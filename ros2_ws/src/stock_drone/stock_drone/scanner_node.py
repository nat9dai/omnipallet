"""Read pallet barcodes from the side cameras, place each label in the aisle with the VIO pose, and count stock."""
import bisect
import json
import math
import os
import threading
from collections import defaultdict

import cv2
import numpy as np
import rclpy
import zxingcpp
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from stock_drone import aisle_spec as spec
from stock_drone import inventory

BAR_MODULES = 156  # GS1-128 with (00) + 18 digits: start, FNC1, 10 pairs, check, stop (no quiet zone)


def quat_to_rot(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def slerp_rot(r0, r1, a):
    d = r0.T @ r1
    rvec, _ = cv2.Rodrigues(d)
    rd, _ = cv2.Rodrigues(rvec * a)
    return r0 @ rd


class PoseBuffer:
    def __init__(self, maxlen=4000):
        self.t, self.p, self.r, self.v = [], [], [], []
        self.maxlen = maxlen
        self.lock = threading.Lock()

    def add(self, t, p, r, v):
        with self.lock:
            self.t.append(t)
            self.p.append(p)
            self.r.append(r)
            self.v.append(v)
            if len(self.t) > self.maxlen:
                del self.t[0], self.p[0], self.r[0], self.v[0]

    def at(self, t):
        with self.lock:
            i = bisect.bisect_left(self.t, t)
            if i == 0 or i >= len(self.t):
                return None
            t0, t1 = self.t[i - 1], self.t[i]
            if t1 - t0 > 0.2:
                return None
            a = (t - t0) / (t1 - t0)
            p = (1 - a) * self.p[i - 1] + a * self.p[i]
            v = (1 - a) * self.v[i - 1] + a * self.v[i]
            return p, slerp_rot(self.r[i - 1], self.r[i], a), v


class Camera:
    def __init__(self, name, side, t_bc, r_bc):
        self.name, self.side = name, side
        self.t_bc = np.asarray(t_bc, float)  # camera origin in the body frame (FLU), m
        self.r_bc = np.asarray(r_bc, float)  # camera (x right, y down, z forward) -> body
        self.k = None
        self.n_frames = 0
        self.n_decoded = 0


def cam_rotation(side):
    """Rotation body (FLU) from optical (x right, y down, z forward) for a camera looking left or right."""
    if side == "L":
        return np.column_stack(([1, 0, 0], [0, 0, -1], [0, 1, 0])).astype(float)
    return np.column_stack(([-1, 0, 0], [0, 0, -1], [0, -1, 0])).astype(float)


def decode_tiles(img, n=3, overlap=700):
    """Decode Code 128 in n overlapping vertical tiles; return [(result, tile x offset)], one per text."""
    w = img.shape[1]
    step = (w - overlap) // n  # zxing-cpp misses labels in the full frame; overlap > ~510 px label width
    found = {}
    for k in range(n):
        x0 = k * step
        x1 = w if k == n - 1 else min(w, x0 + step + overlap)
        for r in zxingcpp.read_barcodes(np.ascontiguousarray(img[:, x0:x1]), formats=zxingcpp.BarcodeFormat.Code128):
            found.setdefault(r.text, (r, x0))
    return list(found.values())


class ScannerNode(Node):
    def __init__(self):
        super().__init__("scanner")
        p = self.declare_parameter
        p("pose_topic", "/d2vins/imu_propagation")
        p("home", [-3.0, 0.0, 0.0])  # m, aisle-frame position of the VIO origin (take-off pad)
        p("cameras", ["barcode_left", "barcode_right"])
        p("cam_offset", [0.0, 0.06, 0.0])  # m, left camera in the body frame; right is mirrored
        p("exposure_ms", 0.5)  # for motion blur; 0 turns it off
        p("noise_sigma", 3.0)  # read noise, grey levels
        p("x_dim_mm", spec.X_DIM_MM)
        p("decode_every", 1)
        p("out_dir", os.path.abspath("results/run"))
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.home = np.array(g("home"), float)
        self.exposure = g("exposure_ms") * 1e-3
        self.noise = g("noise_sigma")
        self.bar_w = BAR_MODULES * g("x_dim_mm") * 1e-3  # m
        self.every = max(1, int(g("decode_every")))
        self.out_dir = g("out_dir")
        os.makedirs(self.out_dir, exist_ok=True)
        self.rng = np.random.default_rng(0)
        self.poses = PoseBuffer()
        self.reads = defaultdict(list)  # sscc -> list of observations
        self.log = open(os.path.join(self.out_dir, "reads.jsonl"), "w")
        off = np.array(g("cam_offset"), float)
        self.cams = {}
        for name in g("cameras"):
            side = "L" if "left" in name else "R"
            t = off if side == "L" else off * np.array([1, -1, 1])
            cam = Camera(name, side, t, cam_rotation(side))
            self.cams[name] = cam
            self.create_subscription(CameraInfo, f"/{name}/camera_info", lambda m, c=cam: self.on_info(c, m), 2)
            self.create_subscription(Image, f"/{name}/image_raw", lambda m, c=cam: self.on_image(c, m), 2)
        self.create_subscription(Odometry, g("pose_topic"), self.on_odom, 200)
        self.pub = self.create_publisher(String, "/scan/summary", 10)
        self.create_timer(2.0, self.publish_summary)
        self.create_timer(10.0, self.save)  # keep inventory.csv current in case the node is killed

    def on_odom(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.poses.add(t, np.array([p.x, p.y, p.z]), quat_to_rot(q.w, q.x, q.y, q.z), np.array([v.x, v.y, v.z]))

    def on_info(self, cam, msg):
        cam.k = np.array(msg.k, float).reshape(3, 3)

    def sensor_model(self, img, blur_px):
        """Add the motion blur (along image x) and read noise that the renderer lacks."""
        img = img.astype(np.float32)
        n = int(round(blur_px))
        if n > 1:
            img = cv2.filter2D(img, -1, np.full((1, n), 1.0 / n, np.float32))
        if self.noise > 0:
            img += self.rng.normal(0, self.noise, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)

    def on_image(self, cam, msg):
        cam.n_frames += 1
        if cam.k is None or cam.n_frames % self.every:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pose = self.poses.at(t)
        if pose is None:
            return
        p_wb, r_wb, v_w = pose
        buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        if msg.encoding in ("rgb8", "bgr8"):
            img = cv2.cvtColor(buf[:, :msg.width * 3].reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2GRAY)
        elif msg.encoding == "rgba8":
            img = cv2.cvtColor(buf[:, :msg.width * 4].reshape(msg.height, msg.width, 4), cv2.COLOR_RGBA2GRAY)
        else:
            img = buf[:, :msg.width]
        fx = cam.k[0, 0]
        d_nominal = spec.WIDTH / 2 - abs(cam.t_bc[1])
        v_along = abs(float((r_wb.T @ v_w)[0]))  # body-x speed, which smears along image x
        blur_px = v_along * self.exposure * fx / d_nominal
        img = self.sensor_model(img, blur_px)
        kinv = np.linalg.inv(cam.k)
        for r, x_off in decode_tiles(img):
            if not r.valid or not r.text.startswith("(00)"):
                continue
            pos = r.position
            pts = np.array([[pos.top_left.x, pos.top_left.y], [pos.top_right.x, pos.top_right.y],
                            [pos.bottom_right.x, pos.bottom_right.y], [pos.bottom_left.x, pos.bottom_left.y]], float)
            pts[:, 0] += x_off
            centre = pts.mean(0)
            w_px = 0.5 * (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3]))
            if w_px < 10:
                continue
            z = fx * self.bar_w / w_px
            p_c = z * (kinv @ np.array([centre[0], centre[1], 1.0]))
            p_w = r_wb @ (cam.r_bc @ p_c + cam.t_bc) + p_wb
            p_aisle = p_w + self.home
            sscc = r.text[4:]
            obs = {"t": t, "cam": cam.name, "side": cam.side, "sscc": sscc, "u": centre[0], "v": centre[1],
                   "w_px": w_px, "depth": z, "x": p_aisle[0], "y": p_aisle[1], "z": p_aisle[2],
                   "blur_px": blur_px}
            self.reads[sscc].append(obs)
            self.log.write(json.dumps(obs) + "\n")
            cam.n_decoded += 1

    def publish_summary(self):
        s = {c.name: {"frames": c.n_frames, "decoded": c.n_decoded} for c in self.cams.values()}
        s["unique_pallets"] = len(self.reads)
        self.pub.publish(String(data=json.dumps(s)))

    def save(self):
        rows = inventory.build(self.reads)
        path = os.path.join(self.out_dir, "inventory.csv")
        inventory.write_csv(rows, path)
        self.log.flush()
        return len(rows)


def main():
    rclpy.init()
    node = ScannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    n = node.save()
    print(f"[scanner] saved {n} pallets", flush=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
