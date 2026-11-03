"""Build a dense aisle map in flight from the side fisheye pairs and publish new voxels on /map/points_delta.

    python -m stock_drone.stereo_map_node --ros-args -p out_dir:=results/runs/<run>
"""
import collections
import os
import time

import cv2
import numpy as np
import rclpy
import rclpy.executors
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField

from stock_drone.stereo_map import StereoMapper, VoxelGrid


def quat_to_rot(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class StereoMapNode(Node):
    def __init__(self):
        super().__init__("stereo_map")
        self.declare_parameter("pose_topic", "/nav/odom")  # scanner pose too, so labels and map share a frame
        self.declare_parameter("min_move", 0.3)  # m between keyframes
        self.declare_parameter("min_turn_deg", 10.0)
        self.declare_parameter("min_z", 0.25)  # m, no mapping on the dock
        self.declare_parameter("out_dir", "")
        cv2.setNumThreads(2)
        self.mapper, self.grid = StereoMapper(), VoxelGrid()
        self.odom = collections.deque(maxlen=400)  # (t, p, q)
        self.pending = collections.deque(maxlen=10)  # (t, image)
        self.last_kf = None
        self.ms = []
        self.pub = self.create_publisher(PointCloud2, "/map/points_delta", 20)
        self.create_subscription(Odometry, self.get_parameter("pose_topic").value, self.on_odom, 50)
        self.create_subscription(Image, "/quadcam/image", self.on_image, 5)
        self.get_logger().info(f"grid {self.grid.shape} at {self.grid.voxel * 100:.0f} cm")

    def on_image(self, msg):
        self.pending.append((stamp(msg.header), np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)))
        self.process()

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.odom.append((stamp(msg.header), np.array([p.x, p.y, p.z]), np.array([q.w, q.x, q.y, q.z])))
        self.process()

    def pose_at(self, t):
        """Pose interpolated at t, or None if odometry does not cover t yet."""
        if len(self.odom) < 2 or t > self.odom[-1][0] or t < self.odom[0][0]:
            return None
        for (t0, p0, q0), (t1, p1, q1) in zip(list(self.odom)[:-1], list(self.odom)[1:]):
            if t0 <= t <= t1:
                a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                q = q0 + a * ((q1 if np.dot(q0, q1) >= 0 else -q1) - q0)  # nlerp, poses 20 ms apart
                q /= np.linalg.norm(q)
                return p0 + a * (p1 - p0), quat_to_rot(*q)
        return None

    def process(self):
        while self.pending:
            t, img = self.pending[0]
            if self.odom and t < self.odom[0][0]:
                self.pending.popleft()
                continue
            pose = self.pose_at(t)
            if pose is None:
                return  # wait for odometry past this image
            self.pending.popleft()
            p, r = pose
            if p[2] < self.get_parameter("min_z").value:
                continue
            if self.last_kf is not None:
                dp = np.linalg.norm(p - self.last_kf[0])
                dang = np.degrees(np.arccos(np.clip((np.trace(self.last_kf[1].T @ r) - 1) / 2, -1, 1)))
                if dp < self.get_parameter("min_move").value and dang < self.get_parameter("min_turn_deg").value:
                    continue
            self.last_kf = (p, r)
            t0 = time.perf_counter()
            pts, gray = self.mapper.points(img, r, p)
            new_p, new_g = self.grid.add(pts, gray, t)
            self.ms.append((time.perf_counter() - t0) * 1000)
            self.publish(t, new_p, new_g)
            if len(self.ms) % 100 == 0:
                n = sum(len(a) for a in self.grid.pts)
                self.get_logger().info(f"{len(self.ms)} keyframes, {n} voxels, {np.mean(self.ms[-100:]):.0f} ms "
                                       f"per keyframe")

    def publish(self, t, p, g):
        m = PointCloud2()
        m.header.stamp.sec = int(t)
        m.header.stamp.nanosec = int(round((t - int(t)) * 1e9)) % 1_000_000_000
        m.header.frame_id = "nav"
        m.height, m.width = 1, len(p)
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                    for i, n in enumerate(("x", "y", "z", "gray"))]
        m.is_bigendian, m.point_step, m.is_dense = False, 16, True
        m.row_step = 16 * len(p)
        m.data = np.column_stack([p, g]).astype(np.float32).tobytes()
        self.pub.publish(m)

    def save(self):
        out = self.get_parameter("out_dir").value
        if out:
            p, g, t = self.grid.arrays()
            np.savez(os.path.join(out, "map.npz"), p=p, g=g, t=t, ms=np.array(self.ms))  # fast: stop kills in 5 s
            self.get_logger().info(f"saved {len(p)} voxels to {out}/map.npz")


def main():
    rclpy.init()
    node = StereoMapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
