"""Camera rig: four fisheyes tiled into one grey image (one shared stamp) and two side barcode cameras."""
import array
import math
import os
import sys

import numpy as np
from isaacsim.sensors.camera import Camera
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design as d  # noqa: E402
import fisheye_model as fm  # noqa: E402

# Charpi camera mount centres, body FLU (m), cam0..cam3 as in design.FISHEYE_YAW_DEG
FISHEYE_POS = [(0.0944, -0.0386, 0.0963), (0.0944, 0.0386, 0.0963), (-0.1074, 0.0386, 0.0963),
               (-0.1074, -0.0386, 0.0963)]


def optical_to_body(yaw_deg):
    """Rotation body <- optical (x right, y down, z forward) for a level camera with heading yaw."""
    p = math.radians(yaw_deg)
    z = np.array([math.cos(p), math.sin(p), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def quat_wxyz(rot):
    x, y, z, w = R.from_matrix(rot).as_quat()
    return np.array([w, x, y, z])


def barcode_rotation(side):
    if side == "left":
        return np.column_stack(([1, 0, 0], [0, 0, -1], [0, 1, 0])).astype(float)
    return np.column_stack(([-1, 0, 0], [0, 0, -1], [0, -1, 0])).astype(float)


class SensorRig:
    def __init__(self, parent, node=None, fisheye=True, barcode=True, barcode_res=None):
        """Create camera prims under `parent` (the drone body). Call initialize() after world.reset()."""
        self.node = node
        self.fisheyes, self.barcodes = [], {}
        if fisheye:
            for i, (pos, yaw) in enumerate(zip(FISHEYE_POS, d.FISHEYE_YAW_DEG)):
                cam = Camera(prim_path=f"{parent}/fisheye{i}", resolution=(fm.W, fm.H))
                cam.set_local_pose(np.array(pos), quat_wxyz(optical_to_body(yaw)), camera_axes="ros")
                self.fisheyes.append(cam)
        if barcode:
            res = tuple(barcode_res or d.BARCODE_RES)
            for side in ("left", "right"):
                cam = Camera(prim_path=f"{parent}/barcode_{side}", resolution=res)
                cam.set_local_pose(np.array(d.BARCODE_POS[side]), quat_wxyz(barcode_rotation(side)),
                                   camera_axes="ros")
                self.barcodes[side] = cam
        self._pubs = {}

    def initialize(self):
        for cam in self.fisheyes:
            cam.initialize()
            cam.set_clipping_range(0.02, 150.0)
            # F-theta: Isaac 5.1 clips its OpenCV fisheye lens to the pinhole frustum, below 180 deg
            cam.set_ftheta_properties(nominal_width=fm.W, nominal_height=fm.H, optical_center=(fm.CX, fm.CY),
                                      max_fov=fm.MAX_FOV_DEG, distortion_coefficients=list(fm.FTHETA_K))
        for side, cam in self.barcodes.items():
            cam.initialize()
            cam.set_clipping_range(0.05, 50.0)
            w, h = cam.get_resolution()
            fpx = d.BARCODE_F_PX * w / d.BARCODE_RES[0]  # keep the field of view if rendered smaller
            cam.set_opencv_pinhole_properties(cx=w / 2, cy=h / 2, fx=fpx, fy=fpx, pinhole=[0.0] * 12)
        if self.node is not None:
            from sensor_msgs.msg import CameraInfo, Image
            self._Image, self._Info = Image, CameraInfo
            self._pubs["quad"] = self.node.create_publisher(Image, "/quadcam/image", 5)
            for side in self.barcodes:
                self._pubs[side] = self.node.create_publisher(Image, f"/barcode_{side}/image_raw", 2)
                self._pubs[side + "_info"] = self.node.create_publisher(CameraInfo, f"/barcode_{side}/camera_info", 2)

    @staticmethod
    def gray(rgba):
        import cv2  # ~10x faster than NumPy on 12 MP frames
        return cv2.cvtColor(np.ascontiguousarray(rgba[:, :, :4]).astype(np.uint8, copy=False), cv2.COLOR_RGBA2GRAY)

    def quad_image(self):
        tiles = []
        for cam in self.fisheyes:
            rgba = cam.get_rgba()
            if rgba is None or rgba.size == 0:
                return None
            tiles.append(self.gray(rgba))
        return np.hstack(tiles)

    def _image_msg(self, img, t, frame_id):
        m = self._Image()
        m.header.stamp.sec = int(t)
        m.header.stamp.nanosec = int(round((t - int(t)) * 1e9)) % 1_000_000_000
        m.header.frame_id = frame_id
        m.height, m.width = img.shape[:2]
        m.encoding = "mono8"
        m.step = m.width
        # rclpy copies bytes element by element; array.array is ~60x faster
        m.data = array.array("B", np.ascontiguousarray(img).tobytes())
        return m

    def publish_quad(self, t):
        """Publish the four fisheye tiles; t must be the sim time of the rendered frame (now - render dt)."""
        q = self.quad_image()
        if q is not None:
            self._pubs["quad"].publish(self._image_msg(q, t, "quadcam"))
        return q

    def publish_barcodes(self, t):
        """Publish both barcode frames; t is the sim time of the rendered frame (now - render dt)."""
        for side, cam in self.barcodes.items():
            rgba = cam.get_rgba()
            if rgba is None or rgba.size == 0:
                self.empty_barcode_frames = getattr(self, "empty_barcode_frames", 0) + 1
                if self.empty_barcode_frames in (1, 10, 100, 1000):
                    print(f"[cameras] empty barcode frame #{self.empty_barcode_frames} ({side})", flush=True)
                continue
            self._pubs[side].publish(self._image_msg(self.gray(rgba), t, f"barcode_{side}"))
            info = self._Info()
            info.header.stamp.sec = int(t)
            info.header.stamp.nanosec = int(round((t - int(t)) * 1e9)) % 1_000_000_000
            info.header.frame_id = f"barcode_{side}"
            w, h = cam.get_resolution()
            fpx = d.BARCODE_F_PX * w / d.BARCODE_RES[0]
            info.width, info.height = w, h
            info.k = [fpx, 0.0, w / 2, 0.0, fpx, h / 2, 0.0, 0.0, 1.0]
            info.distortion_model = "plumb_bob"
            info.d = [0.0] * 5
            self._pubs[side + "_info"].publish(info)
