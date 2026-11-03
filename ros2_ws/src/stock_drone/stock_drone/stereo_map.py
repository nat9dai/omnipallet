"""Dense rack-face map: each side's two fisheyes become rectified virtual pinholes matched with SGBM (as in OmniNxt).

    m = StereoMapper()
    pts, gray = m.points(quad_gray, R_wb, t_wb)   # quad_gray: (360, 2560) uint8, tiles cam0..cam3
    grid = VoxelGrid(); joined, joined_gray = grid.add(pts, gray, t)
"""
import math

import cv2
import numpy as np

# must match the rendered rig in sim/vehicle/fisheye_model.py and sim/vehicle/cameras.py
TILE_W, TILE_H = 640, 360
F_EQ = 310.0 / math.radians(118.0)  # px per rad, equidistant lens, 236 deg across a tile
CAM_POS = [(0.0944, -0.0386, 0.0963), (0.0944, 0.0386, 0.0963), (-0.1074, 0.0386, 0.0963),
           (-0.1074, -0.0386, 0.0963)]  # body FLU, m; cam0 FR, cam1 FL, cam2 RL, cam3 RR
CAM_YAW = [-45.0, 45.0, 135.0, -135.0]  # deg
# side: (left-image cam, right-image cam, virtual heading in deg)
PAIRS = {"left": (2, 1, 90.0), "right": (0, 3, -90.0)}


def optical_to_body(yaw_deg):
    """Rotation body <- optical (x right, y down, z forward) for a level camera with heading yaw."""
    p = math.radians(yaw_deg)
    z = np.array([math.cos(p), math.sin(p), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


class StereoMapper:
    def __init__(self, f=150.0, size=384, max_disp=64, z_range=(0.45, 2.8)):
        """f: virtual focal length, px (the fisheye has 150 px/rad); size: virtual image side, px."""
        self.f, self.size, self.z_range = f, size, z_range
        c = (size - 1) / 2.0
        u, v = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
        self.rays = np.stack([(u - c) / f, (v - c) / f, np.ones_like(u)], axis=-1)  # (S, S, 3) virtual frame
        self.maps, self.r_bv, self.base, self.pos = {}, {}, {}, {}
        for side, (cl, cr, yaw) in PAIRS.items():
            r_bv = optical_to_body(yaw)
            self.r_bv[side] = r_bv
            self.pos[side] = np.array(CAM_POS[cl])
            self.base[side] = float(np.linalg.norm(np.array(CAM_POS[cl]) - np.array(CAM_POS[cr])))
            self.maps[side] = [self._fisheye_map(r_bv, k) for k in (cl, cr)]
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max_disp, blockSize=5,
                                          P1=8 * 25, P2=32 * 25, disp12MaxDiff=1, uniquenessRatio=15,
                                          speckleWindowSize=80, speckleRange=2,
                                          mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.c = c

    def _fisheye_map(self, r_bv, k):
        """cv2.remap tables that resample fisheye tile k into the virtual camera r_bv."""
        d = self.rays @ (optical_to_body(CAM_YAW[k]).T @ r_bv).T  # rays in fisheye k's optical frame
        t = np.arctan2(np.hypot(d[..., 0], d[..., 1]), d[..., 2])
        phi = np.arctan2(d[..., 1], d[..., 0])
        mx = (TILE_W / 2 + F_EQ * t * np.cos(phi)).astype(np.float32)
        my = (TILE_H / 2 + F_EQ * t * np.sin(phi)).astype(np.float32)
        out = (mx < 0) | (mx > TILE_W - 1) | (my < 0) | (my > TILE_H - 1)
        mx[out], my[out] = -1, -1
        return cv2.convertMaps(mx, my, cv2.CV_16SC2)

    def views(self, quad, side):
        """The rectified (left, right) virtual images of one side from the (360, 2560) quad image."""
        tiles = [quad[:, k * TILE_W:(k + 1) * TILE_W] for k in range(4)]
        cl, cr, _ = PAIRS[side]
        return [cv2.remap(tiles[k], m[0], m[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                for k, m in zip((cl, cr), self.maps[side])]

    def points(self, quad, r_wb, t_wb, step=2):
        """World points (N, 3) and their gray level (N,) from both sides; step thins the pixel grid."""
        out_p, out_g = [], []
        for side in PAIRS:
            left, right = self.views(quad, side)
            disp = self.sgbm.compute(left, right).astype(np.float32) / 16.0
            disp = disp[::step, ::step]
            ok = disp > 0.5
            z = np.where(ok, self.f * self.base[side] / np.maximum(disp, 0.5), 0)
            ok &= (z > self.z_range[0]) & (z < self.z_range[1])
            p_v = self.rays[::step, ::step][ok] * z[ok][:, None]  # (N, 3) virtual left-camera frame
            p_b = self.pos[side] + p_v @ self.r_bv[side].T
            out_p.append(t_wb + p_b @ r_wb.T)
            out_g.append(left[::step, ::step][ok])
        return np.concatenate(out_p), np.concatenate(out_g)


class VoxelGrid:
    """Voxel grid over the aisle that keeps a voxel once min_hits keyframes saw it, so one-off bad matches drop out."""

    def __init__(self, lo=(-6.0, -3.2, -0.5), hi=(86.0, 3.2, 9.5), voxel=0.03, min_hits=3):
        self.lo, self.voxel, self.min_hits = np.array(lo), voxel, min_hits
        self.shape = tuple(np.ceil((np.array(hi) - self.lo) / voxel).astype(int))
        self.hits = np.zeros(self.shape, np.uint8)  # 3067 x 214 x 334 = 219 MB for the default aisle box
        self.pts, self.g, self.t = [], [], []

    def add(self, p, g, t=0.0):
        """Count one keyframe's points; return the centres (N, 3) and gray (N,) of voxels that just joined."""
        idx =np.floor((p - self.lo) / self.voxel).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < self.shape), axis=1)
        flat = np.ravel_multi_index(idx[ok].T, self.shape)
        flat, first = np.unique(flat, return_index=True)  # one hit per keyframe
        h = self.hits.reshape(-1)
        n = h[flat].astype(np.int32) + 1
        h[flat] = np.minimum(n, 255).astype(np.uint8)
        j = n == self.min_hits
        c = (self.lo + (np.stack(np.unravel_index(flat[j], self.shape), axis=1) + 0.5) * self.voxel).astype(np.float32)
        jg = g[ok][first][j]
        self.pts.append(c)
        self.g.append(jg)
        self.t.append(np.full(len(c), t, np.float32))
        return c, jg

    def arrays(self):
        if not self.pts:
            return np.zeros((0, 3), np.float32), np.zeros(0, np.uint8), np.zeros(0, np.float32)
        return np.concatenate(self.pts), np.concatenate(self.g), np.concatenate(self.t)
