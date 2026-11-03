"""Score a flight's map against a reference map built by the same stereo mapper at the true pose.

    python scripts/map_eval.py results/runs/<run> --reference   # once: <run>/demo/map_ref.npz from the replay
    python scripts/map_eval.py results/runs/<run>               # writes <run>/map_eval.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "stock_drone"))
from stock_drone import aisle_spec as spec  # noqa: E402
from stock_drone.stereo_map import StereoMapper, VoxelGrid  # noqa: E402


def reference(run):
    demo = os.path.join(run, "demo")
    fr = os.path.join(demo, "frames")
    times = json.load(open(os.path.join(fr, "frames.json")))["times"]
    gt = np.load(os.path.join(demo, "flight.npz"))["gt"]
    sl = Slerp(gt[:, 0], R.from_quat(gt[:, 4:8]))
    m, grid, last = StereoMapper(), VoxelGrid(lo=(-9.0, -3.2, -0.5), hi=(83.0, 3.2, 9.5)), None
    for i, t in enumerate(times):
        p = np.array([np.interp(t, gt[:, 0], gt[:, k]) for k in (1, 2, 3)])
        if p[2] < 0.25 or (last is not None and np.linalg.norm(p - last) < 0.3):
            continue
        last = p
        g = cv2.imread(os.path.join(fr, "fish", f"{i:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
        quad = np.hstack([g[r * 360:(r + 1) * 360, c * 640:(c + 1) * 640] for r, c in ((0, 0), (0, 1), (1, 0), (1, 1))])
        pts, gr = m.points(quad, sl([t]).as_matrix()[0], p)
        grid.add(pts, gr, t)
    p, g, t = grid.arrays()
    np.savez(os.path.join(demo, "map_ref.npz"), p=p, g=g, t=t)
    print(f"reference map: {len(p)} voxels")


def face_errors(p, pallets):
    """Front of the map near each label minus the true face distance (m), per pallet with enough points."""
    tree_xz = cKDTree(p[:, [0, 2]])
    out = []
    for pal in pallets:
        x, y, z = pal.label_pose
        idx = tree_xz.query_ball_point([x, z], 0.2)
        if len(idx) < 10:
            continue
        q = p[idx]
        q = q[np.sign(q[:, 1]) == np.sign(y)]
        if len(q) < 10:
            continue
        front = np.percentile(np.abs(q[:, 1]), 10)  # the nearest surface, robust to a few stray voxels
        out.append(front - abs(y))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reference", action="store_true")
    args = ap.parse_args()
    if args.reference:
        reference(args.run)
        return
    m = np.load(os.path.join(args.run, "demo", "map.npz"))
    p, sp = m["dense_p"].astype(float), m["sparse_p"].astype(float)
    pallets = spec.layout()
    res = {"dense_voxels": int(len(p)), "sparse_landmarks": int(len(sp))}
    raw = os.path.join(args.run, "map.npz")
    if os.path.exists(raw):
        ms = np.load(raw)["ms"]
        res.update(keyframes=int(len(ms)), ms_per_keyframe_mean=float(ms.mean()),
                   ms_per_keyframe_p95=float(np.percentile(ms, 95)))
    e = face_errors(p, pallets)
    res.update(face_err_median_m=float(np.median(e)), face_abs_err_median_m=float(np.median(np.abs(e))),
               face_abs_err_p95_m=float(np.percentile(np.abs(e), 95)), face_pallets=int(len(e)))
    near = sp[np.abs(np.abs(sp[:, 1]) - 1.25) < 0.1]
    res["sparse_within_10cm_of_face_frac"] = float(len(near) / max(1, len(sp)))
    ref_path = os.path.join(args.run, "demo", "map_ref.npz")
    if os.path.exists(ref_path):
        ref = np.load(ref_path)["p"].astype(float)
        sel = p[(np.abs(p[:, 1]) > 1.0) & (np.abs(p[:, 1]) < 1.6) & (p[:, 0] > 0) & (p[:, 0] < 80)]
        dist, _ = cKDTree(ref).query(sel, distance_upper_bound=1.0)
        dist = dist[np.isfinite(dist)]
        res.update(c2c_median_m=float(np.median(dist)), c2c_p95_m=float(np.percentile(dist, 95)),
                   ref_voxels=int(len(ref)))
    json.dump(res, open(os.path.join(args.run, "map_eval.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
