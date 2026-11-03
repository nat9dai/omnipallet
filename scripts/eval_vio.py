"""Accuracy of a VIO trajectory against simulator ground truth.

    python scripts/eval_vio.py results/runs/<run> [--vio d2vins/d2vins_1.csv]   # writes vio_eval.json, vio_traj.npz beside the CSV

"start" ties the VIO to the true pose at the first sample (as in flight); "4dof" best-fits yaw and translation.
"""
import argparse
import json
import os

import numpy as np

import bagtools


def load_vio_csv(path):
    """D2VINS CSV rows: t x y z qw qx qy qz (whitespace or comma separated)."""
    rows = []
    for line in open(path):
        parts = line.replace(",", " ").split()
        if len(parts) >= 8:
            try:
                rows.append([float(v) for v in parts[:8]])
            except ValueError:
                continue
    return np.array(rows)


def yaw_of(qw, qx, qy, qz):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def align_4dof(p_est, p_gt):
    """Yaw and translation minimising |R p_est + t - p_gt| (closed form in the xy plane)."""
    me, mg = p_est.mean(0), p_gt.mean(0)
    a, b = p_est - me, p_gt - mg
    yaw = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum())
    r = rotz(yaw)
    return r, mg - r @ me


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--vio", default="d2vins/d2vins_1.csv")
    args = ap.parse_args()
    gt = bagtools.odom_array(bagtools.read(bagtools.bag_dir(args.run), ["/gt/odom"])["/gt/odom"])
    vio = load_vio_csv(os.path.join(args.run, args.vio))
    t = vio[:, 0]
    keep = (t >= gt[0, 0]) & (t <= gt[-1, 0])
    vio, t = vio[keep], t[keep]
    g = np.column_stack([np.interp(t, gt[:, 0], gt[:, k]) for k in range(1, 8)])  # x y z qx qy qz qw
    p_gt, p_vio = g[:, :3], vio[:, 1:4]
    yaw0 = yaw_of(g[0, 6], g[0, 3], g[0, 4], g[0, 5]) - yaw_of(*vio[0, 4:8])
    r0 = rotz(yaw0)
    p_start = (r0 @ (p_vio - p_vio[0]).T).T + p_gt[0]
    e_start = p_start - p_gt
    r4, t4 = align_4dof(p_vio, p_gt)
    e_4 = (r4 @ p_vio.T).T + t4 - p_gt
    path = np.linalg.norm(np.diff(p_gt, axis=0), axis=1).sum()
    res = {
        "duration_s": float(t[-1] - t[0]),
        "samples": int(len(t)),
        "path_length_m": float(path),
        "ate_rmse_4dof_m": float(np.sqrt((np.linalg.norm(e_4, axis=1) ** 2).mean())),
        "ate_rmse_start_m": float(np.sqrt((np.linalg.norm(e_start, axis=1) ** 2).mean())),
        "final_error_start_m": float(np.linalg.norm(e_start[-1])),
        "drift_percent": float(100 * np.linalg.norm(e_start[-1]) / max(path, 1e-6)),
        "max_lateral_error_start_m": float(np.abs(e_start[:, 1]).max()),
        "max_vertical_error_start_m": float(np.abs(e_start[:, 2]).max()),
    }
    print(json.dumps(res, indent=1))
    out = os.path.dirname(os.path.join(args.run, args.vio))  # next to the estimate, so baselines don't overwrite
    with open(os.path.join(out, "vio_eval.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(out, "vio_traj.npz"), t=t, gt=p_gt, vio_start=p_start, vio_4dof=(r4 @ p_vio.T).T + t4)


if __name__ == "__main__":
    main()
