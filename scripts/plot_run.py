"""Side and top views of a run: mission reference, ground truth and VIO, with the racks drawn in.

    python scripts/plot_run.py results/runs/<run> [--out results/figures/<name>.pdf]
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import bagtools  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "stock_drone", "stock_drone"))
import aisle_spec as spec  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "font.size": 8,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
INK, GT, REF, VIO = "#1f1f1f", "#1f1f1f", "#9aa0a6", "#2a63b8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = bagtools.read(bagtools.bag_dir(args.run), ["/gt/odom", "/mission/reference"])
    gt = bagtools.odom_array(d["/gt/odom"])
    ref = bagtools.odom_array(d["/mission/reference"])
    vio = None
    npz = os.path.join(args.run, "d2vins", "vio_traj.npz")
    if os.path.exists(npz):
        vio = np.load(npz)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.7, 3.4), gridspec_kw={"height_ratios": [1.7, 1]}, sharex=True)
    for k in range(spec.LEVELS):
        ax1.axhline(k * spec.LEVEL_H, color="#d9dde3", lw=0.6, zorder=0)
        ax1.axhline(spec.label_z(k), color="#f0d9b5", lw=0.6, ls=":", zorder=0)
    ax1.plot(ref[:, 1], ref[:, 3], color=REF, lw=1.6, label="reference")
    ax1.plot(gt[:, 1], gt[:, 3], color=GT, lw=0.7, label="ground truth")
    if vio is not None:
        ax1.plot(vio["vio_start"][:, 0], vio["vio_start"][:, 2], color=VIO, lw=0.7, label="VIO (start aligned)")
    ax1.set_ylabel("height z (m)")
    ax1.set_ylim(-0.2, spec.HEIGHT + 0.3)
    ax1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    for s in (1, -1):
        ax2.axhline(s * spec.WIDTH / 2, color="#c0262d", lw=0.8)
    ax2.plot(ref[:, 1], ref[:, 2], color=REF, lw=1.6)
    ax2.plot(gt[:, 1], gt[:, 2], color=GT, lw=0.7)
    if vio is not None:
        ax2.plot(vio["vio_start"][:, 0], vio["vio_start"][:, 1], color=VIO, lw=0.7)
    ax2.set_ylim(-1.45, 1.45)
    ax2.set_ylabel("lateral y (m)")
    ax2.set_xlabel("along the aisle x (m)")
    fig.tight_layout(h_pad=0.4)
    out = args.out or os.path.join(args.run, "trajectory.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
