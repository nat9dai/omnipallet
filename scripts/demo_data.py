"""Pack one flight for the demo video: true pose, flown pose and barcode reads.

    python scripts/demo_data.py results/runs/<run>   # writes <run>/demo/flight.npz

flight.npz (sim s, aisle frame): gt (N, 8) t x y z qx qy qz qw; nav (M, 4) t x y z; ssccs (S,) valid SSCCs;
reads (K, 7) t, cam (0 L, 1 R), u, v, w_px, sscc index, first read; clock (C, 2) sim time, bag wall time.
"""
import argparse
import json
import os
import sys

import numpy as np

import bagtools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "stock_drone", "stock_drone"))
import aisle_spec as spec  # noqa: E402

DOCK_X = -3.0  # m, the VIO origin is the dock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    d = bagtools.read(bagtools.bag_dir(args.run), ["/gt/odom", "/nav/odom"])
    gt = bagtools.odom_array(d["/gt/odom"])[:, :8]
    clock = np.array([[bagtools.stamp(m.header), t] for t, m in d["/gt/odom"][::25]])
    nav = bagtools.odom_array(d["/nav/odom"])[:, :4]
    nav[:, 1] += DOCK_X
    ssccs, rows, seen = [], [], set()
    for line in open(os.path.join(args.run, "reads.jsonl")):
        o = json.loads(line)
        s = o["sscc"]
        if len(s) != 18 or spec.gs1_check_digit(s[:17]) != s[-1]:
            continue
        new = s not in seen
        if new:
            seen.add(s)
            ssccs.append(s)
        rows.append([o["t"], 0 if o["side"] == "L" else 1, o["u"], o["v"], o["w_px"], ssccs.index(s), new])
    out = os.path.join(args.run, "demo")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "flight.npz"), gt=gt, nav=nav, reads=np.array(rows, float), ssccs=np.array(ssccs),
             clock=clock)
    print(f"wrote {out}/flight.npz: {len(gt)} poses, {len(nav)} nav, {len(rows)} reads of {len(ssccs)} pallets, "
          f"t {gt[0, 0]:.1f} to {gt[-1, 0]:.1f} s")


if __name__ == "__main__":
    main()
