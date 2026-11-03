"""Flight summary from ground truth; energy comes from the budget.py power model (the sim battery is not physical).

    python scripts/summarize_run.py results/runs/<run>   # writes <run>/mission_summary.json
"""
import argparse
import json
import os

import numpy as np

import bagtools

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    d = bagtools.read(bagtools.bag_dir(args.run), ["/gt/odom", "/mission/reference"])
    gt = bagtools.odom_array(d["/gt/odom"])
    ref = bagtools.odom_array(d["/mission/reference"])
    air = gt[:, 3] > 0.05
    t_air = gt[air, 0]
    path = float(np.linalg.norm(np.diff(gt[air, 1:4], axis=0), axis=1).sum())
    scan = air & (np.abs(gt[:, 8]) > 0.5) & (gt[:, 1] > 0.5)  # moving along the racks
    zr = np.interp(gt[scan, 0], ref[:, 0], ref[:, 3])
    budget = json.load(open(os.path.join(RES, "budget.json")))
    flight_s = float(t_air[-1] - t_air[0])
    energy_wh = budget["total_power_W"] * flight_s / 3600
    out = {
        "flight_min": flight_s / 60,
        "path_m": path,
        "scan_off_centre_rms_m": float(np.sqrt(np.mean(gt[scan, 2] ** 2))),
        "scan_off_centre_max_m": float(np.abs(gt[scan, 2]).max()),
        "scan_height_err_rms_m": float(np.sqrt(np.mean((gt[scan, 3] - zr) ** 2))),
        "scan_height_err_max_m": float(np.abs(gt[scan, 3] - zr).max()),
        "energy_wh_model": energy_wh,
        "battery_used_pct_model": 100 * energy_wh / budget["battery_Wh"],
    }
    print(json.dumps(out, indent=1))
    with open(os.path.join(args.run, "mission_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
