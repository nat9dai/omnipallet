"""Score a run's inventory against the true layout: recall, precision, slot accuracy, label position error (m).

    python scripts/eval_inventory.py results/run   # writes <run>/eval.json
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "stock_drone", "stock_drone"))
import aisle_spec as spec  # noqa: E402


def evaluate(run_dir, x_range=None, levels=None):
    truth = {p.sscc: p for p in spec.layout()
             if (x_range is None or x_range[0] <= p.x <= x_range[1]) and (levels is None or p.level in levels)}
    inv = os.path.join(run_dir, "inventory.csv")
    if not os.path.exists(inv) or os.path.getmtime(inv) < os.path.getmtime(os.path.join(run_dir, "reads.jsonl")):
        sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "stock_drone"))
        from stock_drone import inventory  # noqa: E402
        inventory.write_csv(inventory.from_jsonl(os.path.join(run_dir, "reads.jsonl")), inv)
    with open(inv) as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if r.get("status", "ok") != "misread"]
    found = [r for r in rows if r["sscc"] in truth]
    errs, slot_ok = [], 0
    per = {}
    for r in found:
        p = truth[r["sscc"]]
        tx, ty, tz = p.label_pose
        errs.append([float(r["x"]) - tx, float(r["y"]) - ty, float(r["z"]) - tz])
        ok = (r["side"] == p.side and int(r["bay"]) == p.bay and int(r["level"]) == p.level
              and int(r["slot"]) == p.slot)
        slot_ok += ok
    for p in truth.values():
        k = f"{p.side}{p.level + 1}"
        per.setdefault(k, [0, 0])
        per[k][1] += 1
    for r in found:
        p = truth[r["sscc"]]
        per[f"{p.side}{p.level + 1}"][0] += 1
    e = np.array(errs) if errs else np.zeros((0, 3))
    out = {
        "pallets_true": len(truth),
        "pallets_read": len(found),
        "recall": len(found) / len(truth),
        "precision": len(found) / max(len(rows), 1),
        "slot_accuracy": slot_ok / max(len(found), 1),
        "pos_err_median_m": float(np.median(np.linalg.norm(e, axis=1))) if len(e) else None,
        "pos_err_p95_m": float(np.percentile(np.linalg.norm(e, axis=1), 95)) if len(e) else None,
        "x_err_mean_m": float(e[:, 0].mean()) if len(e) else None,
        "recall_by_side_level": {k: v[0] / v[1] for k, v in sorted(per.items())},
        "missed": sorted(set(truth) - {r["sscc"] for r in found}),
        "dropped_misreads": sum(r.get("status") == "misread" for r in all_rows),
        "flagged_for_review": sum(r.get("status") == "review" for r in all_rows),
    }
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=os.path.join(ROOT, "results", "run"))
    ap.add_argument("--x-range", type=float, nargs=2, default=None)
    ap.add_argument("--levels", type=int, nargs="*", default=None, help="0-based levels that were flown")
    a = ap.parse_args()
    res = evaluate(a.run, a.x_range, a.levels)
    print(json.dumps({k: v for k, v in res.items() if k != "missed"}, indent=1))
    print(f"missed {len(res['missed'])}")
