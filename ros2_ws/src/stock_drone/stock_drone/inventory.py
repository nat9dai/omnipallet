"""Merge label reads into an inventory: one row per pallet (SSCC) with its rack slot, misreads flagged."""
import csv
import json

import numpy as np

from stock_drone import aisle_spec as spec

FIELDS = ["sscc", "side", "bay", "level", "slot", "x", "y", "z", "n_reads", "first_t", "status"]


def build(reads):
    """Return sorted inventory rows from reads, a dict sscc -> list of observation dicts (x, y, z, side, t)."""
    rows = []
    slots = [(b, s) for b in range(spec.N_BAYS) for s in range(spec.PALLETS_PER_BAY)]
    for sscc, obs in reads.items():
        # a blurred label can decode to a wrong number that passes the Code 128 checksum but not GS1's
        if len(sscc) != 18 or not sscc.isdigit() or spec.gs1_check_digit(sscc[:17]) != sscc[-1]:
            continue
        xs = float(np.median([o["x"] for o in obs]))
        ys = float(np.median([o["y"] for o in obs]))
        zs = float(np.median([o["z"] for o in obs]))
        side = max(("L", "R"), key=lambda s: sum(o["side"] == s for o in obs))
        level = int(np.argmin([abs(zs - spec.label_z(k)) for k in range(spec.LEVELS)]))
        bay, slot = min(slots, key=lambda bs: abs(spec.slot_x(*bs) - xs))
        rows.append({"sscc": sscc, "side": side, "bay": bay, "level": level, "slot": slot, "x": round(xs, 3),
                     "y": round(ys, 3), "z": round(zs, 3), "n_reads": len(obs),
                     "first_t": round(min(o["t"] for o in obs), 2), "status": "ok"})
    # a code read once in the same slot as a code read several times is a misread
    by_slot = {}
    for r in rows:
        by_slot.setdefault((r["side"], r["level"], r["bay"], r["slot"]), []).append(r)
    for group in by_slot.values():
        if len(group) < 2:
            continue
        best = max(r["n_reads"] for r in group)
        kept = []
        for r in group:
            if r["n_reads"] == 1 and best >= 2:
                r["status"] = "misread"
            else:
                kept.append(r)
        if len(kept) > 1:
            for r in kept:
                r["status"] = "review"  # two well-read codes in one slot: a person checks
    return sorted(rows, key=lambda r: (r["side"], r["level"], r["bay"], r["slot"]))


def counted(rows):
    """Rows that count as pallets: everything except dropped misreads."""
    return [r for r in rows if r.get("status", "ok") != "misread"]


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)


def from_jsonl(path):
    reads = {}
    for line in open(path):
        if line.strip():
            o = json.loads(line)
            reads.setdefault(o["sscc"], []).append(o)
    return build(reads)
