"""Pull a flight's dense stereo map and D2VINS landmarks from its bag, in the aisle frame, with join times (sim s).

    python scripts/map_data.py results/runs/<run>   # writes <run>/demo/map.npz
"""
import argparse
import os

import numpy as np

import bagtools

DOCK_X = -3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    topics = ["/map/points_delta", "/d2vins/margined_cloud", "/gt/odom", "/nav/odom", "/d2vins/imu_propagation",
              "/nav/rack_offset", "/nav/aisle_offset"]
    d = bagtools.read(bagtools.bag_dir(args.run), topics)
    clock = np.array([[bagtools.stamp(m.header), t] for t, m in d["/gt/odom"][::25]])  # sim, wall
    to_sim = lambda w: np.interp(w, clock[:, 1], clock[:, 0])  # noqa: E731

    dp, dg, dt = [], [], []
    for _, m in d["/map/points_delta"]:
        a = np.frombuffer(bytes(m.data), np.float32).reshape(-1, 4)
        dp.append(a[:, :3])
        dg.append(a[:, 3].astype(np.uint8))
        dt.append(np.full(len(a), bagtools.stamp(m.header), np.float32))
    dense_p = np.concatenate(dp) + np.float32([DOCK_X, 0, 0])

    # offsets the bridge removed from the VIO pose, vs sim time
    def series(topic):
        rows = [(to_sim(t), m.data) for t, m in d[topic]]
        return np.array(rows) if rows else np.zeros((1, 2))
    rack, aisle = series("/nav/rack_offset"), series("/nav/aisle_offset")
    vio = bagtools.odom_array(d["/d2vins/imu_propagation"])
    nav = bagtools.odom_array(d["/nav/odom"])
    dz = np.column_stack([nav[:, 0], np.interp(nav[:, 0], vio[:, 0], vio[:, 3]) - nav[:, 3]])

    def step(s, t):  # last value at or before t
        k = np.searchsorted(s[:, 0], t, side="right") - 1
        return np.where(k >= 0, s[np.maximum(k, 0), 1], 0.0)

    sp, st = [], []
    for w, m in d["/d2vins/margined_cloud"]:
        if not len(m.points):
            continue
        t = to_sim(w)
        p = np.array([[q.x, q.y, q.z] for q in m.points])
        p -= [step(rack, t), step(aisle, t), np.interp(t, dz[:, 0], dz[:, 1])]
        sp.append(p)
        st.append(np.full(len(p), t))
    sparse_p, sparse_t = np.concatenate(sp), np.concatenate(st)
    _, first = np.unique(sparse_p.round(3), axis=0, return_index=True)  # a cloud repeats until the next marginalisation
    first.sort()
    sparse_p, sparse_t = sparse_p[first] + [DOCK_X, 0, 0], sparse_t[first]

    out = os.path.join(args.run, "demo")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "map.npz"), dense_p=dense_p.astype(np.float32), dense_g=np.concatenate(dg),
             dense_t=np.concatenate(dt), sparse_p=sparse_p.astype(np.float32), sparse_t=sparse_t.astype(np.float32))
    print(f"wrote {out}/map.npz: {len(dense_p)} dense voxels, {len(sparse_p)} D2VINS landmarks")


if __name__ == "__main__":
    main()
