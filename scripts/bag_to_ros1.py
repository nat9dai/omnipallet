"""Convert a run's ROS 2 bag to a ROS 1 bag for D2VINS, indexed by header stamp so it replays at sim speed.

    python scripts/bag_to_ros1.py results/runs/<run> [--topics /imu /quadcam/image /gt/odom] [--tile-order 0 3 2 1]

--tile-order k0 k1 k2 k3 makes output tile i the input tile k_i (D2VINS expects counter-clockwise tiles).
"""
import argparse
import heapq
import os
import subprocess
from pathlib import Path

from rosbags.rosbag1 import Reader as Reader1
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--topics", nargs="+", default=["/imu", "/quadcam/image", "/gt/odom"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--tile-order", type=int, nargs=4, default=None)
    args = ap.parse_args()
    src = Path(args.run) / "bag"
    dst = Path(args.out or os.path.join(args.run, "ros1.bag"))
    tmp = dst.with_suffix(".wall.bag")
    for p in (dst, tmp):
        if p.exists():
            p.unlink()
    cmd = ["rosbags-convert", "--src", str(src), "--dst", str(tmp), "--include-topic", *args.topics]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    # re-index by header stamp (sim time)
    ts1 = get_typestore(Stores.ROS1_NOETIC)
    count, first, last = {}, None, None
    with Reader1(tmp) as reader, Writer(dst) as writer:
        out = {c.topic: writer.add_connection(c.topic, c.msgtype, typestore=ts1) for c in reader.connections}

        def reorder(raw, msgtype):
            m = ts1.deserialize_ros1(raw, msgtype)
            w = m.width // 4
            img = m.data.reshape(m.height, m.width)
            m.data = np.ascontiguousarray(np.hstack([img[:, k * w:(k + 1) * w] for k in args.tile_order])).ravel()
            return ts1.serialize_ros1(m, msgtype)

        def stream(conn):  # header stamps are monotonic within a topic
            for c, _, raw in reader.messages(connections=[conn]):
                h = ts1.deserialize_ros1(raw, c.msgtype).header.stamp
                if args.tile_order and c.topic == "/quadcam/image":
                    raw = reorder(raw, c.msgtype)
                yield h.sec * 1_000_000_000 + h.nanosec, c.topic, raw

        for t_ns, topic, raw in heapq.merge(*(stream(c) for c in reader.connections), key=lambda x: x[0]):
            if t_ns > 0:
                writer.write(out[topic], t_ns, raw)
                count[topic] = count.get(topic, 0) + 1
                first = t_ns if first is None else first
                last = t_ns
    tmp.unlink()
    span = (last - first) * 1e-9 if first is not None else 0
    print(f"wrote {dst}: {count}, {span:.1f} s of sim time")


if __name__ == "__main__":
    main()
