"""Read run bags (MCAP, ROS 2 Humble) into numpy arrays without a ROS install."""
import glob
import os

import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

PX4_MSG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ros2_ws", "src", "px4_msgs", "msg")


def _typestore(px4_types=()):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    add = {}
    for name in px4_types:
        path = os.path.join(PX4_MSG_DIR, f"{name}.msg")
        add.update(get_types_from_msg(open(path).read(), f"px4_msgs/msg/{name}"))
    if add:
        ts.register(add)
    return ts


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def read(bag_dir, topics, px4_types=()):
    """Read topics into a dict topic -> list of (bag receive time s, message)."""
    out = {t: [] for t in topics}
    with AnyReader([Path(bag_dir).resolve()], default_typestore=_typestore(px4_types)) as r:
        conns = [c for c in r.connections if c.topic in topics]
        for c, t_ns, raw in r.messages(connections=conns):
            out[c.topic].append((t_ns * 1e-9, r.deserialize(raw, c.msgtype)))
    return out


def odom_array(msgs):
    """(N, 11): t, x, y, z, qx, qy, qz, qw, vx, vy, vz from nav_msgs/Odometry (header stamps)."""
    rows = []
    for _, m in msgs:
        p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        rows.append([stamp(m.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z])
    return np.array(rows) if rows else np.zeros((0, 11))


def phases(msgs, clock_msgs=None):
    """List of (bag time, phase name) at each change."""
    out, last = [], None
    for t, m in msgs:
        if m.data != last:
            out.append((t, m.data))
            last = m.data
    return out


def bag_dir(run):
    cands = sorted(glob.glob(os.path.join(run, "bag*")))
    return cands[0] if cands else None
