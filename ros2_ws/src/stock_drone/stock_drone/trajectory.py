"""Serpentine scan path with jerk-limited segments, in the aisle frame (x along the aisle, y left, z up, m)."""
import dataclasses
import math

import numpy as np


@dataclasses.dataclass
class DoubleS:
    """Rest-to-rest double-S profile over distance L >= 0 with limits v, a, j (Biagiotti and Melchiorri 2008)."""
    L: float
    v: float
    a: float
    j: float

    def __post_init__(self):
        L, v, a, j = self.L, self.v, self.a, self.j
        if L <= 1e-9:
            self.tj = self.ta = self.tv = 0.0
            self.vlim = 0.0
            return
        if v * j < a * a:
            tj = math.sqrt(v / j)
            ta = 2 * tj
        else:
            tj = a / j
            ta = tj + v / a
        tv = L / v - ta
        if tv < 0:  # v is not reached: shrink the peak speed
            tv = 0.0
            if L >= 2 * a ** 3 / j ** 2:
                tj = a / j
                ta = tj / 2 + math.sqrt((tj / 2) ** 2 + L / a)
            else:
                tj = (L / (2 * j)) ** (1 / 3)
                ta = 2 * tj
        self.tj, self.ta, self.tv = tj, ta, tv
        self.alim = j * tj
        self.vlim = (ta - tj) * self.alim

    @property
    def T(self):
        return 2 * self.ta + self.tv

    def sample(self, t):
        """Distance, speed and acceleration at time t along the segment."""
        if self.L <= 1e-9:
            return 0.0, 0.0, 0.0
        t = min(max(t, 0.0), self.T)
        if t > self.T / 2:  # second half mirrors the first
            s, v, a = self._accel_half(self.T - t)
            return self.L - s, v, -a
        return self._accel_half(t)

    def _accel_half(self, t):
        j, tj, ta, alim, vlim = self.j, self.tj, self.ta, self.alim, self.vlim
        if t < tj:
            return j * t ** 3 / 6, j * t ** 2 / 2, j * t
        if t < ta - tj:
            return (alim / 6 * (3 * t ** 2 - 3 * tj * t + tj ** 2), alim * (t - tj / 2), alim)
        if t < ta:
            r = ta - t
            return (vlim * ta / 2 - vlim * r + j * r ** 3 / 6, vlim - j * r ** 2 / 2, j * r)
        return vlim * ta / 2 + vlim * (t - ta), vlim, 0.0


@dataclasses.dataclass
class Segment:
    p0: np.ndarray
    p1: np.ndarray
    profile: DoubleS
    tag: str  # "scan", "climb", "transit", "hold"
    hold: float = 0.0  # s, wait at p1 after arrival

    @property
    def T(self):
        return self.profile.T + self.hold


class Trajectory:
    def __init__(self, segments, yaw=0.0):
        self.segments = segments
        self.yaw = yaw
        self.t0 = np.cumsum([0.0] + [s.T for s in segments])

    @property
    def duration(self):
        return float(self.t0[-1])

    def sample(self, t):
        """Position, velocity, acceleration (aisle frame), yaw and segment tag at time t."""
        t = min(max(t, 0.0), self.duration)
        i = int(np.searchsorted(self.t0, t, side="right") - 1)
        i = min(i, len(self.segments) - 1)
        seg = self.segments[i]
        d = seg.p1 - seg.p0
        n = np.linalg.norm(d)
        u = d / n if n > 1e-9 else np.zeros(3)
        s, v, a = seg.profile.sample(t - self.t0[i])
        return seg.p0 + u * s, u * v, u * a, self.yaw, seg.tag

    def length(self, tag=None):
        return sum(s.profile.L for s in self.segments if tag is None or s.tag == tag)


def serpentine(x_start, x_end, heights, y=0.0, home=(-3.0, 0.0), v_scan=1.0, v_transit=2.0, v_climb=0.5,
               a_max=1.0, j_max=2.0, hold=4.0, takeoff_z=1.0):
    """Path from the take-off hover at home, one pass per height in alternating directions, back to home."""
    hx, hy = home
    pts = [np.array([hx, hy, takeoff_z]), np.array([hx, y, heights[0]]), np.array([x_start, y, heights[0]])]
    tags = ["climb", "transit"]
    holds = [hold, hold]
    left_to_right = True
    for k, z in enumerate(heights):
        xa, xb = (x_start, x_end) if left_to_right else (x_end, x_start)
        if k > 0:
            pts.append(np.array([xa, y, z]))
            tags.append("climb")
            holds.append(hold)
        pts.append(np.array([xb, y, z]))
        tags.append("scan")
        holds.append(hold)
        left_to_right = not left_to_right
    last = pts[-1]
    pts.append(np.array([last[0], y, heights[0]]))
    tags.append("climb")
    holds.append(hold)
    pts.append(np.array([hx, hy, heights[0]]))
    tags.append("transit")
    holds.append(hold)
    pts.append(np.array([hx, hy, takeoff_z]))
    tags.append("climb")
    holds.append(hold)
    vmax = {"scan": v_scan, "transit": v_transit, "climb": v_climb}
    segs = []
    for p0, p1, tag, h in zip(pts[:-1], pts[1:], tags, holds):
        segs.append(Segment(p0, p1, DoubleS(float(np.linalg.norm(p1 - p0)), vmax[tag], a_max, j_max), tag, h))
    return Trajectory(segs)


if __name__ == "__main__":
    tr = serpentine(0.5, 79.5, [0.5, 2.16, 3.76, 5.36, 6.96])
    print(f"duration {tr.duration:.1f} s, path {tr.length():.1f} m, scan {tr.length('scan'):.1f} m")
    ts = np.arange(0, tr.duration, 0.01)
    S = [tr.sample(t) for t in ts]
    P = np.array([s[0] for s in S])
    V = np.array([s[1] for s in S])
    A = np.array([s[2] for s in S])
    dv = np.abs(np.diff(P, axis=0) / 0.01 - V[:-1]).max()
    print(f"max |v| {np.linalg.norm(V, axis=1).max():.2f} m/s, max |a| {np.linalg.norm(A, axis=1).max():.2f} m/s^2, "
          f"position/velocity consistency {dv:.3f} m/s")
