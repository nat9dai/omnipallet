"""Equidistant 236 deg fisheye of the sim quad camera and its best-fit MEI + radtan calibration for D2VINS.

    python3 sim/vehicle/fisheye_model.py   # prints the calibration and the fit error
"""
import functools
import math

import numpy as np

W, H = 640, 360
CX, CY = 320.0, 180.0
MAX_FOV_DEG = 236.0
F_EQ = 310.0 / math.radians(MAX_FOV_DEG / 2)  # px/rad: the 118 deg half-angle lands at r = 310 px
FTHETA_K = [0.0, 1.0 / F_EQ, 0.0, 0.0, 0.0]  # Isaac F-theta k0..k4, theta = sum k_i r^i


def mei_radius(t, gamma, xi, k1, k2):
    """Pixel radius of a ray at angle t under MEI + radial distortion (tangential terms are zero)."""
    m = np.sin(t) / (np.cos(t) + xi)
    return gamma * m * (1 + k1 * m ** 2 + k2 * m ** 4)


@functools.lru_cache(maxsize=1)
def mei_calibration():
    """Best MEI + radtan fit to the equidistant lens: dict with gamma, xi, k1, k2 and the fit errors."""
    from scipy.optimize import least_squares
    t = np.linspace(0, math.radians(MAX_FOV_DEG / 2), 500)
    r = t * F_EQ
    best = None
    for xi0 in (1.2, 1.6, 2.0, 2.5):
        sol = least_squares(lambda p: mei_radius(t, *p) - r, [(1 + xi0) * F_EQ, xi0, 0.0, 0.0],
                            bounds=([10, 1.0, -2, -2], [3000, 6, 2, 2]))
        if best is None or sol.cost < best.cost:
            best = sol
    e = np.abs(mei_radius(t, *best.x) - r)
    g, xi, k1, k2 = best.x
    return {"gamma": float(g), "xi": float(xi), "k1": float(k1), "k2": float(k2), "max_err_px": float(e.max()),
            "rms_err_px": float(np.sqrt((e ** 2).mean()))}


def equidistant_project(p_cam):
    """Project optical-frame points (N, 3) with the rendered lens. Returns (N, 2) pixels."""
    p = np.atleast_2d(p_cam).astype(float)
    t = np.arctan2(np.linalg.norm(p[:, :2], axis=1), p[:, 2])
    phi = np.arctan2(p[:, 1], p[:, 0])
    r = F_EQ * t
    return np.column_stack([CX + r * np.cos(phi), CY + r * np.sin(phi)])


def mei_project(p_cam):
    """Project optical-frame points (N, 3) with the MEI calibration D2VINS uses. Returns (N, 2) pixels."""
    c = mei_calibration()
    p = np.atleast_2d(p_cam).astype(float)
    rho = np.linalg.norm(p, axis=1)
    den = p[:, 2] + c["xi"] * rho
    mx, my = p[:, 0] / den, p[:, 1] / den
    r2 = mx ** 2 + my ** 2
    s = 1 + c["k1"] * r2 + c["k2"] * r2 ** 2
    return np.column_stack([c["gamma"] * mx * s + CX, c["gamma"] * my * s + CY])


if __name__ == "__main__":
    c = mei_calibration()
    print(f"rendered lens: equidistant, F_EQ {F_EQ:.2f} px/rad, {MAX_FOV_DEG:.0f} deg max, "
          f"{math.degrees(2 * CY / F_EQ):.0f} deg up-down")
    print("MEI + radtan calibration:", {k: round(v, 5) for k, v in c.items()})
    pts = np.random.default_rng(0).standard_normal((2000, 3))
    pts = pts[np.arccos(pts[:, 2] / np.linalg.norm(pts, axis=1)) < math.radians(MAX_FOV_DEG / 2)]
    d = np.linalg.norm(equidistant_project(pts) - mei_project(pts), axis=1)
    print(f"reprojection difference over random rays: max {d.max():.3f} px, mean {d.mean():.3f} px")
