"""Draw point-cloud maps with a pinhole camera in NumPy (z-buffered square splats, no GPU), for the demo video.

    img = render(points, colours, eye=(-8, -7, 11), target=(40, 0, 3), size=(960, 540), fov_deg=50)
"""
import numpy as np


def turbo(x):
    """Turbo colour map, x in [0, 1] -> (N, 3) RGB floats in [0, 1] (polynomial fit by Google)."""
    x = np.clip(x, 0, 1)[:, None]
    c = [[0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
         [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
         [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973]]
    p = np.concatenate([x ** k for k in range(6)], axis=1)
    return np.clip(p @ np.array(c).T, 0, 1)


def height_colours(p, gray=None, z_lo=0.0, z_hi=8.5):
    """RGB uint8 by height, darkened by the camera's gray level so the texture of the loads shows."""
    rgb = turbo((p[:, 2] - z_lo) / (z_hi - z_lo))
    if gray is not None:
        rgb = rgb * (0.45 + 0.55 * (gray[:, None] / 255.0))
    return (rgb * 255).astype(np.uint8)


def look_at(eye, target, up=(0, 0, 1)):
    """Rotation world -> camera (rows: x right, y down, z forward)."""
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def project(p, eye, target, size, fov_deg):
    """Pixel (u, v) and depth of world points for a camera at eye looking at target."""
    w, h = size
    r = look_at(eye, target)
    pc = (p - np.asarray(eye, float)) @ r.T
    f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = f * pc[:, 0] / z + w / 2
        v = f * pc[:, 1] / z + h / 2
    return u, v, z


def render(p, rgb, eye, target, size=(960, 540), fov_deg=50, splat=2.0, bg=(13, 15, 18), near=0.3, edl=1.0):
    """Render points to an (H, W, 3) uint8 image; a point 5 m away covers about splat px."""
    w, h = size
    u, v, z = project(p, eye, target, size, fov_deg)
    ok = (z > near) & (u > -3) & (u < w + 3) & (v > -3) & (v < h + 3)
    ui, vi, z, c = np.round(u[ok]).astype(np.int64), np.round(v[ok]).astype(np.int64), z[ok], rgb[ok]
    rad = np.clip(np.round(splat * 5.0 / z / 2), 0, 2).astype(np.int64)  # square half-size, px
    pix, dep, col = [], [], []
    for r in range(3):
        m = rad >= r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue  # each ring once
                uu, vv = ui[m] + dx, vi[m] + dy
                inb = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                pix.append(vv[inb] * w + uu[inb])
                dep.append(z[m][inb])
                col.append(c[m][inb])
    pix, dep, col = np.concatenate(pix), np.concatenate(dep), np.concatenate(col)
    zbuf = np.full(w * h, np.inf)
    np.minimum.at(zbuf, pix, dep)
    win = dep <= zbuf[pix]
    img = np.empty((h * w, 3), np.float32)
    img[:] = bg
    img[pix[win]] = col[win]
    img = img.reshape(h, w, 3)
    if edl:
        z2 = zbuf.reshape(h, w)
        hit = np.isfinite(z2)
        img[hit] *= eye_dome(z2, edl)[hit][:, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def eye_dome(zbuf, strength=1.0, step=1):
    """Eye-dome lighting factor (H, W) in (0, 1] from depth zbuf (inf = empty); darkens depth edges."""
    ld = np.log2(np.where(np.isfinite(zbuf), zbuf, 1e3))
    pad = np.pad(ld, step, mode="edge")
    h, w = ld.shape
    acc = np.zeros_like(ld)
    for dy, dx in ((-step, 0), (step, 0), (0, -step), (0, step), (-step, -step), (step, step), (-step, step),
                   (step, -step)):
        nb = pad[step + dy:step + dy + h, step + dx:step + dx + w]
        acc += np.maximum(0.0, ld - nb)
    return np.exp(-60.0 * strength * acc / 8)


def render_face(p, rgb, side, x_range=(-4.0, 84.0), z_range=(-0.3, 9.0), px_per_m=12.0, bg=(13, 15, 18)):
    """Orthographic view of rack face "L" (y > 0) or "R" (y < 0); returns (H, W, 3) uint8 and (x, z) -> px."""
    m = p[:, 1] > 0.3 if side == "L" else p[:, 1] < -0.3
    q, c = p[m], rgb[m]
    w = int((x_range[1] - x_range[0]) * px_per_m)
    h = int((z_range[1] - z_range[0]) * px_per_m)
    to_px = lambda x, z: ((x - x_range[0]) * px_per_m, (z_range[1] - z) * px_per_m)  # noqa: E731
    u, v = to_px(q[:, 0], q[:, 2])
    ui, vi = u.astype(np.int64), v.astype(np.int64)
    dep = np.abs(q[:, 1])
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    pix, dep, c = vi[ok] * w + ui[ok], dep[ok], c[ok]
    zbuf = np.full(w * h, np.inf)
    np.minimum.at(zbuf, pix, dep)
    win = dep <= zbuf[pix]
    img = np.empty((h * w, 3), np.uint8)
    img[:] = bg
    img[pix[win]] = c[win]
    return img.reshape(h, w, 3), to_px


def render_ghost(p, rgb, near_mask, eye, target, size=(1280, 720), fov_deg=50, splat=1.2, ghost=0.3,
                 bg=(13, 15, 18), edl=1.0):
    """Render with near_mask points see-through (weight ghost), so the far rack shows behind the near one."""
    far = render(p[~near_mask], rgb[~near_mask], eye, target, size, fov_deg, splat, bg, edl=edl).astype(np.float32)
    nr = render(p[near_mask], rgb[near_mask], eye, target, size, fov_deg, splat, bg, edl=edl).astype(np.float32)
    cover = np.any(nr != np.float32(bg), axis=2)
    out = far.copy()
    out[cover] = (1 - ghost) * far[cover] + ghost * nr[cover]
    return np.clip(out, 0, 255).astype(np.uint8)
