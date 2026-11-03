"""Compose the demo video of one flight: replayed camera views, the QGroundControl recording and the live map.

    python scripts/demo_data.py results/runs/<run>
    python scripts/map_data.py results/runs/<run>
    sim/run_isaac.sh sim/apps/demo_replay.py --run results/runs/<run>
    .venv/bin/python scripts/make_demo_video.py results/runs/<run>   # writes videos/demo.mp4
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "stock_drone", "stock_drone"))
sys.path.insert(0, os.path.join(ROOT, "sim", "vehicle"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import aisle_spec as spec  # noqa: E402
import design as d  # noqa: E402
import maprender as mr  # noqa: E402

W, H = 2560, 1440
FOLLOW, MAP, QGC, INV = (0, 0, 1280, 720), (1280, 0, 1280, 720), (960, 1080, 640, 360), (1600, 1080, 960, 360)
FISH_Y, FISH_W, FISH_H = 720, 640, 360
BAR_Y, STRIP_W, STRIP_H, ZOOM_W = 1080, 760, 180, 200
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BG, PANEL, INK, MUTED, FAINT = (13, 15, 18), (22, 25, 31), (236, 239, 244), (150, 158, 172), (52, 57, 68)
GREEN, FRESH, RED, AMBER, WHITE = (46, 204, 113), (190, 255, 214), (231, 76, 60), (245, 176, 65), (255, 255, 255)
READ_WINDOW = 0.3  # s, a label counts as decoded this close to one of its reads
FRESH_S = 1.0  # s, highlight on a newly counted pallet
NEW_S = 2.0  # s, voxels this new are drawn bright
END_S = 6.0  # s of video held on the last frame
SKIP = 3  # the renderer's first frames are still denoising
ZOOM_HOLD = 2.0  # s, label zoom stays up after the label leaves view
BAND_C = 0.425  # fallback band centre, fraction of image height
GOD_EYE, GOD_TARGET, GOD_FOV = (-20.0, -42.0, 28.0), (23.0, 0.0, 1.5), 50.0
FACE_X, FACE_Z, FACE_PPM, FACE_SS = (-1.0, 81.0), (-0.2, 8.6), 10.6, 3  # inventory panel faces: m, px/m, supersample

_fonts = {}


def font(size, path=FONT):
    if (size, path) not in _fonts:
        _fonts[size, path] = ImageFont.truetype(path, size)
    return _fonts[size, path]


def barcode_rotation(side):
    """Rotation body <- optical (x right, y down, z forward); same as sim/vehicle/cameras.py."""
    if side == "left":
        return np.column_stack(([1, 0, 0], [0, 0, -1], [0, 1, 0])).astype(float)
    return np.column_stack(([-1, 0, 0], [0, 0, -1], [0, -1, 0])).astype(float)


def label_corners(p):
    x, y, z = p.label_pose
    hw, hh = spec.LABEL_W / 2, spec.LABEL_H / 2
    return np.array([[x - hw, y, z + hh], [x + hw, y, z + hh], [x + hw, y, z - hh], [x - hw, y, z - hh]])


def project(pts, body_p, body_r, side, img_w):
    """Pixels (N, 2) of world points in a barcode image img_w wide, or None if any point is behind."""
    r_wc = body_r @ barcode_rotation(side)
    t_wc = body_p + body_r @ np.array(d.BARCODE_POS[side])
    pc = (pts - t_wc) @ r_wc  # (N, 3) optical frame, m
    if np.any(pc[:, 2] < 0.1):
        return None
    s = img_w / d.BARCODE_RES[0]
    f = d.BARCODE_F_PX * s
    return np.column_stack([f * pc[:, 0] / pc[:, 2] + img_w / 2, f * pc[:, 1] / pc[:, 2] + d.BARCODE_RES[1] * s / 2])


def tag(draw, x, y, words, size=17):
    f = font(size, BOLD)
    w = draw.textlength(words, font=f)
    draw.rounded_rectangle((x, y, x + w + 16, y + size + 10), 5, fill=(0, 0, 0, 150))
    draw.text((x + 8, y + 4), words, font=f, fill=INK)


def rgb(path):
    return Image.fromarray(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))


class FaceMaps:
    """Both rack faces of the dense map seen straight on, built up incrementally as voxels join."""

    def __init__(self, p, col):
        self.p, self.col = p, col
        self.w = int((FACE_X[1] - FACE_X[0]) * FACE_PPM * FACE_SS)
        self.h = int((FACE_Z[1] - FACE_Z[0]) * FACE_PPM * FACE_SS)
        self.z = {s: np.full(self.w * self.h, np.inf) for s in "LR"}
        self.c = {s: np.zeros((self.w * self.h, 3), np.float32) for s in "LR"}
        self.n = 0  # voxels added so far; map arrays are sorted by join time

    def add_until(self, k):
        if k <= self.n:
            return
        q, c = self.p[self.n:k], self.col[self.n:k]
        self.n = k
        u = ((q[:, 0] - FACE_X[0]) * FACE_PPM * FACE_SS).astype(np.int64)
        v = ((FACE_Z[1] - q[:, 2]) * FACE_PPM * FACE_SS).astype(np.int64)
        ok = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        for s, m in (("L", ok & (q[:, 1] > 0.3)), ("R", ok & (q[:, 1] < -0.3))):
            pix, dep = v[m] * self.w + u[m], np.abs(q[m, 1])
            order = np.argsort(-dep)  # far first so nearer voxels overwrite
            pix, dep, cc = pix[order], dep[order], c[m][order]
            closer = dep < self.z[s][pix]
            pix, dep, cc = pix[closer], dep[closer], cc[closer]
            self.z[s][pix] = dep
            self.c[s][pix] = cc

    def image(self, side):
        img = self.c[side].reshape(self.h, self.w, 3)
        empty = ~np.isfinite(self.z[side].reshape(self.h, self.w))
        img = np.where(empty[..., None], np.float32(BG), img)
        small = cv2.resize(img, (self.w // FACE_SS, self.h // FACE_SS), interpolation=cv2.INTER_AREA)
        return np.clip(small, 0, 255).astype(np.uint8)


def to_face_px(x, z):
    return (x - FACE_X[0]) * FACE_PPM, (FACE_Z[1] - z) * FACE_PPM


class QgcFeed:
    """QGroundControl frames recorded in wall-clock time, looked up by sim time through the bag's clock map."""

    def __init__(self, run, clock):
        self.dir = os.path.join(run, "qgc")
        path = os.path.join(self.dir, "times.txt")
        self.ok = os.path.exists(path) and clock is not None and len(clock) > 1
        if self.ok:
            a = np.loadtxt(path)
            self.idx, self.wall = a[:, 0].astype(int), a[:, 1]
            self.clock = clock
            self.rtf = np.ptp(clock[:, 0]) / np.ptp(clock[:, 1])  # sim seconds per wall second

    def frame(self, t):
        if not self.ok:
            return None
        w = np.interp(t, self.clock[:, 0], self.clock[:, 1])
        k = max(0, int(np.searchsorted(self.wall, w, side="right")) - 1)
        return os.path.join(self.dir, f"{self.idx[k]:06d}.jpg")


class Flight:
    """Everything one frame needs, loaded once per worker process."""

    def __init__(self, run):
        self.run = run
        demo = os.path.join(run, "demo")
        self.fr = os.path.join(demo, "frames")
        f = np.load(os.path.join(demo, "flight.npz"))
        self.gt, self.nav, self.reads = f["gt"], f["nav"], f["reads"]
        self.ssccs = [str(s) for s in f["ssccs"]]
        self.qgc = QgcFeed(run, f["clock"] if "clock" in f else None)
        air = self.gt[self.gt[:, 3] > 0.05, 0]
        self.t_off, self.t_land = air[0], air[-1]
        if os.path.exists(os.path.join(self.fr, "frames.json")):
            self.meta = json.load(open(os.path.join(self.fr, "frames.json")))
        else:  # render still running: the replay's default timing
            ts = np.arange(self.t_off - 2.0, self.t_land + 2.0, 4.0 / 25.0)
            self.meta = {"speed": 4.0, "fps": 25.0, "times": list(ts)}
        self.times = np.array(self.meta["times"])
        self.slerp = Slerp(self.gt[:, 0], R.from_quat(self.gt[:, 4:8]))
        self.pallets = spec.layout()
        self.by_sscc = {p.sscc: p for p in self.pallets}
        self.ev = json.load(open(os.path.join(run, "eval.json")))
        self.summary = json.load(open(os.path.join(run, "mission_summary.json")))
        self.missed = set(self.ev["missed"])
        self.inv = {r["sscc"]: r for r in csv.DictReader(open(os.path.join(run, "inventory.csv")))}
        self.first_read = {}  # only codes the inventory kept (misreads dropped)
        for r in self.reads:
            s_ = self.ssccs[int(r[5])]
            if self.inv.get(s_, {}).get("status") == "ok":
                self.first_read.setdefault(s_, r[0])
        self.new_reads = sorted((t, s) for s, t in self.first_read.items())
        self.new_t = np.array([t for t, _ in self.new_reads])
        m = np.load(os.path.join(demo, "map.npz"))
        order = np.argsort(m["dense_t"], kind="stable")
        self.map_p, self.map_t = m["dense_p"][order].astype(np.float64), m["dense_t"][order]
        g = m["dense_g"][order].astype(np.float64)
        lo, hi = np.percentile(g, [2, 98])
        gs = np.clip((g - lo) / (hi - lo) * 255, 0, 255)
        self.map_col = (0.35 * mr.height_colours(self.map_p) + 0.65 * gs[:, None]).astype(np.float32)
        self.faces = FaceMaps(self.map_p, self.map_col)
        self.zooms, self.band = {}, {}

    def pose(self, t):
        p = np.array([np.interp(t, self.gt[:, 0], self.gt[:, k]) for k in (1, 2, 3)])
        rot = self.slerp([np.clip(t, self.gt[0, 0], self.gt[-1, 0])]).as_matrix()[0]
        return p, rot

    def nav_at(self, t):
        return np.array([np.interp(t, self.nav[:, 0], self.nav[:, k]) for k in (1, 2, 3)])

    def barcode(self, canvas, i, t, p, rot, end, draw=True):
        near = self.reads[np.abs(self.reads[:, 0] - t) < READ_WINDOW]
        for side, key, cam, oy in (("left", "bl", 0, BAR_Y), ("right", "br", 1, BAR_Y + STRIP_H)):
            img = rgb(os.path.join(self.fr, key, f"{i:05d}.jpg"))
            s = STRIP_W / img.width
            # band follows the label row being scanned (cm of height error shift it by 100s of px), smoothed over 0.3 s
            hb = STRIP_H / s  # band height, image px
            lz = np.array([spec.label_z(k) for k in range(spec.LEVELS)])
            z_cam = p[2] + (rot @ np.array(d.BARCODE_POS[side]))[2]
            face_y = spec.WIDTH / 2 * (1 if side == "left" else -1)
            uv_l = project(np.array([[p[0], face_y, lz[np.argmin(np.abs(lz - z_cam))]]]), p, rot, side, img.width)
            want = BAND_C * img.height if uv_l is None else uv_l[0, 1] - 0.08 * hb
            prev = self.band.get(side)
            if prev is None or not 0 < t - prev[1] < 1.0:
                vc = want
            else:
                vc = prev[0] + (want - prev[0]) * min(1.0, (t - prev[1]) / 0.3)
            self.band[side] = (vc, t)
            v0 = int(np.clip(vc - hb / 2, 0, img.height - hb))
            shown = []
            for sscc in sorted({self.ssccs[int(r[5])] for r in near if int(r[1]) == cam}):
                pal = self.by_sscc.get(sscc)
                uv = None if pal is None else project(label_corners(pal), p, rot, side, img.width)
                if uv is None or not (0 < uv[:, 0].mean() < img.width):
                    continue
                shown.append((sscc, pal, uv))
            if shown:
                sscc, pal, uv = min(shown, key=lambda e: abs(e[2][:, 0].mean() - img.width / 2))
                c = uv.mean(axis=0)
                half = np.ptp(uv[:, 0]) * 0.75
                crop = img.crop((int(c[0] - half), int(c[1] - half * 0.72), int(c[0] + half), int(c[1] + half * 0.72)))
                self.zooms[side] = (crop.resize((ZOOM_W - 14, 128), Image.LANCZOS), sscc, t)
            if not draw:
                continue
            band = img.crop((0, v0, img.width, v0 + int(round(STRIP_H / s))))
            canvas.paste(band.resize((STRIP_W, STRIP_H), Image.BILINEAR), (0, oy))
            dr = ImageDraw.Draw(canvas, "RGBA")
            for sscc, pal, uv in shown:
                poly = [(a * s, oy + (b - v0) * s) for a, b in uv]
                new = 0 <= t - self.first_read[sscc] < FRESH_S
                dr.polygon(poly, outline=FRESH if new else GREEN, width=3)
                words = pal.location[4:]
                ftxt = font(14, BOLD)
                wpx = dr.textlength(words, font=ftxt)
                bx = min(max(np.mean([a for a, _ in poly]) - wpx / 2 - 5, 2), STRIP_W - wpx - 12)
                by = max(oy + 2, min(poly[0][1], poly[1][1]) - 22)
                dr.rectangle((bx, by, bx + wpx + 10, by + 19), fill=(40, 120, 70, 235) if new else (10, 60, 30, 220))
                dr.text((bx + 5, by + 2), words, font=ftxt, fill=INK)
            zx = STRIP_W
            dr.rectangle((zx, oy, zx + ZOOM_W, oy + STRIP_H), fill=PANEL)
            dr.text((zx + ZOOM_W / 2, oy + 6), f"{side} barcode camera", font=font(13, BOLD), fill=INK, anchor="ma")
            if side in self.zooms and 0 <= t - self.zooms[side][2] < ZOOM_HOLD and not end:
                crop, sscc, _ = self.zooms[side]
                canvas.paste(crop, (zx + 7, oy + 26))
                dr = ImageDraw.Draw(canvas, "RGBA")
                dr.text((zx + ZOOM_W / 2, oy + 158), "(00) " + sscc, font=font(12, MONO), fill=GREEN, anchor="ma")
            else:
                dr.text((zx + ZOOM_W / 2, oy + STRIP_H / 2), "no label in view", font=font(14), fill=MUTED,
                        anchor="mm")

    def map3d(self, canvas, i, t):
        k = int(np.searchsorted(self.map_t, t, side="right"))
        p, col, tj = self.map_p[:k], self.map_col[:k], self.map_t[:k]
        col = np.where((tj > t - NEW_S)[:, None], 0.3 * col + 0.7 * np.float32(WHITE), col)
        size = MAP[2:]
        img = mr.render_ghost(p, col, p[:, 1] < 0, eye=GOD_EYE, target=GOD_TARGET, size=size, fov_deg=GOD_FOV,
                              ghost=0.35)
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, "RGBA")
        counted = [self.inv[s] for s, tt in self.first_read.items() if tt <= t]
        if counted:
            lab = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in counted])
            u, v, z = mr.project(lab, GOD_EYE, GOD_TARGET, size, GOD_FOV)
            for a, b, yy in zip(u, v, lab[:, 1]):
                dr.rectangle((a - 1, b - 1, a + 1, b + 1), fill=GREEN + ((235,) if yy > 0 else (90,)))
        if t > self.t_off:
            path_t = np.arange(self.t_off, t, 0.5)
            if len(path_t) > 1:
                path = np.column_stack([np.interp(path_t, self.nav[:, 0], self.nav[:, c]) for c in (1, 2, 3)])
                u, v, _ = mr.project(path, GOD_EYE, GOD_TARGET, size, GOD_FOV)
                dr.line(list(zip(u, v)), fill=AMBER + (200,), width=2)
            d = self.nav_at(t)
            u, v, _ = mr.project(d[None], GOD_EYE, GOD_TARGET, size, GOD_FOV)
            dr.ellipse((u[0] - 9, v[0] - 9, u[0] + 9, v[0] + 9), outline=AMBER, width=3)
            dr.ellipse((u[0] - 3, v[0] - 3, u[0] + 3, v[0] + 3), fill=AMBER)
        canvas.paste(im, MAP[:2])
        dr = ImageDraw.Draw(canvas, "RGBA")
        tag(dr, MAP[0] + 10, MAP[1] + 10, "SLAM map of the whole aisle, built during the flight")
        tag(dr, MAP[0] + 10, MAP[1] + MAP[3] - 66, "Near (right) rack drawn see-through so the left face shows",
            size=14)
        tag(dr, MAP[0] + 10, MAP[1] + MAP[3] - 38, f"{k / 1e6:.2f} M voxels of 3 cm.  White: joined in the last "
            f"{NEW_S:.0f} s.  Green: counted labels.  Orange: drone and its path", size=14)

    def inventory(self, canvas, t, p_nav, end):
        x0, y0, w, h = INV
        dr = ImageDraw.Draw(canvas, "RGBA")
        dr.rectangle((x0, y0, x0 + w, y0 + h), fill=PANEL)
        counted = {s for s, tt in self.first_read.items() if tt <= t}
        n = len(counted)
        dr.text((x0 + 16, y0 + 10), "pallets counted", font=font(15), fill=MUTED)
        big = font(40, BOLD)
        dr.text((x0 + 16, y0 + 28), f"{n}", font=big, fill=GREEN)
        dr.text((x0 + 24 + dr.textlength(f"{n}", font=big), y0 + 46), f"/ {len(self.pallets)}", font=font(20),
                fill=MUTED)
        ft = max(0.0, min(t, self.t_land) - self.t_off)
        p = p_nav
        if t < self.t_off:
            phase = "on the dock"
        elif t > self.t_land:
            phase = "landed"
        elif p[0] < -1.0:
            phase = "taking off" if t < self.new_t[0] else "landing on the dock"
        else:
            lz = np.array([spec.label_z(kk) for kk in range(spec.LEVELS)])
            kk = int(np.argmin(np.abs(lz - p[2])))
            vx = (np.interp(t + 0.25, self.nav[:, 0], self.nav[:, 1]) - np.interp(t - 0.25, self.nav[:, 0],
                                                                                    self.nav[:, 1])) / 0.5
            if abs(lz[kk] - p[2]) < 0.25:
                phase = f"scanning level {kk + 1} of 5" if abs(vx) > 0.15 else "turning at the aisle end"
            else:
                phase = "climbing to the next level"
        rows = [("flight time", f"{int(ft // 60)}:{int(ft % 60):02d} (sim clock)"), ("now", phase)]
        gt_p = np.array([np.interp(t, self.gt[:, 0], self.gt[:, k]) for k in (1, 2, 3)])
        if self.t_off < t < self.t_land:
            e = p - gt_p
            rows.append(("nav error", f"along {e[0]:+.2f}  side {e[1]:+.2f}  height {e[2]:+.2f} m"))
        for j, (a, b) in enumerate(rows):
            dr.text((x0 + 230, y0 + 12 + j * 22), a, font=font(14), fill=MUTED)
            dr.text((x0 + 330, y0 + 12 + j * 22), b, font=font(14, BOLD), fill=INK)
        dr.text((x0 + 700, y0 + 12), f"video at {self.meta['speed']:.0f}x speed", font=font(14), fill=MUTED)
        dr.text((x0 + 700, y0 + 34), "no GPS", font=font(14, BOLD), fill=AMBER)
        top = {"L": y0 + 88, "R": y0 + 222}
        fx = x0 + 52
        for side, name in (("L", "left rack"), ("R", "right rack")):
            img = Image.fromarray(self.faces.image(side))
            canvas.paste(img, (fx, top[side]))
            dr = ImageDraw.Draw(canvas, "RGBA")
            dr.text((x0 + 12, top[side] - 2), name.split()[0], font=font(13, BOLD), fill=MUTED)
            for s in counted | (self.missed if end else set()):
                r = self.inv.get(s)
                pal = self.by_sscc.get(s)
                if pal is None or (r and r["side"] != side) or (r is None and pal.side != side):
                    continue
                if r is not None:
                    xm, lv = spec.slot_x(int(r["bay"]), int(r["slot"])), int(r["level"])
                else:
                    xm, lv = pal.x, pal.level
                ax, ay = to_face_px(xm, spec.label_z(lv))
                col = RED if s in self.missed else FRESH if t - self.first_read[s] < FRESH_S else GREEN
                r_ = 3 if s in self.missed else 2
                dr.rectangle((fx + ax - r_, top[side] + ay - r_, fx + ax + r_, top[side] + ay + r_), fill=col)
            if self.t_off < t < self.t_land:
                ax, ay = to_face_px(p[0], p[2])
                dr.line((fx + ax, top[side] + ay - 9, fx + ax, top[side] + ay + 9), fill=AMBER, width=3)
        ty = y0 + h - 22
        for xm in (0, 20, 40, 60, 80):
            ax, _ = to_face_px(xm, 0)
            dr.text((fx + ax, ty), f"{xm} m", font=font(12), fill=MUTED, anchor="ma")

    def compose(self, i, end=False, faces_ready=False):
        t = self.times[i]
        canvas = Image.new("RGB", (W, H), BG)
        p, rot = self.pose(t)
        p_nav = self.nav_at(t)
        canvas.paste(rgb(os.path.join(self.fr, "chase", f"{i:05d}.jpg")).resize(FOLLOW[2:], Image.BILINEAR),
                     FOLLOW[:2])
        dr = ImageDraw.Draw(canvas, "RGBA")
        tag(dr, FOLLOW[0] + 12, FOLLOW[1] + 12, "Follow camera")
        self.map3d(canvas, i, t)
        # fisheyes: the file holds cam0 FR, cam1 FL, cam2 RL, cam3 RR; shown left to right round the drone
        fish = rgb(os.path.join(self.fr, "fish", f"{i:05d}.jpg"))
        cells = {"FR": (0, 0), "FL": (0, 1), "RL": (1, 0), "RR": (1, 1)}
        for j, (k, name) in enumerate((("RL", "rear left"), ("FL", "front left"), ("FR", "front right"),
                                       ("RR", "rear right"))):
            r_, c_ = cells[k]
            canvas.paste(fish.crop((c_ * 640, r_ * 360, c_ * 640 + 640, r_ * 360 + 360)), (j * FISH_W, FISH_Y))
            dr = ImageDraw.Draw(canvas, "RGBA")
            tag(dr, j * FISH_W + 10, FISH_Y + 10, f"fisheye, {name}", size=15)
        self.barcode(canvas, i, t, p, rot, end)
        q = self.qgc.frame(t)
        if q is not None:
            canvas.paste(rgb(q).resize(QGC[2:], Image.LANCZOS), QGC[:2])
        dr = ImageDraw.Draw(canvas, "RGBA")
        if self.qgc.ok:
            tag(dr, QGC[0] + 8, QGC[1] + QGC[3] - 64, "QGroundControl, recorded live", size=14)
            tag(dr, QGC[0] + 8, QGC[1] + QGC[3] - 36, f"wall clock, sim at {self.qgc.rtf:.2f}x real time", size=12)
        if not faces_ready:
            self.faces.add_until(int(np.searchsorted(self.map_t, t, side="right")))
        self.inventory(canvas, t, p_nav, end)
        dr = ImageDraw.Draw(canvas, "RGBA")
        for line in ((1280, 0, 1280, 720), (0, 720, W, 720), (0, 1080, W, 1080),
                     (640, 720, 640, 1080), (1280, 720, 1280, 1080), (1920, 720, 1920, 1080), (960, 1080, 960, H),
                     (1600, 1080, 1600, H), (0, 1260, 960, 1260), (STRIP_W, 1080, STRIP_W, H)):
            dr.line(line, fill=BG, width=2)
        return canvas

    def end_card(self, base):
        ev, summary = self.ev, self.summary
        top_only = all(self.by_sscc[s].level == spec.LEVELS - 1 for s in self.missed if s in self.by_sscc)
        lines = [(f"{ev['pallets_read']} of {ev['pallets_true']} pallets counted", 48, GREEN),
                 (f"{ev['slot_accuracy'] * 100:.0f}% in the right slot. Median label position error "
                  f"{ev['pos_err_median_m'] * 100:.0f} cm", 24, INK),
                 (f"Flight {summary['flight_min']:.1f} min, {summary['path_m']:.0f} m, no GPS. "
                  f"Map: {len(self.map_p) / 1e6:.1f} M voxels of 3 cm", 21, MUTED)]
        if self.missed:
            lines.append((f"The {len(self.missed)} missed labels are the red dots in the inventory panel"
                          + (", all on the top level" if top_only else ""), 21, MUTED))
        im = base.copy()
        x0, y0, w, h = FOLLOW
        region = im.crop((x0, y0, x0 + w, y0 + h))
        im.paste(Image.blend(region, Image.new("RGB", region.size, BG), 0.72), (x0, y0))
        dr = ImageDraw.Draw(im)
        y = y0 + h / 2 - sum(s + 16 for _, s, _ in lines) / 2
        for words, size, colour in lines:
            dr.text((x0 + w / 2, y), words, font=font(size, BOLD if size >= 30 else FONT), fill=colour, anchor="ma")
            y += size + 16
        return im


_flight = None


def _init(run):
    global _flight
    cv2.setNumThreads(1)
    _flight = Flight(run)


def _work(job):
    """Compose frames [a, b) into out_dir; the faces and zooms are primed from the frames before a."""
    a, b, out_dir = job
    fl = _flight
    for i in range(max(SKIP, a - int(ZOOM_HOLD * fl.meta["fps"] / fl.meta["speed"]) - 1), a):
        p, rot = fl.pose(fl.times[i])
        fl.barcode(None, i, fl.times[i], p, rot, False, draw=False)
    for i in range(a, b):
        fl.compose(i).save(os.path.join(out_dir, f"{i - SKIP:05d}.jpg"), quality=93)
    return b - a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="write these composites as JPEGs instead")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crf", type=int, default=22)
    args = ap.parse_args()
    demo = os.path.join(args.run, "demo")
    if args.frames:
        fl = Flight(args.run)
        for i in args.frames:
            fl.compose(i).save(os.path.join(demo, f"composite_{i:05d}.jpg"), quality=92)
        fl.end_card(fl.compose(args.frames[-1], end=True)).save(os.path.join(demo, "composite_end.jpg"), quality=92)
        print("wrote", len(args.frames) + 1, "composites to", demo)
        return
    out_dir = os.path.join(demo, "composite")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    n = len(json.load(open(os.path.join(demo, "frames", "frames.json")))["times"])
    edges = np.linspace(SKIP, n, args.workers * 3 + 1).astype(int)
    jobs = [(int(a), int(b), out_dir) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    done = 0
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(args.run,)) as pool:
        for k in pool.imap_unordered(_work, jobs):
            done += k
            print(f"composed {done}/{n - SKIP}", flush=True)
    fl = Flight(args.run)
    fl.faces.add_until(len(fl.map_p))
    last = fl.compose(n - 1, end=True, faces_ready=True)
    card = fl.end_card(last)
    fps = fl.meta["fps"]
    for j in range(int(END_S * fps)):
        Image.blend(last, card, min(1.0, j / 15)).save(os.path.join(out_dir, f"{n - SKIP + j:05d}.jpg"), quality=93)
    out = args.out or os.path.join(ROOT, "videos", "demo.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(out_dir, "%05d.jpg"),
                    "-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", out], check=True)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
