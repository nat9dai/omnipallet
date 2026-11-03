"""Procedural floor, wall and roof textures, with stains and wear so the fisheye VIO has corners to track.

    python sim/warehouse/make_textures.py   # writes sim/warehouse/textures/{floor,wall,ceiling}.png
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "textures")
RNG = np.random.default_rng(3)


def fractal_noise(n, octaves=6):
    """Sum of upsampled random grids, values roughly in [-1, 1]."""
    img = np.zeros((n, n), np.float32)
    amp = 1.0
    for o in range(octaves):
        s = 2 ** (o + 2)
        g = RNG.standard_normal((s, s)).astype(np.float32)
        img += amp * np.array(Image.fromarray(g).resize((n, n), Image.BICUBIC))
        amp *= 0.55
    return img / np.abs(img).max()


def floor(n=2048):
    """One 6 m x 6 m concrete tile with a saw-cut joint on two edges, stains and specks."""
    base = 128 + 22 * fractal_noise(n) + 6 * RNG.standard_normal((n, n))
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB")
    stains = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    ds = ImageDraw.Draw(stains)
    for _ in range(45):  # oil stains, soft edged
        x, y, r = RNG.integers(0, n, 2).tolist() + [int(RNG.integers(20, 120))]
        ds.ellipse([x - r, y - int(r * RNG.uniform(0.4, 0.9)), x + r, y + r // 2], fill=(45, 42, 38, int(RNG.integers(15, 45))))
    for _ in range(25):  # tyre marks
        x0, y0 = RNG.integers(0, n, 2)
        ds.line([x0, y0, x0 + RNG.integers(-600, 600), y0 + RNG.integers(-80, 80)], fill=(30, 30, 30, 35),
                width=int(RNG.integers(6, 16)))
    img = Image.alpha_composite(img.convert("RGBA"), stains.filter(ImageFilter.GaussianBlur(10))).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(4000):
        x, y = RNG.integers(0, n, 2)
        c = int(RNG.integers(60, 200))
        d.point((x, y), fill=(c, c, c, 255))
    d.line([0, 2, n, 2], fill=(55, 55, 55, 255), width=5)
    d.line([2, 0, 2, n], fill=(55, 55, 55, 255), width=5)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def wall(n=2048):
    """Insulated metal wall panels: vertical ribs every 1 m (tile = 4 m wide), grime towards the floor."""
    x = np.arange(n)
    ribs = 200 - 40 * (np.sin(2 * np.pi * x / (n / 4)) > 0.85)
    base = np.tile(ribs, (n, 1)).astype(np.float32) + 10 * fractal_noise(n)
    grime = np.linspace(0, 45, n)[:, None]
    img = np.clip(base - grime, 0, 255).astype(np.uint8)
    rgb = np.stack([img * 0.92, img * 0.95, img * 1.0], -1).astype(np.uint8)
    out = Image.fromarray(rgb)
    d = ImageDraw.Draw(out)
    for k in range(4):  # panel seams and fasteners
        d.line([k * n // 4, 0, k * n // 4, n], fill=(90, 90, 95), width=4)
        for y in range(40, n, 160):
            d.ellipse([k * n // 4 + 12, y, k * n // 4 + 20, y + 8], fill=(70, 70, 75))
    return out


def ceiling(n=2048):
    """Trapezoidal steel roof deck with purlins every 1.5 m (tile = 6 m)."""
    x = np.arange(n)
    deck = 150 + 35 * np.sign(np.sin(2 * np.pi * x / (n / 30)))
    base = np.tile(deck, (n, 1)).astype(np.float32) + 8 * fractal_noise(n)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    for k in range(4):
        y = k * n // 4 + 10
        d.rectangle([0, y, n, y + 26], fill=(60, 62, 70))
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("floor", floor), ("wall", wall), ("ceiling", ceiling)]:
        fn().save(os.path.join(OUT, f"{name}.png"))
        print("wrote", name)


if __name__ == "__main__":
    main()
