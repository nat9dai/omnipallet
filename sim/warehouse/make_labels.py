"""Write the label textures: one unique GS1-128 strip per pallet and one shared label face.

    python sim/warehouse/make_labels.py [--tex-ppm 6]   # writes sim/warehouse/textures/
"""
import argparse
import os
import sys

import barcode
import numpy as np
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "ros2_ws", "src", "stock_drone", "stock_drone"))
import aisle_spec as spec  # noqa: E402

OUT = os.path.join(HERE, "textures")
QUIET = 10  # quiet-zone modules each side (GS1 minimum)


def gs1_modules(data):
    """GS1-128 bar pattern as a 0/1 array of modules, with quiet zones, via python-barcode."""
    code = barcode.get("gs1_128", data, writer=ImageWriter())
    pattern = code.build()[0]  # string of '1' (bar) and '0' (space), one char per module
    bits = np.array([c == "1" for c in pattern], dtype=np.uint8)
    return np.concatenate([np.zeros(QUIET, np.uint8), bits, np.zeros(QUIET, np.uint8)])


def strip_image(data, tex_ppm, height):
    bits = gs1_modules(data)
    row = np.where(np.repeat(bits, tex_ppm) == 1, 20, 245).astype(np.uint8)
    return Image.fromarray(np.tile(row, (height, 1)), "L"), len(bits)


def face_image(w_px=420, h_px=296):
    """Generic label print in the area above the barcode: header bars and grey text lines."""
    img = Image.new("L", (w_px, h_px), 245)
    d = ImageDraw.Draw(img)
    try:
        f_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        f_small = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        f_big = f_small = ImageFont.load_default()
    d.rectangle([0, 0, w_px - 1, h_px - 1], outline=40, width=3)
    d.text((14, 10), "ACME LOGISTICS", font=f_big, fill=25)
    d.line([0, 46, w_px, 46], fill=40, width=2)
    for i, txt in enumerate(["SSCC", "CONTENT", "COUNT", "BEST BEFORE"]):
        d.text((14, 56 + 26 * i), txt, font=f_small, fill=60)
        d.rectangle([150, 60 + 26 * i, 150 + 90 + 40 * (i % 3), 70 + 26 * i], fill=110)
    d.line([0, 166, w_px, 166], fill=40, width=2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-ppm", type=int, default=6, help="texels per barcode module")
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "strips"), exist_ok=True)
    face_image().save(os.path.join(OUT, "label_face.png"))
    pallets = spec.layout()
    n_mod = None
    for p in pallets:
        img, n_mod = strip_image(p.gs1, args.tex_ppm, height=96)
        img.save(os.path.join(OUT, "strips", f"{p.sscc}.png"), optimize=True)
    width_mm = n_mod * spec.X_DIM_MM
    print(f"{len(pallets)} strips, {n_mod} modules incl. quiet zones = {width_mm:.1f} mm at X = {spec.X_DIM_MM} mm, "
          f"{n_mod * args.tex_ppm} texels wide")


if __name__ == "__main__":
    main()
