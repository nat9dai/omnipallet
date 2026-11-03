"""Record an X display (QGroundControl in Xephyr) as JPEGs plus times.txt of wall-clock grab times, for the demo video.

    Xephyr :7 -screen 1280x720 &  DISPLAY=:7 QGroundControl &
    .venv/bin/python scripts/qgc_capture.py results/runs/<name>/qgc --display :7 --hz 4
"""
import argparse
import os
import time

import cv2
import numpy as np
from Xlib import X, display


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--display", default=":7")
    ap.add_argument("--hz", type=float, default=4.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    root = display.Display(args.display).screen().root
    g = root.get_geometry()
    times = open(os.path.join(args.out, "times.txt"), "a")
    t_end, i = time.time() + args.max_s, 0
    while time.time() < t_end:
        t = time.time()
        raw = root.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xFFFFFFFF)
        img = np.frombuffer(raw.data, np.uint8).reshape(g.height, g.width, 4)[:, :, :3]  # BGRX
        cv2.imwrite(os.path.join(args.out, f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        times.write(f"{i} {t:.3f}\n")
        times.flush()
        i += 1
        time.sleep(max(0.0, 1.0 / args.hz - (time.time() - t)))


if __name__ == "__main__":
    main()
