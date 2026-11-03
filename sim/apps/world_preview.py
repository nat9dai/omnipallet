"""Build the warehouse and save a few pinhole renders to check the look and lighting.

    sim/run_isaac.sh sim/apps/world_preview.py [--x-range 0 15] [--out results/figures/preview]
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--x-range", type=float, nargs=2, default=None)
ap.add_argument("--out", default="results/figures/preview")
ap.add_argument("--res", type=int, nargs=2, default=[1600, 900])
ap.add_argument("--drone", type=float, nargs=3, default=None, help="place the Charpi airframe here (x y z, m)")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.res[0], "height": args.res[1]})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sim", "warehouse"))
sys.path.insert(0, os.path.join(ROOT, "sim", "vehicle"))
import build_world  # noqa: E402
import design  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1 / 250, rendering_dt=1 / 25)
stage = omni.usd.get_context().get_stage()
pallets = build_world.build(stage, get_assets_root_path(), x_range=tuple(args.x_range) if args.x_range else None)
print(f"[preview] {len(pallets)} pallets", flush=True)
if args.drone:
    from pxr import Gf, UsdGeom
    world.get_physics_context().set_gravity(0.0)  # parked for the picture, not flown
    prim = stage.DefinePrim("/World/drone", "Xform")
    prim.GetReferences().AddReference(design.CHARPI_USD)
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*args.drone))


def look(yaw_deg, pitch_deg):
    """World quaternion (w, x, y, z) for an Isaac 'world' camera: x forward, z up."""
    x, y, z, w = R.from_euler("ZY", [yaw_deg, -pitch_deg], degrees=True).as_quat()
    return np.array([w, x, y, z])


views = {  # name: position, yaw, pitch (deg)
    "aisle_entrance": ((-6.0, 0.0, 3.0), 0.0, -8.0),
    "aisle_low": ((2.0, 0.0, 0.6), 0.0, 0.0),
    "rack_left": ((4.0, -0.2, 2.2), 90.0, 0.0),
    "rack_high": ((6.0, 0.0, 7.0), 60.0, -15.0),
}
if args.drone:
    dx, dy, dz = args.drone
    views = {  # name: position, yaw, pitch, focal length (cm)
        "drone_behind": ((dx - 2.2, 0.35, dz + 0.9), -4.0, -20.0, 1.8),
        "drone_side": ((dx + 1.6, -0.9, dz + 0.45), 152.0, -10.0, 2.4),
        "drone_wide": ((dx - 7.0, 0.0, dz + 2.5), 0.0, -14.0, 1.4),
    }
cam = Camera(prim_path="/World/preview_cam", resolution=tuple(args.res))
world.reset()
cam.initialize()
cam.set_focal_length(1.4)
cam.set_clipping_range(0.05, 200.0)
os.makedirs(args.out, exist_ok=True)
for name, (pos, yaw, pitch, *focal) in views.items():
    cam.set_focal_length(focal[0] if focal else 1.4)
    cam.set_world_pose(np.array(pos), look(yaw, pitch), camera_axes="world")
    for _ in range(60):  # let textures stream and the denoiser settle
        world.step(render=True)
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        print("[preview] no image for", name, flush=True)
        continue
    Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(os.path.join(args.out, f"{name}.png"))
    print("[preview] saved", name, flush=True)
app.close()
