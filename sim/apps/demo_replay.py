"""Replay a recorded flight's ground truth and render the demo video frames into <run>/demo/frames/.

    python scripts/demo_data.py results/runs/<run>
    sim/run_isaac.sh sim/apps/demo_replay.py --run results/runs/<run> [--speed 4 --fps 25]
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--speed", type=float, default=4.0, help="video seconds per simulated second, inverted")
ap.add_argument("--fps", type=float, default=25.0)
ap.add_argument("--pad", type=float, default=2.0, help="seconds shown before take-off and after landing")
ap.add_argument("--barcode-scale", type=int, default=3, help="render the 12 MP barcode cameras this much smaller")
ap.add_argument("--chase-res", type=int, nargs=2, default=[1280, 720])
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--t-start", type=float, default=None, help="start here instead of before take-off (tests)")
ap.add_argument("--x-range", type=float, nargs=2, default=None, help="build only these pallets (tests)")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "anti_aliasing": 0})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
from scipy.spatial.transform import Rotation as R, Slerp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for sub in ("sim/apps", "sim/warehouse", "sim/vehicle", "ros2_ws/src/stock_drone/stock_drone"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import build_world  # noqa: E402
import cameras  # noqa: E402
import design as d  # noqa: E402
import simsettings  # noqa: E402

PROP_DIR = {"prop_front_right_1": 1, "prop_rear_left_1": 1, "prop_front_left_1": -1, "prop_rear_right_1": -1}
SPIN_DEG = 55.0  # deg per video frame, for show (real rotor speed aliases at 25 fps)
PREFIX = "/World/stockdrone"


def set_airframe_visible(stage, visible):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(PREFIX)):
        if prim.GetName() == "visuals" and prim.IsA(UsdGeom.Imageable):
            img = UsdGeom.Imageable(prim)
            img.MakeVisible() if visible else img.MakeInvisible()


def look(yaw_deg, pitch_deg):
    """World quaternion (w, x, y, z) for an Isaac 'world' camera: x forward, z up."""
    x, y, z, w = R.from_euler("ZY", [yaw_deg, -pitch_deg], degrees=True).as_quat()
    return np.array([w, x, y, z])


class Chase:
    """A follow camera behind the drone that swings round its side when the pass direction flips."""

    def __init__(self):
        self.theta, self.dir, self.pos = 0.0, 1.0, None

    def update(self, p, vx, dt):
        if abs(vx) > 0.3:
            self.dir = np.sign(vx)
        target = 0.0 if self.dir > 0 else np.pi
        self.theta += np.clip(target - self.theta, -np.pi / 3 * dt, np.pi / 3 * dt)  # half turn in 3 s
        c, s = np.cos(self.theta), np.sin(self.theta)
        want = p + np.array([-1.6 * c, -(0.45 + 0.4 * s), 0.5])  # stays inside the 2.5 m aisle
        self.pos = want if self.pos is None else self.pos + (want - self.pos) * min(1.0, dt / 0.5)
        aim = p + np.array([0.4 * c, 0.0, -0.05]) - self.pos
        yaw = np.degrees(np.arctan2(aim[1], aim[0]))
        pitch = np.degrees(np.arctan2(aim[2], np.hypot(aim[0], aim[1])))
        return self.pos, look(yaw, pitch)


def main():
    f = np.load(os.path.join(args.run, "demo", "flight.npz"))
    gt = f["gt"]
    air = gt[gt[:, 3] > 0.05, 0]
    t0 = args.t_start if args.t_start is not None else air[0] - args.pad
    times = np.arange(t0, air[-1] + args.pad, args.speed / args.fps)
    if args.max_frames:
        times = times[:args.max_frames]
    slerp = Slerp(gt[:, 0], R.from_quat(gt[:, 4:8]))
    out = os.path.join(args.run, "demo", "frames")
    for sub in ("chase", "fish", "bl", "br"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    simsettings.make_rendering_synchronous()
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 250, rendering_dt=1 / 25)
    world.get_physics_context().set_gravity(0.0)
    stage = omni.usd.get_context().get_stage()
    build_world.build(stage, get_assets_root_path(), x_range=tuple(args.x_range) if args.x_range else None)
    root = stage.DefinePrim(PREFIX, "Xform")
    root.GetReferences().AddReference(d.CHARPI_USD)
    for prim in Usd.PrimRange(root):  # posed by hand, so turn off all physics
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
    spin_ops = []
    for name in ("prop_front_right_1", "prop_rear_left_1", "prop_front_left_1", "prop_rear_right_1"):
        vis = stage.GetPrimAtPath(f"{PREFIX}/{name}/visuals")
        if vis.IsValid():  # scale to 7 in props; local y is the rotor axis
            UsdGeom.Xformable(vis).AddScaleOp(opSuffix="propsize").Set(Gf.Vec3f(7.0 / 4.94, 1.0, 7.0 / 4.94))
            op = UsdGeom.Xformable(vis).AddRotateYOp(opSuffix="spin")
            op.Set(0.0)
            spin_ops.append((op, PROP_DIR[name]))
    xf = UsdGeom.Xformable(root)
    xf.ClearXformOpOrder()
    t_op, q_op = xf.AddTranslateOp(), xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    t_op.Set(Gf.Vec3d(0, 0, 0))
    q_op.Set(Gf.Quatd(1, 0, 0, 0))
    cache = UsdGeom.XformCache()
    body_local = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(PREFIX + "/body"))).T  # body in root
    w, h = d.BARCODE_RES
    rig = cameras.SensorRig(PREFIX + "/body", node=None, barcode=True,
                            barcode_res=(w // args.barcode_scale, h // args.barcode_scale))
    chase_cam = Camera(prim_path="/World/chase_cam", resolution=tuple(args.chase_res))
    world.reset()
    rig.initialize()
    chase_cam.initialize()
    chase_cam.set_focal_length(2.4)
    chase_cam.set_clipping_range(0.05, 200.0)
    chase = Chase()
    for _ in range(30):  # annotators return empty arrays for the first renders
        world.step(render=True)

    def grab(cam):
        for _ in range(5):
            rgba = cam.get_rgba()
            if rgba is not None and rgba.ndim == 3:
                return cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            world.step(render=True)
        raise RuntimeError("camera gave no image")

    meta = []
    spin = 0.0
    import time
    t_wall = time.time()
    for i, t in enumerate(times):
        p = np.array([np.interp(t, gt[:, 0], gt[:, k]) for k in (1, 2, 3)])
        vx = (np.interp(t + 0.1, gt[:, 0], gt[:, 1]) - np.interp(t - 0.1, gt[:, 0], gt[:, 1])) / 0.2
        body = np.eye(4)
        body[:3, :3], body[:3, 3] = slerp([t]).as_matrix()[0], p
        rootm = body @ np.linalg.inv(body_local)
        qx, qy, qz, qw = R.from_matrix(rootm[:3, :3]).as_quat()
        t_op.Set(Gf.Vec3d(*rootm[:3, 3]))
        q_op.Set(Gf.Quatd(qw, qx, qy, qz))
        # pass 1: onboard cameras, airframe hidden; two steps because frames lag one render
        set_airframe_visible(stage, False)
        world.step(render=True)
        world.step(render=True)
        fish = [grab(c) for c in rig.fisheyes]
        cv2.imwrite(os.path.join(out, "fish", f"{i:05d}.jpg"),
                    np.vstack([np.hstack(fish[:2]), np.hstack(fish[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        for side, key in (("left", "bl"), ("right", "br")):
            img = grab(rig.barcodes[side])
            cv2.imwrite(os.path.join(out, key, f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # pass 2: chase camera, airframe shown, props spinning
        level = np.clip(min(t - (air[0] - 1.5), (air[-1] + 2.0) - t), 0.0, 1.0)
        spin += SPIN_DEG * level
        for op, sgn in spin_ops:
            op.Set(float((sgn * spin) % 360.0))
        set_airframe_visible(stage, True)
        cpos, cq = chase.update(p, vx, args.speed / args.fps)
        chase_cam.set_world_pose(cpos, cq, camera_axes="world")
        world.step(render=True)
        world.step(render=True)
        img = grab(chase_cam)
        cv2.imwrite(os.path.join(out, "chase", f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        meta.append(float(t))
        if i % 50 == 0:
            print(f"[demo] frame {i}/{len(times)} t={t:.1f}s, {(time.time() - t_wall) / (i + 1):.2f} s per frame",
                  flush=True)
    json.dump({"times": meta, "speed": args.speed, "fps": args.fps, "barcode_scale": args.barcode_scale},
              open(os.path.join(out, "frames.json"), "w"))
    print("[demo] done", len(meta), "frames", flush=True)
    app.close()


main()
