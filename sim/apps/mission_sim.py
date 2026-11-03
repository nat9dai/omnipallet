"""Full mission sim: warehouse aisle, stock drone, PX4 SITL and cameras, publishing ROS 2 in sim time.

    sim/run_isaac.sh sim/apps/mission_sim.py [--gui] [--x-range 0 20] [--no-barcode] [--duration 1200]

PX4 also needs MicroXRCEAgent on UDP 8888; scripts/stack.sh starts both.
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--gui", action="store_true")
ap.add_argument("--x-range", type=float, nargs=2, default=None, help="only pallets in this span")
ap.add_argument("--no-barcode", action="store_true")
ap.add_argument("--barcode-every", type=int, default=3)
ap.add_argument("--barcode-res", type=int, nargs=2, default=None)
ap.add_argument("--duration", type=float, default=1500.0, help="simulated seconds")
ap.add_argument("--home", type=float, nargs=2, default=[-3.0, 0.0])
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not args.gui, "anti_aliasing": 0})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
app.update()

import rclpy  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from omni.isaac.core.world import World  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for sub in ("sim/apps", "sim/warehouse", "sim/vehicle", "ros2_ws/src/stock_drone/stock_drone"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import build_world  # noqa: E402
import cameras  # noqa: E402
import simsettings  # noqa: E402
import stockdrone  # noqa: E402

simsettings.make_rendering_synchronous()
PHYSICS_DT, RENDER_DT = 1.0 / 250.0, 1.0 / 25.0

rclpy.init()
node = rclpy.create_node("isaac_sim")
pg = PegasusInterface()
pg._world_settings = {"physics_dt": PHYSICS_DT, "stage_units_in_meters": 1.0, "rendering_dt": RENDER_DT}
pg._world = World(**pg._world_settings)
world = pg.world
stage = omni.usd.get_context().get_stage()
pallets = build_world.build(stage, get_assets_root_path(), x_range=tuple(args.x_range) if args.x_range else None)
print(f"[mission_sim] warehouse with {len(pallets)} pallets", flush=True)

prefix = "/World/stockdrone"
drone = stockdrone.spawn(stage, node, position=(args.home[0], args.home[1], 0.02), prefix=prefix, pallets=pallets)
# keep the airframe out of the onboard views, as OmniNxt's vignetting mask does
print(f"[mission_sim] hid {stockdrone.hide_airframe(stage, prefix)} airframe parts", flush=True)
rig = cameras.SensorRig(prefix + "/body", node=node, barcode=not args.no_barcode, barcode_res=args.barcode_res)
world.reset()
rig.initialize()

timeline = omni.timeline.get_timeline_interface()
timeline.play()
step = 0
tim = {"step": 0.0, "quad": 0.0, "barcode": 0.0}  # wall seconds per part of the loop
import time  # noqa: E402

while app.is_running() and world.current_time < args.duration:
    # all cameras render every step (toggling render products froze them); barcodes are read every Nth
    bar = bool(rig.barcodes) and step % args.barcode_every == 0
    t0 = time.perf_counter()
    world.step(render=True)
    t1 = time.perf_counter()
    t = world.current_time
    if step > 5:  # the first frames are still black
        rig.publish_quad(t - RENDER_DT)  # frames lag one render step (simsettings)
        t2 = time.perf_counter()
        if bar:
            rig.publish_barcodes(t - RENDER_DT)
        tim["quad"] += t2 - t1
        tim["barcode"] += time.perf_counter() - t2
    tim["step"] += t1 - t0
    step += 1
    if step % 250 == 0:
        per = {k: round(1000 * v / 250, 1) for k, v in tim.items()}
        print(f"[mission_sim] sim time {t:.1f} s, ms per render step {per}", flush=True)
        tim = dict.fromkeys(tim, 0.0)
print(f"[mission_sim] done at sim time {world.current_time:.1f} s", flush=True)
timeline.stop()
node.destroy_node()
rclpy.try_shutdown()
app.close()
