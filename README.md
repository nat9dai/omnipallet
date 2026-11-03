# OmniPallet

A quadrotor that counts pallets in a GPS-denied warehouse aisle, simulated in Isaac Sim.
Four fisheye cameras feed D2VINS for navigation, a barcode camera reads the labels, and PX4 flies over ROS 2.

Needs Ubuntu 22.04, Isaac Sim 5.1 (`~/isaacsim`), Pegasus Simulator, PX4 v1.16 (`~/PX4-Autopilot`, `make px4_sitl`),
ROS 2 Humble, Micro XRCE-DDS Agent and Docker.

## Setup

```bash
git clone https://github.com/HKUST-Aerial-Robotics/D2SLAM third_party/D2SLAM
git -C third_party/D2SLAM checkout 4c33871
git -C third_party/D2SLAM am "$PWD"/patches/d2slam/*.patch
git clone https://github.com/HKUST-Swarm/swarm_msgs third_party/swarm_msgs
git clone https://github.com/PX4/px4_msgs ros2_ws/src/px4_msgs
git -C ros2_ws/src/px4_msgs checkout 57e96ed
```

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
source /opt/ros/humble/setup.bash
(cd ros2_ws && colcon build --packages-select px4_msgs)
```

```bash
docker build -t omnipallet/d2slam:cpu d2slam/docker
d2slam/run.sh build
```

```bash
.venv/bin/python sim/warehouse/make_textures.py
.venv/bin/python sim/warehouse/make_labels.py
python3 sim/px4/make_overlay.py
export CHARPI_USD=/path/to/charpi.usd
```

## Run

```bash
sim/run_isaac.sh sim/apps/world_preview.py --x-range 0 15
scripts/run_mission.sh oracle_01 oracle
MAP=1 scripts/run_mission.sh vio_01 vio
```

## Evaluate

```bash
.venv/bin/python scripts/eval_inventory.py results/runs/vio_01
.venv/bin/python scripts/eval_vio.py results/runs/vio_01
.venv/bin/python scripts/plot_run.py results/runs/vio_01
```

## Demo video

```bash
.venv/bin/python scripts/demo_data.py results/runs/vio_01
.venv/bin/python scripts/map_data.py results/runs/vio_01
sim/run_isaac.sh sim/apps/demo_replay.py --run results/runs/vio_01
.venv/bin/python scripts/make_demo_video.py results/runs/vio_01
```
