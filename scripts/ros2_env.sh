#!/usr/bin/env bash
# Source this for host ROS 2 nodes: Humble, px4_msgs (v1.16 wire format), our package and the venv (~/.local ignored).
#   source scripts/ros2_env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "$ROOT/ros2_ws/install/local_setup.bash"
source "$ROOT/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/ros2_ws/src/stock_drone:$PYTHONPATH"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
