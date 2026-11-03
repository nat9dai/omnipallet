#!/usr/bin/env bash
# Run a script with Isaac Sim 5.1's Python and its internal ROS 2 Humble libraries.
#   sim/run_isaac.sh sim/apps/<app>.py [args]
set -e
ISAACSIM_PATH="${ISAACSIM_PATH:-$HOME/isaacsim}"
unset ROS_VERSION ROS_PYTHON_VERSION ROS_DISTRO AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH CMAKE_PREFIX_PATH
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v "^/opt/ros/" | paste -sd':' -)
export ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${ISAACSIM_PATH}/exts/isaacsim.ros2.bridge/humble/lib"
exec "${ISAACSIM_PATH}/python.sh" "$@"
