#!/usr/bin/env bash
# Live D2VINS for closed-loop flight: roscore, D2VINS and a ros1_bridge (Foxy) to the host's ROS 2 nodes.
#   d2slam/live.sh <run_dir> [config]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$(cd "$1" && pwd)"; REL="${RUN#$ROOT/results/}"; CFG_SRC="${2:-$ROOT/d2slam/config/sim/quadcam_sim.yaml}"
OUT="$RUN/d2vins"; mkdir -p "$OUT"
sed -e "s|^output_path:.*|output_path: \"/root/results/$REL/d2vins/\"|" "$CFG_SRC" > "$OUT/config.yaml"
cp "$(dirname "$CFG_SRC")/fisheye_cams.yaml" "$OUT/"
export D2SLAM_NAME="d2slam_live_$(basename "$RUN")"
exec "$ROOT/d2slam/run.sh" "roscore > /dev/null 2>&1 & sleep 3; \
  rosparam load /root/config/bridge/topics.yaml; \
  roslaunch /root/config/launch/sim_vio.launch config:=/root/results/$REL/d2vins/config.yaml \
    output_path:=/root/results/$REL/d2vins/ > /root/results/$REL/d2vins/d2vins.log 2>&1 & \
  sleep 4; source /opt/ros/foxy/setup.bash; export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-0}; \
  export FASTRTPS_DEFAULT_PROFILES_FILE=/root/config/dds/udp_only.xml; \
  ros2 run ros1_bridge parameter_bridge"
