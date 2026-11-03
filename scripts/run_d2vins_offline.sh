#!/usr/bin/env bash
# Replay a run's ROS 1 bag through D2VINS (CPU) in Docker; writes <run>/d2vins/d2vins_1.csv (t x y z qw qx qy qz).
#   [DEBUG=1] [DURATION=s] scripts/run_d2vins_offline.sh results/runs/<run> [rate] [config]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$(cd "$1" && pwd)"; RATE="${2:-1.0}"; CFG_SRC="${3:-$ROOT/d2slam/config/sim/quadcam_sim.yaml}"
REL="${RUN#$ROOT/results/}"
OUT="$RUN/d2vins"
[ -f "$RUN/ros1.bag" ] || "$ROOT/.venv/bin/python" "$ROOT/scripts/bag_to_ros1.py" "$RUN"
rm -rf "$OUT"; mkdir -p "$OUT"
# per-run config with output in the run folder, plus the camera calibration file it points to
sed -e "s|^output_path:.*|output_path: \"/root/results/$REL/d2vins/\"|" \
    "$CFG_SRC" > "$OUT/config.yaml"
cp "$(dirname "$CFG_SRC")/$(sed -n 's/^calib_file_path: *"\(.*\)"/\1/p' "$CFG_SRC")" "$OUT/"
SHOW=false
if [ "${DEBUG:-0}" = "1" ]; then
  SHOW=true
  sed -i -e "s|^show_track_id:.*|show_track_id: 1|" "$OUT/config.yaml"
  echo "write_tracking_image_to_file: 1" >> "$OUT/config.yaml"
fi
DUR=""; [ -n "${DURATION:-}" ] && DUR="-u $DURATION"
# a private ROS master port so a live session on 11311 is not disturbed
"$ROOT/d2slam/run.sh" "export ROS_MASTER_URI=http://localhost:11411; roscore -p 11411 > /dev/null 2>&1 & sleep 3; \
  roslaunch /root/config/launch/sim_vio.launch config:=/root/results/$REL/d2vins/config.yaml output_path:=/root/results/$REL/d2vins/ show:=$SHOW \
    > /root/results/$REL/d2vins/d2vins.log 2>&1 & \
  sleep 6; rosbag play -q -r $RATE $DUR /root/results/$REL/ros1.bag; sleep 4; rosnode kill -a > /dev/null 2>&1; sleep 2; kill %1"
ls "$OUT" | head
