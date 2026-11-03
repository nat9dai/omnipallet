#!/usr/bin/env bash
# Fly one mission in the warehouse sim and record it under results/runs/<name>.
#   [SIM_ARGS="--x-range 0 40.5"] [DUR=1500] [MAP=1] scripts/run_mission.sh <name> <oracle|vio> [mission -p args...]
# oracle: PX4 and the scanner get the true pose; vio: they get live D2VINS. MAP=1 also records the dense map.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
NAME="$1"; NAV="$2"; shift 2
RUN="results/runs/$NAME"; rm -rf "$RUN"; mkdir -p "$RUN"
echo "$0 $NAME $NAV $* SIM_ARGS=${SIM_ARGS:-}" > "$RUN/command.txt"

scripts/stack.sh start "$RUN" sim/apps/mission_sim.py --duration "${DUR:-1500}" ${SIM_ARGS:-}
scripts/stack.sh wait_px4 "$RUN" 300
if [ "$NAV" = vio ]; then
  setsid d2slam/live.sh "$RUN" > "$RUN/live.log" 2>&1 < /dev/null &
  POSE=/d2vins/imu_propagation; SCAN_POSE=/nav/odom; HOME_XYZ="[-3.0,0.0,0.0]"  # VIO origin = take-off pad
else
  POSE=/gt/odom; SCAN_POSE=/gt/odom; HOME_XYZ="[0.0,0.0,0.0]"  # simulator world = aisle frame
fi
if [ "$NAV" = vio ]; then
  BRIDGE_ARGS="${BRIDGE_EXTRA:-}"  # e.g. BRIDGE_EXTRA="-p pose_frame:=ned" for a GCS map
else
  BRIDGE_ARGS="-p aisle_lock:=false -p rack_lock:=false"
fi
scripts/stack.sh node "$RUN" vio_bridge stock_drone.vio_bridge_node -p odom_topic:=$POSE $BRIDGE_ARGS
scripts/stack.sh bag "$RUN" /gt/odom /imu /range /range_left /range_right /nav/odom /nav/aisle_offset /nav/rack_offset \
  /mission/phase /mission/reference /d2vins/imu_propagation \
  /fmu/out/vehicle_local_position /fmu/out/estimator_status_flags /fmu/out/vehicle_status_v1 \
  /map/points_delta /d2vins/margined_cloud
scripts/stack.sh node "$RUN" scanner stock_drone.scanner_node -p use_sim_time:=true -p pose_topic:=$SCAN_POSE \
  -p "home:=$HOME_XYZ" -p out_dir:="$ROOT/$RUN"
if [ "${MAP:-0}" = 1 ]; then
  scripts/stack.sh node "$RUN" stereo_map stock_drone.stereo_map_node -p use_sim_time:=true -p out_dir:="$ROOT/$RUN"
fi
sleep 5
scripts/stack.sh node "$RUN" mission stock_drone.mission_node -p use_sim_time:=true "$@"

# a stalled sim never reaches DONE; give up after 3 h of wall time
timeout 10800 bash -c "until grep -q 'phase -> DONE' '$RUN/mission.log'; do sleep 10; done" || echo "mission did not finish"
sleep 15
scripts/stack.sh stop "$RUN" > /dev/null 2>&1 || true
if [ "$NAV" = vio ]; then
  docker stop "d2slam_live_$NAME" > /dev/null 2>&1 || true  # only this run's VIO, not offline replays
fi
echo "done: $RUN"
