#!/usr/bin/env bash
# Start or stop the sim stack; each process logs to <run_dir>/<name>.log, and `stop` kills only what this started.
#   scripts/stack.sh start <run_dir> <isaac_app.py> [app args...]   # agent + Isaac/PX4
#   scripts/stack.sh node  <run_dir> <name> <python module> [ros args...]   # a host ROS 2 node
#   scripts/stack.sh stop  <run_dir>
#   scripts/stack.sh bag   <run_dir> <topics...>
#   scripts/stack.sh wait_px4 <run_dir> [timeout_s]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cmd="$1"; run="$2"; shift 2
mkdir -p "$run"
pids="$run/pids"

launch() {
  local name="$1"; shift
  setsid "$@" > "$run/$name.log" 2>&1 < /dev/null &
  echo "$! $name" >> "$pids"
}

case "$cmd" in
  start)
    app="$1"; shift
    if ! pgrep -x MicroXRCEAgent > /dev/null; then launch xrce_agent MicroXRCEAgent udp4 -p 8888; fi
    launch isaac "$ROOT/sim/run_isaac.sh" "$app" "$@"
    ;;
  node)
    name="$1"; module="$2"; shift 2
    launch "$name" bash -c "source '$ROOT/scripts/ros2_env.sh' && exec python -m $module --ros-args $*"
    ;;
  bag)
    launch bag bash -c "source '$ROOT/scripts/ros2_env.sh' && exec ros2 bag record -s mcap -o '$run/bag' $*"
    ;;
  wait_px4)
    t="${1:-240}"
    timeout "$t" bash -c "until grep -q 'Startup script returned' '$run/isaac.log'; do sleep 2; done" \
      && echo "px4 up" || { echo "px4 not up after ${t}s"; exit 1; }
    ;;
  stop)
    [ -f "$pids" ] || exit 0
    while read -r pid name; do
      kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null
    done < "$pids"
    sleep 5
    while read -r pid name; do
      kill -KILL -- "-$pid" 2>/dev/null || true
    done < "$pids"
    pkill -x px4 2>/dev/null || true
    rm -f "$pids"
    ;;
esac
