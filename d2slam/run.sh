#!/usr/bin/env bash
# Run a command in the D2SLAM container with source, config and results mounted; D2SLAM_NAME names the container.
#   d2slam/run.sh build                 # catkin build d2vins (CPU)
#   d2slam/run.sh <cmd ...>             # any command, ROS 1 Noetic + workspace sourced
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMG=omnipallet/d2slam:cpu
VOLS=(-v "$ROOT/third_party/D2SLAM:/root/swarm_ws/src/D2SLAM"
      -v "$ROOT/third_party/swarm_msgs:/root/swarm_ws/src/swarm_msgs"
      -v "$ROOT/d2slam/ws/build:/root/swarm_ws/build" -v "$ROOT/d2slam/ws/devel:/root/swarm_ws/devel"
      -v "$ROOT/d2slam/ws/logs:/root/swarm_ws/logs"
      -v "$ROOT/d2slam/config:/root/config" -v "$ROOT/d2slam/output:/root/output"
      -v "$ROOT/results:/root/results")
mkdir -p "$ROOT/d2slam/ws/build" "$ROOT/d2slam/ws/devel" "$ROOT/d2slam/ws/logs" "$ROOT/d2slam/output" "$ROOT/results"
TTY=(); [ -t 0 ] && TTY=(-it)
if [ "$1" = "build" ]; then
  CMD='source /opt/ros/noetic/setup.bash && catkin config --extend /opt/ros/noetic -DCMAKE_BUILD_TYPE=Release \
       --cmake-args -DUSE_CUDA=OFF -DTorch_DIR=/opt/libtorch/share/cmake/Torch \
       -DONNXRUNTIME_LIB_DIR=/usr/local/lib -DONNXRUNTIME_INC_DIR=/usr/local/include && catkin build d2vins -j16'
else
  # ';' not '&&': a command starting with 'x &' would otherwise background the sourcing too
  CMD="source /opt/ros/noetic/setup.bash && source /root/swarm_ws/devel/setup.bash; $*"
fi
exec docker run --rm "${TTY[@]}" --net=host --ipc=host --name "${D2SLAM_NAME:-d2slam_$$}" -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  "${VOLS[@]}" "$IMG" bash -c "$CMD"
