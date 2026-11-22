#!/usr/bin/env bash
# Run once per batch, by the SITL operator under C6.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
python=${PERCH_PYTHON:-/home/sup/data/venvs/occwm/bin/python}
px4_ros_setup=${PX4_ROS_SETUP:-/home/sup/ws_px4/install/setup.bash}
state=/home/sup/data/perchpilot/sitl

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  exec "$python" "$here/ros/eval_node.py" --help
fi
cfg=$("$python" "$here/ros/eval_node.py" "$@" --check-config)
policy=${cfg%%$'\n'*}
truth_dir=${cfg#*$'\n'}
export PERCH_NODE_IMPL=${PERCH_NODE_IMPL:-python}
case "$PERCH_NODE_IMPL" in
  python) perch_node=("$python" "$here/ros/perch_node.py") ;;
  rust)
    perch_node=("$here/rs/target/release/perch_node")
    if [[ ! -x ${perch_node[0]} || ! -f $policy/policy.rust.json ]]; then
      echo 'Build the Rust node and export the C1 policy weights first (ros/README_perch_node.md).' >&2
      exit 1
    fi
    ;;
  *) echo 'PERCH_NODE_IMPL must be rust or python.' >&2; exit 1 ;;
esac
# PX4 roll/pitch airmode: keeps rate tracking at low collective thrust, as the GPU sim assumes
# (MC_AIRMODE=0 gives no torque at thrust 0 and the policy's thrust-cut + pitch command does nothing).
airmode=${PERCH_AIRMODE:-1}
export PERCH_AIRMODE=$airmode PERCH_YAW_SCALE=${PERCH_YAW_SCALE:-0} PERCH_FREEZE_PLATE_V=${PERCH_FREEZE_PLATE_V:-1}
export PERCH_DEPART_PUSHOFF=${PERCH_DEPART_PUSHOFF:-1} PERCH_NODE_ARGS=${PERCH_NODE_ARGS:-}
export PERCH_LOAD_CELL=${PERCH_LOAD_CELL:-total} PERCH_SLIP_SENSOR=${PERCH_SLIP_SENSOR:-1}
# Main-evaluation defaults (P5, 2026-11-20): no yaw-rate command to PX4 (PX4 lowers the collective for yaw) and
# plate velocity held during PERCH (EKF drift otherwise shows as plate motion). PERCH_YAW_SCALE=1 and
# PERCH_FREEZE_PLATE_V=0 reproduce the earlier runs. HoldPad footprint: see sitl_up.sh (PERCH_PAD_FOOTPRINT).
node_main=(--yaw-scale "${PERCH_YAW_SCALE:-0}")
# DEPART by push-off in body-rate mode (PERCH_DEPART_PUSHOFF=0: old offboard-position depart).
if [[ ${PERCH_DEPART_PUSHOFF:-1} == 1 ]]; then node_main+=(--depart-pushoff); fi
if [[ ${PERCH_FREEZE_PLATE_V:-1} == 1 ]]; then node_main+=(--freeze-plate-v); fi
# Extra perch_node options, e.g. PERCH_NODE_ARGS='--approach-back 3.5'.
read -r -a node_extra <<< "${PERCH_NODE_ARGS:-}"
so="$here/gz/plugins/hold_pad/build/libHoldPad.so"
if [[ ! -f $so ]]; then
  echo 'Build HoldPad first (gz/plugins/hold_pad/README.md).' >&2
  exit 1
fi
# review D08: refuse a stale plugin build, and one without footprint support when the footprint is requested
for src in "$here"/gz/plugins/hold_pad/*.cc "$here"/gz/plugins/hold_pad/*.hh; do
  if [[ $src -nt $so ]]; then
    echo "libHoldPad.so is older than $src: rebuild HoldPad." >&2
    exit 1
  fi
done
export PERCH_PAD_FOOTPRINT=${PERCH_PAD_FOOTPRINT-0.125 0.132}
if [[ -n $PERCH_PAD_FOOTPRINT ]] && ! grep -qa footprint_half_x "$so"; then
  echo 'libHoldPad.so has no footprint support but PERCH_PAD_FOOTPRINT is set.' >&2
  exit 1
fi

set +u
source /opt/ros/jazzy/setup.bash
# shellcheck source=/dev/null
source "$px4_ros_setup"
set -u
export GZ_PARTITION=perchpilot ROS_DOMAIN_ID=77 PX4_SIM_MODEL=gz_x500_perch
"$python" -c 'import rclpy, px4_msgs.msg, gz.transport13, gz.msgs10.contacts_pb2'
bridge_dir=$(ros2 pkg prefix ros_gz_bridge)/lib/ros_gz_bridge

mkdir -p "$state"
exec 9>"$state/eval.lock"
flock -n 9 || { echo 'An evaluation already owns the SITL slot.' >&2; exit 1; }
if [[ -s "$state/pids" ]]; then
  while read -r label pid; do
    if [[ $pid =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "SITL pid file contains a live $label process ($pid)." >&2
      exit 1
    fi
  done < "$state/pids"
fi
if pgrep -f "^gz sim -r -s --headless-rendering $state/perch_world\.sdf( |$)" >/dev/null; then
  echo 'A Gazebo server already uses the PerchPilot SITL world.' >&2
  exit 1
fi

# shellcheck disable=SC2317
cleanup() {
  local running
  running=$(jobs -pr)
  if [[ -n $running ]]; then
    while read -r pid; do kill "$pid" 2>/dev/null || true; done <<< "$running"
  fi
  "$here/scripts/sitl_down.sh" 9>&-
  wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# One SITL session: PX4 + Gazebo + bridge + nodes until the evaluator exits. Exit code 3 from the evaluator means
# "a trial could not return to standby": the batch restarts PX4 + Gazebo and continues with --continue (that trial
# stays recovered=false / rearm_ok as measured; restarts are counted in the result as sitl_restarts).
session() {
  # review D14: errexit does not apply inside a function called left of ||, so check explicitly
  if ! "$here/scripts/sitl_up.sh" 9>&-; then
    echo 'sitl_up.sh failed' >&2
    "$here/scripts/sitl_down.sh" 9>&-
    return 1
  fi
  export PERCH_EFFECTIVE_NODE_ARGS="--thrust-model x500_perch_gz --pad-offset 0.227 --px4-param MC_AIRMODE=$airmode ${node_main[*]} ${node_extra[*]}"
  "$bridge_dir/parameter_bridge" '/perch/hold_cmd@std_msgs/msg/Float64]gz.msgs.Double' 9>&- &
  local bridge_pid=$!
  "${perch_node[@]}" --run "$policy" --eval-mode \
    --thrust-model x500_perch_gz --pad-offset 0.227 --px4-param "MC_AIRMODE=$airmode" \
    --tick-log "$truth_dir/node_ticks.jsonl" --q2-log "$truth_dir/q2_samples.jsonl" \
    "${node_main[@]}" "${node_extra[@]}" 9>&- &
  local perch_pid=$!
  # eval_node also starts the tag_camera_sim ROS node in its executor, using its single Gazebo pose stream.
  "$python" "$here/ros/eval_node.py" "$@" 9>&- &
  local eval_pid=$!
  local st=0 finished=
  wait -n -p finished "$bridge_pid" "$perch_pid" "$eval_pid" || st=$?
  if [[ ${finished:-} != "$eval_pid" ]]; then
    echo 'Bridge or perch node exited before the evaluator.' >&2
    st=1
  fi
  kill "$bridge_pid" "$perch_pid" "$eval_pid" 2>/dev/null || true
  wait "$bridge_pid" "$perch_pid" "$eval_pid" 2>/dev/null || true
  "$here/scripts/sitl_down.sh" 9>&-
  return "$st"
}

max_restarts=${PERCH_MAX_RESTARTS:-10}
restarts=0
status=0
session "$@" || status=$?
while [[ $status == 3 && $restarts -lt $max_restarts ]]; do
  restarts=$((restarts + 1))
  echo "SITL restart $restarts/$max_restarts after a failed recovery" >&2
  status=0
  session "$@" --continue || status=$?
done
exit "$status"
