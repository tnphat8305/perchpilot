#!/usr/bin/env bash
# Start an ISOLATED PX4 SITL stack for PerchPilot (does not touch other projects):
#   - gz server, partition "perchpilot" (other projects use the default partition)
#   - PX4 instance 7 (others use 0..N small), own working dir under /home/sup/data
#   - Micro XRCE-DDS agent on UDP 8899 (others use 8888), ROS_DOMAIN_ID=77
# PIDs are recorded in $RUN/pids so sitl_down.sh only stops what we started.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
# default: our separate PX4 copy with the perch-mode patch (px4_patches/0001); never the other project's tree
PX4_DIR=${PX4_DIR:-/home/sup/data/perchpilot/PX4-perch}
export PX4_SIM_MODEL=${PX4_SIM_MODEL:-gz_x500_perch}
if [[ ! "$PX4_SIM_MODEL" =~ ^gz_[a-zA-Z0-9_-]+$ ]]; then
  echo "PX4_SIM_MODEL must have the form gz_<model>" >&2
  exit 2
fi
PX4_GZ_MODEL_POSE=${PX4_GZ_MODEL_POSE:-1.55,0,0.25,0,0,0}
RUN=/home/sup/data/perchpilot/sitl
mkdir -p "$RUN/px4_i7"
: > "$RUN/pids"

export GZ_PARTITION=perchpilot
export ROS_DOMAIN_ID=77
export GZ_SIM_RESOURCE_PATH="$HERE/gz/models:$HERE/gz/worlds:$PX4_DIR/Tools/simulation/gz/models:$PX4_DIR/Tools/simulation/gz/worlds${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$HERE/gz/plugins/hold_pad/build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export PX4_GZ_MODELS="$PX4_DIR/Tools/simulation/gz/models"
export PX4_GZ_WORLDS="$HERE/gz/worlds"

# PX4 names the spawned entity <model>_7. Match HoldPad's target to that name.
# HoldPad landing-gear footprint "HALF_X HALF_Y" [m]. C2 v1.1 default = x500 skids "0.125 0.132".
# PERCH_PAD_FOOTPRINT= (set but empty) reproduces the C2 v1 point pad.
fp=${PERCH_PAD_FOOTPRINT-0.125 0.132}
fp_xml=
if [[ -n $fp ]]; then
  read -r fx fy <<< "$fp"
  fp_xml="<footprint_half_x>$fx</footprint_half_x><footprint_half_y>$fy</footprint_half_y>"
fi
# Pad load cell: total (servo + gear/plate collision normal force, default) or servo (pre-2026-11-20 reading)
if [[ ${PERCH_LOAD_CELL:-total} == servo ]]; then fp_xml="$fp_xml<load_cell_mode>servo</load_cell_mode>"; fi
# C5 v2 optical slip sensor on /perch/hold_sensors (2 extra values); PERCH_SLIP_SENSOR=0 keeps the 11-channel C5 v1
if [[ ${PERCH_SLIP_SENSOR:-1} == 0 ]]; then fp_xml="$fp_xml<slip_sensor>0</slip_sensor>"; fi
sed -e "s|<drone_model>x500_7</drone_model>|<drone_model>${PX4_SIM_MODEL#gz_}_7</drone_model>|" \
    -e "s|<pad_offset>0.227</pad_offset>|<pad_offset>0.227</pad_offset>$fp_xml|" \
    "$HERE/gz/worlds/perch_world.sdf" > "$RUN/perch_world.sdf"
gz sim -r -s --headless-rendering "$RUN/perch_world.sdf" > "$RUN/gz.log" 2>&1 &
echo "gz $!" >> "$RUN/pids"
sleep 4

MicroXRCEAgent udp4 -p 8899 > "$RUN/agent.log" 2>&1 &
echo "agent $!" >> "$RUN/pids"

PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 PX4_GZ_WORLD=perch_world PX4_GZ_MODEL_NAME='' \
PX4_GZ_MODEL_POSE="$PX4_GZ_MODEL_POSE" PX4_UXRCE_DDS_PORT=8899 \
  "$PX4_DIR/build/px4_sitl_default/bin/px4" -d -i 7 -w "$RUN/px4_i7" "$PX4_DIR/build/px4_sitl_default/etc" \
  > "$RUN/px4.log" 2>&1 &
echo "px4 $!" >> "$RUN/pids"
echo "started; logs in $RUN"
cat "$RUN/pids"
