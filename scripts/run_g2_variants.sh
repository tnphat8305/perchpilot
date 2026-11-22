#!/usr/bin/env bash
# G2 perch/ride-along/re-arm variants on a fresh SITL each time.
#   V2: forced re-arm sent as internal command (skips preflight checks)
#   V3: V2 + EKF2_GPS_CHECK=21 (drop drift/speed checks that assume a stationary vehicle when at rest)

HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=/home/sup/data/perchpilot/sitl
source /opt/ros/jazzy/setup.bash; source ~/ws_px4/install/setup.bash
export ROS_DOMAIN_ID=77 GZ_PARTITION=perchpilot
for V in "$@"; do
  "$HERE/scripts/sitl_down.sh" >/dev/null; sleep 3
  # g2_probe reads x500_7 truth relative to the original spawn origin.
  PX4_SIM_MODEL=gz_x500 PX4_GZ_MODEL_POSE=0,0,0.25,0,0,0 "$HERE/scripts/sitl_up.sh" >/dev/null; sleep 18
  if [ "$V" = V3 ]; then
    /home/sup/PX4-Autopilot/build/px4_sitl_default/bin/px4-param --instance 7 set EKF2_GPS_CHECK 21
  fi
  FORCE_INTERNAL=1 timeout 160 python3 "$HERE/ros/g2_probe.py" "$OUT/g2_probe_$V.json" 2>&1 \
    | grep -o '{"t".*' > "$OUT/g2_events_$V.jsonl"
  cp "$OUT/px4.log" "$OUT/px4_$V.log"
done
"$HERE/scripts/sitl_down.sh"
