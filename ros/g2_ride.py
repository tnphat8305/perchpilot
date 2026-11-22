"""G2-ride: clean ride-along test without DetachableJoint artifacts.

Drone lands on the cart roof (real contact), disarms normally, the cart ramps up
to 1.5 m/s and drives 10 s, stops; then the drone re-arms and takes off from
the (now displaced) cart roof. Compares stock PX4 vs patched PX4 with perch
mode on (env PERCH_MODE=1). Ground truth from gz for EKF error.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from g2_probe import Probe, gz_pub, gz_truth, VEHICLE_CMD_USER_1, PERCH_MODE  # noqa: E402
import rclpy  # noqa: E402
from px4_msgs.msg import VehicleCommand, VehicleStatus  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/sup/data/perchpilot/sitl/g2_ride.json"


def land_flags():
    """Sample vehicle_land_detected via the PX4 shell client of instance 7."""
    import subprocess
    px4 = os.environ.get("PX4_DIR", "/home/sup/PX4-Autopilot") + "/build/px4_sitl_default/bin/px4-listener"
    out = subprocess.run([px4, "--instance", "7", "vehicle_land_detected", "-n", "1"], capture_output=True, text=True, timeout=10).stdout
    flags = {}
    for line in out.splitlines():
        line = line.strip()
        for key in ("landed", "at_rest", "maybe_landed"):
            if line.startswith(key + ":"):
                flags[key] = line.split(":")[1].strip()
    return flags


def err(n, off):
    tr = gz_truth()
    e = (n.lpos.x - off[0], n.lpos.y - off[1], n.lpos.z - off[2])
    return tr, round(math.dist(tr, e), 2), [round(n.lpos.vx, 2), round(n.lpos.vy, 2), round(n.lpos.vz, 2)]


def main():
    rclpy.init()
    n = Probe()
    for _ in range(30):
        n.spin_for(1, lambda: None)
        if n.lpos is not None and n.status is not None and n.att is not None:
            break
    gz_pub("/perch/detach", "gz.msgs.Empty", "")
    tr0 = gz_truth()
    off = (n.lpos.x - tr0[0], n.lpos.y - tr0[1], n.lpos.z - tr0[2])   # EKF-origin offset vs truth at start
    res = {"perch_mode": PERCH_MODE, "events": []}

    def ev(name, **kw):
        tr, e, v = err(n, off)
        rec = {"t": round(time.time() - n.t0, 2), "event": name, "arming_state": int(n.status.arming_state),
               "landed_truth_z_up": round(-tr[2], 2), "truth_ned": [round(x, 2) for x in tr], "ekf_err_m": e, "ekf_vel": v, **kw}
        res["events"].append(rec)
        print(json.dumps(rec), flush=True)

    ev("start")
    # take off, fly above cart roof centre (ENU 8,0 -> NED 0,8), land on roof (0.85 m)
    x0, y0 = n.lpos.x, n.lpos.y
    n.hold_pos(x0, y0, off[2] - 3.0, 1.0)
    for _ in range(8):                       # preflight checks may need a few seconds after boot
        n.cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)
        n.cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1)
        n.hold_pos(x0, y0, off[2] - 3.0, 2.0)
        if n.status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
            break
    n.hold_pos(x0, y0, off[2] - 3.0, 8.0)
    n.hold_pos(off[0] + 0.0, off[1] + 8.0, off[2] - 3.0, 8.0)
    n.cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
    n.spin_for(15, lambda: None)
    ev("landed_on_roof")
    if n.status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
        n.cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0)
        n.spin_for(2, lambda: None)
    if PERCH_MODE:
        n.cmd(VEHICLE_CMD_USER_1, 1)
    n.spin_for(3, lambda: None)
    ev("disarmed_on_roof", land_detected=land_flags())
    # ride: ramp 0 -> 1.5 m/s over 3 s, cruise 10 s, ramp down 3 s
    for k in range(1, 16):
        gz_pub("/model/perch_cart/cmd_vel", "gz.msgs.Twist", f"linear: {{x: {0.1 * k:.2f}}}")
        n.spin_for(0.2, lambda: None)
    samples = []
    for _ in range(5):
        n.spin_for(2, lambda: None)
        samples.append(land_flags())
    ev("riding_1.5ms", land_detected_samples=samples)
    DEPART_MOVING = os.environ.get("DEPART_MOVING", "0") == "1"
    if not DEPART_MOVING:
        for k in range(14, -1, -1):
            gz_pub("/model/perch_cart/cmd_vel", "gz.msgs.Twist", f"linear: {{x: {0.1 * k:.2f}}}")
            n.spin_for(0.2, lambda: None)
        n.spin_for(3, lambda: None)
        ev("ride_stopped")
    # re-arm + take off 2 m above current EKF position
    hx, hy, hz = n.lpos.x, n.lpos.y, n.lpos.z
    n.hold_pos(hx, hy, hz - 2.0, 1.0)
    n.cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)
    n.cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1)
    n.hold_pos(hx, hy, hz - 2.0, 2.0)
    plain_ok = n.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
    if not plain_ok:
        n.cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1, 21196)
        n.hold_pos(hx, hy, hz - 2.0, 1.0)
    if PERCH_MODE:
        n.cmd(VEHICLE_CMD_USER_1, 0)
    ev("rearm", plain_arm_ok=plain_ok)
    n.hold_pos(hx, hy, hz - 2.0, 8.0)
    ev("after_takeoff_8s", target_up_m=2.0, depart_moving=DEPART_MOVING)
    gz_pub("/model/perch_cart/cmd_vel", "gz.msgs.Twist", "linear: {x: 0.0}")
    n.cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
    n.spin_for(1, lambda: None)
    json.dump(res, open(OUT, "w"))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
