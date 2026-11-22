"""G2 probe: PX4 SITL (instance 7) offboard interface checks for PerchPilot.

Phases
  A. takeoff + hover in offboard position mode
  B. offboard BODY-RATE mode (/fmu/in/vehicle_rates_setpoint): own attitude-hold
     + altitude loop, then rate doublets on roll / pitch / yaw; log gyro to
     identify the PX4 rate-loop lag used by the GPU simulator
  C. fly next to the perch plate, attach (gz DetachableJoint - interface test only),
     force-disarm while attached (motors off), let the cart drive (ride-along),
     re-arm, detach, depart.
All samples go to a JSON log for offline analysis.
"""
import json
import math
import os
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleRatesSetpoint, VehicleCommand,
                          VehicleStatus, VehicleLocalPosition, VehicleAttitude, SensorCombined, VehicleLandDetected)

NS = "/px4_7"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/sup/data/perchpilot/sitl/g2_probe.json"


def gz_pub(topic, msgtype, payload):
    subprocess.run(["gz", "topic", "-t", topic, "-m", msgtype, "-p", payload], check=False,
                   env={**__import__("os").environ, "GZ_PARTITION": "perchpilot"}, timeout=10)


def gz_truth():
    """Ground-truth x500_7 pose from gz, converted ENU->NED (local origin = spawn ENU 0,0)."""
    import os, re
    out = subprocess.run(["gz", "model", "-m", "x500_7", "-p"], capture_output=True, text=True, timeout=10,
                         env={**os.environ, "GZ_PARTITION": "perchpilot"}).stdout
    nums = re.findall(r"\[([-0-9.e ]+)\]", out)
    trip = [m.split() for m in nums if len(m.split()) == 3]
    if not trip:
        return None
    x, y, z = [float(v) for v in trip[0]]
    return (y, x, -z)


FORCE_INTERNAL = os.environ.get("FORCE_INTERNAL", "0") == "1"
PERCH_MODE = os.environ.get("PERCH_MODE", "0") == "1"   # needs patched PX4 (px4_patches/0001)
VEHICLE_CMD_USER_1 = 31010


def quat_to_rpy(q):  # PX4 q = [w,x,y,z], body FRD -> NED
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class Probe(Node):
    def __init__(self):
        super().__init__("perch_g2_probe")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.p_mode = self.create_publisher(OffboardControlMode, NS + "/fmu/in/offboard_control_mode", qos)
        self.p_traj = self.create_publisher(TrajectorySetpoint, NS + "/fmu/in/trajectory_setpoint", qos)
        self.p_rate = self.create_publisher(VehicleRatesSetpoint, NS + "/fmu/in/vehicle_rates_setpoint", qos)
        self.p_cmd = self.create_publisher(VehicleCommand, NS + "/fmu/in/vehicle_command", qos)
        self.status = self.lpos = self.att = None
        self.gyro = (0.0, 0.0, 0.0)
        self.landed = None
        self.create_subscription(VehicleStatus, NS + "/fmu/out/vehicle_status", lambda m: setattr(self, "status", m), qos)
        self.create_subscription(VehicleLocalPosition, NS + "/fmu/out/vehicle_local_position", lambda m: setattr(self, "lpos", m), qos)
        self.create_subscription(VehicleAttitude, NS + "/fmu/out/vehicle_attitude", lambda m: setattr(self, "att", m), qos)
        self.create_subscription(SensorCombined, NS + "/fmu/out/sensor_combined", self._gyro, qos)
        self.log = {"rate": [], "events": [], "track": []}
        self.t0 = time.time()

    def _gyro(self, m):
        self.gyro = tuple(float(v) for v in m.gyro_rad)

    def now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def ev(self, name, **kw):
        st = self.status
        tr = gz_truth() if name in ("start", "attached", "ride_along_done", "after_plain_arm", "after_forced_arm", "detached", "departed") else None
        ekf = (self.lpos.x, self.lpos.y, self.lpos.z) if self.lpos else None
        if tr and ekf:
            kw["truth_ned"] = [round(v, 2) for v in tr]
            kw["ekf_err_m"] = round(math.dist(tr, ekf), 2)
            kw["ekf_vel"] = [round(self.lpos.vx, 2), round(self.lpos.vy, 2), round(self.lpos.vz, 2)]
        rec = {"t": round(time.time() - self.t0, 3), "event": name,
               "arming_state": int(st.arming_state) if st else None, "nav_state": int(st.nav_state) if st else None,
               "z": round(self.lpos.z, 3) if self.lpos else None, **kw}
        self.log["events"].append(rec)
        self.get_logger().info(json.dumps(rec))

    def cmd(self, command, p1=0.0, p2=0.0, p3=0.0):
        m = VehicleCommand()
        m.timestamp = self.now_us()
        m.command, m.param1, m.param2, m.param3 = command, float(p1), float(p2), float(p3)
        m.target_system, m.target_component, m.source_system, m.source_component = 8, 1, 1, 1
        m.from_external = not (FORCE_INTERNAL and command == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM and p2 == 21196)
        self.p_cmd.publish(m)

    def mode(self, body_rate=False):
        m = OffboardControlMode()
        m.timestamp = self.now_us()
        m.position = not body_rate
        m.body_rate = body_rate
        self.p_mode.publish(m)

    def pos_sp(self, x, y, z, yaw=0.0):
        m = TrajectorySetpoint()
        m.timestamp = self.now_us()
        m.position = [float(x), float(y), float(z)]
        m.velocity = [math.nan] * 3
        m.acceleration = [math.nan] * 3
        m.yaw = float(yaw)
        self.p_traj.publish(m)

    def rate_sp(self, r, p, y, thr):
        m = VehicleRatesSetpoint()
        m.timestamp = self.now_us()
        m.roll, m.pitch, m.yaw = float(r), float(p), float(y)
        m.thrust_body = [0.0, 0.0, -float(thr)]
        self.p_rate.publish(m)

    def spin_for(self, secs, fn, hz=100):
        end = time.time() + secs
        period = 1.0 / hz
        while time.time() < end and rclpy.ok():
            t = time.time()
            fn()
            rclpy.spin_once(self, timeout_sec=0)
            dt = period - (time.time() - t)
            if dt > 0:
                time.sleep(dt)

    def hold_pos(self, x, y, z, secs, yaw=0.0):
        def f():
            self.mode(False)
            self.pos_sp(x, y, z, yaw)
            if self.lpos:
                self.log["track"].append([round(time.time() - self.t0, 3), self.lpos.x, self.lpos.y, self.lpos.z])
        self.spin_for(secs, f, hz=50)


def main():
    rclpy.init()
    n = Probe()
    for _ in range(30):
        n.spin_for(1, lambda: None)
        if n.lpos is not None and n.status is not None and n.att is not None:
            break
    if n.lpos is None or n.status is None:
        print("no PX4 data on", NS)
        return 1
    n.ev("start", landed_xy=(n.lpos.x, n.lpos.y))
    gz_pub("/perch/detach", "gz.msgs.Empty", "")       # DetachableJoint attaches on spawn -> release
    n.ev("detached_at_spawn")
    x0, y0 = n.lpos.x, n.lpos.y

    # --- A: arm + offboard position takeoff
    n.hold_pos(x0, y0, -3.0, 1.0)
    n.cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)
    n.cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1)
    n.hold_pos(x0, y0, -3.0, 9.0)
    n.ev("hover_reached")

    # --- B: body-rate mode with own attitude/altitude hold + doublets
    hover_thr = 0.60
    z_ref = n.lpos.z
    seq = []  # (t_start, axis, value, duration)
    t = 2.0
    for axis in (0, 1, 2):
        for amp in (1.5, 3.0):
            seq += [(t, axis, amp, 0.15), (t + 0.15, axis, -amp, 0.15)]
            t += 2.0
    T_total = t + 1.0
    tB = time.time()

    def rate_step():
        el = time.time() - tB
        r, p, y = quat_to_rpy(list(n.att.q))
        cmd = [-4.0 * r, -4.0 * p, 0.0]           # attitude hold (P on angles)
        for (ts, ax, val, dur) in seq:
            if ts <= el < ts + dur:
                cmd = [0.0, 0.0, 0.0]
                cmd[ax] = val
        thr = hover_thr + 0.12 * (n.lpos.z - z_ref) + 0.08 * n.lpos.vz
        thr = thr / max(0.5, math.cos(r) * math.cos(p))
        thr = min(0.95, max(0.1, thr))
        n.mode(True)
        n.rate_sp(*cmd, thr)
        n.log["rate"].append([round(el, 4), *cmd, *n.gyro, thr, n.lpos.z])
    n.spin_for(T_total, rate_step, hz=100)
    n.ev("rate_phase_done", z=n.lpos.z)

    # --- C: go next to the plate; plate centre ENU (7.2, 0, 1.3) -> NED (N=0, E=7.2, D=-1.3)
    # local origin = spawn point (ENU 0,0) so NED target is x=0, y=6.55 (0.65 m in front of plate face)
    n.hold_pos(0.0, 6.55, -1.45, 10.0, yaw=math.pi / 2)
    n.ev("at_plate", pos=(n.lpos.x, n.lpos.y, n.lpos.z))
    gz_pub("/perch/attach", "gz.msgs.Empty", "")
    n.hold_pos(n.lpos.x, n.lpos.y, n.lpos.z, 1.0, yaw=math.pi / 2)
    n.ev("attached")
    if PERCH_MODE:
        n.cmd(VEHICLE_CMD_USER_1, 1)
        n.ev("perch_mode_on")
    # motors off while attached: normal disarm (expected to be rejected in air), then forced
    n.cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0)
    n.hold_pos(n.lpos.x, n.lpos.y, n.lpos.z, 1.0, yaw=math.pi / 2)
    n.ev("after_plain_disarm")
    if n.status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
        n.cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0, 21196)
        n.hold_pos(n.lpos.x, n.lpos.y, n.lpos.z, 1.0, yaw=math.pi / 2)
        n.ev("after_forced_disarm")
    z_att = n.lpos.z
    n.hold_pos(n.lpos.x, n.lpos.y, n.lpos.z, 4.0, yaw=math.pi / 2)
    n.ev("perched_motors_off_4s", dz=n.lpos.z - z_att)
    # ride-along: cart drives +x ENU at 1 m/s for 5 s
    gz_pub("/model/perch_cart/cmd_vel", "gz.msgs.Twist", "linear: {x: 1.0}")
    n.hold_pos(n.lpos.x, n.lpos.y, n.lpos.z, 5.0, yaw=math.pi / 2)
    gz_pub("/model/perch_cart/cmd_vel", "gz.msgs.Twist", "linear: {x: 0.0}")
    n.hold_pos(n.lpos.x, n.lpos.y, n.lpos.z, 2.0, yaw=math.pi / 2)
    n.ev("ride_along_done", pos=(n.lpos.x, n.lpos.y, n.lpos.z), vel=(n.lpos.vx, n.lpos.vy, n.lpos.vz))
    # re-arm (offboard setpoints are streaming at the current position), then detach and depart
    hx, hy, hz = n.lpos.x, n.lpos.y, n.lpos.z
    n.cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)
    n.cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1)
    n.hold_pos(hx, hy, hz, 2.0, yaw=math.pi / 2)
    n.ev("after_plain_arm")
    if n.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
        n.cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1, 21196)
        n.hold_pos(hx, hy, hz, 2.0, yaw=math.pi / 2)
        n.ev("after_forced_arm")
    gz_pub("/perch/detach", "gz.msgs.Empty", "")
    if PERCH_MODE:
        n.cmd(VEHICLE_CMD_USER_1, 0)
    z_det = n.lpos.z
    zmax = [z_det]

    def depart_watch():
        zmax[0] = max(zmax[0], n.lpos.z)
    n.hold_pos(hx, hy, hz, 1.5, yaw=math.pi / 2)
    depart_watch()
    n.ev("detached", drop_m=zmax[0] - z_det)
    n.hold_pos(hx, hy - 2.0, hz - 1.0, 6.0, yaw=math.pi / 2)
    n.ev("departed", pos=(n.lpos.x, n.lpos.y, n.lpos.z))
    n.cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
    n.spin_for(1, lambda: None)
    json.dump(n.log, open(OUT, "w"))
    print("saved", OUT)
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
