"""PerchPilot perch node (phase P5-early): learned perching policy in the loop with PX4 SITL.

  EKF2 (/px4_7/fmu/out/vehicle_odometry) ---------------------------.
  /perch/tag_pose -> EKF2 composition -> PlateKF ----------------------+-> observations -> policy.ts
                                                                         `-> PhaseMachine (APPROACH/PERCH/VERIFY/REST/DEPART/ABORT)
  -> /px4_7/fmu/in/{offboard_control_mode, vehicle_rates_setpoint, trajectory_setpoint, vehicle_command}
  -> /perchpilot/diag (std_msgs/String JSON), /perch/hold_cmd (std_msgs/Float64)

Rule R1: drone odometry comes from EKF2 and plate measurements come from /perch/tag_pose.

Run (torch is not installed for the system python; the occwm venv has the same CPython 3.12, so the
sourced ROS 2 PYTHONPATH works with it):
    source /opt/ros/jazzy/setup.bash && source ~/ws_px4/install/setup.bash
    export ROS_DOMAIN_ID=77
    /home/sup/data/venvs/occwm/bin/python perchpilot/ros/perch_node.py --run RUN_DIR [options]
See README_perch_node.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perch_core import (ActionScale, DroneState, Inputs, Phase, PhaseCfg, PhaseMachine, PolicyRunner,  # noqa: E402
                        RollingStats, action_to_px4, drone_state_from_odometry, enu_to_ned, level_hold_action,
                        read_px4_param, plate_from_cart, quat_to_rot, rot_to_quat, set_px4_param, yaw_enu_to_ned)
from plate_perception import PlateKF, TAG_FRAME, TAG_PERIOD, TAG_TOPIC, plate_in_camera, tag_measurement  # noqa: E402

import rclpy  # noqa: E402
import rclpy.executors  # noqa: E402
import rclpy.task  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402

HOLD_SENSORS_QOS = "reliable/keep_last"     # recorded in eval provenance (q2 *_qos batches)
HOLD_SENSORS_QOS_DEPTH = 50
from geometry_msgs.msg import PoseStamped  # noqa: E402
from std_msgs.msg import Float64, Float64MultiArray, String  # noqa: E402
from px4_msgs.msg import (OffboardControlMode, SensorCombined, TrajectorySetpoint, VehicleCommand,  # noqa: E402
                          VehicleControlMode, VehicleOdometry, VehicleRatesSetpoint, VehicleStatus)
from perch_q2 import BeliefNet, N_CH, Q2Verifier, SensorWindow, load_q2_cfg  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="dir with policy.ts + policy.json (contract C1)")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    ap.add_argument("--ns", default="/px4_7")
    ap.add_argument("--target-system", type=int, default=8, help="MAVLink system id of PX4 instance 7")
    ap.add_argument("--px4-instance", type=int, default=7)
    ap.add_argument("--hover-thrust", type=float, default=None,
                    help="normalised hover thrust; default: read MPC_THR_HOVER via px4-param, else 0.60")
    ap.add_argument("--thrust-model", default="linear", choices=["linear", "x500_gz", "x500_perch_gz"],
                    help="hover-scaled linear mapping or quadratic Gazebo mapping for the selected model")
    ap.add_argument("--pad-offset", type=float, default=0.05)
    ap.add_argument("--rate-hz", type=float, default=100.0)
    ap.add_argument("--diag-hz", type=float, default=20.0)
    ap.add_argument("--perception", default="tag", choices=["tag"])
    ap.add_argument("--kf-q-acc", type=float, default=0.1,
                    help="KF white-acceleration PSD [m^2/s^3]; 0.5 gave |v| noise ~0.3 m/s on a static cart in SITL")
    ap.add_argument("--verify-rule", default="placeholder", choices=["placeholder", "none", "dthr", "fsm_b4b", "fsm_b4e", "probe"],
                    help="VERIFY: Q2 rule (none = motors off at touchdown; probe = full v2 schedule, measurement)")
    ap.add_argument("--q2-belief", default="/home/sup/data/perchpilot/runs/q2_belief_v2_small",
                    help="dir with belief.ts + belief.json (C5 raw window in)")
    ap.add_argument("--q2-decide-json", default=str(Path(__file__).resolve().parents[1] / "results" /
                                                    "q2_decide_v3_v2_small_ens5.json"),
                    help="decide JSON with chosen_on_val thresholds (tau, s for D-thr; e_thr for B4b)")
    ap.add_argument("--q2-dthr-key", default="D-thr_v3_early_stop")
    ap.add_argument("--q2-slip-off", type=float, nargs=3, default=None,
                    help="B4e slip_off_mm per decision point (RELAX, PULL0.3, PULL0.6), e.g. from a v6 lock")
    ap.add_argument("--q2-tau", type=float, default=None, help="D-thr p_hold threshold (overrides the JSON)")
    ap.add_argument("--q2-s", type=float, default=None, help="D-thr p_slip early-stop threshold (overrides)")
    ap.add_argument("--q2-dmin", type=int, default=None, help="earliest decision point for MOTORS_OFF (overrides)")
    ap.add_argument("--q2-log", default=None, help="JSONL of the 200 Hz C5 samples around contact")
    ap.add_argument("--depart-pushoff", action="store_true",
                    help="DEPART in body-rate offboard: arm idle, release, push off along body z, level, then position")
    ap.add_argument("--freeze-plate-v", action="store_true",
                    help="PERCH: keep the plate velocity estimated before the manoeuvre (position-only KF updates)")
    ap.add_argument("--yaw-scale", type=float, default=1.0,
                    help="scale of the yaw-rate command sent to PX4 in PERCH (0 = no yaw; obs keep the policy action)")
    ap.add_argument("--extra-delay-steps", type=int, default=0,
                    help="hold each policy command N extra ticks (match the trained command delay)")
    # phase machine
    ap.add_argument("--approach-back", type=float, default=4.0,
                    help="hold point this far behind the plate centre (horizontal, against vehicle forward) [m]")
    ap.add_argument("--approach-up", type=float, default=1.0, help="hold point height above the plate centre [m]")
    ap.add_argument("--rest-s", type=float, default=5.0)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--perch-lost-s", type=float, default=1.0)
    ap.add_argument("--eval-mode", action="store_true", help="wait for /perchpilot/cmd start/reset")
    ap.add_argument("--standby-offset", type=float, nargs=3, default=(2.5, 0.0, 2.2),
                    help="evaluation hover offset [m ENU] from initial EKF position")
    # logging / offline
    ap.add_argument("--log-dir", default=None, help="write ticks.jsonl + summary.json here")
    ap.add_argument("--tick-log", default=None,
                    help="JSONL with the full observation, action, EKF state and plate estimate of every PERCH tick")
    ap.add_argument("--px4-param", action="append", default=[], metavar="NAME=VALUE",
                    help="set a PX4 parameter at startup through px4-param (e.g. MC_AIRMODE=1)")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N s (0 = run until Ctrl-C)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no PX4/Gazebo: synthetic static state, PERCH forever; measures inference time + loop jitter")
    return ap.parse_args(argv)


class PerchNode(Node):
    def __init__(self, a):
        super().__init__("perchpilot_perch_node")
        self.a = a
        self.policy = PolicyRunner(a.run, a.backend)
        meta = self.policy.meta
        hover, src = a.hover_thrust, "cli"
        if hover is None and not a.dry_run:
            hover, src = read_px4_param("MPC_THR_HOVER", a.px4_instance), "px4-param"
        if hover is None:
            hover, src = 0.60, "default (4001_gz_x500 airframe)"
        self.scale = ActionScale(rate_max=tuple(meta["rate_max"]), tw_ratio=float(meta["tw_ratio_nominal"]),
                                 hover_thrust=float(hover), thrust_model=a.thrust_model)
        for item in a.px4_param:
            name, value = item.split("=", 1)
            ok = set_px4_param(name, value, a.px4_instance)
            self.get_logger().info(f"px4 param {name}={value}: {'ok' if ok else 'FAILED'}")
            if not ok:     # review D08: a required PX4 parameter that did not apply stops the batch
                raise SystemExit(f"PX4 parameter {name}={value} could not be set")
        pcfg = PhaseCfg(approach_back=a.approach_back, approach_up=a.approach_up, rest_s=a.rest_s, max_attempts=a.max_attempts,
                        perch_lost_s=a.perch_lost_s, extra_delay_steps=a.extra_delay_steps,
                        yaw_cmd_scale=a.yaw_scale, depart_pushoff=a.depart_pushoff,
                        standby_offset=tuple(a.standby_offset) if a.eval_mode else None)
        if a.dry_run:
            pcfg.max_steps, pcfg.perch_lost_s = 10 ** 9, 1e9
        verifier = None
        if a.verify_rule != "placeholder":
            q2cfg = load_q2_cfg(a.verify_rule, a.q2_decide_json, a.q2_dthr_key)
            # explicit thresholds (e.g. a Q2 v4 lock file: selection["D-thr"]["0.02"]) override the JSON
            if a.q2_tau is not None:
                q2cfg.tau = a.q2_tau
            if a.q2_s is not None:
                q2cfg.s = a.q2_s
            if a.q2_dmin is not None:
                q2cfg.d_min = a.q2_dmin
            if a.verify_rule == "fsm_b4e":
                if a.q2_slip_off is None:
                    raise SystemExit("--verify-rule fsm_b4e needs --q2-slip-off")
                q2cfg.slip_off_mm = list(a.q2_slip_off)
            verifier = Q2Verifier(q2cfg, BeliefNet(a.q2_belief) if a.verify_rule == "dthr" else None)
            self.get_logger().info(f"Q2 verify rule {a.verify_rule}: tau={q2cfg.tau} s={q2cfg.s} "
                                   f"e_thr={q2cfg.e_thr:.4f} slip_off_mm={q2cfg.slip_off_mm} schedule={q2cfg.schedule}")
        self.pm = PhaseMachine(pcfg, self.policy, self.scale, a.pad_offset,
                               start=Phase.PERCH if a.dry_run else Phase.WAIT, verifier=verifier)
        if a.dry_run:
            self.pm._reset_perch()
        self.kf = PlateKF(q_acc=a.kf_q_acc)
        self.n_meas = 0
        self.trial = 0
        self.contact = None
        self.f_normal = 0.0
        self.slip_disp = None                     # pending u/w displacement from HoldPad, if provided
        self.imu = None                           # (acc_frd, gyro_frd) from sensor_combined
        self.q2win = SensorWindow()
        self.q2_f = None
        self.hold_enabled = False
        self.ekf_hist = deque(maxlen=50)           # ~0.5 s of EKF2 odometry for camera-time composition
        self.rates_only = None                    # PX4 vehicle_control_mode: body-rate control without attitude loop
        self.tick_f = None                        # opened at the first PERCH tick (the evaluator owns the dir)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        ns = a.ns
        self.pubs = {}
        if not a.dry_run:
            self.pubs = {
                "mode": self.create_publisher(OffboardControlMode, ns + "/fmu/in/offboard_control_mode", qos),
                "traj": self.create_publisher(TrajectorySetpoint, ns + "/fmu/in/trajectory_setpoint", qos),
                "rate": self.create_publisher(VehicleRatesSetpoint, ns + "/fmu/in/vehicle_rates_setpoint", qos),
                "cmd": self.create_publisher(VehicleCommand, ns + "/fmu/in/vehicle_command", qos),
                "hold": self.create_publisher(Float64, "/perch/hold_cmd", 10),
            }
            self.create_subscription(VehicleOdometry, ns + "/fmu/out/vehicle_odometry", self._odom, qos)
            self.create_subscription(VehicleStatus, ns + "/fmu/out/vehicle_status", self._status, qos)
            self.create_subscription(VehicleControlMode, ns + "/fmu/out/vehicle_control_mode", self._ctrl_mode, qos)
            if a.verify_rule != "placeholder":
                self.create_subscription(SensorCombined, ns + "/fmu/out/sensor_combined", self._imu, qos)
            tag_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
            self.create_subscription(PoseStamped, TAG_TOPIC, self._tag, tag_qos)
            if a.eval_mode:
                self.create_subscription(String, "/perchpilot/cmd", self._trial_cmd, 10)
                # reliable, deep queue: every 5 ms HoldPad sample carries a slip increment (C5 v2). With the
                # best-effort depth-1 profile, q2_probe_m4 had ~50% exactly-zero window samples at RTF 1.0 and
                # the summed slip read ~0.6-0.7x truth: consistent with dropped samples
                hold_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                                      depth=HOLD_SENSORS_QOS_DEPTH)
                self.create_subscription(Float64MultiArray, "/perch/hold_sensors", self._hold_sensor, hold_qos)
        self.p_diag = self.create_publisher(String, "/perchpilot/diag", 10)

        self.drone = None
        self.armed = self.offboard = False
        self.n_odom = 0
        if a.dry_run:
            self._init_dry()

        self.st_loop, self.st_jit, self.st_inf, self.st_tick = RollingStats(), RollingStats(), RollingStats(), RollingStats()
        self.t_prev = None
        self.t_start = time.monotonic()
        self.t_diag = self.t_report = 0.0
        self.done_future = rclpy.task.Future()
        self.n_ticks = 0
        self.log_f = None
        if a.log_dir:
            Path(a.log_dir).mkdir(parents=True, exist_ok=True)
            self.log_f = open(Path(a.log_dir) / "ticks.jsonl", "w")
        self.get_logger().info(json.dumps({"policy_backend": self.policy.backend, "obs_dim": self.policy.obs_dim,
                                           "hover_thrust": hover, "hover_src": src, "rate_max": self.scale.rate_max,
                                           "tw_ratio": self.scale.tw_ratio, "thrust_model": a.thrust_model,
                                           "max_thrust_g": round(self.scale.max_acc_g(), 3),
                                           "perception": a.perception,
                                           "dry_run": a.dry_run}))
        self.timer = self.create_timer(1.0 / a.rate_hz, self.tick)
        if a.verify_rule != "placeholder":
            self.q2_timer = self.create_timer(0.005, self._q2_sample)

    # ------------------------------------------------------------------ callbacks
    def _odom(self, m):
        # review D10: drop non-finite samples and degenerate quaternions
        if (m.pose_frame != VehicleOdometry.POSE_FRAME_NED or not np.all(np.isfinite(m.position))
                or not np.all(np.isfinite(m.q)) or np.linalg.norm(m.q) < 0.5
                or not np.all(np.isfinite(m.velocity)) or not np.all(np.isfinite(m.angular_velocity))):
            return
        try:
            self.drone = drone_state_from_odometry(m.position, m.q, m.velocity, m.angular_velocity, m.velocity_frame,
                                                   t=time.monotonic())
        except ValueError as e:                    # unknown velocity frame: drop the sample (state goes stale)
            self.get_logger().warning(str(e), throttle_duration_sec=2.0)
            return
        self.n_odom += 1
        self.ekf_hist.append(self.drone)

    def _drone_at(self, t):
        """EKF state closest to time t (the camera frame time), so a fast rotation between the frame and the
        latest odometry sample does not rotate the plate measurement (6 rad/s x 20 ms = 7 deg)."""
        if not self.ekf_hist:
            return self.drone
        return min(self.ekf_hist, key=lambda s: abs(s.t - t))

    def _status(self, m):
        self.armed = m.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.offboard = m.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def _ctrl_mode(self, m):
        self.rates_only = bool(m.flag_control_rates_enabled and not m.flag_control_attitude_enabled
                               and not m.flag_control_position_enabled)

    def _trial_cmd(self, msg):
        if msg.data == "reset":
            self.pm.reset_trial()
        elif msg.data.split(":")[0] == "start" and self.pm.start_trial(time.monotonic()):
            # "start:<k>" carries the evaluator's trial index (survives a SITL restart); plain "start" counts up
            self.trial = int(msg.data.split(":")[1]) if ":" in msg.data else self.trial + 1
            self.kf = PlateKF(q_acc=self.a.kf_q_acc)  # discard state from before the cart set_pose
            self.hold_enabled = True
            self.pubs["hold"].publish(Float64(data=1.0))

    def _hold_sensor(self, msg):
        if len(msg.data) in (3, 5):
            self.contact = time.monotonic(), msg.data[1] == 1.0
            self.f_normal = float(msg.data[2])
            if len(msg.data) == 5:
                if self.slip_disp is None:
                    self.slip_disp = np.zeros(2)
                self.slip_disp += msg.data[3:5]
            else:
                self.slip_disp = None

    def _imu(self, m):
        self.imu = (np.array(m.accelerometer_m_s2, dtype=float), np.array(m.gyro_rad, dtype=float))

    def _q2_sample(self):
        """200 Hz C5 sample: IMU FRD -> FLU (x, -y, -z), load cell, contact, thrust_cmd01, phase_id, probe_level."""
        if self.imu is None:
            return
        acc, gyro = self.imu
        flu = np.array([1.0, -1.0, -1.0])
        now = time.monotonic()
        contact = self.contact is not None and now - self.contact[0] < 0.2 and self.contact[1]
        thr01, ph, lv = self.pm.q2_channels()
        slip = self.slip_disp if self.slip_disp is not None else ()
        self.q2win.add(acc * flu, gyro * flu, self.f_normal, contact, thr01, ph, lv, slip)
        if self.slip_disp is not None:
            self.slip_disp.fill(0.0)  # consume each delivered displacement once
        if self.a.q2_log and self.pm.phase in (Phase.PERCH, Phase.VERIFY, Phase.REST, Phase.DEPART):
            if self.q2_f is None:
                Path(self.a.q2_log).parent.mkdir(parents=True, exist_ok=True)
                self.q2_f = open(self.a.q2_log, "a", buffering=1)
            row = self.q2win.buf[-1]
            self.q2_f.write(json.dumps({"wall": round(now, 5), "trial": self.trial, "phase": self.pm.phase.value,
                                        "x": [round(float(v), 5) for v in row]}) + "\n")

    def _tag(self, msg):
        drone = self.drone
        now = time.monotonic()
        if msg.header.frame_id != TAG_FRAME or drone is None or now - drone.t > 0.2:
            return
        stamp = msg.header.stamp
        stamp_ns = stamp.sec * 10**9 + stamp.nanosec
        age = (self.get_clock().now().nanoseconds - stamp_ns) * 1e-9
        if not 0 <= age <= 0.2:
            return
        t = now - age
        if self.kf.t_meas is not None and t <= self.kf.t_meas:
            return
        p, q = msg.pose.position, msg.pose.orientation
        c, quat = np.array([p.x, p.y, p.z]), np.array([q.w, q.x, q.y, q.z])
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-6:
            return
        self.kf.update(tag_measurement(t, self._drone_at(t), c, quat_to_rot(quat)))
        self.n_meas += 1

    # ------------------------------------------------------------------ dry run
    def _init_dry(self):
        """Static synthetic scene: cart at the SDF spawn pose, drone 2 m in front of the plate facing it."""
        c, Rp = plate_from_cart([8.0, 0.0, 0.45], np.eye(3))
        n = Rp[:, 2]
        self.drone = DroneState(p=c + 2.0 * n, v=np.zeros(3), R=np.eye(3), w=np.zeros(3), t=time.monotonic())
        self.dry_tag = plate_in_camera(self.drone.p, self.drone.R, c, Rp)
        self.next_dry_tag = self.drone.t
        self.armed = self.offboard = True

    # ------------------------------------------------------------------ main loop
    def _now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def tick(self):
        t = time.monotonic()
        if self.t_prev is not None:
            dt = t - self.t_prev
            self.st_loop.add(dt)
            self.st_jit.add(abs(dt - 1.0 / self.a.rate_hz))
        self.t_prev = t
        a = self.a
        drone = self.drone
        if drone is not None and not a.dry_run and t - drone.t > 0.2:
            drone = None                                # stale EKF -> publish nothing -> PX4 offboard failsafe
        if a.dry_run and t >= self.next_dry_tag:
            self.kf.update(tag_measurement(t, drone, *self.dry_tag))
            self.n_meas += 1
            self.next_dry_tag = max(self.next_dry_tag + TAG_PERIOD, t)
        self.kf.freeze_v = a.freeze_plate_v and self.pm.phase in (Phase.PERCH, Phase.VERIFY)
        plate = self.kf.estimate(t)
        contact = self.contact[1] if self.contact is not None and t - self.contact[0] < 0.2 else None
        win = self.q2win.array() if self.pm.phase == Phase.VERIFY and self.q2win.ready() else None
        out = self.pm.step(Inputs(t=t, drone=drone, armed=self.armed, offboard=self.offboard, plate=plate,
                                  contact=contact, rates_mode=None if a.dry_run else bool(self.rates_only),
                                  q2_window=win))
        if out.mode == "position" and self.rates_only and drone is not None:
            # PX4 keeps flying our last rate setpoint until its 2 Hz control-mode update: make it a level hold.
            out.rates_frd, out.thrust_body, out.thrust_norm, _, _ = action_to_px4(
                level_hold_action(drone, self.scale), self.scale)
        if a.tick_log and drone is not None and out.phase in (Phase.PERCH, Phase.VERIFY):
            self._log_tick(t, out, drone, plate)
        if math.isfinite(out.infer_s):
            self.st_inf.add(out.infer_s)
        for e in out.events:
            self.get_logger().info(f"[phase] {e}")
        self._publish(out)

        # --- diagnostics / logging
        self.n_ticks += 1
        rec = {"t": round(t - self.t_start, 4), "phase": out.phase.value, "attempt": self.pm.attempt,
               "rho": round(plate.reliability, 3) if plate else 0.0,
               "plate_age": round(plate.age, 3) if plate else None,
               "plate_vis": round(plate.visible_for, 2) if plate else 0.0,
               "tag_visible": bool(plate and plate.visible),
               "action": None if out.action is None else [round(float(x), 4) for x in out.action],
               "thrust_norm": None if not math.isfinite(out.thrust_norm) else round(out.thrust_norm, 4),
               "rates_frd": None if out.rates_frd is None else [round(float(x), 3) for x in out.rates_frd],
               "loop_dt_ms": round(1e3 * self.st_loop.buf[-1], 3) if self.st_loop.buf else None,
               "infer_ms": round(1e3 * out.infer_s, 4) if math.isfinite(out.infer_s) else None,
               "armed": self.armed, "offboard": self.offboard, "events": out.events,
               "trial": self.trial, "ready": bool(self.pm.ready), "result": self.pm.result,
               "q2": self.pm.q2_result}
        if drone is not None:
            rec["p_enu"] = [round(float(x), 3) for x in drone.p]
        if plate is not None:
            rec["plate_c_enu"] = [round(float(x), 3) for x in plate.c]
            rec["plate_v_enu"] = [round(float(x), 3) for x in plate.v]
        if self.log_f:
            self.log_f.write(json.dumps(rec) + "\n")
        if t - self.t_diag >= 1.0 / a.diag_hz:
            self.t_diag = t
            verifier = self.pm.verifier
            if verifier is not None and out.phase == Phase.VERIFY:
                rec["q2_probe"] = {"macro": verifier.macro,
                                   "last_point": verifier.decisions[-1] if verifier.decisions else None}
                if win is not None and win.shape[1] >= N_CH + 2:
                    rec["q2_probe"]["slip_mm"] = float(np.linalg.norm(win[:, N_CH:N_CH + 2].sum(0)))
            if verifier is not None and verifier.cfg.rule == "fsm_b4e":
                rec["b4e_thresholds"] = dict(zip(verifier.cfg.schedule, verifier.cfg.slip_off_mm))
            rec["loop_ms"] = self.st_loop.summary(1e3)
            rec["jitter_ms"] = self.st_jit.summary(1e3)
            rec["infer_ms_stats"] = self.st_inf.summary(1e3)
            self.p_diag.publish(String(data=json.dumps(rec)))
            if a.eval_mode and self.pubs:
                self.pubs["hold"].publish(Float64(data=float(self.hold_enabled)))
        if t - self.t_report >= 5.0:
            self.t_report = t
            self.get_logger().info(json.dumps({"phase": out.phase.value, "odom_msgs": self.n_odom,
                                               "meas": self.n_meas,
                                               **self.timing()}))
        self.st_tick.add(time.monotonic() - t)            # compute time of this callback
        if a.duration > 0 and t - self.t_start >= a.duration and not self.done_future.done():
            self.done_future.set_result(True)

    def _log_tick(self, t, out, drone, plate):
        def r(v, n=5):
            return [round(float(x), n) for x in np.asarray(v).reshape(-1)]
        rec = {"wall": round(t, 5), "phase": out.phase.value, "k": self.pm.k, "trial": self.trial,
               "switching": bool(self.pm.flag.get("switching")), "rates_only": self.rates_only,
               "obs": None if out.obs is None else r(out.obs),
               "action": None if out.action is None else r(out.action),
               "thrust_norm": None if not math.isfinite(out.thrust_norm) else round(out.thrust_norm, 5),
               "rates_frd": None if out.rates_frd is None else r(out.rates_frd),
               "ekf": {"t": round(drone.t, 5), "p": r(drone.p), "v": r(drone.v), "q": r(rot_to_quat(drone.R)),
                       "w": r(drone.w)}}
        if plate is not None:
            rec["plate"] = {"c": r(plate.c), "q": r(rot_to_quat(plate.R)), "v": r(plate.v),
                            "age": round(plate.age, 4), "visible": bool(plate.visible)}
        if self.tick_f is None:
            Path(self.a.tick_log).parent.mkdir(parents=True, exist_ok=True)
            self.tick_f = open(self.a.tick_log, "a", buffering=1)
        self.tick_f.write(json.dumps(rec) + "\n")

    def _publish(self, out):
        if not self.pubs or out.mode == "none":
            return
        ts = self._now_us()
        mode = OffboardControlMode()
        mode.timestamp = ts
        mode.position = out.mode == "position"
        mode.body_rate = out.mode == "rates"
        self.pubs["mode"].publish(mode)
        if out.pos_enu is not None:          # also while PX4 still runs its position loop after a switch to rates
            sp = TrajectorySetpoint()
            sp.timestamp = ts
            sp.position = [float(x) for x in enu_to_ned(out.pos_enu)]
            sp.velocity = [float(x) for x in enu_to_ned(out.vel_enu)] if out.vel_enu is not None else [math.nan] * 3
            sp.acceleration = [math.nan] * 3
            sp.yaw = float(yaw_enu_to_ned(out.yaw_enu)) if math.isfinite(out.yaw_enu) else math.nan
            sp.yawspeed = math.nan
            self.pubs["traj"].publish(sp)
        if out.rates_frd is not None:        # also while PX4 still runs body-rate control after a switch back
            r = VehicleRatesSetpoint()
            r.timestamp = ts
            r.roll, r.pitch, r.yaw = (float(x) for x in out.rates_frd)
            r.thrust_body = [float(x) for x in out.thrust_body]
            self.pubs["rate"].publish(r)
        for (cmd, p1, p2) in out.commands:
            m = VehicleCommand()
            m.timestamp = ts
            m.command, m.param1, m.param2 = int(cmd), float(p1), float(p2)
            m.target_system, m.target_component, m.source_system, m.source_component = self.a.target_system, 1, 1, 1
            m.from_external = True
            self.pubs["cmd"].publish(m)
            self.get_logger().info(f"[cmd] {cmd} p1={p1} p2={p2}")
        if out.hold_cmd is not None:
            self.hold_enabled = bool(out.hold_cmd)
            self.pubs["hold"].publish(Float64(data=float(out.hold_cmd)))

    def timing(self):
        return {"loop_ms": self.st_loop.summary(1e3), "jitter_ms": self.st_jit.summary(1e3),
                "infer_ms": self.st_inf.summary(1e3), "tick_compute_ms": self.st_tick.summary(1e3)}

    def summary(self):
        return {"ticks": self.n_ticks, "wall_s": round(time.monotonic() - self.t_start, 2),
                "policy_backend": self.policy.backend, "phase": self.pm.phase.value, "result": self.pm.result,
                "meas": self.n_meas, **self.timing()}


def main(argv=None):
    a = parse_args(argv)
    rclpy.init()
    node = PerchNode(a)
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin_until_future_complete(node.done_future)   # one wait-set for timer + subscriptions
    except KeyboardInterrupt:
        pass
    s = node.summary()
    print(json.dumps(s, indent=1))
    if a.log_dir:
        (Path(a.log_dir) / "summary.json").write_text(json.dumps(s, indent=1))
        node.log_f.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
