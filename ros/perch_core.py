"""PerchPilot perch node - ROS-free core (frames, observation v1, action mapping,
policy runner, phase machine, timing stats).

Everything here is plain numpy so it can be unit-tested without ROS / PX4 / Gazebo.
`perch_node.py` only wires this core to rclpy + px4_msgs + the plate perception emulator.

Frame conventions (READ BEFORE EDITING):
  * PX4:  local/world frame NED (x North, y East, z Down); body frame FRD
          (x Forward, y Right, z Down). PX4 quaternion q = [w, x, y, z] rotates
          FRD body -> NED (R_ned_frd).
  * Sim / Gazebo / this core: world ENU (x East, y North, z Up); body FLU
          (x Forward, y Left, z Up = thrust axis). QuadPerchEnv uses q = [w,x,y,z]
          rotating FLU body -> ENU and quat_to_rot() gives R with R[:, 2] = body z.
  * Conversions are pure permutation / sign flips:
        v_enu = T v_ned,  T = [[0,1,0],[1,0,0],[0,0,-1]]   (T = T^-1)
        v_flu = F v_frd,  F = diag(1,-1,-1)                (F = F^-1)
        R_enu_flu = T R_ned_frd F
  * Body rates FRD (p, q, r) <-> FLU (p, -q, -r).
  * Thrust: sim thrust acts along +z_FLU; PX4 VehicleRatesSetpoint.thrust_body is in FRD,
    so thrust_body = [0, 0, -thrust_normalised].
  * Yaw: yaw_ned (from North, clockwise) = pi/2 - yaw_enu (from East, counter-clockwise).
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

G = 9.81
TAG_LOSS_MAX = 2.0

# ----------------------------------------------------------------------------- frames
T_NED_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # NED<->ENU (self-inverse)
F_FRD_FLU = np.diag([1.0, -1.0, -1.0])                                      # FRD<->FLU (self-inverse)


def ned_to_enu(v):
    return T_NED_ENU @ np.asarray(v, dtype=float)


def enu_to_ned(v):
    return T_NED_ENU @ np.asarray(v, dtype=float)


def frd_to_flu(v):
    return F_FRD_FLU @ np.asarray(v, dtype=float)


def flu_to_frd(v):
    return F_FRD_FLU @ np.asarray(v, dtype=float)


def quat_to_rot(q):
    """q = [w, x, y, z] (any frames) -> 3x3 rotation matrix. Same formula as sim.quat_to_rot."""
    w, x, y, z = (float(c) for c in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(R):
    """3x3 rotation -> q = [w, x, y, z] with w >= 0 (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def R_enu_flu_from_px4_q(q_ned_frd):
    """PX4 attitude quaternion (FRD -> NED) -> rotation matrix FLU -> ENU."""
    return T_NED_ENU @ quat_to_rot(q_ned_frd) @ F_FRD_FLU


def px4_q_from_R_enu_flu(R_enu_flu):
    """Rotation FLU -> ENU -> PX4 quaternion FRD -> NED (inverse of R_enu_flu_from_px4_q)."""
    return rot_to_quat(T_NED_ENU @ np.asarray(R_enu_flu) @ F_FRD_FLU)


def yaw_enu_to_ned(psi_enu):
    return wrap_pi(math.pi / 2 - psi_enu)


def yaw_ned_to_enu(psi_ned):
    return wrap_pi(math.pi / 2 - psi_ned)


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def orthonormalize(M):
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ----------------------------------------------------------------------------- states
@dataclass
class DroneState:
    """Drone state in the local ENU frame (EKF origin) with FLU body axes."""
    p: np.ndarray            # position [m], ENU
    v: np.ndarray            # velocity [m/s], ENU
    R: np.ndarray            # rotation FLU body -> ENU
    w: np.ndarray            # body rates [rad/s], FLU
    t: float = 0.0           # receive time [s] (monotonic)


VELOCITY_FRAME_NED = 1
VELOCITY_FRAME_FRD = 2
VELOCITY_FRAME_BODY_FRD = 3


def drone_state_from_odometry(position, q, velocity, angular_velocity, velocity_frame=VELOCITY_FRAME_NED, t=0.0):
    """px4_msgs/VehicleOdometry fields (NED / FRD) -> DroneState (ENU / FLU). EKF2 data only (rule R1)."""
    R_ned_frd = quat_to_rot(q)
    v = np.asarray(velocity, dtype=float)
    if velocity_frame == VELOCITY_FRAME_BODY_FRD:
        v = R_ned_frd @ v
    elif velocity_frame not in (VELOCITY_FRAME_NED, VELOCITY_FRAME_FRD):
        raise ValueError(f"unsupported velocity_frame {velocity_frame}")
    return DroneState(p=ned_to_enu(position), v=ned_to_enu(v),
                      R=T_NED_ENU @ R_ned_frd @ F_FRD_FLU, w=frd_to_flu(angular_velocity), t=t)


def odometry_from_drone_state(s: DroneState):
    """Inverse of drone_state_from_odometry (NED velocity frame). Used by tests / offline plants."""
    return dict(position=enu_to_ned(s.p), q=px4_q_from_R_enu_flu(s.R), velocity=enu_to_ned(s.v),
                angular_velocity=flu_to_frd(s.w), velocity_frame=VELOCITY_FRAME_NED)


@dataclass
class PlateState:
    """Plate estimate in the local ENU frame. R columns: [x_plate, u (lateral), n (outward normal)]."""
    c: np.ndarray            # centre of the plate SURFACE [m], ENU
    R: np.ndarray            # rotation plate -> ENU; R[:, 2] = outward normal n, R[:, 1] = lateral axis u
    v: np.ndarray            # plate translational velocity [m/s], ENU
    valid: bool = True
    age: float = 0.0         # time since last measurement [s]
    visible_for: float = 0.0  # duration of the current uninterrupted measurement streak [s]
    reliability: float = 1.0  # PerchRL Eq. 12
    visible: bool = True
    loss_s: float = 0.0


def plate_axes(R_plate):
    """(u, w, n) exactly as QuadPerchEnv.plate_frame(): n = outward normal, u = lateral, w = n x u."""
    R_plate = np.asarray(R_plate, dtype=float)
    n = R_plate[:, 2]
    u = R_plate[:, 1]
    return u, np.cross(n, u), n


def plate_tilt(R_plate):
    """Plate tilt from horizontal [rad] = angle between outward normal and world up."""
    return math.acos(max(-1.0, min(1.0, float(np.asarray(R_plate)[2, 2]))))


# Plate placement in perch_world.sdf (cart body frame): pose -0.8 0 0.85 0 -1.1345 0, box 0.45x0.45x0.02
SDF_PLATE_OFFSET = (-0.8, 0.0, 0.85)
SDF_PLATE_PITCH = -1.1345
SDF_PLATE_HALF_THICK = 0.01


def plate_from_cart(p_cart, R_cart, offset=SDF_PLATE_OFFSET, pitch=SDF_PLATE_PITCH, face_offset=SDF_PLATE_HALF_THICK):
    """Cart model pose (ENU) -> (plate surface centre, R_plate). Plate frame = cart frame pitched by `pitch`
    about its y axis, so R_plate[:, 2] = R_cart (-sin|pitch|, 0, cos|pitch|) (faces backwards-up) and
    R_plate[:, 1] = cart lateral axis, matching the sim (n = R_veh nb, u = R_veh[:, 1])."""
    R_cart = np.asarray(R_cart, dtype=float)
    Rp = R_cart @ rot_y(pitch)
    c = np.asarray(p_cart, dtype=float) + R_cart @ np.asarray(offset, dtype=float) + face_offset * Rp[:, 2]
    return c, Rp


# ----------------------------------------------------------------------------- observation v1
OBS_V1_LAYOUT = [["rel_p_uwn", 3], ["rel_v_uwn", 3], ["R_body_in_plate", 9], ["omega_body", 3],
                 ["up_in_plate", 3], ["prev_act", 4], ["t_norm", 1], ["veh_speed", 1], ["plate_tilt", 1]]
TAG_OBS_LAYOUT = [["rho", 1], ["visible", 1], ["t_loss_norm", 1]]
ACT_LAYOUT = ["thrust01", "wx", "wy", "wz"]


def build_obs(drone: DroneState, plate: PlateState, prev_act, t_norm, pad_offset=0.05, act_hist=None,
              tag_inputs=False):
    """Build the v1 prefix, optional action history and optional v2 camera inputs.

    act_hist: optional list of past actions (newest first) appended when the policy was trained
    with act_history (= env.act_buf[:, 1:]).
    """
    u, w, n = plate_axes(plate.R)
    if tag_inputs:
        # Sim v2 reconstructs the lateral axis from the filtered normal and world up.
        u = np.cross(n, [0.0, 0.0, 1.0])
        u /= max(float(np.linalg.norm(u)), 1e-6)
        w = np.cross(n, u)
    pad = drone.p - pad_offset * drone.R[:, 2]          # pad point below CoG along -z_body
    Pm = np.stack([u, w, n], 0)                          # rows = plate axes
    rp = Pm @ (pad - plate.c)
    rv = Pm @ (drone.v - plate.v)
    Rp = Pm @ drone.R                                    # body axes in plate frame
    gp = Pm[:, 2]                                        # world up in plate frame
    parts = [rp, rv, Rp.reshape(-1), drone.w, gp, np.asarray(prev_act, dtype=float),
             [float(t_norm)], [float(np.linalg.norm(plate.v))], [plate_tilt(plate.R)]]
    if act_hist:
        parts += [np.asarray(a, dtype=float) for a in act_hist]
    if tag_inputs:
        parts.append([plate.reliability, float(plate.visible), plate.loss_s / TAG_LOSS_MAX])
    return np.concatenate([np.asarray(p, dtype=float).reshape(-1) for p in parts]).astype(np.float32)


# ----------------------------------------------------------------------------- action mapping
@dataclass
class ActionScale:
    rate_max: Tuple[float, float, float] = (6.0, 6.0, 3.0)
    tw_ratio: float = 3.0           # policy.json tw_ratio_nominal
    hover_thrust: float = 0.60      # PX4 MPC_THR_HOVER (normalised thrust that gives 1 g), "linear" model
    thrust_min: float = 0.0
    thrust_max: float = 1.0
    # "linear": thr = hover_thrust * acc / g (C1 default, PX4 THR_MDL_FAC = 0 assumption)
    # "x500_gz" / "x500_perch_gz": omega = w_min + (w_max - w_min) * u,
    #            F = 4 kf omega^2  ->  u = (sqrt(m acc / (4 kf)) - w_min) / (w_max - w_min)
    thrust_model: str = "linear"
    mass: float = 2.064             # x500_base total link mass [kg]
    kf: Optional[float] = None     # override motorConstant [N/(rad/s)^2]
    w_min: float = 150.0            # SIM_GZ_EC_MIN (4001_gz_x500)
    w_max: float = 1000.0           # SIM_GZ_EC_MAX / maxRotVelocity

    def __post_init__(self):
        if self.kf is None:
            self.kf = 1.14400296e-05 if self.thrust_model == "x500_perch_gz" else 8.54858e-06

    def acc_to_u(self, acc):
        if self.thrust_model == "linear":
            return self.hover_thrust * acc / G
        if self.thrust_model in ("x500_gz", "x500_perch_gz"):
            w = math.sqrt(max(acc, 0.0) * self.mass / (4.0 * self.kf))
            return (w - self.w_min) / (self.w_max - self.w_min)
        raise ValueError(self.thrust_model)

    def u_to_acc(self, u):
        if self.thrust_model == "linear":
            return u / self.hover_thrust * G
        w = self.w_min + (self.w_max - self.w_min) * u
        return 4.0 * self.kf * w * w / self.mass

    def max_acc_g(self):
        return self.u_to_acc(self.thrust_max) / G


def action_to_px4(a, sc: ActionScale):
    """Policy action in [-1,1]^4 -> (rates_frd[3], thrust_body[3], thrust_norm, thrust_acc, saturated).

    thrust_acc = thrust_N / m = (a0+1)/2 * TW * g   (contract C1)
    thrust_norm = sc.acc_to_u(thrust_acc)            (linear hover-calibrated model or x500 motor model)
    rates: omega_FLU = a[1:4] * rate_max  ->  FRD (p, -q, -r).
    """
    a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
    thrust_acc = (a[0] + 1.0) * 0.5 * sc.tw_ratio * G
    thr_raw = sc.acc_to_u(thrust_acc)
    thr = float(np.clip(thr_raw, sc.thrust_min, sc.thrust_max))
    w_flu = a[1:4] * np.asarray(sc.rate_max, dtype=float)
    rates_frd = flu_to_frd(w_flu)
    return rates_frd, np.array([0.0, 0.0, -thr]), thr, thrust_acc, thr != thr_raw


def px4_to_action(rates_frd, thrust_body, sc: ActionScale):
    """Inverse of action_to_px4 (exact when not saturated). Used by tests / offline plants."""
    w_flu = frd_to_flu(rates_frd)
    thrust_acc = sc.u_to_acc(-float(thrust_body[2]))
    a0 = 2.0 * thrust_acc / (sc.tw_ratio * G) - 1.0
    return np.concatenate([[a0], w_flu / np.asarray(sc.rate_max, dtype=float)])


def hover_action(tw_ratio):
    """Action that commands 1 g of thrust; the sim resets prev_act to this."""
    return np.array([2.0 / tw_ratio - 1.0, 0.0, 0.0, 0.0])


def level_hold_action(drone: "DroneState", sc: ActionScale, k_att=4.0, k_vz=2.0):
    """Policy-unit action that levels the vehicle (keeps yaw, zero yaw rate) with tilt-compensated thrust that
    also damps the vertical speed (a dive is arrested before PX4's position loop takes over).

    Used while PX4 has not yet switched its control mode (commander publishes vehicle_control_mode only at
    2 Hz or on a nav-state change, so an offboard position <-> body-rate switch lags by up to 0.5 s)."""
    z = drone.R[:, 2]
    w_body = k_att * (drone.R.T @ np.cross(z, [0.0, 0.0, 1.0]))
    w_body[2] = 0.0
    acc_up = float(np.clip(G - k_vz * float(drone.v[2]), 0.5 * G, 1.8 * G))
    acc = acc_up / max(float(z[2]), 0.5)
    a = np.concatenate([[2.0 * acc / (sc.tw_ratio * G) - 1.0], w_body / np.asarray(sc.rate_max, dtype=float)])
    return np.clip(a, -1.0, 1.0)


# ----------------------------------------------------------------------------- policy runner
class NumpyMLP:
    """numpy-only runner for `policy.npz` (proposed C1 extension, see README_perch_node.md):
    keys norm_mean, norm_var, n_layers, W0..W{L-1} ([out,in]), b0..b{L-1}; ELU between layers,
    RunningNorm (x-mean)/(sqrt(var)+1e-6) clamped to [-10, 10], output clamped to [-1, 1]."""

    def __init__(self, path):
        d = np.load(path)
        self.mean, self.var = d["norm_mean"].astype(np.float64), d["norm_var"].astype(np.float64)
        L = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float64) for i in range(L)]
        self.b = [d[f"b{i}"].astype(np.float64) for i in range(L)]

    def __call__(self, obs):
        x = np.clip((np.asarray(obs, dtype=np.float64) - self.mean) / (np.sqrt(self.var) + 1e-6), -10, 10)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = W @ x + b
            if i < len(self.W) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))   # ELU
        return np.clip(x, -1.0, 1.0)


def check_policy_meta(meta):
    for k in ("obs_dim", "obs_layout", "act_layout", "rate_max", "tw_ratio_nominal", "dt", "frame"):
        if k not in meta:
            raise ValueError(f"policy.json missing field '{k}' (contract C1)")
    if sum(int(n) for _, n in meta["obs_layout"]) != int(meta["obs_dim"]):
        raise ValueError("policy.json: obs_layout does not sum to obs_dim")
    if [list(x) for x in meta["obs_layout"][:len(OBS_V1_LAYOUT)]] != OBS_V1_LAYOUT:
        raise ValueError(f"policy.json: obs_layout prefix is not observation v1: {meta['obs_layout']}")
    if list(meta["act_layout"]) != ACT_LAYOUT:
        raise ValueError(f"policy.json: act_layout {meta['act_layout']} != {ACT_LAYOUT}")
    if meta["frame"] != "plate_uwn":
        raise ValueError(f"policy.json: frame {meta['frame']} != plate_uwn")
    if abs(float(meta["dt"]) - 0.01) > 1e-9:
        raise ValueError("policy.json: dt must be 0.01 (100 Hz)")
    extra = int(meta["obs_dim"]) - 28 - (3 if has_tag_inputs(meta) else 0)
    if extra % 4:
        raise ValueError("policy.json: act history must be a multiple of 4 (after v1 and optional tag inputs)")


def has_tag_inputs(meta):
    return [list(x) for x in meta["obs_layout"][-3:]] == TAG_OBS_LAYOUT


class PolicyRunner:
    """Loads RUN/policy.json + RUN/policy.ts (TorchScript, CPU, 1 thread) or RUN/policy.npz (numpy)."""

    def __init__(self, run_dir, backend="auto"):
        run = Path(run_dir)
        self.meta = json.loads((run / "policy.json").read_text())
        check_policy_meta(self.meta)
        self.obs_dim = int(self.meta["obs_dim"])
        self.tag_inputs = has_tag_inputs(self.meta)
        self.hist_len = (self.obs_dim - 28 - (3 if self.tag_inputs else 0)) // 4
        self.backend = None
        if backend in ("auto", "torch"):
            try:
                import torch
                torch.set_num_threads(1)
                self._torch = torch
                self._mod = torch.jit.load(str(run / "policy.ts"), map_location="cpu").eval()
                self.backend = "torch"
            except Exception as e:  # noqa: BLE001
                if backend == "torch":
                    raise
                self._err = repr(e)
        if self.backend is None:
            self._np = NumpyMLP(run / "policy.npz")
            self.backend = "numpy"
        self(np.zeros(self.obs_dim, dtype=np.float32))     # warm-up (JIT profiling passes)
        self(np.zeros(self.obs_dim, dtype=np.float32))

    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"obs dim {obs.shape[0]} != policy obs_dim {self.obs_dim}")
        if self.backend == "torch":
            with self._torch.inference_mode():
                out = self._mod(self._torch.from_numpy(obs).unsqueeze(0))
            return np.clip(out[0].numpy().astype(np.float64), -1.0, 1.0)
        return self._np(obs)


# ----------------------------------------------------------------------------- timing stats
class RollingStats:
    def __init__(self, maxlen=5000):
        self.buf = deque(maxlen=maxlen)
        self.n = 0

    def add(self, x):
        self.buf.append(float(x))
        self.n += 1

    def summary(self, scale=1.0):
        if not self.buf:
            return {"n": 0}
        a = np.asarray(self.buf) * scale
        return {"n": self.n, "p50": round(float(np.percentile(a, 50)), 4), "p95": round(float(np.percentile(a, 95)), 4),
                "p99": round(float(np.percentile(a, 99)), 4), "max": round(float(a.max()), 4)}


# ----------------------------------------------------------------------------- phase machine
class Phase(str, Enum):
    WAIT = "WAIT"
    APPROACH = "APPROACH"
    PERCH = "PERCH"
    VERIFY = "VERIFY"
    REST = "REST"
    DEPART = "DEPART"
    ABORT = "ABORT"
    DONE = "DONE"


# PX4 vehicle_command ids (kept here so the core does not import px4_msgs)
CMD_DO_SET_MODE = 176
CMD_ARM_DISARM = 400
CMD_USER_1_PERCH_MODE = 31010
FORCE_MAGIC = 21196


@dataclass
class PhaseCfg:
    # Hold point = plate centre - approach_back * f + approach_up * z_world, with f the horizontal direction the
    # plate faces away from (= vehicle forward). QuadPerchEnv.reset_idx starts episodes 2.7..5.7 m behind and
    # 0.2..1.8 m above the plate centre, moving with the vehicle -> (4.0, 1.0) is near the centre of that box.
    approach_back: float = 4.0
    approach_up: float = 1.0
    approach_tol: float = 0.5       # [m] position tolerance to start PERCH
    approach_vtol: float = 0.3      # [m/s] relative-velocity tolerance (sim: vehicle velocity + 0.3 m/s noise)
    approach_yaw_tol: float = 0.25  # [rad] heading tolerance (sim: vehicle heading +-0.3 rad)
    approach_settle_s: float = 0.3  # all hand-over conditions must hold this long
    extra_delay_steps: int = 0      # PERCH: hold each policy command this many extra ticks before sending it
    # PERCH: scale of the yaw-rate command sent to PX4 (observation/history keep the policy's own action).
    # PX4 mixes yaw last and lowers the collective to make room for it: the policy's constant +-3 rad/s yaw
    # demand cut the real thrust from 2.2 g to ~1.45 g in SITL (P5, 2026-11-20).
    yaw_cmd_scale: float = 1.0
    switch_timeout_s: float = 1.0   # PERCH: max wait for PX4 to report body-rate control before the policy starts
    tag_stable_s: float = 0.3       # plate seen continuously before PERCH
    approach_timeout_s: float = 90.0
    arm_delay_s: float = 1.0        # stream setpoints this long before the first arm / offboard request
    arm_retry_s: float = 1.0
    perch_timeout_s: float = 6.0    # = sim episode length
    perch_lost_s: float = 1.0       # max estimate age during PERCH (camera loses the plate near touchdown)
    contact_d: float = 0.02         # geometric contact fallback: signed pad distance to plate [m]
    plate_half: float = 0.22
    # [m] PERCH -> ABORT when the CoG is this far below the plate centre estimate. Plate-relative because the
    # EKF2 height origin is not the ground in SITL (offsets of -0.24 m and +1.4 m seen in two sessions).
    min_below_plate: float = 1.0
    verify_s: float = 0.5
    rest_s: float = 5.0
    force_disarm_after_s: float = 1.0   # REST: forced disarm (21196) if a plain disarm is rejected
    force_arm_after_tries: int = 2      # DEPART: forced arm after this many rejected plain arms
    depart_dist: float = 2.0
    depart_timeout_s: float = 15.0
    # DEPART from the plate in body-rate offboard (P5, 2026-11-20). Arming in offboard POSITION on the tilted plate
    # starts PX4's smooth-takeoff thrust ramp and its attitude failure detector: after release the drone fell
    # 1.25 m to the ground (SITL truth), the EKF lost yaw (emergency yaw reset -> latched mag fault) and every later
    # arm was denied. Push-off: arm with idle thrust and zero rates, release, thrust along the body z (~ plate
    # normal) with zero rates, level with vz damping, then hand over to position control.
    depart_pushoff: bool = False
    pushoff_g: float = 1.5
    pushoff_s: float = 0.25
    pushoff_level_s: float = 0.6
    pushoff_idle_u: float = 0.05
    abort_back: float = 1.5
    abort_up: float = 1.0
    abort_hold_s: float = 3.0
    max_attempts: int = 3
    max_steps: int = 600            # episode length of the sim [steps] -> t_norm = k / max_steps
    standby_offset: Optional[Tuple[float, float, float]] = None  # managed evaluation, from initial EKF position


@dataclass
class Inputs:
    t: float                                  # monotonic time [s]
    drone: Optional[DroneState]
    armed: bool
    offboard: bool
    plate: Optional[PlateState]
    contact: Optional[bool] = None            # /perch/hold_sensors contact flag; None = not available
    rates_mode: Optional[bool] = None         # PX4 vehicle_control_mode is body-rate only; None = unknown (tests)
    q2_window: Optional[np.ndarray] = None    # latest C5 window (80 x 11) for the Q2 decision rule


@dataclass
class Output:
    phase: Phase
    mode: str = "none"                        # "position" | "rates" | "none"
    pos_enu: Optional[np.ndarray] = None
    vel_enu: Optional[np.ndarray] = None
    yaw_enu: float = float("nan")
    rates_frd: Optional[np.ndarray] = None
    thrust_body: Optional[np.ndarray] = None
    thrust_norm: float = float("nan")
    action: Optional[np.ndarray] = None
    obs: Optional[np.ndarray] = None
    infer_s: float = float("nan")
    commands: List[Tuple[int, float, float]] = field(default_factory=list)
    hold_cmd: Optional[float] = None
    events: List[str] = field(default_factory=list)


def _horiz_unit(v, fallback=(1.0, 0.0, 0.0)):
    h = np.array([v[0], v[1], 0.0])
    nh = np.linalg.norm(h)
    return h / nh if nh > 1e-6 else np.asarray(fallback, dtype=float)


class PhaseMachine:
    """APPROACH -> PERCH -> VERIFY -> REST -> DEPART -> DONE, with ABORT -> APPROACH (max_attempts)."""

    def __init__(self, cfg: PhaseCfg, policy, scale: ActionScale, pad_offset=0.05, start=Phase.WAIT, verifier=None):
        self.cfg, self.policy, self.sc, self.pad_offset = cfg, policy, scale, pad_offset
        self.verifier = verifier        # perch_q2.Q2Verifier (VERIFY = Q2 probing + decision); None = placeholder
        self.q2_result = None
        self.last_acc = G
        self.hist_len = getattr(policy, "hist_len", 0)
        self.tag_inputs = getattr(policy, "tag_inputs", False)
        self.phase = start
        self.t_enter = None
        self.attempt = 1
        self.k = 0
        self.prev_act = hover_action(scale.tw_ratio)
        self.hist = []
        self.last_thr = scale.hover_thrust
        self.anchor = None          # phase-specific stored position (ENU)
        self.n_anchor = None
        self.t_cmd = -1e9
        self.n_tries = 0
        self.flag = {}
        self.result = None
        self.standby = None
        self.ready = False
        self.returning = False

    def start_trial(self, t):
        if self.cfg.standby_offset is None or not self.ready or self.phase != Phase.WAIT:
            return False
        self.ready = False
        self._go(Phase.APPROACH, t, Output(phase=self.phase), "evaluation start")
        return True

    def reset_trial(self):
        if self.cfg.standby_offset is not None and self.phase != Phase.WAIT:
            self.returning = True

    def _return_to_standby(self, inp, out):
        if self.phase == Phase.DEPART:
            return
        if self.result == "depart_timeout":
            # Still on the plate or disarmed: try DEPART again; airborne and free: fly back to standby (P5: a
            # depart that lifted off after its timeout used to leave the node in DONE for the rest of the batch).
            if self._contact(inp) or not inp.armed:
                self.result = None
                self._go(Phase.DEPART, inp.t, out, "evaluation reset: retry depart")
                return
        if (self.phase == Phase.REST or
                self.phase in (Phase.PERCH, Phase.VERIFY) and self._contact(inp)):
            self._go(Phase.DEPART, inp.t, out, "evaluation reset from plate")
            return
        self.returning = False
        self.result, self.anchor, self.attempt = None, None, 1
        self._reset_perch()
        self.q2_result = None
        self.last_thr = self.sc.hover_thrust
        self._go(Phase.WAIT, inp.t, out, "evaluation return to standby")
        out.hold_cmd = 0.0
        out.commands.append((CMD_USER_1_PERCH_MODE, 0.0, 0.0))

    # -- helpers
    def _go(self, phase, t, out, why):
        out.events.append(f"{self.phase.value}->{phase.value}: {why}")
        self.phase, self.t_enter = phase, t
        self.t_cmd, self.n_tries, self.flag = -1e9, 0, {}
        out.phase = phase

    def _pos(self, out, p, v=None, yaw=float("nan")):
        out.mode, out.pos_enu, out.yaw_enu = "position", np.asarray(p, dtype=float), yaw
        out.vel_enu = None if v is None else np.asarray(v, dtype=float)

    def _arm_offboard(self, inp, out, force=False):
        """Request offboard + arm every arm_retry_s (setpoints must already be streaming)."""
        if inp.t - self.t_cmd >= self.cfg.arm_retry_s:
            self.t_cmd = inp.t
            self.n_tries += 1
            out.commands.append((CMD_DO_SET_MODE, 1.0, 6.0))
            out.commands.append((CMD_ARM_DISARM, 1.0, float(FORCE_MAGIC) if force else 0.0))

    def _reset_perch(self):
        self.k = 0
        self.prev_act = hover_action(self.sc.tw_ratio)
        self.hist = [self.prev_act.copy() for _ in range(self.hist_len)]
        self.cmd_fifo = deque(self.prev_act.copy() for _ in range(self.cfg.extra_delay_steps))

    def _approach_point(self, plate):
        u, w, n = plate_axes(plate.R)
        f = _horiz_unit(-n)                     # vehicle forward = direction the drone faces
        tgt = plate.c - self.cfg.approach_back * f + np.array([0.0, 0.0, self.cfg.approach_up])
        return tgt, math.atan2(f[1], f[0])

    def q2_channels(self):
        """(thrust_cmd01, phase_id, probe_level) for the C5 window at this instant."""
        tw_g = self.sc.tw_ratio * G
        if self.phase == Phase.VERIFY and self.verifier is not None and not self.verifier.done:
            ph, lv = self.verifier.state
            return self.verifier.thrust / tw_g, float(ph), float(lv)
        if self.phase == Phase.REST:
            return 0.0, 3.0, 0.0
        return self.last_acc / tw_g, 0.0, 1.0

    def _verify_q2(self, inp, out):
        v = self.verifier
        thrust, decision = v.step(inp.t, inp.contact, inp.q2_window)
        self._rates_out(out, np.zeros(3), self.sc.acc_to_u(thrust))
        if decision is None:
            return
        self.q2_result = {**v.result, "points": list(v.decisions)}
        if decision == "OFF" and inp.contact is False:
            # review D12: never cut the motors without a current contact report
            decision = "SELF_ABORT"
            self.q2_result.update(decision=decision, why="contact lost before MOTORS_OFF")
        if decision == "OFF":
            self._go(Phase.REST, inp.t, out, f"Q2 {v.result['why']}: MOTORS_OFF")
            return
        # ABORT (controlled) / SELF_ABORT: release the pad and leave with the push-off depart
        self._go(Phase.DEPART, inp.t, out, f"Q2 {decision}: {v.result['why']}")
        out.hold_cmd = 0.0
        self.flag["stage"], self.flag["t_stage"] = ("push" if decision == "ABORT" else "level"), inp.t

    def _rates_out(self, out, rates_flu, thrust_u):
        out.mode, out.rates_frd = "rates", flu_to_frd(np.asarray(rates_flu, dtype=float))
        out.thrust_norm = float(np.clip(thrust_u, 0.0, 1.0))
        out.thrust_body = np.array([0.0, 0.0, -out.thrust_norm])

    def _depart_pushoff(self, inp, out, el, plate_ok):
        c, d, f = self.cfg, inp.drone, self.flag
        stage = f.setdefault("stage", "arm")
        now = inp.t
        if stage == "arm":
            self._rates_out(out, np.zeros(3), c.pushoff_idle_u)
            if inp.armed and inp.offboard and inp.rates_mode is not False:
                f["stage"], f["t_stage"] = "spool", now
            else:
                self._arm_offboard(inp, out, force=c.standby_offset is None and self.n_tries >= c.force_arm_after_tries)
        elif stage == "spool":
            self._rates_out(out, np.zeros(3), c.pushoff_idle_u)
            if not inp.armed:
                f["stage"] = "arm"
            elif now - f["t_stage"] >= 1.0:                     # PX4 default spool-up (COM_SPOOLUP_TIME)
                out.commands.append((CMD_USER_1_PERCH_MODE, 0.0, 0.0))
                out.hold_cmd = 0.0
                f["stage"], f["t_stage"] = "push", now
                out.events.append("depart: released, push-off along body z")
        elif stage == "push":
            self._rates_out(out, np.zeros(3), self.sc.acc_to_u(c.pushoff_g * G))
            if now - f["t_stage"] >= c.pushoff_s:
                f["stage"], f["t_stage"] = "level", now
        elif stage == "level":
            rates, tb, thr, _, _ = action_to_px4(level_hold_action(d, self.sc), self.sc)
            out.mode, out.rates_frd, out.thrust_body, out.thrust_norm = "rates", rates, tb, thr
            if now - f["t_stage"] >= c.pushoff_level_s:
                n = plate_axes(inp.plate.R)[2] if plate_ok else np.array([0.0, 0.0, 1.0])
                self.anchor = d.p + c.depart_dist * _horiz_unit(n) + np.array([0.0, 0.0, 1.0])
                f["stage"] = "climb"
                out.events.append("depart: levelled, position control")
        if f["stage"] == "climb":
            self._pos(out, self.anchor)
            dist = np.linalg.norm(d.p - inp.plate.c) if plate_ok else float("nan")
            if np.linalg.norm(d.p - self.anchor) < 0.5 and (not plate_ok or dist >= c.depart_dist):
                self.result = "departed"
                self._go(Phase.DONE, inp.t, out, f"departed, {dist:.2f} m from plate")
        if el > c.depart_timeout_s and self.phase == Phase.DEPART:
            self.result = "depart_timeout"
            self._go(Phase.DONE, inp.t, out, "depart timeout")

    def _contact(self, inp):
        if inp.contact is not None:
            return bool(inp.contact)
        if inp.plate is None or not inp.plate.valid:
            return False
        u, w, n = plate_axes(inp.plate.R)
        rel = inp.drone.p - self.pad_offset * inp.drone.R[:, 2] - inp.plate.c
        d = float(rel @ n)       # review D10: two-sided (behind the plate is not contact)
        return -0.05 <= d <= self.cfg.contact_d and max(abs(rel @ u), abs(rel @ w)) < self.cfg.plate_half

    # -- main step
    def step(self, inp: Inputs) -> Output:
        c = self.cfg
        out = Output(phase=self.phase)
        if self.t_enter is None:
            self.t_enter = inp.t
        if inp.drone is None:
            self.ready = False
            return out                                   # nothing to do without EKF state
        if self.returning:
            self._return_to_standby(inp, out)
        el = inp.t - self.t_enter
        d = inp.drone
        plate_ok = inp.plate is not None and inp.plate.valid

        if self.phase == Phase.WAIT:
            if c.standby_offset is None:
                self._pos(out, d.p)
                if plate_ok:
                    self._go(Phase.APPROACH, inp.t, out, "plate estimate valid")
            else:
                if self.standby is None:
                    self.standby = d.p + np.asarray(c.standby_offset)
                    self.t_enter = inp.t
                    el = 0.0
                self._pos(out, self.standby, yaw=0.0)
                self.ready = (inp.armed and inp.offboard and
                              np.linalg.norm(d.p - self.standby) < 0.30 and np.linalg.norm(d.v) < 0.35)
                if (not inp.armed or not inp.offboard) and el >= c.arm_delay_s:
                    self._arm_offboard(inp, out)

        elif self.phase == Phase.APPROACH:
            if not plate_ok:
                self._pos(out, self.anchor if self.anchor is not None else d.p)
            else:
                tgt, yaw = self._approach_point(inp.plate)
                self._pos(out, tgt, inp.plate.v, yaw)
                self.anchor = tgt
                err = np.linalg.norm(d.p - tgt)
                verr = np.linalg.norm(d.v - inp.plate.v)
                yerr = abs(wrap_pi(math.atan2(d.R[1, 0], d.R[0, 0]) - yaw))
                ok = (inp.armed and inp.offboard and err < c.approach_tol and verr < c.approach_vtol
                      and yerr < c.approach_yaw_tol and inp.plate.visible_for >= c.tag_stable_s)
                if not ok:
                    self.flag.pop("ok_since", None)
                elif inp.t - self.flag.setdefault("ok_since", inp.t) >= c.approach_settle_s:
                    self._reset_perch()
                    self._go(Phase.PERCH, inp.t, out,
                             f"at approach point err={err:.2f} verr={verr:.2f} yerr={yerr:.2f}")
                    self.flag["switching"] = True    # wait for PX4 body-rate control before the policy runs
            if (not inp.armed or not inp.offboard) and el >= c.arm_delay_s:
                self._arm_offboard(inp, out)
            if el > c.approach_timeout_s and self.phase == Phase.APPROACH:
                self._go(Phase.ABORT, inp.t, out, "approach timeout")

        elif self.phase == Phase.PERCH:
            why = None
            if not plate_ok or inp.plate.age > c.perch_lost_s:
                why = "plate estimate lost"
            elif self.k >= c.max_steps:
                why = "perch timeout"
            elif d.R[2, 2] < -0.3:
                why = "upside down"
            elif d.p[2] < inp.plate.c[2] - c.min_below_plate:
                why = "too low"
            elif not inp.armed:
                why = "disarmed in flight"
            if not why and self.flag.get("switching"):
                if inp.rates_mode is None or inp.rates_mode:
                    self.flag["switching"] = False
                    self.flag["switch_s"] = el
                    out.events.append(f"PX4 body-rate control after {el:.3f} s")
                elif el > c.switch_timeout_s:
                    why = "PX4 did not switch to body-rate control"
            if why:
                self._go(Phase.ABORT, inp.t, out, why)
            elif self.flag.get("switching"):
                # PX4 still runs its position controller: keep the approach setpoint for it and stream a
                # level-hold rate command that takes over at the switch. The policy clock k does not run.
                a = level_hold_action(d, self.sc)
                rates, tb, thr, _, _ = action_to_px4(a, self.sc)
                out.mode, out.rates_frd, out.thrust_body, out.thrust_norm = "rates", rates, tb, thr
                out.pos_enu, out.yaw_enu = self._approach_point(inp.plate)
                out.vel_enu = inp.plate.v.copy()
            elif self.k > 0 and self._contact(inp):
                self._go(Phase.VERIFY, inp.t, out, f"contact after {self.k} steps")
                if self.verifier is not None:
                    self.verifier.reset(inp.t, self.last_acc)
                    self.q2_result = None
            else:
                obs = build_obs(d, inp.plate, self.prev_act, self.k / c.max_steps, self.pad_offset,
                                self.hist if self.hist_len else None, tag_inputs=self.tag_inputs)
                t0 = time.perf_counter()
                a = np.clip(self.policy(obs), -1.0, 1.0)
                out.infer_s = time.perf_counter() - t0
                sent = a
                if c.extra_delay_steps > 0:
                    self.cmd_fifo.append(a)
                    sent = self.cmd_fifo.popleft()
                if c.yaw_cmd_scale != 1.0:
                    sent = sent.copy()
                    sent[3] *= c.yaw_cmd_scale
                rates, tb, thr, _, _ = action_to_px4(sent, self.sc)
                out.mode, out.rates_frd, out.thrust_body, out.thrust_norm = "rates", rates, tb, thr
                out.action, out.obs = a, obs
                if self.hist_len:
                    self.hist = [self.prev_act.copy()] + self.hist[:-1]
                self.prev_act = a
                self.last_thr = thr
                self.last_acc = (float(np.clip(sent[0], -1, 1)) + 1.0) * 0.5 * self.sc.tw_ratio * G
                self.k += 1

        el = inp.t - self.t_enter                        # re-evaluate: the phase may have changed above
        if self.phase == Phase.VERIFY and self.verifier is not None:
            self._verify_q2(inp, out)

        elif self.phase == Phase.VERIFY:
            # placeholder for Q2: keep the last thrust, zero rates, log; decide after verify_s
            out.mode, out.rates_frd = "rates", np.zeros(3)
            out.thrust_body, out.thrust_norm = np.array([0.0, 0.0, -self.last_thr]), self.last_thr
            if el >= c.verify_s:
                if self._contact(inp):
                    self._go(Phase.REST, inp.t, out, "verify placeholder: contact held")
                else:
                    self._go(Phase.ABORT, inp.t, out, "verify placeholder: contact lost")

        elif self.phase == Phase.REST and inp.armed and inp.contact is False and self.verifier is not None:
            # review D12: contact lost before the motors are off -> no disarm, leave like a SELF_ABORT
            self._go(Phase.DEPART, inp.t, out, "REST: contact lost while armed")
            self.q2_result = {**(self.q2_result or {}), "decision": "SELF_ABORT",
                              "why": "contact lost after MOTORS_OFF was commanded, before disarm"}
            out.hold_cmd = 0.0
            self.flag["stage"], self.flag["t_stage"] = "level", inp.t

        elif self.phase == Phase.REST:
            self._pos(out, d.p)                          # keep offboard alive + a sane setpoint for re-arm
            if not self.flag.get("sent"):
                out.commands += [(CMD_USER_1_PERCH_MODE, 1.0, 0.0), (CMD_ARM_DISARM, 0.0, 0.0)]
                self.flag["sent"] = True
            elif (inp.armed and el >= c.force_disarm_after_s and inp.contact is not False
                  and inp.t - self.flag.get("forced_t", -1e9) >= c.arm_retry_s):
                # review D12: a forced disarm only while the pad still reports contact
                # repeated: one best-effort vehicle_command was lost in SITL and PX4 stayed armed for 40 s
                out.commands.append((CMD_ARM_DISARM, 0.0, float(FORCE_MAGIC)))
                if not self.flag.get("forced"):
                    out.events.append("plain disarm rejected -> forced disarm")
                self.flag["forced"], self.flag["forced_t"] = True, inp.t
            if c.standby_offset is None and el >= c.rest_s:
                self._go(Phase.DEPART, inp.t, out, "rest time over")

        elif self.phase == Phase.DEPART and (c.depart_pushoff or "stage" in self.flag):
            self._depart_pushoff(inp, out, el, plate_ok)

        elif self.phase == Phase.DEPART:
            if not self.flag.get("armed_once"):
                self._pos(out, d.p)
                if inp.armed and inp.offboard:
                    since = self.flag.setdefault("armed_since", inp.t)
                    # Keep the pad latched while a newly armed PX4 is in its default spool-up.
                    if not self.flag.get("wait_for_spool") or inp.t - since >= 1.0:
                        self.flag["armed_once"] = True
                        out.commands.append((CMD_USER_1_PERCH_MODE, 0.0, 0.0))
                        out.hold_cmd = 0.0
                        n = plate_axes(inp.plate.R)[2] if plate_ok else np.array([0.0, 0.0, 1.0])
                        self.anchor = d.p + c.depart_dist * n
                else:
                    if c.standby_offset is not None and not inp.armed:
                        self.flag["wait_for_spool"] = True
                    self.flag.pop("armed_since", None)
                    self._arm_offboard(inp, out, force=c.standby_offset is None and
                                       self.n_tries >= c.force_arm_after_tries)
            else:
                self._pos(out, self.anchor)
                dist = np.linalg.norm(d.p - inp.plate.c) if plate_ok else float("nan")
                if np.linalg.norm(d.p - self.anchor) < 0.5 and (not plate_ok or dist >= c.depart_dist):
                    self.result = "departed"
                    self._go(Phase.DONE, inp.t, out, f"departed, {dist:.2f} m from plate")
            if el > c.depart_timeout_s and self.phase == Phase.DEPART:
                self.result = "depart_timeout"
                self._go(Phase.DONE, inp.t, out, "depart timeout")

        elif self.phase == Phase.ABORT:
            if not self.flag.get("init"):
                self.flag["init"] = True
                n = plate_axes(inp.plate.R)[2] if plate_ok else np.array([0.0, 0.0, 1.0])
                self.anchor = d.p + c.abort_back * _horiz_unit(n) + np.array([0.0, 0.0, c.abort_up])
                out.hold_cmd = 0.0
            self._pos(out, self.anchor)
            if c.standby_offset is None and not inp.armed and el >= c.arm_delay_s:
                self._arm_offboard(inp, out, force=self.n_tries >= c.force_arm_after_tries)
            if c.standby_offset is None and el >= c.abort_hold_s:
                if self.attempt < c.max_attempts:
                    self.attempt += 1
                    self._go(Phase.APPROACH, inp.t, out, f"retry, attempt {self.attempt}")
                else:
                    self.result = "aborted"
                    self._go(Phase.DONE, inp.t, out, "max attempts reached")

        elif self.phase == Phase.DONE:
            if self.anchor is None:
                self.anchor = d.p.copy()
            self._pos(out, self.anchor)

        out.phase = self.phase
        return out


def default_px4_dir():
    for d in (os.environ.get("PX4_DIR"), "/home/sup/data/perchpilot/PX4-perch", "/home/sup/PX4-Autopilot"):
        if d and Path(d, "build/px4_sitl_default/bin/px4-param").exists():
            return d
    return None


def read_px4_param(name, instance=7, px4_dir=None, timeout=5.0):
    """Read one PX4 parameter through the SITL shell client (`px4-param --instance N show NAME`)."""
    import re
    import subprocess
    px4_dir = px4_dir or default_px4_dir()
    if px4_dir is None:
        return None
    exe = Path(px4_dir, "build/px4_sitl_default/bin/px4-param")
    try:
        out = subprocess.run([str(exe), "--instance", str(instance), "show", name], capture_output=True,
                             text=True, timeout=timeout).stdout
    except Exception:  # noqa: BLE001
        return None
    m = re.search(name + r".*?:\s*([-0-9.eE]+)", out)
    return float(m.group(1)) if m else None


def set_px4_param(name, value, instance=7, px4_dir=None, timeout=5.0, tries=10, retry_s=1.0):
    """Set one PX4 parameter through the SITL shell client and read it back. Returns True on success."""
    import subprocess
    px4_dir = px4_dir or default_px4_dir()
    if px4_dir is None:
        return False
    exe = Path(px4_dir, "build/px4_sitl_default/bin/px4-param")
    for _ in range(tries):
        try:
            subprocess.run([str(exe), "--instance", str(instance), "set", name, str(value)], capture_output=True,
                           text=True, timeout=timeout)
        except Exception:  # noqa: BLE001
            pass
        got = read_px4_param(name, instance, px4_dir, timeout)
        if got is not None and abs(got - float(value)) < 1e-6:
            return True
        time.sleep(retry_s)
    return False
