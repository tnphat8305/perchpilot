"""Camera geometry and constant-velocity Kalman filter for plate measurements."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from perch_core import PlateState, TAG_LOSS_MAX, orthonormalize, rot_y

TAG_TOPIC = "/perch/tag_pose"
TAG_FRAME = "camera"
TAG_PERIOD = 1.0 / 30.0
R_BODY_CAMERA = rot_y(math.radians(15.0))     # FLU camera at body origin, +x pitched down


def plate_in_camera(p, R, c, R_plate):
    """Transform the plate pose from ENU to the body-fixed FLU camera frame."""
    R_camera_world = (R @ R_BODY_CAMERA).T
    return R_camera_world @ (c - p), R_camera_world @ R_plate


def tag_pos_std(distance):
    return 0.01 + 0.01 * distance             # EnvCfgV2, in metres


def tag_measurement(t, drone, c_camera, R_plate_camera):
    """Compose the camera measurement with EKF2 odometry in local ENU."""
    R_world_camera = drone.R @ R_BODY_CAMERA
    return Meas(t, drone.p + R_world_camera @ c_camera, R_world_camera @ R_plate_camera,
                tag_pos_std(float(np.linalg.norm(c_camera))))


@dataclass
class Meas:
    t: float
    c: np.ndarray         # plate surface centre, local ENU
    R: np.ndarray         # plate rotation, local ENU
    pos_std: float


class PlateKF:
    """Constant-velocity Kalman filter on the plate centre (local ENU) + smoothed attitude.

    State x = [c(3), v(3)]; white-acceleration process noise q_acc [m^2/s^4 per s]; attitude is
    filtered with an exponential moving average on the rotation matrix (plate/cart rotate slowly).
    """

    def __init__(self, q_acc=0.5, att_alpha=0.3, lost_s=2.0, streak_gap_s=0.2, init_vel_std=3.0):
        self.q_acc, self.att_alpha, self.lost_s, self.gap = q_acc, att_alpha, lost_s, streak_gap_s
        self.init_vel_std = init_vel_std
        self.x = None
        self.P = None
        self.R = None
        self.t = None
        self.t_meas = None
        self.t_streak = None
        # True: position-only updates, velocity kept (set by the node during PERCH: the EKF drifts in aggressive
        # flight and the composed plate measurement then shows a spurious plate velocity of ~0.2 m/s in SITL)
        self.freeze_v = False

    def _predict_to(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        q = self.q_acc
        Q = np.zeros((6, 6))
        Q[:3, :3] = q * dt ** 3 / 3 * np.eye(3)
        Q[:3, 3:] = Q[3:, :3] = q * dt ** 2 / 2 * np.eye(3)
        Q[3:, 3:] = q * dt * np.eye(3)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, m: Meas):
        if self.x is None:
            self.x = np.concatenate([m.c, np.zeros(3)])
            self.P = np.diag([m.pos_std ** 2] * 3 + [self.init_vel_std ** 2] * 3)
            self.R, self.t = m.R.copy(), m.t
        else:
            self._predict_to(m.t)
            H = np.hstack([np.eye(3), np.zeros((3, 3))])
            S = H @ self.P @ H.T + m.pos_std ** 2 * np.eye(3)
            K = self.P @ H.T @ np.linalg.inv(S)
            if self.freeze_v:
                K[3:, :] = 0.0
            self.x = self.x + K @ (m.c - H @ self.x)
            IKH = np.eye(6) - K @ H                     # Joseph form: valid for the frozen-velocity gain too
            self.P = IKH @ self.P @ IKH.T + m.pos_std ** 2 * K @ K.T
            self.R = orthonormalize((1 - self.att_alpha) * self.R + self.att_alpha * m.R)
        if self.t_meas is None or m.t - self.t_meas > self.gap:
            self.t_streak = m.t
        self.t_meas = m.t

    def estimate(self, t) -> Optional[PlateState]:
        """Plate state predicted to time t (does not modify the filter)."""
        if self.x is None:
            return None
        dt = max(0.0, t - self.t)
        c = self.x[:3] + dt * self.x[3:]
        age = t - self.t_meas
        vis = (self.t_meas - self.t_streak) if age <= self.gap else 0.0
        # A detection is current for one camera period. Silence after that implies visual loss.
        loss = min(TAG_LOSS_MAX, max(0.0, age - TAG_PERIOD))
        rho = math.exp(-2.0 * loss)
        return PlateState(c=c, R=self.R.copy(), v=self.x[3:].copy(), valid=age <= self.lost_s, age=age,
                          visible_for=vis, reliability=rho, visible=loss == 0.0, loss_s=loss)
