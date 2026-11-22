"""Emulate an AprilTag camera from Gazebo model poses and publish /perch/tag_pose."""
from __future__ import annotations

import argparse
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from perch_core import (DroneState, SDF_PLATE_HALF_THICK, SDF_PLATE_OFFSET, SDF_PLATE_PITCH,
                        plate_from_cart, quat_to_rot, rot_to_quat, rot_x, rot_y, rot_z)
from plate_perception import Meas, TAG_FRAME, TAG_PERIOD, TAG_TOPIC, plate_in_camera, tag_pos_std

HFOV = math.radians(82.96)
VFOV = math.radians(52.90)
MAX_RANGE = 6.0
MIN_VIEW_COS = math.cos(math.radians(70.0))
NORMAL_SIGMA = math.radians(2.0)
# The merged x500_base model offsets base_link (and its IMU) from the spawned model frame.
X500_BASE_LINK_IN_MODEL = np.array([0.0, 0.0, 0.24])


def tag_visible(c_camera, n_camera):
    """Match QuadPerchEnvV2._visibility for camera-frame centre and front normal."""
    distance = max(float(np.linalg.norm(c_camera)), 1e-6)
    x, y, z = c_camera
    visible = (x > 0.0 and math.atan2(abs(y), x) < HFOV / 2
               and math.atan2(abs(z), x) < VFOV / 2 and distance < MAX_RANGE
               and float(-c_camera @ n_camera) / distance > MIN_VIEW_COS)
    return visible, distance


def noisy_tag_pose(c_camera, R_plate_camera, distance, rng):
    """Apply v2's range-dependent position noise and Gaussian noise on the normal."""
    c = c_camera + rng.normal(0.0, tag_pos_std(distance), 3)
    n = R_plate_camera[:, 2] + rng.normal(0.0, NORMAL_SIGMA, 3)
    n /= np.linalg.norm(n)
    u = R_plate_camera[:, 1]
    u = u - (u @ n) * n
    u /= np.linalg.norm(u)
    return c, np.column_stack((np.cross(u, n), u, n))


@dataclass
class PerceptionCfg:
    mode: str = "world"                  # "world" | "relative"
    rate_hz: float = 30.0
    pos_noise: float = 0.02              # [m] std at 1 m (grows linearly with range, see pos_noise_per_m)
    pos_noise_per_m: float = 0.01        # [m/m] extra std per metre of range
    ang_noise_deg: float = 1.0           # [deg] std of a random small rotation on the plate attitude
    fov_h_deg: float = 90.0
    fov_v_deg: float = 60.0
    cam_rpy_deg: tuple = (0.0, 0.0, 0.0)  # camera optical axis = body +x rotated by these FLU angles
    min_range: float = 0.15
    max_range: float = 12.0
    require_facing: bool = True          # tag on the plate face: only visible from the front side
    dropout: float = 0.0                 # probability of dropping a visible frame
    origin_enu: tuple = (0.0, 0.0, 0.0)  # Gazebo ENU position of the EKF local origin (mode "world")
    plate_offset: tuple = SDF_PLATE_OFFSET
    plate_pitch: float = SDF_PLATE_PITCH
    face_offset: float = SDF_PLATE_HALF_THICK
    seed: int = 0


def cam_rotation(cfg: PerceptionCfg):
    """Camera frame (x = optical axis, y left, z up) -> FLU body."""
    r, p, y = (math.radians(a) for a in cfg.cam_rpy_deg)
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def in_fov(cfg: PerceptionCfg, cam_p, R_cam_world, plate_c, plate_n):
    """True if the plate centre is inside the camera frustum and the plate faces the camera."""
    rel_w = np.asarray(plate_c) - np.asarray(cam_p)
    rng = float(np.linalg.norm(rel_w))
    if not (cfg.min_range <= rng <= cfg.max_range):
        return False, rng
    rel = R_cam_world.T @ rel_w
    if rel[0] <= 1e-6:
        return False, rng
    az = math.degrees(math.atan2(rel[1], rel[0]))
    el = math.degrees(math.atan2(rel[2], rel[0]))
    if abs(az) > cfg.fov_h_deg / 2 or abs(el) > cfg.fov_v_deg / 2:
        return False, rng
    if cfg.require_facing and float(np.dot(-rel_w, plate_n)) <= 0.0:
        return False, rng
    return True, rng


def small_rotation(rng, std_rad):
    v = rng.normal(0.0, std_rad, 3)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


class PerceptionEmulator:
    """Legacy offline emulator for the v1 tests using synthetic cart and drone poses."""

    def __init__(self, cfg: PerceptionCfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t_next = -1e9
        self.R_cam_body = cam_rotation(cfg)
        self.n_meas = 0
        self.n_gated = 0

    def measure(self, t, p_cart_w, R_cart_w, drone_ekf: DroneState, drone_true: Optional[DroneState] = None):
        """Returns a Meas or None (not yet time for a frame / outside FOV / dropped).

        p_cart_w, R_cart_w: cart model pose in Gazebo ENU world.
        drone_ekf: EKF2 state (local ENU). drone_true: TRUE drone pose in Gazebo world, only used in mode
        "relative" to synthesise the camera-relative measurement.
        """
        cfg = self.cfg
        period = 1.0 / cfg.rate_hz
        if t < self.t_next - 1e-9:
            return None
        # fixed-rate frame schedule (a 100 Hz caller then gets 30 frames/s on average, not 25)
        self.t_next = self.t_next + period if t - self.t_next < period else t + period
        c_w, Rp_w = plate_from_cart(p_cart_w, R_cart_w, cfg.plate_offset, cfg.plate_pitch, cfg.face_offset)
        origin = np.asarray(cfg.origin_enu, dtype=float)
        if cfg.mode == "world":
            c_true, R_true = c_w - origin, Rp_w            # plate in local ENU (EKF origin)
            # FOV gating with the EKF pose (no drone truth anywhere in this mode)
            ok, rng_m = in_fov(cfg, drone_ekf.p, drone_ekf.R @ self.R_cam_body, c_true, R_true[:, 2])
        elif cfg.mode == "relative":
            if drone_true is None:
                return None
            # plate in the TRUE body frame (what the camera sees), composed with the EKF pose
            rel_b = drone_true.R.T @ (c_w - drone_true.p)
            c_true, R_true = drone_ekf.p + drone_ekf.R @ rel_b, drone_ekf.R @ (drone_true.R.T @ Rp_w)
            ok, rng_m = in_fov(cfg, drone_true.p, drone_true.R @ self.R_cam_body, c_w, Rp_w[:, 2])
        else:
            raise ValueError(cfg.mode)
        if not ok or (cfg.dropout > 0 and self.rng.random() < cfg.dropout):
            self.n_gated += 1
            return None
        std = cfg.pos_noise + cfg.pos_noise_per_m * max(0.0, rng_m - 1.0)
        c = c_true + self.rng.normal(0.0, std, 3)
        R = small_rotation(self.rng, math.radians(cfg.ang_noise_deg)) @ R_true
        self.n_meas += 1
        return Meas(t=t, c=c, R=R, pos_std=std)


def parse_pose_v(msg, names):
    """Extract model positions and rotations from a Gazebo Pose_V frame."""
    out = {}
    for p in msg.pose:
        if p.name in names and p.name not in out:
            pos = np.array([p.position.x, p.position.y, p.position.z])
            q = [p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
            if q == [0.0, 0.0, 0.0, 0.0]:
                q = [1.0, 0.0, 0.0, 0.0]  # Protobuf default for an omitted orientation
            out[p.name] = (pos, quat_to_rot(q))
    return out


class GzPoseSource:
    def __init__(self, args):
        os.environ["GZ_PARTITION"] = args.gz_partition
        from gz.transport13 import Node
        from gz.msgs10.pose_v_pb2 import Pose_V

        self.names = {args.drone_model, args.cart_model}
        self.lock = threading.Lock()
        self.latest = None
        self.node = Node()
        if not self.node.subscribe(Pose_V, args.gz_pose_topic, self._receive):
            raise RuntimeError(f"gz subscribe failed: {args.gz_pose_topic}")

    def _receive(self, msg):
        poses = parse_pose_v(msg, self.names)
        if self.names == poses.keys():  # Use cart and drone from the same Gazebo frame.
            with self.lock:
                self.latest = time.monotonic(), poses

    def take(self):
        with self.lock:
            sample, self.latest = self.latest, None
        return sample


def simulated_tag_measurements(take_poses, drone_model, cart_model, seed):
    """Adapt a Gazebo pose source to a stream of noisy camera-relative measurements."""
    rng = np.random.default_rng(seed)

    def measure():
        sample = take_poses()
        if sample is None or time.monotonic() - sample[0] > 0.1:
            return None
        poses = sample[1]
        p_model, R = poses[drone_model]
        p = p_model + R @ X500_BASE_LINK_IN_MODEL
        c_world, R_plate_world = plate_from_cart(*poses[cart_model])
        c, R_plate = plate_in_camera(p, R, c_world, R_plate_world)
        visible, distance = tag_visible(c, R_plate[:, 2])
        # third element: CLOCK_MONOTONIC time of the pose sample (= camera exposure time)
        return (*noisy_tag_pose(c, R_plate, distance, rng), sample[0]) if visible else None

    return measure


def start_tag_camera(node, measure):
    import rclpy.duration
    from geometry_msgs.msg import PoseStamped

    pub = node.create_publisher(PoseStamped, TAG_TOPIC, 1)

    def capture():
        sample = measure()
        if sample is None:
            return
        c, R_plate, t_sample = sample
        msg = PoseStamped()
        # stamp the exposure (pose sample) time, not the publish time: the policy node composes the
        # measurement with the EKF2 state at the stamp
        age_ns = int(max(0.0, time.monotonic() - t_sample) * 1e9)
        msg.header.frame_id = TAG_FRAME
        msg.header.stamp = (node.get_clock().now() - rclpy.duration.Duration(nanoseconds=age_ns)).to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, c)
        q = msg.pose.orientation
        q.w, q.x, q.y, q.z = map(float, rot_to_quat(R_plate))
        pub.publish(msg)

    return node.create_timer(TAG_PERIOD, capture)


def main(argv=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node

    parser = argparse.ArgumentParser(description=__doc__)
    model = os.environ.get("PX4_SIM_MODEL", "gz_x500_perch").removeprefix("gz_")
    parser.add_argument("--drone-model", default=f"{model}_7")
    parser.add_argument("--cart-model", default="perch_cart")
    parser.add_argument("--gz-partition", default=os.environ.get("GZ_PARTITION", "perchpilot"))
    parser.add_argument("--gz-pose-topic", default="/world/perch_world/dynamic_pose/info")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rclpy.init(args=[])
    node = Node("tag_camera_sim")
    source = GzPoseSource(args)
    start_tag_camera(node, simulated_tag_measurements(source.take, args.drone_model, args.cart_model, args.seed))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
