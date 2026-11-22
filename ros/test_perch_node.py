"""Offline tests for the perch node (no PX4 / Gazebo / ROS graph needed).

Run with the occwm venv (needs torch for the sim comparison):
    /home/sup/data/venvs/occwm/bin/python -m pytest -q perchpilot/ros/test_perch_node.py
"""
import ast
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from perch_core import (ActionScale, DroneState, Inputs, Phase, PhaseCfg, PhaseMachine, PlateState,  # noqa: E402
                        PolicyRunner, RollingStats, CMD_ARM_DISARM, CMD_DO_SET_MODE, CMD_USER_1_PERCH_MODE,
                        FORCE_MAGIC, R_enu_flu_from_px4_q, action_to_px4, build_obs, check_policy_meta,
                        drone_state_from_odometry, enu_to_ned, flu_to_frd, frd_to_flu, hover_action, ned_to_enu,
                        odometry_from_drone_state, plate_axes, plate_from_cart, plate_tilt, px4_q_from_R_enu_flu,
                        px4_to_action, quat_to_rot, rot_to_quat, rot_y, rot_z, yaw_enu_to_ned, yaw_ned_to_enu)
from plate_perception import PlateKF, TAG_PERIOD, plate_in_camera, tag_measurement  # noqa: E402
from tag_camera_sim import PerceptionCfg, PerceptionEmulator, in_fov, parse_pose_v, tag_visible  # noqa: E402

torch = pytest.importorskip("torch")
RNG = np.random.default_rng(0)


def rand_quat(rng=RNG):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


# ----------------------------------------------------------------------------- frames
def test_vector_roundtrips():
    for _ in range(100):
        v = RNG.normal(size=3)
        assert np.allclose(enu_to_ned(ned_to_enu(v)), v)
        assert np.allclose(flu_to_frd(frd_to_flu(v)), v)
    assert np.allclose(ned_to_enu([1.0, 2.0, 3.0]), [2.0, 1.0, -3.0])   # North 1, East 2, Down 3
    assert np.allclose(frd_to_flu([1.0, 2.0, 3.0]), [1.0, -2.0, -3.0])


def test_quaternion_roundtrips():
    for _ in range(200):
        q = rand_quat()
        assert np.allclose(rot_to_quat(quat_to_rot(q)), q, atol=1e-9)
        R = R_enu_flu_from_px4_q(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
        assert np.allclose(px4_q_from_R_enu_flu(R), q, atol=1e-9)
    for _ in range(50):
        a = RNG.uniform(-math.pi, math.pi)
        assert abs(math.remainder(yaw_ned_to_enu(yaw_enu_to_ned(a)) - a, 2 * math.pi)) < 1e-12


def test_known_attitudes():
    # PX4 identity: level, nose North. In ENU: body x = North = (0,1,0), body z (FLU up) = (0,0,1)
    R = R_enu_flu_from_px4_q([1, 0, 0, 0])
    assert np.allclose(R[:, 0], [0, 1, 0]) and np.allclose(R[:, 2], [0, 0, 1]) and np.allclose(R[:, 1], [-1, 0, 0])
    # PX4 yaw +90 deg (nose East) == ENU yaw 0
    qe = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    assert np.allclose(R_enu_flu_from_px4_q(qe), np.eye(3), atol=1e-12)
    assert abs(yaw_ned_to_enu(math.pi / 2)) < 1e-12
    # PX4 pitch +20 deg (nose up, FRD) -> FLU body x has positive ENU z
    th = math.radians(20)
    R = R_enu_flu_from_px4_q([math.cos(th / 2), 0, math.sin(th / 2), 0])
    assert R[2, 0] > 0.3


def test_rate_conversion_consistent_with_attitude():
    """Integrating FRD rates on R_ned_frd == integrating the converted FLU rates on R_enu_flu."""
    for _ in range(50):
        q = rand_quat()
        w_frd = RNG.normal(size=3)
        dt = 1e-6
        Rn = quat_to_rot(q) @ (np.eye(3) + skew(w_frd) * dt)
        Re = R_enu_flu_from_px4_q(q) @ (np.eye(3) + skew(frd_to_flu(w_frd)) * dt)
        T = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
        F = np.diag([1.0, -1, -1])
        assert np.allclose(T @ Rn @ F, Re, atol=1e-12)


def test_drone_state_roundtrip():
    for _ in range(50):
        s = DroneState(p=RNG.normal(size=3), v=RNG.normal(size=3), R=quat_to_rot(rand_quat()), w=RNG.normal(size=3))
        s2 = drone_state_from_odometry(**odometry_from_drone_state(s))
        assert all(np.allclose(getattr(s, k), getattr(s2, k)) for k in ("p", "v", "R", "w"))


def test_body_velocity_frame():
    q = rand_quat()
    v_ned = RNG.normal(size=3)
    v_body = quat_to_rot(q).T @ v_ned
    s1 = drone_state_from_odometry([0, 0, 0], q, v_ned, [0, 0, 0], 1)
    s3 = drone_state_from_odometry([0, 0, 0], q, v_body, [0, 0, 0], 3)
    assert np.allclose(s1.v, s3.v)


# ----------------------------------------------------------------------------- obs == QuadPerchEnv.observe()
def _sim():
    from sim.quad_perch_env import EnvCfg, QuadPerchEnv, quat_mul
    return EnvCfg, QuadPerchEnv, quat_mul


def px4_fields_from_env(env, quat_mul):
    """Independent ENU/FLU -> NED/FRD conversion with quaternion algebra (not perch_core's matrices)."""
    s = 1 / math.sqrt(2)
    q_T = torch.tensor([0.0, s, s, 0.0])           # 180 deg about (1,1,0)/sqrt2: ENU -> NED
    q_F = torch.tensor([0.0, 1.0, 0.0, 0.0])       # 180 deg about x: FRD -> FLU
    q = quat_mul(quat_mul(q_T, env.q[0]), q_F).numpy().astype(float)
    p, v, w = env.p[0].numpy().astype(float), env.v[0].numpy().astype(float), env.w[0].numpy().astype(float)
    return dict(position=[p[1], p[0], -p[2]], q=q, velocity=[v[1], v[0], -v[2]],
                angular_velocity=[w[0], -w[1], -w[2]], velocity_frame=1)


def plate_from_env(env, via_cart=True):
    psi, th = float(env.veh_psi[0]), float(env.tilt[0])
    Rv = rot_z(psi)
    if via_cart:   # exercise plate_from_cart: "cart" at veh_p with z = plate height, plate offset (-0.3, 0, 0)
        p_cart = env.veh_p[0].numpy().astype(float).copy()
        p_cart[2] = float(env.ph[0])
        c, Rp = plate_from_cart(p_cart, Rv, offset=(-0.3, 0.0, 0.0), pitch=-th, face_offset=0.0)
    else:
        pc, _, _, _, _ = env.plate_frame()
        c, Rp = pc[0].numpy().astype(float), Rv @ rot_y(-th)
    v = float(env.veh_s[0]) * Rv[:, 0]
    return PlateState(c=c, R=Rp, v=v)


def test_plate_frame_matches_env():
    EnvCfg, QuadPerchEnv, _ = _sim()
    env = QuadPerchEnv(EnvCfg(num_envs=1, obs_noise=0.0), device="cpu", seed=3)
    for _ in range(20):
        env.reset_idx(torch.arange(1))
        env.veh_p[0] = torch.tensor(RNG.normal(size=3) * [5, 5, 0], dtype=torch.float32)
        pc, n, u, w, vel = env.plate_frame()
        pl = plate_from_env(env)
        uu, ww, nn = plate_axes(pl.R)
        assert np.allclose(pl.c, pc[0].numpy(), atol=1e-5)
        assert np.allclose(nn, n[0].numpy(), atol=1e-6) and np.allclose(uu, u[0].numpy(), atol=1e-6)
        assert np.allclose(ww, w[0].numpy(), atol=1e-6) and np.allclose(pl.v, vel[0].numpy(), atol=1e-6)
        assert abs(plate_tilt(pl.R) - float(env.tilt[0])) < 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_obs_matches_env_observe(seed):
    EnvCfg, QuadPerchEnv, quat_mul = _sim()
    env = QuadPerchEnv(EnvCfg(num_envs=1, obs_noise=0.0), device="cpu", seed=seed)
    rng = np.random.default_rng(100 + seed)
    for _ in range(20):
        env.reset_idx(torch.arange(1))
        # arbitrary physical state (not only reset states)
        env.p[0] += torch.tensor(rng.normal(size=3), dtype=torch.float32)
        env.v[0] = torch.tensor(rng.normal(size=3) * 2, dtype=torch.float32)
        env.q[0] = torch.tensor(rand_quat(rng), dtype=torch.float32)
        env.w[0] = torch.tensor(rng.normal(size=3) * 2, dtype=torch.float32)
        env.prev_act[0] = torch.tensor(rng.uniform(-1, 1, 4), dtype=torch.float32)
        env.t[0] = int(rng.integers(0, 600))
        env.veh_p[0] = torch.tensor(rng.normal(size=3) * [5, 5, 0], dtype=torch.float32)
        ref = env.observe()[0].numpy()
        drone = drone_state_from_odometry(**px4_fields_from_env(env, quat_mul))
        mine = build_obs(drone, plate_from_env(env), env.prev_act[0].numpy(), int(env.t[0]) / env.max_steps,
                         pad_offset=env.cfg.pad_offset)
        assert mine.shape == ref.shape == (28,)
        err = np.abs(mine - ref)
        assert err.max() < 1e-5, (err.argmax(), err.max())


class ScriptedPolicy:
    """Deterministic random actions; exposes hist_len like PolicyRunner."""

    def __init__(self, hist_len, seed=0):
        self.hist_len, self.rng = hist_len, np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.uniform(-0.9, 0.9, 4)


@pytest.mark.parametrize("act_history", [False, True])
def test_closed_loop_core_vs_env(act_history):
    """env state -> PX4 msgs -> PhaseMachine(PERCH) -> PX4 rate setpoint -> env action, over several
    episodes (incl. resets): the node's obs must equal env.observe() at every step (history, t_norm, prev_act)."""
    EnvCfg, QuadPerchEnv, quat_mul = _sim()
    kw = dict(delay_steps=(4, 6), act_history=True) if act_history else {}
    env = QuadPerchEnv(EnvCfg(num_envs=1, obs_noise=0.0, tw_ratio=(3.0, 3.0), **kw), device="cpu", seed=7)
    pol = ScriptedPolicy((env.obs_dim - 28) // 4)
    sc = ActionScale(rate_max=env.cfg.rate_max, tw_ratio=3.0, hover_thrust=0.3)   # no thrust saturation
    pm = PhaseMachine(PhaseCfg(min_below_plate=1e9, max_steps=env.max_steps), pol, sc, env.cfg.pad_offset,
                      start=Phase.PERCH)
    pm._reset_perch()
    n_done = 0
    for k in range(900):
        ref = env.observe()[0].numpy()
        inp = Inputs(t=k * 0.01, drone=drone_state_from_odometry(**px4_fields_from_env(env, quat_mul)), armed=True,
                     offboard=True, plate=plate_from_env(env, via_cart=False), contact=False)
        out = pm.step(inp)
        assert out.phase == Phase.PERCH and out.mode == "rates", out.events
        assert np.abs(out.obs - ref).max() < 1e-5, (k, np.abs(out.obs - ref).argmax())
        a = px4_to_action(out.rates_frd, out.thrust_body, sc)
        assert np.allclose(a, out.action, atol=1e-12)
        _, _, done, _ = env.step(torch.tensor(a, dtype=torch.float32).unsqueeze(0))
        if bool(done[0]):
            n_done += 1
            pm._reset_perch()
    assert n_done >= 1          # at least one reset exercised


# ----------------------------------------------------------------------------- action -> PX4
def test_action_mapping_signs():
    sc = ActionScale(rate_max=(6.0, 6.0, 3.0), tw_ratio=3.0, hover_thrust=0.6)
    r, tb, thr, acc, sat = action_to_px4([0, 1, 0, 0], sc)
    assert np.allclose(r, [6, 0, 0])                        # FLU +roll == FRD +roll (right wing down)
    r, *_ = action_to_px4([0, 0, 1, 0], sc)
    assert np.allclose(r, [0, -6, 0])                       # FLU +wy (nose down) == FRD -pitch rate
    r, *_ = action_to_px4([0, 0, 0, 1], sc)
    assert np.allclose(r, [0, 0, -3])                       # FLU +wz (yaw left / CCW) == FRD -yaw rate
    r, tb, thr, acc, sat = action_to_px4(hover_action(3.0), sc)
    assert abs(acc - 9.81) < 1e-9 and abs(thr - 0.6) < 1e-9 and np.allclose(tb, [0, 0, -0.6]) and not sat
    r, tb, thr, acc, sat = action_to_px4([-1, 0, 0, 0], sc)
    assert thr == 0.0 and tb[2] == 0.0
    r, tb, thr, acc, sat = action_to_px4([1, 0, 0, 0], sc)  # 3 g asked, x500 hover 0.6 -> saturates
    assert sat and thr == 1.0 and tb[2] == -1.0
    assert tb[2] <= 0                                       # FRD z points down: upward thrust is negative
    for _ in range(100):
        a = RNG.uniform(-1, 1, 4)
        a[0] = RNG.uniform(-1, 2 / (3.0 * 0.6) - 1)         # non-saturated range
        r, tb, *_ = action_to_px4(a, sc)
        assert np.allclose(px4_to_action(r, tb, sc), a)


def test_x500_thrust_model():
    sc = ActionScale(thrust_model="x500_gz", tw_ratio=3.0)
    assert abs(sc.acc_to_u(9.81) - 0.729) < 0.005            # physical hover point of the gz x500
    assert abs(sc.max_acc_g() - 1.69) < 0.01                 # x500 thrust-to-weight ~1.69 (policy trained 2.6-3.4)
    for _ in range(100):
        a = RNG.uniform(-1, 1, 4)
        a[0] = RNG.uniform(-0.55, 0.12)                      # non-saturated range for TW 3
        r, tb, thr, acc, sat = action_to_px4(a, sc)
        assert not sat and np.allclose(px4_to_action(r, tb, sc), a)
    perch = ActionScale(thrust_model="x500_perch_gz", tw_ratio=2.26)
    assert perch.max_acc_g() == pytest.approx(2.26)
    assert perch.acc_to_u(9.81) == pytest.approx(0.606106)


# ----------------------------------------------------------------------------- policy runner
def make_test_policy(d, obs_dim=28, seed=0):
    """Tiny random TorchScript MLP + policy.json + policy.npz in temp dir d (tests only, not committed)."""
    from export_policy import actor_from_state_dict, export, make_meta
    from rl.ppo import ActorCritic
    torch.manual_seed(seed)
    ac = ActorCritic(obs_dim, 4, priv_dim=9 if obs_dim % 4 == 3 else 0)
    ac.norm.mean.normal_()
    ac.norm.var.uniform_(0.5, 2.0)
    return export(actor_from_state_dict(ac.state_dict()), d, make_meta(obs_dim))


@pytest.mark.parametrize("obs_dim", [28, 52, 31, 55])
def test_policy_runner_torch_and_numpy(obs_dim):
    with tempfile.TemporaryDirectory() as d:
        make_test_policy(d, obs_dim)
        pt, pn = PolicyRunner(d, "torch"), PolicyRunner(d, "numpy")
        assert pt.backend == "torch" and pn.backend == "numpy" and pt.hist_len == (obs_dim - 28) // 4
        assert pt.tag_inputs == pn.tag_inputs == (obs_dim in (31, 55))
        for _ in range(20):
            o = RNG.normal(size=obs_dim).astype(np.float32) * 3
            at, an = pt(o), pn(o)
            assert at.shape == (4,) and np.all(np.abs(at) <= 1)
            assert np.allclose(at, an, atol=1e-5)
        with pytest.raises(ValueError):
            pt(np.zeros(obs_dim + 1))


def test_policy_meta_validation():
    from export_policy import make_meta
    m = make_meta(28)
    check_policy_meta(m)
    for bad in ({"frame": "world"}, {"dt": 0.02}, {"act_layout": ["wx", "wy", "wz", "thrust01"]}, {"obs_dim": 30}):
        with pytest.raises(ValueError):
            check_policy_meta({**m, **bad})


def test_inference_time_cpu():
    """Real-size net (256-256-128): single-obs CPU inference must fit the 10 ms loop with margin."""
    import time
    with tempfile.TemporaryDirectory() as d:
        make_test_policy(d, 28)
        for backend in ("torch", "numpy"):
            p = PolicyRunner(d, backend)
            st = RollingStats()
            o = np.zeros(28, dtype=np.float32)
            for _ in range(300):
                t0 = time.perf_counter()
                p(o)
                st.add(time.perf_counter() - t0)
            s = st.summary(1e3)
            print(f"\n[infer {backend}] ms {s}")
            assert s["p95"] < 3.0


# ----------------------------------------------------------------------------- perception
def test_plate_from_cart_sdf():
    c, Rp = plate_from_cart([8.0, 0.0, 0.45], np.eye(3))
    n = Rp[:, 2]
    assert np.allclose(n, [-math.sin(1.1345), 0, math.cos(1.1345)], atol=1e-9)
    assert np.allclose(c, np.array([7.2, 0, 1.3]) + 0.01 * n, atol=1e-9)
    assert abs(math.degrees(plate_tilt(Rp)) - 65.0) < 0.01
    c2, Rp2 = plate_from_cart([0, 0, 0.45], rot_z(math.pi / 2))     # cart heading North
    assert np.allclose(Rp2[:, 1], [-1, 0, 0], atol=1e-12)           # lateral axis = cart left = West
    assert Rp2[1, 2] < -0.9                                          # normal faces backwards (South)


def _hover_in_front(c, n, dist=2.0, yaw=0.0):
    return DroneState(p=c + dist * n, v=np.zeros(3), R=rot_z(yaw), w=np.zeros(3))


def test_fov_gating():
    cfg = PerceptionCfg()
    c, Rp = plate_from_cart([8.0, 0.0, 0.45], np.eye(3))
    d = _hover_in_front(c, Rp[:, 2])
    ok, _ = in_fov(cfg, d.p, d.R, c, Rp[:, 2])
    assert ok
    ok, _ = in_fov(cfg, d.p, rot_z(math.pi), c, Rp[:, 2])           # looking away
    assert not ok
    ok, _ = in_fov(cfg, d.p, rot_z(math.radians(50)), c, Rp[:, 2])  # 50 deg off-axis > 45 deg half FOV
    assert not ok
    ok, _ = in_fov(cfg, c - 2.0 * Rp[:, 2], np.eye(3) @ rot_z(math.pi), c, Rp[:, 2])   # behind the plate
    assert not ok
    narrow = PerceptionCfg(fov_h_deg=60.0)
    ok, _ = in_fov(narrow, d.p, rot_z(math.radians(35)), c, Rp[:, 2])
    assert not ok


def test_emulator_rate_and_kf_tracking():
    cfg = PerceptionCfg(pos_noise=0.02, ang_noise_deg=1.0, seed=1)
    emu, kf = PerceptionEmulator(cfg), PlateKF()
    speed = 1.5
    errs, verrs = [], []
    n_meas_first_s = 0
    for k in range(400):                                  # 4 s at 100 Hz
        t = k * 0.01
        p_cart = np.array([8.0 + speed * t, 0.0, 0.45])
        c_true, Rp = plate_from_cart(p_cart, np.eye(3))
        drone = _hover_in_front(c_true, Rp[:, 2], 3.0)
        m = emu.measure(t, p_cart, np.eye(3), drone)
        if m is not None:
            kf.update(m)
            n_meas_first_s += t < 1.0
        est = kf.estimate(t)
        if t > 2.0:
            errs.append(np.linalg.norm(est.c - c_true))
            verrs.append(np.linalg.norm(est.v - [speed, 0, 0]))
            assert est.valid and est.visible_for > 0.3
    assert n_meas_first_s == 30
    assert np.mean(errs) < 0.05 and np.mean(verrs) < 0.3, (np.mean(errs), np.mean(verrs))
    est = kf.estimate(4.0 + 3.0)                          # 3 s without measurements -> invalid
    assert not est.valid and est.reliability == pytest.approx(math.exp(-4.0))


def test_relative_mode_cancels_ekf_bias():
    """EKF position bias biases 'world' mode plate-relative geometry but not 'relative' mode."""
    p_cart = np.array([8.0, 0.0, 0.45])
    c_w, Rp = plate_from_cart(p_cart, np.eye(3))
    truth = _hover_in_front(c_w, Rp[:, 2], 2.0)
    bias = np.array([0.4, -0.3, -0.2])          # keeps the plate inside the FOV from the biased pose
    ekf = DroneState(p=truth.p + bias, v=np.zeros(3), R=truth.R, w=np.zeros(3))
    for mode, expect_err in (("world", np.linalg.norm(bias)), ("relative", 0.0)):
        emu = PerceptionEmulator(PerceptionCfg(mode=mode, pos_noise=0.0, pos_noise_per_m=0.0, ang_noise_deg=0.0))
        m = emu.measure(0.0, p_cart, np.eye(3), ekf, truth)
        rel_est = m.c - ekf.p
        rel_true = c_w - truth.p
        assert abs(np.linalg.norm(rel_est - rel_true) - expect_err) < 1e-9


def test_parse_pose_v():
    pv = pytest.importorskip("gz.msgs10.pose_v_pb2")
    msg = pv.Pose_V()
    a = msg.pose.add()
    a.name = "body"
    a.position.x = 99.0
    b = msg.pose.add()
    b.name = "perch_cart"
    b.position.x, b.position.y, b.position.z = 8.5, 0.1, 0.45
    b.orientation.w, b.orientation.z = math.cos(0.1), math.sin(0.1)
    c = msg.pose.add()
    c.name = "x500_7"                                     # default (all-zero) quaternion -> identity
    out = parse_pose_v(msg, {"perch_cart", "x500_7"})
    assert set(out) == {"perch_cart", "x500_7"}
    assert np.allclose(out["perch_cart"][0], [8.5, 0.1, 0.45])
    assert np.allclose(out["perch_cart"][1], rot_z(0.2), atol=1e-12)
    assert np.allclose(out["x500_7"][1], np.eye(3))


def test_tag_pose_camera_body_world_roundtrip():
    rng = np.random.default_rng(16)
    for _ in range(100):
        p, c, bias = rng.normal(size=(3, 3))
        R, Rp = (quat_to_rot(rand_quat(rng)) for _ in range(2))
        c_cam, Rp_cam = plate_in_camera(p, R, c, Rp)
        ekf = DroneState(p=p + bias, R=R, v=np.zeros(3), w=np.zeros(3))
        m = tag_measurement(1.0, ekf, c_cam, Rp_cam)
        np.testing.assert_allclose(m.c - ekf.p, c - p, atol=1e-12)
        np.testing.assert_allclose(m.R, Rp, atol=1e-12)
        assert m.pos_std == pytest.approx(0.01 + 0.01 * np.linalg.norm(c - p))


def test_tag_fov_matches_sim_v2():
    from sim.quad_perch_env_v2 import EnvCfgV2, QuadPerchEnvV2

    rng = np.random.default_rng(35)
    N = 1024
    env = QuadPerchEnvV2(EnvCfgV2(num_envs=N, cam_pitch_deg=(15.0, 15.0)), device="cpu")
    p = rng.normal(size=(N, 3))
    R = np.stack([quat_to_rot(rand_quat(rng)) for _ in range(N)])
    Rp = np.stack([quat_to_rot(rand_quat(rng)) for _ in range(N)])
    c = p + np.einsum("nij,nj->ni", R, rng.uniform([-2, -6, -5], [8, 6, 4], size=(N, 3)))
    expected, dist, _ = env._visibility(*(torch.from_numpy(x) for x in (p, R, c, Rp[..., 2])))
    assert 0 < expected.sum() < N
    for i in range(N):
        c_cam, Rp_cam = plate_in_camera(p[i], R[i], c[i], Rp[i])
        visible, distance = tag_visible(c_cam, Rp_cam[:, 2])
        assert visible == bool(expected[i])
        assert distance == pytest.approx(float(dist[i]))


def test_tag_reliability_and_v2_observation():
    c, R = plate_from_cart([8.0, 0.0, 0.45], np.eye(3))
    drone = _hover_in_front(c, R[:, 2])
    kf = PlateKF()
    kf.update(tag_measurement(1.0, drone, *plate_in_camera(drone.p, drone.R, c, R)))
    current = kf.estimate(1.0 + TAG_PERIOD / 2)
    assert current.visible and current.reliability == 1.0
    lost = kf.estimate(1.0 + TAG_PERIOD + 0.5)
    assert not lost.visible
    assert lost.reliability == pytest.approx(math.exp(-1.0))
    obs = build_obs(drone, lost, np.zeros(4), 0.0, act_hist=[np.ones(4)], tag_inputs=True)
    np.testing.assert_allclose(obs[-7:], [1.0] * 4 + [math.exp(-1.0), 0.0, 0.25])


# ----------------------------------------------------------------------------- phase machine
def _plate_static():
    c, Rp = plate_from_cart([8.0, 0.0, 0.45], np.eye(3))
    return PlateState(c=c, R=Rp, v=np.zeros(3), valid=True, age=0.0, visible_for=1.0)


def test_phase_machine_full_sequence():
    with tempfile.TemporaryDirectory() as d:
        make_test_policy(d, 28)
        pol = PolicyRunner(d, "numpy")
    sc = ActionScale(hover_thrust=0.6)
    pm = PhaseMachine(PhaseCfg(rest_s=2.0), pol, sc)
    plate = _plate_static()
    n = plate.R[:, 2]
    t = 0.0
    ground = DroneState(p=np.zeros(3), v=np.zeros(3), R=np.eye(3), w=np.zeros(3))

    def run(drone, armed, offboard, plate, contact=None, secs=0.01):
        nonlocal t
        outs = []
        for _ in range(max(1, int(round(secs / 0.01)))):
            outs.append(pm.step(Inputs(t=t, drone=drone, armed=armed, offboard=offboard, plate=plate, contact=contact)))
            t += 0.01
        return outs

    o = run(ground, False, False, None)[-1]
    assert o.phase == Phase.WAIT and o.mode == "position"
    o = run(ground, False, False, plate)[-1]
    assert o.phase == Phase.APPROACH
    outs = run(ground, False, False, plate, secs=1.5)
    cmds = [c for o in outs for c in o.commands]
    assert (CMD_DO_SET_MODE, 1.0, 6.0) in cmds and (CMD_ARM_DISARM, 1.0, 0.0) in cmds
    tgt = outs[-1].pos_enu
    assert np.allclose(tgt, plate.c + np.array([-4.0, 0.0, 1.0]))   # sim start box: 2.7..5.7 m behind, 0.2..1.8 up
    assert abs(outs[-1].yaw_enu) < 1e-9                    # plate normal faces -x -> fly facing +x (East)
    at_tgt = DroneState(p=tgt.copy(), v=np.zeros(3), R=np.eye(3), w=np.zeros(3))
    outs = run(at_tgt, True, True, plate, secs=0.2)
    assert outs[-1].phase == Phase.APPROACH                 # hand-over conditions must hold approach_settle_s
    # PX4 still reports position control: level-hold rates + approach setpoint, policy clock stopped
    for _ in range(15):
        o = pm.step(Inputs(t=t, drone=at_tgt, armed=True, offboard=True, plate=plate, rates_mode=False))
        t += 0.01
    assert o.phase == Phase.PERCH
    assert o.mode == "rates" and o.action is None and o.pos_enu is not None and pm.k == 0
    assert abs(o.thrust_norm - 0.6) < 1e-9 and np.allclose(o.rates_frd, 0)
    o = run(at_tgt, True, True, plate)[-1]
    assert o.mode == "rates" and o.action is not None and o.obs.shape == (28,) and o.pos_enu is None
    o = run(at_tgt, True, True, plate, contact=True)[-1]
    assert o.phase == Phase.VERIFY and o.mode == "rates" and np.allclose(o.rates_frd, 0)
    outs = run(at_tgt, True, True, plate, contact=True, secs=0.6)
    assert outs[-1].phase == Phase.REST
    cmds = [c for o in outs for c in o.commands]
    assert (CMD_USER_1_PERCH_MODE, 1.0, 0.0) in cmds and (CMD_ARM_DISARM, 0.0, 0.0) in cmds
    outs = run(at_tgt, True, False, plate, contact=True, secs=1.2)   # plain disarm rejected -> forced
    assert (CMD_ARM_DISARM, 0.0, float(FORCE_MAGIC)) in [c for o in outs for c in o.commands]
    outs = run(at_tgt, False, False, plate, contact=True, secs=1.0)
    assert outs[-1].phase == Phase.DEPART
    outs = run(at_tgt, False, False, plate, secs=3.5)                 # arm rejected twice -> forced arm
    cmds = [c for o in outs for c in o.commands]
    assert (CMD_ARM_DISARM, 1.0, 0.0) in cmds and (CMD_ARM_DISARM, 1.0, float(FORCE_MAGIC)) in cmds
    o = run(at_tgt, True, True, plate)[-1]
    assert (CMD_USER_1_PERCH_MODE, 0.0, 0.0) in o.commands and o.hold_cmd == 0.0
    o = run(at_tgt, True, True, plate)[-1]
    assert np.allclose(o.pos_enu, at_tgt.p + 2.0 * n)
    away = DroneState(p=at_tgt.p + 2.0 * n, v=np.zeros(3), R=np.eye(3), w=np.zeros(3))
    o = run(away, True, True, plate)[-1]
    assert o.phase == Phase.DONE and pm.result == "departed"


def test_phase_machine_abort_paths():
    with tempfile.TemporaryDirectory() as d:
        make_test_policy(d, 28)
        pol = PolicyRunner(d, "numpy")
    plate = _plate_static()
    drone = DroneState(p=plate.c + 2.0 * plate.R[:, 2], v=np.zeros(3), R=np.eye(3), w=np.zeros(3))
    pm = PhaseMachine(PhaseCfg(max_attempts=2), pol, ActionScale(), start=Phase.PERCH)
    pm._reset_perch()
    o = pm.step(Inputs(t=0.0, drone=drone, armed=True, offboard=True, plate=plate))
    assert o.phase == Phase.PERCH
    lost = PlateState(c=plate.c, R=plate.R, v=plate.v, valid=True, age=1.5)
    o = pm.step(Inputs(t=0.01, drone=drone, armed=True, offboard=True, plate=lost))
    assert o.phase == Phase.ABORT and o.hold_cmd == 0.0 and o.mode == "position"
    assert o.pos_enu[2] > drone.p[2] + 0.9                              # backs off and climbs
    o = pm.step(Inputs(t=3.1, drone=drone, armed=True, offboard=True, plate=plate))
    assert o.phase == Phase.APPROACH and pm.attempt == 2
    # perch timeout -> ABORT -> DONE (max attempts)
    pm.phase = Phase.PERCH
    pm._reset_perch()
    pm.k = pm.cfg.max_steps
    o = pm.step(Inputs(t=4.0, drone=drone, armed=True, offboard=True, plate=plate))
    assert o.phase == Phase.ABORT
    o = pm.step(Inputs(t=7.5, drone=drone, armed=True, offboard=True, plate=plate))
    assert o.phase == Phase.DONE and pm.result == "aborted"


def test_no_drone_state_no_output():
    pm = PhaseMachine(PhaseCfg(), ScriptedPolicy(0), ActionScale())
    o = pm.step(Inputs(t=0.0, drone=None, armed=True, offboard=True, plate=_plate_static()))
    assert o.mode == "none"          # stale EKF -> nothing published -> PX4 offboard-loss failsafe


def test_node_has_no_truth_subscriptions():
    """R1 static check for the policy's topic allowlist and Python imports."""
    from plate_perception import TAG_TOPIC

    tree = ast.parse((HERE / "perch_node.py").read_text())
    subs = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "create_subscription"]
    assert sorted(ast.unparse(node.args[1]) for node in subs) == [
        "'/perch/hold_sensors'", "'/perchpilot/cmd'", "TAG_TOPIC",
        "ns + '/fmu/out/sensor_combined'", "ns + '/fmu/out/vehicle_control_mode'",
        "ns + '/fmu/out/vehicle_odometry'", "ns + '/fmu/out/vehicle_status'"]
    assert TAG_TOPIC == "/perch/tag_pose"
    for file in ("perch_node.py", "plate_perception.py", "perch_core.py", "perch_q2.py"):
        source = (HERE / file).read_text()
        for forbidden in ("GzPoseSource", "tag_camera_sim", "gz.transport", "/world/", "/perch/hold_truth"):
            assert forbidden not in source


@pytest.mark.skipif(not os.environ.get("PERCH_RUN"), reason="set PERCH_RUN=<exported run dir> (+ PERCH_CKPT=<ckpt.pt>)")
def test_real_policy_closed_loop():
    """Opt-in: a trained policy flown through the node core (PX4 message conversions, PhaseMachine) on the
    GPU-sim dynamics (CPU, 1 env) vs. the same policy on env.observe() directly. Success rates should match."""
    EnvCfg, QuadPerchEnv, quat_mul = _sim()
    run = Path(os.environ["PERCH_RUN"])
    pol = PolicyRunner(run, "torch")
    meta = pol.meta
    kw = {}
    if meta.get("delay_steps_trained", [0, 0]) != [0, 0]:
        kw = dict(delay_steps=tuple(meta["delay_steps_trained"]), tau_rate=(0.035, 0.07), tau_yaw=(0.3, 0.6))
    if os.environ.get("PERCH_CKPT"):            # TorchScript export == eager actor
        from export_policy import actor_from_state_dict
        actor = actor_from_state_dict(torch.load(os.environ["PERCH_CKPT"], map_location="cpu"))
        o = torch.randn(64, pol.obs_dim) * 2
        ref = actor(o).detach().numpy()
        assert np.allclose(np.stack([pol(x) for x in o.numpy()]), ref, atol=1e-5)
    n_ep = int(os.environ.get("PERCH_EPISODES", "30"))
    res = {}
    for path in ("direct", "node"):
        env = QuadPerchEnv(EnvCfg(num_envs=1, **kw), device="cpu", seed=11)
        sc = ActionScale(rate_max=tuple(meta["rate_max"]), tw_ratio=meta["tw_ratio_nominal"], hover_thrust=0.3)
        pm = PhaseMachine(PhaseCfg(min_below_plate=1e9, max_steps=env.max_steps), pol, sc, env.cfg.pad_offset,
                          start=Phase.PERCH)
        pm._reset_perch()
        succ = crash = ep = k = 0
        while ep < n_ep:
            if path == "direct":
                a = pol(env.observe()[0].numpy())
            else:
                inp = Inputs(t=k * 0.01, drone=drone_state_from_odometry(**px4_fields_from_env(env, quat_mul)),
                             armed=True, offboard=True, plate=plate_from_env(env, via_cart=False), contact=False)
                out = pm.step(inp)
                assert out.phase == Phase.PERCH, out.events
                a = px4_to_action(out.rates_frd, out.thrust_body, sc)
            _, _, done, info = env.step(torch.tensor(a, dtype=torch.float32).unsqueeze(0))
            k += 1
            if bool(done[0]):
                ep += 1
                succ += int(info["success"][0])
                crash += int(info["crash"][0])
                pm._reset_perch()
        res[path] = (succ, crash, n_ep)
    print(f"\n[closed loop {run.name}] success/crash/episodes: {res}")
    assert res["node"][0] >= res["direct"][0] - max(3, n_ep // 10)


def test_depart_pushoff_sequence():
    """Eval-mode DEPART by push-off: idle rates until armed in rate mode, spool 1 s, release, push, level, position."""
    sc = ActionScale(hover_thrust=0.6, thrust_model="x500_perch_gz", tw_ratio=2.25)
    pm = PhaseMachine(PhaseCfg(standby_offset=(2, 0, 2), depart_pushoff=True), ScriptedPolicy(0), sc)
    plate = _plate_static()
    n = plate.R[:, 2]
    R_tilt = plate.R @ np.diag([1.0, 1.0, 1.0])        # body z along the plate normal
    on_plate = DroneState(p=plate.c + 0.227 * n, v=np.zeros(3), R=R_tilt, w=np.zeros(3))
    pm.phase, pm.t_enter = Phase.DEPART, 0.0
    o = pm.step(Inputs(t=0.0, drone=on_plate, armed=False, offboard=False, plate=plate, rates_mode=False))
    assert o.mode == "rates" and (CMD_ARM_DISARM, 1.0, 0.0) in o.commands and o.hold_cmd is None
    assert abs(o.thrust_norm - 0.05) < 1e-9 and np.allclose(o.rates_frd, 0)
    pm.step(Inputs(t=0.1, drone=on_plate, armed=True, offboard=True, plate=plate, rates_mode=True))
    o = pm.step(Inputs(t=0.6, drone=on_plate, armed=True, offboard=True, plate=plate, rates_mode=True))
    assert o.hold_cmd is None                                  # still spooling, pad latched
    o = pm.step(Inputs(t=1.15, drone=on_plate, armed=True, offboard=True, plate=plate, rates_mode=True))
    assert o.hold_cmd == 0.0 and (CMD_USER_1_PERCH_MODE, 0.0, 0.0) in o.commands
    o = pm.step(Inputs(t=1.2, drone=on_plate, armed=True, offboard=True, plate=plate, rates_mode=True))
    assert o.mode == "rates" and o.thrust_norm > 0.7 and np.allclose(o.rates_frd, 0)   # 1.5 g push
    pm.step(Inputs(t=1.5, drone=on_plate, armed=True, offboard=True, plate=plate, rates_mode=True))
    o = pm.step(Inputs(t=1.55, drone=on_plate, armed=True, offboard=True, plate=plate, rates_mode=True))
    assert o.mode == "rates" and abs(o.rates_frd[1]) > 0                               # levelling
    o = pm.step(Inputs(t=2.2, drone=on_plate, armed=True, offboard=True, plate=plate, rates_mode=True))
    assert o.mode == "position" and pm.anchor[2] > on_plate.p[2] + 0.9
    away = DroneState(p=pm.anchor.copy(), v=np.zeros(3), R=np.eye(3), w=np.zeros(3))
    o = pm.step(Inputs(t=3.0, drone=away, armed=True, offboard=True, plate=plate, rates_mode=False))
    assert o.phase == Phase.DONE and pm.result == "departed"


# ----------------------------------------------------------------------------- Q2 verify (offline)
def _q2_window(slip=False, n=80):
    rng = np.random.default_rng(0)
    w = np.zeros((n, 11), dtype=np.float32)
    w[:, 0:3] = [0.0, -8.9, 4.1]                       # gravity in a 65 deg tilted body (specific force)
    w[:, 0:2] += rng.normal(0, 2.0 if slip else 0.01, (n, 2))
    w[:, 6], w[:, 7], w[:, 9], w[:, 10] = 8.0, 1.0, 1.0, 0.1
    return w


def test_q2_macro_profile_and_rules():
    from perch_q2 import (PH_PULL, PH_RELAX, PULL03, Q2Cfg, Q2Verifier, RELAX, hp_energy, macro_thrust)
    c = Q2Cfg(rule="fsm_b4b")
    assert abs(macro_thrust(RELAX, 0.0, 9.81, c) - 9.81) < 1e-9
    assert abs(macro_thrust(RELAX, 0.1, 9.81, c) - 0.981) < 1e-9                # idle after the 50 ms ramp
    assert abs(macro_thrust(PULL03, 0.1, 0.981, c) - 0.3 * 9.81) < 1e-9          # holding PULL0.3
    assert abs(macro_thrust(PULL03, 0.25, 0.981, c) - 0.981) < 1e-9              # back to idle at the end
    assert hp_energy(_q2_window(True), 0) > c.e_thr > hp_energy(_q2_window(False), 0)
    # B4b: no slip at RELAX / PULL0.3 / PULL0.6 -> MOTORS_OFF after PULL0.6
    v = Q2Verifier(c)
    v.reset(0.0, 9.81)
    t, dec = 0.0, None
    while dec is None and t < 2.0:
        assert v.state[0] in (PH_RELAX, PH_PULL)
        _, dec = v.step(t, True, _q2_window(False))
        t += 0.01
    assert dec == "OFF" and v.result["d"] == 2 and abs(t - 0.66) < 0.02
    # B4b: slip at RELAX -> ABORT at the first decision point
    v.reset(0.0, 9.81)
    t, dec = 0.0, None
    while dec is None:
        _, dec = v.step(t, True, _q2_window(True))
        t += 0.01
    assert dec == "ABORT" and v.result["d"] == 0
    # loss of contact while powered -> SELF_ABORT; rule none -> OFF immediately
    v.reset(0.0, 9.81)
    v.step(0.0, False, None)
    assert v.step(0.05, False, None)[1] == "SELF_ABORT"
    n = Q2Verifier(Q2Cfg(rule="none"))
    n.reset(0.0, 9.81)
    assert n.step(0.0, True, None)[1] == "OFF"


class _FakeBelief:
    def __init__(self, p_hold, p_slip):
        self.p = (p_hold, p_slip)

    def __call__(self, window):
        assert window.shape == (80, 11)
        return self.p


def test_q2_dthr_rule_and_phase_machine():
    from perch_q2 import Q2Cfg, Q2Verifier
    c = Q2Cfg(rule="dthr", tau=0.56, s=0.6)
    for (ph, ps), want, d in (((0.9, 0.1), "OFF", 0), ((0.9, 0.7), "ABORT", 0), ((0.3, 0.1), "ABORT", 2)):
        v = Q2Verifier(c, _FakeBelief(ph, ps))
        v.reset(0.0, 9.81)
        t, dec = 0.0, None
        while dec is None:
            _, dec = v.step(t, True, _q2_window())
            t += 0.01
        assert dec == want and v.result["d"] == d
    # in the phase machine: contact -> VERIFY (rates 0, idle thrust) -> OFF -> REST
    with tempfile.TemporaryDirectory() as dd:
        make_test_policy(dd, 28)
        pol = PolicyRunner(dd, "numpy")
    sc = ActionScale(hover_thrust=0.6, thrust_model="x500_perch_gz", tw_ratio=2.25)
    pm = PhaseMachine(PhaseCfg(min_below_plate=1e9), pol, sc, start=Phase.PERCH,
                      verifier=Q2Verifier(c, _FakeBelief(0.9, 0.1)))
    pm._reset_perch()
    plate = _plate_static()
    drone = DroneState(p=plate.c + 2.0 * plate.R[:, 2], v=np.zeros(3), R=np.eye(3), w=np.zeros(3))
    pm.step(Inputs(t=0.0, drone=drone, armed=True, offboard=True, plate=plate, contact=False))
    o = pm.step(Inputs(t=0.01, drone=drone, armed=True, offboard=True, plate=plate, contact=True))
    assert o.phase == Phase.VERIFY
    outs = [pm.step(Inputs(t=0.02 + 0.01 * i, drone=drone, armed=True, offboard=True, plate=plate, contact=True,
                           q2_window=_q2_window())) for i in range(20)]
    assert all(np.allclose(x.rates_frd, 0) for x in outs if x.phase == Phase.VERIFY)
    assert outs[-1].phase == Phase.REST and pm.q2_result["decision"] == "OFF"
    assert outs[8].thrust_norm < 0.1                                          # RELAX: idle thrust


def test_q2_window_c5v2_columns():
    """C5 v2: the window always stores 13 columns; an 11-channel belief gets the first 11."""
    from perch_q2 import SensorWindow
    w = SensorWindow()
    for i in range(80):
        w.add([0, 0, 9.8], [0, 0, 0], 5.0, True, 0.04, 1, 0.1, slip_disp=(0.25 * (i % 2), -0.1))
    a = w.array()
    assert a.shape == (80, 13) and a[1, 11] == pytest.approx(0.25) and a[0, 12] == pytest.approx(-0.1)
    belief = Path("/home/sup/data/perchpilot/runs/q2_belief_v2_small")
    if belief.exists():
        from perch_q2 import BeliefNet
        net = BeliefNet(belief)
        assert net.n_ch == 11 and len(net(a)) == 2


def test_q2_b4e_rule():
    """B4e (C5 v2): |summed optical displacement| over the window below slip_off_mm[d] -> OFF; else ABORT at the end."""
    from perch_q2 import Q2Cfg, Q2Verifier, G
    still = np.zeros((80, 13), np.float32)
    slipping = still.copy()
    slipping[:, 11] = 0.05                       # 4 mm over the window
    for win, expect, d in ((still, "OFF", 0), (slipping, "ABORT", 2)):
        v = Q2Verifier(Q2Cfg(rule="fsm_b4e", slip_off_mm=[0.5, 0.5, 0.5]))
        v.reset(0.0, G)
        t, dec = 0.0, None
        while dec is None and t < 2.0:
            t += 0.005
            _, dec = v.step(t, True, win)
        assert dec == expect and v.result["d"] == d
    v = Q2Verifier(Q2Cfg(rule="fsm_b4e"))
    v.reset(0.0, G)
    with pytest.raises(ValueError):
        for k in range(1, 100):
            v.step(0.005 * k, True, np.zeros((80, 11), np.float32))
