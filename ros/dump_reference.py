"""Prepare offline PyTorch inference and Python flight replay references for perch_core."""
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from export_weights import export
from perch_core import (ActionScale, Inputs, Phase, PhaseCfg, PhaseMachine, PolicyRunner,
                        action_to_px4, drone_state_from_odometry, level_hold_action,
                        odometry_from_drone_state, DroneState, quat_to_rot,
                        rot_y, rot_z, rot_to_quat)
from perch_q2 import BeliefNet, Q2Cfg, Q2Verifier, SensorWindow
from plate_perception import PlateKF, plate_in_camera, tag_measurement


COUNT = 1000


def rows(path):
    with path.open() as stream:
        yield from map(json.loads, stream)


def logged_windows(path, channels):
    # The q2 log covers only part of each trial. Never fill a window across a gap or a trial boundary.
    samples, ends = [], []
    prev = None
    streak = 0
    for rec in rows(path):
        dt = rec["wall"] - prev["wall"] if prev is not None else 0.0
        if prev is None or rec["trial"] != prev["trial"] or not 0 < dt < 0.02:
            streak = 0
        prev = rec
        x = rec["x"]
        if len(x) < channels:
            streak = 0
            continue
        samples.append(x[:channels])
        streak += 1
        if streak >= 80:
            ends.append(len(samples))
    return np.asarray(samples, dtype=np.float32), ends


def inference_reference(args):
    rng = np.random.default_rng(709)
    policy = torch.jit.load(str(args.policy / "policy.ts"), map_location="cpu").eval()
    belief = torch.jit.load(str(args.belief / "belief.ts"), map_location="cpu").eval()
    n = policy.norm.mean.numel()
    channels = belief.mean.numel()
    obs = [rec["obs"] for rec in rows(args.sitl_root / args.policy_log / "node_ticks.jsonl")
           if rec["obs"] is not None and len(rec["obs"]) == n]
    samples, ends = logged_windows(args.sitl_root / args.q2_log / "q2_samples.jsonl", channels)
    print(f"SITL candidates: {len(obs)} observations, {len(ends)} contiguous C5 windows")
    input_sets = {
        "policy_random": rng.normal(size=(COUNT, n)) * policy.norm.var.sqrt().numpy() + policy.norm.mean.numpy(),
        "policy_sitl": np.asarray(obs)[rng.choice(len(obs), COUNT, replace=False)],
        "belief_random": rng.normal(size=(COUNT, 80, channels)) * belief.std.numpy() + belief.mean.numpy(),
        "belief_sitl": np.stack([samples[i - 80:i] for i in rng.choice(ends, COUNT, replace=False)]),
    }
    with torch.inference_mode():
        for name, data in input_sets.items():
            x = torch.from_numpy(data.astype(np.float32))
            model = policy if name.startswith("policy") else belief
            y = torch.cat([model(batch) for batch in x.split(50)])
            np.concatenate((x.flatten(1).numpy(), y.numpy()), axis=1).astype("<f4").tofile(args.out / f"{name}.f32")


def json_value(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot serialize {type(obj)}")


def write(stream, **row):
    print(json.dumps(row, default=json_value, allow_nan=False, separators=(",", ":")), file=stream)


CASES = (
    ("full_cycle", "placeholder", True, True, None),
    ("b4e_abort", "fsm_b4e", False, False, "ABORT"),
    ("dthr_off", "dthr", False, True, "OFF"),
    ("d12_contact_loss", "dthr", False, True, "SELF_ABORT"),
    ("b4b_off", "fsm_b4b", False, False, "OFF"),
    ("none_off", "none", False, False, "OFF"),
    ("position_depart", "placeholder", False, False, None),
)


def replay_case(stream, policy, belief, name, rule, managed, push, decision):
    options = dict(rest_s=2.6, depart_pushoff=push, extra_delay_steps=4,
                   yaw_cmd_scale=0.0 if managed else 0.35)
    if managed:
        options["standby_offset"] = [0.3, 0.0, 0.4]
    # Exercise d_min by requiring PULL0.3 on the synthetic no-slip sequence.
    thresholds = dict(rule=rule, tau=0.05, s=0.99, d_min=1, slip_off_mm=[0.5, 0.5, 0.5])
    q2 = None if rule == "placeholder" else Q2Verifier(Q2Cfg(**thresholds), belief if rule == "dthr" else None)
    sc = ActionScale(rate_max=tuple(policy.meta["rate_max"]), tw_ratio=policy.meta["tw_ratio_nominal"],
                     thrust_model="x500_perch_gz")
    phase = Phase.WAIT
    p = np.array([2.0, 0.0, 2.0])
    velocity = np.array([0.06, -0.025, 0.0])
    pm = PhaseMachine(PhaseCfg(**options), policy, sc, pad_offset=0.227, start=phase, verifier=q2)
    kf, hist, sensors = PlateKF(q_acc=0.1), deque(maxlen=50), SensorWindow()
    hold = None
    phases, decisions, commands = set(), set(), []
    started = reset = False
    frames = 0
    write(stream, scenario=name, phase=phase.value, config=options,
          q2=thresholds, pad_offset=0.227, freeze_plate_v=True)
    for tick in range(1200):
        t = tick / 100
        rp = rot_z(0.07) @ rot_y(-1.1345)
        n = rp[:, 2]
        f = -n.copy()
        f[2] = 0.0
        f /= np.linalg.norm(f)
        yaw = np.arctan2(f[1], f[0])
        center = np.array([6.0, 0.05, 1.3]) + t * velocity
        target = center - 4 * f + np.array([0, 0, 1])
        el = t - (pm.t_enter or 0)
        rates_mode, contact = False, False
        armed = t >= 1.2
        v, rotation, w = np.zeros(3), rot_z(yaw), np.zeros(3)
        cmd = None
        if pm.phase == Phase.WAIT:
            if managed and armed:
                p = pm.standby.copy()
            if managed and pm.ready and not started:
                cmd = "start"
                started = pm.start_trial(t)
                assert started
                kf = PlateKF(q_acc=0.1)
        elif pm.phase == Phase.APPROACH:
            p, v = target.copy(), velocity.copy()
        elif pm.phase == Phase.PERCH:
            rates_mode = el > 0.12
            if pm.k >= 18:
                contact = True
                p, rotation, v = center + 0.227 * n, rp, velocity.copy()
            elif pm.k > 0:
                p = target + np.array([0.015 * pm.k, 0.004 * pm.k, -0.008 * pm.k])
                rotation = rot_z(yaw + pm.k * 0.002) @ rot_y(-0.024 * pm.k)
                v, w = np.array([0.8, 0.01, -0.4]), np.array([0.1, -0.5, 0.2])
        elif pm.phase in (Phase.VERIFY, Phase.REST):
            rates_mode = pm.phase == Phase.VERIFY
            p, rotation, v = center + 0.227 * n, rp, velocity.copy()
            contact = not (name == "d12_contact_loss" and pm.phase == Phase.REST)
            if pm.phase == Phase.REST:
                armed = el < 2.2
                if managed and not reset and el >= 2.4:
                    cmd = "reset"
                    pm.reset_trial()
                    reset = True
        elif pm.phase == Phase.DEPART:
            stage = pm.flag.get("stage")
            armed = el >= (2.35 if name == "position_depart" else 0.3) or name in ("b4e_abort", "d12_contact_loss")
            rates_mode = stage != "climb" and (push or stage is not None)
            if stage == "climb" or pm.flag.get("armed_once"):
                p = pm.anchor.copy()
            else:
                p, rotation, v = center + 0.227 * n, rp, velocity.copy()

        # Two 200 Hz callbacks between control ticks. Include multiple delivered slip
        # increments in one interval so that consumption and FIFO ordering are checked.
        samples = []
        if q2 is not None and tick > 0:
            for when in (t - 0.006, t - 0.001):
                glide = 0.025 if name == "b4e_abort" else 0.0
                imu = {"acc": [0.005 * np.sin(when), 0.005, -9.81], "gyro": [0.01, -0.02, 0.03]}
                hold = [[when, float(contact), 8.0 if contact else 0.0, glide, 0.0],
                        [when, float(contact), 8.0 if contact else 0.0, glide, 0.0]]
                if rule in ("fsm_b4b", "none"):
                    hold = [message[:3] for message in hold]
                thr, ph, level = pm.q2_channels()
                sensors.add(np.array(imu["acc"]) * [1, -1, -1], np.array(imu["gyro"]) * [1, -1, -1],
                            hold[-1][2], contact, thr, ph, level, np.sum(hold, axis=0)[3:])
                samples.append({"t": when, "imu": imu, "hold": hold, "x": sensors.buf[-1]})
        odom = tag = None
        if tick > 0:
            state = DroneState(p=p, v=v, R=rotation, w=w)
            odom = {k: np.asarray(value, dtype=np.float32) if isinstance(value, np.ndarray) else value
                    for k, value in odometry_from_drone_state(state).items()}
            d = drone_state_from_odometry(**odom, t=t)
            odom["pose_frame"] = 1
            hist.append(d)
            if tick == 1 or tick * 3 // 10 != (tick - 1) * 3 // 10:
                age = 0.0 if tick == 1 else 0.008 + 0.003 * (tick % 4)
                exposure = t - age
                sample = min(hist, key=lambda msg: abs(msg.t - exposure))
                c_cam, r_cam = plate_in_camera(sample.p, sample.R, center - age * velocity, rp)
                c_cam += 0.0001 * np.sin(t * np.array([15, 9, 12]))
                tag = {"age": age, "position": c_cam.astype(np.float32), "q": rot_to_quat(r_cam).astype(np.float32)}
                kf.update(tag_measurement(exposure, sample, tag["position"], quat_to_rot(tag["q"])))
        kf.freeze_v = pm.phase in (Phase.PERCH, Phase.VERIFY)
        d = hist[-1] if hist else None
        plate = kf.estimate(t)
        sensed_contact = bool(hold[-1][1]) if hold is not None else None
        win = sensors.array() if pm.phase == Phase.VERIFY and sensors.ready() else None
        out = pm.step(Inputs(t, d, armed, armed, plate, sensed_contact, rates_mode, win))
        if out.mode == "position" and rates_mode and d is not None:
            out.rates_frd, out.thrust_body, out.thrust_norm, _, _ = action_to_px4(level_hold_action(d, sc), sc)
        phases.add(out.phase.value)
        commands.extend(out.commands)
        if pm.q2_result:
            decisions.add(pm.q2_result["decision"])
        write(stream, t=t, odom=odom, tag=tag, sensors=samples, cmd=cmd, armed=armed, offboard=armed,
              rates_mode=rates_mode, output=dict(
                  phase=out.phase.value, mode=out.mode, pos=out.pos_enu, vel=out.vel_enu,
                  yaw=out.yaw_enu if np.isfinite(out.yaw_enu) else None, rates=out.rates_frd,
                  thrust=out.thrust_norm if np.isfinite(out.thrust_norm) else None, thrust_body=out.thrust_body,
                  commands=out.commands, hold_cmd=out.hold_cmd, action=out.action, obs=out.obs,
                  q2=pm.q2_result, k=pm.k, switching=bool(pm.flag.get("switching")),
                  ready=pm.ready, result=pm.result))
        frames += 1
        if (name not in ("full_cycle", "position_depart") and decisions
                and (name != "d12_contact_loss" or "SELF_ABORT" in decisions)):
            # Keep recording through disarm or through the first departure control commands.
            if pm.phase in (Phase.REST, Phase.DEPART) and t - pm.t_enter > 0.16:
                break
        if (managed and reset and pm.phase == Phase.WAIT) or (not managed and pm.phase == Phase.DONE):
            break
    assert {"WAIT", "APPROACH", "PERCH"} <= phases
    assert rule == "none" or "VERIFY" in phases
    if decision:
        assert decision in decisions, (name, decisions, pm.q2_result)
    if name in ("full_cycle", "position_depart"):
        assert {"REST", "DEPART", "DONE"} <= phases
        assert commands.count((400, 0.0, 21196.0)) >= 2
        assert (31010, 0.0, 0.0) in commands
    if name == "position_depart":
        assert (400, 1.0, 21196.0) in commands
    if name == "d12_contact_loss":
        assert (400, 0.0, 0.0) not in commands and (400, 0.0, 21196.0) not in commands
    print(f"replay {name}: {frames} ticks, phases {', '.join(sorted(phases))}, decisions {sorted(decisions)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--belief", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sitl-root", type=Path, default=Path("/home/sup/data/perchpilot/sitl_eval"))
    parser.add_argument("--policy-log", default="main_p2y_v0")
    parser.add_argument("--q2-log", default="q2_dthr_v7_qos")
    parser.add_argument("--replay-only", action="store_true", help="skip inference fixtures from the SITL logs")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for kind, run in (("policy", args.policy), ("belief", args.belief)):
        export(run / f"{kind}.ts", kind, args.out / kind)
        if (run / "best.pt").is_file():
            export(run / "best.pt", kind, args.out / f"{kind}-pt")
    if not args.replay_only:
        inference_reference(args)
    with (args.out / "replay.jsonl").open("w") as stream:
        policy = PolicyRunner(args.policy, "torch")
        belief = BeliefNet(args.belief)
        for case in CASES:
            replay_case(stream, policy, belief, *case)


if __name__ == "__main__":
    main()
