"""CPU checks for C5 v2 sensor physics, causality and the channel ablation."""
from collections import Counter
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q2.belief_net import BeliefNet  # noqa: E402
from q2.dataset import gen_chunk  # noqa: E402
from q2.decide_v4 import b4e  # noqa: E402
from q2.fsm_features import features, slip_distance  # noqa: E402
from q2.sitl_external import decision_windows  # noqa: E402
from perch_q2 import SensorWindow  # noqa: E402
from sim.contact_phase import N_CH, ContactCfg, ContactPhaseSim, decision_steps  # noqa: E402


def test_c5_v1_bitwise_against_v5():
    source = subprocess.check_output(["git", "show", "ee2f824:perchpilot/sim/contact_phase.py"], cwd=ROOT)
    old = types.ModuleType("contact_phase_v5_reference")
    with patch.dict(sys.modules, {old.__name__: old}):
        exec(compile(source, str(ROOT / "sim/contact_phase.py"), "exec"), old.__dict__)
    old_chunk = types.FunctionType(gen_chunk.__wrapped__.__code__,
                                   {**gen_chunk.__wrapped__.__globals__, "ContactPhaseSim": old.ContactPhaseSim})
    for dr in (False, True):
        kw = dict(num_envs=64, sensor_dr=dr, tw_ratio=(1.9, 2.6))
        args = (100_600_000, "cpu", 2)
        with torch.no_grad():
            ref, ref_stats = old_chunk(old.ContactCfg(**kw), *args)
        legacy, stats = gen_chunk(ContactCfg(**kw), *args)
        optical, optical_stats = gen_chunk(ContactCfg(**kw, slip_sensor=True), *args)
        assert ref_stats == stats == optical_stats
        assert ref.keys() == legacy.keys() == optical.keys()
        for key in ref:
            v6 = optical[key][..., :N_CH] if key == "X" else optical[key]
            assert legacy[key].tobytes() == ref[key].tobytes() == v6.tobytes(), key
        a = old.ContactPhaseSim(old.ContactCfg(**kw), seed=args[0])
        b = ContactPhaseSim(ContactCfg(**kw), seed=args[0])
        for sim in (a, b):
            sim.run_script_to(decision_steps(b.cfg)[-1])
        for key in ("hist", "fn_true", "status", "pos", "thrust"):
            assert getattr(a, key).numpy().tobytes() == getattr(b, key).numpy().tobytes(), key


def test_optical_noise_delay_and_surface_gate():
    sim = ContactPhaseSim(ContactCfg(num_envs=12_000, t_rest=0.01, max_contact_s=0.02, slip_sensor=True), seed=24)
    scale = sim.slip_scale[:, 0]
    assert 0.9 <= scale.min() and scale.max() <= 1.1
    assert abs(scale.mean() - 1) < 0.002
    assert abs(scale.std() - 0.2 / np.sqrt(12)) < 0.002

    pre = sim.hist[:, :sim.pre, N_CH:]
    assert 0.019 < pre.std() < 0.021 and abs(pre.mean()) < 0.0002
    assert torch.allclose(pre * 100, (pre * 100).round(), atol=1e-5, rtol=0)

    # A constant 1.5/-0.8 mm impulse along plate lat/up; measure in the pad axes.
    disp = torch.tensor([0.0015, -0.0008]).expand(sim.N, -1)
    expected = (1.5 * sim.e_lat - 0.8 * sim.e_up).unsqueeze(1) @ sim.R[:, :, :2]
    expected = expected.squeeze(1) * sim.slip_scale
    near = torch.ones(sim.N, dtype=torch.bool)
    for k in (1, 2, 3, 4):
        sim.k.fill_(k)
        # At the delayed arrival of sample 2, the second half has left the surface.
        if k == 4:
            near[sim.N // 2:] = False
        sim._optical_sample(disp if k in (1, 2) else torch.zeros_like(disp), near)
        reading = sim.hist[:, sim.pre + k, N_CH:]
        if k < 3:
            assert 0.019 < reading.std() < 0.021 and reading.abs().max() < 0.11
        else:
            assert torch.count_nonzero(reading[~near]) == 0
            measured = reading[near]
            dropped = (measured == 0).all(-1)
            assert 0.014 < dropped.float().mean() < 0.026
            delta = (measured - expected[near])[~dropped]
            assert abs(delta.mean()) < 0.001 and 0.019 < delta.std() < 0.022
            assert torch.allclose(measured * 100, (measured * 100).round(), atol=2e-5, rtol=0)


def test_step_displacement_and_no_future_label_input():
    cfg = ContactCfg(num_envs=128, t_rest=0.1, slip_sensor=True, sensor_dr=True)
    sim = ContactPhaseSim(cfg, seed=27)
    rec = sim.run_script_to(40, record=True)
    alive = rec["alive"][:, 40]
    delta = (rec["pos"][alive, 38] - rec["pos"][alive, 37]) * 1000
    slide = delta[:, :1] * sim.e_lat[alive] + delta[:, 1:] * sim.e_up[alive]
    expected = (slide[:, None, :] @ sim.R[alive, :, :2]).squeeze(1) * sim.slip_scale[alive]
    measured = sim.hist[alive, sim.pre + 40, N_CH:]
    good = (expected.norm(dim=1) > 0.2) & (measured.norm(dim=1) > 0.05)
    assert good.sum() > 5 and (measured[good] - expected[good]).abs().max() < 0.1

    reference, _ = gen_chunk(cfg, 27, "cpu", 1)
    original_fork = ContactPhaseSim.rest_fork

    def change_labels(*args, **kwargs):
        result = original_fork(*args, **kwargs)
        return {**result, "held": ~result["held"], "fell": ~result["fell"], "slip_path": result["slip_path"] + 10}

    with patch.object(ContactPhaseSim, "rest_fork", change_labels):
        changed, _ = gen_chunk(cfg, 27, "cpu", 1)
    assert np.all(reference["held"] != changed["held"])
    assert np.all(changed["slip_path_if_off_m"] >= 10)
    assert reference["X"].tobytes() == changed["X"].tobytes()


def test_fsm_ablation_and_log_windows():
    w = torch.zeros(2, 80, 13)
    w[:, 0, -2:] = torch.tensor([3.0, 10.0])
    w[:, 1, -1] = -6
    assert slip_distance(w).tolist() == [5.0, 5.0]
    assert torch.equal(features(w, 0.005, use_slip=False), features(w[:, :, :11], 0.005))
    assert torch.equal(features(w, 0.005)[:, -1], slip_distance(w))
    model = BeliefNet(torch.arange(13), torch.ones(13), drop=["slip"], ch=4).eval()
    ts = torch.jit.trace(model, w)
    noisy = w.clone()
    noisy[:, :, -2:] += 10_000
    assert torch.equal(ts(w), ts(noisy))

    masks = np.ones((3, 3), bool)
    ep = {"alive": masks, "held": masks, "strict": masks, "fell": ~masks,
          "slip_mm": np.array([[1, 9, 9], [2, 2, 9], [2, 9, 3]], dtype=float)}
    result = b4e(ep, {"slip_off_mm": [1.5, 2.5, 3.5]})
    assert result["d"].tolist() == [0, 1, 2]
    ep.update(held=~masks, strict=~masks, fell=masks)
    assert np.array_equal(b4e(ep, {"slip_off_mm": [1.5, 2.5, 3.5]})["d"], result["d"])

    buf = SensorWindow()
    rows = []
    for i in range(81):
        phase, level = (0, 1) if i < 50 else ((1, 0.1) if i < 80 else (3, 0))
        buf.add([i] * 3, [0] * 3, 3, True, 0.05, phase, level, (0.01 * i, -0.02 * i))
        rows.append({"x": buf.buf[-1].tolist(), "wall": i * 0.005})
    trial = {"trial": 1, "q2": {"points": [{"d": 0, "macro": "RELAX", "t": 0.15}]}}
    raw = np.asarray([r["x"] for r in rows[:80]], np.float32)
    for width in (11, 13):
        (_, _, _, x), = decision_windows(rows, trial, {}, Counter(), width)
        assert x.tobytes() == raw[:, :width].tobytes()
    rows[0]["x"] = rows[0]["x"][:11]
    skipped = Counter()
    assert not list(decision_windows(rows, trial, {}, skipped, 13)) and skipped["no_C5_v2_slip"] == 1


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_c5_v1_bitwise_against_v5()
    test_optical_noise_delay_and_surface_gate()
    test_step_displacement_and_no_future_label_input()
    test_fsm_ablation_and_log_windows()
    print("4 Q2 v6 CPU tests passed")
