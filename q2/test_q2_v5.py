"""CPU checks for v5 DR, C5 extraction and conservative OFF scoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q2.belief_net import BeliefNet  # noqa: E402
from q2.dataset import gen_chunk  # noqa: E402
from q2.eval_sitl_external import load_dataset, make_trace, replay  # noqa: E402
from q2.decide_v3 import b4b  # noqa: E402
from q2.sitl_external import batch_logs, decision_windows, off_truth  # noqa: E402
from perch_q2 import hp_energy  # noqa: E402
from sim.contact_phase import ContactCfg, ContactPhaseSim, decision_steps  # noqa: E402

V4_REF = "6805224"


def test_dr_off_bitwise_against_v4():
    source = subprocess.check_output(["git", "show", f"{V4_REF}:perchpilot/sim/contact_phase.py"], cwd=ROOT)
    old = types.ModuleType("contact_phase_v4_reference")
    sys.modules[old.__name__] = old
    try:
        exec(compile(source, str(ROOT / "sim/contact_phase.py"), "exec"), old.__dict__)
    finally:
        del sys.modules[old.__name__]
    cfg_kw = {"num_envs": 256, "tw_ratio": (1.9, 2.6)}
    args = (100_000_000, "cpu", 2)
    current = ContactPhaseSim(ContactCfg(**cfg_kw), seed=args[0])
    reference = old.ContactPhaseSim(old.ContactCfg(**cfg_kw), seed=args[0])
    last = decision_steps(current.cfg)[-1]
    current.run_script_to(last)
    reference.run_script_to(last)
    for key in ("hist", "fn_true", "status", "loss_k", "thrust", "pos"):
        a, b = getattr(current, key), getattr(reference, key)
        assert a.numpy().tobytes() == b.numpy().tobytes(), key

    # Inject the historical simulator into a copy of the same dataset wrapper.
    old_chunk = types.FunctionType(gen_chunk.__wrapped__.__code__,
                                   {**gen_chunk.__wrapped__.__globals__, "ContactPhaseSim": old.ContactPhaseSim})
    with torch.no_grad():
        ref, stats = old_chunk(old.ContactCfg(**cfg_kw), *args)
    same, same_stats = gen_chunk(ContactCfg(**cfg_kw), *args)
    assert same_stats == stats
    assert same.keys() == ref.keys()
    for key in ref:
        assert same[key].dtype == ref[key].dtype and same[key].shape == ref[key].shape
        assert same[key].tobytes() == ref[key].tobytes(), key
    dr, dr_stats = gen_chunk(ContactCfg(**cfg_kw, sensor_dr=True), *args)
    assert dr_stats == stats and dr["X"].tobytes() != ref["X"].tobytes()
    for key in ref.keys() - {"X"}:
        assert dr[key].tobytes() == ref[key].tobytes(), key
    assert dr["X"][:, :, 7:].tobytes() == same["X"][:, :, 7:].tobytes()


def test_c5_raw_and_replay():
    rows = []
    for i in range(81):
        if i < 50:
            phase, level, code = "PERCH", 1.0, 0
        elif i < 80:
            phase, level, code = "VERIFY", 0.1, 1
        else:
            phase, level, code = "REST", 0.0, 3
        rows.append({"wall": i * 0.005, "phase": phase, "x": [i] * 6 + [-5, 1, 0.1, code, level]})
    trial = {"trial": 1, "mu": 0.9, "f_hold_n": 60,
             "q2": {"decision": "OFF", "points": [{"d": 0, "macro": "RELAX", "t": 0.15}]}}
    (d, macro, t, x), = decision_windows(rows, trial, {}, {})
    assert (d, macro, t) == (0, "RELAX", rows[79]["wall"])
    assert x.shape == (80, 11) and x.dtype == np.float32
    assert x.tobytes() == np.asarray([r["x"] for r in rows[:80]], np.float32).tobytes()
    trial.update(mu=0.2, f_hold_n=10, held_star=False, truth=1000)
    assert next(decision_windows(rows, trial, {}, {}))[-1].tobytes() == x.tobytes()

    w = torch.from_numpy(x[None])
    mean, std = torch.arange(11, dtype=torch.float32), torch.arange(1, 12, dtype=torch.float32)
    model = BeliefNet(mean, std).eval()
    ts = torch.jit.trace(model, torch.zeros_like(w))
    z = ((w - mean) / std).transpose(1, 2)
    logits = model.head(model.conv(z))
    expected = torch.stack([logits[:, 0].sigmoid(), logits[:, 1].sigmoid(), logits[:, 2]], -1)
    assert torch.equal(ts(w), expected)

    scores = {k: torch.zeros(3) for k in ("p", "ps", "hp_energy", "b4d")}
    windows = [{"d": i, "held_star": None, "fell": None} for i in range(3)]
    episodes = [{"rows": [0, 1, 2], "points": [0, 1, 2], "off_d": 3, "powered_loss": False}]
    ep, ix = make_trace(scores, episodes, windows, last=2)
    off = replay(b4b, ep, ix, episodes, windows, {"e_thr": 1})
    assert off["off_selected"] == 1 and off["off_at_logged_decision"] == off["off_with_hold_truth"] == 0
    assert off["false_safe_on_observed_off"]["p"] is None
    episodes[0]["rows"] = [0]
    ep, ix = make_trace(scores, episodes, windows, last=2)
    assert replay(b4b, ep, ix, episodes, windows, {"e_thr": 1})["trace_censored"] == 1


def test_off_reference_before_disarm():
    samples = [{"wall": 0.5, "phase": "REST", "x": [0] * 9 + [3, 0]}]
    hold = [{"kind": "hold", "wall": t, "sim_t": t, "slip_mps": 0.0,
             "margin_n": 2, "engaged": t < 0.6} for t in (0.49, 0.5, 0.55, 0.6, 0.7)]
    off = off_truth(samples, hold)
    assert off["commanded"] and off["fell"] and not off["held_star"] and not off["disarmed"]
    assert off_truth([], hold)["fell"] is None
    assert off_truth(samples, hold[:3])["held_star"] is None


def check_logged_archive(data, model_dir):
    w, manifest, _ = load_dataset(data)
    assert w.shape == (manifest["n_windows"], 80, 11)
    found = 0
    for batch in ("q2_probe_m1", "q2_probe_m2"):
        path = Path(manifest["raw_log_root"]) / batch
        records = {t["trial"]: (t, samples) for t, samples, _ in batch_logs(path, {})}
        for e in manifest["episodes"]:
            if e["batch"] != batch:
                continue
            trial, samples = records[e["trial"]]
            hp_logged = {p["d"]: p["hp_energy"] for p in (trial["q2"] or {}).get("points", [])}
            for row in e["rows"]:
                meta = manifest["windows"][row]
                end = next(i for i, r in enumerate(samples) if r["wall"] == meta["end_wall"])
                original = np.array([r["x"] for r in samples[end - 79:end + 1]], dtype=np.float32)
                assert w[row].numpy().tobytes() == original.tobytes()
                assert abs(hp_energy(original, meta["d"]) - hp_logged[meta["d"]]) < 1e-4
                if e["off_d"] != meta["d"]:
                    assert meta["held_star"] is None and meta["fell"] is None
                found += 1
    assert found > 0
    ts = torch.jit.load(str(model_dir / "belief.ts"), map_location="cpu").eval()
    meta = json.loads((model_dir / "belief.json").read_text())
    assert meta["channels"] == manifest["channels"] and meta["ts_check_maxabs"] == 0
    with torch.inference_mode():
        z = (w.float() - ts.mean) / ts.std * ts.mask
        logits = ts.head(ts.conv(z.transpose(1, 2)))
        expected = torch.stack([(logits[:, 0] / ts.temp).sigmoid(), logits[:, 1].sigmoid(), logits[:, 2]], -1)
        assert torch.equal(ts(w), expected)
    print(f"Verified {found} original probe windows and exported normalization on {len(w)} windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--belief", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    test_dr_off_bitwise_against_v4()
    test_c5_raw_and_replay()
    test_off_reference_before_disarm()
    if args.external and args.belief:
        check_logged_archive(args.external, args.belief)
    print("Q2 v5 CPU checks passed")
