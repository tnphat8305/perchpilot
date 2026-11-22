"""Small CPU protocol tests. Run directly or with pytest."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q2.dataset import V4_SPLITS, batches, episode_ids, gen_chunk, split_plan  # noqa: E402
from q2.decide_v3 import curve, dthr_fn, online_check  # noqa: E402
from q2.decide_v4 import bootstrap_ci, canonical, collect, digest, locked_test, select, test_summary as summarize_test  # noqa: E402
from sim.contact_phase import NOMINAL_TW, ContactCfg, ContactPhaseSim, decision_steps  # noqa: E402


def test_tw_command():
    n = 512
    cfg = ContactCfg(num_envs=n, t_rest=0.1)
    sim = ContactPhaseSim(cfg, seed=14)
    k = decision_steps(cfg)[0]
    sim.run_script_to(k)
    e = sim.ar[sim.k == k]
    w = sim.windows(e, sim.k[e])
    inferred = w[:, -1, 10] / w[:, -1, 8]
    mae = (inferred - sim.tw[e]).abs().mean().item()
    assert torch.allclose(inferred, torch.full_like(inferred, NOMINAL_TW))
    assert mae > 0.3
    # The varied pre-touchdown command (unlike the flat idle one) supports a correlation check.
    pre = sim.hist[:, sim.pre - 1, 10] / sim.hist[:, sim.pre - 1, 8]
    corr = torch.corrcoef(torch.stack((pre, sim.tw)))[0, 1].item()
    assert abs(corr) < 0.12
    # Control reproduces the historical leak.
    legacy = ContactPhaseSim(ContactCfg(num_envs=n, t_rest=0.1, legacy_thrust=True), seed=14)
    legacy.run_script_to(k)
    e = legacy.ar[legacy.k == k]
    w = legacy.windows(e, legacy.k[e])
    assert torch.allclose(w[:, -1, 10] / w[:, -1, 8], legacy.tw[e])
    print(f"TW: nominal estimate MAE {mae:.3f}, pre-touchdown correlation {corr:+.3f}")


def test_splits_disjoint():
    cfg_kw = {"legacy_thrust": False, "t_rest": 0.1}
    plans = {s: split_plan(s, 63, 32, 3) for s in V4_SPLITS}
    assert plans["train"]["materials"] == plans["val"]["materials"]
    assert plans["dev_ood"]["materials"] == plans["test"]["materials"] != plans["train"]["materials"]
    ids = {}
    for s, plan in plans.items():
        expected = np.concatenate([episode_ids(seed, np.arange(n)) for seed, n in batches(plan)])
        if s in ("train", "val"):
            ids[s] = []
            for seed, n in batches(plan):
                res, _ = gen_chunk(ContactCfg(num_envs=n, **plan["materials"], **cfg_kw), seed, "cpu", 1)
                ids[s].extend(episode_ids(res["episode_seed"], res["env"]))
            assert set(ids[s]) <= set(expected)
        else:
            ids[s] = collect([], None, plan, cfg_kw, "cpu")["id"]
            assert np.array_equal(ids[s], expected)
    assert sum(len(set(v)) for v in ids.values()) == len(set().union(*ids.values()))


def test_test_gate():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        lock = root / "lock.json"
        config = {"test_plan": {"seed": 99}, "cfg_kw": {}, "selection": {"D-thr": {"tau": 0.8}}}
        reads = []

        def read(locked):
            saved = json.loads(lock.read_text())
            assert saved["sha256"] == digest(canonical(locked))
            assert saved["config"] == config
            assert list((root / ".test_claims").glob("*.json"))
            reads.append(locked["test_plan"])
            return np.arange(3)

        sha, chosen, loaded = locked_test(lock, config, root, read)
        assert sha == digest(canonical(chosen)) and loaded.tolist() == [0, 1, 2]
        for other_lock in (lock, root / "different_tag.json"):
            try:
                locked_test(other_lock, config, root, read)
            except FileExistsError:
                pass
            else:
                raise AssertionError("second test read allowed")
        assert len(reads) == 1


def test_bootstrap():
    rng = np.random.default_rng(172)
    # Identical yet nonconstant success flags have zero difference on every paired draw.
    binary = np.tile([1, 0], 50)
    assert bootstrap_ci(binary - binary, np.ones(100), rng) == {"lo": 0.0, "hi": 0.0, "n_defined": 10000}
    # D-thr wins in 20 of 100 pairs, otherwise ties: the exact bootstrap is Binomial(100, .2) / 100.
    delta = np.r_[np.ones(20), np.zeros(80)]
    ci = bootstrap_ci(delta, np.ones(100), rng)
    assert abs(ci["lo"] - 0.12) <= 0.01 and abs(ci["hi"] - 0.28) <= 0.01
    off = np.r_[np.ones(50), np.zeros(50)]
    risk = bootstrap_ci(off, off, rng)
    assert risk == {"lo": 1.0, "hi": 1.0, "n_defined": 10000}
    assert bootstrap_ci(off * 0, off * 0, rng) == {"lo": None, "hi": None, "n_defined": 0}


def test_select_and_zero_off():
    def outcomes(ep, prm):
        i = np.arange(400)
        return {"off": i < prm["n_off"], "held": i < prm["valid"], "strict": i < prm["valid"],
                "fell_off": i < prm.get("n_fell", 0), "self_abort": i < 0, "abort": i >= prm["n_off"],
                "d": np.where(i < prm["n_off"], 0, -1)}

    grid = [{"n_off": 380, "valid": 380}, {"n_off": 384, "valid": 350}, {"n_off": 0, "valid": 0}]
    selected = select(outcomes, {}, grid)
    assert selected["0.005"] is None
    assert selected["0.01"]["params"] == grid[1]  # 0/380 is above 1% before rounding.
    assert selected["0.02"]["params"] == grid[0]
    empty = curve(outcomes, {}, grid[-1:], ["n_off"])[0]
    assert empty["false_safe"] is None and empty["false_safe_strict"] is None
    row = summarize_test(outcomes({}, {"n_off": 100, "valid": 90, "n_fell": 3}), 0.02, np.array([0.15]),
                         np.random.default_rng(1))
    assert row["exceeds_cap"] and row["n_fell"] == 3 and row["false_safe"]["p"] == 0.03


def test_online_same_batch():
    class Model(torch.nn.Module):
        def forward(self, x):
            p = torch.sigmoid((x[:, -1, 6] - 5.0) * 3)
            return torch.stack((p, torch.zeros_like(p), torch.zeros_like(p)), dim=1)

    plan = split_plan("dev_ood", 93, 64, 2)
    cfg_kw = {"legacy_thrust": False, "t_rest": 0.2}
    models = [Model()]
    ep = collect(models, None, plan, cfg_kw, "cpu")
    seed, size = next(batches(plan))
    for d_min in (0, 2):
        result = online_check(models, {"tau": 0.5, "s": 0.8}, 21, plan["materials"], seed, "cpu", cfg_kw,
                              ep, dthr_fn("single", 2, d_min=d_min), "single", d_min, batch_n=size)
        assert result["episode_agreement"] == 1.0 and result["n"] == 21
    print("online: 21/21 across different replay/report sizes, early and late")


def test_sensor_switches():
    simulations = []
    for v, a in ((1, True), (0, True), (0, False)):
        cfg = ContactCfg(num_envs=48, t_rest=0.2, vib_scale=v, slip_acc_in_imu=a)
        sim = ContactPhaseSim(cfg, seed=32)
        k = decision_steps(cfg)[2]
        rec = sim.run_script_to(k, record=True)
        simulations.append((rec, sim.hist))
    base_rec, base_h = simulations[0]
    for rec, h in simulations[1:]:
        assert all(torch.equal(rec[key], base_rec[key]) for key in rec)
        assert torch.equal(h[:, :, 6:], base_h[:, :, 6:])
    assert not torch.equal(simulations[1][1][:, :, :6], base_h[:, :, :6])
    assert not torch.equal(simulations[1][1][:, :, :3], simulations[2][1][:, :, :3])
    assert torch.equal(simulations[1][1][:, :, 3:6], simulations[2][1][:, :, 3:6])


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_tw_command()
    test_splits_disjoint()
    test_test_gate()
    test_bootstrap()
    test_select_and_zero_off()
    test_online_same_batch()
    test_sensor_switches()
    print("7 Q2 v4 tests passed")
