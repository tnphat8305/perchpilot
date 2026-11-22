"""Select Q2 policies on dev_ood, lock the configuration, then score the paired final test once."""
import argparse
import hashlib
import io
import itertools
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q2 import fsm_features  # noqa: E402
from q2.belief_net import wilson  # noqa: E402
from q2.dataset import V4_SPLITS, batches, episode_ids, split_plan  # noqa: E402
from q2.decide import at_touchdown, summarize  # noqa: E402
from q2.decide_v3 import (SCRIPT_V3, SS, TAUS, b4a, b4b, dthr_fn, fsm_features as legacy_features,  # noqa: E402
                          online_check, policy)
from sim.contact_phase import ContactCfg, ContactPhaseSim, decision_steps  # noqa: E402
from sim.quad_perch_env import G  # noqa: E402

CAPS = (0.005, 0.01, 0.02)
FSM = ("B4a", "B4b", "B4c", "B4d")
IMMEDIATE = "OFF-at-touchdown"
RESAMPLES = 10_000
INFERENCE_BATCH = 512
HYBRIDS = ("D-thr+B4e", "D-thr|slip-abort")


@torch.no_grad()
def collect(models, handcrafted, plan, cfg_kw, device, *, touchdown=False):
    """Replay the fixed schedule once, with the same noise and REST futures for all methods."""
    chunks = []
    for seed, n in batches(plan):
        sim = ContactPhaseSim(ContactCfg(num_envs=n, **plan["materials"], **cfg_kw), device=device, seed=seed)
        at_k0 = {}
        if touchdown:
            r = sim.rest_fork(sim.ar, sim.k, sim.thrust, sim.pos)
            at_k0 = {f"{key}_k0": r[label] for key, label in
                     (("held", "held"), ("strict", "held_strict"), ("fell", "fell"))}
        ks = decision_steps(sim.cfg)[:len(SCRIPT_V3)]
        rec = sim.run_script_to(ks[-1], record=True, script=SCRIPT_V3)
        shape = (n, len(ks))
        arr = {k: torch.zeros(shape, dtype=torch.bool, device=device) for k in ("held", "strict", "fell")}
        arr.update({k: torch.zeros(shape, device=device) for k in ("hp_energy", "c_run_s", "fn_mean", "gyro_max", "b4d")})
        if sim.cfg.slip_sensor:
            arr.update({k: torch.zeros(shape, device=device) for k in ("slip_mm", "b4d_drop_slip")})
        arr["alive"] = rec["alive"][:, ks]
        for k in ("p", "ps"):
            arr[k] = torch.zeros((len(models), *shape), device=device)
        for d, k in enumerate(ks):
            live = sim.ar[arr["alive"][:, d]]
            if not live.numel():
                continue
            kk = torch.full_like(live, k)
            r = sim.rest_fork(live, kk, rec["thrust"][live, k], rec["pos"][live, k])
            for key, label in (("held", "held"), ("strict", "held_strict"), ("fell", "fell")):
                arr[key][live, d] = r[label]
            for e in live.split(INFERENCE_BATCH):
                w = sim.windows(e, torch.full_like(e, k))
                for j, model in enumerate(models):
                    o = model(w)
                    arr["p"][j, e, d], arr["ps"][j, e, d] = o[:, 0], o[:, 1]
                for key, value in legacy_features(w, d, sim.cfg).items():
                    arr[key][e, d] = value
                if handcrafted is not None:
                    for key, score in handcrafted.items():
                        arr[key][e, d] = score(w, sim.dt)
                if sim.cfg.slip_sensor:
                    arr["slip_mm"][e, d] = fsm_features.slip_distance(w)
        chunks.append({**{k: t.cpu().numpy() for k, t in (arr | at_k0).items()},
                       "id": episode_ids(seed, np.arange(n)), "mu": sim.mu.cpu().numpy(),
                       "f_hold_mg": (sim.f_hold / G).cpu().numpy()})
        del sim, rec
    return {k: np.concatenate([c[k] for c in chunks], axis=1 if k in ("p", "ps") else 0) for k in chunks[0]}


def b4c(ep, prm):
    energy = ep["hp_energy"]
    return policy(ep, energy < prm["e_off"], energy >= prm["e_thr"], last=2)


def b4d(ep, prm, last=2, d_min=0, score_key="b4d"):
    off = ep[score_key] >= prm["tau"]
    off[:, :d_min] = False
    return policy(ep, off, np.zeros_like(off), last=last)


def b4e(ep, prm):
    off = ep["slip_mm"] < prm["slip_off_mm"]
    return policy(ep, off, np.zeros_like(off), last=2)


def dthr_b4e(ep, prm):
    off = (ep["p"][0] >= prm["tau"]) & (ep["slip_mm"] < prm["slip_off_mm"])
    return policy(ep, off, ep["ps"][0] >= prm["s"], last=2)


def dthr_slip_abort(ep, prm):
    stop = (ep["ps"][0] >= prm["s"]) | (ep["slip_mm"] > prm["slip_abort_mm"])
    return policy(ep, ep["p"][0] >= prm["tau"], stop, last=2)


def hybrid_counts(ranks, sizes, eligible, masks):
    """Count first OFFs and powered losses for every combination of phase thresholds."""
    ranks = np.where(eligible, ranks, sizes)
    counts = np.zeros((4, *sizes), dtype=np.int64)
    for d, outcomes in enumerate(masks):
        # For OFF at d, previous phases must fail their gates and d must pass.
        # The fourth mask counts losses with no OFF at any phase.
        dims = tuple(k + 1 for k in sizes[:d + 1])
        ix = np.ravel_multi_index(ranks[:, :d + 1].T, dims)
        prior = min(d, 3)
        for k, mask in outcomes:
            hist = np.bincount(ix[mask], minlength=np.prod(dims)).reshape(dims)
            for axis in range(prior):
                hist = np.flip(np.flip(hist, axis).cumsum(axis), axis)
            selection = (slice(1, None),) * prior
            if d < 3:
                hist = hist.cumsum(d)
                selection += (slice(None, -1),) + (None,) * (2 - d)
            counts[k] += hist[selection]
    return counts


def select_slip(ep, grid, thresholds, *, abort=False):
    """Exhaust the joint grid using episode counts instead of replaying each tuple."""
    alive, slip = ep["alive"], ep["slip_mm"]
    held, fell = ep["held"], ep["fell"]
    if abort:
        peak = np.maximum.accumulate(np.where(alive, slip, -np.inf), axis=1)
        ranks = np.searchsorted(thresholds, peak, side="left")
    else:
        sizes = tuple(map(len, thresholds))
        ranks = np.column_stack([np.searchsorted(t, slip[:, d], side="right")
                                 for d, t in enumerate(thresholds)])
    best, best_rank = dict.fromkeys(CAPS), {}
    for prm in grid:
        stopped = np.logical_or.accumulate(alive & (ep["ps"][0] >= prm["s"]), axis=1)
        eligible = alive & ~stopped & (ep["p"][0] >= prm["tau"])
        loss = ~alive[:, -1] & ~stopped[:, -1]
        if abort:
            # Slip ABORT can only remove a baseline OFF. Its first phase is unchanged.
            idx = np.flatnonzero(eligible.any(1))
            d = eligible[idx].argmax(1)
            r = ranks[idx, d]
            samples = (r, r[fell[idx, d]], r[held[idx, d]], ranks[loss & ~eligible.any(1), -1])
            counts = np.stack([np.bincount(x, minlength=len(thresholds) + 1)[:-1].cumsum() for x in samples])
        else:
            masks = [[(0, eligible[:, d]), (1, eligible[:, d] & fell[:, d]), (2, eligible[:, d] & held[:, d])]
                     for d in range(3)] + [[(3, loss)]]
            counts = hybrid_counts(ranks, sizes, eligible, masks)
        offs, falls, valid_count, self_abort = counts.reshape(4, -1)
        n = np.maximum(offs, 1)
        p = falls / n
        z = 1.96
        upper = (p + z*z/(2*n) + z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
        for cap in CAPS:
            good = np.flatnonzero((offs > 0) & (upper <= cap))
            if not good.size:
                continue
            good = good[valid_count[good] == valid_count[good].max()]
            i = good[np.lexsort((self_abort[good], upper[good]))[0]]
            exact_upper = wilson(int(falls[i]), int(offs[i]), digits=None)["hi"]
            rank = (int(valid_count[i]), -exact_upper, -int(self_abort[i]))
            if cap in best_rank and rank <= best_rank[cap]:
                continue
            slip_params = ({"slip_abort_mm": thresholds[i]} if abort else
                           {"slip_off_mm": [t[j] for t, j in zip(thresholds, np.unravel_index(i, sizes))]})
            best_rank[cap] = rank
            best[cap] = {"params": {**prm, **slip_params},
                         "dev_ood": {"n": len(alive), "n_off": int(offs[i]), "n_fell": int(falls[i]),
                                     "n_valid": int(valid_count[i]), "false_safe_upper_95": exact_upper}}
    return {str(cap): choice for cap, choice in best.items()}


def threshold_grids(ep):
    alive = ep["alive"]
    q = lambda x, probs: np.unique(np.quantile(x, probs)).tolist() if x.size else []
    energy = ep["hp_energy"]
    shared = q(energy[alive], np.linspace(0.05, 0.995, 80))
    per_phase = [[0.0, *q(energy[:, d][alive[:, d]], np.linspace(0.05, 0.995, 11)), 1e30]
                 for d in range(alive.shape[1])]
    dthr = [{"tau": float(t), "s": s} for t, s in itertools.product(TAUS, SS)]
    grids = {
        "dthr": dthr,
        "B4a": [{"t_c": a, "F_min": b, "g_max": c} for a, b, c in itertools.product(
            [0.025, 0.05, 0.1, 0.15], [0, 2, 4, 6, 8, 10], [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1e30])],
        "B4b": [{"e_thr": e} for e in shared],
        "B4c": [{"e_off": list(e), "e_thr": stop} for e, stop in itertools.product(
            itertools.product(*per_phase), [*q(energy[alive], [0.50, 0.80, 0.95, 0.995]), 1e30])],
        "B4d": [{"tau": float(t)} for t in TAUS],
    }
    if "slip_mm" in ep:
        slip = ep["slip_mm"]
        per_phase = [[0.0, *q(slip[:, d][live], np.linspace(0.01, 0.995, 21)), 1e30]
                     for d, live in enumerate(alive.T)]
        grids["B4e"] = [{"slip_off_mm": list(e)} for e in itertools.product(*per_phase)]
    return grids


def select(fn, ep, grid):
    """Maximize the exact valid-perch count under the unrounded Wilson upper bound."""
    best = {cap: None for cap in CAPS}
    ranks = {}
    for prm in grid:
        o = fn(ep, prm)
        offs = int(o["off"].sum())
        if not offs:
            continue
        falls = int(o["fell_off"].sum())
        upper = wilson(falls, offs, digits=None)["hi"]
        if upper > CAPS[-1]:
            continue
        valid = int(o["held"].sum())
        rank = (valid, -upper, -int(o["self_abort"].sum()))
        for cap in CAPS:
            if upper <= cap and (best[cap] is None or rank > ranks[cap]):
                best[cap] = {"params": prm, "dev_ood": {"n": len(o["off"]), "n_off": offs, "n_fell": falls,
                                                       "n_valid": valid, "false_safe_upper_95": upper}}
                ranks[cap] = rank
    return {str(cap): choice for cap, choice in best.items()}


def model_policy(index, rule):

    def run(ep, prm):
        return rule({**ep, "p": ep["p"][index:index + 1], "ps": ep["ps"][index:index + 1]}, prm)

    return run


def bootstrap_ci(numerator, denominator, rng):
    """Resample episodes through counts of their joint outcomes, exactly for this statistic.

    For a delta, numerator is the paired difference per episode and denominator is 1.
    For a false-safe ratio a non-OFF episode contributes (0, 0).
    """
    cats, counts = np.unique(np.column_stack((numerator, denominator)), axis=0, return_counts=True)
    draws = rng.multinomial(len(numerator), counts / len(numerator), size=RESAMPLES)
    den = draws @ cats[:, 1]
    keep = den > 0
    ratios = (draws @ cats[:, 0])[keep] / den[keep]
    bounds = np.quantile(ratios, [0.025, 0.975]).tolist() if keep.any() else [None, None]
    return {"lo": bounds[0], "hi": bounds[1], "n_defined": int(keep.sum())}


def valid(o):
    return (o["off"] & o["held"]).astype(np.int8)


def compare(a, b, ra, rb, rng, first="D-thr"):
    if a is None or b is None:
        return {"status": "no_dev_ood_configuration", "delta_valid_perch": None, "bootstrap_95": None}
    diff = valid(a) - valid(b)
    ci = bootstrap_ci(diff, np.ones_like(diff), rng)
    status = "at_cap"
    if ra["exceeds_cap"] or rb["exceeds_cap"]:
        status = "cap_exceeded"
    elif ra["exceeds_cap"] is None or rb["exceeds_cap"] is None:
        status = "undefined_test_false_safe"
    return {"status": status, "delta_valid_perch": float(diff.mean()), "bootstrap_95": ci,
            "higher_valid_perch": first if diff.sum() > 0 else "comparator" if diff.sum() < 0 else "tie",
            "ci_includes_zero": ci["lo"] <= 0 <= ci["hi"]}


def test_summary(o, cap, times, rng):
    r = summarize(o)
    off = o["off"]
    d = o["d"][off]
    n_off, n_fell = int(off.sum()), int(o["fell_off"].sum())
    invalid = {"false_safe": o["fell_off"], "false_safe_held": off & ~o["held"],
               "false_safe_strict": off & ~o["strict"]}
    r.update({k: wilson(int(x.sum()), n_off, digits=None) for k, x in invalid.items()})
    r.update(n_off=n_off, n_fell=n_fell, n_valid=int(valid(o).sum()),
             exceeds_cap=n_fell / n_off > cap if n_off else None,
             mean_off_time_s=float(np.where(d < 0, 0.0, times[d]).mean()) if n_off else None,
             n_off_by_decision=np.bincount(d[d >= 0], minlength=len(times)).tolist(),
             n_off_at_touchdown=int((d < 0).sum()),
             false_safe_bootstrap_95={k: bootstrap_ci(x, off, rng) for k, x in invalid.items()})
    return r


def seed_variation(rows, family):
    scored = {k: r for k, r in rows.items()
              if (k == family or k.startswith(family + "/seed=")) and r is not None}
    vals = {}
    for metric in ("valid_perch", "false_safe", "motors_off"):
        p = [r[metric]["p"] for r in scored.values() if r[metric]["p"] is not None]
        vals[metric] = {"n": len(p), "mean": float(np.mean(p)) if p else None,
                        "std": float(np.std(p, ddof=1)) if len(p) > 1 else None}
    return {"scored_methods": list(scored), **vals}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value, *, exclusive=False):
    with path.open("x" if exclusive else "w") as f:
        json.dump(value, f, indent=1, allow_nan=False)
        f.write("\n")
        if exclusive:
            f.flush()
            os.fsync(f.fileno())


def locked_test(lock_path, config, episode_dir, read_test):
    """Only call the test reader after writing and verifying the lock and claiming the test."""
    sha = digest(canonical(config))
    write_json(lock_path, {"sha256": sha, "config": config}, exclusive=True)
    saved = json.loads(lock_path.read_text())
    if digest(canonical(saved["config"])) != sha or saved["sha256"] != sha:
        raise RuntimeError("invalid experiment lock")
    locked = saved["config"]
    # Separate tags for the same test may not turn test results into another tuning round.
    # Sensor sensitivity deliberately varies cfg_kw on the same paired episode plan.
    identity = digest(canonical({k: locked[k] for k in ("test_plan", "cfg_kw")}))
    claims = episode_dir / ".test_claims"
    claims.mkdir(parents=True, exist_ok=True)
    write_json(claims / f"{identity}.json", {"lock_sha256": sha, "lock": str(lock_path.resolve())}, exclusive=True)
    return sha, locked, read_test(locked)


def load_models(beliefs, ablations, manifest_sha, device):
    entries = []
    for kind, paths in (("seed", beliefs), ("ablation", ablations)):
        for path in paths:
            meta = json.loads((path / "belief.json").read_text())
            if meta.get("protocol") != "v4" or meta.get("dataset_sha256") != manifest_sha or meta["label"] != "held":
                raise ValueError(f"{path}: expected a held-star belief trained on this v4 dataset")
            if bool(meta["drop"]) != (kind == "ablation"):
                raise ValueError(f"{path}: unexpected sensor channel mask")
            ts = path / "belief.ts"
            model_bytes = ts.read_bytes()
            entries.append((kind, {"seed": meta["seed"], "drop": meta["drop"], "sha256": digest(model_bytes),
                                   "file": str(ts.resolve())}, torch.jit.load(io.BytesIO(model_bytes), map_location=device).eval()))
    ordered = sorted(entries, key=lambda e: (e[0] != "seed", e[1]["seed"], e[1]["drop"]))
    keys = [(kind, m["seed"] if kind == "seed" else tuple(m["drop"])) for kind, m, _ in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate belief seed or ablation")
    return [m for _, m, _ in ordered], [model for _, _, model in ordered]


def v5_control(path, splits, cfg_kw, device):
    lock = json.loads(path.read_text())
    cfg = lock["config"]
    if (digest(canonical(cfg)) != lock["sha256"] or cfg["protocol"] != "H2-v4"
            or not cfg["cfg_kw"].get("sensor_dr") or cfg["cfg_kw"].get("slip_sensor")
            or canonical(cfg["cfg_kw"]) != canonical({k: v for k, v in cfg_kw.items() if k != "slip_sensor"})
            or cfg["device"] != device or list(cfg["caps"]) != list(CAPS)
            or any(canonical(cfg["splits"][s]) != canonical(splits[s]) for s in ("dev_ood", "test"))):
        raise ValueError("v5 control must have a valid lock and the same dev_ood/test plans, physics and device")
    return {"lock_sha256": lock["sha256"], "lock": str(path.resolve())}


def v5_outcomes(path, lock_sha, test):
    data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as a:
        if (str(a["lock_sha256"]) != lock_sha or not np.array_equal(a["caps"], CAPS)
                or any(not np.array_equal(a[k], test[k]) for k in ("id", "mu", "f_hold_mg"))):
            raise ValueError("v5 episode archive does not match its lock or the v6 episodes")
        methods = a["methods"].tolist()
        keys = ("D-thr", "D-thr-late", "B4b", "B4c", "B4d")
        out = [{key: {"off": a["off"][ci, methods.index(key)], "held": a["held_star"][ci, methods.index(key)],
                      "fell_off": a["fell"][ci, methods.index(key)]} if a["available"][ci, methods.index(key)] else None
                for key in keys} for ci in range(len(CAPS))]
    return out, digest(data)


def evaluate(args):
    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    manifest = (args.data / "meta.json").read_bytes()
    ds = json.loads(manifest)
    if ds.get("protocol") != "v4":
        raise ValueError("decide_v4 needs a v4-protocol dataset")
    cfg = ContactCfg(**ds["cfg_kw"])
    version = "v7" if getattr(cfg, "imu_hold", False) else ("v6" if cfg.slip_sensor else ("v5" if cfg.sensor_dr else "v4"))
    spec = "1.1" if version == "v4" else "1.2"
    name = f"q2_{version}_{args.tag}"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.episode_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.results_dir / f"{name}_lock.json"
    if lock_path.exists():
        raise FileExistsError(lock_path)
    splits = ds["splits"]
    for split in V4_SPLITS:
        expected = split_plan(split, ds["args"]["n_" + split], ds["args"]["chunk"], ds["args"]["seed"])
        if canonical(expected) != canonical({k: splits[split][k] for k in expected}):
            raise ValueError(f"{split}: invalid v4 episode plan")
    if args.test_seed is not None:
        if args.test_seed == ds["args"]["seed"]:
            raise ValueError("--test-seed must differ from the dataset seed")
        original = splits["test"]
        fresh = split_plan("test", original["episodes"], original["chunk"], args.test_seed)
        splits = {**splits, "test": {**original, **fresh}}
    if args.reuse_b4d_lock and not cfg.slip_sensor:
        raise ValueError("--reuse-b4d-lock requires a slip-sensor dataset")
    if cfg.legacy_thrust:
        raise ValueError("v4 requires nominal thrust commands")
    cfg_kw = {k: v for k, v in asdict(cfg).items() if k not in ("num_envs", "mu", "f_hold_mg")}
    if not cfg.sensor_dr:
        cfg_kw.pop("sensor_dr")  # keep the v4 test-claim identity
    if not cfg.slip_sensor:
        cfg_kw.pop("slip_sensor")  # keep the existing v4/v5 test-claim identities
    control = None
    if args.v5_lock:
        if not cfg.slip_sensor or not args.v5_episodes or not args.v5_episodes.is_file():
            raise ValueError("v5 control requires a v6 run and a frozen --v5-episodes archive")
        control = v5_control(args.v5_lock, splits, cfg_kw, args.device)
        control["episodes"] = str(args.v5_episodes.resolve())
    times = np.array(decision_steps(cfg)[:len(SCRIPT_V3)]) * cfg.dt
    metas, models = load_models(args.belief, args.ablations, digest(manifest), args.device)
    frozen = json.loads(args.reuse_b4d_lock.read_text()) if args.reuse_b4d_lock else None
    if frozen and (digest(canonical(frozen["config"])) != frozen["sha256"]
                   or frozen["config"]["protocol"] != "H2-v4"
                   or frozen["config"]["dataset_sha256"] != digest(manifest)):
        raise ValueError("expected a valid B4d lock for this dataset")
    b4d_fit = (frozen["config"]["B4d_model"] if frozen else
               fsm_features.fit(args.data, cfg.dt, args.device, use_slip=cfg.slip_sensor))
    handcrafted = {"b4d": fsm_features.scorer(b4d_fit, args.device)}
    b4d_drop_fit = None
    if cfg.slip_sensor:
        b4d_drop_fit = (frozen["config"]["B4d_drop_slip_model"] if frozen else
                        fsm_features.fit(args.data, cfg.dt, args.device))
        handcrafted["b4d_drop_slip"] = fsm_features.scorer(b4d_drop_fit, args.device)
    after_fit = time.perf_counter()
    dev = collect(models, handcrafted, splits["dev_ood"], cfg_kw, args.device)
    after_dev = time.perf_counter()
    print(f"{name}: collected dev_ood in {after_dev - after_fit:.1f} s", flush=True)

    methods = {name: (fn, name) for name, fn in zip(FSM, (b4a, b4b, b4c, b4d))}
    if cfg.slip_sensor:
        methods.update({"B4e": (b4e, "B4e"),
                        "B4d/drop=slip": (lambda ep, prm: b4d(ep, prm, score_key="b4d_drop_slip"), "B4d")})
    for i, m in enumerate(metas):
        if m["drop"]:
            suffix = "/drop=" + "+".join(m["drop"])
            methods["D-thr" + suffix] = (model_policy(i, dthr_fn("single", 2)), "dthr")
            if m["drop"] == ["slip"]:
                methods["D-thr-late" + suffix] = (model_policy(i, dthr_fn("single", 2, d_min=2)), "dthr")
        else:
            suffix = f"/seed={m['seed']}" if i else ""
            methods["D-thr" + suffix] = (model_policy(i, dthr_fn("single", 2)), "dthr")
            methods["D-thr-late" + suffix] = (model_policy(i, dthr_fn("single", 2, d_min=2)), "dthr")

    grids = threshold_grids(dev)
    selections = {key: select(fn, dev, grids[grid]) for key, (fn, grid) in methods.items()}
    if cfg.slip_sensor:
        gates = [sorted({p["slip_off_mm"][d] for p in grids["B4e"]}) for d in range(3)]
        aborts = np.unique(np.r_[0.0, np.quantile(dev["slip_mm"][dev["alive"]], np.linspace(0.01, 0.995, 80)),
                                 1e30].astype(np.float32)).tolist()
        grids.update({HYBRIDS[0]: {"belief": grids["dthr"], "slip": gates},
                      HYBRIDS[1]: {"belief": grids["dthr"], "slip": aborts}})
        for i, m in enumerate(metas):
            if m["drop"]:
                continue
            suffix = f"/seed={m['seed']}" if i else ""
            ep = {**dev, "p": dev["p"][i:i + 1], "ps": dev["ps"][i:i + 1]}
            for family, rule, thresholds in zip(HYBRIDS, (dthr_b4e, dthr_slip_abort), (gates, aborts)):
                key = family + suffix
                methods[key] = (model_policy(i, rule), family)
                selections[key] = select_slip(ep, grids["dthr"], thresholds, abort=family == HYBRIDS[1])
    first = splits["dev_ood"]
    seed, batch_n = next(batches(first))
    prm = next((r["params"] for r in reversed(list(selections["D-thr"].values())) if r), {"tau": 0.5, "s": 0.6})
    checked = None
    if args.n_online > 0:
        checked = online_check([models[0]], prm, min(batch_n, args.n_online), first["materials"], seed, args.device,
                               cfg_kw, {**dev, "p": dev["p"][:1], "ps": dev["ps"][:1]}, dthr_fn("single", 2),
                               "single", 0, batch_n=batch_n)
        checked.update(split="dev_ood", first_episode_id=int(dev["id"][0]), batch_n=batch_n, params=prm)
        if not checked["all_identical"]:
            raise RuntimeError(f"online replay disagrees: {checked}")

    # Paths and checksums pin all learned models and the chosen thresholds to this test.
    source_files = ("q2/decide_v4.py", "q2/decide_v3.py", "q2/decide.py", "q2/dataset.py",
                    "q2/fsm_features.py", "q2/belief_net.py", "q2/train_belief.py",
                    "sim/contact_phase.py", "sim/quad_perch_env.py")
    config = {"protocol": "H2-v4", "spec": spec, "dataset_sha256": digest(manifest), "dataset": str(args.data.resolve()),
              "splits": splits, "test_plan": splits["test"], "cfg_kw": cfg_kw, "device": args.device,
              "decision_points_s": times.tolist(), "caps": CAPS, "models": metas, "B4d_model": b4d_fit,
              "sources_sha256": {path: digest((ROOT / path).read_bytes()) for path in source_files},
              "grid_sha256": digest(canonical(grids)), "grid_sizes": {
                  k: len(v) for k, v in grids.items() if k not in HYBRIDS},
              "selection": selections, "fixed": {IMMEDIATE: {}},
              "bootstrap": {"seed": args.bootstrap_seed, "resamples": RESAMPLES}}
    if cfg.slip_sensor:
        config["B4d_drop_slip_model"] = b4d_drop_fit
        config["grid_sizes"].update({HYBRIDS[0]: len(grids["dthr"]) * len(grids["B4e"]),
                                     HYBRIDS[1]: len(grids["dthr"]) * len(aborts)})
        config["slip_grids_mm"] = {HYBRIDS[0]: gates, HYBRIDS[1]: aborts}
    if frozen:
        config["B4d_fit_lock_sha256"] = frozen["sha256"]
    if control:
        config["v5_control"] = control
    before_test = time.perf_counter()
    print(f"{name}: dev_ood thresholds selected in {before_test - after_dev:.1f} s", flush=True)
    sha, locked, test = locked_test(lock_path, config, args.episode_dir,
                                    lambda lock: collect(models, handcrafted, lock["test_plan"],
                                                         lock["cfg_kw"], lock["device"], touchdown=True))
    after_test = time.perf_counter()
    v5, v5_sha = v5_outcomes(args.v5_episodes, control["lock_sha256"], test) if control else (None, None)

    n = len(test["id"])
    methods[IMMEDIATE] = (lambda ep, _: at_touchdown(ep, np.ones(n, bool)), None)
    mu_group, f_group = test["mu"] >= 1.0, test["f_hold_mg"] >= 3.0
    shape = (len(CAPS), len(methods), n)
    raw = {k: np.zeros(shape, bool) for k in ("off", "held_star", "held_strict", "fell", "abort", "self_abort")}
    raw.update(id=test["id"], mu=test["mu"], f_hold_mg=test["f_hold_mg"], mu_group=mu_group, f_hold_group=f_group,
               off_time_s=np.full(shape, np.nan, np.float32), available=np.zeros(shape[:2], bool),
               methods=np.array(list(methods)), caps=np.array(CAPS), lock_sha256=np.array(sha))
    rng = np.random.default_rng(locked["bootstrap"]["seed"])
    results = {}
    for ci, cap in enumerate(locked["caps"]):
        rows, outcomes = {}, {}
        for mi, (key, (fn, _)) in enumerate(methods.items()):
            choice = {"params": locked["fixed"][key]} if key in locked["fixed"] else locked["selection"][key][str(cap)]
            o = outcomes[key] = None if choice is None else fn(test, choice["params"])
            rows[key] = None if o is None else test_summary(o, cap, times, rng)
            if o is None:
                continue
            raw["available"][ci, mi] = True
            for field, source in (("off", "off"), ("held_star", "held"), ("held_strict", "strict"),
                                  ("fell", "fell_off"), ("abort", "abort"), ("self_abort", "self_abort")):
                raw[field][ci, mi] = o[source]
            off = o["off"]
            d = o["d"][off]
            raw["off_time_s"][ci, mi, off] = np.where(d < 0, 0.0, times[d])
        comparators = FSM + (("B4e",) if cfg.slip_sensor else ())
        pairs = [(d, f) for d in methods if d == "D-thr" or d.startswith("D-thr/")
                 for f in comparators]
        pairs += [(d, "B4b") for d in methods if d == "D-thr-late" or d.startswith("D-thr-late/")]
        if cfg.slip_sensor:
            pairs += [("D-thr-late", "B4e"), ("B4d", "B4e"), ("B4e", IMMEDIATE)]
            pairs += [(m.split("/")[0], m) for m in methods if m.endswith("/drop=slip")]
            pairs += [(m, f) for m in methods if m.split("/")[0] in HYBRIDS
                      for f in ("B4e", "D-thr" + m[len(m.split("/")[0]):])]
        comparisons = {f"{d} - {f}": compare(outcomes[d], outcomes[f], rows[d], rows[f], rng, first=d)
                       for d, f in pairs}
        results[str(cap)] = {"test": rows, "comparisons": comparisons,
                             "across_seeds": {f: seed_variation(rows, f) for f in ("D-thr", "D-thr-late", *HYBRIDS)
                                              if f in methods}}
        if v5:
            for key, o in v5[ci].items():
                n_off = int(o["off"].sum()) if o else 0
                r = {"exceeds_cap": int(o["fell_off"].sum()) / n_off > cap if n_off else None}
                comparisons[f"{key} - {key} (v5)"] = compare(outcomes[key], o, rows[key], r, rng, first=key)
    archive = args.episode_dir / f"{name}_episodes.npz"
    with archive.open("xb") as f:
        np.savez_compressed(f, **raw)
    result = {"name": name, "protocol": "H2-v4", "spec": spec, "lock": lock_path.name, "lock_sha256": sha,
              "episode_archive": str(archive.resolve()), "test_episodes": n, "online_check": checked,
              "by_cap": results, "timing_s": {"fit_B4d": round(after_fit - t0, 2), "dev_ood": round(after_dev - after_fit, 2),
                                             "select_and_check": round(before_test - after_dev, 2),
                                             "test": round(after_test - before_test, 2), "total": round(time.perf_counter() - t0, 2)}}
    if control:
        result["v5_control"] = {**control, "episodes_sha256": v5_sha}
    write_json(args.results_dir / f"{name}.json", result, exclusive=True)
    print(json.dumps({"name": name, "lock_sha256": sha, "timing_s": result["timing_s"], "online_check": checked}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--belief", type=Path, nargs="+", required=True, help="training run directory per belief seed")
    ap.add_argument("--ablations", type=Path, nargs="*", default=[])
    ap.add_argument("--v5-lock", type=Path, help="optional frozen v5 control paired on the same episodes")
    ap.add_argument("--v5-episodes", type=Path, help="episode archive from that v5 run")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test-seed", type=int, help="generate a fresh same-distribution v4 test plan")
    ap.add_argument("--reuse-b4d-lock", type=Path, help="reuse fitted B4d controls from a lock for this dataset")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n_online", type=int, default=128)
    ap.add_argument("--bootstrap_seed", type=int, default=7024)
    ap.add_argument("--episode_dir", type=Path, default=Path("/home/sup/data/perchpilot/q2eval"))
    ap.add_argument("--results_dir", type=Path, default=ROOT / "results")
    evaluate(ap.parse_args())


if __name__ == "__main__":
    main()
