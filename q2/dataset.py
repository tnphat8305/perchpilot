"""Q2 dataset generation: C5 sensor windows (80 x 11 or 13 @ 200 Hz) at decision points of the fixed
probing schedule (v2) RELAX -> PULL0.3 -> PULL0.6 -> PULL0.9, labelled by forking the state
and simulating 10 s of motors-off REST (sim/contact_phase.py).

Labels per window: held (= held_star v2: on the plate for 10 s, total slip < 2 cm), held_strict
(margin >= 0 at every step, the v1 label), fell_if_off, slip_path_if_off_m, margin_min_g, slip (now).

Per episode we record windows at
  * the 4 fixed decision points (end of each macro: t = 0.15, 0.40, 0.65, 0.90 s), and
  * `--extra` random intermediate time steps (the "random early stopping" variant: the
    episode could have been stopped there, so the belief net sees every phase).
Only envs still attached at that step contribute a window (a loss of contact while the motors
run is a SELF_ABORT, not a fall, and ends the episode).

With --protocol v4, sizes count episodes. Train/val use seen materials; dev_ood/test use
unseen materials and their manifest entries are simulated on demand by decide_v4.
The original window-count protocol is available as --protocol legacy (default).
Output (float16 windows, never in the repo):
  <out>/<split>/X_0000.npy   (n, 80, channels) float16
  <out>/<split>/Y_0000.npz   labels + metadata (see LABEL_KEYS)
  <out>/meta.json            sizes, class balance with Wilson 95% CIs, config
Run: python q2/dataset.py --out /home/sup/data/perchpilot/q2data/<tag> [--device cuda]
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.contact_phase import (FELL, SELF_ABORT, ContactCfg, ContactPhaseSim, decision_steps)  # noqa: E402
from sim.quad_perch_env import G  # noqa: E402
from q2.belief_net import wilson  # noqa: E402

SEEN = dict(mu=((0.25, 1.0),), f_hold_mg=((0.6, 3.0),))
UNSEEN = dict(mu=((0.15, 0.25), (1.0, 1.2)), f_hold_mg=((0.4, 0.6), (3.0, 3.5)))
SPLITS = {"train": (SEEN, 1_000_000), "val": (SEEN, 2_000_000), "test": (UNSEEN, 3_000_000)}
V4_SPLITS = ("train", "val", "dev_ood", "test")
WIN = 80
meta_dt = ContactCfg().dt


def split_plan(name, episodes, chunk, seed):
    if not 0 <= seed < 10_000 or not 0 < chunk < 2**20 or not 0 < episodes <= 100_000 * chunk:
        raise ValueError("v4 requires seed in [0, 10000), 1 to 100000 batches, and chunk < 2**20")
    return {"episodes": episodes, "chunk": chunk, "materials": SEEN if name in ("train", "val") else UNSEEN,
            "seed_start": 100_000_000 + (seed * len(V4_SPLITS) + V4_SPLITS.index(name)) * 100_000}


def episode_ids(batch_seed, env):
    return np.int64(batch_seed) * (2**20) + env


def batches(plan):
    n, chunk = plan["episodes"], plan["chunk"]
    for i, start in enumerate(range(0, n, chunk)):
        yield plan["seed_start"] + i, min(chunk, n - start)


def rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("RssAnon"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return -1.0


@torch.no_grad()
def gen_chunk(cfg, seed, device, extra):
    """Simulate one chunk of episodes; return numpy dict of windows + labels."""
    sim = ContactPhaseSim(cfg, device=device, seed=seed)
    fixed = decision_steps(cfg)
    rec = sim.run_script_to(fixed[-1], record=True)
    N = sim.N
    g = torch.Generator(device=sim.dev).manual_seed(seed + 7)
    cols = [torch.full((N,), k, dtype=torch.long, device=sim.dev) for k in fixed]
    lo = 5
    for _ in range(extra):
        r = torch.randint(lo, fixed[-1], (N,), device=sim.dev, generator=g)
        cols.append(r)
    out = {k: [] for k in ("X", "held", "held_strict", "slip", "margin_min_g", "fell_if_off", "slip_path_if_off_m",
                           "k", "fixed", "env")}
    for ci, kc in enumerate(cols):
        alive = rec["alive"][sim.ar, kc]
        e = sim.ar[alive]
        if e.numel() == 0:
            continue
        k = kc[alive]
        r = sim.rest_fork(e, k, rec["thrust"][e, k], rec["pos"][e, k])
        out["X"].append(sim.windows(e, k, WIN).half().cpu())
        out["held"].append(r["held"].cpu())
        out["held_strict"].append(r["held_strict"].cpu())
        out["slip_path_if_off_m"].append(r["slip_path"].cpu())
        out["slip"].append(rec["slip"][e, k].cpu())
        out["margin_min_g"].append((r["margin_min"] / G).cpu())
        out["fell_if_off"].append(r["fell"].cpu())
        out["k"].append(k.cpu())
        out["fixed"].append(torch.full((e.numel(),), ci < len(fixed), dtype=torch.bool))
        out["env"].append(e.cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    env = torch.from_numpy(res["env"]).to(sim.dev)
    ep = {"mu": sim.mu, "f_hold_mg": sim.f_hold / G, "tilt_deg": torch.rad2deg(sim.tilt), "vn": sim.vn,
          "vt": sim.vt, "misalign_deg": torch.rad2deg(sim.misalign), "w0": sim.w0.norm(dim=-1)}
    for k, v in ep.items():
        res[k] = v[env].float().cpu().numpy()
    res["episode_seed"] = np.full(len(res["k"]), seed, dtype=np.int64)
    # shuffle within the chunk so truncating the last shard does not drop the late decision points
    perm = np.random.default_rng(seed).permutation(len(res["k"]))
    res = {k: v[perm] for k, v in res.items()}
    lost = (sim.status == FELL) | (sim.status == SELF_ABORT)
    stats = {"episodes": N, "lost_contact_during_script": int(lost.sum()),
             "lost_detach": int((lost & (sim.loss_cause == 1)).sum()),
             "lost_slid_off": int((lost & (sim.loss_cause == 2)).sum())}
    del sim, rec
    return res, stats


def parse_cfg(text):
    """'a=1.0,b=0' -> {'a': 1.0, 'b': False} typed after the ContactCfg defaults."""
    base = asdict(ContactCfg())
    out = {}
    for kv in filter(None, text.split(",")):
        k, v = kv.split("=")
        k = k.strip()
        if k not in base:
            raise KeyError(f"unknown ContactCfg field {k}")
        t = type(base[k])
        out[k] = bool(int(v)) if t is bool else t(v)
    return out


LABEL_KEYS = ["held", "held_strict", "slip_path_if_off_m", "slip", "margin_min_g", "fell_if_off", "k", "fixed", "mu", "f_hold_mg", "tilt_deg",
              "vn", "vt", "misalign_deg", "w0", "episode_seed", "env"]


def gen_split(name, n_target, out, args, cfg_kw, plan=None):
    if plan is None:
        mats, base = SPLITS[name]
    else:
        mats = plan["materials"]
        schedule = iter(batches(plan))
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    total, i, t0 = 0, 0, time.time()
    agg = {"held": 0, "held_strict": 0, "slip": 0, "fell_if_off": 0, "n": 0, "episodes": 0,
           "lost_contact_during_script": 0, "lost_detach": 0, "lost_slid_off": 0, "by_k": {}}
    while (agg["episodes"] if plan else total) < n_target:
        seed, size = next(schedule) if plan else (base + args.seed * 10_000 + i, args.chunk)
        cfg = ContactCfg(num_envs=size, **mats, **cfg_kw)
        res, st = gen_chunk(cfg, seed, args.device, args.extra)
        n = len(res["k"]) if plan else min(len(res["k"]), n_target - total)
        np.save(d / f"X_{i:04d}.npy", res["X"][:n])
        labels = {k: res[k][:n] for k in LABEL_KEYS}
        if plan:
            labels["episode_id"] = episode_ids(seed, labels["env"])
        np.savez(d / f"Y_{i:04d}.npz", **labels)
        for k in ("held", "held_strict", "slip", "fell_if_off"):
            agg[k] += int(res[k][:n].sum())
        for k in ("episodes", "lost_contact_during_script", "lost_detach", "lost_slid_off"):
            agg[k] += st[k]
        fx = res["fixed"][:n]
        for kk in np.unique(res["k"][:n][fx]):
            m = fx & (res["k"][:n] == kk)
            b = agg["by_k"].setdefault(int(kk), [0, 0])
            b[0] += int(res["held"][:n][m].sum())
            b[1] += int(m.sum())
        agg["n"] += n
        total += n
        i += 1
        print(json.dumps({"split": name, "shard": i, "windows": total, "wall_s": round(time.time() - t0, 1),
                          "rss_anon_mb": round(rss_mb(), 1)}), flush=True)
        del res
    n = agg["n"]
    return {"windows": n, "shards": i, "episodes": agg["episodes"], "materials": mats,
            "held_star": wilson(agg["held"], n), "held_strict": wilson(agg["held_strict"], n),
            "slip_now": wilson(agg["slip"], n), "fell_if_off": wilson(agg["fell_if_off"], n),
            "self_abort_during_script": wilson(agg["lost_contact_during_script"], agg["episodes"]),
            "self_abort_detach": wilson(agg["lost_detach"], agg["episodes"]),
            "self_abort_slid_off": wilson(agg["lost_slid_off"], agg["episodes"]),
            "held_star_by_decision_step": {f"k{k}_t{k * meta_dt:.2f}s": wilson(*v) for k, v in sorted(agg["by_k"].items())},
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", choices=("legacy", "v4"), default="legacy")
    ap.add_argument("--n_train", type=int, default=200_000)
    ap.add_argument("--n_val", type=int, default=20_000)
    ap.add_argument("--n_dev_ood", type=int, default=20_000)
    ap.add_argument("--n_test", type=int, default=20_000)
    ap.add_argument("--chunk", type=int, default=2048, help="episodes per chunk (bounds peak RAM)")
    ap.add_argument("--extra", type=int, default=2, help="random intermediate decision points per episode")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", help="legacy split subset (default train,val,test)")
    ap.add_argument("--cfg", default="", help="ContactCfg overrides, e.g. acc_vib_slip=1.0,powered_detach_is_fall=0")
    args = ap.parse_args()
    v4 = args.protocol == "v4"
    if v4 and args.splits is not None:
        ap.error("v4 requires all four splits")
    out = Path(args.out)
    cfg_kw = parse_cfg(args.cfg)
    cfg_kw.setdefault("legacy_thrust", not v4)
    if v4:
        cfg_kw.setdefault("tw_ratio", (1.9, 2.6))   # match Q1 / x500_perch (nominal 2.26); v2/v3 used (2.6, 3.4)
    if v4 and (cfg_kw["legacy_thrust"] or {"num_envs", *SEEN} & cfg_kw.keys()):
        ap.error("v4 fixes the command convention, split materials and episode counts")
    sizes = {"train": args.n_train, "val": args.n_val, "dev_ood": args.n_dev_ood, "test": args.n_test}
    names = V4_SPLITS if v4 else (args.splits or "train,val,test").split(",")
    plans = {s: split_plan(s, sizes[s], args.chunk, args.seed) for s in names} if v4 else {}
    out.mkdir(parents=True, exist_ok=not v4)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cfg = ContactCfg(**cfg_kw)
    meta = {"git": git, "design": "v2 (RELAX/PULL, signed load cell, held_star = on plate & slip < 2 cm)",
            "protocol": args.protocol, "args": vars(args), "window": [WIN, len(cfg.sensor_channels)],
            "cfg_kw": cfg_kw,
            "cfg": {k: v for k, v in asdict(cfg).items() if k not in ("mu", "f_hold_mg", "num_envs")},
            "decision_steps": decision_steps(cfg), "splits": {}}
    if v4:
        meta["design"] = "v4 (nominal hover, separate development and final OOD episodes)"
        if cfg_kw.get("sensor_dr"):
            meta["design"] = "v5 (v4 episode protocol with per-episode sensor randomization)"
    if cfg.slip_sensor:
        meta["design"] = "v6 (C5 v2 optical slip displacement)"
        meta["channels"] = cfg.sensor_channels
    for s in names:
        plan = plans.get(s)
        if v4 and s in ("dev_ood", "test"):
            meta["splits"][s] = {**plan, "storage": "on_demand"}
        else:
            stats = gen_split(s, sizes[s], out, args, cfg_kw, plan)
            meta["splits"][s] = {**(plan or {}), **stats}
        if not v4:
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({s: {k: v[k] for k in ("episodes", "windows", "held_star") if k in v}
                      for s, v in meta["splits"].items()}))


if __name__ == "__main__":
    main()
