"""Q2 decision v3: adaptive probing schedule, conservative out-of-distribution variants and
the FSM baselines B4a / B4b, all evaluated on EPISODES of sim/contact_phase.py (design v2 physics).

Schedules (one decision point at the end of each macro):
  v2 : RELAX -> PULL0.3 -> PULL0.6 -> PULL0.9   (t = 0.15 / 0.40 / 0.65 / 0.90 s)
  v3 : RELAX -> PULL0.3 -> PULL0.6              (PULL0.9 dropped)
Episodes are simulated once with the v2 schedule (its prefix IS the v3 schedule), so every
policy below is evaluated on identical physics, noise and cart futures.

Policies (at each decision point d of the schedule, in this order):
  early stop   : p_slip >= s*  -> stop pulling, controlled ABORT (counted as ABORT; modelled as
                 an immediate ABORT at the decision point)
  motors off   : p_hold >= tau* (and d >= d_min) -> MOTORS_OFF
  otherwise    : next macro; after the last macro -> ABORT
  loss of contact while the motors run -> SELF_ABORT
D-thr variants: v2 schedule / v3 without early stop / v3 with early stop; conservative OOD
options: (a) ensemble of M nets, p_hold = min over members, p_slip = mean; (b) d_min = 1
(at least PULL0.3 survived before MOTORS_OFF).

FSM baselines (same sensors, same schedule; thresholds by grid search on VAL):
  B4a (Delft-like, no probing), evaluated at the end of RELAX:
      contact on continuously >= t_c (single-sample dropouts bridged) AND
      |mean f_normal over the last 50 ms| >= F_min AND max ||gyro|| over the last 0.1 s < g_max
      -> MOTORS_OFF, else ABORT.
  B4b (with probing), v3 schedule: slip detector at every decision point = energy of the
      first-difference (high-pass) of the tangential body accelerations (acc_x, acc_y) over the
      macro just finished (RELAX: last 15 samples, after the impact transient; PULL: 50 samples);
      energy >= e_thr -> controlled ABORT; no slip at any step -> MOTORS_OFF after PULL0.6.
Selection rule for every method: on VAL, among settings with false-safe (falls / MOTORS_OFF)
<= 1%, take the largest valid-perch; among settings within VALID_TOL = 1 point of it, the fewest
SELF_ABORTs, then the most conservative threshold.

Run: python q2/decide_v3.py --belief RUN0[,RUN1,...] --tag <tag>
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q2.belief_net import wilson  # noqa: E402
from q2.dataset import SEEN, UNSEEN  # noqa: E402
from q2.decide import at_touchdown, columns, summarize  # noqa: E402
from sim.contact_phase import (ABORT, ABORTED, FELL, MOTORS_OFF, OFF, PULL03, PULL06, PULL09, RELAX,  # noqa: E402
                               SELF_ABORT, ContactCfg, ContactPhaseSim, decision_steps)
from sim.quad_perch_env import G  # noqa: E402

FS_SELECT = 0.01
VALID_TOL = 0.01          # settings within 1 point of the best valid-perch count as ties -> fewest SELF_ABORTs
SCRIPT_V2 = [RELAX, PULL03, PULL06, PULL09]
SCRIPT_V3 = [RELAX, PULL03, PULL06]
FSM_KEYS = ("c_run_s", "fn_mean", "gyro_max", "hp_energy")


@torch.no_grad()
def fsm_features(w, d, cfg):
    """FSM features from raw windows w (E, 80, 11) at decision point index d."""
    c = w[:, :, 7]
    cf = torch.maximum(c, torch.roll(c, 1, dims=1))                 # bridge single-sample dropouts
    run = torch.flip(torch.cumprod(torch.flip(cf, [1]), 1), [1]).sum(1)
    n50, n100 = int(round(0.05 / cfg.dt)), int(round(0.1 / cfg.dt))
    L = 15 if d == 0 else int(round((cfg.ramp_s + cfg.pull_hold_s + cfg.pull_return_s) / cfg.dt))
    hp = torch.diff(w[:, -L:, 0:2], dim=1)
    return {"c_run_s": run * cfg.dt, "fn_mean": w[:, -n50:, 6].mean(1),
            "gyro_max": w[:, -n100:, 3:6].norm(dim=-1).max(1).values,
            "hp_energy": (hp ** 2).sum(-1).mean(1)}


@torch.no_grad()
def collect(models, n, mats, seed0, device, chunk, cfg_kw):
    """Episodes with the v2 schedule; per decision point: alive, p_hold/p_slip per model, labels, FSM features."""
    cfg_kw = {"legacy_thrust": True, **cfg_kw}
    M = len(models)
    keys = ("alive", "held", "strict", "fell") + FSM_KEYS
    acc = {k: [] for k in keys + ("p", "ps", "held_k0", "fell_k0", "strict_k0", "mu", "f_hold_mg", "tilt_deg",
                                  "slip_true")}
    done, ci = 0, 0
    while done < n:
        m = min(chunk, n - done)
        sim = ContactPhaseSim(ContactCfg(num_envs=m, **mats, **cfg_kw), device=device, seed=seed0 + ci)
        r0 = sim.rest_fork(sim.ar, sim.k, sim.thrust, sim.pos)
        ks = decision_steps(sim.cfg)
        rec = sim.run_script_to(ks[-1], record=True, script=SCRIPT_V2)
        D = len(ks)
        arr = {k: torch.zeros(m, D, device=sim.dev) for k in keys + ("slip_true",)}
        p = torch.zeros(M, m, D, device=sim.dev)
        ps = torch.zeros(M, m, D, device=sim.dev)
        for d, k in enumerate(ks):
            al = rec["alive"][:, k]
            e = sim.ar[al]
            arr["alive"][:, d] = al.float()
            if e.numel() == 0:
                continue
            kk = torch.full_like(e, k)
            w = sim.windows(e, kk).float()
            for j, mdl in enumerate(models):
                o = mdl(w)
                p[j, e, d], ps[j, e, d] = o[:, 0], o[:, 1]
            for fk, fv in fsm_features(w, d, sim.cfg).items():
                arr[fk][e, d] = fv
            r = sim.rest_fork(e, kk, rec["thrust"][e, k], rec["pos"][e, k])
            arr["held"][e, d], arr["strict"][e, d] = r["held"].float(), r["held_strict"].float()
            arr["fell"][e, d] = r["fell"].float()
            arr["slip_true"][e, d] = rec["slip"][e, k].float()
        for k in keys + ("slip_true",):
            acc[k].append(arr[k].cpu().numpy())
        acc["p"].append(p.cpu().numpy())
        acc["ps"].append(ps.cpu().numpy())
        acc["held_k0"].append(r0["held"].cpu().numpy())
        acc["strict_k0"].append(r0["held_strict"].cpu().numpy())
        acc["fell_k0"].append(r0["fell"].cpu().numpy())
        acc["mu"].append(sim.mu.cpu().numpy())
        acc["f_hold_mg"].append((sim.f_hold / G).cpu().numpy())
        acc["tilt_deg"].append(torch.rad2deg(sim.tilt).cpu().numpy())
        done += m
        ci += 1
        del sim, rec
    out = {k: np.concatenate(v, axis=1 if k in ("p", "ps") else 0) for k, v in acc.items()}
    for k in ("alive", "held", "strict", "fell", "slip_true"):
        out[k] = out[k].astype(bool)
    return out


def policy(ep, off_ok, stop, last):
    """Generic schedule policy. off_ok, stop: (N, D) bool; slip stop has priority; ABORT after `last`."""
    D = last + 1
    alive = ep["alive"][:, :D]
    ev_stop = alive & stop[:, :D]
    ev_off = alive & ~stop[:, :D] & off_ok[:, :D]
    ev = ev_stop | ev_off
    any_ev = ev.any(1)
    j = ev.argmax(1)
    i = np.arange(len(j))
    is_off = any_ev & ev_off[i, j]
    is_stop = any_ev & ~is_off
    reach = alive[:, last]
    pick = lambda a: np.where(is_off, a[i, j], False)
    return {"off": is_off, "held": pick(ep["held"]), "strict": pick(ep["strict"]), "fell_off": pick(ep["fell"]),
            "abort": is_stop | (~any_ev & reach), "self_abort": ~any_ev & ~reach,
            "d": np.where(is_off, j, -1), "early_stop": is_stop}


def rates(o):
    n, n_off = len(o["off"]), int(o["off"].sum())
    q = lambda x, d: float(x.sum()) / d if d else None
    return {"valid_perch": q(o["off"] & o["held"], n), "false_safe": q(o["off"] & o["fell_off"], n_off),
            "false_safe_held": q(o["off"] & ~o["held"], n_off),
            "false_safe_strict": q(o["off"] & ~o["strict"], n_off), "off_rate": q(o["off"], n), "abort": q(o["abort"], n),
            "self_abort": q(o["self_abort"], n), "n_off": n_off}


def select(cands):
    """cands: list of (params, rates). Rule in the module docstring."""
    ok = [c for c in cands if c[1]["n_off"] > 0 and c[1]["false_safe"] <= FS_SELECT]
    if not ok:
        return None
    best = max(c[1]["valid_perch"] for c in ok)
    near = [c for c in ok if c[1]["valid_perch"] >= best - VALID_TOL]
    return min(near, key=lambda c: (c[1]["self_abort"], -c[0].get("tau", 0), -c[0].get("conserv", 0)))


# ---------------------------------------------------------------- method definitions
def dthr_fn(comb, last, d_min=0, early=True):
    def f(ep, prm):
        p = ep["p"][0] if comb == "single" else ep["p"].min(0)
        ps = ep["ps"][0] if comb == "single" else ep["ps"].mean(0)
        off_ok = p >= prm["tau"]
        off_ok[:, :d_min] = False
        stop = ps >= prm["s"] if early else np.zeros_like(off_ok)
        return policy(ep, off_ok, stop, last)
    return f


def b4a(ep, prm):
    ok = (ep["c_run_s"] >= prm["t_c"] - 1e-9) & (np.abs(ep["fn_mean"]) >= prm["F_min"]) & (ep["gyro_max"] < prm["g_max"])
    return policy(ep, ok, np.zeros_like(ok), last=0)


def b4b(ep, prm, last=2):
    stop = ep["hp_energy"] >= prm["e_thr"]
    off_ok = np.zeros_like(stop)
    off_ok[:, last] = True
    return policy(ep, off_ok, stop, last=last)


TAUS = np.concatenate([np.linspace(0, 0.9, 46), np.linspace(0.9, 1.0, 101)[1:], [1.01]])
SS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]


def grids(ep_val):
    e = ep_val["hp_energy"][:, :3][ep_val["alive"][:, :3]]
    e_thr = np.unique(np.quantile(e, np.linspace(0.05, 0.995, 80)))
    return {
        "dthr_noearly": [{"tau": t, "s": 1.01} for t in TAUS],
        "dthr_early": [{"tau": t, "s": s} for t in TAUS for s in SS],
        "B4a": [{"t_c": a, "F_min": b, "g_max": c, "conserv": -c}
                for a, b, c in itertools.product([0.025, 0.05, 0.1, 0.15], [0, 2, 4, 6, 8, 10],
                                                 [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf])],
        "B4b": [{"e_thr": float(x), "conserv": -float(x)} for x in e_thr],
    }


def methods(M):
    m = {"D-thr_v2_schedule": (dthr_fn("single", 3, early=False), "dthr_noearly"),
         "D-thr_v3_no_early_stop": (dthr_fn("single", 2, early=False), "dthr_noearly"),
         "D-thr_v3_early_stop": (dthr_fn("single", 2), "dthr_early"),
         "D-thr_v3_early_stop_minPULL0.3": (dthr_fn("single", 2, d_min=1), "dthr_early")}
    if M > 1:
        m["D-thr_v3_early_stop_ensemble"] = (dthr_fn("ens", 2), "dthr_early")
        m["D-thr_v3_early_stop_ensemble_minPULL0.3"] = (dthr_fn("ens", 2, d_min=1), "dthr_early")
    m["B4a_FSM_no_probe"] = (b4a, "B4a")
    m["B4b_FSM_probe"] = (b4b, "B4b")
    return m


def mu_bins(ep, o):
    mu = ep["mu"]
    out = {}
    for nm, msk in (("mu_0.15-0.25", mu < 0.25 + 1e-6), ("mu_1.0-1.2", mu >= 1.0)):
        if msk.any():
            out[nm] = summarize(o, mask=msk)
    return out


def curve(fn, ep, params, key_order):
    rows = []
    for prm in params:
        r = rates(fn(ep, prm))
        rows.append({**{k: (float(prm[k]) if np.isfinite(prm[k]) else "inf") for k in key_order}, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}})
    return rows


def pareto(rows):
    """Upper-left frontier: for increasing false_safe, the best valid_perch so far."""
    rs = sorted((r for r in rows if r["n_off"]), key=lambda r: (r["false_safe"], -r["valid_perch"]))
    out, best = [], -1.0
    for r in rs:
        if r["valid_perch"] > best:
            out.append(r)
            best = r["valid_perch"]
    return out


def best_at(rows, fs):
    ok = [r for r in rows if r["n_off"] > 0 and r["false_safe"] <= fs]
    return max(ok, key=lambda r: r["valid_perch"]) if ok else None


@torch.no_grad()
def online_check(models, prm, n, mats, seed, device, cfg_kw, ep_ref, fn, comb, d_min, *, batch_n=None):
    """Genuine closed loop for the adaptive D-thr (v3 schedule, early stop, combination rule)."""
    cfg_kw = {"legacy_thrust": True, **cfg_kw}
    sim = ContactPhaseSim(ContactCfg(num_envs=batch_n or n, **mats, **cfg_kw), device=device, seed=seed)
    ks = decision_steps(sim.cfg)[:3]
    script = torch.tensor(SCRIPT_V3, device=sim.dev)
    for _ in range(ks[-1] + 1):
        r = sim.ready
        if r.any():
            e = sim.ar[r]
            a = torch.full((sim.N,), -1, dtype=torch.long, device=sim.dev)
            na = sim.n_actions[e]
            w = sim.windows(e, sim.k[e]).float()
            outs = torch.stack([mdl(w) for mdl in models])
            p = outs[0, :, 0] if comb == "single" else outs[:, :, 0].min(0).values
            ps = outs[0, :, 1] if comb == "single" else outs[:, :, 1].mean(0)
            d = na - 1                                          # decision index just reached
            nxt = script[na.clamp(max=len(SCRIPT_V3) - 1)]
            act = torch.where(ps >= prm["s"], ABORT,
                              torch.where((p >= prm["tau"]) & (d >= d_min), MOTORS_OFF,
                                          torch.where(na >= len(SCRIPT_V3), ABORT, nxt)))
            a[e] = torch.where(na == 0, script[0], act)
            sim.act(a)
        if (sim.status == 0).any() and sim.k.max() < ks[-1]:
            sim.step()
    st = sim.status[:n].cpu().numpy()
    off = st == OFF
    on = {"off": off, "held": sim.off_held[:n].cpu().numpy() & off,
          "strict": sim.off_held_strict[:n].cpu().numpy() & off, "fell_off": sim.off_fell[:n].cpu().numpy() & off,
          "d": np.where(off, np.searchsorted(ks, sim.off_k[:n].cpu().numpy()), -1),
          "abort": st == ABORTED, "self_abort": (st == SELF_ABORT) | (st == FELL)}
    sub = {k: (v[:, :n] if k in ("p", "ps") else v[:n]) for k, v in ep_ref.items()}
    ref = fn(sub, prm)
    agree = np.mean(np.all(np.stack([on[k] == ref[k] for k in on]), 0))
    return {"n": n, "episode_agreement": round(float(agree), 5), "all_identical": bool(agree == 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--belief", required=True, help="comma-separated run dirs with belief.ts (first = single model)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_val", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n_online", type=int, default=1024)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirs = [Path(d) for d in args.belief.split(",")]
    models = [torch.jit.load(str(d / "belief.ts"), map_location=args.device).eval() for d in dirs]
    bmeta = json.loads((dirs[0] / "belief.json").read_text())
    dmeta_p = Path(bmeta["dataset"]) / "meta.json"
    cfg_kw = json.loads(dmeta_p.read_text()).get("cfg_kw", {}) if dmeta_p.exists() else {}
    seeds = {"val": 5_000_000 + args.seed * 10_000, "test": 6_000_000 + args.seed * 10_000}
    ep = {"val": collect(models, args.n_val, SEEN, seeds["val"], args.device, args.chunk, cfg_kw),
          "test": collect(models, args.n_test, UNSEEN, seeds["test"], args.device, args.chunk, cfg_kw)}
    G_ = grids(ep["val"])
    res_eval = {"val": {}, "test": {}}
    chosen, curves = {}, {}
    for name, (fn, gk) in methods(len(models)).items():
        cands = [(prm, rates(fn(ep["val"], prm))) for prm in G_[gk]]
        sel = select(cands)
        if sel is None:
            chosen[name] = None
            continue
        prm = sel[0]
        chosen[name] = {k: (float(v) if np.isfinite(v) else "inf") for k, v in prm.items() if k != "conserv"}
        for s in ("val", "test"):
            o = fn(ep[s], prm)
            r = summarize(o)
            r["early_stop"] = wilson(int(o.get("early_stop", np.zeros(1, bool)).sum()), len(o["off"]))
            r["bins_mu"] = mu_bins(ep[s], o)
            res_eval[s][name] = r
        # risk-performance curves: sweep the main threshold at the chosen secondary parameters
        if gk.startswith("dthr"):
            sweep = [{**prm, "tau": t} for t in TAUS]
            keys = ["tau", "s"]
        elif gk == "B4a":
            sweep, keys = G_["B4a"], ["t_c", "F_min", "g_max"]
        else:
            sweep, keys = G_["B4b"], ["e_thr"]
        curves[name] = {s: curve(fn, ep[s], sweep, keys) for s in ("val", "test")}
    # baselines
    N = {s: len(ep[s]["mu"]) for s in ep}
    for s in ep:
        res_eval[s]["B7_oracle_at_touchdown"] = summarize(at_touchdown(ep[s], ep[s]["held_k0"]))
        res_eval[s]["B7_oracle_on_v3_schedule"] = summarize(policy(ep[s], ep[s]["held"], np.zeros_like(ep[s]["held"]), 2))
        res_eval[s]["always_off_at_touchdown_no_check"] = summarize(at_touchdown(ep[s], np.ones(N[s], bool)))
        res_eval[s]["B7_oracle_at_touchdown"]["bins_mu"] = mu_bins(ep[s], at_touchdown(ep[s], ep[s]["held_k0"]))
    # H2: D-thr vs FSM at the same false-safe, read off the TEST curves (both sides treated alike)
    h2 = {}
    for name in curves:
        h2[name] = {f"fs<={fs}": best_at(curves[name]["test"], fs) for fs in (0.005, 0.01, 0.02, 0.05)}
    # slip detector quality (D-thr p_slip vs FSM energy) at the v3 decision points
    al = ep["test"]["alive"][:, :3]
    from q2.belief_net import auroc
    yt = torch.from_numpy(ep["test"]["slip_true"][:, :3][al])
    slip_q = {"auroc_p_slip_single": auroc(torch.from_numpy(ep["test"]["ps"][0][:, :3][al]), yt),
              "auroc_fsm_hp_energy": auroc(torch.from_numpy(ep["test"]["hp_energy"][:, :3][al]), yt)}
    # online closed-loop check of the main adaptive policy
    main_name = "D-thr_v3_early_stop_ensemble" if len(models) > 1 else "D-thr_v3_early_stop"
    chk = None
    if chosen.get(main_name) and args.n_online > 0:
        fn = methods(len(models))[main_name][0]
        prm = {k: (np.inf if v == "inf" else v) for k, v in chosen[main_name].items()}
        chk = online_check(models, prm, min(args.n_online, args.n_test, args.chunk), UNSEEN, seeds["test"],
                           args.device, cfg_kw, ep["test"], fn, "ens" if len(models) > 1 else "single", 0,
                           batch_n=min(args.n_test, args.chunk))
        chk["policy"] = main_name
    name = f"q2_decide_v3_{args.tag}"
    res = {"name": name, "phase": "P3", "design": "v3 (adaptive schedule, OOD-conservative, FSM B4a/B4b)",
           "git": git, "seed": args.seed,
           "config": {**vars(args), "fs_select_on_val": FS_SELECT, "valid_tol": VALID_TOL, "cfg_kw": cfg_kw, "n_models": len(models),
                      "ensemble_rule": "p_hold = min over members, p_slip = mean",
                      "decision_points_s": [round(k * ContactCfg().dt, 3) for k in decision_steps(ContactCfg())],
                      "materials": {"val": SEEN, "test_unseen": UNSEEN}, "episode_seeds": seeds},
           "chosen_on_val": chosen, "eval": res_eval, "h2_same_false_safe_test": h2, "slip_detection_test": slip_q,
           "online_check": chk, "wall_min": round((time.time() - t0) / 60, 2)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    (ROOT / "results" / f"{name}_curve.json").write_text(json.dumps(
        {"name": name, "x": "false_safe (falls / MOTORS_OFF)", "y": "valid_perch",
         "curves": {k: {s: columns(v) for s, v in c.items()} for k, c in curves.items()},
         "pareto_B4a": {s: columns(pareto(curves["B4a_FSM_no_probe"][s])) for s in ("val", "test")}
         if "B4a_FSM_no_probe" in curves else None}, separators=(",", ":"), default=float))
    pr = lambda r: {k: r[k]["p"] for k in ("valid_perch", "false_safe", "abort", "self_abort")}
    print(json.dumps({"chosen": chosen, "test": {k: pr(v) for k, v in res_eval["test"].items()},
                      "online_check": chk, "slip": slip_q}, indent=1, default=float))


if __name__ == "__main__":
    main()
