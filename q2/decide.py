"""Q2 decision rule D-thr + baselines, evaluated on EPISODES of sim/contact_phase.py (design v2).

D-thr: run the schedule RELAX -> PULL0.3 -> PULL0.6 -> PULL0.9; at the end of each macro
(decision points t = 0.15/0.40/0.65/0.90 s) MOTORS_OFF if p_hold >= tau, else continue; after
PULL0.9 without reaching tau -> ABORT. A loss of contact while the motors run (detach or pad
slid off during RELAX/PULL) is a SELF_ABORT (failed perch, not a fall).

Because the schedule up to the stopping point does not depend on tau, one scripted run per
episode gives p_hold and the motors-off outcome (rest fork, identical to stepping MOTORS_OFF)
at every decision point; every tau is evaluated exactly from these arrays. A genuine
step-by-step closed loop with the belief net in the loop is run for tau* on the first TEST
chunk and must agree with the offline evaluation (`online_check`).

Metrics (proportions with Wilson 95%):
  valid_perch       = MOTORS_OFF and held_star                         / episodes
  false_safe        = MOTORS_OFF and falls within the 10 s REST        / MOTORS_OFF   (PRIMARY)
  false_safe_held   = MOTORS_OFF and not held_star (slip >= 2 cm/fall) / MOTORS_OFF
  false_safe_strict = MOTORS_OFF and margin < 0 at some step           / MOTORS_OFF   (v1 label)
  fall              = MOTORS_OFF and falls / episodes (falls only happen after MOTORS_OFF)
  abort, self_abort = ABORT after the schedule / loss of contact with motors on, per episode
tau* = largest valid_perch on VAL with false_safe <= 1% (safety margin under the 2% target,
because v1 showed the threshold does not transfer well to unseen materials).

Baselines: B7 oracle (knows mu, f_hold: MOTORS_OFF right at touchdown iff held_star, else ABORT),
B7 on the schedule (perfect p_hold at the 4 decision points), B7-static (knows mu, f_hold and the
CURRENT cart acceleration only), always MOTORS_OFF at touchdown (no confirmation), always OFF
after RELAX, and the no-PULL ablation of D-thr (decide only after RELAX, else ABORT).

Run: python q2/decide.py --belief /home/sup/data/perchpilot/runs/q2_belief_<tag> --tag <tag>
"""
import argparse
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
from sim.contact_phase import (ABORT, ABORTED, FELL, MOTORS_OFF, OFF, SCRIPT, SELF_ABORT,  # noqa: E402
                               ContactCfg, ContactPhaseSim, decision_steps)
from sim.quad_perch_env import G  # noqa: E402

FS_TARGET = 0.02
FS_SELECT = 0.01          # tau* is chosen on VAL with this margin under the target


@torch.no_grad()
def collect(model, n, mats, seed0, device, chunk, cfg_kw):
    """Scripted episodes; per decision point: alive, p_hold, held_star, strict, fell_if_off, static oracle."""
    cfg_kw = {"legacy_thrust": True, **cfg_kw}
    keys = ("alive", "p", "held", "strict", "fell", "static_ok")
    acc = {k: [] for k in keys + ("held_k0", "strict_k0", "fell_k0", "mu", "f_hold_mg", "tilt_deg",
                                  "loss_k", "loss_cause")}
    done, ci = 0, 0
    while done < n:
        m = min(chunk, n - done)
        sim = ContactPhaseSim(ContactCfg(num_envs=m, **mats, **cfg_kw), device=device, seed=seed0 + ci)
        r0 = sim.rest_fork(sim.ar, sim.k, sim.thrust, sim.pos)             # motors off right at touchdown
        ks = decision_steps(sim.cfg)
        rec = sim.run_script_to(ks[-1], record=True)
        D = len(ks)
        arr = {k: torch.zeros(m, D, device=sim.dev) for k in keys}
        for d, k in enumerate(ks):
            al = rec["alive"][:, k]
            e = sim.ar[al]
            kk = torch.full_like(e, k)
            arr["alive"][:, d] = al.float()
            if e.numel() == 0:
                continue
            arr["p"][e, d] = model(sim.windows(e, kk).float())[:, 0]
            r = sim.rest_fork(e, kk, rec["thrust"][e, k], rec["pos"][e, k])
            arr["held"][e, d] = r["held"].float()
            arr["strict"][e, d] = r["held_strict"].float()
            arr["fell"][e, d] = r["fell"].float()
            arr["static_ok"][e, d] = (sim.static_margin_off(e, kk) >= 0).float()
        for k in keys:
            acc[k].append(arr[k].cpu().numpy())
        acc["held_k0"].append(r0["held"].cpu().numpy())
        acc["strict_k0"].append(r0["held_strict"].cpu().numpy())
        acc["fell_k0"].append(r0["fell"].cpu().numpy())
        acc["mu"].append(sim.mu.cpu().numpy())
        acc["f_hold_mg"].append((sim.f_hold / G).cpu().numpy())
        acc["tilt_deg"].append(torch.rad2deg(sim.tilt).cpu().numpy())
        acc["loss_k"].append(sim.loss_k.cpu().numpy())
        acc["loss_cause"].append(sim.loss_cause.cpu().numpy())
        done += m
        ci += 1
        del sim, rec
    out = {k: np.concatenate(v) for k, v in acc.items()}
    for k in keys:
        out[k] = out[k].astype(np.float32 if k == "p" else bool)
    return out


def outcome(ep, cut, last=None):
    """cut: (N, D) bool 'would cut at decision point d'; `last` = index of the last decision point
    of the schedule (ABORT right after it; default: the full schedule). Returns per-episode outcomes."""
    alive = ep["alive"]
    go = alive & cut
    off = go.any(1)
    d_star = np.where(off, go.argmax(1), -1)
    i = np.arange(len(d_star))
    pick = lambda a: np.where(off, a[i, d_star.clip(0)], False)
    last = alive.shape[1] - 1 if last is None else last
    reach_end = alive[:, last]
    return {"off": off, "held": pick(ep["held"]), "strict": pick(ep["strict"]), "fell_off": pick(ep["fell"]),
            "abort": ~off & reach_end, "self_abort": ~off & ~reach_end, "d": d_star}


def at_touchdown(ep, cut0):
    """Policies that decide at k = 0 (no probing): MOTORS_OFF where cut0, else ABORT."""
    z = np.zeros(len(cut0), bool)
    return {"off": cut0, "held": cut0 & ep["held_k0"], "strict": cut0 & ep["strict_k0"],
            "fell_off": cut0 & ep["fell_k0"], "abort": ~cut0, "self_abort": z, "d": np.full(len(cut0), -1)}


def summarize(o, mask=None):
    if mask is not None:
        o = {k: v[mask] for k, v in o.items()}
    n = len(o["off"])
    n_off = int(o["off"].sum())
    return {"episodes": n,
            "valid_perch": wilson(int((o["off"] & o["held"]).sum()), n),
            "false_safe": wilson(int((o["off"] & o["fell_off"]).sum()), n_off),
            "false_safe_held": wilson(int((o["off"] & ~o["held"]).sum()), n_off),
            "false_safe_strict": wilson(int((o["off"] & ~o["strict"]).sum()), n_off),
            "fall": wilson(int((o["off"] & o["fell_off"]).sum()), n),
            "abort": wilson(int(o["abort"].sum()), n),
            "self_abort": wilson(int(o["self_abort"].sum()), n),
            "motors_off": wilson(n_off, n),
            "mean_decision_point": round(float(o["d"][o["off"]].mean()), 3) if n_off else None}


def dthr_cut(ep, tau, points=None):
    cut = ep["p"] >= tau
    if points is not None:
        m = np.zeros(cut.shape[1], bool)
        m[points] = True
        cut = cut & m
    return cut


def sweep(ep, taus, points=None):
    rows = []
    for t in taus:
        o = outcome(ep, dthr_cut(ep, t, points), last=None if points is None else max(points))
        n_off, n = int(o["off"].sum()), len(o["off"])
        q = lambda x, d: round(float(x.sum()) / d, 5) if d else None
        rows.append({"tau": round(float(t), 4), "valid_perch": q(o["off"] & o["held"], n),
                     "false_safe": q(o["off"] & o["fell_off"], n_off),
                     "false_safe_held": q(o["off"] & ~o["held"], n_off),
                     "false_safe_strict": q(o["off"] & ~o["strict"], n_off),
                     "fall": q(o["off"] & o["fell_off"], n), "abort": q(o["abort"], n),
                     "self_abort": q(o["self_abort"], n), "n_off": n_off})
    return rows


def columns(rows):
    """List of row dicts -> dict of columns (compact JSON for the risk-performance curve)."""
    return {k: [r[k] for r in rows] for k in rows[0]}


def pick_tau(rows, key="false_safe", thr=FS_SELECT):
    ok = [r for r in rows if r["n_off"] > 0 and r[key] <= thr]
    if not ok:
        return 1.01
    best = max(r["valid_perch"] for r in ok)
    return max(r["tau"] for r in ok if r["valid_perch"] == best)    # ties -> the safer (larger) tau


def baselines(ep):
    N, D = ep["alive"].shape
    first = np.zeros((N, D), bool)
    first[:, 0] = True
    return {"B7_oracle_at_touchdown": at_touchdown(ep, ep["held_k0"]),
            "B7_oracle_on_schedule": outcome(ep, ep["held"]),
            "B7_static_oracle_no_future": outcome(ep, ep["static_ok"]),
            "always_off_at_touchdown_no_check": at_touchdown(ep, np.ones(N, bool)),
            "always_off_after_RELAX": outcome(ep, first)}


def bins(ep, o):
    mu, fh, tl = ep["mu"], ep["f_hold_mg"], ep["tilt_deg"]
    out = {}
    for nm, m in (("mu_low", mu < 0.5), ("mu_high", mu >= 0.5), ("fhold_low", fh < 1.0), ("fhold_high", fh >= 1.0),
                  ("tilt_55-62", tl < 62), ("tilt_62-70", tl >= 62)):
        if m.any():
            out[nm] = summarize(o, mask=m)
    return out


@torch.no_grad()
def online_check(model, tau, n, mats, seed, device, cfg_kw, ep_ref, *, batch_n=None):
    """Genuine closed loop: belief net queried at each ready point, actions issued step by step."""
    cfg_kw = {"legacy_thrust": True, **cfg_kw}
    sim = ContactPhaseSim(ContactCfg(num_envs=batch_n or n, **mats, **cfg_kw), device=device, seed=seed)
    ks = decision_steps(sim.cfg)
    script = torch.tensor(SCRIPT, device=sim.dev)
    for _ in range(ks[-1] + 1):
        r = sim.ready
        if r.any():
            e = sim.ar[r]
            a = torch.full((sim.N,), -1, dtype=torch.long, device=sim.dev)
            first = sim.n_actions[e] == 0
            p = model(sim.windows(e, sim.k[e]).float())[:, 0]
            nxt = script[sim.n_actions[e].clamp(max=len(SCRIPT) - 1)]
            act = torch.where(p >= tau, MOTORS_OFF, torch.where(sim.n_actions[e] >= len(SCRIPT), ABORT, nxt))
            a[e] = torch.where(first, script[0], act)              # touchdown: always start the schedule
            sim.act(a)
        if (sim.status == 0).any() and sim.k.max() < ks[-1]:
            sim.step()
    st = sim.status[:n].cpu().numpy()
    off = st == OFF
    on = {"off": off, "held": sim.off_held[:n].cpu().numpy() & off, "strict": sim.off_held_strict[:n].cpu().numpy() & off,
          "fell_off": sim.off_fell[:n].cpu().numpy() & off,
          "d": np.where(off, np.searchsorted(ks, sim.off_k[:n].cpu().numpy()), -1), "abort": st == ABORTED,
          "self_abort": (st == SELF_ABORT) | (st == FELL)}
    sub = {k: v[:n] for k, v in ep_ref.items()}
    ref = outcome(sub, dthr_cut(sub, tau))
    per_ep = np.mean(np.all(np.stack([on[k] == ref[k] for k in on]), 0))
    return {"n": n, "all_fields_identical": bool(per_ep == 1.0), "episode_agreement": round(float(per_ep), 5),
            "online": summarize(on)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--belief", required=True, help="run dir with belief.ts")
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
    model = torch.jit.load(str(Path(args.belief) / "belief.ts"), map_location=args.device).eval()
    # the sim must use the same sensor/contact config as the dataset the belief net was trained on
    bmeta = json.loads((Path(args.belief) / "belief.json").read_text())
    dmeta_p = Path(bmeta["dataset"]) / "meta.json"
    cfg_kw = json.loads(dmeta_p.read_text()).get("cfg_kw", {}) if dmeta_p.exists() else {}
    seeds = {"val": 5_000_000 + args.seed * 10_000, "test": 6_000_000 + args.seed * 10_000}
    ep = {"val": collect(model, args.n_val, SEEN, seeds["val"], args.device, args.chunk, cfg_kw),
          "test": collect(model, args.n_test, UNSEEN, seeds["test"], args.device, args.chunk, cfg_kw)}
    taus = np.concatenate([np.linspace(0, 0.9, 91), np.linspace(0.9, 1.0, 201)[1:], [1.01]])
    curves = {s: sweep(ep[s], taus) for s in ep}
    tau = pick_tau(curves["val"])                                   # primary: falls, <= 1% on val
    tau_2 = pick_tau(curves["val"], thr=FS_TARGET)                  # no safety margin (reference)
    tau_held = pick_tau(curves["val"], key="false_safe_held")       # held_star-based, <= 1%
    curves_np = {s: sweep(ep[s], taus, points=[0]) for s in ep}
    tau_np = pick_tau(curves_np["val"])
    res_eval = {}
    for s in ep:
        o = outcome(ep[s], dthr_cut(ep[s], tau))
        r = {"D-thr": summarize(o),
             "D-thr_tau_2pct_no_margin": summarize(outcome(ep[s], dthr_cut(ep[s], tau_2))),
             "D-thr_tau_held_star_based": summarize(outcome(ep[s], dthr_cut(ep[s], tau_held))),
             "D-thr_noPULL": summarize(outcome(ep[s], dthr_cut(ep[s], tau_np, points=[0]), last=0))}
        for nm, ob in baselines(ep[s]).items():
            r[nm] = summarize(ob)
        r["held_star_at_decision_points"] = [wilson(int((ep[s]["held"][:, d] & ep[s]["alive"][:, d]).sum()),
                                                    int(ep[s]["alive"][:, d].sum())) for d in range(4)]
        r["alive_at_decision_points"] = [wilson(int(ep[s]["alive"][:, d].sum()), len(ep[s]["alive"])) for d in range(4)]
        r["held_star_at_touchdown"] = wilson(int(ep[s]["held_k0"].sum()), len(ep[s]["held_k0"]))
        if s == "test":
            r["bins_D-thr"] = bins(ep[s], o)
            r["bins_B7_oracle_at_touchdown"] = bins(ep[s], at_touchdown(ep[s], ep[s]["held_k0"]))
        res_eval[s] = r
    chk = online_check(model, tau, min(args.n_online, args.n_test, args.chunk), UNSEEN, seeds["test"], args.device,
                       cfg_kw, ep["test"], batch_n=min(args.n_test, args.chunk))
    name = f"q2_decide_{args.tag}"
    dts = ContactCfg(**cfg_kw).dt
    res = {"name": name, "phase": "P3", "design": "v2", "git": git, "seed": args.seed,
           "config": {**vars(args), "fs_target": FS_TARGET, "fs_select_on_val": FS_SELECT, "cfg_kw": cfg_kw,
                      "belief_label": bmeta.get("label", "held"), "belief_drop": bmeta.get("drop", []),
                      "schedule": ["RELAX", "PULL0.3", "PULL0.6", "PULL0.9"],
                      "decision_points_s": [round(k * dts, 3) for k in decision_steps(ContactCfg(**cfg_kw))],
                      "materials": {"val": SEEN, "test_unseen": UNSEEN}, "episode_seeds": seeds},
           "tau_star": tau, "tau_2pct_no_margin": tau_2, "tau_held_star_based": tau_held, "tau_noPULL": tau_np,
           "eval": res_eval, "online_check": chk, "wall_min": round((time.time() - t0) / 60, 2)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(res, indent=1))
    (ROOT / "results" / f"{name}_curve.json").write_text(json.dumps(
        {"name": name, "x": "false_safe (falls / MOTORS_OFF)", "y": "valid_perch", "tau_star": tau,
         "D-thr": {s: columns(v) for s, v in curves.items()},
         "D-thr_noPULL": {s: columns(v) for s, v in curves_np.items()}}, separators=(",", ":")))
    pr = lambda r: {k: r[k]["p"] for k in ("valid_perch", "false_safe", "false_safe_held", "abort", "self_abort")}
    print(json.dumps({"tau*": tau, "test": {k: pr(v) for k, v in res_eval["test"].items()
                                           if isinstance(v, dict) and "valid_perch" in v},
                      "online_check": {k: chk[k] for k in ("all_fields_identical", "episode_agreement")}}, indent=1))


if __name__ == "__main__":
    main()
