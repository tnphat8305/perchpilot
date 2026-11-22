"""Compare the C5 contact-phase signals measured in Gazebo/PX4 SITL with sim/contact_phase.py at the same material.

SITL side: q2_samples.jsonl (perch_node --q2-log, 200 Hz C5 rows) + truth_<k>.jsonl (HoldPad hold_truth: engaged,
margin, slip speed) of a `--verify-rule probe` batch (full RELAX -> PULL0.3 -> PULL0.6 -> PULL0.9 schedule).
Sim side: ContactPhaseSim with mu, F_hold, tilt, cart speed and TW fixed to the trial's values, same schedule.

Per macro segment (RELAX, PULL0.3, PULL0.6, PULL0.9) and per trial / sim batch:
  f_end    mean load cell [N] over the last 50 ms of RELAX, or over the PULL hold (samples 15..39 of the macro)
  hp       B4b feature: mean squared first difference of (acc_x, acc_y) over the segment (RELAX: last 15 samples)
  acc_t    std of (acc_x, acc_y) over the segment [m/s^2]
  gyro     rms body rate [rad/s]
  slip     SITL: mean HoldPad slip speed [m/s] (truth); sim: fraction of steps with margin < 0
  contact  fraction of samples with the contact flag set

    python perchpilot/ros/q2_signal_check.py /home/sup/data/perchpilot/sitl_eval/<probe run> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MACROS = [(1, 0.1, "RELAX"), (2, 0.3, "PULL0.3"), (2, 0.6, "PULL0.6"), (2, 0.9, "PULL0.9")]
G = 9.81
MASS = 2.0643


def seg_stats(X, name):
    if len(X) < (15 if name == "RELAX" else 40):
        return None
    if name == "RELAX":
        fwin, hpwin = X[-10:, 6], X[-15:]
    else:
        fwin, hpwin = X[15:40, 6], X
    hp = np.diff(hpwin[:, 0:2], axis=0)
    a_t = X[:, 0:2] - X[:, 0:2].mean(0)
    return {"n": int(len(X)), "f_end": float(fwin.mean()), "hp": float((hp ** 2).sum(-1).mean()),
            "acc_t": float(np.sqrt((a_t ** 2).sum(-1).mean())), "acc_mean": X[:, 0:3].mean(0).round(3).tolist(),
            "gyro": float(np.sqrt((X[:, 3:6] ** 2).sum(-1).mean())), "contact": float(X[:, 7].mean())}


def segments(X):
    """Split a contact-phase sample array by (phase_id, probe_level) runs -> {macro name: rows}."""
    out = {}
    for ph, lv, name in MACROS:
        m = (np.abs(X[:, 9] - ph) < 0.1) & (np.abs(X[:, 10] - lv) < 0.05)
        idx = np.nonzero(m)[0]
        if len(idx):
            run = idx[: np.argmax(np.diff(np.r_[idx, 10 ** 9]) > 1) + 1]   # first contiguous run
            out[name] = run
    return out


def sitl_trials(run_dir):
    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / "q2_samples.jsonl").open()]
    trials = [json.loads(line) for line in (run_dir / "trials.jsonl").open()]
    res = []
    for tr in trials:
        k = tr["trial"]
        truth = [json.loads(line) for line in (run_dir / f"truth_{k}.jsonl").open()]
        w0, w1 = truth[0]["wall"], truth[-1]["wall"]          # node trial ids restart after a SITL restart
        R = [r for r in rows if w0 <= r["wall"] <= w1]
        if not R:
            continue
        X = np.array([r["x"] for r in R])
        W = np.array([r["wall"] for r in R])
        hold = [(t["wall"], t["engaged"], t["margin_n"], t["slip_mps"]) for t in truth if t["kind"] == "hold"]
        load = [(t["wall"], t["servo_n"], t["collision_n"]) for t in truth if t["kind"] == "load"]
        lw = np.array([x[0] for x in load]) if load else np.zeros(0)
        hw = np.array([h[0] for h in hold]) if hold else np.zeros(0)
        segs = segments(X)
        st = {}
        for name, idx in segs.items():
            s = seg_stats(X[idx], name)
            if s is None:
                continue
            if len(hw):
                m = (hw >= W[idx[0]]) & (hw <= W[idx[-1]])
                if m.any():
                    s["slip"] = float(np.mean([hold[i][3] for i in np.nonzero(m)[0]]))
                    s["engaged"] = float(np.mean([hold[i][1] for i in np.nonzero(m)[0]]))
                    s["margin_min_N"] = float(np.min([hold[i][2] for i in np.nonzero(m)[0]]))
            if len(lw):
                m = (lw >= W[idx[-1]] - 0.05) & (lw <= W[idx[-1]])
                if m.any():
                    s["servo_n"] = float(np.mean([load[i][1] for i in np.nonzero(m)[0]]))
                    s["collision_n"] = float(np.mean([load[i][2] for i in np.nonzero(m)[0]]))
            st[name] = s
        res.append({"trial": k, "mu": tr["mu"], "f_hold_n": tr["f_hold_n"], "outcome": tr["outcome"],
                    "crash_reason": tr["crash_reason"], "valid_perch": tr.get("valid_perch"),
                    "q2": (tr.get("q2") or {}).get("decision"), "touch": tr.get("touch"), "segments": st})
    return res


def sim_stats(mu, f_hold_n, n=256, tilt=65.0, tw=2.26, seed=0, sensor_dr=False):
    import torch
    from sim.contact_phase import SCRIPT, ContactCfg, ContactPhaseSim, decision_steps
    fm = f_hold_n / (MASS * G)
    cfg = ContactCfg(num_envs=n, mu=((mu, mu + 1e-6),), f_hold_mg=((fm, fm + 1e-6),),
                     plate_tilt_deg=(tilt, tilt + 1e-6), veh_speed=(0.0, 1e-6), tw_ratio=(tw, tw + 1e-6),
                     sensor_dr=sensor_dr)
    sim = ContactPhaseSim(cfg, device="cpu", seed=seed)
    ks = decision_steps(sim.cfg)
    rec = sim.run_script_to(ks[-1], record=True, script=SCRIPT)
    H = sim.hist[:, sim.pre + 1: sim.pre + ks[-1] + 1].numpy()
    slip = rec["slip"][:, 1:].numpy()
    alive = rec["alive"][:, 1:].numpy()
    agg = {}
    for name in ("RELAX", "PULL0.3", "PULL0.6", "PULL0.9"):
        vals = []
        for e in range(n):
            segs = segments(H[e])
            if name not in segs:
                continue
            idx = segs[name]
            if not alive[e, idx].all():
                continue
            s = seg_stats(H[e][idx], name)
            if s:
                s["slip"] = float(slip[e, idx].mean())
                vals.append(s)
        if vals:
            agg[name] = {k: float(np.mean([v[k] for v in vals])) for k in ("f_end", "hp", "acc_t", "gyro", "contact",
                                                                            "slip")}
            agg[name]["n_alive"] = len(vals)
            agg[name]["slip_frac_envs"] = float(np.mean([v["slip"] > 0 for v in vals]))
            agg[name]["p05_p95"] = {
                k: np.quantile([v[k] for v in vals], (0.05, 0.95)).tolist() for k in ("f_end", "hp", "gyro")}
    from sim.contact_phase import STATUS_NAMES
    st = sim.status.numpy()
    agg["status_after_schedule"] = {STATUS_NAMES[i]: int((st == i).sum()) for i in range(len(STATUS_NAMES))}
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n-sim", type=int, default=256)
    ap.add_argument("--sensor-dr", action="store_true", help="use v5 sensor randomization for the sim reference")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    sitl = sitl_trials(a.run_dir)
    mats = sorted({(t["mu"], t["f_hold_n"]) for t in sitl})
    sim = {f"{mu}_{f}": sim_stats(mu, f, a.n_sim, sensor_dr=a.sensor_dr) for mu, f in mats}
    out = {"sensor_dr": a.sensor_dr, "n_sim_per_material": a.n_sim, "sitl": sitl, "sim": sim}
    for t in sitl:
        s = sim[f"{t['mu']}_{t['f_hold_n']}"]
        print(f"trial {t['trial']} mu={t['mu']} F={t['f_hold_n']} N -> {t['outcome']} {t['crash_reason']} q2={t['q2']}")
        for name in ("RELAX", "PULL0.3", "PULL0.6", "PULL0.9"):
            g = t["segments"].get(name)
            m = s.get(name)
            gs = (f"f={g['f_end']:7.2f} hp={g['hp']:.4f} acc_t={g['acc_t']:.3f} gyro={g['gyro']:.3f} "
                  f"slip={g.get('slip', float('nan')):.3f} eng={g.get('engaged', float('nan')):.2f} "
                  f"servo={g.get('servo_n', float('nan')):.2f} coll={g.get('collision_n', float('nan')):.2f}") if g else "-"
            ms = (f"f={m['f_end']:7.2f} hp={m['hp']:.4f} acc_t={m['acc_t']:.3f} gyro={m['gyro']:.3f} "
                  f"slipfrac={m['slip']:.2f} alive={m['n_alive']}") if m else "-"
            print(f"   {name:8s} SITL {gs}\n            SIM  {ms}")
        print("   sim status after schedule:", s["status_after_schedule"])
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
