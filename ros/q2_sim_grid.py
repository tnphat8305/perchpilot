"""Sim reference for the SITL Q2 batches: q2/decide_v3 policies on sim/contact_phase.py episodes at the SAME
3x3 material grid, tilt 65 deg, static cart, TW 2.26 (CPU). Prints valid perch / false-safe (fall) / ABORT /
SELF_ABORT per rule and per material.

    python perchpilot/ros/q2_sim_grid.py [--n 256] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q2.decide_v3 import b4b, collect, dthr_fn  # noqa: E402

MASS, G = 2.0643, 9.81


def rule_rates(o):
    n, n_off = len(o["off"]), int(o["off"].sum())
    return {"n": n, "valid_perch": int((o["off"] & o["held"]).sum()), "off": n_off,
            "fell_after_off": int((o["off"] & o["fell_off"]).sum()), "abort": int(o["abort"].sum()),
            "self_abort": int(o["self_abort"].sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--belief", default="/home/sup/data/perchpilot/runs/q2_belief_v2_small")
    ap.add_argument("--decide-json", default=str(ROOT / "results" / "q2_decide_v3_v2_small_ens5.json"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    chosen = json.loads(Path(a.decide_json).read_text())["chosen_on_val"]
    model = torch.jit.load(str(Path(a.belief) / "belief.ts"), map_location="cpu").eval()
    cfg_kw = {"plate_tilt_deg": (65.0, 65.0 + 1e-6), "veh_speed": (0.0, 1e-6), "tw_ratio": (2.26, 2.26 + 1e-6)}
    rules = {
        "dthr": lambda ep: dthr_fn("single", 2)(ep, chosen["D-thr_v3_early_stop"]),
        "dthr_min": lambda ep: dthr_fn("single", 2, d_min=1)(ep, chosen["D-thr_v3_early_stop_minPULL0.3"]),
        "fsm_b4b": lambda ep: b4b(ep, chosen["B4b_FSM_probe"]),
    }
    out = {}
    tot = {r: None for r in list(rules) + ["none"]}
    for mu in (0.2, 0.5, 0.9):
        for fh in (10.13, 24.30, 60.75):
            fm = fh / (MASS * G)
            mats = {"mu": ((mu, mu + 1e-6),), "f_hold_mg": ((fm, fm + 1e-6),)}
            ep = collect([model], a.n, mats, 9_000_000 + int(mu * 100) * 1000 + int(fh), "cpu", a.n, cfg_kw)
            res = {r: rule_rates(f(ep)) for r, f in rules.items()}
            n = len(ep["held_k0"])
            res["none"] = {"n": n, "valid_perch": int(ep["held_k0"].sum()), "off": n,
                           "fell_after_off": int(ep["fell_k0"].sum()), "abort": 0, "self_abort": 0}
            out[f"{mu}_{fh}"] = res
            for r, v in res.items():
                tot[r] = dict(v) if tot[r] is None else {k: tot[r][k] + v[k] for k in v}
            print(mu, fh, {r: (v["valid_perch"], v["off"], v["fell_after_off"], v["abort"], v["self_abort"])
                           for r, v in res.items()}, flush=True)
    for r, v in tot.items():
        print(f"{r:9s} valid {v['valid_perch'] / v['n']:.3f}  false-safe(fall) {v['fell_after_off'] / max(v['off'], 1):.3f} "
              f"(off {v['off']})  abort {v['abort'] / v['n']:.3f}  self_abort {v['self_abort'] / v['n']:.3f}")
    out["total"] = tot
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
