"""Slow-slip cell check: HoldPad REST margin, slip-speed time course (truth) and total REST slip per trial.

Calibration used to pick the cell (from the held / sliding REST margins of q2_probe_m4b and q2_none_qos, motors off,
plate 65 deg): M = mu (8.55 + F_hold) - 18.35 N, i.e. m g cos(65) = 8.55 N, m g sin(65) = 18.35 N (m = 2.06 kg).
HoldPad (C2) asks for a tangential slip speed slip_gain * |M| = 0.05 (m/s)/N * |M|; Gazebo's own gear / plate
contact friction (mu 0.9) acts on top of it, so the realised REST slip can be much slower than C2 predicts.

    python perchpilot/ros/q2_slow_slip_check.py RUN_DIR [RUN_DIR ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BINS = [(-1.0, -0.5), (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (2.0, 2.5), (3.0, 3.5), (4.0, 4.5), (6.0, 6.5),
        (8.0, 8.5), (9.5, 10.0)]


def trial(run, tr):
    rows = [json.loads(line) for line in (run / f"truth_{tr['trial']}.jsonl").open()]
    rest = [r["sim_t"] for r in rows if r["kind"] == "diag" and r["diag"].get("phase") == "REST"]
    out = {k: tr.get(k) for k in ("trial", "mu", "f_hold_n", "outcome", "crash_reason", "valid_perch", "rest_slip_m")}
    out["predicted_margin_n"] = round(tr["mu"] * (8.55 + tr["f_hold_n"]) - 18.35, 3)
    if not rest:
        return out
    t0 = rest[0]
    s = np.array([(r["sim_t"] - t0, r["d_u_mm"], r["d_w_mm"]) for r in rows if r["kind"] == "slip" and r["valid"] > 0])
    h = np.array([(r["sim_t"] - t0, r["margin_n"], r["slip_mps"], r["engaged"]) for r in rows if r["kind"] == "hold"])
    steady = h[(h[:, 0] > 2.0) & (h[:, 3] > 0)]
    out["rest_margin_n_median"] = round(float(np.median(steady[:, 1])), 3) if len(steady) else None
    out["bins"] = []
    for a, b in BINS:
        m = (s[:, 0] >= a) & (s[:, 0] < b) if len(s) else np.zeros(0, bool)
        hm = (h[:, 0] >= a) & (h[:, 0] < b)
        out["bins"].append({"t_rest_s": [a, b], "slip_mm": round(float(np.linalg.norm(s[m, 1:3].sum(0))), 3) if m.any()
                            else 0.0, "holdpad_v_mm_s": round(1000 * float(h[hm, 2].mean()), 2) if hm.any() else None,
                            "margin_n": round(float(h[hm, 1].mean()), 3) if hm.any() else None})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = []
    for rd in a.run_dirs:
        run = Path(rd)
        for tr in (json.loads(line) for line in (run / "trials.jsonl").open()):
            r = {"run": run.name, **trial(run, tr)}
            res.append(r)
            print(f"{run.name} trial {r['trial']} mu {r['mu']} F {r['f_hold_n']} -> {r['outcome']} valid "
                  f"{r['valid_perch']} rest_slip {r['rest_slip_m']} m  margin pred {r['predicted_margin_n']} "
                  f"meas {r.get('rest_margin_n_median')}")
            for b in r.get("bins", []):
                print(f"     REST {b['t_rest_s'][0]:5.1f}..{b['t_rest_s'][1]:5.1f} s  slip {b['slip_mm']:7.3f} mm  "
                      f"v {b['holdpad_v_mm_s']} mm/s  margin {b['margin_n']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
