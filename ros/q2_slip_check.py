"""Check the C5 v2 slip_disp channels (node C5 window, columns 11-12) against HoldPad's slip truth.

Per trial of a probe run (`--verify-rule probe`, PERCH_SLIP_SENSOR=1): for each macro segment (RELAX, PULL0.3,
PULL0.6, PULL0.9) and for the first 2 s of REST, the accumulated displacement [mm] measured by the window vs the
truth (/perch/slip_truth, valid samples only: d_true over the same wall-time span, shifted by the 10 ms sensor delay), and the HoldPad
slip speed (hold_truth). Also the per-50-ms-bin error of the measured path length.

    python perchpilot/ros/q2_slip_check.py /home/sup/data/perchpilot/sitl_eval/q2_probe_m4 [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2_signal_check import segments  # noqa: E402

DELAY = 0.010


def trial_check(run, tr):
    truth = [json.loads(line) for line in (run / f"truth_{tr['trial']}.jsonl").open()]
    w0, w1 = truth[0]["wall"], truth[-1]["wall"]
    rows = [json.loads(line) for line in (run / "q2_samples.jsonl").open()]
    rows = [r for r in rows if w0 <= r["wall"] <= w1 and len(r["x"]) == 13]
    if not rows:
        return None
    X = np.array([r["x"] for r in rows])
    W = np.array([r["wall"] for r in rows])
    P = np.array([r["phase"] for r in rows])
    st = np.array([(t["wall"], t["d_u_mm"], t["d_w_mm"], t["valid"]) for t in truth if t["kind"] == "slip"])
    if len(st):     # the sensor reads only noise when the pad is off the plate (gap > 2 mm): compare valid truth
        st = st[st[:, 3] > 0]
    hold = np.array([(t["wall"], t["slip_mps"], t["engaged"]) for t in truth if t["kind"] == "hold"])
    out = {"trial": tr["trial"], "mu": tr["mu"], "f_hold_n": tr["f_hold_n"], "outcome": tr["outcome"],
           "crash_reason": tr["crash_reason"], "segments": {}}

    def span(name, a, b):
        meas = X[(W >= a) & (W <= b), 11:13].sum(0)
        m = (st[:, 0] >= a - DELAY) & (st[:, 0] <= b - DELAY) if len(st) else np.zeros(0, bool)
        true = st[m, 1:3].sum(0) if m.any() else np.zeros(2)
        hm = (hold[:, 0] >= a) & (hold[:, 0] <= b) if len(hold) else np.zeros(0, bool)
        v = float(hold[hm, 1].mean()) if hm.any() else float("nan")
        out["segments"][name] = {"meas_mm": np.round(meas, 3).tolist(), "true_mm": np.round(true, 3).tolist(),
                                 "meas_path_mm": round(float(np.linalg.norm(meas)), 3),
                                 "true_path_mm": round(float(np.linalg.norm(true)), 3),
                                 "holdpad_slip_mps": round(v, 4), "dur_s": round(b - a, 3)}

    segs = segments(X)
    for name, idx in segs.items():
        span(name, W[idx[0]], W[idx[-1]])
    rest = np.nonzero(P == "REST")[0]
    if len(rest):
        span("REST_2s", W[rest[0]], min(W[rest[0]] + 2.0, W[-1]))
    # per-bin comparison during VERIFY + REST (50 ms bins)
    m = np.isin(P, ["VERIFY", "REST"])
    if m.sum() > 20 and len(st):
        t0, t1 = W[m][0], W[m][-1]
        edges = np.arange(t0, t1, 0.05)
        em, et = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            em.append(np.linalg.norm(X[(W >= a) & (W < b), 11:13].sum(0)))
            sm = (st[:, 0] >= a - DELAY) & (st[:, 0] < b - DELAY)
            et.append(np.linalg.norm(st[sm, 1:3].sum(0)))
        em, et = np.array(em), np.array(et)
        out["bins_50ms"] = {"n": int(len(em)), "rmse_mm": round(float(np.sqrt(np.mean((em - et) ** 2))), 4),
                            "corr": round(float(np.corrcoef(em, et)[0, 1]), 3) if et.std() > 0 and em.std() > 0
                            else None, "true_max_mm": round(float(et.max()), 3)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    trials = [json.loads(line) for line in (run / "trials.jsonl").open()]
    res = [r for r in (trial_check(run, t) for t in trials) if r]
    for r in res:
        print(f"trial {r['trial']} mu={r['mu']} F={r['f_hold_n']} -> {r['outcome']} {r['crash_reason']} "
              f"bins={r.get('bins_50ms')}")
        for name, s in r["segments"].items():
            print(f"   {name:8s} meas {s['meas_path_mm']:8.3f} mm  true {s['true_path_mm']:8.3f} mm  "
                  f"holdpad v {s['holdpad_slip_mps']:.3f} m/s  ({s['dur_s']} s)")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
