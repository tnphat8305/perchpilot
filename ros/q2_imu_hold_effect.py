"""Offline: effect of 100 Hz sample-and-hold IMU on belief v6 outputs.

A) sim val windows (v6_full, 200 Hz IMU) vs the same windows with IMU cols 0-5 held every other sample (both
   phases), belief v6 s0: |dp_hold|, |dp_slip|, decision flips at tau 0.44.
B) SITL m4b windows at decision points (VERIFY rows, 80-sample windows from q2_samples): raw (held IMU) vs IMU
   linearly interpolated over the repeated samples (approximate 200 Hz).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perch_q2 import BeliefNet  # noqa: E402

TAU = 0.44
net = BeliefNet("/home/sup/data/perchpilot/runs/q2_belief_v6_full_s0")


def run(W):
    with torch.inference_mode():
        o = net.mod(torch.from_numpy(np.asarray(W, np.float32)))
    return o[:, 0].numpy(), o[:, 1].numpy()


def hold(W, phase):
    H = W.copy()
    idx = np.arange(W.shape[1])
    src = idx - ((idx + phase) % 2)
    src[src < 0] = 0
    H[:, :, :6] = W[:, src, :6]
    return H


def summary(a, b, tag):
    (ph0, ps0), (ph1, ps1) = a, b
    flip = np.mean((ph0 >= TAU) != (ph1 >= TAU))
    print(f"{tag}: n={len(ph0)} |dp_hold| mean {np.abs(ph1 - ph0).mean():.4f} p95 {np.percentile(np.abs(ph1 - ph0), 95):.4f}"
          f"  |dp_slip| mean {np.abs(ps1 - ps0).mean():.4f}  OFF-flip@tau{TAU} {flip:.4f}"
          f"  mean p_hold {ph0.mean():.3f}->{ph1.mean():.3f}")
    return {"n": int(len(ph0)), "dp_hold_mean": float(np.abs(ph1 - ph0).mean()),
            "dp_hold_p95": float(np.percentile(np.abs(ph1 - ph0), 95)),
            "dp_slip_mean": float(np.abs(ps1 - ps0).mean()), "off_flip": float(flip),
            "p_hold_mean": [float(ph0.mean()), float(ph1.mean())]}


out = {}
X = np.load("/home/sup/data/perchpilot/q2data/v6_full/val/X_0000.npy")[:20000]
base = run(X)
for p in (0, 1):
    out[f"sim_hold_phase{p}"] = summary(base, run(hold(X, p)), f"sim val, IMU held phase {p}")

rows = [json.loads(l) for l in open("/home/sup/data/perchpilot/sitl_eval/q2_probe_m4b/q2_samples.jsonl")]
Xs = np.array([r["x"] for r in rows if len(r["x"]) == 13], np.float32)
P = np.array([r["phase"] for r in rows if len(r["x"]) == 13])
Wl = []
for i in np.nonzero(P == "VERIFY")[0]:
    if i >= 79 and (P[i - 79:i + 1] != "REST").all():
        Wl.append(Xs[i - 79:i + 1])
Wr = np.stack(Wl)
Wi = Wr.copy()
for w in Wi:
    rep = np.r_[False, (w[1:, :6] == w[:-1, :6]).all(1)]
    good = np.nonzero(~rep)[0]
    for c in range(6):
        w[:, c] = np.interp(np.arange(80), good, w[good, c])
out["sitl_m4b_raw_vs_interp"] = summary(run(Wr), run(Wi), "SITL m4b VERIFY windows, held -> interpolated")
(Path(__file__).resolve().parents[1] / "results" / "sitl_q2_imu_hold_effect_v6.json").write_text(
    json.dumps(out, indent=1))
