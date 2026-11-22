"""Which C5 channel makes the Q2 belief net out of distribution on SITL windows?

Takes the window ending at each decision point (end of RELAX / PULL0.3 / PULL0.6) of a `--verify-rule probe`
SITL run (q2_samples.jsonl), runs belief.ts on it, then on variants that move one channel group towards what
sim/contact_phase.py produces: load-cell offset, motor vibration on the IMU (0.4 m/s^2 and 0.02 rad/s rms per
unit hover thrust, plus white noise), thrust_cmd01 rescaled to the sim TW, the touchdown body-rate transient.

    python perchpilot/ros/q2_belief_probe.py RUN_DIR [--belief DIR] [--f-offset 5.3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perch_q2 import BeliefNet  # noqa: E402
from q2_signal_check import segments  # noqa: E402


def windows(run_dir):
    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / "q2_samples.jsonl").open()]
    trials = [json.loads(line) for line in (run_dir / "trials.jsonl").open()]
    out = []
    for tr in trials:
        truth = [json.loads(line) for line in (run_dir / f"truth_{tr['trial']}.jsonl").open()]
        w0, w1 = truth[0]["wall"], truth[-1]["wall"]
        X = np.array([r["x"] for r in rows if w0 <= r["wall"] <= w1], dtype=np.float32)
        if len(X) < 100:
            continue
        segs = segments(X)
        for d, name in enumerate(("RELAX", "PULL0.3", "PULL0.6")):
            if name in segs and segs[name][-1] >= 79:
                end = segs[name][-1]
                out.append({"trial": tr["trial"], "mu": tr["mu"], "f_hold_n": tr["f_hold_n"], "d": d,
                            "macro": name, "w": X[end - 79:end + 1]})
    return out


def variants(w, f_offset, rng, tw_real=2.25, tw_sim=3.0):
    v = {"raw": w}
    a = w.copy(); a[:, 6] += f_offset; v["f_normal+offset"] = a
    T = (w[:, 8:9] * tw_real)                                  # thrust in hover units
    b = w.copy()
    b[:, 0:3] += rng.normal(0, 1, (len(w), 3)) * (0.007 + 0.4 * T)
    b[:, 3:6] += rng.normal(0, 1, (len(w), 3)) * (0.0005 + 0.02 * T)
    v["imu+motor_vib"] = b
    c = b.copy(); c[:, 6] += f_offset; v["vib+offset"] = c
    e = w.copy(); e[:, 8] *= tw_real / tw_sim; v["thrust01@simTW"] = e
    f = c.copy(); f[:, 8] *= tw_real / tw_sim; v["vib+offset+TW"] = f
    # pre-touchdown part as contact_phase._fill_pre_history models it: specific force (0, 0, T), gyro = rate at
    # touchdown (the SITL window instead holds the end of the policy's flare manoeuvre)
    g = w.copy()
    pre = g[:, 9] < 0.5
    if pre.any() and (~pre).any():
        k0 = int(np.argmin(pre))
        g[pre, 0:2] = 0.0
        g[pre, 2] = g[pre, 8] * tw_real * 9.81
        g[pre, 3:6] = w[k0, 3:6]
    v["pre_touch_as_sim"] = g
    h = g.copy(); h[:, 6] += f_offset; v["pre_as_sim+offset"] = h
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--belief", default="/home/sup/data/perchpilot/runs/q2_belief_v2_small")
    ap.add_argument("--f-offset", type=float, default=5.3)
    a = ap.parse_args(argv)
    net = BeliefNet(a.belief)
    rng = np.random.default_rng(0)
    rows = []
    for item in windows(a.run_dir):
        r = {k: item[k] for k in ("trial", "mu", "f_hold_n", "macro")}
        for name, w in variants(item["w"], a.f_offset, rng).items():
            r[name] = [round(x, 3) for x in net(w)]
        rows.append(r)
        print(json.dumps(r))


if __name__ == "__main__":
    main()
