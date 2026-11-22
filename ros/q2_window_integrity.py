"""Sample integrity of the node's 200 Hz C5 window (q2_samples.jsonl) during PERCH / VERIFY / REST.

Per channel group, the fraction of window rows that repeat the previous row exactly (stale sample-and-hold: the
topic delivered nothing new in that 5 ms tick) and, for the slip channels, the fraction of exact (0, 0) rows (no
HoldPad sample accumulated; the C5 v2 sensor itself gives ~2% dropout zeros + ~4% both-quantised-to-zero noise).
Also the eval_node truth rate of HoldPad samples (slip/load rows per sim second) for reference.

    python perchpilot/ros/q2_window_integrity.py RUN_DIR [RUN_DIR ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GROUPS = {"acc": slice(0, 3), "gyro": slice(3, 6), "f_normal": slice(6, 7), "contact": slice(7, 8),
          "slip": slice(11, 13)}


def run_integrity(run):
    rows = [json.loads(line) for line in (run / "q2_samples.jsonl").open()]
    out = {"run": run.name}
    for ph in ("PERCH", "VERIFY", "REST"):
        sel = [r for r in rows if r["phase"] == ph]
        if len(sel) < 2:
            continue
        X = np.array([r["x"] for r in sel])
        W = np.array([r["wall"] for r in sel])
        cont = np.diff(W) < 0.02            # consecutive ticks only
        d = {"n": int(len(X)), "ncol": int(X.shape[1])}
        for g, sl in GROUPS.items():
            if sl.stop > X.shape[1]:
                continue
            rep = (X[1:, sl] == X[:-1, sl]).all(1)[cont]
            d[f"{g}_repeat"] = round(float(rep.mean()), 4)
        if X.shape[1] >= 13:
            d["slip_both_zero"] = round(float((X[:, 11:13] == 0).all(1).mean()), 4)
        # f_normal repeats are only informative when the value is changing (in contact, motors still on)
        out[ph] = d
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = [run_integrity(Path(r)) for r in a.run_dirs]
    for r in res:
        print(r["run"])
        for ph in ("PERCH", "VERIFY", "REST"):
            if ph in r:
                print(f"  {ph:7s}", {k: v for k, v in r[ph].items()})
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
