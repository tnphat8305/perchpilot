"""Q2 SITL batch table (Wilson 95%) from trials.jsonl: valid perch, false-safe (fall after OFF / OFF), controlled
ABORT, SELF_ABORT, other crash, recovery after ABORT, per rule and per material.

A trial whose OFF was commanded but whose contact was lost before the disarm (no motors-off, outcome timeout in
batches run before commit 'Q2: contact lost after OFF ...') is counted as SELF_ABORT here.

    python perchpilot/ros/q2_report.py q2_dthr q2_none q2_fsm_b4b q2_dthr_min [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_node import TRUTH_ROOT, wilson  # noqa: E402


def classify(t):
    q2 = (t.get("q2") or {}).get("decision")
    if t["outcome"] == "crash" and t.get("fell_after_off"):
        return "fall_after_off"
    if t["outcome"] == "crash":
        return "crash_other"
    if t["outcome"] == "q2_abort":
        return "abort"
    if t["outcome"] == "self_abort" or (t["outcome"] == "timeout" and q2 in ("OFF", "SELF_ABORT")
                                        and not t["motors_off"]):
        return "self_abort"
    if t["outcome"] == "rest":
        return "valid" if t.get("valid_perch") else "rest_not_valid"
    return t["outcome"]


def table(trials):
    n = len(trials)
    cls = [classify(t) for t in trials]
    off = [t for t in trials if t["motors_off"]]
    ab = [t for t, c in zip(trials, cls) if c in ("abort", "self_abort")]
    return {"n": n,
            "valid_perch": wilson(sum(c == "valid" for c in cls), n),
            "motors_off": wilson(len(off), n),
            "false_safe_fall": wilson(sum(t.get("fell_after_off") is True for t in off), len(off)),
            "abort": wilson(sum(c == "abort" for c in cls), n),
            "self_abort": wilson(sum(c == "self_abort" for c in cls), n),
            "crash_other": wilson(sum(c == "crash_other" for c in cls), n),
            "recovered_after_abort": wilson(sum(t.get("recovered") is True for t in ab), len(ab)),
            "classes": {c: cls.count(c) for c in sorted(set(cls))}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = {}
    for name in a.names:
        trials = [json.loads(line) for line in (TRUTH_ROOT / name / "trials.jsonl").open()]
        res = {"all": table(trials), "materials": {}}
        by = defaultdict(list)
        for t in trials:
            by[f"mu{t['mu']}_F{t['f_hold_n']}"].append(t)
        for k, v in sorted(by.items()):
            res["materials"][k] = {"n": len(v), "classes": table(v)["classes"]}
        out[name] = res
        f = lambda w: f"{w['p']:.3f} [{w['lo']:.3f}; {w['hi']:.3f}] ({round(w['p'] * w['n'])}/{w['n']})" \
            if w["p"] is not None else "n/a"
        r = res["all"]
        print(f"{name} n={r['n']}: valid {f(r['valid_perch'])} | false-safe {f(r['false_safe_fall'])} | "
              f"ABORT {f(r['abort'])} | SELF_ABORT {f(r['self_abort'])} | other crash {f(r['crash_other'])} | "
              f"recovered after abort {f(r['recovered_after_abort'])}")
        for k, v in res["materials"].items():
            print(f"    {k}: {v['classes']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
