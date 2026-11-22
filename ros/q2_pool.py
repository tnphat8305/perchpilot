"""Pool SITL Q2 batches of the same rule and configuration (e.g. eval seeds 2027 + 2028) and compare with `none`.

Per pooled group: Wilson 95% for valid perch (held_star), false-safe v1.2 (contact lost within the REST window after
MOTORS_OFF was commanded / OFF commanded; includes loss of contact before PX4 finished disarming, which the node
logs as SELF_ABORT "after MOTORS_OFF was commanded"), false_safe_held (OFF with held_star = 0 / OFF), Q2 ABORT,
SELF_ABORT and the OFF rate. Against the baseline group: difference with the Newcombe hybrid-score (Wilson) 95% CI
and the two-sided Fisher exact p.

Batches are pooled only when their effective configuration matches (node args, slip sensor, load cell, QoS, speeds,
materials, REST, policy, HoldPad / PX4 binaries); the eval seed may differ. Trial rows come from
/home/sup/data/perchpilot/sitl_eval/<name>/trials.jsonl (last row per trial id), configuration from
perchpilot/results/sitl_<name>.json.

    python perchpilot/ros/q2_pool.py [--json perchpilot/results/sitl_q2_pooled.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
EVAL = Path("/home/sup/data/perchpilot/sitl_eval")
Z = 1.959964

# label, baseline label, batches
GROUPS = [
    ("none", None, ["q2_none_qos", "q2_none_qos_s2"]),
    ("b4e", "none", ["q2_b4e_qos", "q2_b4e_qos_s2"]),
    ("dthr_v7", "none", ["q2_dthr_v7_qos", "q2_dthr_v7_qos_s2"]),
    ("none_s2027", None, ["q2_none_qos"]),
    ("dthr_v6_s2027", "none_s2027", ["q2_dthr_v6_qos"]),
    ("dthr_v4full_s2027", "none_s2027", ["q2_dthr_v4full_qos"]),
    ("none_v15", None, ["q2_none_qos_v15", "q2_none_qos_v15_s2"]),
    ("b4e_v15", "none_v15", ["q2_b4e_qos_v15", "q2_b4e_qos_v15_s2"]),
    ("none_v15_s2027", None, ["q2_none_qos_v15"]),
    ("dthr_v7_v15", "none_v15_s2027", ["q2_dthr_v7_qos_v15"]),     # same eval seed as its baseline
    ("none_slow", None, ["q2_none_slow"]),          # slow-slip cell (single mu / F_hold, REST margin slightly < 0)
    ("b4e_slow", "none_slow", ["q2_b4e_slow"]),
]


def wilson(k, n):
    if n == 0:
        return {"k": 0, "n": 0, "p": None, "lo": None, "hi": None}
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return {"k": k, "n": n, "p": round(p, 4), "lo": round(max(0.0, c - h), 4), "hi": round(min(1.0, c + h), 4)}


def newcombe(a, b):
    """95% CI of p_a - p_b (Newcombe 1998, method 10) and the two-sided Fisher exact p."""
    if a["n"] == 0 or b["n"] == 0:
        return None
    pa, pb = a["k"] / a["n"], b["k"] / b["n"]
    d = pa - pb
    lo = d - math.sqrt((pa - a["lo"]) ** 2 + (b["hi"] - pb) ** 2)
    hi = d + math.sqrt((a["hi"] - pa) ** 2 + (pb - b["lo"]) ** 2)
    _, p = fisher_exact([[a["k"], a["n"] - a["k"]], [b["k"], b["n"] - b["k"]]])
    return {"diff": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4), "fisher_p": round(float(p), 4)}


def load_trials(name):
    rows = {}
    for line in (EVAL / name / "trials.jsonl").open():
        t = json.loads(line)
        rows[t["trial"]] = t
    return [rows[k] for k in sorted(rows)]


def config_key(res):
    prov, cfg = res["provenance"], res["config"]
    env = prov.get("env", {})
    return {"node_args": env.get("PERCH_NODE_ARGS"), "slip_sensor": env.get("PERCH_SLIP_SENSOR"),
            "load_cell": env.get("PERCH_LOAD_CELL"), "q2_lock_file_sha256": prov.get("q2_lock_file_sha256"),
            "qos": prov.get("perch_node_hold_sensors_qos"), "holdpad_so_sha256": prov.get("holdpad_so_sha256"),
            "px4_bin_sha256": prov.get("px4_bin_sha256"), "policy_ts_sha256": prov.get("policy_ts_sha256"),
            **{k: cfg.get(k) for k in ("speeds", "mu", "f_hold_n", "rest_s", "policy_run", "tilt_deg")}}


def off_commanded(t):
    q = t.get("q2") or {}
    return q.get("decision") == "OFF" or "after MOTORS_OFF" in (q.get("why") or "")


def metrics(trials):
    n = len(trials)
    off = [t for t in trials if off_commanded(t)]
    fell = [t for t in off if t["outcome"] in ("crash", "self_abort")]
    return {
        "n": n,
        "valid_perch": wilson(sum(t.get("valid_perch") is True for t in trials), n),
        "off_rate": wilson(len(off), n),
        "false_safe_v12": wilson(len(fell), len(off)),
        "false_safe_held": wilson(sum(t.get("valid_perch") is not True for t in off), len(off)),
        "q2_abort": wilson(sum(t["outcome"] == "q2_abort" for t in trials), n),
        "self_abort": wilson(sum(t["outcome"] == "self_abort" and not off_commanded(t) for t in trials), n),
        "crash": wilson(sum(t["outcome"] == "crash" for t in trials), n),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out, groups = {"groups": {}, "vs_baseline": {}}, {}
    for label, base, names in GROUPS:
        names = [nm for nm in names if (ROOT / "results" / f"sitl_{nm}.json").exists() and (EVAL / nm).exists()]
        if not names:
            continue
        results = [json.loads((ROOT / "results" / f"sitl_{nm}.json").read_text()) for nm in names]
        keys = [config_key(r) for r in results]
        diff = {f: [k[f] for k in keys] for f in keys[0] if any(k[f] != keys[0][f] for k in keys)}
        # rule `none` never reads the sensor window: the passive optical slip sensor and the lock do not affect it
        ignorable = {"slip_sensor", "q2_lock_file_sha256"} if "--verify-rule none" in (keys[0]["node_args"] or "") \
            else set()
        if set(diff) - ignorable:
            raise SystemExit(f"{label}: batches differ in configuration: {diff}")
        trials = [t for nm in names for t in load_trials(nm)]
        m = metrics(trials)
        groups[label] = m
        out["groups"][label] = {"batches": names, "eval_seeds": [r["eval"].get("eval_seed") for r in results],
                                "status": [r["status"] for r in results], "config": keys[0],
                                "config_differences_ignored": diff, **m}
        if base and base in groups:
            out["vs_baseline"][label] = {"baseline": base, **{
                k: newcombe(m[k], groups[base][k]) for k in ("valid_perch", "false_safe_v12", "false_safe_held",
                                                               "q2_abort", "off_rate")}}
    fmt = lambda w: f"{w['k']}/{w['n']} [{w['lo']}; {w['hi']}]" if w["n"] else "-"
    for label, g in out["groups"].items():
        print(f"{label:18s} n={g['n']:3d} valid {fmt(g['valid_perch'])}  FS v1.2 {fmt(g['false_safe_v12'])}  "
              f"FS held {fmt(g['false_safe_held'])}  ABORT {fmt(g['q2_abort'])}  seeds {g['eval_seeds']}")
        if label in out["vs_baseline"]:
            v = out["vs_baseline"][label]
            print(" " * 20 + "vs " + v["baseline"] + ": " + "  ".join(
                f"{k} {d['diff']:+.3f} [{d['lo']:+.3f}; {d['hi']:+.3f}] p={d['fisher_p']}"
                for k, d in v.items() if k != "baseline" and d))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
