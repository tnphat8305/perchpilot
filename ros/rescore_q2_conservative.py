"""Re-score logged Q2 batches from the MOTORS_OFF decision (spec 1.2), offline."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
from pathlib import Path

import numpy as np

from eval_node import HOLD_MAX_GAP_S, PROJECT, TRUTH_ROOT, wilson


def read_jsonl(path):
    data = path.read_bytes()
    return [json.loads(line) for line in data.splitlines()], hashlib.sha256(data).hexdigest()


def batch_logs(run_dir, hashes):
    """Use evaluator wall intervals: node-side trial numbers may reset after restart."""
    rows, hashes["q2_samples.jsonl"] = read_jsonl(run_dir / "q2_samples.jsonl")
    trials, hashes["trials.jsonl"] = read_jsonl(run_dir / "trials.jsonl")
    rows.sort(key=lambda row: row["wall"])
    wall = [r["wall"] for r in rows]
    for trial in trials:
        name = f"truth_{trial['trial']}.jsonl"
        if not (run_dir / name).is_file():
            yield trial, [], []
            continue
        truth, hashes[name] = read_jsonl(run_dir / name)
        if not truth or truth[0]["kind"] != "start" or truth[-1]["kind"] != "end":
            yield trial, [], []
            continue
        for key in ("trial", "mu", "f_hold_n"):
            if truth[0][key] != trial[key]:
                raise ValueError(f"{run_dir / name}: trial metadata mismatch for {key}")
        start, end = truth[0]["wall"], truth[-1]["wall"]
        yield trial, rows[bisect_left(wall, start):bisect_right(wall, end)], truth


def hold_rows(truth):
    return sorted((r for r in truth if r["kind"] == "hold"), key=lambda r: r["wall"])


def off_truth(samples, truth, rest_s=10.0):
    """REST begins in the tick issuing the Q2 OFF decision, before PX4 disarms.

    Historical logs bracket that tick by C5 samples or by diagnostic snapshots.
    Use the first REST C5 stamp when present and retain the timing source.
    """
    off = {"commanded": None, "wall": None, "source": None, "held_star": None,
           "held_strict": None, "fell": None, "slip_path_m": None, "disarmed": False}
    if not truth:
        return off
    rest = next((r for r in samples if r["phase"] == "REST" and r["x"][9] == 3), None)
    diags = [r for r in truth if r["kind"] == "diag"]
    diag_rest = [r for r in diags if r["diag"]["phase"] == "REST"
                 and (r["diag"].get("q2") or {}).get("decision") == "OFF"]
    off["disarmed"] = any(not r["diag"]["armed"] for r in diag_rest)
    post_off_abort = next((r for r in diags if "after MOTORS_OFF was commanded"
                           in (r["diag"].get("q2") or {}).get("why", "")), None)
    if rest is not None:
        off.update(commanded=True, wall=rest["wall"], source="first_REST_C5")
    elif diag_rest:
        off.update(commanded=True, wall=diag_rest[0]["wall"], source="first_REST_diag")
    elif post_off_abort:
        off.update(commanded=True, source="node_reported_post_OFF_loss")
    else:
        off["commanded"] = False
        return off

    hold = hold_rows(truth)
    if not hold:
        return off
    wall = off["wall"]
    if wall is not None:
        # C5 and truth use the same wall clock. Integrate with the truth sim clock.
        t0 = float(np.interp(wall, [r["wall"] for r in hold], [r["sim_t"] for r in hold]))
        deadline = t0 + rest_s
        # Do not include a deliberate release after a completed REST. If the controller
        # departed because contact was lost, the disengaged truth normally precedes it.
        depart = next((r["wall"] for r in samples if r["wall"] > wall and r["phase"] == "DEPART"), float("inf"))
        # DEPART after confirmed disarm may be re-arm/release, or recovery after a
        # loss. Look until the scoring deadline if loss is already visible at departure.
        last_t, loss, complete, gap = t0, None, False, False
        path_slip, strict = 0.0, True
        for r in hold:
            t = r["sim_t"]
            if r["wall"] < wall or t < last_t:
                continue
            if r["wall"] >= depart:
                break
            gap |= t - last_t > HOLD_MAX_GAP_S
            path_slip += r["slip_mps"] * max(0.0, min(t, deadline) - last_t)
            if t <= deadline + 1e-9:
                strict &= r["margin_n"] >= 0 and r["engaged"]
                if not r["engaged"]:
                    loss = r
                    break
            last_t = t
            if t >= deadline:
                complete = True
                break
        if loss is not None:
            off.update(held_star=False, held_strict=False, fell=True, slip_path_m=path_slip,
                       loss_sim_t=loss["sim_t"], loss_wall=loss["wall"])
        elif complete and not gap:
            off.update(held_star=path_slip < 0.02, held_strict=strict, fell=False, slip_path_m=path_slip)

    # This diagnostic explicitly records the sequence. Require a disengaged
    # truth sample near that report (even if the C5 transition itself was missed).
    if off["fell"] is None and post_off_abort:
        loss = next((r for r in hold if not r["engaged"]
                     and abs(r["wall"] - post_off_abort["wall"]) <= 0.25), None)
        if loss is not None:
            off.update(held_star=False, held_strict=False, fell=True,
                       loss_sim_t=loss["sim_t"], loss_wall=loss["wall"])
    return off


def rescored_metrics(trials, outcomes):
    """Summarize the command-based Q2 outcomes alongside the disarm record."""
    n = len(trials)
    cut = [r for r in outcomes if r["commanded"] is True]
    known_command = sum(r["commanded"] is not None for r in outcomes)

    def metric(k, negate=False):
        known = sum(r[k] is not None for r in cut)
        successes = sum(r[k] is (not negate) for r in cut)
        return {**(wilson(successes, len(cut)) if known == len(cut) else
                   {"p": None, "lo": None, "hi": None, "n": len(cut)}),
                "k": successes, "unknown": len(cut) - known}

    unknown = n - known_command + metric("held_star")["unknown"]
    valid = sum(r["held_star"] is True for r in cut)
    return {"off_commanded": wilson(len(cut), known_command), "off_command_unknown": n - known_command,
            "valid_perch": {**(wilson(valid, n) if not unknown else {"p": None, "lo": None, "hi": None, "n": n}),
                            "k": valid, "unknown": unknown},
            "false_safe_fall": metric("fell"), "false_safe_held": metric("held_star", negate=True),
            "false_safe_strict": metric("held_strict", negate=True),
            "falls_before_disarm": sum(r["fell"] is True and not r["disarmed"] for r in cut)}


def run_batch(run_dir, rest_s):
    hashes = {}
    trials, outcomes = [], []
    for trial, samples, truth in batch_logs(run_dir, hashes):
        trials.append(trial)
        outcomes.append({"trial": trial["trial"], "mu": trial["mu"], "f_hold_n": trial["f_hold_n"],
                         **off_truth(samples, truth, rest_s)})
    return trials, outcomes, hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, default=TRUTH_ROOT)
    ap.add_argument("--results", type=Path, default=PROJECT / "results")
    ap.add_argument("--out", type=Path, default=None, help="directory for *_conservative.json (default: --results)")
    args = ap.parse_args()
    target = args.out or args.results
    target.mkdir(parents=True, exist_ok=True)
    reports = {}
    for path in sorted(args.results.glob("sitl_q2_*.json")):
        if path.stem.endswith("_conservative") or path.stem == "sitl_q2_report":
            continue
        original = json.loads(path.read_text())
        batch = path.stem.removeprefix("sitl_")
        trials, outcomes, hashes = run_batch(args.logs / batch, original["config"]["rest_s"])
        if len(trials) != original["eval"]["n"]:
            raise ValueError(f"{batch}: trial count differs from {path}")
        metrics = rescored_metrics(trials, outcomes)
        cells = {}
        for key in original["eval"]["bins"]:
            ids = [i for i, t in enumerate(trials)
                   if f"speed_{t['speed_mps']}_mu_{t['mu']}_f_hold_n_{t['f_hold_n']}" == key]
            if ids:
                cells[key] = rescored_metrics([trials[i] for i in ids], [outcomes[i] for i in ids])
        result = {**original, "spec": "1.2",
                  "eval": {**original["eval"], "nominal": {**original["eval"]["nominal"], **metrics},
                           "bins": {k: {**v, **cells.get(k, {})} for k, v in original["eval"]["bins"].items()}},
                  "conservative": {"batch": batch, "raw_log_dir": str(args.logs / batch),
                                   "sha256": hashes, "trials": outcomes}}
        (target / f"{path.stem}_conservative.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
        reports[batch] = metrics
        print(batch, metrics["valid_perch"], metrics["false_safe_fall"], "pre-disarm falls", metrics["falls_before_disarm"])
    report_path = args.results / "sitl_q2_report.json"
    if report_path.exists():
        original = json.loads(report_path.read_text())
        result = {name: {**row, "spec": "1.2", "all": {**row["all"], **reports[name]}} for name, row in original.items()}
        (target / "sitl_q2_report_conservative.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
