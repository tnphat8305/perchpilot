"""Build an external-test-only C5 dataset from historical Gazebo/SITL logs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros"))
sys.path.insert(0, str(ROOT))

from perch_q2 import DT, N_CH, SCHEDULE_V2, WIN  # noqa: E402
from q2_signal_check import segments  # noqa: E402
from rescore_q2_conservative import TRUTH_ROOT, batch_logs, hold_rows, off_truth  # noqa: E402
from sim.contact_phase import SENSOR_CHANNELS, SLIP_CHANNELS  # noqa: E402

DEFAULT_OUT = TRUTH_ROOT.parent / "q2data" / "sitl_ext"
TOUCHDOWN = -1
BAD_BATCHES = {"q2_none_invalid_ekf": "batch invalidated after EKF failure; see ROS README §14"}
QOS_BATCHES = {"q2_probe_m4b"} | {
    f"q2_{rule}_qos{suffix}" for rule in ("none", "b4e", "dthr_v6", "dthr_v7", "dthr_v4full")
    for suffix in ("", "_s2")
}


def decision_windows(samples, trial, off, skipped, n_channels=None):
    """Use recorded decisions, ending at the last C5 sample before the macro transition."""
    if not samples:
        return
    values = [r["x"] for r in samples]
    widths = np.array([len(v) for v in values])
    x = np.asarray([v[:N_CH] for v in values], dtype=np.float32)
    stamps = np.asarray([r["wall"] for r in samples])
    if not np.isin(widths, (N_CH, N_CH + len(SLIP_CHANNELS))).all() or not np.isfinite(x).all():
        raise ValueError(f"invalid C5 sample for trial {trial['trial']}")
    q2 = trial.get("q2") or {}
    points = q2.get("points", [])
    runs = segments(x)
    candidates = []
    for point in points:
        d, macro = point["d"], point["macro"]
        if not 0 <= d < len(SCHEDULE_V2) or macro != SCHEDULE_V2[d]:
            raise ValueError(f"unexpected decision point: {point}")
        idx = runs.get(macro)
        # A run ending with log truncation does not establish a completed macro.
        if idx is None or idx[-1] + 1 >= len(x):
            skipped["no_C5_transition"] += 1
            continue
        elapsed = stamps[idx[-1] + 1] - stamps[idx[0]]
        if abs(elapsed - point["t"]) > 0.025:
            skipped["transition_does_not_match_decision"] += 1
            continue
        candidates.append((d, macro, idx[-1]))
    if not points and off["commanded"] and "rule none:" in q2.get("why", ""):
        idx = np.flatnonzero(x[:, 9] == 3)
        if idx.size and idx[0] > 0:
            candidates.append((TOUCHDOWN, "TOUCHDOWN", idx[0] - 1))
    for d, macro, end in candidates:
        start = end + 1 - WIN
        if start < 0:
            skipped["fewer_than_80_samples"] += 1
            continue
        wstamps = stamps[start:end + 1]
        gaps = np.diff(wstamps)
        if abs(wstamps[-1] - wstamps[0] - (WIN - 1) * DT) > 0.035 or gaps.max() > 0.03:
            skipped["C5_time_gap"] += 1
            continue
        selected = slice(start, end + 1)
        has_slip = (widths[selected] > N_CH).all()
        if n_channels is not None and n_channels > N_CH and not has_slip:
            skipped["no_C5_v2_slip"] += 1
            continue
        w = np.asarray(values[selected], dtype=np.float32) if has_slip and n_channels != N_CH else x[selected].copy()
        if not np.isfinite(w).all():
            raise ValueError(f"invalid C5 sample for trial {trial['trial']}")
        yield d, macro, float(stamps[end]), w


def truth_slip(hold, wall):
    """Align slip to the last sample of the decision window on the shared wall clock."""
    if not hold:
        return None
    stamp = min(hold, key=lambda r: abs(wall - r["wall"]))
    if abs(stamp["wall"] - wall) > 0.03 or not stamp["engaged"]:
        return None
    return bool(stamp["slip_mps"] > 0)


def build(logs, out, results, slip_sensor=False, post_qos=False):
    if post_qos and not slip_sensor:
        raise ValueError("post-QoS evaluation requires 13-channel windows")
    channels = SENSOR_CHANNELS + (SLIP_CHANNELS if slip_sensor else [])
    manifest = {"spec": "1.2", "role": "external_test_only", "raw_log_root": str(logs.resolve()),
                "window": [WIN, len(channels)], "hz_nominal": int(round(1 / DT)), "channels": channels,
                "normalization": "raw float32, model normalizes once using training buffers",
                "labels": {"slip": "HoldPad truth at window end while engaged",
                           "held_star": "observed 10 s REST after OFF at this decision only",
                           "fell": "observed loss after OFF at this decision only",
                           "null": "unobserved or censored. Episode outcome cannot label earlier decisions."},
                "batches": {}, "excluded_batches": dict(BAD_BATCHES), "episodes": [], "windows": []}
    if post_qos:
        manifest["post_qos"] = True
        manifest["excluded_batches"] = {batch: "log directory absent" for batch in sorted(QOS_BATCHES)
                                        if not (logs / batch).is_dir()}
    xs = []
    for path in sorted(logs.glob("q2_*")):
        batch = path.name
        if batch in BAD_BATCHES or not path.is_dir() or (post_qos and batch not in QOS_BATCHES):
            continue
        if not all((path / name).is_file() for name in ("trials.jsonl", "q2_samples.jsonl")):
            manifest["excluded_batches"][batch] = "missing trials or C5 log"
            continue
        source = results / f"sitl_{batch}.json"
        original = json.loads(source.read_text()) if source.exists() else {}
        rest_s = original.get("config", {}).get("rest_s", 10.0)
        hashes, skipped = {}, Counter()
        count, with_windows, start = 0, 0, len(xs)
        for tr, samples, truth in batch_logs(path, hashes):
            count += 1
            if not truth:
                skipped["missing_truth_log"] += 1
                continue
            if slip_sensor and not any(len(r["x"]) == len(channels) for r in samples):
                skipped["no_C5_v2_samples"] += 1
                continue
            hold = hold_rows(truth)
            off = off_truth(samples, truth, rest_s)
            q2 = tr.get("q2") or {}
            off_d = None
            if off["commanded"]:
                if q2.get("points"):
                    off_d = q2["d"]
                elif batch.startswith("q2_none"):
                    off_d = TOUCHDOWN
            if off_d == TOUCHDOWN and "rule none:" not in q2.get("why", ""):
                # The pre-disarm loss handler can overwrite the original decision text.
                tr = {**tr, "q2": {**q2, "why": "rule none: historical OFF at touchdown"}}
            episode = {"batch": batch, "trial": tr["trial"], "mu": tr["mu"], "f_hold_n": tr["f_hold_n"],
                       "speed_mps": tr["speed_mps"], "start_wall": truth[0]["wall"], "end_wall": truth[-1]["wall"],
                       "truth_file": f"truth_{tr['trial']}.jsonl", "outcome": tr["outcome"],
                       "behavior_decision": q2.get("decision"), "off_d": off_d, "off_truth": off,
                       "points": [p["d"] for p in q2.get("points", [])],
                       "started_probe": any(r["phase"] == "VERIFY" for r in samples),
                       "powered_loss": q2.get("decision") == "SELF_ABORT" and off["commanded"] is False,
                       "rows": []}
            for d, macro, end, x in decision_windows(samples, tr, off, skipped, len(channels)):
                episode["rows"].append(len(xs))
                window = {"episode": len(manifest["episodes"]), "d": d, "macro": macro, "end_wall": end,
                          "slip": truth_slip(hold, end), "held_star": None, "fell": None}
                if d == off_d:
                    window.update(held_star=off["held_star"], fell=off["fell"])
                manifest["windows"].append(window)
                xs.append(x)
            with_windows += bool(episode["rows"])
            manifest["episodes"].append(episode)
        manifest["batches"][batch] = {"episodes": count, "episodes_with_windows": with_windows,
                                      "windows": len(xs) - start, "skipped": dict(skipped), "rest_s": rest_s,
                                      "sha256": hashes, "git": original.get("git"),
                                      "source_result": source.name if original else None,
                                      "provenance": original.get("provenance"),
                                      "prereview": "prereview" in batch or batch == "q2_probe_m1"}
    if not xs:
        raise ValueError("no C5 decision windows were extracted")
    manifest.update(n_episodes=len(manifest["episodes"]), n_windows=len(xs),
                    n_episodes_with_windows=sum(bool(e["rows"]) for e in manifest["episodes"]))
    out.mkdir(parents=True, exist_ok=True)
    target = out / "X.npy"
    np.save(target, np.stack(xs), allow_pickle=False)
    manifest["X_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n_episodes", "n_episodes_with_windows", "n_windows")}))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, default=TRUTH_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--slip-sensor", action="store_true", help="export only complete 13-channel C5 v2 windows")
    ap.add_argument("--post-qos", action="store_true", help="restrict to m4b and decision batches after the QoS fix")
    args = ap.parse_args()
    build(args.logs, args.out, args.results, args.slip_sensor or args.post_qos, args.post_qos)


if __name__ == "__main__":
    main()
