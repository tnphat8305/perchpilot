"""Evaluate frozen Q2 models and decision rules on the logged external C5 test."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from q2.belief_net import auroc, ece, wilson  # noqa: E402
from q2.decide_v3 import b4b, dthr_fn, fsm_features as legacy_features  # noqa: E402
from q2.decide_v4 import HYBRIDS, b4d, b4e, canonical, digest, dthr_b4e, dthr_slip_abort  # noqa: E402
from q2.fsm_features import scorer, slip_distance  # noqa: E402
from q2.sitl_external import DEFAULT_OUT, DT, N_CH, SENSOR_CHANNELS, WIN  # noqa: E402
from sim.contact_phase import ContactCfg, SLIP_CHANNELS  # noqa: E402


def read_lock(path):
    lock = json.loads(path.read_text())
    if digest(canonical(lock["config"])) != lock["sha256"] or lock["config"]["protocol"] != "H2-v4":
        raise ValueError(f"invalid Q2 lock: {path}")
    return lock


def load_dataset(path):
    raw = (path / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    channels = manifest["channels"]
    if manifest["role"] != "external_test_only" or channels not in (SENSOR_CHANNELS, SENSOR_CHANNELS + SLIP_CHANNELS):
        raise ValueError("expected a C5 external-test manifest")
    x_bytes = (path / "X.npy").read_bytes()
    if digest(x_bytes) != manifest["X_sha256"]:
        raise ValueError("C5 archive checksum differs from manifest")
    x = np.load(io.BytesIO(x_bytes), allow_pickle=False)
    if x.dtype != np.float32 or x.shape != (len(manifest["windows"]), WIN, len(channels)) or not np.isfinite(x).all():
        raise ValueError("invalid C5 archive")
    return torch.from_numpy(x), manifest, digest(raw)


def binary_metrics(scores, labels, probability=True):
    idx = [i for i, label in enumerate(labels) if label is not None]
    y = torch.tensor([labels[i] for i in idx], dtype=torch.bool)
    s = scores[idx]
    return {"n": len(y), "positive": int(y.sum()), "auroc": auroc(s, y),
            "ece15": ece(s, y) if probability and len(y) else None}


def make_trace(scores, episodes, windows, last):
    """Build a prefix-only trace, leaving unknown outcome labels out of the policy input."""
    shape = (len(episodes), last + 1)
    alive = np.zeros(shape, dtype=bool)
    ep = {k: np.zeros(shape, dtype=np.float32) for k in scores}
    row_idx = np.full(shape, -1, dtype=int)
    for i, episode in enumerate(episodes):
        by_d = {windows[row]["d"]: row for row in episode["rows"]}
        for d in range(last + 1):
            if d not in by_d:
                break
            row = by_d[d]
            alive[i, d] = True
            row_idx[i, d] = row
            for key, score in scores.items():
                ep[key][i, d] = score[row]
    # The shared v4 policy needs label arrays to return sim outcomes. Replay
    # supplies placeholders and reads truth only after selecting the action.
    ep.update(p=ep["p"][None], ps=ep["ps"][None], alive=alive,
              held=np.zeros_like(alive), fell=np.zeros_like(alive), strict=np.zeros_like(alive))
    return ep, row_idx


def replay(rule, ep, row_idx, episodes, windows, params):
    if params is None:
        return None
    actions = rule(ep, params)
    off_rows = [windows[row_idx[i, actions["d"][i]]] for i in np.flatnonzero(actions["off"])]
    n, n_off = len(episodes), len(off_rows)
    known_loss = sum(bool(missing and e["powered_loss"] and ep["alive"][i].sum() == len(e["points"]))
                     for i, (missing, e) in enumerate(zip(actions["self_abort"], episodes)))
    censored = int(actions["self_abort"].sum()) - known_loss
    held = sum(w["held_star"] is True for w in off_rows)
    held_unknown = sum(w["held_star"] is None for w in off_rows)
    falls = sum(w["fell"] is True for w in off_rows)
    known_falls = n_off - sum(w["fell"] is None for w in off_rows)
    return {"params": params, "episodes": n, "off_selected": n_off,
            "off_at_logged_decision": sum(bool(episodes[i]["off_d"] == actions["d"][i])
                                          for i in np.flatnonzero(actions["off"])),
            "abort": int(actions["abort"].sum()), "powered_loss": known_loss, "trace_censored": censored,
            "valid_confirmed": held, "off_with_hold_truth": n_off - held_unknown,
            "valid_perch_bounds": [held / n, (held + held_unknown + censored) / n] if n else None,
            "false_safe_on_observed_off": {**wilson(falls, known_falls), "k": falls},
            "false_safe_bounds_given_selected_off": [falls / n_off, (falls + n_off - known_falls) / n_off] if n_off else None}


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(args.threads)
    # Resolve all rules and checkpoints before opening the external data.
    locks = {name: read_lock(path) for name, path in
             (("v4", args.v4_lock), ("v5", args.v5_lock), ("v6", args.v6_lock), ("v7", args.v7_lock)) if path}
    if not locks:
        raise ValueError("supply at least one v4–v7 lock")
    for name, lock in locks.items():
        cfg = lock["config"]["cfg_kw"]
        version = "v7" if cfg.get("imu_hold") else ("v6" if cfg.get("slip_sensor") else
                                                  ("v5" if cfg.get("sensor_dr") else "v4"))
        if version != name:
            raise ValueError(f"{name} lock option received a {version} lock")
    models = {}
    for version, lock in locks.items():
        seeds = [m for m in lock["config"]["models"] if not m["drop"]]
        for i, meta in enumerate(seeds):
            data = Path(meta["file"]).read_bytes()
            if digest(data) != meta["sha256"]:
                raise ValueError(f"checkpoint differs from lock: {meta['file']}")
            method = "D-thr" if i == 0 else f"D-thr/seed={meta['seed']}"
            model = torch.jit.load(io.BytesIO(data), map_location="cpu").eval()
            models[f"belief_{version}_s{meta['seed']}"] = (version, method, model)
    b4d_scores = {f"B4d_{v}": scorer(lock["config"]["B4d_model"], "cpu") for v, lock in locks.items()}
    x, manifest, data_sha = load_dataset(args.data)
    has_slip = x.shape[-1] > N_CH
    if {"v6", "v7"} & locks.keys() and not has_slip:
        raise ValueError("v6/v7 scoring requires a 13-channel archive from sitl_external.py --slip-sensor")
    windows = manifest["windows"]
    energy = torch.empty(len(x))
    for d in range(-1, 4):
        mask = torch.tensor([w["d"] == d for w in windows], dtype=torch.bool)
        if mask.any():
            energy[mask] = legacy_features(x[mask], max(0, d), ContactCfg())["hp_energy"]
    scores = {name: list(model(x[:, :, :model.mean.numel()])[:, :2].T)
              for name, (_, _, model) in models.items()}
    scores["B4b"] = [-energy, energy]
    scores.update({key: [score(x, DT), None] for key, score in b4d_scores.items()})
    if has_slip:
        disp = slip_distance(x)
        scores["B4e"] = [-disp, disp]
    cohorts = {
        "all": lambda e: True,
        "probes": lambda e: e["batch"].startswith("q2_probe") if has_slip else e["batch"] in ("q2_probe_m1", "q2_probe_m2"),
        "postreview": lambda e: (not manifest["batches"][e["batch"]]["prereview"]
                                 and manifest["batches"][e["batch"]]["source_result"] is not None),
    }
    if manifest.get("post_qos"):
        cohorts.update({batch: lambda e, b=batch: e["batch"] == b for batch in manifest["batches"]})
    methods = {name: (version, method, name) for name, (version, method, _) in models.items()}
    for name, (version, method, _) in models.items():
        for family in HYBRIDS:
            key = method.replace("D-thr", family, 1)
            if key in locks[version]["config"]["selection"]:
                methods[f"{family}_{name}"] = version, key, name
    methods.update({f"B4b_{name}": (name, "B4b", "B4b") for name in locks})
    methods.update({name: (name.removeprefix("B4d_"), "B4d", name) for name in b4d_scores})
    methods.update({f"B4e_{name}": (name, "B4e", "B4e") for name in ("v6", "v7") if name in locks})
    rules = {"D-thr": dthr_fn("single", 2), "B4b": b4b, "B4d": b4d, "B4e": b4e,
             HYBRIDS[0]: dthr_b4e, HYBRIDS[1]: dthr_slip_abort}

    # Fixed diagnostic settings. Neither these nor the 4-point probe replay
    # should be interpreted as satisfying a locked false-safe cap.
    diagnostic = {"D-thr": {"tau": 0.5, "s": 0.6}, "B4b": {"e_thr": 0.6571889521200449}, "B4d": {"tau": 0.5}}
    result = {"spec": "1.2", "dataset_sha256": data_sha, "dataset": str(args.data.resolve()),
              "lock_sha256": {k: v["sha256"] for k, v in locks.items()}, "classification": {}, "replay": {},
              "scoring_limits": "An OFF outcome is observable only if OFF was logged at the very same decision. "
                                "A missing policy continuation is censored, except documented powered contact loss."}
    for cohort, include in cohorts.items():
        ids = [i for i, w in enumerate(windows) if include(manifest["episodes"][w["episode"]])]
        group = [windows[i] for i in ids]
        result["classification"][cohort] = {
            name: {"held_star": binary_metrics(hold[ids], [w["held_star"] for w in group], name not in ("B4b", "B4e")),
                   "slip": binary_metrics(slip[ids], [w["slip"] for w in group], name not in ("B4b", "B4e"))
                   if slip is not None else None}
            for name, (hold, slip) in scores.items()
        }
        episodes = [e for e in manifest["episodes"] if include(e) and e["started_probe"]]
        if not episodes:
            continue
        rows = result["replay"][cohort] = {}
        for name, (version, method, score_name) in methods.items():
            family = method.split("/")[0]
            s_hold, s_slip = scores[score_name]
            raw = {"p": s_hold, "ps": s_slip if family in ("D-thr", *HYBRIDS) else torch.zeros(len(x)),
                   "hp_energy": energy, "b4d": s_hold}
            if has_slip:
                raw["slip_mm"] = disp
            ep, row_idx = make_trace(raw, episodes, windows, last=2)
            rule = rules[family]
            selections = locks[version]["config"]["selection"][method]
            rows[name] = {"locked": {
                cap: replay(rule, ep, row_idx, episodes, windows, pick["params"] if pick else None)
                for cap, pick in selections.items()},
                "fixed_diagnostic": replay(rule, ep, row_idx, episodes, windows, diagnostic.get(family))}
            if cohort == "probes" and family in diagnostic:
                ep, row_idx = make_trace(raw, episodes, windows, last=3)
                rule = {"D-thr": dthr_fn("single", 3, d_min=3),
                        "B4b": lambda e, p: b4b(e, p, last=3),
                        "B4d": lambda e, p: b4d(e, p, last=3, d_min=3)}[family]
                rows[name]["fixed_probe_only_off_at_P09"] = replay(
                    rule, ep, row_idx, episodes, windows, diagnostic[family])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--v4-lock", type=Path)
    ap.add_argument("--v5-lock", type=Path)
    ap.add_argument("--v6-lock", type=Path)
    ap.add_argument("--v7-lock", type=Path)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    evaluate(ap.parse_args())


if __name__ == "__main__":
    main()
