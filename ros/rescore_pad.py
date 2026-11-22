"""Offline rescoring of finished SITL batches with the touch velocity at the PAD point (review D01) and the D04
success rule (a cycle needs a valid touch and no crash).

Old HoldPad builds published v_n / v_t of the CoG. Here the pad velocity is rebuilt from the logged Gazebo poses
(base_link CoG velocity by central differences + body rate x r_pad, r_pad = 0.227 m along -z_body), relative to
the plate (cart translation), just before the touch stamp. Envelope and summary as eval_node (C3, Wilson).

    python perchpilot/ros/rescore_pad.py main_v0 main_v15 main_v3 [--dt-before 0.01]
writes perchpilot/results/sitl_<name>_padscored.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_node import PROJECT, TRUTH_ROOT, score_touch, summarize  # noqa: E402
from perch_core import plate_axes  # noqa: E402
from sitl_obs_check import Truth  # noqa: E402

PAD = 0.227


def pad_touch(truth_rows, touch, dt_before, h):
    tr = Truth(truth_rows)
    t_wall = touch["sim_t"] * tr.fit[0] + tr.fit[1] - dt_before
    if tr.ok(t_wall - h) and tr.ok(t_wall + h):
        d, p = tr.state(t_wall, h=h)
        method = "central"
    elif tr.ok(t_wall - 2 * h) and t_wall <= tr.t[-1] + 1e-3:
        # the log ends at the touch: backward difference over [t - 2h, t], evaluated at t - h
        d, p = tr.state(t_wall - h, h=h)
        method = "backward"
    else:
        return None
    u, w, n = plate_axes(p.R)
    v_pad = d.v + d.R @ np.cross(d.w, [0.0, 0.0, -PAD]) - p.v
    vn = float(-(v_pad @ n))
    vt = float(np.linalg.norm(v_pad + vn * n))
    v_cog = d.v - p.v
    return {"vn_pad_mps": vn, "vt_pad_mps": vt, "vn_cog_rebuilt": float(-(v_cog @ n)),
            "vt_cog_rebuilt": float(np.linalg.norm(v_cog + (-(v_cog @ n)) * n)), "diff": method}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--dt-before", type=float, default=0.01)
    ap.add_argument("--h", type=float, default=0.02, help="central-difference half window [s]")
    a = ap.parse_args(argv)
    for name in a.names:
        run = TRUTH_ROOT / name
        trials = [json.loads(line) for line in (run / "trials.jsonl").open()]
        changed = 0
        for t in trials:
            tc = t.get("touch")
            rows = [json.loads(line) for line in (run / f"truth_{t['trial']}.jsonl").open()]
            if tc is not None:
                touch_row = next(r for r in rows if r["kind"] == "touch")
                tc["sim_t"] = touch_row["sim_t"]
                r = pad_touch(rows, tc, a.dt_before, a.h)
                if r:
                    tc.update(r, vn_cog_holdpad=tc["vn_mps"], vt_cog_holdpad=tc["vt_mps"],
                              vn_mps=r["vn_pad_mps"], vt_mps=r["vt_pad_mps"], v_point="pad (rebuilt from poses)")
            ok = tc is not None and score_touch(tc)
            new_success = bool(t["held"] is True and ok and (t["crash_reason"] in (None, "touch_envelope")))
            if t["crash_reason"] == "touch_envelope" and ok:
                t["crash_reason_holdpad_cog"] = "touch_envelope"
                t["crash_reason"] = None       # the envelope miss came only from the CoG velocity
            if new_success != t["success"] or ok != t["touch_success"]:
                changed += 1
                print(f"{name} trial {t['trial']}: touch_success {t['touch_success']}->{ok}, "
                      f"success {t['success']}->{new_success} (vt cog {tc and tc.get('vt_cog_holdpad')}, "
                      f"pad {tc and round(tc['vt_mps'], 3)})")
            t["touch_success"], t["success"] = ok, new_success
            t["cycle_complete"] = bool(new_success and t.get("recovered") and t.get("rearm_ok"))
        orig = json.loads((PROJECT / "results" / f"sitl_{name}.json").read_text())
        orig["name"] = f"{name}_padscored"
        orig["rescore"] = {"tool": "ros/rescore_pad.py", "touch_velocity": "pad point rebuilt from Gazebo poses",
                           "dt_before_s": a.dt_before, "diff_half_window_s": a.h,
                           "success_rule": "held and valid touch and no crash (review D04)", "trials_changed": changed}
        orig["eval"]["nominal"] = summarize(trials)
        out = PROJECT / "results" / f"sitl_{name}_padscored.json"
        out.write_text(json.dumps(orig, indent=2, allow_nan=False) + "\n")
        e = orig["eval"]["nominal"]
        print(f"{name}: touch_success {e['touch_success']['p']} success {e['success']['p']} "
              f"cycle {e['cycle_complete']['p']} vt_mean {e['touch_vt_mean']} vn_mean {e['touch_vn_mean']} "
              f"changed {changed} -> {out.name}")


if __name__ == "__main__":
    main()
