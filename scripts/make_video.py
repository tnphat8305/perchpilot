#!/usr/bin/env python3
"""Dựng video demo PerchPilot từ log Gazebo và PX4 SITL trên CPU."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
from tempfile import NamedTemporaryFile
import time

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

# These modules run offline and do not import ROS at module scope.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros"))
from eval_node import score_touch
from perch_core import (PhaseCfg, R_enu_flu_from_px4_q, TAG_LOSS_MAX, ned_to_enu, plate_from_cart,
                       quat_to_rot, rot_to_quat)
from perch_q2 import N_CH, PH_PULL, PH_RELAX, SCHEDULE_V3, WIN
from plate_perception import TAG_PERIOD
from tag_camera_sim import X500_BASE_LINK_IN_MODEL

ROOT = Path("/home/sup/data/perchpilot")
CODE_ROOT = Path(__file__).resolve().parents[1]   # folder that holds scripts/, results/
DEFAULT_OUT = ROOT / "video/perchpilot_demo.mp4"
MONTAGE_OUT = CODE_ROOT / "video/perchpilot_demo_v2.mp4"
RESULTS = CODE_ROOT / "results"
SUMMARY_SECONDS = 5
PUSHOFF_SECONDS = PhaseCfg().pushoff_s
DRONE, CART = "x500_perch_7", "perch_cart"
FOOTER = "Dựng lại từ log Gazebo + PX4 SITL; không phải quay màn hình"
COLORS = {
    "APPROACH": "#78c8ff", "PERCH": "#5ee4e4", "VERIFY": "#f8bf65",
    "REST": "#7be4a5", "DEPART": "#b9a5ff", "ABORT": "#ff8583", "DONE": "#7be4a5",
}
BG, PANEL, GRID = "#101923", "#162330", "#344559"
FG, MUTED = "#eef5fa", "#aabed0"
CYAN, GOLD, GREEN, RED = "#55e0ec", "#ffc468", "#82e5a6", "#ff8583"


def jsonl(path):
    with path.open(encoding="utf-8") as src:
        for line_number, line in enumerate(src, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc


def trial_rows(batch, name):
    path = batch / name
    return list(jsonl(path)) if path.is_file() else []


def b4e_thresholds(group):
    args = shlex.split(group["config"]["node_args"])
    start = args.index("--q2-slip-off") + 1
    values = [float(v) for v in args[start:start + len(SCHEDULE_V3)]]
    if len(values) != len(SCHEDULE_V3) or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Invalid B4e threshold configuration")
    return dict(zip(SCHEDULE_V3, values))


def material_cell(row):
    return row["speed_mps"], row["mu"], row["f_hold_n"]


def select_montage(root, thresholds):
    def records(name, eligible):
        batch = root / name
        return [(batch, r) for r in trial_rows(batch, "trials.jsonl")
                if eligible(r) and (batch / f"truth_{r['trial']}.jsonl").is_file()]

    def pick(candidates, rank, scene):
        if not candidates:
            raise ValueError(f"Couldn't select montage scene {scene}")
        return min(candidates, key=lambda item: rank(item[1]))

    for number, (name, title) in enumerate(
            (("main_p2y_v15", "Chu kỳ ở xe 1,5 m/s"), ("main_p2y_v3", "Chu kỳ ở xe 3 m/s")), 1):
        batch, row = pick(records(name, lambda r: r.get("cycle_complete")), lambda r: r["trial"], number)
        yield batch, row, f"Màn {number}  /  {title}"

    def rule(row, prefix, decision):
        q2 = row.get("q2") or {}
        return q2.get("decision") == decision and q2.get("why", "").startswith(prefix)

    baseline = records("q2_none_qos", lambda r: r.get("motors_off") and r.get("fell_after_off")
                       and rule(r, "rule none:", "OFF"))
    cells = {material_cell(r) for _, r in baseline}

    def is_slip_abort(row):
        if row.get("outcome") != "q2_abort" or not rule(row, "fsm_b4e at ", "ABORT") or material_cell(row) not in cells:
            return False
        points = row["q2"]["points"]
        return ([p["macro"] for p in points] == SCHEDULE_V3
                and all(p["slip_mm"] >= thresholds[p["macro"]] for p in points))

    aborts = records("q2_b4e_qos", is_slip_abort) or records("q2_b4e_qos_s2", is_slip_abort)
    abort_batch, abort = pick(aborts, lambda r: (r["mu"], r["f_hold_n"], r["trial"]), 3)
    yield abort_batch, abort, "Màn 3  /  B4e · Bề mặt trơn → ABORT"

    batch, row = pick(records("q2_b4e_qos", lambda r: rule(r, "fsm_b4e at ", "OFF")
                             and r.get("held") and r.get("valid_perch") and r.get("cycle_complete")),
                      lambda r: (-r["mu"], -r["f_hold_n"], r["trial"]), 4)
    yield batch, row, "Màn 4  /  B4e · Bám chắc → OFF"

    batch, row = pick([item for item in baseline if material_cell(item[1]) == material_cell(abort)],
                      lambda r: (abort_batch.name.endswith("_s2") or r["trial"] != abort["trial"], r["trial"]), 5)
    yield batch, row, "Màn 5  /  Đối chiếu · Tắt ngay rồi tuột tấm"


def recovery_events(path, trial):
    """Console timestamps delimit recovery, which eval_node records after closing truth JSONL."""
    if not path.is_file():
        return {}
    events = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        summary = re.search(r"\[perchpilot_eval_node\]: Trial (\d+)/", line)
        if summary:
            if events:
                break
            if int(summary[1]) == trial:
                events["end"] = float(line.split("] [")[1].split("]")[0])
        if events and "[perchpilot_perch_node]: [phase]" in line:
            for key, marker in (("depart", "REST->DEPART"), ("push", "depart: released, push-off"),
                                ("level", "depart: levelled"), ("done", "DEPART->DONE: departed,")):
                if marker in line:
                    events.setdefault(key, float(line.split("] [")[1].split("]")[0]))
            if "->WAIT:" in line:
                break
    return events


def interpolate(t, clock, values):
    return np.array([np.interp(t, clock, column) for column in values.T])


def interpolate_quat(t, clock, values):
    # Normalized linear interpolation along the short arc between adjacent pose samples.
    hi = int(np.clip(np.searchsorted(clock, t), 1, len(clock) - 1))
    lo = hi - 1
    a, b = values[lo], values[hi]
    frac = np.clip((t - clock[lo]) / (clock[hi] - clock[lo]), 0, 1)
    return (1 - frac) * a + frac * (b if np.dot(a, b) >= 0 else -b)


def append_recovery(clip, console_dir, px4_dir):
    """Add logged drone ground truth during recovery, after validating the pose join."""
    if not clip.result.get("cycle_complete") or any(d["phase"] == "DEPART" for _, d in clip.diag):
        return
    events = recovery_events(console_dir / f"{clip.batch.name}.log", clip.result["trial"])
    if not {"end", "depart", "push", "done"} <= events.keys():
        return

    try:
        from pyulog import ULog
    except ImportError:
        print("Recovery ULogs require pyulog; using trial JSONL only.", file=sys.stderr)
        return
    absolute_end = clip.start + clip.end
    depart = datetime.fromtimestamp(events["depart"], timezone.utc)
    done = absolute_end + events["done"] - events["end"]
    pos_topic = "vehicle_local_position_groundtruth"
    att_topic = "vehicle_attitude_groundtruth"
    cmd_topic = "vehicle_rates_setpoint"
    for offset in (0, 1, -1, 2):
        stamp = depart + timedelta(seconds=offset)
        path = px4_dir / stamp.strftime("%Y-%m-%d/%H_%M_%S.ulg")
        if not path.is_file():
            continue
        topics = {d.name: d.data for d in ULog(str(path), [pos_topic, att_topic, cmd_topic]).data_list}
        if not {pos_topic, att_topic} <= topics.keys():
            continue
        p, q = (topics[k] for k in (pos_topic, att_topic))
        ts, ts_q = (data["timestamp_sample"] * 1e-6 for data in (p, q))
        xyz = ned_to_enu(np.array([p[k] for k in "xyz"])).T
        quats = np.column_stack([q[f"q[{i}]"] for i in range(4)])
        i = np.searchsorted(ts, absolute_end, side="right")
        if i == len(ts) or ts[i] - absolute_end > 0.2 or min(ts[-1], ts_q[-1]) < done:
            continue
        join_q = R_enu_flu_from_px4_q(interpolate_quat(ts[i], ts_q, quats))
        if (np.linalg.norm(xyz[i] - clip.drone[-1]) > 0.30
                or np.trace(join_q.T @ quat_to_rot(clip.drone_q[-1])) < 2.9):
            continue
        mask = (ts > absolute_end) & (ts <= done)
        added = ts[mask] - clip.start
        # Cart control stays on at the measured terminal speed through DEPART.
        # ponytail: retain constant cart velocity over this short gap until a cart pose log covers recovery.
        clip.cart = np.vstack((clip.cart, clip.cart[-1] + (added - clip.pose_t[-1])[:, None] * clip.cart_v))
        clip.cart_q = np.vstack((clip.cart_q, np.repeat(clip.cart_q[-1:], len(added), axis=0)))
        clip.drone = np.vstack((clip.drone, xyz[mask]))
        added_q = [rot_to_quat(R_enu_flu_from_px4_q(interpolate_quat(t, ts_q, quats))) for t in ts[mask]]
        clip.drone_q = np.vstack((clip.drone_q, added_q))
        clip.pose_t = np.r_[clip.pose_t, added]
        clip.duration = done - clip.start
        clip.recovery = {key: clip.end + events[key] - events["end"] for key in ("depart", "push", "level", "done")
                         if key in events}
        if cmd_topic in topics:
            d = topics[cmd_topic]
            t = d["timestamp"] * 1e-6 - clip.start
            keep = (t > clip.end) & (t <= clip.duration)
            for j in np.flatnonzero(keep):
                clip.commands.append((t[j], -d["thrust_body[2]"][j], [d[k][j] for k in ("roll", "pitch", "yaw")]))
        print(f"  DEPART: {path.name} + console; cart extrapolated", flush=True)
        return


class Clip:
    def __init__(self, batch, result, title, ticks, q2_samples, thresholds=None):
        self.batch, self.result, self.title = batch, result, title
        self.thresholds = thresholds or {}
        truth = defaultdict(list)
        for row in jsonl(batch / f"truth_{result['trial']}.jsonl"):
            truth[row["kind"]].append(row)
        poses = sorted(truth["pose"], key=lambda r: r["sim_t"])
        if len(poses) < 2:
            raise ValueError(f"{batch.name}/{result['trial']}: need at least two pose rows")
        self.start = truth["start"][0]["sim_t"] if truth["start"] else poses[0]["sim_t"]
        self.end = (truth["end"][-1] if truth["end"] else poses[-1])["sim_t"] - self.start
        self.duration, self.recovery = self.end, {}
        self.pose_t = np.array([p["sim_t"] - self.start for p in poses])
        self.drone, self.drone_q, self.cart, self.cart_q = (
            np.array([p["models"][name][key] for p in poses], dtype=float)
            for name in (DRONE, CART) for key in ("xyz", "wxyz")
        )
        j = min(len(poses) - 2, np.searchsorted(self.pose_t, self.pose_t[-1] - 0.25))
        self.cart_v = (self.cart[-1] - self.cart[j]) / (self.pose_t[-1] - self.pose_t[j])
        wall_poses = sorted((p["wall"], p["sim_t"] - self.start) for p in poses if "wall" in p)
        wall, sim = zip(*wall_poses) if wall_poses else ([], [])
        raw_diags = sorted(truth["diag"], key=lambda r: r["sim_t"])
        self.diag = [(r["sim_t"] - self.start, r["diag"]) for r in raw_diags]
        self.cut_t = next((t for t, d in self.diag if d["phase"] == "REST" and d.get("armed") is False), None)
        self.touch = result.get("touch") or (truth["touch"][0] if truth["touch"] else None)
        self.touch_t = self.touch["sim_t"] - self.start if self.touch else None
        self.touch_ok = result.get("touch_success")
        if self.touch_ok is None and self.touch:
            self.touch_ok = score_touch(self.touch)
        self.hold = [(r["sim_t"] - self.start, r) for r in sorted(truth["hold"], key=lambda r: r["sim_t"])]
        self.commands = [(t, d.get("thrust_norm"), d.get("rates_frd")) for t, d in self.diag]
        self.tick, self.samples = [], []
        if wall:
            for raw, dest in ((ticks, self.tick), (q2_samples, self.samples)):
                rows = [r for r in raw if wall[0] - 0.05 <= r["wall"] <= wall[-1] + 0.05]
                for row, t in zip(rows, np.interp([r["wall"] for r in rows], wall, sim)):
                    dest.append((float(t), row))
                dest.sort(key=lambda item: item[0])
        self.commands += [(t, r.get("thrust_norm"), r.get("rates_frd")) for t, r in self.tick]
        self.loads = sorted((r["sim_t"] - self.start, r["servo_n"], r["collision_n"]) for r in truth["load"])
        self.q2 = result.get("q2") or {}
        self.immediate_off = self.q2.get("why", "").startswith("rule none:")
        self.decisions = []
        for t, diag in self.diag:
            q2 = diag.get("q2") or {}
            if q2.get("decision") and (not self.decisions or self.decisions[-1][1] != q2):
                self.decisions.append((t, q2))
        self.macros, verify_exit = [], None
        for t, row in self.samples:
            x = row["x"]
            if row["phase"] == "VERIFY" and x[9] in (PH_RELAX, PH_PULL):
                macro = "RELAX" if x[9] == PH_RELAX else f"PULL{x[10]:g}"
                if not self.macros or self.macros[-1][1] != macro:
                    self.macros.append((t, macro))
            elif self.macros and verify_exit is None and row["phase"] != "VERIFY":
                verify_exit = t
        # Q2 result points record elapsed time within each macro. Use these if there is no C5 log.
        points = self.q2.get("points") or []
        if not self.macros and points:
            onset = (self.decisions[0][0] - sum(p["t"] for p in points) if self.decisions
                     else next((t for t, d in self.diag if d["phase"] == "VERIFY"), None))
            if onset is not None:
                for point in points:
                    self.macros.append((onset, point["macro"]))
                    onset += point["t"]
        last_point = self.decisions[0][0] if self.decisions else self.end
        if self.thresholds and verify_exit is not None:
            last_point = verify_exit
        self.points = list(zip([t for t, _ in self.macros[1:]] + [last_point], points))
        slip_rows = [(t, r["x"][N_CH:N_CH + 2] if len(r["x"]) >= N_CH + 2 else [np.nan, np.nan])
                     for t, r in self.samples]
        self.slip, self.slip_window = [], []
        if slip_rows:
            _, slip_values = zip(*slip_rows)
            self.slip = [(t, uv) for t, uv in slip_rows if np.isfinite(uv).all()]
            if self.thresholds:
                for i, (t, row) in enumerate(self.samples):
                    if row["phase"] == "VERIFY" and i >= WIN - 1:
                        # B4e sums each component across the live buffer before taking the norm.
                        uv = np.asarray(slip_values[i - WIN + 1:i + 1], dtype=np.float32)
                        if np.isfinite(uv).all():
                            self.slip_window.append((t, float(np.linalg.norm(uv.sum(axis=0)))))
        self.verify_end = verify_exit if verify_exit is not None else last_point
        self.failure = failure_text(result)
        self.failure_t = self.end
        if result.get("crash_reason") == "detached_after_disarm" and self.cut_t is not None:
            self.failure_t = next((t for t, r in self.hold if t > self.cut_t and not r["engaged"]), self.end)
        elif result.get("crash_reason") in ("ground", "cart"):
            self.failure_t = next((r["sim_t"] - self.start for r in truth["impact"]
                                   if r["surface"] == result["crash_reason"]), self.end)
        elif result.get("crash_reason") == "touch_envelope" and self.touch_t is not None:
            self.failure_t = self.touch_t
        if self.end <= 0 or np.any(np.diff(self.pose_t) <= 0):
            raise ValueError(f"{batch.name}/{result['trial']}: invalid pose clock or trial duration")

    def finish_setup(self):
        for name in ("diag", "tick", "hold", "decisions", "macros", "points"):
            setattr(self, f"{name}_t", [t for t, _ in getattr(self, name)])

    def latest(self, name, t):
        i = bisect_right(getattr(self, f"{name}_t"), t) - 1
        return getattr(self, name)[i] if i >= 0 else (-math.inf, {})

    def state(self, t):
        diag_t, d = self.latest("diag", t)
        tick_t, tick = self.latest("tick", t)
        phase = d.get("phase", "WAIT")
        plate = {}
        if tick_t >= diag_t and t - tick_t <= 0.12:
            phase = tick.get("phase", phase)
            plate = tick.get("plate") or {}
        visible = plate.get("visible", d.get("tag_visible", d.get("det_visible")))
        age = plate.get("age", d.get("plate_age"))
        loss = max(0.0, age - TAG_PERIOD) if age is not None else None
        rho = math.exp(-2 * min(TAG_LOSS_MAX, loss)) if plate and loss is not None else d.get("rho")
        if self.recovery and t >= self.recovery["depart"]:
            phase = "DONE" if t >= self.recovery["done"] - 1e-6 else "DEPART"
        return phase, visible, loss, rho, d


def failure_text(row):
    reason = row.get("crash_reason")
    return {
        "detached_after_disarm": "FALSE-SAFE · Đã OFF, sau đó drone trượt và tuột khỏi tấm.",
        "touch_envelope": "CRASH · Cú chạm nằm ngoài đường bao cho phép.",
        "plate_missed_pad": "CRASH · Thân chạm tấm trước khi đế bám.",
        "ground": "CRASH · Drone va vào mặt đất.",
        "cart": "CRASH · Drone va vào thân xe.",
    }.get(reason, f"CRASH · {reason}" if reason else
          "FALSE-SAFE · Motors OFF nhưng lượt không đạt tiêu chí giữ." if row.get("false_safe") else "")


CART_VERTS = np.array([[x, y, z] for z in (-0.4, 0.4) for x, y in
                       ((-1, -1), (1, -1), (1, 1), (-1, 1))])
CART_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
              [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
PLATE_SQUARE = np.array([[-.225, -.225, 0], [.225, -.225, 0], [.225, .225, 0], [-.225, .225, 0]])
MOTORS = np.array([[x, y, .06] for x, y in ((.174, -.174), (-.174, .174), (.174, .174), (-.174, -.174))])
LEGS = np.array([[[x * .055, y * .065, 0], [x * .125, y * .132, -.2195]]
                 for x in (-1, 1) for y in (-1, 1)])
ANGLES = np.linspace(0, 2 * np.pi, 25)
ROTORS = MOTORS[:, None, :] + .12 * np.column_stack((np.cos(ANGLES), np.sin(ANGLES), np.zeros_like(ANGLES)))
DRONE_FRAME = np.concatenate((MOTORS[:2][None], MOTORS[2:][None], LEGS, [[[0, 0, .06], [.20, 0, .06]]]))


class View:
    def __init__(self, clip, dpi, camera, speed):
        self.clip, self.camera = clip, camera
        fig = self.fig = Figure(figsize=(12.8, 7.2), dpi=dpi, facecolor=BG)
        self.canvas = FigureCanvasAgg(fig)
        fig.text(.05, .951, clip.title, color=FG, fontsize=23, weight="bold", va="top")
        fig.text(.95, .931, f"PHÁT {speed:g}×", ha="right", color=MUTED, fontsize=10, weight="bold")
        fig.text(.05, .038, FOOTER, color=MUTED, fontsize=9)
        fig.text(.95, .038, f"{clip.batch.name}  /  Lượt {clip.result['trial']}", ha="right", color=MUTED, fontsize=9)

        scene_rect = (.034, .226, .579, .621) if clip.thresholds or clip.immediate_off else (.034, .132, .579, .715)
        scene = self.scene = fig.add_axes(scene_rect, projection="3d", facecolor=BG)
        scene.set_axis_off()
        scene.view_init(elev=23, azim=-66)
        scene.set_proj_type("ortho")
        self.floor = Line3DCollection([], colors=GRID, linewidths=.55)
        self.cart = Poly3DCollection([], facecolors=["#263d55", "#46657d", "#304b62",
                                                    "#375368", "#294358", "#3c5970"],
                                     edgecolor="#7692a3", linewidth=.7)
        self.plate = Poly3DCollection([], facecolor=GOLD, edgecolor="#fff0ca", linewidth=1.0)
        self.rotors = Line3DCollection([], colors=[CYAN, "#78aebf", CYAN, "#78aebf"], linewidths=1.3)
        self.frame = Line3DCollection([], colors=[FG, FG, "#b8d3df", "#b8d3df", "#b8d3df", "#b8d3df", CYAN],
                                      linewidths=[3, 3, 1.4, 1.4, 1.4, 1.4, 2.4])
        self.post, = scene.plot([], [], [], color="#9cabb7", lw=2.0)
        self.trail, = scene.plot([], [], [], color=CYAN, lw=1.45, alpha=.73)
        self.contact, = scene.plot([], [], [], marker="o", markersize=5, markeredgecolor=FG, linestyle="")
        for artist in (self.floor, self.cart, self.plate, self.rotors, self.frame):
            scene.add_collection3d(artist)
        fig.text(.05, .857, "Drone & quỹ đạo", color=CYAN, fontsize=10)
        fig.text(.195, .857, "Tấm 0,45 m", color=GOLD, fontsize=10)
        fig.text(.305, .857, "Xe", color="#9bb7cc", fontsize=10)
        self.source = fig.text(.05, .144, "", color=MUTED, fontsize=9)
        self.phase = fig.text(.05, .812, "", color=FG, fontsize=17, weight="bold", va="top")
        self.clock = fig.text(.05, .765, "", color=FG, fontsize=10, va="top")
        self.tag = fig.text(.05, .732, "", color=FG, fontsize=10, va="top")
        if clip.thresholds or clip.immediate_off:
            fig.text(.05, .699, f"Vật liệu   ·   μ = {clip.result['mu']:g}   ·   "
                     f"Lực giữ cài đặt {clip.result['f_hold_n']:g} N", color=MUTED, fontsize=10, va="top")
        self.info = fig.text(.05, .196, "", color=FG, fontsize=10, va="bottom", linespacing=1.35)
        self.outcome = fig.text(.05, .089, "", color=FG, fontsize=12, weight="bold")
        self.setup_charts()
        if camera == "fixed":
            positions = np.vstack((clip.cart, clip.drone))
            lo, hi = positions.min(axis=0), positions.max(axis=0)
            self.frame_scene((lo + hi) / 2, max(5, hi[0] - lo[0] + 3), max(4.5, hi[1] - lo[1] + 3),
                             max(4, hi[2] + 0.6))

    def setup_charts(self):
        c = self.clip
        self.fig.text(.64, .857, "Tín hiệu điều khiển và tiếp xúc", color=FG, fontsize=11, weight="bold")
        self.cursors = []
        t, thrust, rates = zip(*sorted(c.commands, key=lambda r: r[0])) if c.commands else ([], [], [])
        rate_values = np.asarray([r if r is not None else [np.nan] * 3 for r in rates]).reshape(-1, 3)
        load = [(t, r["x"][6]) for t, r in c.samples]
        if load:
            lt, lv = zip(*load)
            load_series = [(lt, lv, GOLD, "C5")]
        elif c.loads:
            lt, servo, collision = zip(*c.loads)
            load_series = [(lt, servo, GOLD, "servo"), (lt, collision, GREEN, "va chạm")]
        else:
            load_series = []
        charts = [("Thrust lệnh [0–1]", [(t, thrust, CYAN, "")]),
                  ("Rate lệnh FRD [rad/s]", [(t, vs, color, label) for vs, color, label in
                                             zip(rate_values.T, (CYAN, GOLD, "#cfb2ff"), ("roll", "pitch", "yaw"))]),
                  ("Load cell [N]" if load or not c.loads else "Lực pháp tuyến HoldPad [N]", load_series)]
        zoom = None
        slip = c.slip
        if c.thresholds and c.macros:
            first = c.macros[0][0]
            zoom = (max(0, first - .04), min(c.duration, c.verify_end + .05))
            slip = [(t, uv) for t, uv in slip if zoom[0] <= t <= zoom[1]]
        if slip or zoom or c.immediate_off:
            st, uv = zip(*slip) if slip else ([], [])
            u, w = np.asarray(uv).reshape(-1, 2).T
            charts.append(("slip_disp quang học [mm/mẫu]", [(st, u, CYAN, "u"), (st, w, GREEN, "w")]))
        if zoom:
            wt, displacement = zip(*c.slip_window) if c.slip_window else ([], [])
            charts.append((f"|Σ slip_disp| · cửa sổ {WIN} mẫu [mm]", [(wt, displacement, CYAN, "|Σ|")]))
        height = .66 / len(charts)
        for i, (title, series) in enumerate(charts):
            ax = self.fig.add_axes((.68, .78 - (i + 1) * height + .035, .267, height - .048), facecolor=PANEL)
            ax.set_title(title, color=FG, fontsize=10, loc="left", pad=5)
            limits = zoom if zoom and i >= 3 else (0, c.duration)
            ax.set_xlim(*limits)
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.grid(color=GRID, linewidth=.55)
            ax.tick_params(colors=MUTED, labelsize=8, length=0, labelbottom=i == len(charts) - 1)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            usable = False
            for times, values, color, label in series:
                y = np.asarray(values, dtype=float)
                if np.isfinite(y).any():
                    ax.plot(times, y, color=color, linewidth=1, label=label)
                    usable = True
            if zoom and i == len(charts) - 1:
                macro_t, thresholds = zip(*[(t, c.thresholds[macro]) for t, macro in c.macros])
                ax.step((*macro_t, c.verify_end), (*thresholds, thresholds[-1]), where="post",
                        color=GOLD, ls="--", lw=1.3, label="Ngưỡng")
                if c.points:
                    pt, pv = zip(*[(t, p["slip_mm"]) for t, p in c.points])
                    ax.plot(pt, pv, "o", color=FG, ms=3.5, label="Điểm chốt")
                ax.set_ylim(bottom=0)
                usable = True
            if not usable:
                ax.set_yticks([])
                label = "Lô này thiếu kênh u/w" if c.immediate_off and i == 3 else "Không có mẫu"
                ax.text(.5, .52, label, color=MUTED, fontsize=9, ha="center", transform=ax.transAxes)
            elif (labels := ax.get_legend_handles_labels()[1]) and len(labels) > 1:
                ax.legend(loc="upper right", ncol=len(labels), frameon=False, fontsize=7, labelcolor=FG,
                          handlelength=1.2, columnspacing=.8)
            if i == 0 and usable:
                ax.set_ylim(-.025, 1.04)
            if c.touch_t is not None:
                ax.axvline(c.touch_t, color=GOLD, alpha=.65, lw=.8, ls="--")
            if c.recovery:
                ax.axvspan(c.end, c.duration, color=COLORS["DEPART"], alpha=.12)
            self.cursors.append(ax.axvline(0, color=FG, lw=.9, alpha=.8))
            if i == len(charts) - 1:
                label = "Cận cảnh VERIFY [s]" if zoom else "Thời gian mô phỏng [s]"
                ax.set_xlabel(label, color=MUTED, fontsize=9, labelpad=3)

    def frame_scene(self, center, xspan, yspan, top):
        lo, hi = np.array(center[:2]) - [xspan / 2, yspan / 2], np.array(center[:2]) + [xspan / 2, yspan / 2]
        self.scene.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(-.05, top))
        self.scene.set_box_aspect((xspan, yspan, top), zoom=1.12)
        grid = [[(x, lo[1], 0), (x, hi[1], 0)] for x in np.arange(math.ceil(lo[0]), hi[0])]
        grid += [[(lo[0], y, 0), (hi[0], y, 0)] for y in np.arange(math.ceil(lo[1]), hi[1])]
        self.floor.set_segments(grid)

    def draw(self, t):
        c = self.clip
        pos = interpolate(t, c.pose_t, c.drone)
        drone_R = quat_to_rot(interpolate_quat(t, c.pose_t, c.drone_q))
        cart = interpolate(t, c.pose_t, c.cart)
        cart_R = quat_to_rot(interpolate_quat(t, c.pose_t, c.cart_q))
        plate_c, plate_R = plate_from_cart(cart, cart_R)
        body = pos + drone_R @ X500_BASE_LINK_IN_MODEL
        self.frame.set_segments(DRONE_FRAME @ drone_R.T + body)
        self.rotors.set_segments(ROTORS @ drone_R.T + body)
        self.cart.set_verts((CART_VERTS @ cart_R.T + cart)[CART_FACES])
        self.plate.set_verts([PLATE_SQUARE @ plate_R.T + plate_c])
        post = np.array([[-.75, 0, .4], [-.75, 0, .71]]) @ cart_R.T + cart
        self.post.set_data_3d(*post.T)
        begin, end = np.searchsorted(c.pose_t, [t - 3.0, t], side="right")
        trail = np.vstack((c.drone[begin:end:2], pos))
        self.trail.set_data_3d(*trail.T)
        if self.camera == "follow":
            center = (pos + cart) / 2
            xspan = max(4.4, abs(pos[0] - cart[0]) + 2.7)
            yspan = max(3.5, abs(pos[1] - cart[1]) + 2.7)
            self.frame_scene(center, xspan, yspan, max(2.7, body[2] + .75))
        if c.touch_t is not None and c.touch_t <= t <= c.touch_t + .7:
            self.contact.set_data_3d(*np.array([plate_c]).T)
            self.contact.set_markerfacecolor(GREEN if c.touch_ok else RED)
        else:
            self.contact.set_data_3d([], [], [])
        self.draw_text(t)
        for cursor in self.cursors:
            cursor.set_xdata([t, t])
        self.canvas.draw()

    def draw_text(self, t):
        c = self.clip
        phase, visible, loss, rho, diag = c.state(t)
        result = c.result
        recovering = bool(c.recovery and t > c.end)
        if recovering:
            visible, loss, rho = None, None, None
        self.phase.set(text=phase, color=COLORS.get(phase, MUTED))
        self.clock.set_text(f"t = {t:05.2f} s   ·   Xe {result.get('speed_mps', np.linalg.norm(c.cart_v)):g} m/s")
        tag = "—" if visible is None else "thấy" if visible else "mất"
        tag_color = MUTED if visible is None else GREEN if visible else RED
        tag_extra = f"   ·   ρ {rho:.3f}" if rho is not None else ""
        if loss is not None:
            tag_extra += f"   ·   mất {loss:.2f} s"
        self.tag.set(text=f"● TAG {tag}{tag_extra}", color=tag_color)
        self.source.set_text("Pose drone: PX4 ground truth  ·  Pose xe: ngoại suy" if recovering else "")
        lines = []
        if c.touch_t is not None and t >= c.touch_t:
            touch = c.touch
            lines.append(f"Chạm {c.touch_t:.2f} s  ·  {'PASS' if c.touch_ok else 'FAIL'} đường bao"
                         f"   ·   vₙ {touch['vn_mps']:.2f} m/s   ·   vₜ {touch['vt_mps']:.2f} m/s"
                         f"   ·   lệch {touch['align_deg']:.1f}°")
        decision_t, q2 = c.latest("decisions", t)
        if not q2 and not c.decisions and t >= c.end:
            q2 = c.q2
        _, macro = c.latest("macros", t)
        _, point = c.latest("points", t)
        if c.thresholds:
            for point_t, point in c.points:
                if t < point_t:
                    break
                threshold = c.thresholds[point["macro"]]
                slip = point["slip_mm"]
                below = slip < threshold
                decision = c.q2["decision"] if point["macro"] == c.q2.get("macro") else "thử tiếp"
                lines.append(f"{point['macro']}  ·  |Σ slip_disp| {slip:.3f} "
                             f"{'<' if below else '≥'} {threshold:.3f} mm  →  {decision}")
            if phase == "VERIFY" and isinstance(macro, str) and t < c.verify_end:
                lines.append(f"Đang đo {macro}  ·  Ngưỡng OFF {c.thresholds[macro]:.3f} mm")
        elif c.immediate_off and q2.get("decision") == "OFF":
            lines.append("Q2 none  ·  Lệnh OFF ngay khi chạm")
        elif q2 or (phase == "VERIFY" and macro):
            macro_text = q2.get("macro") or (macro if isinstance(macro, str) else "")
            point = q2 or point
            probabilities = [f"{k} {point[k]:.3f}" for k in ("p_hold", "p_slip") if point.get(k) is not None]
            if probabilities and point.get("macro") != macro_text:
                probabilities.insert(0, f"điểm {point['macro']}")
            lines.append("   ·   ".join(v for v in (f"Q2 {macro_text}".strip(), *probabilities,
                                                   "OFF" if q2.get("decision") == "OFF" else "") if v))
        status, color = "", FG
        if phase == "REST":
            _, hold = c.latest("hold", t)
            engaged = hold.get("engaged")
            held = "đã tuột" if engaged is False else "đang bám" if engaged else "thiếu mẫu giữ"
            if engaged and hold.get("margin_n", 0) < 0:
                held = "đang bám, biên lực âm"
            rest = f"{t - c.cut_t:.2f} s sau disarm" if c.cut_t is not None and t >= c.cut_t else "chờ disarm"
            lines.append(f"REST  ·  {rest}  ·  {held}")
        elif c.thresholds and recovering and c.cut_t is not None and result.get("held"):
            lines.append(f"REST  ·  Đã giữ {c.end - c.cut_t:.2f} s sau disarm")
        if phase == "DEPART":
            if c.recovery:
                stage = ("tái kích hoạt" if t < c.recovery["push"] else
                         "push-off" if t < c.recovery.get("level", c.duration) else "đã cân bằng")
            elif q2.get("decision") == "ABORT" or (c.thresholds and c.q2.get("decision") == "ABORT"):
                push_t = c.verify_end if c.thresholds else decision_t
                stage = "push-off" if t - push_t < PUSHOFF_SECONDS else "rời tấm sau ABORT"
            else:
                stage = ("push-off" if any("push-off" in event for event in diag.get("events", []))
                         else "rời tấm")
            lines.append(f"DEPART  ·  {stage}")
        if q2.get("decision") in ("ABORT", "SELF_ABORT"):
            status, color = f"Q2 {q2['decision']}  ·  μ = {result.get('mu', '—')}", RED
            if c.thresholds and q2["decision"] == "ABORT":
                status = "B4e ABORT  ·  Động cơ vẫn chạy để rời tấm"
        if t >= c.end and result.get("cycle_complete") and (not c.recovery or phase == "DONE"):
            status = "Chu kỳ hoàn tất · Đã tái kích hoạt và về hover"
            if not c.recovery and phase != "DONE":
                status += " (tổng kết lượt)"
            color = GREEN
        if c.failure and t >= c.failure_t:
            status, color = c.failure, RED
            if c.immediate_off and result.get("rest_slip_m") is not None:
                lines.append(f"Trượt tích lũy trên tấm  ·  {100 * result['rest_slip_m']:.2f} cm")
        self.info.set_text("\n".join(lines))
        self.outcome.set(text=status, color=color)


def output_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".mp4", delete=False)


def summary_canvas(results, groups, dpi):
    def fraction(metric):
        n = metric["n"]
        k = metric["k"] if "k" in metric else round(metric["p"] * n)
        return f"{k}/{n}"

    main = [json.loads(path.read_text(encoding="utf-8")) for path in results.glob("sitl_main_p2y_v*.json")]
    if not main:
        raise ValueError(f"{results}: missing SITL main p2y results")
    main.sort(key=lambda row: row["config"]["speeds"][0])
    main_rows = [[f"{row['config']['speeds'][0]:g} m/s".replace(".", ","),
                  *(fraction(row["eval"]["nominal"][metric])
                    for metric in ("touch_success", "valid_perch", "cycle_complete"))] for row in main]
    q2_rows = [[label, *(fraction(groups[key][metric])
                        for metric in ("valid_perch", "off_rate", "false_safe_v12"))]
               for key, label in (("b4e", "B4e"), ("none", "Tắt ngay"), ("dthr_v7", "D-thr v7"))]

    fig = Figure(figsize=(12.8, 7.2), dpi=dpi, facecolor=BG)
    fig.text(.07, .91, "Kết quả SITL", color=FG, fontsize=30, weight="bold")
    speed = groups["b4e"]["config"]["speeds"][0]
    seeds = " + ".join(map(str, groups["b4e"]["eval_seeds"]))
    fig.text(.07, .80, f"Q2  /  Xe {speed:g} m/s  ·  Seed {seeds}", color=CYAN, fontsize=15, weight="bold")
    fig.text(.07, .41, "p2y  /  Điều kiện danh định theo tốc độ xe", color=CYAN, fontsize=15, weight="bold")
    for y, rows, columns in (
        (.52, q2_rows, ("Luật VERIFY", "Đậu hợp lệ / lượt", "Lệnh OFF / lượt", "False-safe v1.2 / OFF")),
        (.13, main_rows, ("Tốc độ xe", "Chạm đạt / lượt", "Đậu hợp lệ / lượt", "Trọn chu kỳ / lượt")),
    ):
        ax = fig.add_axes((.07, y, .86, .25), frame_on=False)
        ax.set_axis_off()
        table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", bbox=(0, 0, 1, 1))
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        for (row, _), cell in table.get_celld().items():
            cell.set(facecolor=GRID if row == 0 else PANEL, edgecolor=BG)
            cell.get_text().set(color=MUTED if row == 0 else GREEN if y > .5 and row == 1 else FG,
                                weight="bold" if row == 0 else "normal")
    fig.text(.07, .475, "False-safe v1.2  ·  Mất bám sau lệnh OFF trong cửa sổ REST", color=MUTED, fontsize=10)
    fig.text(.07, .055, "Nguồn  ·  sitl_q2_pooled.json  /  sitl_main_p2y_v*.json", color=MUTED, fontsize=10)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return canvas


def render(clips, args, summary=None):
    output = args.out
    try:
        temporary = output_file(output)
    except OSError as exc:
        if output != DEFAULT_OUT or exc.errno not in (1, 13, 30):
            raise
        output = CODE_ROOT / "video" / output.name
        print(f"{exc.strerror}; using {output}", flush=True)
        temporary = output_file(output)
    temporary.close()
    target = Path(temporary.name)
    total = 0
    started = time.monotonic()
    try:
        for index, clip in enumerate(clips):
            view = View(clip, args.dpi, args.camera, args.speed)
            if index == 0:
                width, height = view.canvas.get_width_height()
                command = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                    "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-framerate", str(args.fps), "-i", "-",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-an", "-c:v", "libx264", "-threads", "2",
                    "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
                ]
                encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
            frames = max(1, math.ceil(clip.duration * args.fps / args.speed - 1e-8))
            tail = round(1.6 * args.fps) if args.montage else 0
            print(f"  {clip.batch.name}/{clip.result['trial']}: log {clip.duration:.3f} s, "
                  f"{frames + tail} frames", flush=True)
            for frame in range(0, frames, args.decimate):
                view.draw(min(clip.duration, frame * args.speed / args.fps))
                pixels = view.canvas.buffer_rgba().cast("B")
                for _ in range(min(args.decimate, frames - frame)):
                    encoder.stdin.write(pixels)
            if tail:
                view.draw(clip.duration)
                pixels = view.canvas.buffer_rgba().cast("B")
                for _ in range(tail):
                    encoder.stdin.write(pixels)
            total += frames + tail
        if summary is not None:
            pixels = summary.buffer_rgba().cast("B")
            frames = SUMMARY_SECONDS * args.fps
            for _ in range(frames):
                encoder.stdin.write(pixels)
            total += frames
            print(f"  Summary: {frames} frames", flush=True)
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg failed to encode video")
        target.replace(output)
    finally:
        if "encoder" in locals() and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        target.unlink(missing_ok=True)
    print(f"{output}: {total} frames, {total / args.fps:.3f} s; render {time.monotonic() - started:.1f} s",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, help="batch directory, or the SITL evaluation root with --montage")
    parser.add_argument("--trial", type=int, action="append", help="trial ID (repeatable)")
    parser.add_argument("--montage", action="store_true", help="join five demo scenes and a five-second results card")
    parser.add_argument("--out", type=Path, help="MP4 path (montage default: video/perchpilot_demo_v2.mp4)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=1.0, help="simulated seconds per video second")
    parser.add_argument("--dpi", type=int, default=100, help="100 gives 1280×720")
    parser.add_argument("--decimate", type=int, default=1, help="render every Nth frame, repeating it at output FPS")
    parser.add_argument("--camera", choices=("follow", "fixed"), default="follow")
    parser.add_argument("--console-dir", type=Path, help="optional batch console logs for recovery (default: logs_agents)")
    parser.add_argument("--px4-log-dir", type=Path, help="optional PX4 ULog archive for recovery (requires pyulog)")
    args = parser.parse_args()
    if args.montage and args.trial or not args.montage and (not args.batch or not args.trial):
        parser.error("supply --batch DIR --trial ID [--trial ID ...], or --montage [--batch SITL_ROOT]")
    if min(args.fps, args.dpi, args.decimate) < 1 or not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--fps, --speed, --dpi and --decimate must be positive")
    args.out = args.out or (MONTAGE_OUT if args.montage else DEFAULT_OUT)
    log_root = (args.batch or ROOT / "sitl_eval") if args.montage else args.batch.parent
    console_dir = args.console_dir or log_root.parent / "logs_agents"
    px4_dir = args.px4_log_dir or log_root.parent / "sitl/px4_i7/log"
    try:
        groups = {}
        if args.montage or args.batch.name.startswith("q2_b4e_qos"):
            groups = json.loads((RESULTS / "sitl_q2_pooled.json").read_text(encoding="utf-8"))["groups"]
        by_batch = {name: b4e_thresholds(group) for key in ("b4e", "b4e_v15") if (group := groups.get(key))
                    for name in group["batches"]}
        summary = None
        if args.montage:
            selected = list(select_montage(log_root, by_batch["q2_b4e_qos"]))
            summary = summary_canvas(RESULTS, groups, args.dpi)
        else:
            records = {r["trial"]: r for r in jsonl(args.batch / "trials.jsonl")}
            missing = [str(t) for t in args.trial if t not in records]
            if missing:
                raise ValueError(f"{args.batch.name}: trial ID(s) not found: {', '.join(missing)}")
            selected = [(args.batch, records[t], f"PerchPilot  /  Lượt {t}") for t in args.trial]
        print("Selected trials:", flush=True)
        for batch, row, _ in selected:
            print(f"  {batch.name}/{row['trial']} ({row.get('outcome', '—')})", flush=True)
        clips, cached = [], {}
        for batch, row, title in selected:
            if batch not in cached:
                cached[batch] = [trial_rows(batch, name) for name in ("node_ticks.jsonl", "q2_samples.jsonl")]
            ticks, samples = ([r for r in rows if r.get("trial") == row["trial"]] for rows in cached[batch])
            clip = Clip(batch, row, title, ticks, samples, by_batch.get(batch.name))
            append_recovery(clip, console_dir, px4_dir)
            clip.finish_setup()
            clips.append(clip)
        render(clips, args, summary)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
