"""Read-only PerchPilot dashboard. Replay needs the occwm venv; live also needs sourced ROS 2."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import shlex
import threading
import time
from urllib.parse import parse_qs, urlsplit

from eval_node import dashboard_batch, score_touch
from perch_core import TAG_LOSS_MAX
from perch_q2 import N_CH, PH_PULL, PH_RELAX, SCHEDULE_V3, WIN
from plate_perception import TAG_PERIOD

HOST = "127.0.0.1"
ROS_DOMAIN = 77
DIAG_TOPIC = "/perchpilot/diag"
SENSOR_TOPIC = "/perch/hold_sensors"
ODOM_TOPIC = "/px4_7/fmu/out/vehicle_odometry"
EVAL_TOPIC = "/perchpilot/eval"
PAGE = Path(__file__).with_suffix(".html")
INTERVAL = 0.05
HISTORY = 10.0
SERIES = ("commands", "sensors", "b4e")


def jsonl(path):
    if path.is_file():
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except ValueError as exc:
                        raise ValueError(f"{path}:{number}: {exc}") from exc


def threshold_config(directory):
    name = f"sitl_{directory.name}.json"
    for path in (directory / name, PAGE.parents[1] / "results" / name):
        if path.is_file():
            data = json.loads(path.read_text())
            env = data.get("provenance", {}).get("env", {})
            args = shlex.split(env.get("PERCH_EFFECTIVE_NODE_ARGS") or env.get("PERCH_NODE_ARGS") or "")
            if "--q2-slip-off" in args:
                start = args.index("--q2-slip-off") + 1
                values = list(map(float, args[start:start + len(SCHEDULE_V3)]))
                if len(values) == len(SCHEDULE_V3) and all(math.isfinite(v) and v >= 0 for v in values):
                    return dict(zip(SCHEDULE_V3, values))
    return {}


def recovery_phases(directory, trial, end):
    """The evaluator closes truth before recovery. Align console transitions to its trial summary."""
    filename = f"{directory.name}.log"
    paths = (directory / filename, directory.parent.parent / "logs_agents" / filename)
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        return []
    phases = []
    summary_time = None
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            summary = re.search(r"\[(\d+\.\d+)\] \[perchpilot_eval_node\]: Trial (\d+)/", line)
            if summary:
                if summary_time is not None:
                    break
                if int(summary[2]) == trial:
                    summary_time = float(summary[1])
            if summary_time is not None:
                change = re.search(r"\[(\d+\.\d+)\] \[perchpilot_perch_node\]: \[phase\] (REST->DEPART|DEPART->DONE):", line)
                if change:
                    phases.append((end + float(change[1]) - summary_time, change[2].split("->")[1]))
                    if phases[-1][1] == "DONE":
                        break
    return phases if phases and phases[0][1] == "DEPART" else []


class Monitor:
    """Sources for the flight display and the evaluator arrive independently."""

    def __init__(self, source, thresholds=None):
        self.source = source
        self.thresholds = thresholds or {}
        self.trial = None
        self.phase = None
        self.phases = []
        self.diag = {}
        self.pose = None
        self.q2 = None
        self.macro = None
        self.last_point = None
        self.truth = None
        self.series = {name: [] for name in SERIES}

    def set_phase(self, phase):
        self.phase = phase
        if phase and phase != "WAIT" and phase not in self.phases:
            self.phases.append(phase)

    def flight(self, t, row):
        trial = row.get("trial")
        phase = row.get("phase")
        if trial != self.trial or (phase == "APPROACH" and self.phase != "APPROACH"):
            self.trial = trial
            self.phases = []
            self.q2 = self.macro = self.last_point = self.pose = None
        self.set_phase(phase)
        self.diag = row
        self.position(t, row.get("p_enu"))
        self.series["commands"].append((t, row.get("thrust_norm"), *(row.get("rates_frd") or (None,) * 3)))

        if row.get("b4e_thresholds"):
            self.thresholds = row["b4e_thresholds"]
        if probe := row.get("q2_probe"):
            self.macro = probe["macro"]
            self.last_point = probe.get("last_point")
            if probe.get("slip_mm") is not None:
                self.b4e(t, probe["slip_mm"], self.macro)
        if q2 := row.get("q2"):
            if q2 != self.q2 and q2.get("slip_mm") is not None:
                self.b4e(t, q2["slip_mm"], q2.get("macro"))
            self.q2 = q2
        elif self.phase == "REST" and row.get("armed") is False and not self.q2:
            self.q2 = {"decision": "OFF", "why": "REST · PX4 đã disarm"}

    def position(self, t, position):
        center = self.diag.get("plate_c_enu")
        if position is not None and center is not None:
            self.pose = {"t": t, "enu": [p - c for p, c in zip(position, center)]}

    def b4e(self, t, slip, macro):
        self.series["b4e"].append((t, slip, self.thresholds.get(macro)))

    def frame(self, t):
        age = self.diag.get("plate_age")
        truth = self.truth
        if truth and truth.get("trial") not in (None, self.trial):
            truth = {"batch": truth.get("batch")}
        payload = {"t": t, "source": self.source, "trial": self.trial, "phase": self.phase,
                   "phases": list(self.phases), "tag_visible": self.diag.get("tag_visible"),
                   "rho": self.diag.get("rho"), "loss": max(0.0, age - TAG_PERIOD) if age is not None else None,
                   "pose": self.pose, "q2": self.q2, "macro": self.macro if self.phase == "VERIFY" else None,
                   "last_point": self.last_point, "thresholds": self.thresholds,
                   "truth": truth, "series": self.series}
        self.series = {name: [] for name in SERIES}
        return payload


def tick_diag(row):
    plate = row.get("plate") or {}
    age = plate.get("age")
    return {"trial": row["trial"], "phase": row["phase"], "plate_age": age, "tag_visible": plate.get("visible"),
            "rho": math.exp(-2 * min(TAG_LOSS_MAX, max(0.0, age - TAG_PERIOD))) if age is not None else None,
            "p_enu": (row.get("ekf") or {}).get("p"), "plate_c_enu": plate.get("c"),
            "thrust_norm": row.get("thrust_norm"), "rates_frd": row.get("rates_frd")}


class Replay:
    def __init__(self, directory, trial, speed=1.0):
        self.speed = speed
        result_rows = list(jsonl(directory / "trials.jsonl"))
        result = next((r for r in result_rows if r["trial"] == trial), {})
        evaluation = [r for r in jsonl(directory / f"truth_{trial}.jsonl")
                      if r["kind"] in ("start", "diag", "touch", "end")]
        # Old evaluator archives may only have sim_t. Current archives share CLOCK_MONOTONIC with the node logs.
        use_wall = any("wall" in row for row in evaluation)
        anchor = next((r for r in evaluation if "wall" in r and "sim_t" in r), {})

        def stamp(row):
            if use_wall:
                return row["wall"] if "wall" in row else anchor["wall"] + row["sim_t"] - anchor["sim_t"]
            return row["sim_t"]

        events = [(stamp(r), "diag", r["diag"]) for r in evaluation if r["kind"] == "diag"]
        for name, kind in (("node_ticks.jsonl", "tick"), ("q2_samples.jsonl", "sample")):
            events.extend((r["wall"], kind, r) for r in jsonl(directory / name)
                          if r.get("trial") == trial and ("wall" in r) and (use_wall or not evaluation))
        if not events:
            raise ValueError(f"không có telemetry cho lượt {trial} trong {directory}")

        model = Monitor(directory.name, threshold_config(directory))
        start = next((stamp(r) for r in evaluation if r["kind"] == "start"), min(e[0] for e in events))
        end = next((stamp(r) for r in reversed(evaluation) if r["kind"] == "end"), max(e[0] for e in events))
        stop = end
        if evaluation:
            model.truth = {"trial": trial, "batch": dashboard_batch(result_rows) if result_rows else None}
            touch = next((r for r in evaluation if r["kind"] == "touch"), None)
            if touch:
                touch_data = result.get("touch") or touch
                passed = result.get("touch_success")
                if passed is None:
                    passed = score_touch(touch_data)
                events.append((stamp(touch), "truth", {"touch": touch_data, "touch_pass": passed}))
            events.append((end, "truth", {"outcome": result.get("outcome"), "cycle_complete": result.get("cycle_complete")}))
            if result.get("cycle_complete"):
                for when, phase in recovery_phases(directory, trial, end):
                    events.append((when, "phase", phase))
                    stop = max(when, stop)

        start = min(start, end)
        self.duration = round(max(0.0, stop - start), 5)
        events = [(max(0.0, t - start), kind, row) for t, kind, row in events if start - 0.00001 <= t <= stop]
        events.sort(key=lambda event: event[0])
        q2 = next((row["q2"] for _, kind, row in reversed(events) if kind == "diag" and row.get("q2")), {})
        points = {point["macro"]: point for point in q2.get("points", [])}
        slip_window = deque(maxlen=WIN)
        previous_sample = -math.inf
        index = 0
        self.frames = []
        for n in range(math.ceil(self.duration / INTERVAL) + 1):
            t = min(round(n * INTERVAL, 5), self.duration)
            while index < len(events) and events[index][0] <= t + 1e-5:
                stamp_s, kind, row = events[index]
                index += 1
                if kind == "diag":
                    model.flight(stamp_s, row)
                elif kind == "tick":
                    model.flight(stamp_s, tick_diag(row))
                elif kind == "truth":
                    model.truth = {**model.truth, **row}
                elif kind == "phase":
                    model.diag = {}
                    model.set_phase(row)
                else:
                    x = row["x"]
                    if stamp_s - previous_sample > 0.2 or len(x) < N_CH + 2:
                        slip_window.clear()
                    previous_sample = stamp_s
                    slip = x[N_CH:N_CH + 2] if len(x) >= N_CH + 2 else (None, None)
                    model.series["sensors"].append((stamp_s, x[6], *slip))
                    if len(x) >= N_CH + 2:
                        slip_window.append(slip)
                    macro = sample_macro(row)
                    if macro:
                        if model.macro and macro != model.macro:
                            model.last_point = points.get(model.macro)
                        model.macro = macro
                        if len(slip_window) == WIN:
                            model.b4e(stamp_s, math.hypot(*(math.fsum(axis) for axis in zip(*slip_window))), macro)
            self.frames.append(model.frame(t))
        self.times = [frame["t"] for frame in self.frames]


def sample_macro(row):
    if row["phase"] == "VERIFY":
        x = row["x"]
        if x[9] == PH_RELAX:
            return "RELAX"
        if x[9] == PH_PULL:
            return f"PULL{x[10]:g}"
    return None


def with_history(frames):
    last = frames[-1]
    since = last["t"] - HISTORY
    return {**last, "series": {name: [p for frame in frames for p in frame["series"][name] if p[0] >= since]
                               for name in SERIES}}


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, port, replay=None):
        self.replay = replay
        self.page = PAGE.read_bytes()
        self.stop = threading.Event()
        self.live_frames = deque(maxlen=math.ceil(HISTORY / INTERVAL) + 1)
        self.lock = threading.Lock()
        super().__init__((HOST, port), Handler)

    def server_close(self):
        self.stop.set()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(self.server.page)))
            self.end_headers()
            self.wfile.write(self.server.page)
        elif url.path == "/events":
            query = parse_qs(url.query)
            try:
                start = float(query.get("start", ["0"])[0])
                if not math.isfinite(start) or start < 0:
                    raise ValueError
            except ValueError:
                self.send_error(400, "Invalid replay start")
                return
            try:
                self.connection.settimeout(5)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                if self.server.replay:
                    self.replay(start, query.get("pause") == ["1"])
                else:
                    self.live()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
        else:
            self.send_error(404)

    def event(self, value, name=None):
        header = f"event: {name}\n" if name else ""
        self.wfile.write(f"{header}data: {json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))}\n\n".encode())

    def replay(self, start, paused):
        replay = self.server.replay
        index = bisect_right(replay.times, start) - 1
        first = bisect_right(replay.times, replay.times[index] - HISTORY - INTERVAL)
        self.event({"mode": "replay", "duration": replay.duration, "speed": replay.speed}, "meta")
        self.event(with_history(replay.frames[first:index + 1]))
        if not paused:
            start_wall = time.monotonic()
            for frame in replay.frames[index + 1:]:
                target = start_wall + (frame["t"] - replay.times[index]) / replay.speed
                while (wait := target - time.monotonic()) > 0:
                    if self.server.stop.wait(min(wait, 2.0)):
                        return
                    if wait > 2:
                        self.wfile.write(b": keepalive\n\n")
                self.event(frame)
        self.event({}, "end")

    def live(self):
        self.event({"mode": "live"}, "meta")
        previous = None
        next_ping = time.monotonic() + 2
        while not self.server.stop.is_set():
            with self.server.lock:
                frames = list(self.server.live_frames)
            if frames and previous is None:
                self.event(with_history(frames))
                previous = frames[-1]["t"]
            elif frames:
                for frame in frames:
                    if frame["t"] > previous:
                        self.event(frame)
                previous = frames[-1]["t"]
            if time.monotonic() >= next_ping:
                self.wfile.write(b": keepalive\n\n")
                next_ping = time.monotonic() + 2
            self.server.stop.wait(INTERVAL)

    def log_message(self, *args):
        pass


def run_live(server):
    import rclpy
    from px4_msgs.msg import VehicleOdometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray, String

    model = Monitor("/px4_7")
    started = time.monotonic()
    last_diag = -math.inf
    odom = None

    def seconds():
        return time.monotonic() - started

    def on_diag(msg):
        nonlocal last_diag
        t = seconds()
        row = json.loads(msg.data)
        # Keep the policy pose for older/missing odometry, and prefer a recent subscribed EKF2 position.
        if odom is not None and t - odom[0] <= 0.2:
            row["p_enu"] = odom[1]
        model.flight(t, row)
        last_diag = t

    def on_odom(msg):
        nonlocal odom
        if msg.pose_frame == VehicleOdometry.POSE_FRAME_NED and all(map(math.isfinite, msg.position)):
            n, e, d = msg.position
            odom = (seconds(), (e, n, -d))
            if odom[0] - last_diag <= 0.2:
                model.position(*odom)

    def on_sensor(msg):
        if len(msg.data) in (3, 5) and all(map(math.isfinite, msg.data[2:])):
            slip = tuple(msg.data[3:]) or (None, None)
            model.series["sensors"].append((seconds(), msg.data[2], *slip))

    def on_eval(msg):
        model.truth = json.loads(msg.data)

    def snapshot():
        t = seconds()
        if t - last_diag > 0.5:
            model.diag = {}
            model.phase = None
        if model.truth and node.count_publishers(EVAL_TOPIC) == 0:
            model.truth = None
        with server.lock:
            server.live_frames.append(model.frame(t))

    rclpy.init(args=[], domain_id=ROS_DOMAIN)
    node = Node("perchpilot_dashboard")
    try:
        node.create_subscription(String, DIAG_TOPIC, on_diag, 10)
        node.create_subscription(Float64MultiArray, SENSOR_TOPIC, on_sensor, 50)
        node.create_subscription(VehicleOdometry, ODOM_TOPIC, on_odom,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        node.create_subscription(String, EVAL_TOPIC, on_eval,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        node.create_timer(INTERVAL, snapshot)
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--replay", type=Path, metavar="DIR")
    parser.add_argument("--trial", type=int, metavar="ID")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    if args.replay and (args.trial is None or args.trial < 1):
        parser.error("--replay requires a positive --trial")
    if args.live and (args.trial is not None or args.speed != 1.0):
        parser.error("--trial and --speed apply to --replay")
    try:
        replay = Replay(args.replay.resolve(), args.trial, args.speed) if args.replay else None
        with DashboardServer(args.port, replay) as server:
            print(f"PerchPilot → http://{HOST}:{args.port}", flush=True)
            if replay is not None:
                server.serve_forever()
            else:
                http = threading.Thread(target=server.serve_forever, daemon=True)
                http.start()
                try:
                    run_live(server)
                finally:
                    server.shutdown()
                    http.join()
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
