"""Replay regression tests. No ROS or simulator process is started."""
from contextlib import closing, contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import os
import socket
import threading
import time
from unittest.mock import patch

import pytest

from perch_dashboard import DashboardServer, Handler, Replay, SERIES


def write_rows(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def sse(response):
    event = {}
    while line := response.readline().decode():
        if line.startswith("event: "):
            event["event"] = line[7:].strip()
        if line.startswith("data: "):
            event["data"] = json.loads(line[6:])
        if line == "\n" and event:
            yield event
            event = {}


class PipeSocket:
    def __init__(self, sock):
        self.sock = sock

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def sendall(self, data, flags=0):
        # The sandbox allows pipe writes on an anonymous socketpair.
        with os.fdopen(os.dup(self.fileno()), "wb") as pipe:
            pipe.write(data)


@contextmanager
def replay_http(replay):
    try:
        server = DashboardServer(0, replay)
    except PermissionError:
        # The sandbox forbids TCP and UNIX bind. Verify the requested address and run
        # the actual HTTP handler on a socketpair instead.
        with patch.object(ThreadingHTTPServer, "__init__") as bind:
            server = DashboardServer(0, replay)
        bind.assert_called_once_with(("127.0.0.1", 0), Handler)
        conn = HTTPConnection("127.0.0.1", timeout=5)

        def handle(peer):
            with peer:
                Handler(PipeSocket(peer), ("127.0.0.1", 0), server)

        def connect():
            local, peer = socket.socketpair()
            conn.sock = PipeSocket(local)
            conn.sock.settimeout(5)
            threading.Thread(target=handle, args=(peer,), daemon=True).start()

        conn.connect = connect
        with closing(conn):
            yield conn
        return

    with server, closing(HTTPConnection(*server.server_address, timeout=5)) as conn:
        assert server.socket.getsockname()[0] == "127.0.0.1"
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield conn
        finally:
            server.shutdown()
            thread.join()


def test_replay_http_sse_loopback(tmp_path):
    base = 100.0
    write_rows(tmp_path / "node_ticks.jsonl", [
        {"wall": base + t, "trial": 1, "phase": "PERCH" if t < 0.5 else "VERIFY",
         "ekf": {"p": [3, 2, 1]}, "plate": {"c": [1, 2, 0], "age": 0.01, "visible": True},
         "thrust_norm": 0.25, "rates_frd": [1, 0, -1]} for t in (0, 0.25, 0.5, 1, 1.5, 2)
    ])
    write_rows(tmp_path / "q2_samples.jsonl", [
        {"wall": base + i / 200, "trial": 1, "phase": "VERIFY" if i >= 100 else "PERCH",
         "x": [0] * 6 + [12, 1, 0.3, 2, 0.3, 0.01, 0]} for i in range(401)
    ])
    replay = Replay(tmp_path, trial=1)
    with replay_http(replay) as conn:
        conn.request("GET", "/")
        response = conn.getresponse()
        assert response.status == 200
        assert b"PerchPilot" in response.read()
        conn.request("GET", "/events")
        response = conn.getresponse()
        assert response.status == 200
        assert response.headers["Content-Type"] == "text/event-stream"
        started = time.monotonic()
        events = list(sse(response))
        assert 1.8 <= time.monotonic() - started < 5.0
        assert events[0] == {"event": "meta", "data": {"mode": "replay", "duration": 2.0, "speed": 1}}
        assert events[-1]["event"] == "end"
        frames = [e["data"] for e in events if "event" not in e]
        assert len(frames) >= 35
        assert frames[0]["t"] == 0 and frames[-1]["t"] == 2
        assert frames[-1]["phases"] == ["PERCH", "VERIFY"]
        assert frames[-1]["pose"]["enu"] == [2, 0, 1]
        assert all(frame["truth"] is None for frame in frames)
        assert frames[-1]["series"]["b4e"][-1][1] == pytest.approx(0.8)

        conn.request("GET", "/events?start=1.8&pause=1")
        events = list(sse(conn.getresponse()))
        assert len(events) == 3
        assert events[1]["data"]["t"] == 1.8
        assert all(events[1]["data"]["series"][name] for name in SERIES)
        assert events[-1]["event"] == "end"


def test_replay_keeps_evaluator_truth_separate(tmp_path):
    score = {"vn_mps": 1.2, "vt_mps": 0.6, "align_deg": 11.0}
    write_rows(tmp_path / "trials.jsonl", [
        {"trial": 1, "touch": score, "touch_success": True, "motors_off": False, "outcome": "q2_abort",
         "q2": {"decision": "ABORT"}}
    ])
    write_rows(tmp_path / "truth_1.jsonl", [
        {"kind": "start", "wall": 100},
        {"kind": "diag", "wall": 100, "diag": {"phase": "VERIFY", "trial": 1, "rho": 0.8}},
        {"kind": "touch", "wall": 100.5, **score},
        {"kind": "pose", "wall": 100.5, "models": {"drone": {"xyz": [1, 2, 3]}}},
        {"kind": "load", "wall": 100.5, "servo_n": 10},
        {"kind": "end", "wall": 101}
    ])
    replay = Replay(tmp_path, trial=1)
    assert replay.frames[0]["truth"]["batch"]["touch_pass"] == 1
    assert "touch" not in replay.frames[0]["truth"]
    assert replay.frames[-1]["truth"]["touch"] == score
    assert replay.frames[-1]["truth"]["touch_pass"] is True
    assert all(f["pose"] is None and not f["series"]["sensors"] and f["q2"] is None for f in replay.frames)
