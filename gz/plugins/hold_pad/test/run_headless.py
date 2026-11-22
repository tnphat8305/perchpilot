"""Chạy hai ca HoldPad theo C6, lưu log gz topic và chấm từ truth + pose."""

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def records(path):
    decoder = json.JSONDecoder()
    data = path.read_text()
    index = 0
    while index < len(data):
        if data[index].isspace():
            index += 1
            continue
        message, index = decoder.raw_decode(data, index)
        yield message


def check(logs, strong):
    truth = [row["data"] for row in records(logs / "hold_truth.json")]
    expected_params = [0.9, 58.86] if strong else [0.2, 5.886]
    applied = [row["data"] for row in records(logs / "params.json")]
    if applied != [expected_params]:
        raise ValueError(f"Thiếu xác nhận vật liệu {expected_params}: {applied}")
    touch = next(records(logs / "touch.json"), {}).get("data", [])
    if len(touch) != 8 or abs(touch[3]) > 0.01 or not (0.3 < touch[5] < 0.5):
        raise ValueError(f"Truth chạm không khớp hình học/vận tốc của box: {touch}")
    poses = {}
    for row in records(logs / "pose.json"):
        stamp = row["header"]["stamp"]
        millis = 1000 * int(stamp.get("sec", 0)) + round(int(stamp.get("nsec", 0)) / 1e6)
        poses[millis] = tuple(row["position"].get(axis, 0.0) for axis in "xyz")

    first = next((i for i, row in enumerate(truth) if row[1] == 1), None)
    if first is None or first == 0:
        raise ValueError("Thiếu chuyển tiếp 0 → 1 trên /perch/hold_truth")
    if any(abs(y[0] - x[0] - 0.005) > 1e-7 for x, y in zip(truth, truth[1:])):
        raise ValueError("Chuỗi truth có khoảng lấy mẫu khác 5 ms")

    t, _, margin, _ = truth[first]
    t_ms = round(t * 1000)
    tilt = math.radians(65)
    normal = (-math.sin(tilt), 0, math.cos(tilt))
    contact = [poses[step] for step in range(t_ms - 10, t_ms + 11)]
    approach = -sum((y - x) * n for x, y, n in zip(contact[0], contact[1], normal)) / 0.001
    jump = max(math.dist(x, y) for x, y in zip(contact, contact[1:]))
    if jump >= 0.02:
        raise ValueError(f"Bước dịch quanh lúc gắn {jump:.6g} m")
    print(f"Gắn ở t={t:.3f} s, margin={margin:.4f} N, "
          f"v_n ngay trước gắn {approach:.4f} m/s, bước dịch lớn nhất {jump:.6g} m")

    if strong:
        window = truth[first:first + 1001]
        if len(window) != 1001 or any(row[1] != 1 for row in window):
            raise ValueError("Box không bám liên tục đủ 5 s")
        drift = max(math.dist(poses[t_ms], poses[step]) for step in
                    range(t_ms, t_ms + 5001))
        print(f"Bám 5,000 s, độ trôi 3D lớn nhất {drift:.6g} m")
        if drift >= 0.01:
            raise ValueError("Độ trôi ≥ 1 cm")
    else:
        release = next((i for i in range(first + 1, len(truth)) if truth[i][1] == 0), None)
        if release is None or any(row[1] for row in truth[release:]):
            raise ValueError("Box chưa nhả hẳn")
        slips = [row[3] for row in truth[first:release] if row[2] < 0 and row[3] > 0]
        release_time = truth[release][0]
        x, _, z = poses[round(release_time * 1000)]
        plate_x = x * math.cos(tilt) + (z - 1.5) * math.sin(tilt)
        if not slips or abs(plate_x) < 0.224:
            raise ValueError("Thiếu mẫu trượt trước khi box rời mép tấm")
        print(f"Trượt tối đa {max(slips):.4f} m/s, nhả ở t={release_time:.3f} s")


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def run_case(world, logs, env, strong):
    def request(*args):
        return subprocess.run(["gz", *args], env=env, text=True, capture_output=True,
                              timeout=5)

    with ExitStack() as stack:
        def start(log, *args):
            output = stack.enter_context((logs / log).open("w"))
            process = subprocess.Popen(
                ["timeout", "--kill-after=2s", "45s", "gz", *args],
                env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            )
            stack.callback(stop, process)
            return process

        memory = subprocess.check_output(["free", "-m"], text=True)
        print(memory, end="", flush=True)
        available = int(next(row.split()[-1] for row in memory.splitlines()
                             if row.startswith("Mem:")))
        if available < 1500:
            raise ValueError(f"C6 yêu cầu ≥ 1500 MB RAM available, hiện có {available} MB")

        # Chờ server ở trạng thái pause để nối subscriber trước khi thả box.
        server = start("server.log", "sim", "-s", "-v", "3", str(world))
        control = "/world/hold_test/control"
        deadline = time.monotonic() + 12
        while server.poll() is None and time.monotonic() < deadline:
            result = request("service", "-l")
            if control in result.stdout.splitlines():
                break
            if result.returncode:
                raise ValueError(result.stderr.strip() or f"gz service -l: {result.returncode}")
            time.sleep(0.2)
        else:
            raise ValueError("Không thấy dịch vụ điều khiển world")

        echoes = [start(f"{log}.json", "topic", "-e", "-t", topic,
                        "--json-output", "-d", "18")
                  for topic, log in (("/perch/hold_truth", "hold_truth"),
                                     ("/perch/hold_sensors", "hold_sensors"),
                                     ("/perch/touch_truth", "touch"),
                                     ("/perch/hold_params_applied", "params"),
                                     ("/model/hold_box/pose", "pose"))]
        for args in (
            ("topic", "-t", "/perch/hold_params", "-m", "gz.msgs.Double_V", "-p",
             "data: [0.9, 58.86]" if strong else "data: [0.2, 5.886]"),
            ("topic", "-t", "/perch/hold_cmd", "-m", "gz.msgs.Double", "-p", "data: 1"),
            ("service", "-s", control, "--reqtype", "gz.msgs.WorldControl",
             "--reptype", "gz.msgs.Boolean", "--timeout", "3000",
             "--req", "pause: true, multi_step: 6500"),
        ):
            result = request(*args)
            if result.returncode:
                raise ValueError(result.stderr.strip() or str(args))
        for echo in echoes:
            if echo.wait(timeout=20):
                raise ValueError("gz topic -e thoát với lỗi")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="Thư mục trống để lưu log")
    args = parser.parse_args()
    plugin = Path(__file__).resolve().parents[1]
    env = {**os.environ, "GZ_PARTITION": "perchpilot_codex"}
    paths = [str(plugin / "build"), env.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")]
    env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = os.pathsep.join(filter(None, paths))
    world = Path(__file__).with_name("hold_test.sdf").read_text()
    failed = False
    for name, strong in (("giu", True), ("truot", False)):
        print(f"Ca {name}", flush=True)
        logs = args.log_dir.resolve() / name
        logs.mkdir(parents=True)
        fixture = logs / "hold_test.sdf"
        fixture.write_text(world)
        try:
            run_case(fixture, logs, env, strong)
            check(logs, strong)
        except (ValueError, KeyError, subprocess.TimeoutExpired) as error:
            print(f"Ca {name} lỗi {error}. Xem {logs}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
