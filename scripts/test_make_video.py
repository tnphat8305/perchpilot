"""Exercise the SITL replayer on synthetic logs."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from make_video import Clip, SCHEDULE_V3, View, WIN, b4e_thresholds, render, select_montage, summary_canvas


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class MakeVideoTest(unittest.TestCase):
    def test_montage_pairs_the_abort_with_a_fall_in_the_same_cell(self):
        thresholds = dict(zip(SCHEDULE_V3, (1., 2., 3.)))
        points = [{"macro": macro, "slip_mm": threshold + 1} for macro, threshold in thresholds.items()]
        abort = {"trial": 5, "speed_mps": 0., "mu": .2, "f_hold_n": 24.3, "outcome": "q2_abort",
                 "q2": {"decision": "ABORT", "why": "fsm_b4e at PULL0.6", "points": points}}
        off = {**abort, "trial": 1, "mu": .9, "held": True, "valid_perch": True, "cycle_complete": True,
               "q2": {"decision": "OFF", "why": "fsm_b4e at RELAX"}}
        fall = {**abort, "motors_off": True, "fell_after_off": True,
                "q2": {"decision": "OFF", "why": "rule none: motors off at touchdown"}}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in (
                ("main_p2y_v15", [off]), ("main_p2y_v3", [off]),
                ("q2_b4e_qos", [off, {**abort, "q2": {**abort["q2"], "decision": "SELF_ABORT"}}]),
                ("q2_b4e_qos_s2", [abort]),
                ("q2_none_qos", [{**fall, "trial": 2, "f_hold_n": 10.13}, fall]),
            ):
                batch = root / name
                batch.mkdir()
                write_jsonl(batch / "trials.jsonl", rows)
                for row in rows:
                    (batch / f"truth_{row['trial']}.jsonl").touch()
            selected = [(batch.name, row["trial"]) for batch, row, _ in select_montage(root, thresholds)]
            self.assertEqual(selected, [
                ("main_p2y_v15", 1), ("main_p2y_v3", 1), ("q2_b4e_qos_s2", 5),
                ("q2_b4e_qos", 1), ("q2_none_qos", 5),
            ])

    def test_b4e_window_and_summary_card(self):
        group = {
            "config": {"node_args": "--verify-rule fsm_b4e --q2-slip-off 3 2 1", "speeds": [0]},
            "eval_seeds": [11, 12], "valid_perch": {"k": 7, "n": 11},
            "off_rate": {"k": 8, "n": 11}, "false_safe_v12": {"k": 1, "n": 8},
        }
        thresholds = b4e_thresholds(group)
        self.assertEqual(thresholds, {"RELAX": 3, "PULL0.3": 2, "PULL0.6": 1})
        q2 = {"decision": "OFF", "macro": "RELAX", "points": [
            {"d": 0, "macro": "RELAX", "t": .4, "slip_mm": 2.5},
        ]}
        row = {"trial": 1, "mu": .9, "f_hold_n": 60, "speed_mps": 0, "q2": q2}
        models = {name: {"xyz": [x, 0, 1], "wxyz": [1, 0, 0, 0]}
                  for name, x in (("x500_perch_7", 1), ("perch_cart", 3))}
        samples = [{"wall": 500 + i / 200, "phase": "VERIFY",
                    "x": [0, 0, 9.81, 0, 0, 0, 12, 1, .3, 1, .1, (-1)**i, 2 / WIN]}
                   for i in range(WIN)]
        samples.append({**samples[-1], "wall": 500.4, "phase": "REST"})
        with TemporaryDirectory() as directory:
            batch = Path(directory)
            write_jsonl(batch / "truth_1.jsonl", [
                {"kind": "pose", "sim_t": 10, "wall": 500, "models": models},
                {"kind": "diag", "sim_t": 10.4, "diag": {"phase": "REST", "q2": q2}},
                {"kind": "pose", "sim_t": 11, "wall": 501, "models": models},
            ])
            clip = Clip(batch, row, "Test B4e", [], samples, thresholds)
            clip.finish_setup()
            self.assertEqual(len(clip.slip_window), 1)
            self.assertAlmostEqual(clip.slip_window[0][1], 2., places=5)
            view = View(clip, 25, "fixed", 2)
            view.draw(.5)
            self.assertIn("2.500 < 3.000 mm  →  OFF", view.info.get_text())
            self.assertEqual(list(view.fig.axes[-1].lines[1].get_ydata()), [3, 3])

            (batch / "sitl_main_p2y_v15.json").write_text(json.dumps({
                "config": {"speeds": [1.5]}, "eval": {"nominal": {
                    key: {"p": .625, "n": 8} for key in ("touch_success", "valid_perch", "cycle_complete")
                }},
            }), encoding="utf-8")
            summary = summary_canvas(batch, dict.fromkeys(("b4e", "none", "dthr_v7"), group), 25)
            q2_table, main_table = (ax.tables[0] for ax in summary.figure.axes)
            self.assertEqual(q2_table[1, 3].get_text().get_text(), "1/8")
            self.assertEqual(main_table[1, 3].get_text().get_text(), "5/8")
            output = batch / "montage.mp4"
            render([clip], SimpleNamespace(out=output, dpi=25, camera="fixed", speed=2, fps=4,
                                           decimate=1, montage=True), summary)
            frames = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(output),
            ], text=True)
            self.assertEqual(int(frames), 28)  # Two replay frames, six tail frames, and twenty summary frames.

    def test_two_second_clip(self):
        script = Path(__file__).with_name("make_video.py")
        with TemporaryDirectory() as directory:
            batch = Path(directory)
            output = batch / "replay.mp4"
            write_jsonl(batch / "trials.jsonl", [{"trial": 7, "speed_mps": 0.5, "outcome": "rest"}])
            truth = [{"kind": "start", "sim_t": 20}]
            ticks, samples = [], []
            for i in range(41):
                t = i / 20
                clock = {"sim_t": 20 + t, "wall": 500 + 1.03 * t}
                phase = "APPROACH" if t < 0.5 else "PERCH" if t < 1 else "VERIFY" if t < 1.4 else "REST"
                q = [0.92388, 0, -0.38268, 0] if t >= 1 else [1, 0, 0, 0]
                truth.extend([
                    {"kind": "pose", **clock, "models": {
                        "x500_perch_7": {"xyz": [1.7 * min(t, 1.5), 0.1, 2 - 0.4 * t], "wxyz": q},
                        "perch_cart": {"xyz": [2.2 + 0.5 * t, 0, 0.45], "wxyz": [1, 0, 0, 0]},
                    }},
                    {"kind": "hold", **clock, "engaged": t >= 1, "margin_n": 10, "slip_mps": 0},
                    {"kind": "diag", **clock, "diag": {
                        "phase": phase, "armed": phase != "REST", "tag_visible": t < 1, "plate_age": t,
                        "rho": 0.7, "thrust_norm": 0.5 if t < 1 else 0, "rates_frd": [0.4, -0.2, 0.05],
                    }},
                ])
                if t == 1:
                    truth.append({"kind": "touch", **clock, "vn_mps": 1, "vt_mps": .3, "align_deg": 10,
                                  "omega_radps": 1, "pad_u_m": .03, "pad_w_m": .1})
                if phase in ("PERCH", "VERIFY"):
                    ticks.append({"trial": 7, "wall": clock["wall"], "phase": phase, "thrust_norm": .35,
                                  "rates_frd": [0.1, -0.5, 0], "plate": {"age": t, "visible": t < 1}})
                    samples.append({"trial": 7, "wall": clock["wall"], "phase": phase,
                                    "x": [0, 0, 9.81, 0, 0, 0, 12, 1, .3, 1, .1, .03, -.05]})
            truth.append({"kind": "end", "sim_t": 22, "outcome": "rest"})
            write_jsonl(batch / "truth_7.jsonl", truth)
            write_jsonl(batch / "node_ticks.jsonl", ticks)
            write_jsonl(batch / "q2_samples.jsonl", samples)

            subprocess.run([sys.executable, str(script), "--batch", str(batch), "--trial", "7",
                            "--out", str(output), "--fps", "30", "--dpi", "50",
                            "--camera", "fixed", "--decimate", "3"], check=True, capture_output=True, text=True)
            probe = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                "stream=width,height,nb_read_frames,r_frame_rate,duration", "-of", "json", str(output),
            ], text=True)
            stream, = json.loads(probe)["streams"]
            self.assertEqual((stream["width"], stream["height"]), (640, 360))
            self.assertEqual(int(stream["nb_read_frames"]), 60)
            self.assertEqual(stream["r_frame_rate"], "30/1")
            self.assertAlmostEqual(float(stream["duration"]), 2.0)


if __name__ == "__main__":
    unittest.main()
