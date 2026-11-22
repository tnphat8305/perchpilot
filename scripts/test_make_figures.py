"""Kiểm tra lệnh tổng hợp P6 bằng các JSON giả lập."""
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from make_figures import generate, plt, q2_v4_plot, sensitivity_plot


def rate(p, lo, hi, n=100):
    return {"p": p, "lo": lo, "hi": hi, "n": n}


def touch_run(seed, p):
    nominal = {
        "success": rate(p, p - 0.1, p + 0.1),
        "held_if_motors_cut": rate(0.6, 0.5, 0.7),
        "touch_vn_mean": 0.8,
        "touch_vt_mean": 0.3,
        "touch_align_deg_mean": 4.2,
        "visible_frac_mean": 0.75,
    }
    hard = {**nominal, "success": rate(0.65, 0.55, 0.75)}
    return {
        "seed": seed,
        "config": {"env": {"tw_ratio": [1.9, 2.6], "pad_offset": 0.227, "delay_steps": [4, 6]}},
        "eval": {
            "nominal": nominal,
            "hard": hard,
            "bins": {"speed_0-1": nominal["success"], "tilt_55-62": nominal["success"]},
        },
    }


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def table_rows(markdown, heading):
    section = markdown.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in section.splitlines()
        if line.startswith("|") and not line.startswith("| ---")
    ]


class MakeFiguresTest(unittest.TestCase):
    def test_cli_reads_json_and_skips_unfinished_runs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            results, runs = root / "results", root / "runs"

            write(results / "p1b_s0.json", touch_run(0, 0.8))
            write(results / "p1b_s1.json", touch_run(1, 0.7))
            write(results / "p1b_s0_rescored.json", touch_run(0, 0.9))
            write(runs / "p1b_s0/train_log.json", [
                {"it": 1, "succ_rate": 0.5, "delay_hi": 0},
                {"it": 2, "succ_rate": 0.8, "eval_succ": 0.75, "delay_hi": 1},
            ])
            for seed, (perc, norho) in enumerate(((0.8, 0.7), (0.9, 0.8))):
                write(results / f"p2_perc_s{seed}.json", touch_run(seed, perc))
                write(results / f"p2_norho_s{seed}.json", touch_run(seed, norho))

            for suffix, drop, auroc in (("", "", 0.93), ("_noimu", "imu", 0.56)):
                write(results / f"q2_belief_test{suffix}.json", {
                    "seed": 0, "config": {"data": "/data/test", "label": "held", "drop": drop},
                    "eval": {"test_unseen_materials": {"n": 100, "auroc_hold": auroc,
                                                        "ece15_hold": 0.0234, "brier_hold": 0.0678}},
                })
            dthr, b4b, oracle = "D-thr_v3_early_stop", "B4b_FSM_probe", "B7_oracle_at_touchdown"
            decision = {
                "episodes": 100, "valid_perch": rate(0.6, 0.5, 0.7),
                "false_safe": rate(None, None, None, n=0), "abort": rate(0.3, 0.2, 0.4),
                "self_abort": rate(0.1, 0.05, 0.15),
            }
            write(results / "q2_decide_test.json", {
                "seed": 0, "eval": {"test": {
                    dthr: decision, b4b: decision, oracle: {**decision, "false_safe": rate(0, 0, 0.03)},
                }},
                "h2_same_false_safe_test": {
                    policy: {f"fs<={limit}": {"valid_perch": p} for limit in (0.005, 0.01, 0.02)}
                    for policy, p in ((dthr, 0.6), (b4b, 0.5))
                },
            })
            write(results / "q2_decide_test_curve.json", {"curves": {
                policy: {"test": {"false_safe": [0, 0.02], "valid_perch": [0, p], "n_off": [0, 50]}}
                for policy, p in ((dthr, 0.6), (b4b, 0.5))
            }})
            write(results / "sitl_complete.json", {
                "status": "complete", "seed": 0,
                "eval": {"n": 100, "nominal": {"success": rate(0.8, 0.7, 0.9), "tag_lost_pct": 25.5}},
            })
            write(results / "sitl_incomplete.json", {
                "status": "incomplete", "eval": {"nominal": {"success": rate(0.99, 0.98, 1)}},
            })
            (results / "p1a_s0.json").write_text("{", encoding="utf-8")

            command = [sys.executable, str(Path(__file__).with_name("make_figures.py")),
                       "--results-dir", str(results), "--runs-dir", str(runs)]
            run = subprocess.run(command, check=True, capture_output=True, text=True)
            report = (results / "TABLES.md").read_text(encoding="utf-8")

            headers, p1_s0, rescored, p1_s1, p1_mean, _ = table_rows(report, "Q1 (P1)")
            self.assertEqual(headers, [
                "Nguồn JSON", "Phiên bản", "TW", "pad_offset (m)", "Trễ (bước)", "Seed",
                "Chạm đạt danh nghĩa (% [Wilson])", "Chạm đạt ca khó (% [Wilson])",
                "Giữ được nếu tắt motor (% [Wilson])", "vn (m/s)", "vt (m/s)", "align (°)",
            ])
            self.assertEqual(p1_s0[1:8], [
                "Gốc", "1.9–2.6", "0.227", "4–6", "0", "80.00 [70.00, 90.00]", "65.00 [55.00, 75.00]",
            ])
            self.assertEqual(p1_s0[8:], [
                "60.00 [50.00, 70.00] / 60.00 [50.00, 70.00]", "0.800 / 0.800",
                "0.300 / 0.300", "4.20 / 4.20",
            ])
            self.assertIn("p1b_s1.json", p1_s1[0])
            self.assertEqual(rescored[1], "Chấm lại")
            self.assertEqual(rescored[6], "90.00 [80.00, 100.00]")
            self.assertEqual(p1_mean[6], "75.00 ± 7.07")

            p2 = table_rows(report, "P2 — ablation ρ")
            self.assertEqual(p2[0][-1], "visible_frac_mean (%)")
            self.assertEqual(p2[1][-1], "75.00 / 75.00")
            self.assertEqual([row[2:4] for row in p2 if row[1].startswith("Δ")], [["0", "+10.00"], ["1", "+10.00"]])
            self.assertEqual(p2[-1][3], "+10.00 [+0.00, +20.00]")

            belief = table_rows(report, "Q2 — mạng niềm tin và ablation kênh")
            self.assertEqual(belief[0][-3:], ["AUROC", "ECE", "Brier"])
            self.assertEqual(belief[2][3], "Bỏ IMU")
            self.assertEqual(belief[2][-3:], ["0.5600", "0.0234", "0.0678"])
            policies = table_rows(report, "Q2 — bộ quyết định")
            self.assertEqual(policies[1][-4:], [
                "60.00 [50.00, 70.00]", "—", "30.00 [20.00, 40.00]", "10.00 [5.00, 15.00]",
            ])
            h2 = table_rows(report, "Q2 — H2")
            self.assertEqual([row[2] for row in h2[1:]], ["0.5", "1", "2"])
            self.assertTrue(all(row[-1] == "+10.000" for row in h2[1:]))

            _, sitl = table_rows(report, "SITL")
            self.assertIn("sitl_complete.json", sitl[0])
            self.assertEqual(sitl[4], "80.00 [70.00, 90.00]")
            self.assertEqual(sitl[-1], "25.5")
            for name in ("sitl_incomplete.json", "p1a_s0.json", "p1b_s2.json", "p1b_s1/train_log.json"):
                self.assertIn(name, run.stderr)

            images = {path.name for path in (results / "fig").glob("*.png")}
            self.assertEqual(images, {"p1_learning.png", "p1_success_bins.png", "p2_success_bins.png",
                                      "q2_decide_test_curve.png", "q2_sensor_ablation.png"})
            for name in images:
                with Image.open(results / "fig" / name) as image:
                    self.assertEqual(tuple(round(dpi) for dpi in image.info["dpi"]), (150, 150))

    def test_q1_families_rescoring_failures_and_shared_sensitivity(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures = {
                "p1b_s0": (0, 0.6), "p1b_s0_rescored": (0, 0.8), "p1b_s1": (1, 0.7),
                "p2_perc_s0": (0, 0.8), "p2_perc_s0_rescored": (0, 0.85), "p2_norho_s0": (0, 0.75),
                "p2m_perc_s0": (0, 0.75), "p2m_perc_s0_noyaw": (0, 0.85), "p2y_perc_s0": (0, 0.85),
                "b0_direct": (12345, 0.2), "b6_hover": (12345, 0.1),
            }
            for stem, (seed, p) in fixtures.items():
                run = touch_run(seed, p)
                if stem.endswith("_noyaw"):
                    run["rescored"] = {"overrides": ["yaw_scale=0"]}
                if stem in ("p1b_s0_rescored", "p1b_s1"):
                    run["eval"]["nominal"]["fail_breakdown"] = {"bad_vn": round(100 * (1 - p))}
                write(root / f"{stem}.json", run)
            write(root / "invalid/p2_norho_s1.json", touch_run(1, 0.95))
            for stem in ("p1b_s0", "p2_perc_s0", "p2_norho_s0", "p2y_perc_s0"):
                run = touch_run(0, 0.8)
                fail = {**run["eval"]["nominal"], "fail_breakdown": {"bad_align": 20}}
                run["eval"]["nominal"] = fail
                run["eval"]["sweeps"] = {
                    "delay_steps": [{"delay_ms": [40, 60], "metrics": fail}],
                    "odo_scale": [] if stem == "p1b_s0" else [{"odo_scale": 1, "metrics": fail}],
                }
                run["config"]["v2"] = stem != "p1b_s0"
                write(root / f"sens_{stem}.json", run)
            outputs = generate(root, root / "runs")
            report = outputs[0].read_text(encoding="utf-8")

            header, *q1 = table_rows(report, "Q1 sim — bảng chính")
            nominal = header.index("Chạm đạt danh nghĩa (% [Wilson])")
            self.assertEqual(q1[0][nominal], "75.00 ± 7.07")
            self.assertEqual(q1[1][nominal], "80.00 [70.00, 90.00]")
            self.assertIn("p1b_s0_rescored.json", q1[1][6])
            self.assertNotIn("p1b_s0.json", q1[0][6])
            for family, actuator, yaw in (("p2m_perc", "motor", "1"), ("p2m_perc_noyaw", "motor", "0"),
                                          ("p2y_perc", "motor", "0"), ("b0_direct", "ideal", "1"),
                                          ("b6_hover", "ideal", "1")):
                self.assertTrue(any(row[:3] == [family, actuator, yaw] for row in q1))
            self.assertIn("p2y_perc (chính để triển khai)", report)
            self.assertNotIn("](p2_norho_s1.json)", report)

            _, p1_fail, *failures = table_rows(report, "Q1 — phân loại lỗi")
            self.assertEqual(p1_fail[4:7], ["2", "200", "bad_vn=50"])
            self.assertTrue(any(row[0] == "p2_perc" and row[6] == "bad_align=20" for row in failures))
            p2 = table_rows(report, "P2 — ablation ρ")
            self.assertEqual(p2[-2][2:4], ["0", "+10.00"])
            self.assertEqual(p2[-1][3], "+10.00 [—]")
            sens = table_rows(report, "Q1 — độ nhạy")
            self.assertTrue(any("Bỏ qua (v1)" in row for row in sens))
            self.assertEqual(sum("40–60" in row for row in sens), 4)
            self.assertIn("q1_sensitivity.png", {path.name for path in outputs})
            plots = {
                stem: ({"delay_steps": [(50, "40–60", p, p - 0.1, p + 0.1)]}, {"delay_steps": "—"})
                for stem, p in (("sens_p1b_s0", 0.8), ("sens_p2y_perc_s0", 0.9))
            }
            fig = sensitivity_plot(plots)
            self.addCleanup(plt.close, fig)
            self.assertEqual(fig.axes[0].get_legend_handles_labels()[1], ["p1b_s0", "p2y_perc_s0"])

    def test_sitl_main_speed_table_and_debug_appendix(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            metrics = {
                "touch_success": rate(0.8, 0.49, 0.94, 10), "held_if_motors_cut": rate(0.7, 0.4, 0.9, 10),
                "false_safe": rate(0.125, 0.02, 0.47, 8), "rearm": rate(1, 0.68, 1, 8),
                "recovered": rate(0.9, 0.60, 0.98, 10), "cycle_complete": rate(0.7, 0.4, 0.9, 10),
                "touch_n": 8, "touch_vn_mean": 1.15, "touch_vt_mean": 0.24, "touch_align_deg_mean": 15.3,
                "touch_pos_cm_mean": 10.5, "tag_lost_pct": 50.5, "tag_lost_s": 20.2, "perch_s": 40,
            }
            for suffix, speed in (("v15", 1.5), ("v3", 3), ("v0", 0)):
                write(root / f"sitl_main_{suffix}.json", {
                    "status": "complete", "config": {"speeds": [speed], "mu": [0.9], "f_hold_n": [60.75]},
                    "eval": {"n": 10, "nominal": metrics},
                })
            debug = "sitl_p5m_dbg_fake.json"
            (root / debug).write_text("{", encoding="utf-8")
            outputs = generate(root, root / "runs")
            report = outputs[0].read_text(encoding="utf-8")
            _, *main = table_rows(report, "SITL — theo tốc độ xe")
            self.assertEqual([row[1] for row in main], ["0", "1.5", "3"])
            self.assertEqual(main[0][4:10], [
                "8/10 (80.00 [49.00, 94.00])", "7/10 (70.00 [40.00, 90.00])", "1/8 (12.50 [2.00, 47.00])",
                "8/8 (100.00 [68.00, 100.00])", "9/10 (90.00 [60.00, 98.00])", "7/10 (70.00 [40.00, 90.00])",
            ])
            self.assertEqual(main[0][10:], ["8", "1.15", "0.24", "15.3", "10.5", "50.5", "20.2/40"])
            for note in ("C2 v1.1", "--yaw-scale 0", "--freeze-plate-v", "--depart-pushoff", "μ = 0,9"):
                self.assertIn(note, report)
            self.assertNotIn(debug, report.split("## Phụ lục")[0])
            self.assertIn(f"[{debug}]({debug})", report.split("## Phụ lục")[1])
            self.assertIn("sitl_speed.png", {path.name for path in outputs})

    def test_sitl_q2_v12_and_pooled_tables(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            qos = {"perch_node_hold_sensors_qos": "reliable/keep_last/50"}
            baseline = {"false_safe": rate(1, 0.5, 1, 2)}
            trials = [
                {"q2": {"decision": "OFF"}, "outcome": "crash"},
                {"q2": {"decision": "SELF_ABORT", "why": "contact lost after MOTORS_OFF was commanded"},
                 "outcome": "self_abort"},
                {"q2": {"decision": "OFF"}, "outcome": "rest", "valid_perch": False},
                {"q2": {"decision": "SELF_ABORT", "why": "contact lost before MOTORS_OFF"}, "outcome": "self_abort"},
                {"q2": None, "outcome": "crash"},
            ]
            stored = {
                "false_safe_v12": rate(0.25, 0.05, 0.7, 4),
                "off_commanded": rate(0.5, 0.2, 0.8, 8),
            }
            write(root / "sitl_q2_b4e_qos.json", {
                "provenance": {**qos, "env": {"PERCH_NODE_ARGS": "--verify-rule=fsm_b4e"}},
                "config": {"verify_rule": "none"}, "eval": {"nominal": {**baseline, **stored}}, "trials": trials,
            })
            bins = {f"speed_{speed}.0_mu_{mu}_f_hold_n_10.0": baseline
                    for speed, mu in ((0, 0.9), (0, 0.5), (3, 0.9))}
            write(root / "sitl_q2_dthr_lc_v4.json", {
                "config": {"speeds": [0, 3], "verify_rule": "D-thr", "q2_belief": "/runs/q2_belief_v4_small_s0"},
                "provenance": qos,
                "eval": {"bins": bins, "trials": [
                    *({**trial, "speed_mps": 0, "mu": 0.9, "f_hold_n": 10} for trial in trials),
                    {**trials[4], "speed_mps": 0, "mu": 0.5, "f_hold_n": 10},
                    {**trials[2], "speed_mps": 3, "mu": 0.9, "f_hold_n": 10},
                ]},
            })
            expected = {}
            for suffix, args, provenance, law, note in (
                ("r6", "--verify-rule dthr --q2-belief /runs/q2_belief_v6_full_s0", qos,
                 "dthr (q2_belief_v6_full_s0)", "—"),
                ("r7", "--verify-rule dthr --q2-belief=/runs/q2_belief_v7_full_s0", qos,
                 "dthr (q2_belief_v7_full_s0)", "—"),
                ("min", "--verify-rule dthr --q2-dthr-key D-thr_v3_early_stop_minPULL0.3", {},
                 "dthr_min", "trước sửa QoS"),
                ("none_qos", "--verify-rule none", qos, "none", "—"),
                ("fsm_b4b_prereview", "", qos, "fsm_b4b", "trước sửa QoS"),
                ("probe_m4b", "", qos, "probe", "lô đo tín hiệu (ngoài đánh giá luật quyết định)"),
            ):
                write(root / f"sitl_q2_{suffix}.json", {
                    "provenance": {**provenance, "env": {"PERCH_NODE_ARGS": args}},
                    "eval": {"nominal": baseline},
                })
                expected[suffix] = (law, note)
            write(root / "sitl_q2_partial.json", {
                "config": {"rule": "none", "speeds": [1.5]}, "off_commanded": rate(0.375, 0.1, 0.7, 8),
                "trials": trials,
            })
            write(root / "sitl_q2_pooled.json", {
                "groups": {"b4e": {
                    "batches": ["q2_b4e_qos"], "eval_seeds": [2027], "status": ["complete"], "n": 8,
                    "config": {"speeds": [1.5]},
                    "off_rate": {**stored["off_commanded"], "k": 4},
                    "false_safe_v12": {**stored["false_safe_v12"], "k": 1},
                }},
                "vs_baseline": {"b4e": {
                    "baseline": "none", "false_safe_v12": {"diff": -0.1217, "lo": -0.4321, "hi": 0.0987,
                                                           "fisher_p": 0.0421},
                    "false_safe_held": None,
                }},
            })

            report = generate(root, root / "runs")[0].read_text(encoding="utf-8")
            header, *rows = table_rows(report, "SITL Q2 — theo luật")
            q2 = [dict(zip(header, row)) for row in rows]

            def batch(suffix):
                name = f"sitl_q2_{suffix}.json"
                return [row for row in q2 if row["Nguồn JSON"] == f"[{name}]({name})"]

            self.assertNotIn("False-safe", header)
            for suffix, (law, note) in expected.items():
                row, = batch(suffix)
                self.assertEqual((row["Luật"], row["Ghi chú lô"]), (law, note))
                self.assertEqual((row["Lệnh OFF"], row["False-safe v1.2"]), ("—", "—"))
            row, = batch("b4e_qos")
            self.assertEqual(row["Luật"], "fsm_b4e")
            self.assertEqual(row["Lệnh OFF"], "4/8 (50.00 [20.00, 80.00])")
            self.assertEqual(row["False-safe v1.2"], "1/4 (25.00 [5.00, 70.00])")
            row, = batch("partial")
            self.assertEqual(row["Tốc độ xe (m/s)"], "1.5")
            self.assertEqual(row["Lệnh OFF"], "3/8 (37.50 [10.00, 70.00])")
            self.assertEqual(row["False-safe v1.2"], "2/3 (66.67 [20.77, 93.85])")
            materials = {row["μ"]: row for row in batch("dthr_lc_v4") if row["Tốc độ xe (m/s)"] == "0.0"}
            self.assertEqual(materials["0.9"]["Luật"], "dthr (q2_belief_v4_small_s0)")
            self.assertEqual(materials["0.9"]["Ghi chú lô"], "trước sửa QoS")
            self.assertEqual(materials["0.9"]["Lệnh OFF"], "3/5 (60.00 [23.07, 88.24])")
            self.assertEqual(materials["0.9"]["False-safe v1.2"], "2/3 (66.67 [20.77, 93.85])")
            self.assertEqual(materials["0.5"]["False-safe v1.2"], "0/0")
            row, = (row for row in batch("dthr_lc_v4") if row["Tốc độ xe (m/s)"] == "3.0")
            self.assertEqual(row["False-safe v1.2"], "0/1 (0.00 [0.00, 79.35])")

            header, group = table_rows(report, "SITL Q2 — kết quả gộp")
            pooled = dict(zip(header, group))
            self.assertEqual([pooled[key] for key in ("Nhóm", "Tốc độ xe (m/s)", "Seed eval", "Trạng thái các lô")],
                             ["b4e", "1.5", "2027", "complete"])
            self.assertEqual(pooled["Lệnh OFF"], "4/8 (50.00 [20.00, 80.00])")
            self.assertEqual(pooled["False-safe v1.2"], "1/4 (25.00 [5.00, 70.00])")
            _, *comparisons = table_rows(report, "SITL Q2 — so sánh với đối chứng")
            self.assertEqual(comparisons, [
                ["b4e", "none", "False-safe v1.2", "-12.17 [-43.21, +9.87]", "0.0421"],
                ["b4e", "none", "False-safe held", "—", "—"],
            ])
            self.assertNotIn("`sitl_q2_pooled.json` — thiếu kết quả SITL", report)

    def test_q2_v4_locked_caps_bootstrap_seeds_sensor_and_curves(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            score = {
                "episodes": 100, "n_valid": 60, "n_off": 66, "n_fell": 1, "exceeds_cap": True,
                "valid_perch": rate(0.6, 0.5, 0.7), "motors_off": rate(0.66, 0.56, 0.75),
                "false_safe": rate(1 / 66, 0.002, 0.08, 66), "false_safe_held": rate(6 / 66, 0.04, 0.18, 66),
                "false_safe_strict": rate(7 / 66, 0.05, 0.19, 66),
            }
            off_zero = {**score, "n_valid": 0, "n_off": 0, "n_fell": 0, "exceeds_cap": None,
                        "valid_perch": rate(0, 0, 0.04), "motors_off": rate(0, 0, 0.04),
                        **dict.fromkeys(("false_safe", "false_safe_held", "false_safe_strict"),
                                        rate(None, None, None, 0))}
            block = {
                "test": {**dict.fromkeys(("D-thr", "D-thr-late", "B4b", "B4c", "B7_oracle_at_touchdown"), score),
                         "B4a": None, "B4d": off_zero, "always_off_at_touchdown_no_check": score},
                "comparisons": {"D-thr - B4b": {
                    "status": "cap_exceeded", "delta_valid_perch": -0.1,
                    "bootstrap_95": {"lo": -0.13, "hi": -0.07, "n_defined": 10000},
                }},
                "across_seeds": {"D-thr": {
                    "scored_methods": ["D-thr", *(f"D-thr/seed={i}" for i in range(1, 5))],
                    "valid_perch": {"n": 5, "mean": 0.61, "std": 0.03},
                }},
            }
            lock = {"cfg_kw": {"vib_scale": 0.5, "slip_acc_in_imu": False}, "selection": {"D-thr": {
                "0.02": {"dev_ood": {"n": 100, "n_valid": 70, "n_off": 80, "n_fell": 1}},
            }}}
            curve = {"curves": {"D-thr": {
                "dev_ood": {"false_safe": [0.01, 0.02], "valid_perch": [0.65, 0.72], "n_off": [80, 81]},
                "test": {"false_safe": [0.01, 0.03], "valid_perch": [0.9, 0.6], "n_off": [70, 72]},
            }}}
            for stem in ("q2_v4_small_main", "q2_v4_sens_small_v0.5_a0"):
                write(root / f"{stem}.json", {
                    "lock": f"{stem}_lock.json", "lock_sha256": f"sha-{stem}",
                    "by_cap": dict.fromkeys(("0.005", "0.01", "0.02"), block),
                })
                write(root / f"{stem}_lock.json", {"sha256": f"sha-{stem}", "config": lock})
            outputs = generate(root, root / "runs")
            report = outputs[0].read_text(encoding="utf-8")
            _, *table = table_rows(report, "Q2 v4 — cấu hình khóa trên dev_ood")
            self.assertEqual({row[2] for row in table}, {"0.5", "1", "2"})
            self.assertEqual(len(table), 24)
            self.assertEqual(table[0][1], "quy mô nhỏ")
            self.assertIn("sha-q2_v4_small_main", table[0][3])
            self.assertEqual(table[0][5:11], [
                "60/100 (60.00 [50.00, 70.00])", "66/100 (66.00 [56.00, 75.00])", "1/66 (1.52 [0.20, 8.00])",
                "6/66 (9.09 [4.00, 18.00])", "7/66 (10.61 [5.00, 19.00])", "True",
            ])
            self.assertEqual(table[2][-1], "Không có cấu hình trên dev_ood")
            self.assertEqual(table[5][7:11], ["0/0", "0/0", "0/0", "—"])
            self.assertEqual([row[4] for row in table[6:8]], ["B7", "Tắt ngay"])
            self.assertTrue(all(row[-1] == "Đã chấm" for row in table[6:8]))
            bootstrap = table_rows(report, "Q2 v4 — Δ bootstrap ghép cặp")
            self.assertEqual(bootstrap[1][-3:], ["cap_exceeded", "-10.00 [-13.00, -7.00]", "10000"])
            seeds = table_rows(report, "Q2 v4 — qua các seed mạng niềm tin")
            self.assertIn("D-thr/seed=4", seeds[1][4])
            self.assertEqual(seeds[1][5], "61.00 ± 3.00 (n=5)")
            _, sensor, *_ = table_rows(report, "Q2 v4 — độ nhạy cảm biến")
            self.assertEqual(sensor[1:5], ["quy mô nhỏ", "0.5", "0.5", "False"])
            self.assertIn("q2_v4_small_main_locked.png", {path.name for path in outputs})

            write(root / "q2_v4_full_main.json", {"by_cap": {"0.02": block}})
            write(root / "q2_v4_full_main_curve.json", curve)
            outputs = generate(root, root / "runs")
            report = outputs[0].read_text(encoding="utf-8")
            self.assertNotIn("q2_v4_small_main.json", report)
            self.assertIn("q2_v4_full_main_locked.png", {path.name for path in outputs})
            self.assertIn("thiếu `q2_v4_full_main_lock.json`", report)
            fig = q2_v4_plot({"by_cap": {"0.02": block}}, lock, curve)
            self.addCleanup(plt.close, fig)
            self.assertEqual([ax.get_title() for ax in fig.axes], ["dev_ood", "test"])
            self.assertEqual(list(fig.axes[0].lines[1].get_ydata()), [70])
            self.assertEqual(list(fig.axes[1].lines[0].get_ydata()), [90, 60])


if __name__ == "__main__":
    unittest.main()
