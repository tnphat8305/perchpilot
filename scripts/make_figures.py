"""Tổng hợp kết quả P6 từ JSON đã lưu, dùng Matplotlib Agg trên CPU."""
import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, quantiles, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
RUNS = Path("/home/sup/data/perchpilot/runs")
P1 = re.compile(r"(p1[ab])_s\d+(_rescored)?")
P2 = re.compile(r"(p2_perc|p2_norho)_s(\d+)")
CASES = {"nominal": "Danh nghĩa", "hard": "Ca khó"}
TOUCH_METRICS = (
    ("success", ("nominal",), "Chạm đạt danh nghĩa (% [Wilson])", 100, 2),
    ("success", ("hard",), "Chạm đạt ca khó (% [Wilson])", 100, 2),
    ("held_if_motors_cut", CASES, "Giữ được nếu tắt motor (% [Wilson])", 100, 2),
    ("touch_vn_mean", CASES, "vn (m/s)", 1, 3),
    ("touch_vt_mean", CASES, "vt (m/s)", 1, 3),
    ("touch_align_deg_mean", CASES, "align (°)", 1, 2),
)
VISIBLE = ("visible_frac_mean", CASES, "visible_frac_mean (%)", 100, 2)
BELIEF_METRICS = ("auroc_hold", "ece15_hold", "brier_hold")
DTHR = {
    "D-thr_v3_early_stop": "D-thr v3 dừng sớm",
    "D-thr_v3_early_stop_ensemble": "D-thr v3 ensemble",
    "D-thr": "D-thr",
}
FSM = {"B4a_FSM_no_probe": "B4a", "B4b_FSM_probe": "B4b"}
ORACLE = "B7_oracle_at_touchdown"
POLICIES = {**DTHR, **FSM, ORACLE: "B7 oracle", "always_off_at_touchdown_no_check": "Tắt ngay"}
DECISION_METRICS = ("valid_perch", "false_safe", "abort", "self_abort")


def get(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def estimate(value):
    return get(value, "p") if isinstance(value, dict) else value


def fmt(value, digits=None):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}" if digits is not None else str(value)


def metric(value, scale=100, digits=2):
    p = estimate(value)
    if not number(p):
        return "—"
    lo, hi = get(value, "lo"), get(value, "hi")
    point = fmt(scale * p, digits)
    if number(lo) and number(hi):
        point += f" [{scale * lo:.{digits}f}, {scale * hi:.{digits}f}]"
    return point


def wilson_point(value):
    p, lo, hi = (get(value, key) for key in ("p", "lo", "hi"))
    if all(map(number, (p, lo, hi))) and 0 <= lo <= p <= hi <= 1:
        return p, lo, hi
    return None


def count_rate(value, *, k=None, n=None):
    n = get(value, "n") if n is None else n
    if k is None:
        k = get(value, "k")
    p = estimate(value)
    if k is None and number(p) and number(n) and n > 0:
        k = round(p * n)
    rate = metric(value)
    if n == 0:
        return "0/0" if k in (0, None) else "—"
    if number(n):
        return f"{fmt(k)}/{n} ({rate})" if rate != "—" else f"{fmt(k)}/{n}"
    return rate


def read_json(path, warn, expected=dict):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        warn(f"`{path}` — bỏ qua, không đọc được JSON ({error}).")
        return None
    if not isinstance(data, expected):
        shape = "danh sách" if expected is list else "đối tượng"
        warn(f"`{path}` — bỏ qua, JSON phải là một {shape}.")
        return None
    return data


def source(stem):
    return f"[{stem}.json]({stem}.json)"


def table(lines, title, note, headers, rows):
    lines.extend([f"## {title}", "", note, ""])
    if rows:
        lines.extend(
            "| " + " | ".join(str(cell).replace("|", r"\|") for cell in row) + " |"
            for row in [headers, ["---"] * len(headers), *rows]
        )
    else:
        lines.append("Chưa có kết quả đủ dữ liệu.")
    lines.append("")


def touch_values(result, metrics):
    for key, cases, _, _, _ in metrics:
        yield [get(result, "eval", case, key) for case in cases]


def touch_row(result, metrics):
    return [
        " / ".join(metric(value, scale, digits) for value in values)
        for values, (_, _, _, scale, digits) in zip(touch_values(result, metrics), metrics)
    ]


def seed_values(results, metrics):
    for column in zip(*(touch_values(result, metrics) for result in results)):
        yield [[p for value in case if number(p := estimate(value))] for case in zip(*column)]


def seed_summary(values, scale, digits):
    if not values:
        return "—"
    spread = fmt(scale * stdev(values), digits) if len(values) > 1 else "—"
    return f"{scale * mean(values):.{digits}f} ± {spread}"


def range_text(value):
    if isinstance(value, str) and "," in value:
        value = [part.strip() for part in value.split(",")]
    if isinstance(value, list):
        return "–".join(f"{part:g}" if number(part) else str(part) for part in value)
    return fmt(value)


def p1_config(result, run, runs_dir, warn):
    config = result.get("config")
    tw = get(config, "env", "tw_ratio")
    if tw is None:
        tw = get(config, "tw_ratio") or get(config, "tw")
    pad = get(config, "env", "pad_offset")
    if pad is None:
        pad = get(config, "pad_offset")
    delay = get(config, "env", "delay_steps")
    if delay is None:
        delay = get(config, "delay_steps")

    fields = [tw, pad, delay]
    path = runs_dir / run / "policy.json"
    if None in fields and path.is_file():
        policy = read_json(path, warn)
        if (result.get("git") is not None and result.get("seed") is not None
                and get(policy, "git") == result.get("git") and get(policy, "seed") == result.get("seed")):
            if tw is None and number(get(policy, "tw_ratio_nominal")):
                tw = f"{policy['tw_ratio_nominal']:g} (danh nghĩa)"
            if pad is None and number(get(policy, "pad_offset")):
                pad = policy["pad_offset"]
            if delay is None:
                delay = get(policy, "delay_steps_trained")
    extra_source = f"`runs/{run}/policy.json`" if fields != [tw, pad, delay] else ""
    return (range_text(tw), fmt(pad), range_text(delay)), extra_source


def p1_table(lines, results, runs_dir, warn):
    rows = []
    groups = defaultdict(list)
    for stem, result in results.items():
        family, rescored = P1.fullmatch(stem).groups()
        run = stem.removesuffix("_rescored")
        config, policy_source = p1_config(result, run, runs_dir, warn)
        file_source = ", ".join(filter(None, (source(stem), policy_source)))
        rows.append([file_source, "Chấm lại" if rescored else "Gốc", *config,
                     fmt(result.get("seed")), *touch_row(result, TOUCH_METRICS)])
        groups[(family, rescored, config)].append((file_source, result))
    for (family, rescored, config), entries in groups.items():
        files, runs = zip(*entries)
        summary = [
            " / ".join(seed_summary(values, scale, digits) for values in column)
            for column, (_, _, _, scale, digits) in zip(seed_values(runs, TOUCH_METRICS), TOUCH_METRICS)
        ]
        rows.append([", ".join(dict.fromkeys(files)), f"TB ± ĐLC {family} ({'chấm lại' if rescored else 'gốc'})",
                     *config, f"{len(runs)} seed", *summary])
    table(lines, "Q1 (P1)",
          "Nguồn theo từng dòng. Cặp giá trị theo thứ tự danh nghĩa / ca khó. ĐLC mẫu cần ít nhất hai seed. "
          "Trễ là dải cấu hình tính theo bước điều khiển. Bản gốc và bản chấm lại có các dòng tổng hợp riêng. "
          "TW danh nghĩa và cấu hình bổ sung, nếu có, lấy từ `policy.json` cùng git và seed trong thư mục log.",
          ["Nguồn JSON", "Phiên bản", "TW", "pad_offset (m)", "Trễ (bước)", "Seed",
           *(label for _, _, label, _, _ in TOUCH_METRICS)], rows)


def p2_effect(perc, norho, scale, digits, rng):
    if not perc or not norho:
        return "—"
    delta = scale * (mean(perc) - mean(norho))
    point = f"{delta:+.{digits}f}"
    if min(len(perc), len(norho)) < 2:
        return point + " [—]"
    samples = [
        scale * (mean(rng.choices(perc, k=len(perc))) - mean(rng.choices(norho, k=len(norho))))
        for _ in range(5000)
    ]
    bounds = quantiles(samples, n=40)
    return f"{point} [{bounds[0]:+.{digits}f}, {bounds[-1]:+.{digits}f}]"


def p2_table(lines, results, warn):
    metrics = TOUCH_METRICS
    if any(number(get(run, "eval", case, VISIBLE[0])) for run in results.values() for case in CASES):
        metrics += (VISIBLE,)
    arms = defaultdict(dict)
    rows = []
    for stem, run in results.items():
        arm, suffix_seed = P2.fullmatch(stem.removesuffix("_rescored")).groups()
        seed = run.get("seed")
        if type(seed) is not int:
            seed = int(suffix_seed)
        arms[arm][seed] = (stem, run)
        rows.append([source(stem), arm, fmt(seed), *touch_row(run, metrics)])
    perc, norho = arms["p2_perc"], arms["p2_norho"]
    paired = perc.keys() & norho.keys()
    for seed in sorted(paired):
        sources, runs = zip(perc[seed], norho[seed])
        delta = []
        columns = zip(*(touch_values(run, metrics) for run in runs))
        for (left, right), (_, _, _, scale, digits) in zip(columns, metrics):
            differences = []
            for a, b in zip(map(estimate, left), map(estimate, right)):
                differences.append(f"{scale * (a - b):+.{digits}f}" if number(a) and number(b) else "—")
            delta.append(" / ".join(differences))
        rows.append([", ".join(map(source, sources)), "Δ p2_perc − p2_norho", seed, *delta])
    if perc and norho:
        perc, norho = (seed_values([run for _, run in arm.values()], metrics) for arm in (perc, norho))
        rng = random.Random(0)
        delta = [
            " / ".join(p2_effect(a, b, scale, digits, rng) for a, b in zip(perc_cases, norho_cases))
            for perc_cases, norho_cases, (_, _, _, scale, digits) in zip(perc, norho, metrics)
        ]
        rows.append([", ".join(source(stem) for stem in results), "Hiệu TB p2_perc − p2_norho", "—", *delta])
    else:
        warn("P2 — cần kết quả ở cả hai nhóm để tính hiệu số.")
    table(lines, "P2 — ablation ρ",
          "Nguồn theo từng dòng. Cặp giá trị theo thứ tự danh nghĩa / ca khó. Hiệu số tỉ lệ tính theo điểm phần trăm. "
          "Δ từng seed ghép theo seed huấn luyện. KTC 95% của hiệu hai trung bình dùng phân vị bootstrap, "
          "lấy lại có hoàn lại 5.000 lần theo seed trong từng nhóm "
          "một cách độc lập. KTC cần ít nhất hai seed có số liệu ở mỗi nhóm.",
          ["Nguồn JSON", "Nhóm", "Seed", *(label for _, _, label, _, _ in metrics)], rows)


def failure_text(metrics):
    counts = get(metrics, "fail_breakdown")
    if not isinstance(counts, dict):
        return "—"
    return ", ".join(f"{key}={count}" for key, count in counts.items() if count) or "0"


def q1_case_columns(metrics):
    return [*(metric(get(metrics, key)) for key in ("success", "crash", "timeout")), failure_text(metrics)]


Q1_CASE_COLUMNS = ["Chạm đạt (% [Wilson])", "Va chạm (% [Wilson])", "Hết giờ (% [Wilson])", "fail_breakdown"]
Q1_FAMILIES = ("p1b", "p2_perc", "p2_norho", "p2m_perc", "p2y_perc", "b0_direct", "b6_hover")
Q1_BASELINES = {"b0_direct": "B0 Bay thẳng", "b6_hover": "B6 Bay treo"}
Q1_METRICS = TOUCH_METRICS + (
    ("crash", CASES, "Va chạm (% [Wilson])", 100, 2),
    ("timeout", CASES, "Hết giờ (% [Wilson])", 100, 2),
)


def preferred_runs(results):
    selected = {}
    for stem, run in results.items():
        if not number(estimate(get(run, "eval", "nominal", "success"))):
            continue
        name = stem.removesuffix("_rescored")
        if name not in selected or stem.endswith("_rescored"):
            selected[name] = stem, run
    return selected


def q1_group(name, run):
    original = name.removesuffix("_noyaw")
    family = next((key for key in Q1_FAMILIES if original == key or original.startswith(key + "_")), None)
    if family is None:
        if original == name:
            return None
        family = re.sub(r"_s\d+$", "", original)
    env = get(run, "config", "env")
    motor = get(env, "motor_model")
    yaw = get(env, "yaw_scale")
    overrides = get(run, "rescored", "overrides")
    if isinstance(overrides, list):
        for entry in overrides:
            if not isinstance(entry, str):
                continue
            key, _, value = entry.partition("=")
            if key == "motor_model" and value.lower() in ("true", "false", "0", "1"):
                motor = value.lower() in ("true", "1")
            if key == "yaw_scale":
                try:
                    yaw = float(value)
                except ValueError:
                    pass
    if motor is None:
        motor = family in ("p2m_perc", "p2y_perc")
    if original != name:
        family += "_noyaw"
        yaw = 0
    elif not number(yaw):
        yaw = 0 if family == "p2y_perc" else 1
    return family, "motor" if motor else "ideal", yaw


def q1_tables(lines, selected, sensitivity, warn):
    groups, diagnostics = defaultdict(list), defaultdict(list)
    for name, (stem, run) in selected.items():
        if group := q1_group(name, run):
            groups[group].append((stem, run))
    for family in Q1_FAMILIES:
        if not any(group[0] == family for group in groups):
            warn(f"Q1 sim — chưa có kết quả họ `{family}`.")
    if not any(group[0].endswith("_noyaw") for group in groups):
        warn("Q1 sim — chưa có bản chấm lại `*_noyaw.json`.")
    ordered = sorted(groups, key=lambda group: (
        Q1_FAMILIES.index(group[0]) if group[0] in Q1_FAMILIES else len(Q1_FAMILIES), group))
    rows = []
    for group in ordered:
        family, actuator, yaw = group
        entries = groups[group]
        files, runs = zip(*entries)
        summary = [
            " / ".join(seed_summary(values, scale, digits) for values in column)
            for column, (_, _, _, scale, digits) in zip(seed_values(runs, Q1_METRICS), Q1_METRICS)
        ]
        label = "**p2y_perc (chính để triển khai)**" if family == "p2y_perc" else Q1_BASELINES.get(family, family)
        rows.append([label, actuator, f"{yaw:g}", "—", "—",
                     f"TB ± ĐLC ({len(entries)} run)", ", ".join(map(source, files)), *summary])
        for stem, run in entries:
            rows.append([family, actuator, f"{yaw:g}", fmt(run.get("seed")) if family not in Q1_BASELINES else "—",
                         fmt(get(run, "eval", "eval_seed")), fmt(get(run, "eval", "n")), source(stem),
                         *touch_row(run, Q1_METRICS)])
            for case in CASES:
                if isinstance(get(run, "eval", case, "fail_breakdown"), dict):
                    diagnostics[group, case].append((stem, run))

    for stem, run in sensitivity.items():
        name = stem.removeprefix("sens_")
        group = q1_group(name, run)
        if group is None:
            continue
        for case in CASES:
            primary = get(selected.get(name, (None, None))[1], "eval", case, "fail_breakdown")
            if not isinstance(primary, dict) and isinstance(get(run, "eval", case, "fail_breakdown"), dict):
                diagnostics[group, case].append((stem, run))

    table(lines, "Q1 sim — bảng chính",
          "Mỗi cặp theo thứ tự danh nghĩa / ca khó. TB ± ĐLC mẫu qua các seed có số liệu trong từng ô. "
          "Các dòng seed ghi Wilson 95% cho tỉ lệ. Yaw là hệ số lệnh yaw. Họ `*_noyaw` giữ riêng theo policy gốc. "
          "Cấu hình mô phỏng lấy từ JSON và ghi đè khi chấm lại, hoặc từ mặc định của họ run. "
          "`p2y_perc` là họ chính để triển khai.",
          ["Họ run", "actuator", "yaw", "Seed", "Seed eval", "n lượt / Tổng hợp", "Nguồn JSON",
           *(label for _, _, label, _, _ in Q1_METRICS)], rows)

    rows = []
    for group in dict.fromkeys([*ordered, *(group for group, _ in diagnostics)]):
        family, actuator, yaw = group
        for case, label in CASES.items():
            entries = diagnostics[group, case]
            totals = defaultdict(int)
            for _, run in entries:
                for reason, count in get(run, "eval", case, "fail_breakdown").items():
                    if number(count):
                        totals[reason] += count
            sizes = [get(run, "eval", case, "success", "n") or get(run, "eval", "n") for _, run in entries]
            rows.append([family, actuator, f"{yaw:g}", label, len(entries),
                         fmt(sum(sizes) if sizes and all(map(number, sizes)) else None),
                         failure_text({"fail_breakdown": totals}) if entries else "—",
                         ", ".join(source(stem) for stem, _ in entries) or "—"])
    table(lines, "Q1 — phân loại lỗi",
          "Tổng số lượt theo lý do chính trong các JSON có `fail_breakdown`. "
          "Có thể dùng số liệu tham chiếu từ `sens_*` khi dòng chính cùng run chưa có chẩn đoán.",
          ["Họ run", "actuator", "yaw", "Ca", "Số run có chẩn đoán", "n lượt", "fail_breakdown", "Nguồn JSON"], rows)


SENS_AXES = {"odo_scale": "Hệ số nhiễu odometry", "delay_steps": "Trễ lệnh (ms)"}


def sensitivity_reports(lines, results, warn):
    rows, plots = [], {}
    for stem, result in results.items():
        sweeps = get(result, "eval", "sweeps")
        if not isinstance(sweeps, dict):
            warn(f"`{stem}.json` — thiếu lưới độ nhạy.")
            continue
        config = result.get("config")
        odo_reference = get(config, "reference_odo_scale")
        references = {"odo_scale": f"Trễ {range_text(get(config, 'reference_delay_steps'))} bước",
                      "delay_steps": f"odo_scale {fmt(odo_reference)}" if number(odo_reference) else "—"}
        series = {}
        prefix = [source(stem), fmt(get(result, "eval", "eval_seed")), fmt(get(result, "eval", "n"))]
        for axis, label in SENS_AXES.items():
            cells = sweeps.get(axis)
            if not isinstance(cells, list) or not cells:
                if axis == "odo_scale" and get(config, "v2") is False:
                    rows.append([*prefix, label, "Bỏ qua (v1)", "—", *q1_case_columns(None), "—", "—", "—"])
                else:
                    warn(f"`{stem}.json` — thiếu điểm {axis}.")
                continue
            points = []
            for cell in cells:
                x = get(cell, "odo_scale")
                tick = fmt(x)
                if axis == "delay_steps":
                    bounds = get(cell, "delay_ms")
                    if not isinstance(bounds, list) or len(bounds) != 2 or not all(map(number, bounds)):
                        warn(f"`{stem}.json` — bỏ qua điểm thiếu dải trễ.")
                        continue
                    x = mean(bounds)
                    tick = fmt(bounds[0]) if bounds[0] == bounds[1] else range_text(bounds)
                metrics = get(cell, "metrics")
                interval = wilson_point(get(metrics, "success"))
                if not number(x) or interval is None:
                    warn(f"`{stem}.json` — bỏ qua điểm {axis} thiếu mức thử hoặc khoảng Wilson.")
                    continue
                points.append((x, tick, *interval))
                rows.append([*prefix, label, tick, references[axis], *q1_case_columns(metrics),
                             *(fmt(get(metrics, key)) for key in ("touch_vn_mean", "touch_vt_mean", "touch_align_deg_mean"))])
            if points:
                series[axis] = sorted(points)
        if series:
            plots[stem] = (series, references)

    if not results:
        warn("P6 — chưa có kết quả độ nhạy `sens_*.json`.")
    table(lines, "Q1 — độ nhạy", "Các điểm dùng phân bố danh nghĩa và thay đổi từng yếu tố. "
          "vn, vt và góc đế–tấm lấy trung bình trên các lượt chạm đạt. "
          "JSON cũng lưu mốc tham chiếu danh nghĩa và ca khó theo cấu hình checkpoint.",
          ["Nguồn JSON", "Seed eval", "n lượt", "Yếu tố", "Mức thử", "Yếu tố giữ cố định",
           *Q1_CASE_COLUMNS, "vn (m/s)", "vt (m/s)", "Góc đế–tấm (°)"], rows)
    return plots


def wilson_errorbar(ax, x, intervals, **kwargs):
    p, lo, hi = zip(*intervals)
    ax.errorbar(x, [100 * value for value in p],
                yerr=([100 * (value - low) for value, low in zip(p, lo)],
                      [100 * (high - value) for value, high in zip(p, hi)]),
                fmt="o-", capsize=4, **kwargs)


def sensitivity_plot(plots):
    axes_keys = [axis for axis in SENS_AXES if any(axis in series for series, _ in plots.values())]
    fig, axes = plt.subplots(1, len(axes_keys), figsize=(5 * len(axes_keys), 4.2),
                             squeeze=False, sharey=True, layout="constrained")
    for ax, axis in zip(axes.flat, axes_keys):
        labels = defaultdict(dict)
        for index, (stem, (series, references)) in enumerate(plots.items()):
            if axis not in series:
                continue
            x, ticks, p, lo, hi = zip(*series[axis])
            for coord, tick in zip(x, ticks):
                labels[coord][tick] = None
            label = stem.removeprefix("sens_")
            if references[axis] != "—":
                label += f" · {references[axis]}"
            wilson_errorbar(ax, x, zip(p, lo, hi), label=label, color=f"C{index % 10}")
        ax.set_xticks(sorted(labels), [" / ".join(labels[x]) for x in sorted(labels)])
        ax.set(xlabel=SENS_AXES[axis])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Chạm đạt (%)")
    fig.suptitle("Q1 — độ nhạy, khoảng Wilson 95%")
    return fig


def drop_label(value):
    if isinstance(value, list):
        value = ",".join(map(str, value))
    if not isinstance(value, str):
        return "—"
    value = value.strip()
    names = {"imu": "IMU", "fnormal": "lực pháp tuyến", "contact": "tiếp xúc"}
    return "Bỏ " + " + ".join(names.get(key, key) for key in map(str.strip, value.split(","))) if value else "Đủ kênh"


def belief_table(lines, results, warn):
    rows = []
    ablations = defaultdict(list)
    for stem, run in results.items():
        test = get(run, "eval", "test_unseen_materials")
        if not any(number(get(test, key)) for key in BELIEF_METRICS):
            warn(f"`{stem}.json` — bỏ qua, thiếu số liệu mạng trên tập test.")
            continue
        config = run.get("config")
        data = get(config, "data")
        dataset = Path(data).name if isinstance(data, str) and data else "—"
        label = get(config, "label")
        drop = drop_label(get(config, "drop"))
        rows.append([source(stem), dataset, fmt(label), drop, fmt(run.get("seed")), fmt(get(test, "n")),
                     *(metric(test.get(key), 1, 4) for key in BELIEF_METRICS)])
        if label == "held" and number(test.get("auroc_hold")) and drop != "—":
            ablations[(dataset, data)].append((stem, drop, run.get("seed"), test["auroc_hold"]))
    table(lines, "Q2 — mạng niềm tin và ablation kênh",
          "Nguồn theo từng dòng. Chỉ số đầu ra giữ bám trên tập test vật liệu chưa thấy. ECE dùng 15 bin.",
          ["Nguồn JSON", "Dataset", "Nhãn", "Kênh", "Seed", "n cửa sổ", "AUROC", "ECE", "Brier"], rows)
    return ablations


def decision_tables(lines, results, warn):
    rows, h2_rows = [], []
    for stem, run in results.items():
        test = get(run, "eval", "test")
        seen = False
        for key, label in POLICIES.items():
            decision = get(test, key)
            if not isinstance(decision, dict) or "valid_perch" not in decision:
                continue
            seen = True
            rows.append([source(stem), label, fmt(run.get("seed")), fmt(decision.get("episodes")),
                         fmt(get(decision, "false_safe", "n")),
                         *(metric(decision.get(key)) for key in DECISION_METRICS)])
        if not seen:
            warn(f"`{stem}.json` — bỏ qua, thiếu kết quả quyết định trên tập test.")
        h2 = run.get("h2_same_false_safe_test")
        if not isinstance(h2, dict):
            continue
        for limit in (0.005, 0.01, 0.02):
            key = f"fs<={limit}"
            b4b = get(h2, "B4b_FSM_probe", key, "valid_perch")
            for policy, label in DTHR.items():
                dthr = get(h2, policy, key, "valid_perch")
                if not isinstance(get(h2, policy), dict):
                    continue
                b4a = get(h2, "B4a_FSM_no_probe", key, "valid_perch")
                delta = f"{100 * (dthr - b4b):+.3f}" if number(dthr) and number(b4b) else "—"
                h2_rows.append([source(stem), fmt(run.get("seed")), f"{limit * 100:g}", label,
                                metric(dthr, digits=3), metric(b4a, digits=3), metric(b4b, digits=3), delta])
    table(lines, "Q2 — bộ quyết định",
          "Nguồn theo từng dòng. Test vật liệu chưa thấy. False-safe là số lượt rơi trên số lượt tắt motor. "
          "Tỉ lệ và khoảng Wilson 95% lấy từ JSON.",
          ["Nguồn JSON", "Luật", "Seed", "n lượt", "n tắt motor", "Đậu hợp lệ (%)",
           "False-safe (%)", "Hủy (%)", "SELF_ABORT (%)"], rows)
    table(lines, "Q2 — H2",
          "Nguồn theo từng dòng. Mỗi ô dùng tỉ lệ đậu hợp lệ tốt nhất đã ghi trên đường cong test "
          "ở cùng trần false-safe. D-thr và B4b được đọc từ cùng file.",
          ["Nguồn JSON", "Seed", "False-safe tối đa (%)", "Luật D-thr", "D-thr đậu hợp lệ (%)",
           "B4a (%)", "B4b (%)", "D-thr − B4b (điểm %)"], h2_rows)


def sitl_table(lines, results, warn):
    rows = []
    for stem, run in results.items():
        test = get(run, "eval", "nominal")
        if not isinstance(test, dict) or not number(estimate(test.get("success"))):
            warn(f"`{stem}.json` — bỏ qua, thiếu kết quả SITL.")
            continue
        rows.append([source(stem), fmt(run.get("seed")), fmt(get(run, "eval", "n")),
                     *(metric(test.get(key)) for key in
                       ("touch_success", "success", "motors_off", "false_safe", "crash", "abort", "timeout", "recovered")),
                     fmt(test.get("tag_lost_pct"))])
    table(lines, "SITL",
          "Nguồn theo từng dòng. Kết quả danh nghĩa của các lượt đã hoàn tất, kèm khoảng Wilson 95% nếu JSON có.",
          ["Nguồn JSON", "Seed", "n lượt", "Chạm đạt (%)", "Đậu hợp lệ (%)", "Tắt motor (%)",
           "False-safe (%)", "Va chạm (%)", "Hủy (%)", "Hết giờ (%)", "Thu hồi (%)", "Mất dấu (%)"], rows)


SITL_MAIN = ("sitl_main_v0", "sitl_main_v15", "sitl_main_v3")
SITL_RATES = {
    "touch_success": "Chạm đạt", "held_if_motors_cut": "Giữ khi cắt motor", "false_safe": "False-safe",
    "rearm": "Re-arm", "recovered": "Về điểm chờ", "cycle_complete": "Chu kỳ trọn",
}
SITL_Q2_RATES = {key: label for key, label in SITL_RATES.items() if key != "false_safe"} | {
    "off_commanded": "Lệnh OFF", "false_safe_v12": "False-safe v1.2",
}
SITL_Q2_POOL_METRICS = {
    "valid_perch": "Đậu hợp lệ", "off_rate": "Lệnh OFF", "false_safe_v12": "False-safe v1.2",
    "false_safe_held": "False-safe held", "q2_abort": "Q2 ABORT", "self_abort": "SELF_ABORT", "crash": "Va chạm",
}
SITL_Q2_NOTE = ("False-safe v1.2 đếm crash hoặc SELF_ABORT sau khi phát lệnh OFF trong cửa sổ REST, "
                "chia cho số lượt đã phát lệnh OFF.")
SITL_MEANS = {
    "touch_vn_mean": "v_n (m/s)", "touch_vt_mean": "v_t (m/s)", "touch_align_deg_mean": "Lệch (°)",
    "touch_pos_cm_mean": "Sai vị trí (cm)", "tag_lost_pct": "Mất tag (%)",
}


def sitl_cases(run, *, has_trials=False):
    config = run.get("config")
    speeds = get(config, "speeds")
    nominal = get(run, "eval", "nominal")
    speed = speeds[0] if isinstance(speeds, list) and len(speeds) == 1 and number(speeds[0]) else None
    mu, force = range_text(get(config, "mu")), range_text(get(config, "f_hold_n"))
    if speed is not None and isinstance(nominal, dict):
        yield speed, mu, force, nominal
        return
    bins = get(run, "eval", "bins")
    emitted = False
    if isinstance(bins, dict):
        for name, cell in bins.items():
            match = re.fullmatch(r"speed_(\d+(?:\.\d+)?)(?:_mu_(.+)_f_hold_n_(.+))?", name)
            if match and isinstance(cell, dict):
                yield float(match[1]), fmt(match[2]), fmt(match[3]), cell
                emitted = True
    if not emitted and (isinstance(nominal, dict) or has_trials):
        yield speed, mu, force, nominal if isinstance(nominal, dict) else None


def sitl_law(stem, run):
    config = run.get("config")
    node_args = get(run, "provenance", "env", "PERCH_NODE_ARGS")
    args = dict(re.findall(r"(?:^|\s)--(verify-rule|q2-belief|q2-dthr-key)(?:=|\s+)(\S+)",
                           node_args if isinstance(node_args, str) else ""))
    args = {key: value.strip("\"'") for key, value in args.items()}
    batch = stem.removeprefix("sitl_q2_")
    match = re.match(r"(dthr_min|dthr|fsm_b4[be]|b4e|none|probe)(?:_|$)", batch)
    from_name = match[1] if match else "—"
    fields = ("verify_rule", "verify_mode", "verify", "q2_rule", "rule", "law")
    for candidate in (args.get("verify-rule"), *(get(config, field) or run.get(field) for field in fields), from_name):
        if not isinstance(candidate, str):
            continue
        law = candidate.lower().replace("-", "_").replace("d_thr", "dthr")
        if law == "b4e":
            law = "fsm_b4e"
        if law in ("none", "dthr", "dthr_min", "fsm_b4b", "fsm_b4e", "probe"):
            break
    else:
        return "—"
    if law == "dthr":
        dthr_key = args.get("q2-dthr-key") or get(config, "q2_dthr_key")
        if from_name == "dthr_min" or "minpull" in str(dthr_key).lower():
            return "dthr_min"
        belief = args.get("q2-belief") or get(config, "q2_belief") or get(config, "belief")
        if isinstance(belief, str) and belief:
            return f"dthr ({Path(belief).name})"
        if version := re.match(r"dthr_(v[467](?:full)?)(?:_|$)", batch):
            return f"dthr ({version[1]})"
    return law


def sitl_q2_trials(run):
    trials = run.get("trials")
    return trials if isinstance(trials, list) else get(run, "eval", "trials")


def sitl_q2_rates(run, cell, speed, mu, force):
    evaluation = run.get("eval")
    nominal = cell is None or cell is get(evaluation, "nominal")
    scopes = (cell, evaluation, run) if nominal else (cell,)
    rates = {key: next((value for scope in scopes if (value := get(scope, key)) is not None), None)
             for key in ("off_commanded", "false_safe_v12")}
    if all(value is not None for value in rates.values()):
        return rates
    trials = sitl_q2_trials(run)
    if not isinstance(trials, list):
        return rates
    if not nominal:
        case = {"speed_mps": speed}
        if mu != "—" and force != "—":
            try:
                case.update(mu=float(mu), f_hold_n=float(force))
            except ValueError:
                return rates
        trials = [trial for trial in trials if all(get(trial, key) == value for key, value in case.items())]
        if not trials:
            return rates

    # Load the scorer only when raw trials need scoring. The CLI starts in scripts/.
    sys.path.insert(0, str(PROJECT))
    try:
        from ros.q2_pool import off_commanded, wilson
    finally:
        sys.path.pop(0)
    off = [trial for trial in trials if off_commanded(trial)]
    if rates["off_commanded"] is None:
        rates["off_commanded"] = wilson(len(off), len(trials))
    if rates["false_safe_v12"] is None:
        fell = sum(trial.get("outcome") in ("crash", "self_abort") for trial in off)
        rates["false_safe_v12"] = (wilson(fell, len(off)) if all(trial.get("outcome") for trial in off)
                                   else {"n": len(off)})
    return rates


def sitl_q2_pooled_tables(lines, pooled):
    groups = pooled.get("groups", {})
    metrics = SITL_Q2_POOL_METRICS
    rows = [[name, ", ".join(source(f"sitl_{batch}") for batch in group.get("batches", [])) or "—",
             range_text(get(group, "config", "speeds")), ", ".join(map(str, group.get("eval_seeds", []))) or "—",
             ", ".join(group.get("status", [])) or "—", fmt(group.get("n")),
             *(count_rate(group.get(key)) for key in metrics)] for name, group in groups.items()]
    table(lines, "SITL Q2 — kết quả gộp",
          f"Tỉ lệ ghi k/n (% [Wilson 95%]). {SITL_Q2_NOTE} Nguồn {source('sitl_q2_pooled')}.",
          ["Nhóm", "Lô", "Tốc độ xe (m/s)", "Seed eval", "Trạng thái các lô", "n lượt", *metrics.values()], rows)

    rows = []
    for group, comparison in pooled.get("vs_baseline", {}).items():
        for key, stats in comparison.items():
            if key == "baseline":
                continue
            diff, lo, hi = (get(stats, field) for field in ("diff", "lo", "hi"))
            interval = f"{100 * diff:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]" if all(
                map(number, (diff, lo, hi))) else "—"
            rows.append([group, comparison["baseline"], metrics.get(key, key), interval,
                         fmt(get(stats, "fisher_p"), 4)])
    table(lines, "SITL Q2 — so sánh với đối chứng",
          "Hiệu số theo nhóm trừ đối chứng. Fisher hai phía, với độ làm tròn của JSON. "
          f"Nguồn {source('sitl_q2_pooled')}.",
          ["Nhóm", "Đối chứng", "Chỉ số", "Δ (điểm % [Newcombe 95%])", "p Fisher"], rows)


def sitl_reports(lines, results, warn):
    main, q2 = [], []
    for stem, run in results.items():
        if stem in SITL_MAIN:
            cases = main
        elif stem.startswith("sitl_q2_") and stem != "sitl_q2_pooled":
            cases = q2
        else:
            continue
        has_trials = cases is q2 and isinstance(sitl_q2_trials(run), list)
        entries = [(speed, stem, mu, force, cell, run) for speed, mu, force, cell in sitl_cases(run, has_trials=has_trials)]
        if not entries:
            warn(f"`{stem}.json` — thiếu kết quả SITL.")
        cases.extend(entries)
    for stem in SITL_MAIN:
        if stem not in results:
            warn(f"SITL — chưa có kết quả `{stem}.json`.")
    main.sort(key=lambda row: row[0] if number(row[0]) else math.inf)
    q2.sort(key=lambda row: (sitl_law(row[1], row[-1]), row[0] if number(row[0]) else math.inf))
    prefix = ["Nguồn JSON", "Tốc độ xe (m/s)", "μ", "Lực giữ (N)"]
    suffix = ["n đo chạm", *SITL_MEANS.values(), "Mất tag / PERCH (s)"]

    def row(speed, stem, mu, force, cell, run, *, q2_law=None):
        labels, rates = [], SITL_RATES
        if q2_law is not None:
            notes = []
            if "_lc" in stem or "_prereview" in stem or not get(run, "provenance", "perch_node_hold_sensors_qos"):
                notes.append("trước sửa QoS")
            if q2_law == "probe" or stem.startswith("sitl_q2_probe_"):
                notes.append("lô đo tín hiệu (ngoài đánh giá luật quyết định)")
            labels = [q2_law, ", ".join(notes) or "—"]
            cell = {**(cell or {}), **sitl_q2_rates(run, cell, speed, mu, force)}
            rates = SITL_Q2_RATES
        lost, perch = cell.get("tag_lost_s"), cell.get("perch_s")
        duration = f"{lost:g}/{perch:g}" if number(lost) and number(perch) else "—"
        return [*labels, source(stem), fmt(speed), mu, force,
                *(count_rate(cell.get(key)) for key in rates), fmt(cell.get("touch_n")),
                *(fmt(cell.get(key)) for key in SITL_MEANS), duration]

    note = ("Tỉ lệ ghi k/n (% [Wilson 95%]) theo mẫu số của từng metric. "
            "Số đo chạm là trung bình trên `n đo chạm`. Mất tag tính theo thời lượng PERCH.")
    table(lines, "SITL — theo tốc độ xe", note, [*prefix, *SITL_RATES.values(), *suffix],
          [row(*entry) for entry in main])
    if main:
        materials = sorted({entry[2] for entry in main if entry[2] != "—"})
        material_note = ""
        if len(materials) == 1 and "–" not in materials[0]:
            material_note = f" Lô chính dùng một vật liệu μ = {materials[0].replace('.', ',')}."
        lines.extend(["Lô chính dùng C2 v1.1 (gắn theo diện tích càng), "
                      "`--yaw-scale 0`, `--freeze-plate-v`, `--depart-pushoff`." + material_note, ""])
    if q2:
        table(lines, "SITL Q2 — theo luật", f"{note} {SITL_Q2_NOTE}",
              ["Luật", "Ghi chú lô", *prefix, *SITL_Q2_RATES.values(), *suffix],
              [row(*entry, q2_law=sitl_law(entry[1], entry[-1])) for entry in q2])
    else:
        warn("SITL Q2 — chưa có kết quả `sitl_q2_*.json`.")
    if "sitl_q2_pooled" in results:
        sitl_q2_pooled_tables(lines, results["sitl_q2_pooled"])
    return main


def sitl_speed_plot(entries):
    series = {}
    for key, label in SITL_RATES.items():
        points = [(speed, interval) for speed, _, _, _, cell, _ in entries
                  if number(speed) and (interval := wilson_point(cell.get(key))) is not None]
        if points:
            series[label] = points
    if not series:
        return None
    columns = min(3, len(series))
    rows = math.ceil(len(series) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 2.6 * rows),
                             squeeze=False, sharey=True, layout="constrained")
    for ax, (label, points) in zip(axes.flat, series.items()):
        speed, intervals = zip(*points)
        wilson_errorbar(ax, speed, intervals)
        ax.set(title=label, xticks=sorted(set(speed)), ylim=(-5, 105))
        ax.grid(axis="y", alpha=0.25)
    for ax in list(axes.flat)[len(series):]:
        ax.set_visible(False)
    fig.supxlabel("Tốc độ xe (m/s)")
    fig.supylabel("Tỉ lệ (%)")
    fig.suptitle("SITL — khoảng Wilson 95%")
    return fig


def learning_plot(results, runs_dir, warn):
    series = {}
    for run in dict.fromkeys(stem.removesuffix("_rescored") for stem in results):
        path = runs_dir / run / "train_log.json"
        log = read_json(path, warn, expected=list)
        if log is None:
            continue
        rows = sorted((row for row in log if number(get(row, "it"))), key=lambda row: row["it"])
        if not any(number(row.get("succ_rate")) or number(row.get("eval_succ")) for row in rows):
            warn(f"`{path}` — bỏ qua, thiếu tỉ lệ thành công theo vòng lặp.")
            continue
        series[run] = rows
    if not series:
        return None
    fig, axes = plt.subplots(len(series), 1, figsize=(9, 3 * len(series)), squeeze=False, layout="constrained")
    for ax, (run, rows) in zip(axes.flat, series.items()):
        for key, label, style in (("succ_rate", "Huấn luyện (succ_rate)", "-"),
                                  ("eval_succ", "Đánh giá (eval_succ)", "o-")):
            points = [(row["it"], 100 * row[key]) for row in rows if number(row.get(key))]
            if points:
                ax.plot(*zip(*points), style, label=label, linewidth=1.2, markersize=3)
            else:
                warn(f"`runs/{run}/train_log.json` — thiếu {key}.")
        delay_rows = [(row["it"], row["delay_hi"]) for row in rows if number(row.get("delay_hi"))]
        first = True
        for (_, before), (iteration, after) in zip(delay_rows, delay_rows[1:]):
            if after > before:
                ax.axvline(iteration, color="0.4", linestyle="--", linewidth=0.8,
                           label="Tăng trễ tối đa (delay_hi)" if first else None)
                first = False
        ax.set(title=run, xlabel="Vòng lặp (iteration)", ylabel="Chạm đạt (%)", ylim=(-3, 103))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    return fig


def success_bins_plot(results, title, warn):
    bins = {}
    for stem, result in results.items():
        run_bins = {}
        for case in CASES:
            case_bins = get(result, "eval", case, "bins")
            if case == "nominal" and (not case_bins or not isinstance(case_bins, dict)):
                case_bins = get(result, "eval", "bins")  # C3 also shows bins at this level.
            if isinstance(case_bins, dict):
                run_bins[case] = case_bins
        if not run_bins:
            warn(f"`{stem}.json` — bỏ qua hình theo bin, thiếu eval.bins.")
        else:
            bins[stem] = run_bins
    if not bins:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), sharex="col", sharey=True, layout="constrained")
    handles = {}
    for col, (prefix, xlabel) in enumerate((("speed_", "Tốc độ xe (m/s)"), ("tilt_", "Góc tấm (°)"))):
        series = defaultdict(dict)
        for stem, run_bins in bins.items():
            for case in CASES:
                for key, value in run_bins.get(case, {}).items():
                    interval = key.removeprefix(prefix)
                    if not key.startswith(prefix) or not re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", interval):
                        continue
                    value = get(value, "success") or value
                    p, lo, hi = (get(value, field) for field in ("p", "lo", "hi"))
                    if all(map(number, (p, lo, hi))) and lo <= p <= hi and get(value, "n") != 0:
                        series[case, stem][interval] = p, lo, hi
        intervals = sorted({key for points in series.values() for key in points},
                           key=lambda label: tuple(map(float, label.split("-"))))
        width = min(0.13, 0.65 / len(bins))
        for row, (case, case_label) in enumerate(CASES.items()):
            ax = axes[row, col]
            for index, stem in enumerate(bins):
                points = series.get((case, stem))
                if not points:
                    continue
                offset = (index - (len(bins) - 1) / 2) * width
                bars = []
                for idx, interval in enumerate(intervals):
                    if interval in points:
                        p, lo, hi = points[interval]
                        bars.append((idx + offset, 100 * p, 100 * (p - lo), 100 * (hi - p)))
                x, y, lower, upper = zip(*bars)
                handles[stem] = ax.errorbar(x, y, yerr=(lower, upper), fmt="o", capsize=3, markersize=4,
                                            color=f"C{index % 10}")
            ax.set(title=case_label, ylabel="Chạm đạt (%)" if col == 0 else "")
            ax.grid(axis="y", alpha=0.25)
            if not ax.has_data():
                ax.text(0.5, 0.5, "Chưa có số liệu theo bin", transform=ax.transAxes, ha="center")
        ax.set_xlabel(xlabel)
        ax.set_xticks(range(len(intervals)), [label.replace("-", "–") for label in intervals])
        ax.set_xlim(-0.5, max(0.5, len(intervals) - 0.5))
    if not handles:
        plt.close(fig)
        warn(f"{title} — bỏ qua hình theo bin, thiếu khoảng Wilson.")
        return None
    fig.suptitle(f"{title} — chạm đạt theo điều kiện, khoảng Wilson 95%")
    handles = {stem: handles[stem] for stem in bins if stem in handles}
    axes[0, 0].legend(handles.values(), handles.keys(), loc="lower left", fontsize=8)
    return fig


def risk_points(test, *, pareto=True):
    x, y = get(test, "false_safe"), get(test, "valid_perch")
    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        n_off = get(test, "n_off")
        if not isinstance(n_off, list) or len(n_off) != len(x):
            n_off = [None] * len(x)
        points = sorted({(a, b) for a, b, n in zip(x, y, n_off) if number(a) and number(b) and n != 0},
                        key=lambda point: (point[0], -point[1]))
        if not pareto:
            return points
        frontier = []
        for point in points:
            if not frontier or point[1] > frontier[-1][1]:
                frontier.append(point)
        return frontier
    x, y = estimate(x), estimate(y)
    return [(x, y)] if number(x) and number(y) else []


Q2_V4_METHODS = {
    "D-thr": (), "D-thr-late": (), "B4a": ("B4a_FSM_no_probe",), "B4b": ("B4b_FSM_probe",),
    "B4c": (), "B4d": (), "B7": (ORACLE,),
    "Tắt ngay": ("always_off", "always_off_at_touchdown_no_check"),
}


def q2_method_key(data, method):
    return next((key for key in (method, *Q2_V4_METHODS[method]) if isinstance(data, dict) and key in data), method)


def q2_caps(run):
    caps = get(run, "by_cap")
    if not isinstance(caps, dict):
        return []
    points = [(float(key), key, value) for key, value in caps.items()
              if re.fullmatch(r"0\.\d+|1(?:\.0+)?", key) and isinstance(value, dict)]
    return sorted(points)


def q2_lock_name(stem, run):
    ref = run.get("lock")
    return Path(ref).stem if isinstance(ref, str) and ref else f"{stem}_lock"


def q2_scope(stem, run):
    return "quy mô nhỏ" if any("small" in str(value) for value in (stem, get(run, "name"), get(run, "tag"))) else "đầy đủ"


def q2_lock(stem, run, results, warn):
    lock_stem = q2_lock_name(stem, run)
    lock = results.get(lock_stem)
    sha = run.get("lock_sha256")
    if not isinstance(lock, dict):
        warn(f"`{stem}.json` — thiếu `{lock_stem}.json`.")
        return {}, fmt(sha)
    if sha is not None and sha != lock.get("sha256"):
        warn(f"`{stem}.json` — hash không khớp `{lock_stem}.json`.")
        return {}, fmt(sha)
    return lock.get("config"), f"{fmt(sha or lock.get('sha256'))} · {source(lock_stem)}"


def q2_risk_cells(test):
    episodes = get(test, "episodes")
    n_off = get(test, "n_off")
    return [
        count_rate(get(test, "valid_perch"), k=get(test, "n_valid"), n=episodes),
        count_rate(get(test, "motors_off"), k=n_off, n=episodes),
        count_rate(get(test, "false_safe"), k=get(test, "n_fell"), n=n_off),
        *(count_rate(get(test, key), n=n_off) for key in ("false_safe_held", "false_safe_strict")),
        fmt(get(test, "exceeds_cap")),
    ]


def q2_seed_stat(value):
    p, sd, n = (get(value, key) for key in ("mean", "std", "n"))
    mean_sd = f"{100 * p:.2f} ± {fmt(100 * sd if number(sd) else None, 2)}" if number(p) else "—"
    return f"{mean_sd} (n={n})" if number(n) else mean_sd


def q2_v4_tables(lines, main, sensitivity, results, warn):
    rows, comparisons, seeds, sensor, locks = [], [], [], [], {}
    for stem, run in {**main, **sensitivity}.items():
        lock, sha = q2_lock(stem, run, results, warn)
        locks[stem] = lock
        reference = [source(stem), q2_scope(stem, run)]
        for cap, _, block in q2_caps(run):
            prefix = [*reference, f"{cap * 100:g}"]
            test = block.get("test")
            test = test if isinstance(test, dict) else {}
            for method in Q2_V4_METHODS:
                key = q2_method_key(test, method)
                score = get(test, key)
                if isinstance(score, dict):
                    status = "Đã chấm"
                elif key in test:
                    status = "Không có cấu hình trên dev_ood"
                else:
                    status = "Chưa có kết quả"
                cells = [sha, method, *q2_risk_cells(score), status]
                if stem in sensitivity:
                    if key in test:
                        cfg = get(lock, "cfg_kw")
                        sensor.append([*prefix, fmt(get(cfg, "vib_scale")), fmt(get(cfg, "slip_acc_in_imu")), *cells])
                else:
                    rows.append([*prefix, *cells])
            if stem in sensitivity:
                continue
            for comparison, score in (get(block, "comparisons") or {}).items():
                interval = get(score, "bootstrap_95")
                delta = {"p": get(score, "delta_valid_perch"),
                         "lo": get(interval, "lo"), "hi": get(interval, "hi")}
                comparisons.append([*prefix, comparison, fmt(get(score, "status")), metric(delta),
                                    fmt(get(interval, "n_defined"))])
            for method, score in (get(block, "across_seeds") or {}).items():
                methods = get(score, "scored_methods")
                seeds.append([*prefix, method, ", ".join(methods) if isinstance(methods, list) and methods else "—",
                              *(q2_seed_stat(get(score, key)) for key in ("valid_perch", "motors_off", "false_safe"))])

    prefix_headers = ["Nguồn JSON", "Quy mô", "Trần (%)"]
    score_headers = ["Hash cấu hình khóa", "Luật", "Đậu hợp lệ", "OFF", "Rơi/OFF", "FS held", "FS strict",
                     "exceeds_cap", "Trạng thái"]
    note = ("Các tỉ lệ ghi k/n (% [Wilson 95%]) nếu JSON có. `exceeds_cap` so ước lượng rơi/OFF trên test "
            "với trần. Các cấu hình có điều chỉnh ngưỡng được khóa trên dev_ood.")
    table(lines, "Q2 v4 — cấu hình khóa trên dev_ood", note, [*prefix_headers, *score_headers], rows)
    table(lines, "Q2 v4 — Δ bootstrap ghép cặp",
          "Δ đậu hợp lệ trên cùng các episode test, tính theo điểm phần trăm. "
          "Cột trạng thái và KTC bootstrap 95% lấy từ JSON.",
          [*prefix_headers, "Cặp luật", "Trạng thái", "Δ (điểm % [KTC])", "n mẫu bootstrap xác định"], comparisons)
    table(lines, "Q2 v4 — qua các seed mạng niềm tin",
          "TB ± ĐLC mẫu (%) và số seed có số liệu trong từng ô. Seed thiếu cấu hình khóa nằm ngoài thống kê.",
          [*prefix_headers, "Luật", "Run đã chấm", "Đậu hợp lệ", "OFF", "Rơi/OFF"], seeds)
    table(lines, "Q2 v4 — độ nhạy cảm biến", note + " Thông số cảm biến lấy từ tệp khóa tương ứng.",
          [*prefix_headers, "vib_scale", "slip_acc_in_imu", *score_headers], sensor)
    return locks


def q2_dev_point(dev):
    fell, off, valid, n = (get(dev, key) for key in ("n_fell", "n_off", "n_valid", "n"))
    if all(map(number, (fell, off, valid, n))) and off > 0 and n > 0:
        return [(fell / off, valid / n)]
    return risk_points(dev)


def q2_v4_plot(run, lock, curve, *, scope=""):
    curves = get(curve, "curves") or curve
    selection = get(lock, "selection")
    caps = q2_caps(run)
    panels = {}
    for split in ("dev_ood", "test"):
        series = {}
        for method in Q2_V4_METHODS:
            sweep = risk_points(get(curves, q2_method_key(curves, method), split), pareto=False)
            locked = []
            for _, key, block in caps:
                if split == "dev_ood":
                    dev = get(selection, q2_method_key(selection, method), key, split)
                    locked.extend(q2_dev_point(dev))
                else:
                    test = get(block, "test")
                    locked.extend(risk_points(get(test, q2_method_key(test, method))))
            if sweep or locked:
                series[method] = sweep, list(dict.fromkeys(locked))
        if series:
            panels[split] = series
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6),
                             squeeze=False, sharex=True, sharey=True, layout="constrained")
    for ax, (split, series) in zip(axes.flat, panels.items()):
        for index, method in enumerate(Q2_V4_METHODS):
            sweep, locked = series.get(method, ([], []))
            color = f"C{index}"
            if sweep:
                ax.plot(*zip(*[(100 * x, 100 * y) for x, y in sweep]), ".:", color=color, alpha=0.45,
                        label=f"{method} (lưới)" if not locked else None)
            if locked:
                ax.plot(*zip(*[(100 * x, 100 * y) for x, y in locked]), "o-", color=color, label=method)
        ax.set(title=split, xlabel="Rơi/OFF (%)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Đậu hợp lệ (%)")
    fig.suptitle("Q2 v4 — đậu hợp lệ theo false-safe" + (f" ({scope})" if scope else ""))
    return fig


def risk_plots(curves, decisions, warn):
    plots = {}
    for stem, data in curves.items():
        curve = data.get("curves", data)
        points = {policy: risk_points(get(curve, policy, "test")) for policy in (*DTHR, "B4b_FSM_probe", ORACLE)}
        if not any(points[policy] for policy in DTHR):
            warn(f"`{stem}.json` — bỏ qua hình, thiếu đường cong D-thr trên tập test.")
            continue
        decision = stem.removesuffix("_curve")
        # V2/v3 serialize the oracle as an operating point in the matching summary JSON.
        test = get(decisions, decision, "eval", "test")
        oracle_source = stem
        if not points[ORACLE]:
            points[ORACLE] = risk_points(get(test, ORACLE))
            oracle_source = decision
        if "B4b_FSM_probe" in (curve or {}) and not points["B4b_FSM_probe"]:
            warn(f"`{stem}.json` — thiếu điểm B4b có số lượt tắt motor trên tập test.")
        plots[stem] = (points, oracle_source)
    # Use same-suite comparisons when available. Older sweeps have only D-thr and an oracle point.
    comparative = {stem: data for stem, data in plots.items() if data[0]["B4b_FSM_probe"]}
    for stem, (points, oracle_source) in (comparative or plots).items():
        if not points["B4b_FSM_probe"]:
            warn(f"`{stem}.json` — hình Q2 thiếu B4b.")
        if not points[ORACLE]:
            warn(f"`{stem}.json` — hình Q2 thiếu điểm B7 oracle.")
        fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
        for policy, series in points.items():
            if series:
                x, y = zip(*[(100 * a, 100 * b) for a, b in series])
                if policy == ORACLE:
                    ax.scatter(x, y, marker="*", s=110, color="black", label=POLICIES[policy], zorder=4)
                else:
                    ax.step(x, y, where="post", marker=".", markersize=4, label=POLICIES[policy])
        ax.set(title="Q2 — đường cong rủi ro–hiệu quả trên tập test",
               xlabel="False-safe (% lượt tắt motor)", ylabel="Đậu hợp lệ (% lượt test)")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
        sources = ", ".join(source(name) for name in dict.fromkeys((stem, oracle_source))
                            if name in curves or name in decisions)
        description = "Biên Pareto trên tập test"
        if points[ORACLE]:
            description += ", với điểm B7 oracle"
        yield stem, fig, f"{description}. Nguồn {sources}."


def ablation_plot(ablations):
    groups = {}
    for dataset_key, entries in ablations.items():
        arms = defaultdict(list)
        for stem, drop, seed, auc in entries:
            arms[drop].append((stem, seed, auc))
        if any(drop != "Đủ kênh" for drop in arms):
            groups[dataset_key] = arms
    if not groups:
        return None, ""
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 4 * len(groups)), squeeze=False, layout="constrained")
    sources = []
    for ax, ((dataset, _), arms) in zip(axes.flat, groups.items()):
        shared_seeds = set.intersection(*(set(seed for _, seed, _ in rows if seed is not None) for rows in arms.values()))
        names, aurocs = [], []
        for drop in sorted(arms, key=lambda label: (label != "Đủ kênh", label)):
            rows = [(stem, auc) for stem, seed, auc in arms[drop] if not shared_seeds or seed in shared_seeds]
            names.append(drop)
            aurocs.append(mean(auc for _, auc in rows))
            sources.extend(source(stem) for stem, _ in rows)
        labels = [name.replace(" + ", " +\n") for name in names]
        bars = ax.bar(labels, aurocs, color=["#2a6f97" if name == "Đủ kênh" else "#659d70" for name in names])
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        seeds = ", ".join(map(str, sorted(shared_seeds)))
        ax.set(title=f"{dataset} · seed chung {seeds}" if seeds else f"{dataset} · trung bình các seed có số liệu",
               ylabel="AUROC giữ bám (không thứ nguyên)", ylim=(0, 1.1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Q2 — ablation kênh cảm biến trên tập test")
    return fig, ", ".join(sources)


def generate(results_dir, runs_dir):
    warnings = []

    def warn(message):
        warnings.append(message)
        print("Cảnh báo:", message, file=sys.stderr)

    results, debug = {}, []
    for path in sorted(results_dir.glob("*.json")):
        if re.fullmatch(r"sitl_p5.*_dbg.*", path.stem):
            debug.append(path.stem)
            continue
        data = read_json(path, warn)
        if data is None:
            continue
        status = data.get("status")
        if status in ("incomplete", "running", "failed", "pending", "cancelled", "canceled"):
            warn(f"`{path.name}` — bỏ qua, trạng thái {status}.")
            continue
        results[path.stem] = data

    selected = preferred_runs({name: run for name, run in results.items() if not name.startswith(("sens_", "sitl_"))})
    p1 = {name: run for name, run in results.items()
          if P1.fullmatch(name) and number(estimate(get(run, "eval", "nominal", "success")))}
    p2 = {stem: run for name, (stem, run) in selected.items() if P2.fullmatch(name)}
    expected = ["p1a_s0", *(f"p1b_s{seed}" for seed in range(3)),
                *(f"{family}_s{seed}" for family in ("p2_perc", "p2_norho", "p2y_perc") for seed in range(3))]
    missing = [f"`{name}.json`" for name in expected if name not in selected]
    if missing:
        warn("Thiếu kết quả " + ", ".join(missing) + ".")
    sens = {name: run for name, run in results.items() if name.startswith("sens_")}
    belief = {name: run for name, run in results.items() if name.startswith("q2_belief_")}
    decisions = {name: run for name, run in results.items()
                 if name.startswith("q2_decide_") and not name.endswith("_curve")}
    curves = {name: run for name, run in results.items()
              if name.startswith("q2_") and not name.startswith("q2_v4_") and name.endswith("_curve")}
    v4 = {name: run for name, run in results.items() if name.startswith("q2_v4_") and q2_caps(run)}
    v4_main = ({name: run for name, run in v4.items() if name.startswith("q2_v4_full")}
               or {name: run for name, run in v4.items() if name.startswith("q2_v4_small")})
    has_full_v4 = any(name.startswith("q2_v4_full") for name in v4_main)
    v4_sens = {name: run for name, run in v4.items() if name.startswith("q2_v4_sens_")}
    v4_curves = {name.removesuffix("_curve"): run for name, run in results.items()
                 if name.startswith(("q2_v4_full", "q2_v4_small")) and name.endswith("_curve")}
    sitl = {name: run for name, run in results.items() if name.startswith("sitl_")}
    for name in decisions:
        if name + "_curve" not in curves:
            warn(f"Thiếu đường cong `{name}_curve.json`.")
    if not belief:
        warn("Q2 — chưa có kết quả mạng niềm tin.")
    if not decisions:
        warn("Q2 — chưa có kết quả bộ quyết định.")
    if not curves and not decisions:
        warn("Q2 — chưa có dữ liệu đường cong.")
    if not has_full_v4:
        warn("Q2 v4 — chưa có kết quả quy mô đầy đủ `q2_v4_full*.json`.")
    if not v4_sens:
        warn("Q2 v4 — chưa có kết quả độ nhạy cảm biến `q2_v4_sens_*.json`.")

    lines = ["# Bảng kết quả PerchPilot", "", "`—` biểu thị số liệu chưa có hoặc tỉ lệ chưa xác định.", ""]
    q1_tables(lines, selected, sens, warn)
    p2_table(lines, p2, warn)
    sensitivity = sensitivity_reports(lines, sens, warn)
    sitl_main = sitl_reports(lines, sitl, warn)
    locks = q2_v4_tables(lines, v4_main, v4_sens, results, warn)
    p1_table(lines, p1, runs_dir, warn)
    ablations = belief_table(lines, belief, warn)
    decision_tables(lines, decisions, warn)
    legacy_sitl = {name: run for name, run in sitl.items() if name not in SITL_MAIN and not name.startswith("sitl_q2_")}
    if legacy_sitl:
        sitl_table(lines, legacy_sitl, warn)
    lines.extend(["## Hình", ""])
    outputs = []

    def save(filename, fig, description):
        if fig is None:
            return
        path = results_dir / "fig" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)
        lines.append(f"- [{filename}](fig/{filename}) — {description}")

    save("p1_learning.png", learning_plot(p1, runs_dir, warn),
         "Đường học P1 từ `runs/<name>/train_log.json`. Tên run nằm trên từng đồ thị.")
    for title, runs in (("P1", p1), ("P2", p2)):
        sources = ", ".join(source(name) for name in runs)
        save(f"{title.lower()}_success_bins.png", success_bins_plot(runs, title, warn),
             f"Chạm đạt theo bin. Nguồn {sources}.")
    if sensitivity:
        save("q1_sensitivity.png", sensitivity_plot(sensitivity),
             "So sánh policy theo lưới độ nhạy với khoảng Wilson 95%. Nguồn "
             + ", ".join(map(source, sensitivity)) + ".")
    for stem, series in sensitivity.items():
        save(f"{stem}.png", sensitivity_plot({stem: series}),
             f"Chạm đạt danh nghĩa theo từng yếu tố, với khoảng Wilson 95%. Nguồn {source(stem)}.")
    save("sitl_speed.png", sitl_speed_plot(sitl_main), "Tỉ lệ theo tốc độ xe, với khoảng Wilson 95%. Nguồn "
         + ", ".join(source(stem) for stem in dict.fromkeys(entry[1] for entry in sitl_main)) + ".")
    for stem in dict.fromkeys([*v4_main, *v4_curves]):
        if has_full_v4 and stem.startswith("q2_v4_small"):
            continue
        run, lock, curve = v4_main.get(stem), locks.get(stem), v4_curves.get(stem)
        refs = [source(stem)] if run else []
        if lock:
            refs.append(source(q2_lock_name(stem, run)))
        if curve:
            refs.append(source(f"{stem}_curve"))
        scope = q2_scope(stem, run or curve)
        save(f"{stem}_locked.png", q2_v4_plot(run, lock, curve, scope=scope),
             f"Đậu hợp lệ theo rơi/OFF ({scope}). Nguồn " + ", ".join(refs) + ".")
    for stem, fig, description in risk_plots(curves, decisions, warn):
        save(f"{stem}.png", fig, description)
    fig, sources = ablation_plot(ablations)
    save("q2_sensor_ablation.png", fig, f"AUROC ablation kênh. Nguồn {sources}.")
    if not outputs:
        lines.append("Chưa có dữ liệu để vẽ.")
    if debug:
        lines.extend(["", "## Phụ lục — tên file SITL debug", "", *(f"- {source(stem)}" for stem in debug)])
    if warnings:
        lines.extend(["", "## Dữ liệu thiếu hoặc chưa hoàn tất", "", *(f"- {msg}" for msg in warnings)])
    results_dir.mkdir(parents=True, exist_ok=True)
    tables = results_dir / "TABLES.md"
    tables.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [tables, *outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    args = parser.parse_args()
    for path in generate(args.results_dir, args.runs_dir):
        print(f"Đã sinh {path}")


if __name__ == "__main__":
    main()
