"""Đối chiếu CSV từ hold_margin_test với hold_margin của mô phỏng PerchPilot."""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from sim.quad_perch_env import hold_margin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1000:
        parser.error(f"cần đúng 1000 ca, nhận được {len(rows)}")

    def values(*columns, dtype=torch.float32):
        return torch.tensor(
            [[float(row[column]) for column in columns] for row in rows],
            dtype=dtype,
        )

    reference = hold_margin(
        values("tilt").squeeze(-1), values("mu").squeeze(-1),
        values("f_hold").squeeze(-1), values("ax", "ay", "az"),
        values("nx", "ny", "nz"), values("tx", "ty", "tz"),
    ).double()
    cpp = values("margin", dtype=torch.float64).squeeze(-1)
    matches = (reference.sign() == cpp.sign()).sum().item()
    max_error = (reference - cpp).abs().max().item()
    passed = matches >= 990 and max_error <= 1e-4
    print(f"Trùng dấu {matches}/{len(rows)} ({matches / len(rows):.2%})")
    print(f"Sai số tuyệt đối lớn nhất {max_error:.9g} m/s²")
    print("ĐẠT" if passed else "KHÔNG ĐẠT")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
