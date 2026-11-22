"""Train and lock independent Q2 policies for the six slip-sensor models."""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / "q2" / script), *map(str, args)], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--vib_scale", type=float, choices=(1.0, 0.5, 0.0))
    ap.add_argument("--slip_acc_in_imu", type=int, choices=(1, 0))
    ap.add_argument("--stage", choices=("all", "data", "train", "eval"), default="all")
    ap.add_argument("--data_root", type=Path, default=Path("/home/sup/data/perchpilot/q2data"))
    ap.add_argument("--out_root", type=Path, default=Path("/home/sup/data/perchpilot/runs"))
    ap.add_argument("--episode_dir", type=Path, default=Path("/home/sup/data/perchpilot/q2eval"))
    ap.add_argument("--results_dir", type=Path, default=ROOT / "results")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--n_train", type=int, default=200_000)
    ap.add_argument("--n_val", type=int, default=20_000)
    ap.add_argument("--n_dev_ood", type=int, default=20_000)
    ap.add_argument("--n_test", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--dataset_seed", type=int, default=1, help="separate from main v4; paired across sensor models")
    ap.add_argument("--belief_seed", type=int, default=0)
    args = ap.parse_args()
    vib = (1, 0.5, 0) if args.vib_scale is None else (args.vib_scale,)
    slip = (1, 0) if args.slip_acc_in_imu is None else (args.slip_acc_in_imu,)
    for v, s in itertools.product(vib, slip):
        tag = f"sens_{args.tag}_v{v:g}_a{s}"
        print(f"{tag} {args.stage}", flush=True)
        data = args.data_root / f"v4_{tag}"
        common = ("--device", args.device, "--threads", args.threads)
        if args.stage in ("all", "data"):
            run("dataset.py", "--protocol", "v4", "--device", args.device, "--out", data,
                "--chunk", args.chunk, "--n_train", args.n_train, "--n_val", args.n_val,
                "--n_dev_ood", args.n_dev_ood, "--n_test", args.n_test, "--seed", args.dataset_seed,
                "--cfg", f"vib_scale={v},slip_acc_in_imu={s}")
        if args.stage in ("all", "train"):
            run("train_belief.py", *common, "--data", data, "--tag", f"v4_{tag}", "--out_root", args.out_root,
                "--results_dir", args.out_root / "metrics", "--epochs", args.epochs, "--seed", args.belief_seed)
        if args.stage in ("all", "eval"):
            run("decide_v4.py", *common, "--data", data, "--tag", tag,
                "--belief", args.out_root / f"q2_belief_v4_{tag}",
                "--results_dir", args.results_dir, "--episode_dir", args.episode_dir)


if __name__ == "__main__":
    main()
