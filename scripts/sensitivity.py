"""Evaluate one-factor odometry and command-delay sweeps for a Q1 checkpoint."""

import argparse
from pathlib import Path

from eval_ckpt import git_describe, hard_config, load_actor, run_config, write_result
from train import EnvCfg, EnvCfgV2, evaluate


ODO_SCALES = (0, 1, 2, 4)
DELAY_STEPS = ((3, 3), (4, 6), (7, 7), (9, 9))


def sensitivity(run, n=4096, seed=12345, device="cuda"):
    result, v2, env = run_config(run)
    cfg = (EnvCfgV2 if v2 else EnvCfg)(**env)
    if cfg.dt != 0.01:
        raise ValueError("The 30–90 ms grid requires a 100 Hz checkpoint")
    actor = load_actor(run, v2, env, device)
    options = dict(n=n, seed=seed, v2=v2, device=device, fail_breakdown=True)
    nominal = evaluate(actor, **options, **env)
    hard = evaluate(actor, **options, **hard_config(env))

    odo = []
    if v2:
        for scale in ODO_SCALES:
            metrics = nominal if scale == cfg.odo_scale else evaluate(actor, **options, **{**env, "odo_scale": scale})
            odo.append({"odo_scale": scale, "metrics": metrics})

    delay = []
    reference_delay = cfg.command_delay_steps or cfg.delay_steps
    for steps in DELAY_STEPS:
        metrics = nominal if steps == reference_delay else evaluate(
            actor, **options, **{**env, "command_delay_steps": steps})
        delay.append({"delay_steps": steps, "delay_ms": [10 * step for step in steps], "metrics": metrics})

    return {
        "name": f"sens_{result['name']}", "phase": "P6", "git": git_describe(), "seed": seed,
        "config": {"env": env, "v2": v2, "device": str(device), "checkpoint_run": str(run),
                   "checkpoint_git": result.get("git"), "checkpoint_seed": result.get("seed"),
                   "act_history_steps": int(cfg.delay_steps[1]) if cfg.act_history else 0,
                   "reference_delay_steps": reference_delay,
                   "reference_odo_scale": cfg.odo_scale if v2 else None,
                   "odo_status": "evaluated" if v2 else "skipped because v1 lacks an odometry noise model"},
        "eval": {"n": n, "eval_seed": seed, "nominal": nominal, "hard": hard,
                 "sweeps": {"odo_scale": odo, "delay_steps": delay}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = sensitivity(args.run_dir, n=args.n, seed=args.seed, device=args.device)
    print(write_result(result["name"], result))


if __name__ == "__main__":
    main()
