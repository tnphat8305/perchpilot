"""Re-evaluate a finished run's best.pt with the CURRENT simulator/metric code (e.g. after a metric fix).

Usage: python scripts/eval_ckpt.py RUN_DIR [--n 4096] [--seed 12345] [--tag rescored] [--device cuda]
Reads RUN_DIR/result.json for the env config, writes results/<name>_<tag>.json (C3 format).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from rl.ppo import ActorCritic  # noqa: E402
from train import evaluate, make_env  # noqa: E402


def run_config(run):
    result = json.loads((run / "result.json").read_text())
    env = {key: tuple(value) if isinstance(value, list) else value
           for key, value in result["config"]["env"].items()}
    return result, bool(result["config"].get("v2")), env


def load_actor(run, v2, env, device):
    probe = make_env(1, 0, v2=v2, device=device, **env)
    sd = torch.load(run / "best.pt", map_location=device, weights_only=True)
    obs_dim = sd["norm.mean"].numel()
    if probe.obs_dim != obs_dim:
        raise ValueError(f"Checkpoint expects {obs_dim} observations, env produces {probe.obs_dim}")
    priv_dim = sd["critic.0.weight"].shape[1] - obs_dim
    ac = ActorCritic(obs_dim, probe.act_dim, priv_dim=priv_dim).to(device)
    ac.load_state_dict(sd)
    return ac.eval()


def hard_config(env):
    return {**env, "veh_speed": (2.0, 3.0), "plate_tilt_deg": (65.0, 70.0)}


def git_describe():
    return subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=ROOT, text=True).strip()


def write_result(name, result, results_dir=ROOT / "results"):
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", default="rescored")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an env config field for evaluation, e.g. --set yaw_scale=0 --set motor_model=true")
    a = ap.parse_args()
    run = a.run
    res, v2, env_kw = run_config(run)
    for kv in a.set:
        k, v = kv.split("=", 1)
        env_kw[k] = {"true": True, "false": False}.get(v.lower(), None) if v.lower() in ("true", "false") else float(v)
    ac = load_actor(run, v2, env_kw, a.device)
    options = dict(n=a.n, seed=a.seed, v2=v2, device=a.device, fail_breakdown=True)
    ev = {"n": a.n, "eval_seed": a.seed,
          "nominal": evaluate(ac, **options, **env_kw), "hard": evaluate(ac, **options, **hard_config(env_kw))}
    out = {**res, "eval": ev, "rescored": {"tag": a.tag, "eval_code_git": git_describe(), "from": str(run),
                                                "overrides": a.set}}
    name = f"{res['name']}_{a.tag}"
    write_result(name, out)
    print(json.dumps({"name": name, "nominal": ev["nominal"]["success"], "hard": ev["hard"]["success"]}))


if __name__ == "__main__":
    main()
