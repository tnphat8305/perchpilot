"""Export a PPO checkpoint (rl/ppo.py ActorCritic state_dict, e.g. RUN/ckpt.pt) to contract C1:
policy.ts (TorchScript: RunningNorm + actor mean, clamped to [-1, 1]), policy.json, and the numpy
fallback policy.npz (proposed C1 extension, see README_perch_node.md).

Interim tool until train_pilot.py writes these files itself. Run with the occwm venv:
    /home/sup/data/venvs/occwm/bin/python perchpilot/ros/export_policy.py \
        --ckpt /home/sup/data/perchpilot/runs/g1_state_ppo/ckpt.pt --out /path/to/export_dir
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl.ppo import RunningNorm, mlp  # noqa: E402
from perch_core import OBS_V1_LAYOUT, TAG_OBS_LAYOUT, ACT_LAYOUT  # noqa: E402


class PolicyTS(nn.Module):
    def __init__(self, norm, actor):
        super().__init__()
        self.norm, self.actor = norm, actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.norm(obs)).clamp(-1.0, 1.0)


def actor_from_state_dict(state):
    """Load the actor and its normalizer from a symmetric or asymmetric PPO checkpoint."""
    obs_dim = state["norm.mean"].numel()
    act_dim = state["log_std"].numel()
    actor = PolicyTS(RunningNorm(obs_dim), mlp(obs_dim, act_dim))
    actor.load_state_dict({key: value for key, value in state.items() if key.startswith(("actor.", "norm."))})
    return actor


def mlp_to_npz(norm, actor, path):
    lin = [m for m in actor if isinstance(m, nn.Linear)]
    acts = [m for m in actor if not isinstance(m, nn.Linear)]
    if not all(isinstance(a, nn.ELU) for a in acts):
        raise ValueError("numpy runner supports ELU only")
    d = {"norm_mean": norm.mean.cpu().numpy(), "norm_var": norm.var.cpu().numpy(), "n_layers": np.array(len(lin))}
    for i, m in enumerate(lin):
        d[f"W{i}"] = m.weight.detach().cpu().numpy()
        d[f"b{i}"] = m.bias.detach().cpu().numpy()
    np.savez(path, **d)


def export(ac, out, meta):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mod = torch.jit.script(PolicyTS(ac.norm, ac.actor).eval().cpu())
    mod.save(str(out / "policy.ts"))
    mlp_to_npz(ac.norm, ac.actor, out / "policy.npz")
    (out / "policy.json").write_text(json.dumps(meta, indent=1))
    return out


def make_meta(obs_dim, rate_max=(6.0, 6.0, 3.0), tw=3.0, delay=(0, 0), git="unknown", seed=0, source=""):
    hist = obs_dim - 28
    tag_inputs = hist % 4 == 3             # v2 appends three values after action history
    if tag_inputs:
        hist -= 3
    layout = [list(x) for x in OBS_V1_LAYOUT] + ([["act_hist", hist]] if hist else [])
    if tag_inputs:
        layout += [list(x) for x in TAG_OBS_LAYOUT]
    return {"obs_dim": obs_dim, "obs_layout": layout, "act_layout": ACT_LAYOUT, "rate_max": list(rate_max),
            "tw_ratio_nominal": tw, "dt": 0.01, "delay_steps_trained": list(delay), "frame": "plate_uwn",
            "git": git, "seed": seed, "source": source}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tw", type=float, default=3.0, help="tw_ratio_nominal (EnvCfg.tw_ratio midpoint)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu")
    ac = actor_from_state_dict(sd)
    obs_dim = ac.norm.mean.numel()
    delay = (0, 0)
    summ = Path(a.ckpt).parent / "summary.json"
    if summ.exists():
        delay = tuple(json.loads(summ.read_text()).get("calib", {}).get("delay_steps", (0, 0)))
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent).stdout.strip() + " (export time, not train time)"
    except Exception:  # noqa: BLE001
        git = "unknown"
    out = export(ac, a.out, make_meta(obs_dim, tw=a.tw, delay=delay, git=git, seed=a.seed, source=str(a.ckpt)))
    print("exported", out, "obs_dim", obs_dim)


if __name__ == "__main__":
    main()
