"""Score hand-written B6 (hover) and B0 (direct) using only the v2 actor observation."""

import argparse
from pathlib import Path

import torch

from eval_ckpt import git_describe, hard_config, run_config, write_result
from train import EnvCfgV2, evaluate
from sim.quad_perch_env import G


P2_ENV = dict(delay_steps=(4, 6), tau_rate=(0.035, 0.07), tau_yaw=(0.3, 0.6),
              act_history=True, pad_offset=0.227, tw_ratio=(1.9, 2.6), odo_scale=1.0)


def limit_norm(v, limit):
    return v * (limit / v.norm(dim=1, keepdim=True).clamp(min=limit))


class Baseline:
    def __init__(self, direct, tw_nominal, rate_max, pad_offset):
        self.direct = direct
        self.tw_nominal = tw_nominal
        self.rate_max = rate_max
        self.pad_offset = pad_offset
        self.initial_height = None

    def act_deterministic(self, obs):
        rp, rv = obs[:, :3], obs[:, 3:6]
        rotation = obs[:, 6:15].reshape(-1, 3, 3)
        up = obs[:, 18:21]
        up = up / up.norm(dim=1, keepdim=True).clamp(min=1e-6)
        to_body = rotation.transpose(1, 2)
        if self.direct:
            desired_velocity = -1.2 * rp / rp.norm(dim=1, keepdim=True).clamp(min=1e-6)
            accel = 2.0 * (desired_velocity - rv)
            heading = -rp + (rp * up).sum(1, keepdim=True) * up
            heading_body = (to_body @ heading.unsqueeze(-1)).squeeze(-1)
            yaw = 2.0 * torch.atan2(heading_body[:, 1], heading_body[:, 0])
        else:
            # Reconstruct estimated drone velocity from estimated relative velocity and plate speed.
            forward = up * up[:, 2:3]
            forward[:, 2] -= 1.0  # horizontal projection of the inward normal
            forward = forward / forward.norm(dim=1, keepdim=True).clamp(min=1e-6)
            velocity = rv + obs[:, 26:27] * forward
            height = ((rp + self.pad_offset * rotation[:, :, 2]) * up).sum(1, keepdim=True)
            if self.initial_height is None:
                self.initial_height = height
            self.initial_height = torch.where(obs[:, 25:26] == 0, height, self.initial_height)
            accel = -2.0 * velocity + 1.5 * (self.initial_height - height) * up
            yaw = torch.zeros_like(obs[:, 0])

        thrust = G * up + limit_norm(accel, 5.0)
        magnitude = thrust.norm(dim=1, keepdim=True)
        thrust_direction = (to_body @ (thrust / magnitude).unsqueeze(-1)).squeeze(-1)
        rates = torch.stack([-4.0 * thrust_direction[:, 1], 4.0 * thrust_direction[:, 0], yaw], dim=1)
        action = torch.cat([2.0 * magnitude / (self.tw_nominal * G) - 1.0,
                            rates / obs.new_tensor(self.rate_max)], dim=1)
        return action.clamp(-1, 1)


def score_baselines(env, n=4096, seed=12345, device="cuda", matched_run=None):
    cfg = EnvCfgV2(**env)
    git = git_describe()
    options = dict(n=n, seed=seed, v2=True, device=device, fail_breakdown=True)
    for name, direct in (("b6_hover", False), ("b0_direct", True)):
        cases = {}
        for case, env_kw in (("nominal", env), ("hard", hard_config(env))):
            controller = Baseline(direct, sum(cfg.tw_ratio) / 2, cfg.rate_max, cfg.pad_offset)
            cases[case] = evaluate(controller, **options, **env_kw)
        yield {"name": name, "phase": "P6", "git": git, "seed": seed,
               "config": {"env": env, "v2": True, "device": str(device), "matched_run": matched_run},
               "eval": {"n": n, "eval_seed": seed, **cases}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Match a v2 run's environment from its result.json")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    env = P2_ENV
    if args.run is not None:
        _, v2, env = run_config(args.run)
        if not v2:
            parser.error("--run must use the v2 actor observation")
    for result in score_baselines(env, args.n, args.seed, args.device, str(args.run) if args.run else None):
        print(write_result(result["name"], result))


if __name__ == "__main__":
    main()
