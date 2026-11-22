"""G1: state-based PPO pilot for agile perching on a tilted plate carried by a moving vehicle.

Privileged state (no vision yet) - goal is only to measure learnability and
wall-clock training time on the RTX 4060 before building the visuomotor stack.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.quad_perch_env import EnvCfg, QuadPerchEnv  # noqa: E402
from rl.ppo import PPO  # noqa: E402


@torch.no_grad()
def evaluate(ac, n=4096, seed=123, **cfg_kw):
    env = QuadPerchEnv(EnvCfg(num_envs=n, **cfg_kw), device="cuda", seed=seed)
    obs = env.observe()
    first = torch.full((n,), -1, dtype=torch.long, device="cuda")       # outcome of first episode per env
    vel = env.veh_s.clone()
    tilt = env.tilt.clone()
    vn = torch.zeros(n, device="cuda")
    vt = torch.zeros(n, device="cuda")
    al = torch.zeros(n, device="cuda")
    held = torch.zeros(n, dtype=torch.bool, device="cuda")
    for _ in range(env.max_steps + 1):
        obs, _, done, info = env.step(ac.act_deterministic(obs))
        new = done & (first < 0)
        code = torch.where(info["success"], 1, torch.where(info["crash"], 2, 3))
        first = torch.where(new, code, first)
        vn = torch.where(new, info["vn"], vn)
        vt = torch.where(new, info["vt"], vt)
        al = torch.where(new, info["align"], al)
        held = torch.where(new, info["held"], held)
        if (first >= 0).all():
            break
    s = first == 1
    out = {"success": s.float().mean().item(), "crash": (first == 2).float().mean().item(),
           "timeout": (first == 3).float().mean().item(),
           "held_if_motors_cut": held.float().mean().item(),
           "touch_vn_mean": vn[s].mean().item() if s.any() else None,
           "touch_vt_mean": vt[s].mean().item() if s.any() else None,
           "touch_align_deg_mean": torch.rad2deg(torch.acos(al[s].clamp(-1, 1))).mean().item() if s.any() else None,
           "by_speed": {}}
    for lo, hi in ((0, 1), (1, 2), (2, 3.01)):
        m = (vel >= lo) & (vel < hi)
        out["by_speed"][f"{lo}-{hi:g}m/s"] = s[m].float().mean().item()
    for lo, hi in ((55, 62), (62, 70.01)):
        m = (torch.rad2deg(tilt) >= lo) & (torch.rad2deg(tilt) < hi)
        out["by_speed"][f"tilt{lo}-{hi:g}deg"] = s[m].float().mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--calib", action="store_true", help="use PX4-SITL-identified delay / rate lags (G2)")
    ap.add_argument("--hist", action="store_true", help="action history in obs + timeout penalty + delay curriculum")
    ap.add_argument("--init", default=None, help="warm-start from a checkpoint (same obs dim)")
    ap.add_argument("--gate", action="store_true", help="success-gated delay curriculum: +1 step when success >= 0.7")
    ap.add_argument("--out", default="/home/sup/data/perchpilot/runs/g1_state_ppo")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    base = dict(delay_steps=(4, 6), tau_rate=(0.035, 0.07), tau_yaw=(0.3, 0.6)) if args.calib else {}
    if args.hist:
        base.update(act_history=True, timeout_penalty=10.0)
    env = QuadPerchEnv(EnvCfg(num_envs=args.envs, **base), device="cuda")
    algo = PPO(env)
    if args.init:
        algo.ac.load_state_dict(torch.load(args.init, map_location="cuda"))
    if args.gate:
        env.delay_hi = 0
    recent = []
    log, t0, it, steps = [], time.time(), 0, 0
    while time.time() - t0 < args.minutes * 60:
        if args.gate:
            if len(recent) >= 10 and sum(recent[-10:]) / 10 >= 0.7 and env.delay_hi < env.dmax:
                env.delay_hi += 1
                recent = []
        elif args.hist and env.dmax > 0:          # delay curriculum: 0 -> max over the first 600 iterations
            env.delay_hi = min(env.dmax, it * (env.dmax + 1) // 600)
        ep = algo.collect()
        st = algo.update()
        it += 1
        steps += algo.H * env.N
        n_end = ep["success"] + ep["crash"] + ep["timeout"]
        row = {"delay_hi": env.delay_hi, "it": it, "wall_s": round(time.time() - t0, 1), "env_steps": steps,
               "succ_rate": ep["success"] / max(n_end, 1), "crash_rate": ep["crash"] / max(n_end, 1),
               **{k: round(ep[k] / max(n_end, 1), 3) for k in ep if k.startswith("c_")},
               "std": algo.ac.log_std.exp().mean().item(), "lr": algo.lr, **st}
        log.append(row)
        recent.append(row["succ_rate"])
        if it % 25 == 0:
            print(json.dumps(row), flush=True)
        if it % 200 == 0:
            torch.save(algo.ac.state_dict(), out / "ckpt.pt")
    torch.save(algo.ac.state_dict(), out / "ckpt.pt")
    algo.ac.eval()
    ev = {"nominal": evaluate(algo.ac, **base)}
    ev["vehicle_2_3ms_tilt_65_70"] = evaluate(algo.ac, veh_speed=(2.0, 3.0), plate_tilt_deg=(65.0, 70.0), **base)
    summary = {"calib": base, "envs": args.envs, "iterations": it, "env_steps": steps, "wall_min": (time.time() - t0) / 60,
               "gpu": torch.cuda.get_device_name(0), "eval_deterministic": ev}
    (out / "train_log.json").write_text(json.dumps(log))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
