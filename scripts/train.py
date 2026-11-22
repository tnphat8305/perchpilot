"""PerchPilot training entry point (P0/P1+).

Adds to the pilot script (train_pilot.py, kept for reproducing G1-G1d):
  * --seed, asymmetric critic (--asym), std clamp, critic warm-up after a warm start
  * phase-dependent entropy / lr cap once command delay is active
  * best checkpoint by periodic deterministic evaluation at the TARGET config
  * Wilson 95% intervals, results JSON (docs/CONTRACTS.md C3), TorchScript export (C1)
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.quad_perch_env import EnvCfg, QuadPerchEnv  # noqa: E402
from sim.quad_perch_env_v2 import EnvCfgV2, QuadPerchEnvV2  # noqa: E402
from rl.ppo import PPO, load_expanded  # noqa: E402


def make_env(n, seed, v2=False, *, device="cuda", **kw):
    if v2:
        return QuadPerchEnvV2(EnvCfgV2(num_envs=n, **kw), device=device, seed=seed)
    return QuadPerchEnv(EnvCfg(num_envs=n, **kw), device=device, seed=seed)


def wilson(k, n, z=1.96):
    if n == 0:
        return {"p": None, "lo": None, "hi": None, "n": 0}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"p": round(p, 4), "lo": round(c - h, 4), "hi": round(c + h, 4), "n": n}


LOST_TRACK_S = 0.5


def failure_masks(info, cfg):
    """Ordered primary reasons. The first match wins on each failed episode."""
    plate = info["c_plate"]
    reasons = {
        "ground": info["c_ground"],
        "crash_cart": info["c_vehicle"],
        "crash_plate_edge": info["c_plate_edge"],
        "bad_vn": plate & (info["vn"] >= cfg.vn_max),
        "bad_vt": plate & (info["vt"] >= cfg.vt_max),
        "bad_align": plate & (info["align"] <= cfg.align_cos_min),
        "bad_body_rate": plate & (info["body_rate"] >= cfg.w_max),
        "crash_plate_behind": info["c_behind"],
        "crash_flip": info["c_flip"],
        "crash_other": info["crash"],
    }
    if "t_loss" in info:
        reasons["lost_track"] = info["t_loss"] > LOST_TRACK_S
    reasons["timeout"] = info["timeout"]
    return reasons


@torch.no_grad()
def evaluate(ac, n=4096, seed=12345, v2=False, *, device="cuda", fail_breakdown=False, **cfg_kw):
    if n <= 0:
        raise ValueError("n must be positive")
    env = make_env(n, seed, v2=v2, device=device, **cfg_kw)
    obs = env.observe()
    first = torch.full((n,), -1, dtype=torch.long, device=device)
    vel, tilt = env.veh_s.clone(), env.tilt.clone()
    vn, vt, al = (torch.zeros(n, device=device) for _ in range(3))
    held = torch.zeros(n, dtype=torch.bool, device=device)
    vis_sum, vis_cnt = torch.zeros(n, device=device), torch.zeros(n, device=device)
    if fail_breakdown:
        first_reason = torch.zeros_like(first)
        lost_at_end = torch.zeros_like(held)
    for _ in range(env.max_steps + 1):
        obs, _, done, info = env.step(ac.act_deterministic(obs), diagnostics=fail_breakdown)
        if "visible" in info:
            alive = (first < 0).float()
            vis_sum += alive * info["visible"].float()
            vis_cnt += alive
        new = done & (first < 0)
        code = torch.where(info["success"], 1, torch.where(info["crash"], 2, 3))
        first = torch.where(new, code, first)
        if fail_breakdown:
            failed = new & ~info["success"]
            reasons = failure_masks(info, env.cfg)
            remaining = failed
            for i, mask in enumerate(reasons.values(), start=1):
                first_reason = torch.where(remaining & mask, i, first_reason)
                remaining = remaining & ~mask
            if v2:
                lost_at_end |= failed & reasons["lost_track"]
        vn, vt, al = (torch.where(new, info[k], x) for k, x in (("vn", vn), ("vt", vt), ("align", al)))
        held = torch.where(new, info["held"], held)
        if (first >= 0).all():
            break
    s = first == 1
    out = {"success": wilson(int(s.sum()), n), "crash": wilson(int((first == 2).sum()), n),
           "timeout": wilson(int((first == 3).sum()), n),
           "held_if_motors_cut": wilson(int(held.sum()), n),
           "touch_vn_mean": round(vn[s].mean().item(), 3) if s.any() else None,
           "touch_vt_mean": round(vt[s].mean().item(), 3) if s.any() else None,
           "touch_align_deg_mean": round(torch.rad2deg(torch.acos(al[s].clamp(-1, 1))).mean().item(), 2) if s.any() else None,
           "bins": {}}
    if v2:
        out["visible_frac_mean"] = round((vis_sum / vis_cnt.clamp(min=1)).mean().item(), 4)
    if fail_breakdown:
        counts = torch.bincount(first_reason, minlength=len(reasons) + 1).tolist()
        out["fail_breakdown"] = dict(zip(reasons, counts[1:]))
        out["fail_diagnostics"] = {
            "plate_edge_proxy": "Behind slab outside plate footprint. Physical edge geometry is unavailable.",
            "timeout_includes_out_of_range": True,
        }
        if v2:
            out["fail_diagnostics"].update(lost_track_threshold_s=LOST_TRACK_S,
                                            lost_track_at_end=int(lost_at_end.sum()))
        else:
            out["fail_diagnostics"]["unavailable"] = {"lost_track": "v1 has no detector-loss timer"}
    for lo, hi in ((0, 1), (1, 2), (2, 3.01)):
        m = (vel >= lo) & (vel < hi)
        out["bins"][f"speed_{lo}-{hi:g}"] = wilson(int((s & m).sum()), int(m.sum()))
    td = torch.rad2deg(tilt)
    for lo, hi in ((55, 62), (62, 70.01)):
        m = (td >= lo) & (td < hi)
        out["bins"][f"tilt_{lo}-{hi:g}"] = wilson(int((s & m).sum()), int(m.sum()))
    return out


class Exported(nn.Module):
    def __init__(self, ac):
        super().__init__()
        self.norm, self.actor = ac.norm, ac.actor

    def forward(self, obs):
        return self.actor(self.norm(obs)).clamp(-1.0, 1.0)


def export(ac, out, cfg, obs_dim, seed, git):
    m = Exported(ac).cpu().eval()
    ts = torch.jit.trace(m, torch.zeros(1, obs_dim))
    ts.save(str(out / "policy.ts"))
    layout = [["rel_p_uwn", 3], ["rel_v_uwn", 3], ["R_body_in_plate", 9], ["omega_body", 3], ["up_in_plate", 3],
              ["prev_act", 4], ["t_norm", 1], ["veh_speed", 1], ["plate_tilt", 1]]
    if cfg.act_history and int(cfg.delay_steps[1]) > 0:
        layout.append(["pending_act_buf", 4 * int(cfg.delay_steps[1])])
    if isinstance(cfg, EnvCfgV2):
        layout += [["rho", 1], ["visible", 1], ["t_loss_norm", 1]]
    meta = {"obs_dim": obs_dim, "obs_layout": layout, "act_layout": ["thrust01", "wx", "wy", "wz"],
            "rate_max": list(cfg.rate_max), "tw_ratio_nominal": sum(cfg.tw_ratio) / 2, "dt": cfg.dt,
            "delay_steps_trained": list(cfg.delay_steps), "pad_offset": cfg.pad_offset, "frame": "plate_uwn",
            "git": git, "seed": seed}
    (out / "policy.json").write_text(json.dumps(meta, indent=1))
    ac.cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--phase", default="P1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--calib", action="store_true", help="PX4-SITL delay (4-6 steps) and rate lags")
    ap.add_argument("--hist", action="store_true", help="pending-action buffer in obs")
    ap.add_argument("--timeout_penalty", type=float, default=None)
    ap.add_argument("--asym", action="store_true", help="asymmetric critic with privileged inputs")
    ap.add_argument("--v2", action="store_true", help="sim v2: odometry noise + camera FOV + KF + rho (P2)")
    ap.add_argument("--perc", action="store_true", help="v2 active-perception reward (PerchRL Eq. 13)")
    ap.add_argument("--motor", action="store_true", help="x500-like 4-motor allocation with saturation + airmode")
    ap.add_argument("--yaw_scale", type=float, default=None, help="scale on commanded yaw rate (0 = no yaw, as deployed)")
    ap.add_argument("--odo_scale", type=float, default=1.0)
    ap.add_argument("--pad_offset", type=float, default=None, help="pad below CoG [m]; x500 landing-gear bottom = 0.227")
    ap.add_argument("--tw", default=None, help="thrust-to-weight range 'lo,hi' (PerchRL 2.26 +-15%% -> 1.9,2.6)")
    ap.add_argument("--no_rho", action="store_true", help="ablation: rho, visible, t_loss inputs fixed (rho=1)")
    ap.add_argument("--init", default=None, help="warm start (state dict, dims may grow)")
    ap.add_argument("--critic_warmup", type=int, default=0, help="iterations with frozen actor after --init")
    ap.add_argument("--gate", action="store_true", help="success-gated delay curriculum")
    ap.add_argument("--gate_thr", type=float, default=0.7)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--ent_delay", type=float, default=None, help="entropy coef once delay_hi > 0")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_max", type=float, default=1e-3)
    ap.add_argument("--lr_max_delay", type=float, default=None)
    ap.add_argument("--std_max", type=float, default=None)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_n", type=int, default=1024)
    ap.add_argument("--out_root", default="/home/sup/data/perchpilot/runs")
    args = ap.parse_args()

    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    git = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "HEAD", "--", "."], cwd=ROOT, capture_output=True, text=True).stdout
    if diff:
        (out / "git.diff").write_text(diff)           # exact source of a run made from a dirty tree (review #20)
    base = dict(delay_steps=(4, 6), tau_rate=(0.035, 0.07), tau_yaw=(0.3, 0.6)) if args.calib else {}
    if args.hist:
        base["act_history"] = True
    if args.timeout_penalty is not None:
        base["timeout_penalty"] = args.timeout_penalty
    if args.pad_offset is not None:
        base["pad_offset"] = args.pad_offset
    if args.motor:
        base["motor_model"] = True
    if args.yaw_scale is not None:
        base["yaw_scale"] = args.yaw_scale
    if args.tw:
        base["tw_ratio"] = tuple(float(x) for x in args.tw.split(","))
    if args.v2:
        base["odo_scale"] = args.odo_scale
        if args.perc:
            base.update(perc_beta=0.05, perc_eta=0.02)
        if args.no_rho:
            base["aug_inputs"] = False
        base["v2"] = True
    v2 = base.pop("v2", False)
    cfg = (EnvCfgV2 if v2 else EnvCfg)(num_envs=args.envs, **base)
    env = make_env(args.envs, args.seed, v2=v2, **base)
    base_eval = dict(base, v2=v2)
    algo = PPO(env, lr=args.lr, ent=args.ent, asym=args.asym, lr_max=args.lr_max, std_max=args.std_max)
    if args.init:
        sd = torch.load(args.init, map_location="cuda")
        resized = load_expanded(algo.ac, sd)
        print(json.dumps({"warm_start": args.init, "resized": resized}), flush=True)
        if args.critic_warmup:
            algo.actor_frozen = True
        if args.std_max is not None:                  # apply the std cap before the first rollout (review #15)
            with torch.no_grad():
                algo.ac.log_std.clamp_(max=math.log(args.std_max))
    if args.gate:
        env.delay_hi = 0
        env.reset_idx(torch.arange(env.N, device="cuda"))   # resample delays under the gate (review #13)
        algo.obs = env.observe()
        if algo.priv_dim:
            algo.priv = env.privileged()
    recent, log = [], []
    best = {"succ": -1.0, "it": 0}
    t0, it, steps = time.monotonic(), 0, 0
    while time.monotonic() - t0 < args.minutes * 60:
        if algo.actor_frozen and it >= args.critic_warmup:
            algo.actor_frozen = False
        if args.gate and len(recent) >= 10 and sum(recent[-10:]) / 10 >= args.gate_thr and env.delay_hi < env.dmax:
            env.delay_hi += 1
            recent = []
        if env.delay_hi > 0 or not args.gate:
            if args.ent_delay is not None and env.dmax > 0:
                algo.ent = args.ent_delay
            if args.lr_max_delay is not None and env.dmax > 0:
                algo.lr_max = args.lr_max_delay
                if algo.lr > algo.lr_max:
                    algo.set_lr(algo.lr_max)
        ep = algo.collect()
        st = algo.update()
        it += 1
        steps += algo.H * env.N
        n_end = ep["success"] + ep["crash"] + ep["timeout"]
        row = {"it": it, "delay_hi": env.delay_hi, "wall_s": round(time.monotonic() - t0, 1), "env_steps": steps,
               "succ_rate": round(ep["success"] / max(n_end, 1), 4), "crash_rate": round(ep["crash"] / max(n_end, 1), 4),
               "std": round(algo.ac.log_std.exp().mean().item(), 4), "lr": algo.lr, "ent": algo.ent,
               "frozen": algo.actor_frozen, **st}
        log.append(row)
        recent.append(row["succ_rate"])
        if it % args.eval_every == 0 and (not args.gate or env.delay_hi == env.dmax):
            algo.ac.eval()
            q = evaluate(algo.ac, n=args.eval_n, seed=777, **base_eval)["success"]["p"]
            algo.ac.train()
            row["eval_succ"] = q
            if q > best["succ"]:
                best = {"succ": q, "it": it}
                torch.save(algo.ac.state_dict(), out / "best.pt")
        if it % 25 == 0:
            print(json.dumps(row), flush=True)
        if it % 200 == 0:
            torch.save(algo.ac.state_dict(), out / "ckpt.pt")
            (out / "train_log.json").write_text(json.dumps(log))
    torch.save(algo.ac.state_dict(), out / "ckpt.pt")
    if best["succ"] < 0:          # never reached target delay: keep last as "best"
        torch.save(algo.ac.state_dict(), out / "best.pt")
    algo.ac.load_state_dict(torch.load(out / "best.pt", map_location="cuda"))
    algo.ac.eval()
    ev = {"n": 4096, "eval_seed": 12345,
          "nominal": evaluate(algo.ac, **base_eval),
          "hard": evaluate(algo.ac, veh_speed=(2.0, 3.0), plate_tilt_deg=(65.0, 70.0), **base_eval)}
    export(algo.ac, out, cfg, env.obs_dim, args.seed, git)
    res = {"name": args.name, "phase": args.phase, "git": git, "seed": args.seed,
           "config": {**vars(args), "env": {k: list(v) if isinstance(v, tuple) else v for k, v in base.items()}},
           "eval": ev,
           "train": {"iterations": it, "env_steps": steps, "wall_min": round((time.monotonic() - t0) / 60, 2),
                     "best_iter": best["it"], "best_quick_eval": best["succ"], "final_delay_hi": env.delay_hi,
                     "gpu": torch.cuda.get_device_name(0)}}
    (out / "train_log.json").write_text(json.dumps(log))
    (out / "result.json").write_text(json.dumps(res, indent=1))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{args.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"name": args.name, "nominal": ev["nominal"]["success"], "hard": ev["hard"]["success"],
                      "best_iter": best["it"]}), flush=True)


if __name__ == "__main__":
    main()
