"""G0: measure raw simulator throughput (env-steps/s) vs number of envs."""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.quad_perch_env import EnvCfg, QuadPerchEnv, hold_margin, G  # noqa: E402


def sanity_hold():
    """Hand-checked cases for the finite-hold model (motors off, static plate)."""
    dev = "cpu"
    th = torch.tensor([65.0]).deg2rad()
    n = torch.stack([-torch.sin(th), torch.zeros(1), torch.cos(th)], -1)
    zero = torch.zeros(1, 3)
    out = {}
    # need mu*(mg cos th + F) >= mg sin th  ->  with mu=0.5: F >= (sin/0.5 - cos) mg = 1.39 mg
    for f in (1.2, 1.5):
        m = hold_margin(th, torch.tensor([0.5]), torch.tensor([f * G]), zero, n, zero)
        out[f"mu0.5_F{f}mg_holds"] = bool(m.item() >= 0)
    return out


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    res = {"device": torch.cuda.get_device_name(0), "hold_sanity": sanity_hold(), "runs": []}
    for N in (1024, 4096, 8192, 16384):
        env = QuadPerchEnv(EnvCfg(num_envs=N), device="cuda")
        a = torch.zeros(N, 4, device="cuda")
        for _ in range(20):
            env.step(a)
        torch.cuda.synchronize()
        t0, K = time.time(), 200
        for _ in range(K):
            env.step(torch.rand(N, 4, device="cuda") * 2 - 1)
        torch.cuda.synchronize()
        sps = N * K / (time.time() - t0)
        res["runs"].append({"num_envs": N, "env_steps_per_s": round(sps), "sim_s_per_wall_s": round(sps * 0.01)})
        print(res["runs"][-1], flush=True)
        del env
        torch.cuda.empty_cache()
    print(json.dumps(res, indent=1))
    out = Path("/home/sup/data/perchpilot/g0_bench.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
