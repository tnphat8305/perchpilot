"""Self-tests of sim/contact_phase.py v2 (CPU, small N). Run: python q2/test_contact_phase.py"""
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.contact_phase import (ABORT, ABORTED, FELL, MOTORS_OFF, N_CH, OFF, PULL09, RELAX, RUNNING,  # noqa: E402
                               SELF_ABORT, ContactCfg, ContactPhaseSim, decision_steps)
from sim.quad_perch_env import G, hold_margin  # noqa: E402


def test_fork_matches_stepping():
    """REST computed in closed form == stepping the simulator with thrust_cmd = 0."""
    cfg = ContactCfg(num_envs=256, t_rest=1.0, max_contact_s=1.4)
    a = ContactPhaseSim(cfg, seed=3)
    a.run_script_to(decision_steps(cfg)[0])          # after RELAX
    idx = a.ar[a.status == RUNNING]
    ref = a.rest_fork(idx, a.k[idx], a.thrust[idx], a.pos[idx])
    pos0 = a.pos.clone()
    zero = torch.zeros(a.N)
    min_m = torch.full((a.N,), math.inf)
    path = torch.zeros(a.N)
    for _ in range(a.H):
        run = a.status == RUNNING
        a.step(T_cmd_override=zero)
        min_m = torch.minimum(min_m, a.margin_now)       # frozen at the loss step after a loss
        path += torch.where(run, a.v_slip.norm(dim=-1) * a.dt, torch.zeros_like(path))
    lost = (a.status[idx] == FELL) | (a.status[idx] == SELF_ABORT)
    strict = min_m[idx] >= 0
    held = ~lost & (path[idx] < cfg.held_slip_max)
    ag = {k: (x == ref[r]).float().mean().item() for k, x, r in
          (("fell", lost, "fell"), ("held_strict", strict, "held_strict"), ("held_star", held, "held"))}
    err_p = (path[idx] - ref["slip_path"]).abs()[~lost].max().item()
    print(f"fork vs step agreement {ag}, held_star frac {ref['held'].float().mean():.3f}, "
          f"strict frac {ref['held_strict'].float().mean():.3f}, max |d slip_path| {err_p:.2e} m")
    assert all(v == 1.0 for v in ag.values())


def test_extremes():
    """Sticky material holds; weak material never reaches a safe MOTORS_OFF and is never a FALL while powered."""
    for mu, fh, expect in (((1.5, 1.6),), ((5.0, 5.1),), 1.0), (((0.05, 0.06),), ((0.1, 0.11),), 0.0):
        cfg = ContactCfg(num_envs=128, mu=mu, f_hold_mg=fh, t_rest=2.0)
        s = ContactPhaseSim(cfg, seed=1)
        s.act(torch.full((s.N,), RELAX))
        for _ in range(decision_steps(cfg)[0]):
            s.step()
        s.act(torch.full((s.N,), MOTORS_OFF))
        off = s.status == OFF
        frac = (off & s.off_held).float().mean().item()
        print(f"mu={mu} f_hold={fh}: OFF {off.float().mean():.2f}, SELF_ABORT {(s.status == SELF_ABORT).float().mean():.2f}, "
              f"FELL {(s.status == FELL).float().mean():.2f}, held frac {frac:.3f} (expect {expect})")
        assert abs(frac - expect) < 0.05 and not (s.status == FELL).any()


def test_sensors_and_windows():
    cfg = ContactCfg(num_envs=128)
    s = ContactPhaseSim(cfg, seed=0)
    ks = decision_steps(cfg)
    assert ks == [30, 80, 130, 180], ks
    s.run_script_to(ks[0])
    alive = s.status == RUNNING
    e = s.ar[alive]
    w = s.windows(e, torch.full_like(e, ks[0]))
    assert w.shape == (e.numel(), 80, N_CH) and torch.isfinite(w).all()
    npre = 80 - (ks[0] + 1)                        # pre-touchdown samples in the first window
    print("contact pre/post:", w[:, :npre - 1, 7].mean().item(), w[:, npre + 1:, 7].mean().item())
    assert w[:, :npre - 1, 7].max() == 0 and w[:, npre + 1:, 7].mean() > 0.95
    accn = w[:, -10:, 0:3].norm(dim=-1).mean().item()
    fn_relax = w[:, -10:, 6].mean().item()
    # continue to the hold phase of PULL0.9 (signed load cell must show tension on average)
    s.run_script_to(ks[2] - ks[0] + 0)             # schedule continues from current state: RELAX done -> PULL0.3 -> PULL0.6
    s.act(torch.where(s.ready, torch.full((s.N,), PULL09), torch.full((s.N,), -1)))
    for _ in range(25):                            # ramp 10 + 15 into the hold
        s.step()
    al = s.status == RUNNING
    fn_pull = s.hist[s.ar[al], s.pre + s.k[al], 6].mean().item()
    print(f"|acc| end of RELAX {accn:.2f} (~{G:.2f}); f_normal RELAX {fn_relax:+.2f} N, PULL0.9 hold {fn_pull:+.2f} N; "
          f"attached during PULL0.9 {al.float().mean():.2f}")
    assert abs(accn - G) < 1.5 and fn_relax > 0 and fn_pull < 0
    a = torch.where(torch.arange(s.N) % 2 == 0, MOTORS_OFF, ABORT)
    for _ in range(25):
        s.step()
    s.act(a)
    assert ((s.status == OFF) | (s.status == ABORTED) | (s.status == SELF_ABORT)).all()


def test_margin_is_hold_margin():
    """forces()['margin'] is exactly hold_margin with the documented inputs (static case, t >> tau_imp)."""
    cfg = ContactCfg(num_envs=32, vn=(0.0, 0.0), vt=(0.0, 0.0), misalign_deg=(0.0, 0.0))
    s = ContactPhaseSim(cfg, seed=5)
    k = torch.full((s.N,), 150)
    f = s.forces(s.ar, k, torch.zeros(s.N))
    va = s.veh[s.ar, s.pre + k]
    acc = torch.stack([va[:, 0], va[:, 1], torch.zeros(s.N)], -1)
    ref = hold_margin(s.tilt, s.mu, s.f_hold, acc, s.n, torch.zeros(s.N, 3))
    assert torch.allclose(f["margin"], ref, atol=1e-4)
    fn_ref = G * torch.cos(s.tilt) + (acc * s.n).sum(-1)
    assert torch.allclose(f["fn"], fn_ref, atol=1e-3)
    # thrust along n reduces the normal load one-for-one (pulls the pad off)
    f2 = s.forces(s.ar, k, torch.full((s.N,), 0.9 * G))
    assert torch.allclose(f2["fn"], fn_ref - 0.9 * G, atol=1e-3)


if __name__ == "__main__":
    t0 = time.time()
    test_margin_is_hold_margin()
    test_sensors_and_windows()
    test_extremes()
    test_fork_matches_stepping()
    print(f"all tests passed in {time.time() - t0:.1f} s")
