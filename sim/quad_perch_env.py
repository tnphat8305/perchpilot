"""PerchPilot — batched GPU simulator (pilot version, G0).

Quadrotor commanded by collective thrust + body rates (same interface as PX4
/fmu/in/vehicle_rates_setpoint + vehicle_thrust_setpoint), a ground vehicle
driving a random smooth path and carrying a tilted perch plate, and a
finite-holding attachment model (magnetic/velcro pad + friction) whose
material parameters are hidden from the policy.

Everything is vectorised over N environments with torch (no Python loop over
envs). World frame: ENU-like, z up. Body frame: x forward, z up (thrust axis).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

G = 9.81


@dataclass
class EnvCfg:
    num_envs: int = 4096
    dt: float = 0.01            # policy period (100 Hz)
    substeps: int = 2           # physics at 200 Hz
    episode_s: float = 6.0
    # action scaling
    rate_max: tuple = (6.0, 6.0, 3.0)   # rad/s roll, pitch, yaw
    # domain randomisation ranges (lo, hi)
    tw_ratio: tuple = (2.6, 3.4)        # thrust-to-weight
    tau_rate: tuple = (0.025, 0.06)     # PX4 rate-loop first-order lag [s] (re-identify in G2)
    tau_yaw: tuple = None               # None -> same as tau_rate; SITL fit gives ~0.5 s
    delay_steps: tuple = (0, 0)         # pure command delay in policy steps (SITL fit: 45-55 ms incl. ROS)
    tau_thrust: tuple = (0.02, 0.05)
    drag: tuple = (0.05, 0.3)           # linear drag [1/s]
    plate_tilt_deg: tuple = (55.0, 70.0)
    veh_speed: tuple = (0.0, 3.0)       # m/s
    plate_height: tuple = (1.1, 1.5)    # m above ground
    plate_half: float = 0.22            # half side of the square plate [m]
    pad_offset: float = 0.05            # pad below CoG along -z_body [m]
    # touchdown envelope (kinematic success)
    align_cos_min: float = math.cos(math.radians(25.0))
    vn_max: float = 1.5                 # approach speed into plate [m/s]
    vt_max: float = 1.0                 # tangential slip speed [m/s]
    w_max: float = 5.0                  # body-rate norm at contact [rad/s]
    # hidden attachment parameters
    hold_force_mg: tuple = (0.6, 3.0)   # pad normal holding force, in units of m*g
    mu: tuple = (0.25, 1.0)             # pad/plate friction
    obs_noise: float = 0.01
    act_history: bool = False           # append the pending (delayed) action buffer to the observation
    timeout_penalty: float = 0.0
    # Evaluation override. delay_steps still defines the trained observation layout.
    command_delay_steps: tuple | None = field(default=None, kw_only=True)
    # Actuator model. False = ideal (rates follow a first-order lag regardless of thrust; all published P1/P2 runs).
    # True = x500-like 4-motor allocation with per-motor saturation and PX4 airmode semantics (P5 finding, 2026-11-19:
    # the ideal model lets the policy use full collective AND full roll/pitch authority at once, which x500 cannot).
    motor_model: bool = field(default=False, kw_only=True)
    inertia_per_mass: tuple = field(default=(0.0217 / 2.064, 0.0217 / 2.064, 0.040 / 2.064), kw_only=True)  # I/m [m^2]
    inertia_scale: tuple = field(default=(0.9, 1.1), kw_only=True)
    rotor_xy: tuple = field(default=((0.13, -0.22), (-0.13, 0.20), (0.13, 0.22), (-0.13, -0.20)), kw_only=True)
    rotor_spin: tuple = field(default=(1.0, 1.0, -1.0, -1.0), kw_only=True)
    km: float = field(default=0.016, kw_only=True)            # yaw torque / thrust [m]
    f_min_frac: float = field(default=0.26 / 11.44, kw_only=True)  # idle motor thrust / max motor thrust
    # Scale on the commanded yaw rate (deployment sends 0: PX4 lowers collective to make room for yaw, P5 2026-11-20)
    yaw_scale: float = field(default=1.0, kw_only=True)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_to_rot(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


def rotz(psi):
    c, s = torch.cos(psi), torch.sin(psi)
    z, o = torch.zeros_like(psi), torch.ones_like(psi)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).view(*psi.shape, 3, 3)


def hold_margin(tilt, mu, f_hold, extra_acc_world, n, thrust_acc_world):
    """Static holding check of a pad stuck on a plane with outward normal n.

    All forces per unit mass (accelerations). Required contact force from the
    plate on the drone in the plate (non-inertial) frame:
        f_c = a_plate - g_vec - thrust_acc
    Pad can pull at most f_hold (adhesion) and press arbitrarily; Coulomb
    friction with normal load N = f_hold + (pressing part).
    Returns margin >= 0 when holding (slack of the friction cone).
    """
    g_vec = torch.zeros_like(n)
    g_vec[..., 2] = -G
    f_c = extra_acc_world - g_vec - thrust_acc_world      # force plate must exert
    fn = (f_c * n).sum(-1)                                 # + = plate pushes out (drone pressed in)
    ft = (f_c - fn.unsqueeze(-1) * n).norm(dim=-1)
    # normal: plate can push (fn>0) freely, can pull at most f_hold (fn >= -f_hold)
    normal_ok = fn + f_hold
    # friction available: mu * (pressing force + magnet preload)
    fric_ok = mu * (fn.clamp(min=0.0) + f_hold) - ft
    return torch.minimum(normal_ok, fric_ok)


class QuadPerchEnv:
    act_dim = 4

    def __init__(self, cfg: EnvCfg, device="cuda", seed=0):
        self.cfg, self.dev = cfg, torch.device(device)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        N = cfg.num_envs
        self.N = N
        self.max_steps = int(cfg.episode_s / cfg.dt)
        self.obs_dim = 28 + (4 * int(cfg.delay_steps[1]) if cfg.act_history else 0)
        command_delay_steps = cfg.command_delay_steps or cfg.delay_steps
        self.delay_hi = int(command_delay_steps[1])    # curriculum can lower this
        f = lambda *s: torch.zeros(*s, device=self.dev)
        self.p, self.v, self.w = f(N, 3), f(N, 3), f(N, 3)
        self.q = f(N, 4)
        self.thrust = f(N)                      # actual collective accel [m/s^2]
        self.prev_act = f(N, 4)
        self.dmax = int(cfg.delay_steps[1])
        self.delay_buf_steps = max(self.dmax, self.delay_hi)
        self.act_buf = f(N, self.delay_buf_steps + 1, 4)
        self.delay = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.ar = torch.arange(N, device=self.dev)
        # privileged vector for an asymmetric critic (never given to the actor)
        self.priv_dim = 12 + 4 * (self.dmax + 1)
        # vehicle
        self.veh_p, self.veh_psi, self.veh_s = f(N, 3), f(N), f(N)
        self.veh_s_tgt, self.veh_kappa = f(N), f(N)
        self.veh_acc = f(N, 3)
        # randomised params
        self.tw, self.tau_T, self.kd = f(N), f(N), f(N)
        self.tau_w = f(N, 3)
        self.tilt, self.ph, self.f_hold, self.mu = f(N), f(N), f(N), f(N)
        self.t = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.prev_dist = f(N)
        self.prev_d = f(N)
        self.rate_max = torch.tensor(cfg.rate_max, device=self.dev)
        if cfg.motor_model:
            P = torch.tensor(cfg.rotor_xy, device=self.dev)
            S = torch.tensor(cfg.rotor_spin, device=self.dev)
            self.B = torch.stack([P[:, 1], -P[:, 0], cfg.km * S])     # 3x4: body torque per unit motor thrust
            self.Bpinv = torch.linalg.pinv(self.B)                     # 4x3
            self.Jm = f(N, 3)                                          # inertia / mass per env
        # episode stats
        self.stat_success = f(N)
        self.stat_hold = f(N)
        self.reset_idx(torch.arange(N, device=self.dev))

    # ------------------------------------------------------------------ utils
    def _u(self, n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, device=self.dev, generator=self.gen)

    def plate_frame(self):
        R = rotz(self.veh_psi)
        th = self.tilt
        nb = torch.stack([-torch.sin(th), torch.zeros_like(th), torch.cos(th)], -1)
        n = (R @ nb.unsqueeze(-1)).squeeze(-1)                         # outward normal (faces backwards-up)
        u = R[..., :, 1]                                               # lateral axis
        w = torch.cross(n, u, dim=-1)                                  # up-slope axis in plate
        c = self.veh_p + (R @ torch.tensor([-0.3, 0.0, 0.0], device=self.dev).expand(self.N, 3).unsqueeze(-1)).squeeze(-1)
        c = c.clone()
        c[:, 2] = self.ph
        vel = self.veh_s.unsqueeze(-1) * R[..., :, 0]
        return c, n, u, w, vel

    # ------------------------------------------------------------------ reset
    def reset_idx(self, idx):
        if idx.numel() == 0:
            return
        c, n = self.cfg, idx.numel()
        u = lambda lo_hi: self._u(n, *lo_hi)
        self.tw[idx] = u(c.tw_ratio)
        trp = u(c.tau_rate)
        self.tau_w[idx] = torch.stack([trp, trp, u(c.tau_yaw) if c.tau_yaw else trp], -1)
        command_delay_steps = c.command_delay_steps or c.delay_steps
        lo = min(int(command_delay_steps[0]), self.delay_hi)
        self.delay[idx] = torch.randint(lo, self.delay_hi + 1, (n,), device=self.dev, generator=self.gen)
        self.tau_T[idx] = u(c.tau_thrust)
        if c.motor_model:
            js = u(c.inertia_scale)
            self.Jm[idx] = torch.tensor(c.inertia_per_mass, device=self.dev) * js.unsqueeze(-1)
        self.kd[idx] = u(c.drag)
        self.tilt[idx] = torch.deg2rad(u(c.plate_tilt_deg))
        self.ph[idx] = u(c.plate_height)
        self.f_hold[idx] = u(c.hold_force_mg) * G
        self.mu[idx] = u(c.mu)
        self.veh_p[idx] = 0.0
        self.veh_psi[idx] = self._u(n, -math.pi, math.pi)
        self.veh_s[idx] = u(c.veh_speed)
        self.veh_s_tgt[idx] = u(c.veh_speed)
        self.veh_kappa[idx] = self._u(n, -0.08, 0.08)
        self.veh_acc[idx] = 0.0
        # drone behind & above the plate, moving with the vehicle
        R = rotz(self.veh_psi[idx])
        off = torch.stack([self._u(n, -6.0, -3.0), self._u(n, -1.5, 1.5), self._u(n, 0.2, 1.8)], -1)
        base = torch.zeros(n, 3, device=self.dev)
        base[:, 2] = self.ph[idx]
        self.p[idx] = base + (R @ off.unsqueeze(-1)).squeeze(-1)
        self.v[idx] = self.veh_s[idx].unsqueeze(-1) * R[..., :, 0] + 0.3 * torch.randn(n, 3, device=self.dev, generator=self.gen)
        yaw = self.veh_psi[idx] + self._u(n, -0.3, 0.3)
        rp = 0.1 * torch.randn(n, 2, device=self.dev, generator=self.gen)
        qy = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        qr = torch.stack([torch.ones(n, device=self.dev), rp[:, 0] / 2, rp[:, 1] / 2, torch.zeros(n, device=self.dev)], -1)
        q = quat_mul(qy, qr)
        self.q[idx] = q / q.norm(dim=-1, keepdim=True)
        self.w[idx] = 0.0
        self.thrust[idx] = G
        self.prev_act[idx] = 0.0
        # nominal TW (not the randomised one) so the initial command history does not leak TW to the actor
        tw_nom = 0.5 * (c.tw_ratio[0] + c.tw_ratio[1])
        self.prev_act[idx, 0] = 2.0 / tw_nom - 1
        self.act_buf[idx] = self.prev_act[idx].unsqueeze(1)
        self.t[idx] = 0
        pc, pn, _, _, _ = self.plate_frame()
        pad = self._pad_pos()
        self.prev_dist[idx] = (pad[idx] - pc[idx]).norm(dim=-1)
        self.prev_d[idx] = ((pad[idx] - pc[idx]) * pn[idx]).sum(-1)

    def _pad_pos(self):
        R = quat_to_rot(self.q)
        return self.p - self.cfg.pad_offset * R[..., :, 2]

    # ------------------------------------------------------------------ obs
    def observe(self):
        pc, n, u, w, pv = self.plate_frame()
        R = quat_to_rot(self.q)
        pad = self.p - self.cfg.pad_offset * R[..., :, 2]
        rel_p = pad - pc
        rel_v = self.v - pv
        # express relative quantities in the plate frame (u, w, n) -> heading invariant
        Pm = torch.stack([u, w, n], -2)                                # rows = plate axes
        rp = (Pm @ rel_p.unsqueeze(-1)).squeeze(-1)
        rv = (Pm @ rel_v.unsqueeze(-1)).squeeze(-1)
        Rp = Pm @ R                                                    # body axes in plate frame
        gp = Pm[..., :, 2]                                             # world up in plate frame
        obs = torch.cat([rp, rv, Rp.flatten(1), self.w, gp, self.prev_act,
                         (self.t.float() / self.max_steps).unsqueeze(-1),
                         pv.norm(dim=-1, keepdim=True), self.tilt.unsqueeze(-1)]
                        + ([self.act_buf[:, 1:self.dmax + 1].flatten(1)] if self.cfg.act_history and self.dmax > 0 else []), -1)
        obs = obs + self.cfg.obs_noise * torch.randn(obs.shape, device=self.dev, generator=self.gen)
        return obs

    def privileged(self):
        """Hidden parameters + true pending command buffer (critic only, scaled to O(1))."""
        return torch.cat([
            (self.delay.float() / max(self.dmax, 1)).unsqueeze(-1),
            (self.tw / 3.0).unsqueeze(-1),
            self.tau_w * 10.0,
            (self.tau_T * 20.0).unsqueeze(-1),
            (self.kd * 3.0).unsqueeze(-1),
            (self.f_hold / G / 3.0).unsqueeze(-1),
            self.mu.unsqueeze(-1),
            self.veh_acc,
            self.act_buf[:, :self.dmax + 1].flatten(1)], -1)

    # ------------------------------------------------------------------ step
    def _failure_info(self):
        return {}

    def _allocate(self, T, alpha_des):
        """Per-mass units: T collective [m/s^2], alpha_des desired body angular accel [rad/s^2].
        Returns achieved collective and angular accel. PX4-like: roll/pitch first with airmode (collective is shifted
        to fit the motor range; if the span does not fit, roll/pitch are scaled), yaw last and clipped."""
        fmax = (self.tw * G / 4).unsqueeze(-1)                     # per-motor max specific thrust
        fmin = self.cfg.f_min_frac * fmax
        tau = alpha_des * self.Jm                                    # torque / mass
        rp = tau.clone()
        rp[:, 2] = 0
        f = T.unsqueeze(-1) / 4 + rp @ self.Bpinv.T
        fhi, flo = f.max(-1, keepdim=True).values, f.min(-1, keepdim=True).values
        span = fhi - flo
        fits = span <= fmax - fmin
        shift = torch.where(fhi > fmax, fmax - fhi, (fmin - flo).clamp(min=0))
        scale = ((fmax - fmin) / span.clamp(min=1e-6)).clamp(max=1.0)
        mid = 0.5 * (fhi + flo)
        f = torch.where(fits, f + shift, 0.5 * (fmax + fmin) + (f - mid) * scale)
        yw = torch.zeros_like(tau)
        yw[:, 2] = tau[:, 2]
        f = torch.minimum(torch.maximum(f + yw @ self.Bpinv.T, fmin), fmax)
        return f.sum(-1), (f @ self.B.T) / self.Jm

    def step(self, act, *, diagnostics=False):
        c = self.cfg
        act = act.clamp(-1, 1)
        if self.delay_buf_steps > 0:
            self.act_buf = torch.cat([act.unsqueeze(1), self.act_buf[:, :-1]], 1)
            applied = self.act_buf[self.ar, self.delay]
        else:
            applied = act
        T_cmd = (applied[:, 0] + 1) * 0.5 * self.tw * G
        w_cmd = applied[:, 1:] * self.rate_max
        if c.yaw_scale != 1.0:
            w_cmd = w_cmd * torch.tensor([1.0, 1.0, c.yaw_scale], device=self.dev)
        h = c.dt / c.substeps
        for _ in range(c.substeps):
            self.thrust += (T_cmd - self.thrust) * (h / self.tau_T)
            if c.motor_model:
                T_act, alpha = self._allocate(self.thrust, (w_cmd - self.w) / self.tau_w)
                self.w += alpha * h
            else:
                T_act = self.thrust
                self.w += (w_cmd - self.w) * (h / self.tau_w)
            R = quat_to_rot(self.q)
            acc = R[..., :, 2] * T_act.unsqueeze(-1) - self.kd.unsqueeze(-1) * self.v
            acc[:, 2] -= G
            self.v += acc * h
            self.p += self.v * h
            dq = torch.cat([torch.zeros(self.N, 1, device=self.dev), self.w * 0.5 * h], -1)
            self.q = self.q + quat_mul(self.q, dq)
            self.q = self.q / self.q.norm(dim=-1, keepdim=True)
        self._step_vehicle()
        self.t += 1

        pc, n, u, w, pv = self.plate_frame()
        R = quat_to_rot(self.q)
        pad = self.p - c.pad_offset * R[..., :, 2]
        rel = pad - pc
        d = (rel * n).sum(-1)
        inplane = torch.maximum((rel * u).sum(-1).abs(), (rel * w).sum(-1).abs())
        dist = rel.norm(dim=-1)
        # velocity of the PAD point (not the CoG): v + omega_world x r_pad
        r_pad = -c.pad_offset * R[..., :, 2]
        w_world = (R @ self.w.unsqueeze(-1)).squeeze(-1)
        rv = self.v + torch.cross(w_world, r_pad, dim=-1) - pv
        vn = -(rv * n).sum(-1)
        vt = (rv + vn.unsqueeze(-1) * n).norm(dim=-1)
        align = (R[..., :, 2] * n).sum(-1)
        wn = self.w.norm(dim=-1)

        contact = (self.prev_d > 0) & (d <= 0) & (inplane < c.plate_half)
        ok = (align > c.align_cos_min) & (vn < c.vn_max) & (vt < c.vt_max) & (wn < c.w_max)
        success = contact & ok
        crash_plate = contact & ~ok
        # passed through the plate region from the wrong side, or hit vehicle body / ground
        behind = (d < -0.05) & (inplane < c.plate_half + 0.2)
        ground = self.p[:, 2] < 0.15
        veh_rel = (rotz(self.veh_psi).transpose(-1, -2) @ (self.p - self.veh_p).unsqueeze(-1)).squeeze(-1)
        in_vehicle = (veh_rel[:, 0].abs() < 1.2) & (veh_rel[:, 1].abs() < 0.7) & (self.p[:, 2] < self.ph - 0.25)
        far = dist > 12.0
        tilt_bad = R[..., 2, 2] < -0.3                                  # upside down
        crash = crash_plate | ((behind | ground | in_vehicle | tilt_bad) & ~success)
        timeout = self.t >= self.max_steps
        done = success | crash | far | timeout

        # finite-hold check if motors are cut right after touchdown (scoring only, hidden from policy)
        held = hold_margin(self.tilt, self.mu, self.f_hold, self.veh_acc, n, torch.zeros_like(n)) >= 0

        # --- reward
        r = 1.0 * (self.prev_dist - dist)
        near = torch.exp(-2.0 * dist)
        r += 0.05 * near * align
        r -= 0.02 * near * (vn.clamp(min=c.vn_max) - c.vn_max + vt)
        r -= 0.01 * ((act - self.prev_act) ** 2).sum(-1) + 0.002 * (self.w ** 2).sum(-1)
        r -= 0.005
        # graded touchdown quality on any plate contact (gives gradient towards the envelope)
        q_al = ((align - 0.5) / 0.5).clamp(0, 1)
        q_vn = (1 - vn / (2 * c.vn_max)).clamp(0, 1)
        q_vt = (1 - vt / (2 * c.vt_max)).clamp(0, 1)
        r += contact.float() * (8.0 * q_al + 4.0 * q_vn + 4.0 * q_vt)
        r += success.float() * 20.0
        r -= crash.float() * 15.0 + far.float() * 15.0
        r -= (timeout & ~success & ~crash).float() * c.timeout_penalty

        self.prev_dist, self.prev_d, self.prev_act = dist, d, act
        info = {"success": success, "crash": crash, "timeout": timeout | far, "held": success & held,
                "vn": vn, "vt": vt, "align": align,
                "c_plate": crash_plate, "c_ground": ground & ~success, "c_behind": behind & ~contact,
                "c_vehicle": in_vehicle, "c_flip": tilt_bad}
        if diagnostics:
            info.update(c_plate_edge=behind & (inplane >= c.plate_half), body_rate=wn,
                        **self._failure_info())
        done_idx = done.nonzero(as_tuple=False).squeeze(-1)
        self.reset_idx(done_idx)
        return self.observe(), r, done, info

    def _step_vehicle(self):
        c = self.cfg
        dt = c.dt
        # piecewise random target speed, first-order tracking, bounded accel
        resample = torch.rand(self.N, device=self.dev, generator=self.gen) < dt / 2.0
        self.veh_s_tgt = torch.where(resample, self._u(self.N, *c.veh_speed), self.veh_s_tgt)
        ds = ((self.veh_s_tgt - self.veh_s) * 0.8).clamp(-1.5, 1.5)
        self.veh_kappa = (self.veh_kappa + 0.02 * torch.randn(self.N, device=self.dev, generator=self.gen) * math.sqrt(dt)).clamp(-0.1, 0.1)
        self.veh_s = self.veh_s + ds * dt
        dpsi = self.veh_kappa * self.veh_s
        self.veh_psi = self.veh_psi + dpsi * dt
        R = rotz(self.veh_psi)
        fwd, lat = R[..., :, 0], R[..., :, 1]
        self.veh_acc = ds.unsqueeze(-1) * fwd + (self.veh_s * dpsi).unsqueeze(-1) * lat
        self.veh_p = self.veh_p + self.veh_s.unsqueeze(-1) * fwd * dt
