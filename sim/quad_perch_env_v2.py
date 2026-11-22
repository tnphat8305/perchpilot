"""PerchPilot sim v2 (phase P2): no privileged target info in the actor observation.

Adds on top of QuadPerchEnv (v1 untouched, so v1 runs stay reproducible):
  * EKF2-like drone odometry error: random-walk biases + white noise on p, v, attitude, gyro
  * body-fixed RGB camera with a limited FOV; the plate (tag) is measured only when inside the
    FOV, within range and facing the camera, at ~33 Hz, with distance-dependent noise
  * batched constant-velocity Kalman filter for the plate centre (Mahalanobis-gated updates),
    low-pass plate normal, and a reliability indicator rho from the visual-loss duration
  * optional active-perception reward (visibility + estimation error)

Actor observation v2 = v1 layout computed from ESTIMATES (plate from the filter, drone from
noisy odometry), then the pending-action buffer if act_history, then [rho, det_visible, t_loss_norm].
act_history: layout [v1 28][hist 4*dmax][rho, det_visible, t_loss_norm]. The v1 prefix keeps its order so
v1 checkpoints warm-start (rl.ppo.load_expanded maps actor and critic blocks explicitly).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sim.quad_perch_env import EnvCfg, QuadPerchEnv, quat_to_rot, rotz, G  # noqa: F401


@dataclass
class EnvCfgV2(EnvCfg):
    # camera (body FLU frame, optical axis = body x pitched DOWN by cam_pitch_deg)
    cam_pitch_deg: tuple = (10.0, 20.0)  # PerchRL §IV-A: 15 +- 5 deg (agy PIPE_D1 §8.4)
    cam_hfov_deg: float = 82.96          # PerchRL §IV-C2 camera
    cam_vfov_deg: float = 52.90
    cam_range: float = 6.0
    cam_view_max_deg: float = 70.0       # max angle between plate normal and the line of sight
    meas_every: tuple = (2, 5)           # policy steps between camera frames (20-50 Hz; PerchRL f_det 20-60 Hz)
    meas_pos_sigma: tuple = (0.01, 0.01)  # sigma = a + b * distance [m]
    meas_normal_sigma_deg: float = 2.0
    # odometry error (EKF2-like), scaled by odo_scale for sensitivity studies
    odo_scale: float = 1.0
    p_white: float = 0.02
    v_white: float = 0.05
    att_white_deg: float = 0.3
    gyro_white: float = 0.02
    p_rw: float = 0.02                   # m / sqrt(s)
    v_rw: float = 0.03
    att_rw_deg: float = 0.3              # deg / sqrt(s)
    p_bias_max: float = 0.3
    v_bias_max: float = 0.2
    att_bias_max_deg: float = 3.0
    # filter
    kf_acc_sigma: float = 2.0            # plate acceleration process noise [m/s^2]
    kf_gate_chi2: float = 16.3           # 3 dof, p = 0.999
    rho_lambda: float = 2.0              # PerchRL Eq. 12 form; value 2.0 s^-1 is agy's proposal (not in paper)
    t_loss_max: float = 2.0
    aug_inputs: bool = True              # False = ablation "no input augmentation": rho=1, visible=1, t_loss=0
    # active-perception reward, PerchRL Eq. 13a-d (0 = off). Values from agy PIPE_D1 §8.3 (SUY RA):
    # alpha 2.0, tau 0.5 s, beta 0.05, eta 0.02
    perc_alpha: float = 2.0
    perc_tau: float = 0.5
    perc_beta: float = 0.0
    perc_eta: float = 0.0


def skew_exp_small(theta):
    """First-order rotation matrix for small rotation vectors (N,3) -> (N,3,3)."""
    N = theta.shape[0]
    x, y, z = theta.unbind(-1)
    o = torch.zeros_like(x)
    K = torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).view(N, 3, 3)
    I = torch.eye(3, device=theta.device).expand(N, 3, 3)
    return I + K


class QuadPerchEnvV2(QuadPerchEnv):
    def __init__(self, cfg: EnvCfgV2, device="cuda", seed=0):
        N, dev = cfg.num_envs, torch.device(device)
        f = lambda *s: torch.zeros(*s, device=dev)
        # estimator / noise state must exist before the base __init__ calls reset_idx
        self.kf_p, self.kf_v = f(N, 3), f(N, 3)
        self.kf_P = f(N, 2, 2)
        self.n_hat = f(N, 3)
        self.t_loss = f(N)
        self.t_since_meas = f(N)              # time since the last ACCEPTED detector update
        self.visible = torch.zeros(N, dtype=torch.bool, device=dev)       # geometric (scoring/info only)
        self.det_visible = torch.zeros(N, dtype=torch.bool, device=dev)   # detector-event based (actor input)
        self.b_p, self.b_v, self.b_att = f(N, 3), f(N, 3), f(N, 3)
        self._r_perc = f(N)
        self._est_err = f(N)
        self.cam_c, self.cam_s = f(N), f(N)
        self.meas_k = torch.ones(N, dtype=torch.long, device=dev)
        super().__init__(cfg, device=device, seed=seed)
        self.obs_dim = self.obs_dim + 3
        self.priv_dim = self.priv_dim + 9

    # ------------------------------------------------------------------ helpers
    def _randn(self, *shape):
        return torch.randn(*shape, device=self.dev, generator=self.gen)

    def _plate_center(self):
        pc, n, u, w, pv = self.plate_frame()
        return pc, n, pv

    def _visibility(self, p, R, pc, n):
        c = self.cfg
        cc, cs = self.cam_c.unsqueeze(-1), self.cam_s.unsqueeze(-1)
        axis = cc * R[..., :, 0] - cs * R[..., :, 2]
        upc = cs * R[..., :, 0] + cc * R[..., :, 2]
        d = pc - p
        dist = d.norm(dim=-1).clamp(min=1e-6)
        du = d / dist.unsqueeze(-1)
        a = (du * axis).sum(-1)
        b = (du * R[..., :, 1]).sum(-1)
        e = (du * upc).sum(-1)
        ok = a > 0
        ok &= torch.atan2(b.abs(), a) < math.radians(c.cam_hfov_deg / 2)
        ok &= torch.atan2(e.abs(), a) < math.radians(c.cam_vfov_deg / 2)
        ok &= dist < c.cam_range
        ok &= (-du * n).sum(-1) > math.cos(math.radians(c.cam_view_max_deg))
        return ok, dist, a

    def _odo(self):
        """Noisy drone state as an EKF would report it."""
        c, s = self.cfg, self.cfg.odo_scale
        N = self.N
        p = self.p + self.b_p + s * c.p_white * self._randn(N, 3)
        v = self.v + self.b_v + s * c.v_white * self._randn(N, 3)
        dth = self.b_att + s * math.radians(c.att_white_deg) * self._randn(N, 3)
        R = quat_to_rot(self.q) @ skew_exp_small(dth)
        w = self.w + s * c.gyro_white * self._randn(N, 3)
        return p, v, R, w

    # ------------------------------------------------------------------ reset
    def reset_idx(self, idx):
        super().reset_idx(idx)
        if idx.numel() == 0:
            return
        c, n = self.cfg, idx.numel()
        cp = torch.deg2rad(self._u(n, *c.cam_pitch_deg))
        self.cam_c[idx], self.cam_s[idx] = torch.cos(cp), torch.sin(cp)
        self.meas_k[idx] = torch.randint(c.meas_every[0], c.meas_every[1] + 1, (n,), device=self.dev, generator=self.gen)
        s = c.odo_scale
        am = math.radians(c.att_bias_max_deg)
        self.b_p[idx] = (s * 0.3 * c.p_bias_max * self._randn(n, 3)).clamp(-s * c.p_bias_max, s * c.p_bias_max)
        self.b_v[idx] = (s * 0.3 * c.v_bias_max * self._randn(n, 3)).clamp(-s * c.v_bias_max, s * c.v_bias_max)
        self.b_att[idx] = (s * 0.3 * am * self._randn(n, 3)).clamp(-s * am, s * am)
        pc, nn_, pv = self._plate_center()
        # Episodes start mid-APPROACH, where the deployed stack has been tracking the tag (see ros/perch_node.py),
        # so the filter starts CONVERGED-but-imperfect: position from the same odometry+relative-measurement path
        # as at run time, velocity with a 0.3 m/s error prior. Visibility is NOT forced: it is computed geometrically.
        R = quat_to_rot(self.q)
        p_o, _, R_o, _ = self._odo()
        vis, dist, _ = self._visibility(self.p, R, pc, nn_)
        sig = c.meas_pos_sigma[0] + c.meas_pos_sigma[1] * dist
        rel_body = (R.transpose(-1, -2) @ (pc - self.p).unsqueeze(-1)).squeeze(-1) + sig.unsqueeze(-1) * self._randn(self.N, 3)
        z = p_o + (R_o @ rel_body.unsqueeze(-1)).squeeze(-1)
        self.kf_p[idx] = z[idx]
        self.kf_v[idx] = pv[idx] + 0.3 * self._randn(n, 3)
        P = torch.zeros(n, 2, 2, device=self.dev)
        P[:, 0, 0], P[:, 1, 1] = sig[idx] ** 2 + 0.02 ** 2, 0.3 ** 2
        self.kf_P[idx] = P
        nh = nn_[idx] + math.radians(c.meas_normal_sigma_deg) * self._randn(n, 3)
        self.n_hat[idx] = nh / nh.norm(dim=-1, keepdim=True)
        v0 = vis[idx]
        # not visible at the start = target was lost recently (0.3 s) during approach
        self.t_since_meas[idx] = torch.where(v0, torch.zeros(n, device=self.dev), torch.full((n,), 0.3, device=self.dev))
        self.visible[idx] = v0
        period = self.meas_k[idx].float() * c.dt
        self.det_visible[idx] = self.t_since_meas[idx] <= period + 1e-6
        self.t_loss[idx] = (self.t_since_meas[idx] - period).clamp(min=0)

    # ------------------------------------------------------------------ estimator step (hooked after physics)
    def _step_vehicle(self):
        super()._step_vehicle()
        c, N, dt = self.cfg, self.N, self.cfg.dt
        s = c.odo_scale
        sq = math.sqrt(dt)
        lim = lambda x, m: x.clamp(-m, m)
        self.b_p = lim(self.b_p + s * c.p_rw * sq * self._randn(N, 3), s * c.p_bias_max + 1e-9)
        self.b_v = lim(self.b_v + s * c.v_rw * sq * self._randn(N, 3), s * c.v_bias_max + 1e-9)
        am = math.radians(c.att_bias_max_deg)
        self.b_att = lim(self.b_att + s * math.radians(c.att_rw_deg) * sq * self._randn(N, 3), s * am + 1e-9)
        # KF predict (per axis, shared 2x2 covariance)
        self.kf_p = self.kf_p + self.kf_v * dt
        F = torch.tensor([[1.0, dt], [0.0, 1.0]], device=self.dev)
        q = c.kf_acc_sigma ** 2
        Q = q * torch.tensor([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]], device=self.dev)
        self.kf_P = F @ self.kf_P @ F.T + Q
        # camera measurement
        R = quat_to_rot(self.q)
        pc, n, pv = self._plate_center()
        vis, dist, _ = self._visibility(self.p, R, pc, n)
        frame = (self.t % self.meas_k) == 0
        meas = vis & frame
        p_o, _, R_o, _ = self._odo()
        sig = c.meas_pos_sigma[0] + c.meas_pos_sigma[1] * dist
        rel_body = (R.transpose(-1, -2) @ (pc - self.p).unsqueeze(-1)).squeeze(-1) + sig.unsqueeze(-1) * self._randn(N, 3)
        z = p_o + (R_o @ rel_body.unsqueeze(-1)).squeeze(-1)
        S = self.kf_P[:, 0, 0] + sig ** 2
        r = z - self.kf_p
        d2 = (r ** 2).sum(-1) / S
        upd = meas & (d2 < c.kf_gate_chi2)
        K = self.kf_P[:, :, 0] / S.unsqueeze(-1)                       # (N,2)
        um = upd.float()
        self.kf_p = self.kf_p + (um * K[:, 0]).unsqueeze(-1) * r
        self.kf_v = self.kf_v + (um * K[:, 1]).unsqueeze(-1) * r
        I_KH = torch.eye(2, device=self.dev).expand(N, 2, 2).clone()
        I_KH[:, 0, 0] -= K[:, 0]
        I_KH[:, 1, 0] -= K[:, 1]
        newP = I_KH @ self.kf_P
        self.kf_P = torch.where(upd.view(N, 1, 1), newP, self.kf_P)
        nm = n + math.radians(c.meas_normal_sigma_deg) * self._randn(N, 3)
        nm = nm / nm.norm(dim=-1, keepdim=True)
        nh = torch.where(upd.unsqueeze(-1), 0.7 * self.n_hat + 0.3 * nm, self.n_hat)
        self.n_hat = nh / nh.norm(dim=-1, keepdim=True)
        # Actor-facing visibility comes from DETECTOR EVENTS (accepted updates), not from geometry between frames:
        # visual loss = no accepted measurement for longer than one camera period (PerchRL §III-C2, Eq. 12).
        self.t_since_meas = torch.where(upd, torch.zeros_like(self.t_since_meas), self.t_since_meas + dt)
        period = self.meas_k.float() * dt
        self.det_visible = self.t_since_meas <= period + 1e-6
        self.t_loss = (self.t_since_meas - period).clamp(min=0, max=c.t_loss_max)
        self.visible = vis                      # geometric, for metrics only
        err = (self.kf_p - pc).norm(dim=-1)
        self._est_err = err
        _, _, cosax = self._visibility(self.p, R, pc, n)          # v_cam . d_target
        g = 1.0 + c.perc_alpha * (self.t_loss / c.perc_tau).clamp(0, 1)     # Eq. 13a
        self._r_perc = -c.perc_beta * g * err - c.perc_eta * (1.0 - cosax)  # Eq. 13b + 13c

    # ------------------------------------------------------------------ observation from estimates
    def rho(self):
        """PerchRL Eq. 12: 1 when visible, exp(-lambda * T_loss) during visual loss."""
        return torch.where(self.det_visible, torch.ones_like(self.t_loss), torch.exp(-self.cfg.rho_lambda * self.t_loss))

    def observe(self):
        c = self.cfg
        p, v, R, w = self._odo()
        n = self.n_hat
        ez = torch.zeros_like(n)
        ez[:, 2] = 1.0
        u = torch.cross(n, ez, dim=-1)
        u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        wv = torch.cross(n, u, dim=-1)
        Pm = torch.stack([u, wv, n], -2)
        pad = p - c.pad_offset * R[..., :, 2]
        rp = (Pm @ (pad - self.kf_p).unsqueeze(-1)).squeeze(-1)
        rv = (Pm @ (v - self.kf_v).unsqueeze(-1)).squeeze(-1)
        Rp = Pm @ R
        gp = Pm[..., :, 2]
        tilt_hat = torch.acos(n[:, 2].clamp(-1, 1))
        parts = [rp, rv, Rp.flatten(1), w, gp, self.prev_act,
                 (self.t.float() / self.max_steps).unsqueeze(-1),
                 self.kf_v.norm(dim=-1, keepdim=True), tilt_hat.unsqueeze(-1)]
        if c.act_history and self.dmax > 0:
            parts.append(self.act_buf[:, 1:self.dmax + 1].flatten(1))
        if c.aug_inputs:
            parts += [self.rho().unsqueeze(-1), self.det_visible.float().unsqueeze(-1),
                      (self.t_loss / c.t_loss_max).unsqueeze(-1)]
        else:
            o = torch.ones(self.N, 1, device=self.dev)
            parts += [o, o, torch.zeros_like(o)]
        return torch.cat(parts, -1)

    def privileged(self):
        pc, n, u, w, pv = self.plate_frame()
        Pm = torch.stack([u, w, n], -2)
        pad = self._pad_pos()
        rp = (Pm @ (pad - pc).unsqueeze(-1)).squeeze(-1)
        rv = (Pm @ (self.v - pv).unsqueeze(-1)).squeeze(-1)
        e = (Pm @ (self.kf_p - pc).unsqueeze(-1)).squeeze(-1)
        return torch.cat([super().privileged(), rp, rv, e], -1)

    def _failure_info(self):
        # Uncapped loss duration before reset_idx overwrites the terminal slots.
        return {"t_loss": (self.t_since_meas - self.meas_k.float() * self.cfg.dt).clamp(min=0)}

    def step(self, act, *, diagnostics=False):
        obs, r, done, info = super().step(act, diagnostics=diagnostics)
        r = r + self._r_perc
        info["visible"] = self.visible
        info["est_err"] = self._est_err
        return obs, r, done, info
