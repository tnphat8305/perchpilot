"""PerchPilot Q2 -- batched simulator of the POST-TOUCHDOWN contact phase (200 Hz), design v2.

Vectorised over N envs with torch (CPU or CUDA). It starts at the touchdown
instant (k = 0) and simulates the pad stuck on the tilted plate carried by the
moving cart while a controller issues discrete macro actions.

Geometry that drives the design: the drone perches belly-first, z_body ~ plate
normal n, so thrust T*z_body PULLS the pad away from the plate; gravity presses
it in with g*cos(theta) and drags it down-slope with g*sin(theta).

Macro actions (v2; v1 PRESS/PROBE dropped because thrust cannot press):
    RELAX        thrust_cmd -> idle (0.1 hover) over 50 ms, hold to 0.15 s: gravity loads the pad,
                 motors keep spinning; observe slip under (almost) static load
    PULL(k)      load test, k in {0.3, 0.6, 0.9}: ramp to k*hover in 50 ms, hold 150 ms,
                 ramp back to idle in 50 ms (0.25 s). Normal pull tests f_hold, the reduced
                 normal load tests mu; surviving PULL(k) = spare margin for later cart accelerations
    MOTORS_OFF   terminal: thrust_cmd = 0, outcome = 10 s REST under the env's own future cart motion
    ABORT        terminal: counted as abort (no fall)
Default schedule RELAX -> PULL0.3 -> PULL0.6 -> PULL0.9, one decision point after each macro.

Touchdown states are sampled PARAMETRICALLY inside the Q1 success envelope
(docs/DAC_TA_PerchPilot.md 3.1). Later they are to be replaced by the
touchdown states of rollouts of the trained Q1 policy (same fields: vn, vt,
misalignment, body rate, tilt, cart speed).

Physics (per unit mass, docs/CONTRACTS.md C2 -- the holding decision ALWAYS
comes from `hold_margin`, imported from quad_perch_env, never re-derived):
  * frame: cart heading frame (z up). hold_margin is invariant to a yaw rotation,
    so this equals the world-frame computation of the env.
  * extra_acc = cart acceleration + touchdown impact transient
        a_imp(t) = (vn * n - vt * t_hat) / tau_imp * exp(-t / tau_imp)
    (relative velocity killed with tau_imp = 20 ms -- ASSUMPTION).
  * thrust_acc = T * z_body (z_body tilted from n by the misalignment); T follows
    T_cmd through the first-order motor lag tau_T (same range as the Q1 env).
  * normal_ok < 0 -> detach; margin < 0 -> slip along -f_t with speed
    slip_gain * |margin| * m (C2); pad leaving the square plate (half side 0.22 m
    from the centre) -> off the plate.
  * Loss of contact (detach or off the plate) while the motors are still running
    is NOT a fall: status SELF_ABORT (the drone has thrust and flies off; it may
    re-perch). cfg.powered_detach_is_fall=True restores the v1 behaviour (FELL).
    A FALL only happens after MOTORS_OFF (in the REST).
  * rigid pad: attitude relative to the plate fixed while attached -- ASSUMPTION.

Slip does not feed back into the margin, so a REST with motors off from step k0
is evaluated in closed form, batched over the whole 10 s horizon (`rest_fork`);
the step simulator and the fork share the force routine and the discrete thrust
decay, so the closed-loop MOTORS_OFF outcome equals the fork label.

Sensors (C5, 200 Hz, 11 channels, see SENSOR_CHANNELS): acc_body(3), gyro_body(3),
f_normal_meas(1) [N, SIGNED: + compression, - tension], contact(1), thrust_cmd01(1),
phase_id(1), probe_level(1). C5 v2 appends optical slip_disp (2, mm per step).
Labels (training/scoring only): held_star (on the plate for 10 s and total slip < 2 cm),
held_strict (margin >= 0 at every step), fell_if_off, margin_min, slip_now.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sim.quad_perch_env import G, hold_margin

# ---------------------------------------------------------------- constants
RELAX, PULL03, PULL06, PULL09, MOTORS_OFF, ABORT = range(6)
ACTION_NAMES = ["RELAX", "PULL0.3", "PULL0.6", "PULL0.9", "MOTORS_OFF", "ABORT"]
PULL_LEVEL = {PULL03: 0.3, PULL06: 0.6, PULL09: 0.9}
PH_APPROACH, PH_RELAX, PH_PULL, PH_OFF, PH_ABORT = range(5)
RUNNING, FELL, OFF, ABORTED, SELF_ABORT = range(5)
STATUS_NAMES = ["RUNNING", "FELL", "OFF", "ABORTED", "SELF_ABORT"]
SENSOR_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                   "f_normal_N", "contact", "thrust_cmd01", "phase_id", "probe_level"]
N_CH = len(SENSOR_CHANNELS)
SLIP_CHANNELS = ["slip_disp_u_mm", "slip_disp_w_mm"]
SCRIPT = [RELAX, PULL03, PULL06, PULL09]            # default probing schedule
BIG_MU = 1.0e6                                      # hold_margin(mu=BIG) == normal_ok
NOMINAL_TW = 2.26


@dataclass
class ContactCfg:
    num_envs: int = 1024
    dt: float = 0.005                 # 200 Hz
    pre_s: float = 0.4                # free-flight history before touchdown (fills the first windows)
    max_contact_s: float = 1.2        # longest powered contact phase covered by the cart series
    t_rest: float = 10.0              # REST horizon for the label / MOTORS_OFF outcome
    held_slip_max: float = 0.02       # held_star tolerates < 2 cm total slip during the REST
    mass: float = 2.0                 # kg, PX4 x500-like (ASSUMPTION)
    plate_half: float = 0.22
    touch_pos_max: float = 0.15       # touchdown point uniform in +-this (plate coords, from centre)
    # touchdown envelope (sampled parametrically, inside the Q1 success envelope)
    vn: tuple = (0.2, 1.5)
    vt: tuple = (0.0, 1.0)
    misalign_deg: tuple = (0.0, 25.0)
    w0_max: float = 5.0               # body-rate norm at touchdown [rad/s]
    yaw_rel: tuple = (-0.3, 0.3)      # body heading about z_body vs. up-slope axis [rad]
    plate_tilt_deg: tuple = (55.0, 70.0)
    veh_speed: tuple = (0.0, 3.0)
    tw_ratio: tuple = (2.6, 3.4)
    legacy_thrust: bool = False     # reproduce the v2/v3 command and sensor channel
    tau_thrust: tuple = (0.02, 0.05)
    thrust_pre: tuple = (0.6, 1.2)    # actual thrust at touchdown, x hover
    # hidden material params: union of intervals, sampled proportionally to width
    mu: tuple = ((0.25, 1.0),)
    f_hold_mg: tuple = ((0.6, 3.0),)
    # contact dynamics (ASSUMPTIONS)
    tau_imp: float = 0.02             # impact transient time constant [s]
    tau_w_imp: float = 0.03           # body-rate decay after touchdown [s]
    slip_gain: float = 0.05           # (m/s)/N, C2 default
    powered_detach_is_fall: bool = False   # v2: loss of contact with motors on = SELF_ABORT
    # macro timing
    idle_frac: float = 0.1            # idle thrust, x hover (motors spinning)
    relax_s: float = 0.15
    ramp_s: float = 0.05
    pull_hold_s: float = 0.15
    pull_return_s: float = 0.05
    # sensors (ICM-42688-like + vibration ASSUMPTIONS)
    acc_sigma: float = 0.007          # 70 ug/sqrt(Hz) * sqrt(100 Hz)
    acc_vib_motor: float = 0.4        # m/s^2 rms at hover thrust (motor/prop vibration)
    acc_vib_slip: float = 2.0         # m/s^2 rms per m/s of slip speed (stick-slip chatter, v1 used 10)
    acc_bias: float = 0.05            # per-episode bias sigma [m/s^2]
    gyro_sigma: float = 0.0005        # 2.8 mdps/sqrt(Hz) * sqrt(100 Hz)
    gyro_vib_motor: float = 0.02
    gyro_vib_slip: float = 0.1        # rad/s rms per m/s of slip speed (v1 used 0.5)
    gyro_bias: float = 0.005
    vib_scale: float = 1.0           # scale slip-induced vibration, keep motor vibration
    slip_acc_in_imu: bool = True     # include the slip-velocity derivative in measured acceleration
    fn_full_scale_mg: float = 2.0     # load cell range +-FS, FS = this * m * g (saturates)
    fn_noise_fs: float = 0.03         # noise sigma = 3% FS
    fn_delay_s: float = 0.01
    contact_dropout: float = 0.005
    sensor_dr: bool = False          # v5: episode-level sensor variation, independent of physics
    slip_sensor: bool = False        # C5 v2: optical slip displacement under the pad
    imu_hold: bool = False           # v7: per episode p=0.5, IMU (acc, gyro) sample-and-hold at 100 Hz (PX4 SITL
                                     # publishes sensor_combined at 100 Hz, ĐO 2026-11-20); applied in windows() only

    @property
    def sensor_channels(self):
        return SENSOR_CHANNELS + (SLIP_CHANNELS if self.slip_sensor else [])


def sample_union(n, intervals, gen, dev):
    """Uniform sample over a union of disjoint intervals ((lo, hi), ...)."""
    iv = torch.tensor(intervals, dtype=torch.float32, device=dev).view(-1, 2)
    w = iv[:, 1] - iv[:, 0]
    pick = torch.multinomial(w / w.sum(), n, replacement=True, generator=gen)
    r = torch.rand(n, device=dev, generator=gen)
    return iv[pick, 0] + r * w[pick]


def vehicle_series(n, steps, dt, speed, gen, dev):
    """Cart acceleration process, a copy of QuadPerchEnv._step_vehicle with a dt argument.

    Expressed in the cart heading frame. Returns (n, steps, 3): [acc_fwd, acc_lat, yaw_rate].
    """
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev, generator=gen)
    s, s_tgt, kappa = u(*speed), u(*speed), u(-0.08, 0.08)
    out = torch.empty(n, steps, 3, device=dev)
    for i in range(steps):
        resample = torch.rand(n, device=dev, generator=gen) < dt / 2.0
        s_tgt = torch.where(resample, u(*speed), s_tgt)
        ds = ((s_tgt - s) * 0.8).clamp(-1.5, 1.5)
        kappa = (kappa + 0.02 * torch.randn(n, device=dev, generator=gen) * math.sqrt(dt)).clamp(-0.1, 0.1)
        s = s + ds * dt
        dpsi = kappa * s
        out[:, i, 0] = ds
        out[:, i, 1] = s * dpsi
        out[:, i, 2] = dpsi
    return out


class ContactPhaseSim:
    def __init__(self, cfg: ContactCfg, device="cpu", seed=0):
        self.cfg, self.dev = cfg, torch.device(device)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        c, N, dev = cfg, cfg.num_envs, self.dev
        self.N = N
        self.dt = c.dt
        r = lambda s: int(round(s / c.dt))
        self.pre, self.kmax, self.H = r(c.pre_s), r(c.max_contact_s), r(c.t_rest)
        self.ramp_n, self.relax_n = r(c.ramp_s), r(c.relax_s)
        self.hold_n, self.ret_n = r(c.pull_hold_s), r(c.pull_return_s)
        self.pull_n = self.ramp_n + self.hold_n + self.ret_n
        self.fn_delay = r(c.fn_delay_s)
        if c.slip_sensor:
            if c.dt != 0.005:
                raise ValueError("C5 v2 slip_sensor requires 5 ms steps")
            self.slip_gen = torch.Generator(device=dev).manual_seed(seed ^ 0xC502)
            self.slip_scale = 0.9 + 0.2 * torch.rand(N, 1, device=dev, generator=self.slip_gen)
            # Rigid contact models zero pad gap until contact is lost. The ring
            # starts with zero pre-touch displacement and delays it by 10 ms.
            self.slip_pending = torch.zeros(N, 2, 2, device=dev)
        self.idle = c.idle_frac * G
        self.fs_n = c.fn_full_scale_mg * c.mass * G
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(N, device=dev, generator=self.gen)

        # --- hidden material + vehicle params
        self.mu = sample_union(N, c.mu, self.gen, dev)
        self.f_hold = sample_union(N, c.f_hold_mg, self.gen, dev) * G
        self.tilt = torch.deg2rad(u(*c.plate_tilt_deg))
        self.tw = u(*c.tw_ratio)
        self.command_max = (self.tw if c.legacy_thrust else NOMINAL_TW) * G
        self.motor_gain = 1.0 if c.legacy_thrust else self.tw / NOMINAL_TW
        self.tau_T = u(*c.tau_thrust)
        # --- plate frame in cart heading frame: n outward normal, e_up up-slope, e_lat lateral
        st, ct = torch.sin(self.tilt), torch.cos(self.tilt)
        z0 = torch.zeros_like(st)
        self.n = torch.stack([-st, z0, ct], -1)
        self.e_up = torch.stack([ct, z0, st], -1)
        self.e_lat = torch.stack([z0, torch.ones_like(st), z0], -1)
        # --- touchdown state
        self.vn = u(*c.vn)
        self.vt = u(*c.vt)
        phi = u(-math.pi, math.pi)
        self.t_hat = torch.cos(phi)[:, None] * self.e_up + torch.sin(phi)[:, None] * self.e_lat
        al = torch.deg2rad(u(*c.misalign_deg))
        be = u(-math.pi, math.pi)
        zb = torch.cos(al)[:, None] * self.n + torch.sin(al)[:, None] * (
            torch.cos(be)[:, None] * self.e_up + torch.sin(be)[:, None] * self.e_lat)
        x0 = self.e_up - (self.e_up * zb).sum(-1, keepdim=True) * zb
        x0 = x0 / x0.norm(dim=-1, keepdim=True)
        y0 = torch.cross(zb, x0, dim=-1)
        ga = u(*c.yaw_rel)
        xb = torch.cos(ga)[:, None] * x0 + torch.sin(ga)[:, None] * y0
        yb = torch.cross(zb, xb, dim=-1)
        self.R = torch.stack([xb, yb, zb], -1)                 # columns = body axes in cart frame
        self.misalign = al
        wdir = torch.randn(N, 3, device=dev, generator=self.gen)
        self.w0 = wdir / wdir.norm(dim=-1, keepdim=True) * u(0.0, c.w0_max)[:, None]
        self.T_pre = u(*c.thrust_pre) * G
        # --- cart acceleration series covering approach history + contact + REST
        L = self.pre + self.kmax + self.H + 2
        self.veh = vehicle_series(N, L, c.dt, c.veh_speed, self.gen, dev)
        # --- sensor biases (per episode)
        self.acc_b = c.acc_bias * torch.randn(N, 3, device=dev, generator=self.gen)
        self.gyro_b = c.gyro_bias * torch.randn(N, 3, device=dev, generator=self.gen)
        if c.sensor_dr:
            sensor_gen = torch.Generator(device=dev).manual_seed(seed ^ 0x513D5)

            def sensor_uniform(lo, hi):
                return lo + (hi - lo) * torch.rand(N, device=dev, generator=sensor_gen)

            self.fn_offset = sensor_uniform(-8.0, 3.0)
            self.fn_scale = sensor_uniform(0.6, 1.2)
            self.fn_supported = sensor_uniform(0.0, 0.6)
            self.slip_vib_scale = sensor_uniform(0.0, 1.5)
            self.slip_acc_enabled = sensor_uniform(0.0, 1.0) < 0.5
            self.acc_noise_scale = sensor_uniform(0.5, 2.0)
            self.gyro_noise_scale = sensor_uniform(0.5, 2.0)
            self.motor_vib_enabled = sensor_uniform(0.0, 1.0) < 0.5
        if c.imu_hold:
            hold_gen = torch.Generator(device=dev).manual_seed(seed ^ 0x1A0D)
            self.imu_hold_on = torch.rand(N, device=dev, generator=hold_gen) < 0.5
            self.imu_hold_phase = torch.randint(0, 2, (N,), device=dev, generator=hold_gen)
        # --- dynamic state
        self.k = torch.zeros(N, dtype=torch.long, device=dev)
        self.thrust = self.T_pre * self.motor_gain
        self.T_cmd = self.T_pre.clone()
        self.pos = (2 * torch.rand(N, 2, device=dev, generator=self.gen) - 1) * c.touch_pos_max  # (lat, up) on plate
        self.v_slip = torch.zeros(N, 2, device=dev)
        self.status = torch.full((N,), RUNNING, dtype=torch.long, device=dev)
        self.loss_cause = torch.zeros(N, dtype=torch.long, device=dev)      # 1 detach, 2 slid off (powered)
        self.loss_k = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.action = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.act_t = torch.zeros(N, dtype=torch.long, device=dev)
        self.act_T0 = self.T_pre.clone()
        self.phase = torch.full((N,), PH_APPROACH, dtype=torch.long, device=dev)
        self.level = torch.ones(N, device=dev)
        self.n_actions = torch.zeros(N, dtype=torch.long, device=dev)
        # terminal outcome (MOTORS_OFF)
        self.off_k = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.off_held = torch.zeros(N, dtype=torch.bool, device=dev)
        self.off_held_strict = torch.zeros(N, dtype=torch.bool, device=dev)
        self.off_fell = torch.zeros(N, dtype=torch.bool, device=dev)
        self.off_margin_min = torch.zeros(N, device=dev)
        # histories (sample index = pre + k)
        Lh = self.pre + self.kmax + 1
        self.hist = torch.zeros(N, Lh, len(c.sensor_channels), device=dev)
        self.fn_true = torch.zeros(N, Lh, device=dev)
        self.margin_now = torch.zeros(N, device=dev)
        self.slip_now = torch.zeros(N, dtype=torch.bool, device=dev)
        self.ar = torch.arange(N, device=dev)
        self._fill_pre_history()
        self._contact_sample(self.ar, torch.zeros(N, dtype=torch.long, device=dev))
        if c.slip_sensor:
            self.hist[:, self.pre, N_CH:] = self._optical_reading(self.slip_pending[:, 0])

    # ------------------------------------------------------------ physics core
    def forces(self, env, k, thrust):
        """Contact forces for flat index tensors env (S,), k (S,), thrust (S,).

        Returns margin, normal_ok, fn (signed f_c.n, + = compression), slip_dir (S, 2)
        in plate (lat, up) coordinates, spec (S, 3) = a_drone - g in the cart frame.
        """
        c = self.cfg
        va = self.veh[env, self.pre + k]
        acc = torch.stack([va[:, 0], va[:, 1], torch.zeros_like(va[:, 0])], -1)
        t = k.float() * c.dt
        dec = (torch.exp(-t / c.tau_imp) / c.tau_imp).unsqueeze(-1)
        n = self.n[env]
        extra = acc + (self.vn[env, None] * n - self.vt[env, None] * self.t_hat[env]) * dec
        thrust_acc = thrust.unsqueeze(-1) * self.R[env, :, 2]
        tilt, f_hold = self.tilt[env], self.f_hold[env]
        margin = hold_margin(tilt, self.mu[env], f_hold, extra, n, thrust_acc)
        normal_ok = hold_margin(tilt, torch.full_like(f_hold, BIG_MU), f_hold, extra, n, thrust_acc)
        fn = normal_ok - f_hold
        # direction only (sensor/kinematics helper): tangential part of the C2 contact force
        f_c = extra - thrust_acc
        f_c[:, 2] += G
        ft = f_c - fn.unsqueeze(-1) * n
        d = -torch.stack([(ft * self.e_lat[env]).sum(-1), (ft * self.e_up[env]).sum(-1)], -1)
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        spec = extra.clone()
        spec[:, 2] += G
        return {"margin": margin, "normal_ok": normal_ok, "fn": fn, "slip_dir": d, "spec": spec,
                "yaw_rate": va[:, 2]}

    def slip_speed(self, margin, normal_ok):
        c = self.cfg
        return c.slip_gain * c.mass * (-margin).clamp(min=0.0) * (normal_ok >= 0).float()

    # ------------------------------------------------------------ sensors
    def _noise(self, *shape):
        return torch.randn(*shape, device=self.dev, generator=self.gen)

    def _load_cell(self, force, env):
        return (force - self.fn_supported[env] * force.clamp_min(0)) * self.fn_scale[env] + self.fn_offset[env]

    def _optical_reading(self, disp_mm):
        noise = 0.02 * torch.randn(disp_mm.shape, device=self.dev, generator=self.slip_gen)
        drop = torch.rand(disp_mm.shape[:-1], device=self.dev, generator=self.slip_gen) < 0.02
        return ((disp_mm + noise) * 100).round().div(100).masked_fill(drop[..., None], 0)

    def _fill_pre_history(self):
        """Free flight before touchdown: specific force = (0, 0, T) in body, gyro = touchdown rate."""
        c, N, P = self.cfg, self.N, self.pre
        Tn = (self.thrust / G)[:, None, None]
        acc = torch.zeros(N, P, 3, device=self.dev)
        acc[..., 2] = self.thrust[:, None]
        acc_sigma, gyro_sigma = c.acc_sigma, c.gyro_sigma
        if c.sensor_dr:
            acc_sigma = acc_sigma * self.acc_noise_scale[:, None, None]
            gyro_sigma = gyro_sigma * self.gyro_noise_scale[:, None, None]
            Tn = Tn * self.motor_vib_enabled[:, None, None]
        acc = acc + self.acc_b[:, None] + self._noise(N, P, 3) * (acc_sigma + c.acc_vib_motor * Tn)
        gyro = self.w0[:, None] + self.gyro_b[:, None] + self._noise(N, P, 3) * (gyro_sigma + c.gyro_vib_motor * Tn)
        fnm = c.fn_noise_fs * self.fs_n * self._noise(N, P)
        if c.sensor_dr:
            fnm = fnm + self.fn_offset[:, None]
        fnm = fnm.clamp(-self.fs_n, self.fs_n)
        h = self.hist
        h[:, :P, 0:3] = acc
        h[:, :P, 3:6] = gyro
        h[:, :P, 6] = fnm
        h[:, :P, 7] = 0.0
        h[:, :P, 8] = (self.T_pre / self.command_max)[:, None]
        h[:, :P, 9] = PH_APPROACH
        h[:, :P, 10] = 1.0
        if c.slip_sensor:
            h[:, :P, N_CH:] = self._optical_reading(h[:, :P, N_CH:])

    def _contact_sample(self, env, k, a_slip=None, slip_speed=None, attached=None, f=None):
        """Write the 11-channel sensor sample at step k (contact phase) for env indices."""
        c = self.cfg
        if f is None:
            f = self.forces(env, k, self.thrust[env])
        nE = env.numel()
        Rt = self.R[env].transpose(-1, -2)
        spec = f["spec"]
        if a_slip is not None and c.slip_acc_in_imu:
            if c.sensor_dr:
                a_slip = a_slip * self.slip_acc_enabled[env, None]
            spec = spec + a_slip
        ss = torch.zeros(nE, device=self.dev) if slip_speed is None else c.vib_scale * slip_speed
        att = torch.ones(nE, dtype=torch.bool, device=self.dev) if attached is None else attached
        Tn = (self.thrust[env] / G).unsqueeze(-1)
        acc_sigma, gyro_sigma = c.acc_sigma, c.gyro_sigma
        if c.sensor_dr:
            ss = ss * self.slip_vib_scale[env]
            acc_sigma = acc_sigma * self.acc_noise_scale[env, None]
            gyro_sigma = gyro_sigma * self.gyro_noise_scale[env, None]
            Tn = Tn * self.motor_vib_enabled[env, None]
        acc = (Rt @ spec.unsqueeze(-1)).squeeze(-1) + self.acc_b[env] + self._noise(nE, 3) * (
            acc_sigma + c.acc_vib_motor * Tn + c.acc_vib_slip * ss.unsqueeze(-1))
        t = k.float() * c.dt
        yaw = torch.zeros(nE, 3, device=self.dev)
        yaw[:, 2] = f["yaw_rate"]
        w_true = (Rt @ yaw.unsqueeze(-1)).squeeze(-1) + self.w0[env] * torch.exp(-t / c.tau_w_imp).unsqueeze(-1)
        gyro = w_true + self.gyro_b[env] + self._noise(nE, 3) * (
            gyro_sigma + c.gyro_vib_motor * Tn + c.gyro_vib_slip * ss.unsqueeze(-1))
        hi = self.pre + k
        # signed net normal contact force [N]: + compression, - tension (pad pulled by thrust)
        self.fn_true[env, hi] = c.mass * f["fn"] * att.float()
        fd = self.fn_true[env, hi - self.fn_delay]
        if c.sensor_dr:
            fd = self._load_cell(fd, env)
        fnm = (fd + c.fn_noise_fs * self.fs_n * self._noise(nE)).clamp(-self.fs_n, self.fs_n)
        drop = torch.rand(nE, device=self.dev, generator=self.gen) < c.contact_dropout
        contact = (att & ~drop).float()
        s = torch.cat([acc, gyro, fnm[:, None], contact[:, None],
                       (self.T_cmd / self.command_max)[env, None],
                       self.phase[env].float()[:, None], self.level[env][:, None]], -1)
        self.hist[env, hi, :N_CH] = s
        self.margin_now[env] = f["margin"]
        self.slip_now[env] = (f["margin"] < 0) & att

    def _optical_sample(self, disp, near_surface):
        slot = self.k % 2
        delayed = self.slip_pending[self.ar, slot]
        slide = disp[:, :1] * self.e_lat + disp[:, 1:] * self.e_up
        # Measure along the two body axes in the pad plane, in mm per sample.
        pad_mm = (slide[:, None, :] @ self.R[:, :, :2]).squeeze(1) * 1000
        self.slip_pending[self.ar, slot] = pad_mm.masked_fill(~near_surface[:, None], 0)
        meas = self._optical_reading(delayed * self.slip_scale)
        self.hist[self.ar, self.pre + self.k, N_CH:] = meas.masked_fill(~near_surface[:, None], 0)

    # ------------------------------------------------------------ control
    @property
    def ready(self):
        """Envs waiting for a new macro action (running and current macro finished)."""
        dur = torch.where(self.action == RELAX, self.relax_n, self.pull_n)
        return (self.status == RUNNING) & ((self.action < 0) | (self.act_t >= dur))

    def act(self, a):
        """a: (N,) long action ids; entries < 0 are ignored. Only applied to ready envs."""
        a = a.to(self.dev)
        m = self.ready & (a >= 0)
        if not m.any():
            return
        off = m & (a == MOTORS_OFF)
        if off.any():
            idx = off.nonzero().squeeze(-1)
            r = self.rest_fork(idx, self.k[idx], self.thrust[idx], self.pos[idx])
            self.status[idx] = OFF
            self.off_k[idx] = self.k[idx]
            self.off_held[idx], self.off_held_strict[idx] = r["held"], r["held_strict"]
            self.off_fell[idx], self.off_margin_min[idx] = r["fell"], r["margin_min"]
            self.phase[idx] = PH_OFF
        ab = m & (a == ABORT)
        self.status = torch.where(ab, torch.full_like(self.status, ABORTED), self.status)
        self.phase = torch.where(ab, torch.full_like(self.phase, PH_ABORT), self.phase)
        mv = m & (a <= PULL09)
        if mv.any():
            self.action = torch.where(mv, a, self.action)
            self.act_t = torch.where(mv, torch.zeros_like(self.act_t), self.act_t)
            self.act_T0 = torch.where(mv, self.T_cmd, self.act_T0)
            self.phase = torch.where(mv, torch.where(a == RELAX, PH_RELAX, PH_PULL), self.phase)
            lv = torch.full_like(self.level, self.cfg.idle_frac)
            for ai, k in PULL_LEVEL.items():
                lv = torch.where(a == ai, torch.full_like(lv, k), lv)
            self.level = torch.where(mv, lv, self.level)
            self.n_actions += mv.long()

    def _cmd(self):
        """Macro command in nominal acceleration units, before the physical motor gain."""
        tt = (self.act_t + 1).float()
        up = (tt / self.ramp_n).clamp(max=1.0)
        relax = self.act_T0 + (self.idle - self.act_T0) * up
        tgt = torch.full_like(self.T_cmd, self.idle)
        for ai, k in PULL_LEVEL.items():
            tgt = torch.where(self.action == ai, torch.full_like(tgt, k * G), tgt)
        rise = self.act_T0 + (tgt - self.act_T0) * up
        down = ((tt - self.ramp_n - self.hold_n) / self.ret_n).clamp(0.0, 1.0)
        pull = torch.where(tt <= self.ramp_n + self.hold_n, rise, tgt + (self.idle - tgt) * down)
        cmd = torch.where(self.action == RELAX, relax, pull)
        return torch.where(self.action < 0, self.T_cmd, cmd)

    def step(self, T_cmd_override=None):
        """Advance all RUNNING envs by one 200 Hz tick. Noise is drawn for all envs every tick
        (keeps the RNG stream independent of which envs are still running)."""
        c = self.cfg
        run = self.status == RUNNING
        if self.k.max() >= self.kmax:
            raise RuntimeError("contact phase longer than cfg.max_contact_s")
        T_cmd = self._cmd() if T_cmd_override is None else T_cmd_override / self.motor_gain
        self.T_cmd = torch.where(run, T_cmd, self.T_cmd)
        target = self.T_cmd * self.motor_gain
        a = torch.exp(-c.dt / self.tau_T)
        self.thrust = torch.where(run, target + (self.thrust - target) * a, self.thrust)
        self.k = self.k + run.long()
        self.act_t = self.act_t + run.long()
        f = self.forces(self.ar, self.k, self.thrust)
        detach = f["normal_ok"] < 0
        ss = self.slip_speed(f["margin"], f["normal_ok"])
        v = ss.unsqueeze(-1) * f["slip_dir"]
        disp = v * c.dt
        pos = self.pos + disp
        off_plate = pos.abs().max(-1).values > c.plate_half
        a_slip2 = (v - self.v_slip) / c.dt
        a_slip = (a_slip2[:, :1] * self.e_lat + a_slip2[:, 1:] * self.e_up).clamp(-50, 50)
        self.pos = torch.where(run.unsqueeze(-1), pos, self.pos)
        self.v_slip = torch.where(run.unsqueeze(-1), v, self.v_slip)
        lost = run & (detach | off_plate)
        # sensors for all envs (written only where running, so RNG use is status-independent)
        keep_h = self.hist[self.ar, self.pre + self.k].clone()
        keep_fn = self.fn_true[self.ar, self.pre + self.k].clone()
        keep_m, keep_s = self.margin_now.clone(), self.slip_now.clone()
        self._contact_sample(self.ar, self.k, a_slip, ss, ~detach, f)
        if c.slip_sensor:
            self._optical_sample(disp, ~(detach | off_plate))
        nr = ~run
        self.hist[self.ar[nr], self.pre + self.k[nr]] = keep_h[nr]
        self.fn_true[self.ar[nr], self.pre + self.k[nr]] = keep_fn[nr]
        self.margin_now = torch.where(run, self.margin_now, keep_m)
        self.slip_now = torch.where(run, self.slip_now, keep_s)
        # loss of contact with motors running: SELF_ABORT (v2 default) or FELL (v1 behaviour)
        self.loss_cause = torch.where(lost, torch.where(detach, 1, 2), self.loss_cause)
        self.loss_k = torch.where(lost, self.k, self.loss_k)
        new = FELL if c.powered_detach_is_fall else SELF_ABORT
        self.status = torch.where(lost, torch.full_like(self.status, new), self.status)
        return f

    # ------------------------------------------------------------ labels
    @torch.no_grad()
    def rest_fork(self, env, k0, thrust0, pos0, sub=None):
        """Motors off from step k0 (T_cmd = 0) for t_rest, batched over envs and the horizon.

        Returns held (= held_star: no detach, pad on the plate, total slip path < held_slip_max),
        held_strict (margin >= 0 at every step), fell (detach or off the plate), margin_min
        [m/s^2], slip_path [m], fall_t [s] (inf if none).
        """
        c, H = self.cfg, self.H
        if sub is None:                     # bounds peak memory (~0.5 MB per env on CPU)
            sub = 1024 if self.dev.type == "cuda" else 128
        j = torch.arange(1, H + 1, device=self.dev)
        outs = {"held": [], "held_strict": [], "fell": [], "margin_min": [], "slip_path": [], "fall_t": []}
        for s0 in range(0, env.numel(), sub):
            e = env[s0:s0 + sub]
            m = e.numel()
            kk = (k0[s0:s0 + sub, None] + j).reshape(-1)
            ee = e[:, None].expand(m, H).reshape(-1)
            decay = torch.exp(-c.dt / self.tau_T[e])[:, None] ** j.float()
            th = (thrust0[s0:s0 + sub, None] * decay).reshape(-1)
            f = self.forces(ee, kk, th)
            mg = f["margin"].view(m, H)
            nok = f["normal_ok"].view(m, H)
            ss = self.slip_speed(f["margin"], f["normal_ok"])
            v = (ss.unsqueeze(-1) * f["slip_dir"]).view(m, H, 2)
            pos = pos0[s0:s0 + sub, None] + torch.cumsum(v * c.dt, 1)
            bad = (nok < 0) | (pos.abs().max(-1).values > c.plate_half)
            first = torch.where(bad.any(1), bad.float().argmax(1) + 1, torch.full((m,), -1, device=self.dev))
            fell = bad.any(1)
            path = ss.view(m, H).sum(1) * c.dt
            outs["held"].append(~fell & (path < c.held_slip_max))
            outs["held_strict"].append((mg >= 0).all(1))
            outs["fell"].append(fell)
            outs["margin_min"].append(mg.min(1).values)
            outs["slip_path"].append(path)
            outs["fall_t"].append(torch.where(first > 0, first.float() * c.dt, torch.full_like(first.float(), math.inf)))
        return {k: torch.cat(v) for k, v in outs.items()}

    @torch.no_grad()
    def static_margin_off(self, env, k):
        """Privileged: margin with zero thrust at the CURRENT cart acceleration (no future)."""
        return self.forces(env, k, torch.zeros_like(k, dtype=torch.float32))["margin"]

    # ------------------------------------------------------------ windows
    def windows(self, env, k, length=80):
        """Sensor windows (len(env), length, 11 or 13) ending at step k (inclusive)."""
        i = self.pre + k[:, None] - (length - 1) + torch.arange(length, device=self.dev)
        w = self.hist[env[:, None], i]
        if self.cfg.imu_hold:
            # 100 Hz sample-and-hold of acc/gyro: odd samples (w.r.t. a per-episode phase) repeat the previous one
            odd = ((i + self.imu_hold_phase[env, None]) % 2 == 1) & self.imu_hold_on[env, None]
            j = (i - odd.long()).clamp(min=0)
            w = w.clone()
            w[..., :6] = self.hist[env[:, None], j][..., :6]
        return w

    def run_script_to(self, k_end, record=False, script=SCRIPT):
        """Run the fixed schedule (no decisions) until step k_end; optionally record the state."""
        rec = None
        if record:
            K = k_end + 1
            rec = {"thrust": torch.zeros(self.N, K, device=self.dev), "pos": torch.zeros(self.N, K, 2, device=self.dev),
                   "alive": torch.zeros(self.N, K, dtype=torch.bool, device=self.dev),
                   "margin": torch.zeros(self.N, K, device=self.dev), "slip": torch.zeros(self.N, K, dtype=torch.bool, device=self.dev)}
            self._rec(rec, 0)
        sc = torch.tensor(script, device=self.dev)
        for kk in range(k_end):
            r = self.ready
            if r.any():
                a = sc[self.n_actions.clamp(max=len(script) - 1)]
                a = torch.where(self.n_actions >= len(script), torch.full_like(a, -1), a)
                self.act(torch.where(r, a, torch.full_like(a, -1)))
            self.step()
            if record:
                self._rec(rec, kk + 1)
        return rec

    def _rec(self, rec, k):
        rec["thrust"][:, k] = self.thrust
        rec["pos"][:, k] = self.pos
        rec["alive"][:, k] = self.status == RUNNING
        rec["margin"][:, k] = self.margin_now
        rec["slip"][:, k] = self.slip_now


def decision_steps(cfg: ContactCfg):
    """Step indices at the end of each macro of the default schedule: [RELAX, PULL0.3, PULL0.6, PULL0.9]."""
    r = lambda s: int(round(s / cfg.dt))
    a, p = r(cfg.relax_s), r(cfg.ramp_s) + r(cfg.pull_hold_s) + r(cfg.pull_return_s)
    return [a, a + p, a + 2 * p, a + 3 * p]
