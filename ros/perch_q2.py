"""Q2 in the loop (ROS-free): probing macros, the C5 200 Hz sensor window and the decision rules.

The macros, decision points and sensor channels follow sim/contact_phase.py (design v2, docs/CONTRACTS.md C5):
    RELAX     thrust -> idle (0.1 hover) over 50 ms, hold to 0.15 s
    PULL(k)   ramp to k*hover in 50 ms, hold 150 ms, ramp back to idle in 50 ms (0.25 s)
    one decision point at the end of each macro.
Channels: acc_body(3) [m/s^2, specific force, body FLU], gyro_body(3) [rad/s, FLU], f_normal_N (signed,
+ compression), contact, thrust_cmd01, phase_id (0 approach, 1 RELAX, 2 PULL, 3 OFF, 4 ABORT), probe_level,
and in C5 v2, optical slip displacement (u, w) in mm per sample.

Rules (`--verify-rule`):
    none      MOTORS_OFF as soon as contact is detected (baseline "turn off at touchdown, no check")
    dthr      belief net (belief.ts, raw window in, [p_hold, p_slip, margin] out) with the v3 schedule
              RELAX -> PULL0.3 -> PULL0.6: p_slip >= s -> ABORT; p_hold >= tau -> MOTORS_OFF; else next macro;
              after the last macro -> ABORT  (q2/decide_v3.py dthr_fn / policy, single net)
    fsm_b4b   high-pass energy of (acc_x, acc_y) over the macro just finished (RELAX: last 15 samples,
              PULL: 50) >= e_thr -> ABORT; no slip at any point -> MOTORS_OFF after PULL0.6 (decide_v3.b4b)
    fsm_b4e   C5 v2 optical displacement: |sum of (slip_u, slip_w) over the 80-sample window| < slip_off_mm[d]
              -> MOTORS_OFF at decision point d; else next macro; ABORT after PULL0.6 (q2/decide_v4.b4e,
              fsm_features.slip_distance; thresholds from a Q2 v6 lock selection["B4e"][cap])
    probe     measurement mode: full v2 schedule RELAX -> PULL0.3 -> PULL0.6 -> PULL0.9, then MOTORS_OFF
Thresholds and the belief checkpoint come from paths given on the command line (swap for Q2 v4 by replacing
the checkpoint / JSON).
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

G = 9.81
RELAX, PULL03, PULL06, PULL09 = "RELAX", "PULL0.3", "PULL0.6", "PULL0.9"
PULL_LEVEL = {PULL03: 0.3, PULL06: 0.6, PULL09: 0.9}
PH_APPROACH, PH_RELAX, PH_PULL, PH_OFF, PH_ABORT = range(5)
SCHEDULE_V3 = [RELAX, PULL03, PULL06]
SCHEDULE_V2 = [RELAX, PULL03, PULL06, PULL09]
N_CH, WIN = 11, 80
DT = 0.005


@dataclass
class Q2Cfg:
    rule: str = "placeholder"        # placeholder | none | dthr | fsm_b4b | fsm_b4e | probe
    idle_frac: float = 0.1
    relax_s: float = 0.15
    ramp_s: float = 0.05
    pull_hold_s: float = 0.15
    pull_return_s: float = 0.05
    tau: float = 0.56                # D-thr thresholds (decide_v3 "D-thr_v3_early_stop", chosen on val)
    s: float = 0.6
    d_min: int = 0                   # D-thr: earliest decision point allowed to turn the motors off (1 = after PULL0.3)
    e_thr: float = 0.6571889521200449   # B4b threshold (chosen on val)
    slip_off_mm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])   # B4e, per decision point
    contact_loss_s: float = 0.03     # contact flag low this long during VERIFY -> SELF_ABORT
    schedule: List[str] = field(default_factory=lambda: list(SCHEDULE_V3))

    @property
    def pull_s(self):
        return self.ramp_s + self.pull_hold_s + self.pull_return_s

    def duration(self, macro):
        return self.relax_s if macro == RELAX else self.pull_s


def load_q2_cfg(rule, decide_json=None, dthr_key="D-thr_v3_early_stop", b4b_key="B4b_FSM_probe"):
    cfg = Q2Cfg(rule=rule)
    if rule == "probe":
        cfg.schedule = list(SCHEDULE_V2)
    if decide_json:
        chosen = json.loads(Path(decide_json).read_text())["chosen_on_val"]
        cfg.tau, cfg.s = float(chosen[dthr_key]["tau"]), float(chosen[dthr_key]["s"])
        cfg.d_min = 1 if "minPULL0.3" in dthr_key else 0
        cfg.e_thr = float(chosen[b4b_key]["e_thr"])
    return cfg


def macro_thrust(macro, t, T0, cfg: Q2Cfg):
    """Thrust command [m/s^2 per unit mass] t seconds into `macro` that started at thrust T0 (contact_phase._cmd)."""
    idle = cfg.idle_frac * G
    up = min(1.0, max(0.0, t / cfg.ramp_s))
    if macro == RELAX:
        return T0 + (idle - T0) * up
    tgt = PULL_LEVEL[macro] * G
    if t <= cfg.ramp_s + cfg.pull_hold_s:
        return T0 + (tgt - T0) * up
    down = min(1.0, max(0.0, (t - cfg.ramp_s - cfg.pull_hold_s) / cfg.pull_return_s))
    return tgt + (idle - tgt) * down


def hp_energy(window, d):
    """B4b slip feature (decide_v3.fsm_features): mean squared first difference of acc_x, acc_y."""
    L = 15 if d == 0 else 50
    hp = np.diff(np.asarray(window)[-L:, 0:2], axis=0)
    return float((hp ** 2).sum(-1).mean())


class BeliefNet:
    """belief.ts (TorchScript, raw C5 window -> [p_hold, p_slip, margin]); CPU, 1 thread."""

    def __init__(self, run_dir):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.meta = json.loads((Path(run_dir) / "belief.json").read_text())
        self.mod = torch.jit.load(str(Path(run_dir) / "belief.ts"), map_location="cpu").eval()
        self.n_ch = self.mod.mean.numel()

    def __call__(self, window):
        x = self.torch.from_numpy(np.asarray(window, dtype=np.float32)[None, :, :self.n_ch])
        if x.shape[-1] != self.n_ch:
            raise ValueError("belief checkpoint requires C5 v2 optical slip samples")
        with self.torch.inference_mode():
            o = self.mod(x)[0].numpy()
        return float(o[0]), float(o[1])


class SensorWindow:
    """Rolling 200 Hz C5 buffer (last 80 samples)."""

    def __init__(self):
        self.buf = deque(maxlen=WIN)

    def add(self, acc_flu, gyro_flu, f_normal, contact, thrust_cmd01, phase_id, probe_level, slip_disp=()):
        if self.buf and len(self.buf[-1]) != N_CH + len(slip_disp):
            self.buf.clear()
        self.buf.append(np.concatenate([np.asarray(acc_flu, float), np.asarray(gyro_flu, float),
                                        [f_normal, float(contact), thrust_cmd01, phase_id, probe_level], slip_disp]))

    def ready(self):
        return len(self.buf) == WIN

    def array(self):
        return np.stack(self.buf).astype(np.float32)


class Q2Verifier:
    """Macro scheduler + decision rule. step() returns (thrust_acc, decision) with decision in
    {None, 'OFF', 'ABORT', 'SELF_ABORT'}; `state` gives the phase_id / probe_level for the sensor window."""

    def __init__(self, cfg: Q2Cfg, belief: Optional[BeliefNet] = None):
        self.cfg, self.belief = cfg, belief
        if cfg.rule == "dthr" and belief is None:
            raise ValueError("verify rule dthr needs a belief net")
        self.reset(0.0, G)

    def reset(self, t, T0):
        self.t0, self.T0, self.i, self.no_contact_since = t, T0, 0, None
        self.thrust, self.done = T0, False
        self.result = {}
        self.decisions = []

    @property
    def macro(self):
        return self.cfg.schedule[min(self.i, len(self.cfg.schedule) - 1)]

    @property
    def state(self):
        m = self.macro
        return (PH_RELAX, self.cfg.idle_frac) if m == RELAX else (PH_PULL, PULL_LEVEL[m])

    def _finish(self, decision, why, **kw):
        self.done = True
        self.result = {"decision": decision, "why": why, "d": self.i, **kw}
        return decision

    def step(self, t, contact, window: Optional[np.ndarray]):
        c = self.cfg
        if self.done:
            return self.thrust, None
        if c.rule == "none":
            return self.thrust, self._finish("OFF", "rule none: motors off at touchdown")
        # loss of contact with the motors running -> SELF_ABORT (contact_phase v2)
        if contact is False:
            self.no_contact_since = t if self.no_contact_since is None else self.no_contact_since
            if t - self.no_contact_since >= c.contact_loss_s:
                return self.thrust, self._finish("SELF_ABORT", "contact lost while powered")
        else:
            self.no_contact_since = None
        tm = t - self.t0
        dur = c.duration(self.macro)
        if tm < dur - 1e-9:
            self.thrust = macro_thrust(self.macro, tm, self.T0, c)
            return self.thrust, None
        # decision point at the end of the macro
        d = self.i
        rec = {"d": d, "macro": self.macro, "t": round(tm, 4)}
        decision = None
        if c.rule == "dthr":
            p_hold, p_slip = self.belief(window)
            rec.update(p_hold=round(p_hold, 4), p_slip=round(p_slip, 4))
            if p_slip >= c.s:
                decision = "ABORT"
            elif p_hold >= c.tau and d >= c.d_min:
                decision = "OFF"
            elif d == len(c.schedule) - 1:
                decision = "ABORT"
        elif c.rule == "fsm_b4b":
            e = hp_energy(window, d)
            rec["hp_energy"] = round(e, 5)
            if e >= c.e_thr:
                decision = "ABORT"
            elif d == len(c.schedule) - 1:
                decision = "OFF"
        elif c.rule == "fsm_b4e":
            if window is None or window.shape[1] < N_CH + 2:
                raise ValueError("verify rule fsm_b4e needs the C5 v2 window (PERCH_SLIP_SENSOR=1)")
            slip_mm = float(np.linalg.norm(window[:, N_CH:N_CH + 2].sum(0)))
            rec["slip_mm"] = round(slip_mm, 4)
            if slip_mm < c.slip_off_mm[d]:
                decision = "OFF"
            elif d == len(c.schedule) - 1:
                decision = "ABORT"
        elif c.rule == "probe":
            if window is not None:
                rec["hp_energy"] = round(hp_energy(window, d), 5)
            if d == len(c.schedule) - 1:
                decision = "OFF"
        self.decisions.append(rec)
        if decision:
            return self.thrust, self._finish(decision, f"{c.rule} at {self.macro}", **{k: v for k, v in rec.items()
                                                                                       if k not in ("d",)})
        self.i += 1
        self.t0, self.T0 = t, self.thrust
        self.thrust = macro_thrust(self.macro, 0.0, self.T0, c)
        return self.thrust, None
