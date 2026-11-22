"""Offline check of the SITL closed loop against Gazebo truth.

Recomputes the IDEAL v1 observation (same formulas as QuadPerchEnv.observe via perch_core.build_obs) from the
Gazebo model poses logged by eval_node and compares it element-wise with the observation perch_node fed to the
policy (perch_node --tick-log). Both logs carry CLOCK_MONOTONIC wall stamps, so they are aligned in wall time.

Also reports: EKF2 vs truth (position offset/scatter, velocity, attitude, yaw), plate estimate vs truth,
command->motion delay of the body rates (cross-correlation of commanded vs true body rates), and the Gazebo
real-time factor (sim_t vs wall over the pose stream).

    python perchpilot/ros/sitl_obs_check.py /home/sup/data/perchpilot/sitl_eval/<name> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perch_core import (DroneState, OBS_V1_LAYOUT, PlateState, build_obs, orthonormalize, plate_from_cart,  # noqa: E402
                        quat_to_rot)

DRONE, CART = 'x500_perch_7', 'perch_cart'
BASE_LINK_IN_MODEL = np.array([0.0, 0.0, 0.24])   # x500_base merge offset (see tag_camera_sim.py)
RATE_MAX = np.array([6.0, 6.0, 3.0])


def log_so3(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return a / (2 * math.sin(a)) * v


class Truth:
    """Gazebo model poses of one trial, indexed by arrival wall time."""

    def __init__(self, rows):
        poses = [r for r in rows if r['kind'] == 'pose' and 'wall' in r]
        self.wall = np.array([r['wall'] for r in poses])
        self.sim = np.array([r['sim_t'] for r in poses])
        self.pm = np.array([r['models'][DRONE]['xyz'] for r in poses])
        self.Rm = np.array([quat_to_rot(r['models'][DRONE]['wxyz']) for r in poses])
        self.pc = np.array([r['models'][CART]['xyz'] for r in poses])
        self.Rc = np.array([quat_to_rot(r['models'][CART]['wxyz']) for r in poses])
        self.pb = self.pm + np.einsum('nij,j->ni', self.Rm, BASE_LINK_IN_MODEL)
        # Gazebo stamps are exact; arrival wall has jitter -> map sim -> wall with a robust linear fit
        if len(self.sim) > 5:
            A = np.vstack([self.sim, np.ones_like(self.sim)]).T
            self.fit = np.linalg.lstsq(A, self.wall, rcond=None)[0]
            res = self.wall - A @ self.fit
            keep = np.abs(res - np.median(res)) < 3 * (np.median(np.abs(res - np.median(res))) + 1e-4)
            self.fit = np.linalg.lstsq(A[keep], self.wall[keep], rcond=None)[0]
            # arrival lags generation; the earliest arrivals bound the latency from below
            self.fit[1] += np.min(self.wall[keep] - A[keep] @ self.fit)
        else:
            self.fit = np.array([1.0, 0.0])
        self.t = self.sim * self.fit[0] + self.fit[1]   # generation time on the wall clock

    @property
    def rtf(self):
        return 1.0 / self.fit[0] if self.fit[0] else float('nan')

    def ok(self, t):
        return len(self.t) > 5 and self.t[2] <= t <= self.t[-3]

    def _interp(self, arr, t):
        i = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
        a = (t - self.t[i - 1]) / max(self.t[i] - self.t[i - 1], 1e-9)
        return arr[i - 1] * (1 - a) + arr[i] * a, i

    def state(self, t, h=0.02):
        p, i = self._interp(self.pb, t)
        R = orthonormalize(self._interp(self.Rm, t)[0])
        pm_, _ = self._interp(self.pb, t - h)
        pp_, _ = self._interp(self.pb, t + h)
        v = (pp_ - pm_) / (2 * h)
        Rm_ = orthonormalize(self._interp(self.Rm, t - h)[0])
        Rp_ = orthonormalize(self._interp(self.Rm, t + h)[0])
        w = log_so3(Rm_.T @ Rp_) / (2 * h)              # body-frame rates (FLU)
        pc, _ = self._interp(self.pc, t)
        Rc = orthonormalize(self._interp(self.Rc, t)[0])
        vc = (self._interp(self.pc, t + 0.05)[0] - self._interp(self.pc, t - 0.05)[0]) / 0.1
        c, Rp = plate_from_cart(pc, Rc)
        return DroneState(p=p, v=v, R=R, w=w), PlateState(c=c, R=Rp, v=vc)


def obs_slices(obs_dim):
    """v1 prefix, pending-action buffer (4*d) and, when obs_dim % 4 == 3, the v2 suffix [rho, visible, t_loss]."""
    out, k = {}, 0
    for name, n in OBS_V1_LAYOUT:
        out[name] = slice(k, k + n)
        k += n
    tag = 3 if (obs_dim - k) % 4 == 3 else 0
    if obs_dim - tag > k:
        out['act_hist'] = slice(k, obs_dim - tag)
    if tag:
        out['rho'], out['visible'], out['t_loss_norm'] = (slice(obs_dim - 3 + i, obs_dim - 2 + i) for i in range(3))
    return out


def stats(err):
    err = np.asarray(err)
    return {'mean': np.round(err.mean(0), 4).tolist(), 'std': np.round(err.std(0), 4).tolist(),
            'rms': np.round(np.sqrt((err ** 2).mean(0)), 4).tolist(), 'max_abs': np.round(np.abs(err).max(0), 4).tolist()}


def analyse(run_dir, pad_offset=0.227):
    run_dir = Path(run_dir)
    ticks = [json.loads(line) for line in (run_dir / 'node_ticks.jsonl').open()]
    truths = {}
    for f in sorted(run_dir.glob('truth_*.jsonl')):
        rows = [json.loads(line) for line in f.open()]
        trial_rows = [r for r in rows if r['kind'] == 'pose']
        if trial_rows:
            truths[int(f.stem.split('_')[1])] = Truth(rows)
    res = {'rtf': {k: round(t.rtf, 4) for k, t in truths.items()}, 'trials': {}}
    comp_err = {}
    ekf_err = {'p': [], 'v': [], 'att_deg': [], 'yaw_deg': []}
    plate_err = {'c_rel': [], 'v': [], 'n_deg': []}
    rate_cmd, rate_t, trial_of = [], [], []
    for tk in ticks:
        tr = truths.get(tk['trial'])
        if tr is None or not tr.ok(tk['wall']):
            continue
        d_true, p_true = tr.state(tk['wall'])
        e = tk['ekf']
        R_ekf = quat_to_rot(e['q'])
        ekf_err['p'].append(np.array(e['p']) - d_true.p)          # constant EKF-origin offset + drift
        ekf_err['v'].append(np.array(e['v']) - d_true.v)
        dR = R_ekf.T @ d_true.R
        ekf_err['att_deg'].append(math.degrees(np.linalg.norm(log_so3(dR))))
        ekf_err['yaw_deg'].append(math.degrees(math.remainder(
            math.atan2(R_ekf[1, 0], R_ekf[0, 0]) - math.atan2(d_true.R[1, 0], d_true.R[0, 0]), 2 * math.pi)))
        if tk.get('plate'):
            pl = tk['plate']
            Rp_est = quat_to_rot(pl['q'])
            # plate relative to the drone, in the drone's (EKF / true) body frame: frame-offset free
            rel_est = R_ekf.T @ (np.array(pl['c']) - np.array(e['p']))
            rel_true = d_true.R.T @ (p_true.c - d_true.p)
            plate_err['c_rel'].append(rel_est - rel_true)
            plate_err['v'].append(np.array(pl['v']) - p_true.v)
            n_est_b, n_true_b = R_ekf.T @ Rp_est[:, 2], d_true.R.T @ p_true.R[:, 2]
            plate_err['n_deg'].append(math.degrees(math.acos(np.clip(n_est_b @ n_true_b, -1, 1))))
        if tk.get('obs') is None:
            continue
        obs = np.array(tk['obs'])
        sl = obs_slices(len(obs))
        tag = 'rho' in sl
        hl = (len(obs) - 28 - (3 if tag else 0)) // 4
        hist = [obs[28 + 4 * i: 32 + 4 * i] for i in range(hl)] if hl else None
        if tag:   # camera-state inputs are node bookkeeping, not physics: take them from the node
            p_true.reliability, p_true.visible = float(obs[sl['rho']][0]), bool(obs[sl['visible']][0] > 0.5)
            p_true.loss_s = float(obs[sl['t_loss_norm']][0]) * 2.0
        ideal = build_obs(d_true, p_true, obs[sl['prev_act']], obs[sl['t_norm']][0], pad_offset, hist,
                          tag_inputs=tag)
        for name, s in sl.items():
            comp_err.setdefault(name, []).append(obs[s] - ideal[s])
        comp_err.setdefault('_ideal_rel_p', []).append(ideal[sl['rel_p_uwn']])
        comp_err.setdefault('_node_rel_p', []).append(obs[sl['rel_p_uwn']])
        if tk.get('action') is not None:
            rate_cmd.append(np.array(tk['action'][1:4]) * RATE_MAX)
            rate_t.append(tk['wall'])
            trial_of.append(tk['trial'])
    res['obs_minus_ideal'] = {k: stats(v) for k, v in comp_err.items() if not k.startswith('_') and len(v)}
    if comp_err.get('_ideal_rel_p'):
        a, b = np.array(comp_err['_ideal_rel_p']), np.array(comp_err['_node_rel_p'])
        res['rel_p_fit'] = {'scale_node_vs_ideal': [round(float(np.polyfit(a[:, i], b[:, i], 1)[0]), 3)
                                                    for i in range(3)],
                            'corr': [round(float(np.corrcoef(a[:, i], b[:, i])[0, 1]), 3) for i in range(3)]}
    res['n_obs_ticks'] = len(comp_err.get('rel_p_uwn', []))
    if ekf_err['p']:
        p = np.array(ekf_err['p'])
        res['ekf'] = {'p_offset_median': np.round(np.median(p, 0), 3).tolist(),
                      'p_minus_offset': stats(p - np.median(p, 0)), 'v': stats(ekf_err['v']),
                      'att_err_deg': stats(np.array(ekf_err['att_deg'])[:, None]),
                      'yaw_err_deg': stats(np.array(ekf_err['yaw_deg'])[:, None])}
    if plate_err['c_rel']:
        res['plate'] = {'c_rel_body': stats(plate_err['c_rel']), 'v': stats(plate_err['v']),
                        'n_err_deg': stats(np.array(plate_err['n_deg'])[:, None])}
    res['rate_delay'] = rate_delay(truths, np.array(rate_t), np.array(rate_cmd), np.array(trial_of))
    res['ekf_omega_lag'] = ekf_lag(ticks, truths)
    return res


def ekf_lag(ticks, truths, lags=np.arange(-0.1, 0.1001, 0.005)):
    """Age of the EKF2 body rates used in the observation: lag that best matches truth (negative = EKF late)."""
    rows = [(tk['wall'], np.array(tk['ekf']['w']), truths[tk['trial']]) for tk in ticks
            if tk['trial'] in truths and tk.get('obs') is not None]
    if len(rows) < 50:
        return None
    best = None
    for lag in lags:
        err = [float(np.sum((w - tr.state(t + lag)[0].w) ** 2)) for t, w, tr in rows if tr.ok(t + lag)]
        if len(err) > 30 and (best is None or np.mean(err) < best[1]):
            best = (lag, float(np.mean(err)))
    return {'lag_s': round(best[0], 3), 'rms_rad_s': round(best[1] ** 0.5, 3)} if best else None


def rate_delay(truths, t, cmd, trial_of, lags=np.arange(0.0, 0.205, 0.005)):
    """Lag that best explains the true body rates as the commanded rates delayed (+ first-order lag fit)."""
    if len(t) < 50:
        return None
    out = {}
    for axis, name in ((0, 'roll'), (1, 'pitch'), (2, 'yaw')):
        best = None
        for lag in lags:
            x, y = [], []
            for tt, c, tr_id in zip(t, cmd[:, axis], trial_of):
                tr = truths[tr_id]
                if tr.ok(tt + lag):
                    x.append(c)
                    y.append(tr.state(tt + lag)[0].w[axis])
            if len(x) < 30:
                continue
            x, y = np.array(x), np.array(y)
            if x.std() < 1e-6 or y.std() < 1e-6:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if best is None or r > best[1]:
                best = (lag, r, float(np.polyfit(x, y, 1)[0]))
        if best:
            out[name] = {'lag_s': round(best[0], 3), 'corr': round(best[1], 3), 'gain': round(best[2], 3)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--pad-offset', type=float, default=0.227)
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    res = analyse(a.run_dir, a.pad_offset)
    text = json.dumps(res, indent=1)
    print(text)
    if a.json:
        Path(a.json).write_text(text)


if __name__ == '__main__':
    main()
