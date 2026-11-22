"""Swap channel groups between SITL RELAX-end windows (strong grip, q2_probe_m1) and sim RELAX-end windows (same
material) to find what drives p_slip -> 1 on SITL (ROS-free, CPU)."""
import sys, json
from pathlib import Path
import numpy as np, torch
R = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, R); sys.path.insert(0, R + '/ros')
from perch_q2 import BeliefNet
from q2_belief_probe import windows
from sim.contact_phase import SCRIPT, ContactCfg, ContactPhaseSim, decision_steps
net = BeliefNet('/home/sup/data/perchpilot/runs/q2_belief_v2_small')
W = [w for w in windows('/home/sup/data/perchpilot/sitl_eval/q2_probe_m1') if w['macro'] == 'RELAX' and w['trial'] in (1, 4)]
cfg = ContactCfg(num_envs=64, mu=((0.9, 0.9001),), f_hold_mg=((3.0, 3.0001),), plate_tilt_deg=(65, 65.001),
                 veh_speed=(0, 1e-6), tw_ratio=(2.26, 2.2601))
sim = ContactPhaseSim(cfg, device='cpu', seed=1)
ks = decision_steps(sim.cfg)
sim.run_script_to(ks[0], script=SCRIPT)
sw = sim.windows(sim.ar, torch.full_like(sim.ar, ks[0])).numpy()
print('sim RELAX p:', np.round(np.mean([net(x) for x in sw], 0), 3))
groups = {'acc': [0, 1, 2], 'gyro': [3, 4, 5], 'f': [6], 'contact': [7], 'thr': [8], 'phase+level': [9, 10]}
np.set_printoptions(precision=2, suppress=True, linewidth=180)
for it in W:
    w = it['w']
    print('trial', it['trial'], 'raw', np.round(net(w), 3))
    print('  sitl phase col:', w[:, 9].astype(int).tolist())
    print('  sitl contact  :', w[:, 7].astype(int).tolist())
    print('  sim  phase col:', sw[0, :, 9].astype(int).tolist())
    print('  sim  contact  :', sw[0, :, 7].astype(int).tolist())
    print('  sitl acc mean/std', w[:, 0:3].mean(0), w[:, 0:3].std(0), ' sim', sw[0, :, 0:3].mean(0), sw[0, :, 0:3].std(0))
    print('  sitl f', w[-10:, 6].mean(), 'thr01 start/end', w[0, 8], w[-1, 8], ' sim thr01', sw[0, 0, 8], sw[0, -1, 8])
    for g, ch in groups.items():
        x = w.copy(); x[:, ch] = sw[0][:, ch]
        y = sw[0].copy(); y[:, ch] = w[:, ch]
        print(f'  SITL with sim {g:12s} -> {np.round(net(x), 3)}   sim with SITL {g:12s} -> {np.round(net(y), 3)}')
