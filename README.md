# PerchPilot

Quadrotor learning **agile perching on inclined plates (55–70°) mounted on a moving cart (0–3 m/s)**, with **attachment verification before motors-off** and a full **perch–rest–depart** cycle. Simulation only: a GPU PyTorch simulator, Gazebo Harmonic and PX4 v1.15.4 SITL. No truth pose at test time and no depth camera.

Course project for *Robotics & AI (PRRO317129)*. Detailed setup and reproduction commands (Vietnamese): [`README_vi.md`](README_vi.md).

| Folder | Content |
|---|---|
| [`sim/`](sim) | GPU simulator: quad + moving cart + inclined plate, PX4-like command delay, camera FOV, odometry noise, Kalman filter, optional 4-motor saturation/airmode model; contact-phase simulator for attachment verification |
| [`rl/`](rl) | Minimal PPO (asymmetric critic, warm start with input expansion) |
| [`q2/`](q2) | Attachment verification: dataset, 1D-CNN belief net, decision rules (learned D-thr, FSM baselines B4a–B4e), locked-threshold evaluation protocol with paired bootstrap |
| [`ros/`](ros) | ROS 2 nodes: policy node, tag/camera simulator, evaluator, Q2 verifier, browser dashboard |
| [`gz/`](gz) | Gazebo world, `x500_perch` model, `HoldPad` C++ system plugin (finite holding force, load cell, optical slip sensor) |
| [`px4_patches/`](px4_patches) | PX4 patch: "perch mode" so the land detector allows re-arming on a moving vehicle |
| [`scripts/`](scripts) | Training, evaluation, sensitivity, isolated SITL launch, batch SITL evaluation, tables, figures, video from logs |

## Quick start

```bash
PY=python3   # Python 3.12 with torch (CUDA), numpy, matplotlib
$PY scripts/train.py --name g1 --phase P0 --minutes 30 --seed 1 --out_root runs   # Q1, privileged state, no delay
$PY scripts/train.py --name p1 --phase P1 --minutes 25 --calib --hist --asym --gate --out_root runs \
    --tw 1.9,2.6 --pad_offset 0.227 --init runs/g1/best.pt              # Q1 with PX4-like delay
(cd gz/plugins/hold_pad && cmake -S . -B build && cmake --build build)  # Gazebo HoldPad plugin + C++ tests
```

## Notes

- Default data paths assume `/home/sup/data/perchpilot/...`; pass the CLI arguments to change them.
- Experiment logs, checkpoints, results and the written report are not included.
- `gz/models/x500_perch` is derived from the PX4 `x500` model; see its `LICENSE`.
