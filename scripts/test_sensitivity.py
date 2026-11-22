"""Small P6 tests. Run directly with --device cpu (default) or --device cuda."""

import argparse
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import torch

from baselines_q1 import score_baselines
from eval_ckpt import ROOT, write_result
from make_figures import generate
from rl.ppo import ActorCritic
from sensitivity import sensitivity
from sim.quad_perch_env import G, quat_to_rot
from train import evaluate, make_env


# Compare with the exact source before the buffer change on the same PyTorch backend.
PRE_P6 = "ea6ea8adfedfb2fd9789141ce27a795f7210ee2a"
DEVICE = "cpu"
N = 16
TRAIN_ENV = {"delay_steps": (4, 6), "act_history": True}


def legacy_envs():
    def load(name):
        path = f"perchpilot/sim/{name.rsplit('.', 1)[1]}.py"
        source = subprocess.check_output(["git", "show", f"{PRE_P6}:{path}"], cwd=ROOT)
        module = ModuleType(name)
        sys.modules[name] = module
        exec(compile(source, f"{PRE_P6}:{path}", "exec"), module.__dict__)
        return module

    with patch.dict(sys.modules):
        v1 = load("sim.quad_perch_env")
        v2 = load("sim.quad_perch_env_v2")
    return ((v1.QuadPerchEnv, v1.EnvCfg), (v2.QuadPerchEnvV2, v2.EnvCfgV2))


class SensitivityTest(unittest.TestCase):
    def assert_bits(self, left, right, label):
        self.assertEqual((left.dtype, left.shape), (right.dtype, right.shape), label)
        self.assertTrue(torch.equal(left.flatten().contiguous().view(torch.uint8),
                                    right.flatten().contiguous().view(torch.uint8)), label)

    def test_legacy_trajectories_bit_for_bit(self):
        # needs the pre-P6 commit of the development repository; skipped in copies without that git history
        if subprocess.run(["git", "cat-file", "-e", f"{PRE_P6}^{{commit}}"], cwd=ROOT, capture_output=True).returncode:
            self.skipTest(f"git history with commit {PRE_P6} not available")
        commands = torch.rand(200, 8, 4, generator=torch.Generator().manual_seed(441)).to(DEVICE) * 2 - 1
        for v2, (legacy, cfg_cls) in enumerate(legacy_envs()):
            for config in ({}, TRAIN_ENV):
                with self.subTest(v2=bool(v2), config=config):
                    old = legacy(cfg_cls(num_envs=8, **config), device=DEVICE, seed=312)
                    new = make_env(8, 312, v2=bool(v2), device=DEVICE, **config)
                    self.assert_bits(old.observe(), new.observe(), "initial observation")
                    for step, command in enumerate(commands):
                        *old_step, old_info = old.step(command)
                        *new_step, new_info = new.step(command)
                        for key, a, b in zip(("obs", "reward", "done"), old_step, new_step):
                            self.assert_bits(a, b, (step, key))
                        self.assertEqual(old_info.keys(), new_info.keys())
                        for key in old_info:
                            self.assert_bits(old_info[key], new_info[key], (step, "info", key))
                        for key, value in vars(old).items():
                            if isinstance(value, torch.Tensor):
                                self.assert_bits(value, getattr(new, key), (step, "state", key))
                        self.assert_bits(old.privileged(), new.privileged(), (step, "critic"))
                        self.assert_bits(old.gen.get_state(), new.gen.get_state(), (step, "rng"))

    def test_delay_changes_without_moving_actor_columns(self):
        for v2 in (False, True):
            for steps in (3, 7, 9):
                with self.subTest(v2=v2, steps=steps):
                    env = make_env(4, 51, v2=v2, device=DEVICE, **TRAIN_ENV, substeps=1,
                                   obs_noise=0, command_delay_steps=(steps, steps))
                    self.assertEqual(env.obs_dim, 28 + 4 * 6 + 3 * v2)
                    self.assertEqual(env.act_buf.shape[1], max(steps, 6) + 1)
                    self.assertTrue((env.delay == steps).all())
                    initial = env.prev_act[:, 0].clone()
                    commands = [initial + 0.02 * i for i in range(1, steps + 3)]
                    for i, command in enumerate(commands):
                        act = torch.zeros_like(env.prev_act)
                        act[:, 0] = command
                        delayed = commands[i - steps] if i >= steps else initial
                        target = (delayed + 1) * 0.5 * env.tw * G
                        expected = env.thrust + (target - env.thrust) * (env.cfg.dt / env.tau_T)
                        obs, _, done, _ = env.step(act)
                        self.assertFalse(done.any())
                        self.assert_bits(expected, env.thrust, (steps, i, "delayed thrust"))
                        self.assert_bits(obs[:, 28:52], env.act_buf[:, 1:7].flatten(1), "actor history")

    def test_lost_track_is_scored_before_reset(self):
        env = make_env(8, 51, v2=True, device=DEVICE, episode_s=0.01, cam_range=0.0, t_loss_max=0.25)
        env.t_since_meas.fill_(1.2)
        actor = ActorCritic(env.obs_dim, 4).to(DEVICE)
        with patch("train.make_env", return_value=env):
            report = evaluate(actor, n=8, v2=True, device=DEVICE, fail_breakdown=True)
        self.assertEqual(report["timeout"]["p"], 1.0)
        self.assertEqual(report["fail_breakdown"]["lost_track"], 8)
        self.assertEqual(report["fail_diagnostics"]["lost_track_at_end"], 8)
        self.assertTrue((env.t_loss <= 0.3).all())

    def test_failure_breakdown_separates_collision_and_contact_failures(self):
        reasons = ("crash_plate_edge", "crash_cart", "ground", "bad_vn", "bad_vt", "bad_align",
                   "crash_plate_behind", "timeout", "success")
        env = make_env(len(reasons), 51, device=DEVICE, episode_s=0.01, veh_speed=(0, 0),
                       plate_tilt_deg=(0, 0), plate_height=(1.2, 1.2), drag=(0, 0))
        env.veh_psi.zero_()
        env.q.zero_()
        env.q[:, 0] = 1
        env.q[5, :2] = env.q.new_tensor([math.cos(math.pi / 8), math.sin(math.pi / 8)])
        env.v[:] = env.v.new_tensor([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, -2), (1.5, 0, -1),
                                     (0, 0, -1), (0, 0, 0), (0, 0, 0), (0, 0, -1)])
        offsets = env.p.new_tensor([(0.3, 0, -0.07), (0.3, 0, -0.6), (0, 2, -1.15), (0, 0, 0.005),
                                    (0, 0, 0.005), (0, 0, 0.005), (0.1, 0, -0.07), (0, 2, 0), (0, 0, 0.005)])
        pc = env.plate_frame()[0]
        env.p[:] = pc + offsets + env.cfg.pad_offset * quat_to_rot(env.q)[:, :, 2]
        env.prev_dist[:] = offsets.norm(dim=1)
        env.prev_d[:] = offsets[:, 2]

        class ConstantHover:
            def act_deterministic(self, obs):
                return env.prev_act

        with patch("train.make_env", return_value=env):
            report = evaluate(ConstantHover(), n=len(reasons), device=DEVICE, fail_breakdown=True)
        self.assertEqual({key: count for key, count in report["fail_breakdown"].items() if count},
                         dict.fromkeys(reasons[:-1], 1))
        self.assertEqual(report["success"]["p"], 0.1111)

    def assert_metrics(self, metrics):
        for key in ("success", "crash", "timeout", "held_if_motors_cut"):
            interval = metrics[key]
            self.assertLessEqual(interval["lo"], interval["p"])
            self.assertLessEqual(interval["p"], interval["hi"])
            self.assertEqual(interval["n"], N)
        n_success = round(metrics["success"]["p"] * N)
        self.assertEqual(sum(metrics["fail_breakdown"].values()), N - n_success)
        for key in ("touch_vn_mean", "touch_vt_mean", "touch_align_deg_mean"):
            self.assertIn(key, metrics)

    def test_fake_checkpoints_baselines_and_report(self):
        torch.manual_seed(447)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            config = {**TRAIN_ENV, "episode_s": 0.2}
            for v2, name in ((False, "p1b_s0"), (True, "p2_perc_s0")):
                run = root / name
                run.mkdir()
                env = make_env(N, 12, v2=v2, device=DEVICE, **config)
                actor = ActorCritic(env.obs_dim, env.act_dim, priv_dim=env.priv_dim)
                torch.save(actor.state_dict(), run / "best.pt")
                (run / "result.json").write_text(json.dumps({"name": name, "seed": 0, "git": "fake",
                                                             "config": {"v2": v2, "env": config}}))
                result = sensitivity(run, n=N, seed=12, device=DEVICE)
                path = write_result(result["name"], result, results)
                self.assertEqual(path.name, f"sens_{name}.json")
                stored = json.loads(path.read_text())
                self.assertTrue(stored["git"])
                self.assertEqual((stored["seed"], stored["eval"]["eval_seed"], stored["eval"]["n"]), (12, 12, N))
                sweeps = stored["eval"]["sweeps"]
                self.assertEqual([point["odo_scale"] for point in sweeps["odo_scale"]], [0, 1, 2, 4] if v2 else [])
                self.assertEqual([point["delay_ms"] for point in sweeps["delay_steps"]],
                                 [[30, 30], [40, 60], [70, 70], [90, 90]])
                self.assertEqual(sweeps["delay_steps"][1]["metrics"], stored["eval"]["nominal"])
                for metrics in (stored["eval"]["nominal"], stored["eval"]["hard"],
                                *(cell["metrics"] for cells in sweeps.values() for cell in cells)):
                    self.assert_metrics(metrics)
                    self.assertEqual("lost_track" in metrics["fail_breakdown"], v2)
                if not v2:
                    self.assertIn("skipped", stored["config"]["odo_status"])

            for result in score_baselines(config, n=N, seed=12, device=DEVICE):
                write_result(result["name"], result, results)
                for case in ("nominal", "hard"):
                    self.assert_metrics(result["eval"][case])
            outputs = generate(results, root / "runs")
            self.assertEqual({path.name for path in outputs},
                             {"TABLES.md", "q1_sensitivity.png", "sens_p1b_s0.png", "sens_p2_perc_s0.png"})
            report = (results / "TABLES.md").read_text(encoding="utf-8")
            for phrase in ("Bỏ qua (v1)", "40–60", "B0 Bay thẳng", "B6 Bay treo", "fail_breakdown"):
                self.assertIn(phrase, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default=DEVICE)
    args, remaining = parser.parse_known_args()
    DEVICE = args.device
    torch.set_num_threads(1)
    unittest.main(argv=[sys.argv[0], *remaining])
