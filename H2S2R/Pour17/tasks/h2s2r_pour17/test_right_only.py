"""CPU tests of actual diagnostic methods, without importing the Isaac app.

The environment class AST is loaded unchanged; only the simulator base class
and tensor state are replaced by a small deterministic fixture.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile

import torch

from rl_rebuild.baselines.h2s2r.contract import ControlledSide, SIDE_ACTION_DIM, SIDE_OBSERVATION_DIM
from rl_rebuild.baselines.h2s2r.pour17.bimanual import (
    SharedReferenceClock, mean_fingertip_object_distance, object_keypoint_distance,
)
from rl_rebuild.baselines.h2s2r.pour17.observation import build_side_observation
from rl_rebuild.baselines.h2s2r.reward import advance_reference_mask, object_tracking_reward


tree = ast.parse(Path(__file__).with_name("right_env.py").read_text())
namespace = dict(globals(), Pour17H2S2REnv=object)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]), "right_env.py", "exec"), namespace)
Env = namespace["RightBottleH2S2REnv"]


def fixture():
    env = Env.__new__(Env)
    env.num_envs, env.device = 2, "cpu"
    env.cfg = SimpleNamespace(
        action_smoothing_weight=0.0, fingertips_close_threshold_m=0.3,
        stop_reference_threshold_m=0.2, success_region_radius_m=0.05,
        table_top_z=0.87, early_reset_if_fingertips_far=True,
        terminate_when_reference_ends=True,
        external_evaluator_controls_termination=False, clip_obs=5.0,
    )
    env.cup = torch.tensor([[0., 0., 0.9, 1., 0., 0., 0.]]).repeat(2, 1)
    env.bottle = env.cup.clone()
    env._object_poses = lambda: (env.cup, env.bottle)
    env._tips = lambda side: env.bottle[:, None, :3].expand(-1, 5, -1) + 0.03
    env._wrist = lambda side: (env.bottle[:, :3], env.bottle[:, 3:])
    env.reference_object_pose = {1: env.bottle[:1].repeat(3, 1)}
    env.reset_object_pose = {1: env.bottle[0].clone()}
    env.clock = SharedReferenceClock.create(
        2, maximum_index=2, device="cpu", control_dt=1 / 30, reference_dt=1 / 30
    )
    env.clock.speed_factors[:] = 1.0
    env._actions = torch.zeros(2, 11)
    env._previous_actions = torch.zeros_like(env._actions)
    env._certification_alpha = torch.zeros(2)
    env._target_q = torch.zeros(2, 58)
    env.side_joint_ids = {ControlledSide.RIGHT: torch.arange(29), ControlledSide.LEFT: torch.arange(29, 58)}
    env.hand = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(2, 58), joint_vel=torch.zeros(2, 58)))
    right = SimpleNamespace(q=torch.zeros(2, 29), qd=torch.zeros(2, 29))
    right.step = lambda action: right.q + action[:, :1]
    left = SimpleNamespace(q=torch.ones(2, 29), qd=torch.zeros(2, 29))
    def forbidden(*args, **kwargs):
        raise AssertionError("The left controller must never step")
    left.step = forbidden
    env.controller = SimpleNamespace(right=right, left=left)
    env._apply_warmup_clamp = lambda: None
    env._warmup_clamp_left = torch.zeros(2, dtype=torch.long)
    env._last_distances = {}
    env._success_run = torch.zeros(2, dtype=torch.long)
    env._success_latched = torch.zeros(2, dtype=torch.bool)
    env.extras = {}
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.max_episode_length = 903
    env._previous_object_pose = {1: env.bottle.clone()}
    env._previous_previous_object_pose = {1: env.bottle.clone()}
    return env


class RightIsolationTests(unittest.TestCase):
    def test_stop_at_saved_boundary_preserves_epoch_counter(self):
        source = ast.parse(Path(__file__).with_name("train_right_lstm.py").read_text())
        class AgentStub:
            def update_epoch(self):
                self.epoch_num += 1
                return self.epoch_num
        scope = {"PpoAgent": AgentStub, "args": SimpleNamespace(diagnostic_stop_epoch=500),
                 "torch": torch, "_slot": None}
        exec(compile(ast.Module(body=[n for n in source.body if isinstance(n, ast.ClassDef)], type_ignores=[]), "train_right_lstm.py", "exec"), scope)
        agent = scope["LoggedPpoAgent"]()
        agent.epoch_num, agent.writer = 499, None
        with tempfile.TemporaryDirectory() as directory:
            agent.nn_dir = Path(directory)
            self.assertEqual(agent.update_epoch(), 500)
            with self.assertRaises(RuntimeError):
                agent.update_epoch()
            (agent.nn_dir / "ep_500_rew_1.0.pth").touch()
            with self.assertRaises(scope["DiagnosticComplete"]):
                agent.update_epoch()
            self.assertEqual(agent.epoch_num, 500)

    def test_left_cannot_change_reward_clock_success_or_done(self):
        baseline, perturbed = fixture(), fixture()
        perturbed.cup[:] = 100.0
        perturbed.cup[:, 2] = -100.0
        perturbed.hand.data.joint_pos[:, 29:] = -7.0
        for _ in range(65):
            torch.testing.assert_close(baseline._get_rewards(), perturbed._get_rewards())
            torch.testing.assert_close(baseline.clock.indices, perturbed.clock.indices)
            torch.testing.assert_close(baseline._success_latched, perturbed._success_latched)
        self.assertTrue(baseline._success_latched.all())
        torch.testing.assert_close(baseline._get_dones()[0], perturbed._get_dones()[0])

    def test_original_reward_no_half_scale_or_event_bonus(self):
        env = fixture()
        env.bottle[1, 0] = 0.25
        expected, _ = object_tracking_reward(torch.tensor([0., 0.25]), torch.full((2,), 0.03 * 3**0.5))
        for _ in range(65):
            torch.testing.assert_close(env._get_rewards(), expected)
        self.assertEqual(env.clock.indices.tolist(), [2, 0])

    def test_warmup_freezes_official_float_clock(self):
        env = fixture()
        env._warmup_clamp_left[:] = 1
        env._get_rewards()
        torch.testing.assert_close(env.clock.float_indices, torch.zeros(2))

    def test_reference_completion_terminates_episode(self):
        env = fixture()
        env.clock.float_indices[:] = 3.0
        self.assertTrue(env._get_dones()[0].all())

    def test_right_drop_and_far_fingertips_still_reset(self):
        env = fixture()
        env.bottle[0, 2] = 0.7
        env._tips = lambda side: torch.ones(2, 5, 3) * 10.0
        self.assertTrue(env._get_dones()[0].all())
        env.episode_length_buf[:] = 902
        self.assertTrue(env._get_dones()[1].all())

    def test_actions_only_step_right_and_hold_left(self):
        env = fixture()
        for action in (torch.ones(2, 11), -torch.ones(2, 11)):
            env._pre_physics_step(action)
            torch.testing.assert_close(env._target_q[:, 29:], env.controller.left.q)
            torch.testing.assert_close(env._target_q[:, :29], env.controller.right.q + action[:, :1])
        with self.assertRaises(ValueError):
            env._pre_physics_step(torch.zeros(2, 22))

    def test_observation_excludes_left_and_cup(self):
        baseline, perturbed = fixture(), fixture()
        perturbed.cup[:] = 100.0
        perturbed.hand.data.joint_pos[:, 29:] = -7.0
        a, b = baseline._get_observations(), perturbed._get_observations()
        self.assertEqual(a["policy"].shape, (2, 171))
        torch.testing.assert_close(a["policy"], b["policy"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
