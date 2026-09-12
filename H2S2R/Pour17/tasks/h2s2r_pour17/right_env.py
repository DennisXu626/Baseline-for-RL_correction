"""Right-bottle diagnostic; the copied bimanual scene and reset stay unchanged.

Only the right controller receives policy actions.  The left controller remains
at its reset configuration and supplies a fixed PD target, not a welded object.
This is a single-hand diagnostic, not a bimanual Pour17 benchmark result.
"""

import torch

from rl_rebuild.baselines.h2s2r.contract import (
    ControlledSide, SIDE_ACTION_DIM, SIDE_OBSERVATION_DIM,
)
from rl_rebuild.baselines.h2s2r.pour17.bimanual import (
    mean_fingertip_object_distance, object_keypoint_distance,
)
from rl_rebuild.baselines.h2s2r.pour17.observation import build_side_observation
from rl_rebuild.baselines.h2s2r.reward import object_tracking_reward
from .cfg import build_cfg as build_bimanual_cfg
from .env import Pour17H2S2REnv
from .right_workspace import apply_centered_bounds
from .ours_physics import configure_physics, apply_object_physics, verify_runtime


def build_cfg(**kwargs):
    if kwargs.get("reference_start_index") != 14:
        raise ValueError("Right-only v8 requires the independently selected start frame 14")
    cfg = build_bimanual_cfg(**kwargs)
    if cfg.input_regime != "estimated":
        raise ValueError("Right-only diagnostic accepts Estimated input only")
    if cfg.joint_progress_reward_weight or cfg.joint_success_reward_weight:
        raise ValueError("Right-only diagnostic forbids added progress/success rewards")
    cfg.action_space = SIDE_ACTION_DIM
    cfg.observation_space = SIDE_OBSERVATION_DIM
    return configure_physics(cfg)


class RightBottleH2S2REnv(Pour17H2S2REnv):
    """H2S2R single-side policy in the unchanged two-hand physical scene."""

    def _setup_scene(self):
        super()._setup_scene()
        apply_object_physics(self.cfg)

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # V8: right near-grasp is normalized zero; absolute targets stay bounded.
        apply_centered_bounds(self.controller.right)
        self.physics_runtime_record = verify_runtime(self)
        # The parent allocates bimanual buffers; only this diagnostic replaces
        # them. The original environment module remains byte-for-byte intact.
        self._actions = torch.zeros(self.num_envs, SIDE_ACTION_DIM, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

    def _pre_physics_step(self, actions):
        if actions.shape != self._actions.shape:
            raise ValueError(f"Expected right actions {self._actions.shape}, got {actions.shape}")
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1.0, 1.0))
        # No bimanual G2 certification override is used by this diagnostic.
        self._certification_alpha.zero_()
        # Set the normalized policy target once per control.  The inherited
        # _apply_action advances right FABRICS and submits its position target
        # once for each of the 12 physics substeps.
        self.controller.right.set_targets(self._actions)
        self._target_q.copy_(self.hand.data.joint_pos)
        self._target_q[:, self.side_joint_ids[ControlledSide.LEFT]] = self.controller.left.q

    def _right_quantities(self):
        _, bottle = self._object_poses()
        return {
            "bottle_error": object_keypoint_distance(
                bottle, self.reference_object_pose[1][self.clock.indices]
            ),
            "right_tip_distance": mean_fingertip_object_distance(
                self._tips(ControlledSide.RIGHT), bottle[:, :3]
            ),
        }

    def _get_rewards(self):
        warmup_active = self._warmup_clamp_left > 0
        self._apply_warmup_clamp()
        values = self._right_quantities()
        reward, terms = object_tracking_reward(
            values["bottle_error"], values["right_tip_distance"],
            previous_actions=self._previous_actions, actions=self._actions,
            smoothing_weight=self.cfg.action_smoothing_weight,
            close_threshold=self.cfg.fingertips_close_threshold_m,
            stop_reference_threshold=self.cfg.stop_reference_threshold_m,
        )
        self._last_distances[1] = values["bottle_error"]
        clock_updated = self.clock.step_single(
            values["bottle_error"],
            frozen=warmup_active,
            stop_reference_threshold=self.cfg.stop_reference_threshold_m,
        )
        # Training ends with the released H2S2R clock.  This instantaneous
        # final-pose check is diagnostic only; formal Pour success remains the
        # shared external evaluator and is never added to the reward.
        _, bottle = self._object_poses()
        final_error = object_keypoint_distance(
            bottle, self.reference_object_pose[1][-1].expand(self.num_envs, -1)
        )
        in_success = self.clock.is_complete & (
            final_error < self.cfg.success_region_radius_m
        )
        self._success_latched |= in_success
        left_ids = self.side_joint_ids[ControlledSide.LEFT]
        left_deviation = (self.hand.data.joint_pos[:, left_ids] - self.controller.left.q).abs()
        self.extras.update({
            "success_rate": self._success_latched.float().mean(),
            "h2s2r/reference_end_within_5cm_rate": in_success.float().mean(),
            "h2s2r/reference_index_mean": self.clock.indices.float().mean(),
            "h2s2r/reference_float_index_mean": self.clock.float_indices.mean(),
            "h2s2r/reference_speed_factor_mean": self.clock.speed_factors.mean(),
            "h2s2r/warmup_frozen_rate": warmup_active.float().mean(),
            "h2s2r/bottle_tracking_reward": terms["object_tracking"].mean(),
            "h2s2r/bottle_tracking_error_m": values["bottle_error"].mean(),
            "h2s2r/right_fingertip_distance_m": values["right_tip_distance"].mean(),
            "h2s2r/right_clock_update_rate": clock_updated.float().mean(),
            "h2s2r/bottle_height_from_reset_m": (
                bottle[:, 2] - self.reset_object_pose[1][2]
            ).mean(),
            "h2s2r/left_hold_max_deviation_rad": left_deviation.max(),
        })
        return reward

    def _get_dones(self):
        if self.cfg.external_evaluator_controls_termination:
            never = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return never, never
        _, bottle = self._object_poses()
        object_fell_off_table = bottle[:, 2] < self.cfg.table_top_z - 0.05
        fingertips_far = self._right_quantities()["right_tip_distance"] >= (
            self.cfg.fingertips_close_threshold_m
        )
        reference_done = self.clock.is_complete
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = object_fell_off_table
        if self.cfg.early_reset_if_fingertips_far:
            terminated |= fingertips_far
        if self.cfg.terminate_when_reference_ends:
            terminated |= reference_done
        self.extras.update({
            "h2s2r/reset_object_fell_off_table_rate": object_fell_off_table.float().mean(),
            "h2s2r/reset_fingertips_far_rate": fingertips_far.float().mean(),
            "h2s2r/reset_reference_done_rate": reference_done.float().mean(),
            "h2s2r/reset_timeout_rate": time_out.float().mean(),
            "h2s2r/episode_step_mean": self.episode_length_buf.float().mean(),
        })
        return terminated, time_out

    def _get_observations(self):
        _, bottle = self._object_poses()
        side = ControlledSide.RIGHT
        ids = self.side_joint_ids[side]
        wrist_position, wrist_quaternion = self._wrist(side)
        result = build_side_observation(
            q=self.hand.data.joint_pos[:, ids], qd=self.hand.data.joint_vel[:, ids],
            fingertip_positions=self._tips(side), palm_position=wrist_position,
            palm_quaternion_wxyz=wrist_quaternion, object_pose_wxyz=bottle,
            goal_object_pose_wxyz=self.reference_object_pose[1][self.clock.indices],
            previous_object_pose_wxyz=self._previous_object_pose[1],
            previous_previous_object_pose_wxyz=self._previous_previous_object_pose[1],
            fabric_q=self.controller.right.q, fabric_qd=self.controller.right.qd,
        ).clamp(-self.cfg.clip_obs, self.cfg.clip_obs).nan_to_num(0.0)
        self._previous_previous_object_pose[1].copy_(self._previous_object_pose[1])
        self._previous_object_pose[1].copy_(bottle)
        return {"policy": result, "priv_info": torch.empty(self.num_envs, 0, device=self.device)}
