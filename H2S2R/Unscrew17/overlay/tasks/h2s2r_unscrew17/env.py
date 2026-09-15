"""Direct-58D H2S2R policy surface over the frozen Clip17 Unscrew world."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from isaaclab.utils.math import quat_apply

from rl_rebuild.baselines.h2s2r.contract import (
    ControlledSide,
    fingertip_body_names,
    joint_names_for_side,
)
from rl_rebuild.baselines.h2s2r.pour17.observation import (
    build_bimanual_observation,
    build_side_observation,
)

from .cfg import CRITIC_STATE_DIM, OBSERVATION_DIM, unscrew_task_module


_TASK = unscrew_task_module()
_KEYPOINT_OFFSETS_M = ((0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2))


class Unscrew17H2S2REnv(_TASK.UnscrewEnv):
    """Keep Clip17 physics/reward/curriculum and replace only policy I/O/control."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        names = list(self.hand.joint_names)
        self.side_joint_ids = {
            side: torch.tensor(
                [names.index(name) for name in joint_names_for_side(side)],
                dtype=torch.long,
                device=self.device,
            )
            for side in (ControlledSide.RIGHT, ControlledSide.LEFT)
        }
        body_names = list(self.hand.body_names)
        self.tip_body_ids = {
            side: torch.tensor(
                [body_names.index(name) for name in fingertip_body_names(side)],
                dtype=torch.long,
                device=self.device,
            )
            for side in (ControlledSide.RIGHT, ControlledSide.LEFT)
        }
        self.wrist_body_ids = {
            ControlledSide.RIGHT: body_names.index("right_hand_C_MC"),
            ControlledSide.LEFT: body_names.index("left_hand_C_MC"),
        }
        limits = self.hand.data.joint_pos_limits[0, self.map_ids_t]
        if not bool(torch.isfinite(limits).all()):
            raise RuntimeError("Unscrew17 direct-58D requires finite physical joint limits")
        self.action_lower_q58 = limits[:, 0].clone()
        self.action_upper_q58 = limits[:, 1].clone()
        if not bool((self.action_lower_q58 < self.action_upper_q58).all()):
            raise RuntimeError("Unscrew17 has an invalid physical joint limit")

        n = self.num_envs
        current = self.hand.data.joint_pos[:, self.map_ids_t]
        self.policy_center_q58 = current.clone()
        self._target_q58 = current.clone()
        self._previous_target_q58 = current.clone()
        self._actions = torch.zeros(n, 58, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        with np.load(_TASK.MASTER, allow_pickle=False) as reference:
            bottle = np.concatenate(
                [reference["obj_pos_0"], reference["obj_quat_0"]], axis=1
            )
            cap = np.concatenate(
                [reference["obj_pos_1"], reference["obj_quat_1"]], axis=1
            )
        self.reference_object_pose = {
            0: torch.tensor(bottle, dtype=torch.float32, device=self.device),
            1: torch.tensor(cap, dtype=torch.float32, device=self.device),
        }
        current_bottle, current_cap = self._read_objs()
        self._previous_object_pose = {
            0: current_bottle.clone(),
            1: current_cap.clone(),
        }
        self._previous_previous_object_pose = {
            0: current_bottle.clone(),
            1: current_cap.clone(),
        }

    def _decode_direct_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Map each env's -1/0/+1 to lower/reset-center/upper joint targets."""

        if actions.shape != (self.num_envs, 58):
            raise ValueError(f"actions must be ({self.num_envs}, 58), got {actions.shape}")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("actions must be finite")
        center = self.policy_center_q58
        lower = self.action_lower_q58.unsqueeze(0)
        upper = self.action_upper_q58.unsqueeze(0)
        if not bool(((center >= lower) & (center <= upper)).all()):
            raise RuntimeError("an Unscrew17 reset center lies outside physical joint limits")
        clipped = actions.clamp(-1.0, 1.0)
        decoded = center + torch.where(
            clipped <= 0.0,
            clipped * (center - lower),
            clipped * (upper - center),
        )
        return torch.maximum(torch.minimum(decoded, upper), lower)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1.0, 1.0))
        self._previous_target_q58.copy_(self._target_q58)
        self._target_q58.copy_(self._decode_direct_targets(self._actions))
        self.last_act.copy_(self._actions)
        # These quantities drive the unchanged analytic screw physics and task
        # accounting.  They do not modify the direct joint target.
        self._update_screw_drive_gain()
        self._ff = self._ff_row(self.row.clamp(max=self.T_ROW - 1))
        self.q_tgt = self._target_q58

    def _apply_action(self) -> None:
        self.hand.set_joint_position_target(self._target_q58, joint_ids=self.map_ids)
        self._SA.apply_screw(self)

    def _object_keypoints(self, pose_wxyz: torch.Tensor) -> torch.Tensor:
        offsets = torch.tensor(
            _KEYPOINT_OFFSETS_M, dtype=pose_wxyz.dtype, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1, -1)
        quat = pose_wxyz[:, 3:].unsqueeze(1).expand(-1, 3, -1)
        return (
            pose_wxyz[:, :3].unsqueeze(1) + quat_apply(quat, offsets)
        ).reshape(self.num_envs, 9)

    def _goal_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.row.clamp(min=0, max=self.T_ROW - 1)
        return self.reference_object_pose[0][row], self.reference_object_pose[1][row]

    def _tips(self, side: ControlledSide) -> torch.Tensor:
        return (
            self.hand.data.body_pos_w[:, self.tip_body_ids[side]]
            - self.scene.env_origins.unsqueeze(1)
        )

    def _wrist(self, side: ControlledSide) -> tuple[torch.Tensor, torch.Tensor]:
        body_id = self.wrist_body_ids[side]
        return (
            self.hand.data.body_pos_w[:, body_id] - self.scene.env_origins,
            self.hand.data.body_quat_w[:, body_id],
        )

    @staticmethod
    def _contract_columns(side: ControlledSide) -> tuple[int, ...]:
        if side is ControlledSide.RIGHT:
            return tuple(range(0, 7)) + tuple(range(14, 36))
        return tuple(range(7, 14)) + tuple(range(36, 58))

    def _critic_state(
        self,
        policy_observation: torch.Tensor,
        bottle: torch.Tensor,
        cap: torch.Tensor,
        goal_bottle: torch.Tensor,
        goal_cap: torch.Tensor,
    ) -> torch.Tensor:
        sensors = self._all_sensors[:10]
        net_forces = torch.cat(
            [sensor.data.net_forces_w.reshape(self.num_envs, -1, 3).sum(1) for sensor in sensors],
            dim=-1,
        )
        filtered_forces = torch.cat(
            [sensor.data.force_matrix_w.reshape(self.num_envs, -1, 3).sum(1) for sensor in sensors],
            dim=-1,
        )
        max_angle = 2.0 * np.pi * float(self.screw_spec.turns)
        released = self.screw_has_depth & ~self.screw_engaged
        screw_state = torch.stack(
            [
                (self.screw_angle / max_angle).clamp(0.0, 1.0),
                self.screw_engaged.float(),
                self.screw_has_depth.float(),
                released.float(),
            ],
            dim=-1,
        )
        state = torch.cat(
            [
                policy_observation,
                self._object_keypoints(bottle),
                self._object_keypoints(cap),
                self._object_keypoints(goal_bottle),
                self._object_keypoints(goal_cap),
                self.object.data.root_lin_vel_w,
                self.object.data.root_ang_vel_w,
                self.aux.data.root_lin_vel_w,
                self.aux.data.root_ang_vel_w,
                self.episode_length_buf.float().unsqueeze(-1),
                self.hand.data.applied_torque[:, self.map_ids_t],
                net_forces,
                filtered_forces,
                screw_state,
            ],
            dim=-1,
        ).nan_to_num(0.0)
        if state.shape != (self.num_envs, CRITIC_STATE_DIM):
            raise RuntimeError(
                f"Unscrew17 asymmetric critic must be {CRITIC_STATE_DIM}D, got {state.shape}"
            )
        return state

    def _get_observations(self) -> dict[str, torch.Tensor]:
        bottle, cap = self._read_objs()
        goal_bottle, goal_cap = self._goal_poses()
        observations = {}
        for side, object_pose, goal_pose, object_id in (
            (ControlledSide.RIGHT, cap, goal_cap, 1),
            (ControlledSide.LEFT, bottle, goal_bottle, 0),
        ):
            joint_ids = self.side_joint_ids[side]
            columns = torch.tensor(
                self._contract_columns(side), dtype=torch.long, device=self.device
            )
            wrist_position, wrist_quaternion = self._wrist(side)
            observations[side] = build_side_observation(
                q=self.hand.data.joint_pos[:, joint_ids],
                qd=self.hand.data.joint_vel[:, joint_ids],
                fingertip_positions=self._tips(side),
                palm_position=wrist_position,
                palm_quaternion_wxyz=wrist_quaternion,
                object_pose_wxyz=object_pose,
                goal_object_pose_wxyz=goal_pose,
                previous_object_pose_wxyz=self._previous_object_pose[object_id],
                previous_previous_object_pose_wxyz=self._previous_previous_object_pose[object_id],
                command_q=self._target_q58[:, columns],
                command_qd=(self._target_q58[:, columns] - self._previous_target_q58[:, columns])
                / (self.cfg.sim.dt * self.cfg.decimation),
            )
        policy = build_bimanual_observation(
            observations[ControlledSide.RIGHT], observations[ControlledSide.LEFT]
        ).clamp(-self.cfg.clip_obs, self.cfg.clip_obs).nan_to_num(0.0)
        if policy.shape != (self.num_envs, OBSERVATION_DIM):
            raise RuntimeError(f"Unscrew17 actor must be 342D, got {policy.shape}")
        critic = self._critic_state(
            policy, bottle, cap, goal_bottle, goal_cap
        )
        for object_id, pose in ((0, bottle), (1, cap)):
            self._previous_previous_object_pose[object_id].copy_(
                self._previous_object_pose[object_id]
            )
            self._previous_object_pose[object_id].copy_(pose)
        return {
            "policy": policy,
            "critic": critic,
            "priv_info": torch.empty(self.num_envs, 0, device=self.device),
        }

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        if not hasattr(self, "policy_center_q58"):
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return
        reset_q = self.hand.data.joint_pos[env_ids][:, self.map_ids_t]
        self.policy_center_q58[env_ids] = reset_q
        self._target_q58[env_ids] = reset_q
        self._previous_target_q58[env_ids] = reset_q
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        bottle, cap = self._read_objs()
        for object_id, pose in ((0, bottle), (1, cap)):
            self._previous_object_pose[object_id][env_ids] = pose[env_ids]
            self._previous_previous_object_pose[object_id][env_ids] = pose[env_ids]
