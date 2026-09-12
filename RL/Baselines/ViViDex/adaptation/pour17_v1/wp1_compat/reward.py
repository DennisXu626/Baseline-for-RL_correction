"""Torch port of the pinned ViViDex ObjectMimic reward terms."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def quaternion_distance_wxyz(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Shortest SO(3) angle in radians; q and -q are identical."""
    dot = torch.abs(torch.sum(actual * target, dim=-1)).clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


@dataclass(frozen=True)
class RewardConfig:
    hand_error_scale: float = 10.0
    object_error_scale: float = 50.0
    object_reward_scale: float = 10.0
    lift_bonus_threshold_m: float = 0.02
    lift_bonus_magnitude: float = 2.5
    hand_bonus_object_tolerance_m: float = 0.01
    contact_force_threshold_n: float = 0.5
    object_termination_m: float = 0.25


def object_mimic_bimanual(
    *,
    actual_hand_points: torch.Tensor,
    target_hand_points: torch.Tensor,
    actual_object_pose: torch.Tensor,
    target_object_pose: torch.Tensor,
    fingertip_contacts: torch.Tensor,
    object_table_contacts: torch.Tensor,
    pregrasp_success: torch.Tensor,
    initial_object_z: torch.Tensor,
    config: RewardConfig = RewardConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute real-state reward for [right,bottle] and [left,cup].

    Shapes are (N,2,10,3), (N,2,7), and (N,2,5).  The two side rewards are
    averaged.  No target quantity is substituted for a measured state.
    """
    hand_error = torch.linalg.vector_norm(
        actual_hand_points - target_hand_points, dim=-1
    ).mean(dim=-1)
    object_position_error = torch.linalg.vector_norm(
        actual_object_pose[..., :3] - target_object_pose[..., :3], dim=-1
    )
    object_rotation_error = quaternion_distance_wxyz(
        actual_object_pose[..., 3:], target_object_pose[..., 3:]
    ) / torch.pi
    pregrasp_reward = 10.0 * torch.exp(-config.hand_error_scale * hand_error)

    any_contact = fingertip_contacts.any(dim=-1)
    contact_reward = 0.5 * fingertip_contacts.float().sum(dim=-1)
    object_reward = config.object_reward_scale * torch.exp(
        -config.object_error_scale
        * (object_position_error + 0.1 * object_rotation_error)
    )
    lifted_target = target_object_pose[..., 2] >= (
        initial_object_z + config.lift_bonus_threshold_m
    )
    lifted_actual = actual_object_pose[..., 2] >= (
        initial_object_z + config.lift_bonus_threshold_m
    )
    lift_bonus = config.lift_bonus_magnitude * (
        lifted_target & lifted_actual & ~object_table_contacts
    ).float()
    hand_bonus_gate = object_position_error.square() < (
        config.hand_bonus_object_tolerance_m ** 2
    )
    hand_reward = 4.0 * torch.exp(-config.hand_error_scale * hand_error)
    manipulation_reward = any_contact.float() * (
        contact_reward + object_reward + lift_bonus
        + hand_bonus_gate.float() * hand_reward
    )
    side_reward = torch.where(
        pregrasp_success[:, None], manipulation_reward, pregrasp_reward
    )
    return side_reward.mean(dim=1), {
        "hand_error_m": hand_error,
        "object_position_error_m": object_position_error,
        "object_rotation_error_over_pi": object_rotation_error,
        "contact_reward": contact_reward,
        "object_reward": object_reward,
        "lift_bonus": lift_bonus,
        "hand_reward": hand_reward * hand_bonus_gate.float(),
        "side_reward": side_reward,
    }

