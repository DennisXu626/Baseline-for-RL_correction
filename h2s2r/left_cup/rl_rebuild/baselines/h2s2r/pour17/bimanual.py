"""Bimanual extensions of H2S2R's original single-object reward and clock."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..reward import advance_reference_mask, object_tracking_reward


OBJECT_KEYPOINT_OFFSETS_M = (
    (0.2, 0.0, 0.0),
    (0.0, 0.2, 0.0),
    (0.0, 0.0, 0.2),
)


def _quat_apply_wxyz(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    q_xyz = q[..., 1:]
    qw = q[..., :1]
    uv = torch.cross(q_xyz, vectors, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return vectors + 2.0 * (qw * uv + uuv)


def object_keypoint_distance(
    object_pose_wxyz: torch.Tensor,
    goal_pose_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Official H2S2R mean distance over three 20 cm axis keypoints."""

    if object_pose_wxyz.shape != goal_pose_wxyz.shape:
        raise ValueError("object and goal poses must have the same shape")
    if object_pose_wxyz.ndim != 2 or object_pose_wxyz.shape[1] != 7:
        raise ValueError("poses must have shape (N, 7)")
    n = len(object_pose_wxyz)
    offsets = torch.tensor(
        OBJECT_KEYPOINT_OFFSETS_M,
        dtype=object_pose_wxyz.dtype,
        device=object_pose_wxyz.device,
    ).unsqueeze(0).expand(n, -1, -1)
    oq = object_pose_wxyz[:, 3:].unsqueeze(1).expand(-1, 3, -1)
    gq = goal_pose_wxyz[:, 3:].unsqueeze(1).expand(-1, 3, -1)
    current = object_pose_wxyz[:, :3].unsqueeze(1) + _quat_apply_wxyz(oq, offsets)
    goal = goal_pose_wxyz[:, :3].unsqueeze(1) + _quat_apply_wxyz(gq, offsets)
    return torch.linalg.vector_norm(current - goal, dim=-1).mean(dim=-1)


def mean_fingertip_object_distance(
    fingertip_positions: torch.Tensor,
    object_position: torch.Tensor,
) -> torch.Tensor:
    if fingertip_positions.ndim != 3 or fingertip_positions.shape[-1] != 3:
        raise ValueError("fingertip_positions must have shape (N, F, 3)")
    return torch.linalg.vector_norm(
        fingertip_positions - object_position.unsqueeze(1), dim=-1
    ).mean(dim=-1)


def shared_reference_advance_mask(
    object_0_keypoint_distance: torch.Tensor,
    object_1_keypoint_distance: torch.Tensor,
    *,
    stop_reference_threshold: float = 0.2,
) -> torch.Tensor:
    """Advance only when cup and bottle both satisfy H2S2R's threshold."""

    return advance_reference_mask(
        object_0_keypoint_distance,
        stop_reference_threshold=stop_reference_threshold,
    ) & advance_reference_mask(
        object_1_keypoint_distance,
        stop_reference_threshold=stop_reference_threshold,
    )


@dataclass
class SharedReferenceClock:
    """Official H2S2R floating reference clock, with optional bimanual gating.

    The released implementation stores a floating-point index, samples one
    trajectory speed factor per environment in [0.5, 1.0], and converts the
    policy control period to the 30 Hz reference timebase.  ``indices`` is only
    the clamped integer view used to gather the current target pose.
    """

    float_indices: torch.Tensor
    speed_factors: torch.Tensor
    maximum_index: int
    control_dt: float
    reference_dt: float
    min_speed_factor: float = 0.5
    max_speed_factor: float = 1.0

    @property
    def indices(self) -> torch.Tensor:
        return self.float_indices.long().clamp(max=self.maximum_index)

    @property
    def is_complete(self) -> torch.Tensor:
        return self.float_indices.long() > self.maximum_index

    @classmethod
    def create(
        cls,
        num_envs: int,
        maximum_index: int,
        device: str,
        *,
        control_dt: float = 1.0 / 20.0,
        reference_dt: float = 1.0 / 30.0,
    ) -> "SharedReferenceClock":
        if num_envs <= 0 or maximum_index < 1:
            raise ValueError("clock requires at least one environment and two targets")
        if control_dt <= 0.0 or reference_dt <= 0.0:
            raise ValueError("clock time steps must be positive")
        clock = cls(
            float_indices=torch.zeros(num_envs, dtype=torch.float, device=device),
            speed_factors=torch.empty(num_envs, dtype=torch.float, device=device),
            maximum_index=int(maximum_index),
            control_dt=float(control_dt),
            reference_dt=float(reference_dt),
        )
        clock._sample_speed_factors(torch.arange(num_envs, device=device))
        return clock

    def _sample_speed_factors(self, env_ids: torch.Tensor) -> None:
        self.speed_factors[env_ids] = torch.empty(
            len(env_ids), dtype=self.speed_factors.dtype, device=self.speed_factors.device
        ).uniform_(self.min_speed_factor, self.max_speed_factor)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.float_indices[env_ids] = 0.0
        self._sample_speed_factors(env_ids)

    def _advance(self, eligible: torch.Tensor, frozen: torch.Tensor) -> torch.Tensor:
        if eligible.shape != self.float_indices.shape or frozen.shape != eligible.shape:
            raise ValueError("clock masks must match the per-environment clock shape")
        active = eligible & ~frozen & ~self.is_complete
        step_size = self.speed_factors * (self.control_dt / self.reference_dt)
        self.float_indices += torch.where(active, step_size, torch.zeros_like(step_size))
        return active

    def step_single(
        self,
        object_keypoint_distance: torch.Tensor,
        *,
        frozen: torch.Tensor,
        stop_reference_threshold: float = 0.2,
    ) -> torch.Tensor:
        eligible = advance_reference_mask(
            object_keypoint_distance,
            stop_reference_threshold=stop_reference_threshold,
        )
        return self._advance(eligible, frozen)

    def step(
        self,
        object_0_keypoint_distance: torch.Tensor,
        object_1_keypoint_distance: torch.Tensor,
        *,
        frozen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        eligible = shared_reference_advance_mask(
            object_0_keypoint_distance, object_1_keypoint_distance
        )
        if frozen is None:
            frozen = torch.zeros_like(eligible)
        return self._advance(eligible, frozen)


def joint_progress_success_reward(
    shared_reference_advanced: torch.Tensor,
    new_joint_success: torch.Tensor,
    *,
    progress_weight: float,
    success_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reward only progress and first success events that are joint by design."""

    if shared_reference_advanced.dtype is not torch.bool:
        raise TypeError("shared_reference_advanced must be a boolean tensor")
    if new_joint_success.dtype is not torch.bool:
        raise TypeError("new_joint_success must be a boolean tensor")
    if shared_reference_advanced.shape != new_joint_success.shape:
        raise ValueError("joint progress and success masks must have the same shape")
    if progress_weight < 0.0 or success_weight < 0.0:
        raise ValueError("joint reward weights must be non-negative")
    progress = shared_reference_advanced.float() * float(progress_weight)
    success = new_joint_success.float() * float(success_weight)
    return progress + success, {
        "joint_progress": progress,
        "joint_success": success,
    }


def bimanual_tracking_reward(
    object_0_keypoint_distance: torch.Tensor,
    object_1_keypoint_distance: torch.Tensor,
    left_fingertip_object_distance: torch.Tensor,
    right_fingertip_object_distance: torch.Tensor,
    *,
    previous_actions: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    smoothing_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean two unmodified H2S2R side rewards to preserve single-side scale."""

    left, left_terms = object_tracking_reward(
        object_0_keypoint_distance,
        left_fingertip_object_distance,
    )
    right, right_terms = object_tracking_reward(
        object_1_keypoint_distance,
        right_fingertip_object_distance,
    )
    smoothing = torch.zeros_like(left)
    if actions is not None or previous_actions is not None:
        if actions is None or previous_actions is None:
            raise ValueError("actions and previous_actions must be supplied together")
        smoothing = torch.linalg.vector_norm(actions - previous_actions, dim=-1).square()
    total = 0.5 * (left + right) + float(smoothing_weight) * smoothing
    return total, {
        "left_object_tracking": left_terms["object_tracking"],
        "right_object_tracking": right_terms["object_tracking"],
        "left_fingertips_close": left_terms["fingertips_close"],
        "right_fingertips_close": right_terms["fingertips_close"],
        "action_smoothing_penalty": smoothing,
    }
