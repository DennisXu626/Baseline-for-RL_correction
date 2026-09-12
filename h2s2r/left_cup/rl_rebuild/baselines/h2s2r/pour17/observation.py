"""Official H2S2R observation order, duplicated symmetrically for two sides."""

from __future__ import annotations

import torch

from ..contract import (
    BIMANUAL_OBSERVATION_DIM,
    FABRIC_DOF_DIM,
    NUM_FINGERTIPS,
    SIDE_OBSERVATION_DIM,
)


def _wxyz_to_xyzw(pose_wxyz: torch.Tensor) -> torch.Tensor:
    return torch.cat([pose_wxyz[:, :3], pose_wxyz[:, 4:], pose_wxyz[:, 3:4]], dim=-1)


def _quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    q_xyz = q[:, 1:]
    qw = q[:, :1]
    uv = torch.cross(q_xyz, vector, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return vector + 2.0 * (qw * uv + uuv)


def palm_helper_points(
    palm_position: torch.Tensor,
    palm_quaternion_wxyz: torch.Tensor,
    *,
    axis_length_m: float = 0.25,
) -> torch.Tensor:
    """Return palm origin followed by its +x, +y and +z helper points."""

    n = len(palm_position)
    axes = torch.eye(3, dtype=palm_position.dtype, device=palm_position.device)
    axes = axes.unsqueeze(0).expand(n, -1, -1) * float(axis_length_m)
    q = palm_quaternion_wxyz.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4)
    rotated = _quat_apply_wxyz(q, axes.reshape(-1, 3)).reshape(n, 3, 3)
    points = torch.cat([palm_position.unsqueeze(1), palm_position.unsqueeze(1) + rotated], dim=1)
    return points.reshape(n, 12)


def build_side_observation(
    *,
    q: torch.Tensor,
    qd: torch.Tensor,
    fingertip_positions: torch.Tensor,
    palm_position: torch.Tensor,
    palm_quaternion_wxyz: torch.Tensor,
    object_pose_wxyz: torch.Tensor,
    goal_object_pose_wxyz: torch.Tensor,
    previous_object_pose_wxyz: torch.Tensor,
    previous_previous_object_pose_wxyz: torch.Tensor,
    fabric_q: torch.Tensor,
    fabric_qd: torch.Tensor,
) -> torch.Tensor:
    n = q.shape[0]
    expected_matrix = (n, FABRIC_DOF_DIM)
    for name, value in (("q", q), ("qd", qd), ("fabric_q", fabric_q), ("fabric_qd", fabric_qd)):
        if value.shape != expected_matrix:
            raise ValueError(f"{name} must be {expected_matrix}, got {value.shape}")
    if fingertip_positions.shape != (n, NUM_FINGERTIPS, 3):
        raise ValueError(
            f"fingertip_positions must be ({n}, {NUM_FINGERTIPS}, 3), "
            f"got {fingertip_positions.shape}"
        )
    poses = (
        object_pose_wxyz,
        goal_object_pose_wxyz,
        previous_object_pose_wxyz,
        previous_previous_object_pose_wxyz,
    )
    if any(value.shape != (n, 7) for value in poses):
        raise ValueError("all object poses must be (N, 7)")
    observation = torch.cat(
        [
            q,
            qd,
            fingertip_positions.reshape(n, -1),
            palm_helper_points(palm_position, palm_quaternion_wxyz),
            *(_wxyz_to_xyzw(value) for value in poses),
            fabric_q,
            fabric_qd,
        ],
        dim=-1,
    )
    if observation.shape != (n, SIDE_OBSERVATION_DIM):
        raise AssertionError(
            f"side observation is {observation.shape}, expected ({n}, {SIDE_OBSERVATION_DIM})"
        )
    return observation


def build_bimanual_observation(
    right_observation: torch.Tensor,
    left_observation: torch.Tensor,
) -> torch.Tensor:
    if right_observation.shape != left_observation.shape:
        raise ValueError("right and left observations must have the same shape")
    if right_observation.ndim != 2 or right_observation.shape[1] != SIDE_OBSERVATION_DIM:
        raise ValueError(f"each side observation must have width {SIDE_OBSERVATION_DIM}")
    result = torch.cat([right_observation, left_observation], dim=-1)
    if result.shape[1] != BIMANUAL_OBSERVATION_DIM:
        raise AssertionError("bimanual observation contract changed")
    return result
