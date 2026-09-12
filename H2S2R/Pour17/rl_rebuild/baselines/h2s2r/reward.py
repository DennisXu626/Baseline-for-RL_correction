"""Method-internal H2S2R tracking reward, separate from benchmark metrics."""

from __future__ import annotations

import torch


def object_tracking_reward(
    object_keypoint_distance: torch.Tensor,
    fingertip_object_distance: torch.Tensor,
    *,
    previous_actions: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    smoothing_weight: float = 0.0,
    close_threshold: float = 0.3,
    stop_reference_threshold: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reproduce the official forced-reference H2S2R reward equations."""

    close = fingertip_object_distance < close_threshold
    tracking = torch.where(
        close,
        torch.exp(-10.0 * object_keypoint_distance),
        torch.zeros_like(object_keypoint_distance),
    )
    tracking = torch.where(
        object_keypoint_distance > stop_reference_threshold,
        0.1 * tracking,
        tracking,
    )
    smoothing = torch.zeros_like(tracking)
    if actions is not None or previous_actions is not None:
        if actions is None or previous_actions is None:
            raise ValueError("actions and previous_actions must be supplied together")
        smoothing = torch.linalg.vector_norm(actions - previous_actions, dim=-1).square()
    total = tracking + float(smoothing_weight) * smoothing
    return total, {
        "object_tracking": tracking,
        "action_smoothing_penalty": smoothing,
        "fingertips_close": close.float(),
    }


def advance_reference_mask(
    object_keypoint_distance: torch.Tensor,
    *,
    stop_reference_threshold: float = 0.2,
) -> torch.Tensor:
    """Official H2S2R pauses the moving goal when tracking error is too large."""

    return object_keypoint_distance <= stop_reference_threshold

