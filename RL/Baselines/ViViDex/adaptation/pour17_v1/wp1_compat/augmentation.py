"""Faithful, global Pour reference placement augmentation for the dual-hand task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlacementSample:
    stage: int
    translation_xy_m: tuple[float, float]
    yaw_rad: float = 0.0


def sample_placement(stage: int, rng: np.random.Generator) -> PlacementSample:
    """Map the upstream Pour stage ranges without adding an unobserved yaw.

    The source Pour branch has yaw fixed to zero at stages 0, 1, and 2.
    Stage 0 is identity.  Stages 1/2 share x in [-0.06, 0.04] around the
    source's 0.22 m task anchor and y in [0, 0.1].  Because this fixed-base
    adaptation already contains its absolute world candidate, only the sampled
    deltas are applied here, jointly to both hands and both objects.
    """
    if stage not in (0, 1, 2):
        raise ValueError("curriculum stage must be 0, 1, or 2")
    if stage == 0:
        return PlacementSample(stage=0, translation_xy_m=(0.0, 0.0))
    return PlacementSample(
        stage=stage,
        translation_xy_m=(float(rng.uniform(-0.06, 0.04)),
                          float(rng.uniform(0.0, 0.1))),
    )


def apply_global_placement(reference: dict[str, np.ndarray],
                           sample: PlacementSample) -> dict[str, np.ndarray]:
    """Apply one common XY offset to all task-space reference quantities."""
    dx, dy = sample.translation_xy_m
    output = {name: np.asarray(value).copy() for name, value in reference.items()}
    for name in ("right_wrist_pose_wxyz", "left_wrist_pose_wxyz"):
        if name in output:
            output[name][..., 0] += dx
            output[name][..., 1] += dy
    if "object_pose_wxyz" in output:
        output["object_pose_wxyz"][..., 0] += dx
        output["object_pose_wxyz"][..., 1] += dy
    for name in ("right_hand_target_m", "left_hand_target_m"):
        if name in output:
            output[name][..., 0] += dx
            output[name][..., 1] += dy
    return output

