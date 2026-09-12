"""Five-dimensional Sharpa hand-synergy support for the H2S2R adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contract import FINGERS, HAND_DOF_DIM


@dataclass(frozen=True)
class HandSynergy:
    """Frozen linear map from 22 hand joints to five H2S2R action coordinates."""

    matrix: np.ndarray
    center: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    source: str

    def validate(self) -> None:
        if self.matrix.shape != (5, HAND_DOF_DIM):
            raise ValueError(f"matrix must be (5, 22), got {self.matrix.shape}")
        if self.center.shape != (HAND_DOF_DIM,):
            raise ValueError(f"center must be (22,), got {self.center.shape}")
        if self.minimum.shape != (5,) or self.maximum.shape != (5,):
            raise ValueError("minimum and maximum must both be (5,)")
        if np.linalg.matrix_rank(self.matrix) != 5:
            raise ValueError("synergy matrix must have rank 5")
        if not np.all(self.maximum > self.minimum):
            raise ValueError("each synergy maximum must be greater than its minimum")
        for name, value in (
            ("matrix", self.matrix),
            ("center", self.center),
            ("minimum", self.minimum),
            ("maximum", self.maximum),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")


def analytic_finger_synergy(hand_joint_names: tuple[str, ...]) -> HandSynergy:
    """Build a deterministic one-coordinate-per-finger fallback.

    This is for adapter bring-up and smoke tests. Production experiments should
    fit and freeze a PCA basis from the declared Sharpa retarget corpus with
    :func:`fit_pca_synergy`, then record the resulting file hash.
    """

    if len(hand_joint_names) != HAND_DOF_DIM:
        raise ValueError(f"expected 22 hand joints, got {len(hand_joint_names)}")
    matrix = np.zeros((5, HAND_DOF_DIM), dtype=np.float32)
    for row, finger in enumerate(FINGERS):
        # Flexion joints carry the closure coordinate; ab/adduction stays under
        # the c-space posture attractor, matching H2S2R's low-dimensional intent.
        ids = [
            i
            for i, name in enumerate(hand_joint_names)
            if f"_{finger}_" in name and not name.endswith("_AA")
        ]
        if not ids:
            raise ValueError(f"no flexion joints found for {finger}")
        matrix[row, ids] = 1.0 / np.sqrt(len(ids))

    synergy = HandSynergy(
        matrix=matrix,
        center=np.zeros(HAND_DOF_DIM, dtype=np.float32),
        minimum=np.full(5, -0.25, dtype=np.float32),
        maximum=np.full(5, 2.5, dtype=np.float32),
        source="analytic_per_finger_bringup",
    )
    synergy.validate()
    return synergy


def fit_pca_synergy(
    samples: np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    source: str = "sharpa_retarget_pca",
) -> HandSynergy:
    """Fit a frozen five-component PCA map from Sharpa joint samples."""

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != HAND_DOF_DIM:
        raise ValueError(f"samples must be (N, 22), got {samples.shape}")
    if samples.shape[0] < 6:
        raise ValueError("at least six samples are required to fit five components")
    if not np.isfinite(samples).all():
        raise ValueError("samples contain non-finite values")
    center = samples.mean(axis=0)
    _, _, vt = np.linalg.svd(samples - center, full_matrices=False)
    matrix = vt[:5]
    projected = (samples - center) @ matrix.T
    minimum = np.percentile(projected, lower_percentile, axis=0)
    maximum = np.percentile(projected, upper_percentile, axis=0)
    synergy = HandSynergy(
        matrix=matrix.astype(np.float32),
        center=center.astype(np.float32),
        minimum=minimum.astype(np.float32),
        maximum=maximum.astype(np.float32),
        source=source,
    )
    synergy.validate()
    return synergy


def save_synergy(path: str | Path, synergy: HandSynergy) -> None:
    synergy.validate()
    np.savez_compressed(
        path,
        matrix=synergy.matrix,
        center=synergy.center,
        minimum=synergy.minimum,
        maximum=synergy.maximum,
        source=np.asarray(synergy.source),
    )


def load_synergy(path: str | Path) -> HandSynergy:
    with np.load(path, allow_pickle=False) as data:
        synergy = HandSynergy(
            matrix=np.asarray(data["matrix"], dtype=np.float32),
            center=np.asarray(data["center"], dtype=np.float32),
            minimum=np.asarray(data["minimum"], dtype=np.float32),
            maximum=np.asarray(data["maximum"], dtype=np.float32),
            source=str(data["source"].item()),
        )
    synergy.validate()
    return synergy

