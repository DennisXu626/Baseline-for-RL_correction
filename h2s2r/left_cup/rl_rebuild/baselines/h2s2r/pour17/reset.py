"""Reset contracts shared with the current Pour17 simulation.

Only the single pre-manipulation row is imported from the frozen ours tape.
No robot reference trajectory, reward weight, or confidence value is exposed to
the H2S2R policy or reward.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Phase1Reset:
    joint_q_rad: np.ndarray
    joint_qd_rad_s: np.ndarray
    object_0_pose_wxyz: np.ndarray
    object_1_pose_wxyz: np.ndarray
    interaction_row: int
    reference_path: Path
    reference_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_frozen_reference(root: Path, expected_sha256: str) -> Path:
    candidates = sorted((root / "reference").glob("*.npz"))
    for candidate in candidates:
        if _sha256(candidate) == expected_sha256:
            return candidate
    raise FileNotFoundError(
        "no reference tape in bundle/reference matches world_manifest sha256 "
        f"{expected_sha256}"
    )


def load_phase1_g2_reset(bundle_root: str | Path) -> Phase1Reset:
    """Load the deterministic already-at-grasp Pour17 phase-one reset.

    This mirrors ours' ``g2`` RSI entry: robot q is the interaction-first row,
    both objects use the bundle's physically corrected initial poses, all
    velocities are zero, and reset randomization is disabled.
    """

    root = Path(bundle_root).resolve()
    canonical = json.loads(
        (root / "world/canonical_reset_v1.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "world/world_manifest.json").read_text(encoding="utf-8")
    )
    expected_sha256 = str(manifest["reference_tape"]["sha256"])
    reference_path = _resolve_frozen_reference(root, expected_sha256)

    with np.load(reference_path, allow_pickle=True) as archive:
        names = [str(value) for value in archive["seg_names"].tolist()]
        lengths = np.asarray(archive["seg_lens"], dtype=np.int64)
        if names[:2] != ["approach", "seam1"] or len(lengths) != len(names):
            raise ValueError(f"unexpected Pour17 segment contract: {names}")
        interaction_row = int(lengths[0] + lengths[1])
        expected_row = int(
            canonical["objects_initial_state"]["interaction_first_row_index"]
        )
        if interaction_row != expected_row:
            raise ValueError(
                f"reference interaction row {interaction_row} != reset manifest "
                f"row {expected_row}"
            )
        q58 = np.concatenate(
            [
                np.asarray(archive["right_q"][interaction_row], np.float32),
                np.asarray(archive["left_q"][interaction_row], np.float32),
                np.asarray(archive["right_f"][interaction_row], np.float32),
                np.asarray(archive["left_f"][interaction_row], np.float32),
            ]
        )
    if q58.shape != (58,) or not np.isfinite(q58).all():
        raise ValueError("phase-one reset must contain 58 finite joint positions")

    objects = canonical["objects_initial_state"]
    object_0 = np.asarray(
        objects["object_0_cup"]["pose_xyz_wxyz_env_frame"], np.float32
    )
    object_1 = np.asarray(
        objects["object_1_bottle"]["pose_xyz_wxyz_env_frame"], np.float32
    )
    if object_0.shape != (7,) or object_1.shape != (7,):
        raise ValueError("phase-one object reset poses must each have seven values")

    return Phase1Reset(
        joint_q_rad=q58,
        joint_qd_rad_s=np.zeros(58, dtype=np.float32),
        object_0_pose_wxyz=object_0,
        object_1_pose_wxyz=object_1,
        interaction_row=interaction_row,
        reference_path=reference_path,
        reference_sha256=expected_sha256,
    )
