"""Fail-closed input loader for the Estimated and explicit Oracle regimes.

Calibration can be shared across methods.  Task supervision cannot silently be
upgraded from a model estimate or dataset pseudo-label to ground truth.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class InputRegime(str, Enum):
    ESTIMATED = "estimated"
    ORACLE = "oracle"

    @classmethod
    def parse(cls, value: str | "InputRegime") -> "InputRegime":
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except ValueError as exc:
            raise ValueError("input regime must be 'estimated' or 'oracle'") from exc


class TaskSupervisionKind(str, Enum):
    MODEL_ESTIMATE = "model_estimate"
    DATASET_ANNOTATION = "dataset_annotation"
    SENSOR_MEASURED_GROUND_TRUTH = "sensor_measured_ground_truth"


@dataclass(frozen=True)
class InputProvenance:
    regime: InputRegime
    task_supervision_kind: TaskSupervisionKind
    task_supervision_source: str
    camera_calibration_source: str
    sha256: str


@dataclass(frozen=True)
class PerceptionInput:
    frame_ids: np.ndarray
    source_times_s: np.ndarray
    reconstruction_fps: float
    camera_K: np.ndarray
    camera_to_world: np.ndarray
    left_wrist_pose_wxyz: np.ndarray
    right_wrist_pose_wxyz: np.ndarray
    left_hand_q: np.ndarray
    right_hand_q: np.ndarray
    object_0_pose_wxyz: np.ndarray
    object_1_pose_wxyz: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray
    object_0_valid: np.ndarray
    object_1_valid: np.ndarray
    metadata: dict
    provenance: InputProvenance


_REQUIRED_ARRAYS = (
    "frame_ids",
    "K",
    "c2w",
    "left_wrist_pose_wxyz",
    "right_wrist_pose_wxyz",
    "left_hand_q",
    "right_hand_q",
    "object_0_pose_wxyz",
    "object_1_pose_wxyz",
    "left_valid",
    "right_valid",
    "object_0_valid",
    "object_1_valid",
    "meta",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_common(path: Path) -> tuple[dict[str, np.ndarray], dict, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in _REQUIRED_ARRAYS if name not in archive]
        if missing:
            raise ValueError(f"input archive is missing fields: {missing}")
        values = {name: np.asarray(archive[name]).copy() for name in _REQUIRED_ARRAYS}
    metadata = json.loads(str(values.pop("meta").item()))
    if metadata.get("units") != "meters":
        raise ValueError(f"expected meter units, got {metadata.get('units')!r}")
    if metadata.get("quat") != "wxyz":
        raise ValueError(f"expected wxyz quaternions, got {metadata.get('quat')!r}")
    fps = float(metadata.get("fps_reconstruction", 0.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid reconstruction fps: {fps}")
    n = len(values["frame_ids"])
    if metadata.get("frame_time_mapping") == "one_row_per_source_video_frame":
        source_fps = float(metadata.get("fps_source_video", 0.0))
        source_frame_count = int(metadata.get("source_video_frame_count", -1))
        source_duration_s = float(metadata.get("source_video_duration_s", -1.0))
        if not np.isfinite(source_fps) or source_fps <= 0.0:
            raise ValueError(f"invalid source video fps: {source_fps}")
        if not np.isclose(fps, source_fps, rtol=0.0, atol=1e-9):
            raise ValueError(
                "one-row-per-source-frame input must use the source video fps; "
                f"got reconstruction={fps}, source={source_fps}"
            )
        if source_frame_count != n:
            raise ValueError(
                "source video frame count does not match the perception rows; "
                f"got source={source_frame_count}, rows={n}"
            )
        frame_ids = np.asarray(values["frame_ids"], dtype=np.int64)
        if not np.array_equal(frame_ids, np.arange(n, dtype=np.int64)):
            raise ValueError(
                "one-row-per-source-frame input requires contiguous frame_ids 0..N-1"
            )
        expected_duration_s = n / source_fps
        if not np.isclose(
            source_duration_s, expected_duration_s, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "source video duration is inconsistent with frame count and fps; "
                f"got {source_duration_s}, expected {expected_duration_s}"
            )
    for name in (
        "c2w",
        "left_wrist_pose_wxyz",
        "right_wrist_pose_wxyz",
        "left_hand_q",
        "right_hand_q",
        "object_0_pose_wxyz",
        "object_1_pose_wxyz",
        "left_valid",
        "right_valid",
        "object_0_valid",
        "object_1_valid",
    ):
        if len(values[name]) != n:
            raise ValueError(f"{name} has {len(values[name])} rows, expected {n}")
    for name in (
        "left_wrist_pose_wxyz",
        "right_wrist_pose_wxyz",
        "left_hand_q",
        "right_hand_q",
        "object_0_pose_wxyz",
        "object_1_pose_wxyz",
    ):
        if not np.isfinite(values[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    return values, metadata, fps


def _make_input(
    path: Path,
    values: dict[str, np.ndarray],
    metadata: dict,
    fps: float,
    provenance: InputProvenance,
) -> PerceptionInput:
    frame_ids = values["frame_ids"].astype(np.int64, copy=False)
    source_times = (frame_ids - frame_ids[0]).astype(np.float64) / fps
    return PerceptionInput(
        frame_ids=frame_ids,
        source_times_s=source_times,
        reconstruction_fps=fps,
        camera_K=values["K"].astype(np.float64, copy=False),
        camera_to_world=values["c2w"].astype(np.float32, copy=False),
        left_wrist_pose_wxyz=values["left_wrist_pose_wxyz"].astype(np.float32),
        right_wrist_pose_wxyz=values["right_wrist_pose_wxyz"].astype(np.float32),
        left_hand_q=values["left_hand_q"].astype(np.float32),
        right_hand_q=values["right_hand_q"].astype(np.float32),
        object_0_pose_wxyz=values["object_0_pose_wxyz"].astype(np.float32),
        object_1_pose_wxyz=values["object_1_pose_wxyz"].astype(np.float32),
        left_valid=values["left_valid"].astype(bool),
        right_valid=values["right_valid"].astype(bool),
        object_0_valid=values["object_0_valid"].astype(bool),
        object_1_valid=values["object_1_valid"].astype(bool),
        metadata={**metadata, "loaded_from": str(path)},
        provenance=provenance,
    )


def load_estimated_inputs(path: str | Path) -> PerceptionInput:
    """Load the method-neutral EgoDex perception archive for the main result."""

    path = Path(path)
    values, metadata, fps = _read_common(path)
    provenance = InputProvenance(
        regime=InputRegime.ESTIMATED,
        task_supervision_kind=TaskSupervisionKind.MODEL_ESTIMATE,
        task_supervision_source=(
            "EgoDex reconstruction: estimated wrist/hand state and object 6D pose"
        ),
        camera_calibration_source=(
            "dataset/device camera intrinsics and device-SLAM c2w (shared calibration)"
        ),
        sha256=_sha256(path),
    )
    return _make_input(path, values, metadata, fps, provenance)


def load_oracle_inputs(
    path: str | Path,
    *,
    provenance_manifest: str | Path,
) -> PerceptionInput:
    """Load a sensor-measured Oracle archive, rejecting pseudo-GT by default."""

    path = Path(path)
    manifest_path = Path(provenance_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "h2s2r_input_provenance_v1":
        raise ValueError("Oracle provenance manifest has an unknown schema")
    if manifest.get("regime") != InputRegime.ORACLE.value:
        raise ValueError("Oracle provenance manifest must declare regime='oracle'")
    task = manifest.get("task_supervision", {})
    kind = TaskSupervisionKind(task.get("kind"))
    if kind is not TaskSupervisionKind.SENSOR_MEASURED_GROUND_TRUTH:
        raise ValueError(
            "Oracle requires sensor_measured_ground_truth; dataset annotations, "
            "model estimates and pseudo-GT must be reported under their own names"
        )
    expected_sha = str(manifest.get("archive_sha256", "")).lower()
    actual_sha = _sha256(path)
    if expected_sha != actual_sha:
        raise ValueError("Oracle archive SHA-256 does not match its provenance manifest")
    values, metadata, fps = _read_common(path)
    provenance = InputProvenance(
        regime=InputRegime.ORACLE,
        task_supervision_kind=kind,
        task_supervision_source=str(task.get("source", "")),
        camera_calibration_source=str(manifest.get("camera_calibration_source", "")),
        sha256=actual_sha,
    )
    if not provenance.task_supervision_source:
        raise ValueError("Oracle provenance must name the measurement source")
    return _make_input(path, values, metadata, fps, provenance)


def load_inputs(
    path: str | Path,
    *,
    regime: str | InputRegime = InputRegime.ESTIMATED,
    provenance_manifest: str | Path | None = None,
) -> PerceptionInput:
    regime = InputRegime.parse(regime)
    if regime is InputRegime.ESTIMATED:
        if provenance_manifest is not None:
            raise ValueError("Estimated input does not accept an Oracle provenance manifest")
        return load_estimated_inputs(path)
    if provenance_manifest is None:
        raise ValueError("Oracle input requires provenance_manifest")
    return load_oracle_inputs(path, provenance_manifest=provenance_manifest)
