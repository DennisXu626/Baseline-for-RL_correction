"""Build H2S2R object references from declared video perception inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inputs import PerceptionInput


@dataclass(frozen=True)
class ReferenceTrajectory:
    times_s: np.ndarray
    object_0_pose_wxyz: np.ndarray
    object_1_pose_wxyz: np.ndarray
    object_0_source_valid: np.ndarray
    object_1_source_valid: np.ndarray
    source_sha256: str
    source_regime: str
    source_start_index: int
    anchoring: str

    @property
    def length(self) -> int:
        return len(self.times_s)


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("zero-length quaternion")
    return q / norm


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _normalize_quaternion(q)
    vector_q = np.concatenate(
        [np.zeros((*np.asarray(v).shape[:-1], 1)), np.asarray(v, dtype=np.float64)],
        axis=-1,
    )
    return _quat_multiply_wxyz(
        _quat_multiply_wxyz(q, vector_q), _quat_conjugate_wxyz(q)
    )[..., 1:]


def anchor_pose_sequence(
    poses_wxyz: np.ndarray,
    reset_pose_wxyz: np.ndarray,
) -> np.ndarray:
    """Apply one constant SE(3) transform so frame zero equals the sim reset."""

    poses = np.asarray(poses_wxyz, dtype=np.float64)
    reset = np.asarray(reset_pose_wxyz, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or reset.shape != (7,):
        raise ValueError("poses must be (T,7) and reset_pose must be (7,)")
    source_q = _normalize_quaternion(poses[:, 3:])
    reset_q = _normalize_quaternion(reset[3:])
    anchor_q = _quat_multiply_wxyz(reset_q, _quat_conjugate_wxyz(source_q[0]))
    anchored_pos = reset[:3] + _quat_rotate_wxyz(
        anchor_q, poses[:, :3] - poses[0, :3]
    )
    anchored_q = _normalize_quaternion(_quat_multiply_wxyz(anchor_q, source_q))
    result = np.concatenate([anchored_pos, anchored_q], axis=-1)
    result[0] = np.concatenate([reset[:3], reset_q])
    return result.astype(np.float32)


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where((dot < 0.0)[..., None], -q1, q1)
    dot = np.abs(dot).clip(-1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    f = np.asarray(fraction, dtype=np.float64)
    near = sin_theta < 1e-7
    safe_denominator = np.where(near, 1.0, sin_theta)
    w0 = np.where(
        near, 1.0 - f, np.sin((1.0 - f) * theta) / safe_denominator
    )
    w1 = np.where(near, f, np.sin(f * theta) / safe_denominator)
    return _normalize_quaternion(w0[..., None] * q0 + w1[..., None] * q1)


def resample_pose_sequence(
    source_times_s: np.ndarray,
    poses_wxyz: np.ndarray,
    target_times_s: np.ndarray,
) -> np.ndarray:
    source_times = np.asarray(source_times_s, dtype=np.float64)
    target_times = np.asarray(target_times_s, dtype=np.float64)
    poses = np.asarray(poses_wxyz, dtype=np.float64)
    if len(source_times) != len(poses) or len(source_times) < 2:
        raise ValueError("pose sequence must contain at least two timed samples")
    if np.any(np.diff(source_times) <= 0.0):
        raise ValueError("source times must be strictly increasing")
    clipped = np.clip(target_times, source_times[0], source_times[-1])
    upper = np.searchsorted(source_times, clipped, side="right")
    upper = np.clip(upper, 1, len(source_times) - 1)
    lower = upper - 1
    span = source_times[upper] - source_times[lower]
    fraction = (clipped - source_times[lower]) / span
    position = poses[lower, :3] + fraction[:, None] * (
        poses[upper, :3] - poses[lower, :3]
    )
    quaternion = _slerp_wxyz(poses[lower, 3:], poses[upper, 3:], fraction)
    return np.concatenate([position, quaternion], axis=-1).astype(np.float32)


def _nearest_valid(
    source_times_s: np.ndarray,
    valid: np.ndarray,
    target_times_s: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(source_times_s, target_times_s, side="left")
    indices = np.clip(indices, 0, len(source_times_s) - 1)
    previous = np.clip(indices - 1, 0, len(source_times_s) - 1)
    choose_previous = (
        np.abs(target_times_s - source_times_s[previous])
        <= np.abs(source_times_s[indices] - target_times_s)
    )
    indices = np.where(choose_previous, previous, indices)
    return np.asarray(valid, dtype=bool)[indices]


def build_reference_trajectory(
    inputs: PerceptionInput,
    *,
    object_0_reset_pose_wxyz: np.ndarray,
    object_1_reset_pose_wxyz: np.ndarray,
    control_hz: float = 20.0,
    source_start_index: int = 0,
) -> ReferenceTrajectory:
    """Create the main H2S2R reference without confidence gating or cleanup.

    ``source_start_index`` selects the one globally shared interaction-start
    frame before either object's trajectory is processed.  A separate constant
    SE(3) alignment per object maps that selected pose to the declared simulator
    reset.  It is a coordinate-frame adaptation only: all subsequent relative
    motion and frame-to-frame estimation error remain.
    Validity masks are carried into reports but never gate reward or the clock.
    """

    if not np.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    if not isinstance(source_start_index, (int, np.integer)):
        raise TypeError("source_start_index must be an integer frame index")
    source_start_index = int(source_start_index)
    if source_start_index < 0 or source_start_index >= len(inputs.source_times_s) - 1:
        raise ValueError(
            "source_start_index must leave at least two source frames; "
            f"got {source_start_index} for {len(inputs.source_times_s)} frames"
        )

    source_times = np.asarray(
        inputs.source_times_s[source_start_index:], dtype=np.float64
    )
    source_times = source_times - source_times[0]
    object_0_source = inputs.object_0_pose_wxyz[source_start_index:]
    object_1_source = inputs.object_1_pose_wxyz[source_start_index:]
    object_0_valid = inputs.object_0_valid[source_start_index:]
    object_1_valid = inputs.object_1_valid[source_start_index:]

    duration = float(source_times[-1])
    target_times = np.arange(int(round(duration * control_hz)) + 1) / control_hz
    target_times = np.minimum(target_times, duration)
    object_0 = anchor_pose_sequence(
        object_0_source, object_0_reset_pose_wxyz
    )
    object_1 = anchor_pose_sequence(
        object_1_source, object_1_reset_pose_wxyz
    )
    return ReferenceTrajectory(
        times_s=target_times.astype(np.float64),
        object_0_pose_wxyz=resample_pose_sequence(
            source_times, object_0, target_times
        ),
        object_1_pose_wxyz=resample_pose_sequence(
            source_times, object_1, target_times
        ),
        object_0_source_valid=_nearest_valid(
            source_times, object_0_valid, target_times
        ),
        object_1_source_valid=_nearest_valid(
            source_times, object_1_valid, target_times
        ),
        source_sha256=inputs.provenance.sha256,
        source_regime=inputs.provenance.regime.value,
        source_start_index=source_start_index,
        anchoring=(
            "per-object constant SE(3): selected estimated source frame "
            f"{source_start_index} -> declared sim reset; relative motion/noise preserved"
        ),
    )
