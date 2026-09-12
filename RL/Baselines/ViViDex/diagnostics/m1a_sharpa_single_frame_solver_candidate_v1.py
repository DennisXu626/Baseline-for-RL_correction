"""Proposed single-frame Sharpa solver for isolated synthetic diagnostics.

The module is independent of MANO, calibration A, perception artifacts, robot
references, replay, and training.  NLopt is imported only when ``solve`` is
called so the objective and analytic Jacobian remain independently auditable
when that optional runtime dependency is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import m1a_sharpa_left_fk_candidate_v1 as fk_candidate


@dataclass(frozen=True)
class ObjectiveEvaluation:
    tracking: float
    smoothness: float
    total: float
    gradient: np.ndarray
    landmarks_m: np.ndarray
    residual_m: np.ndarray


@dataclass(frozen=True)
class SolverSettings:
    alpha: float = 4e-3
    maxeval: int = 500
    ftol_abs: float = 1e-12
    xtol_abs_rad: float = 1e-8


def _finite_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def landmark_jacobian(
    interface: fk_candidate.SharpaLeftFKCandidateV1,
    q_rad: np.ndarray,
    result: fk_candidate.FKResult | None = None,
) -> np.ndarray:
    """Return analytic landmark Jacobians with shape ``(10, 3, 22)``.

    For each ancestor revolute joint, the column is
    ``axis_in_root x (point_in_root - pivot_in_root)``.  Axis and pivot come
    from the joint frame before applying that joint's motion.
    """
    q = _finite_array(q_rad, (22,), "q_rad")
    evaluated = interface.forward(q) if result is None else result
    frames = evaluated.transforms_by_link()
    jacobian = np.zeros((10, 3, 22), dtype=np.float64)
    descendants = {
        name: interface.descendants_of_joint(name) for name in interface.joint_names
    }
    for joint_index, joint_name in enumerate(interface.joint_names):
        joint = interface.joints[joint_name]
        pre_motion = interface.pre_motion_frame(frames, joint_name)
        pivot_in_root = pre_motion[:3, 3]
        axis = joint.axis / np.linalg.norm(joint.axis)
        axis_in_root = pre_motion[:3, :3] @ axis
        for point_index, (landmark, point_in_root) in enumerate(
            zip(interface.landmarks, evaluated.landmarks_m)
        ):
            if landmark.link in descendants[joint_name]:
                jacobian[point_index, :, joint_index] = np.cross(
                    axis_in_root, point_in_root - pivot_in_root
                )
    return jacobian


def evaluate_objective(
    interface: fk_candidate.SharpaLeftFKCandidateV1,
    q_rad: np.ndarray,
    y_m: np.ndarray,
    q_prev_rad: np.ndarray,
    *,
    alpha: float = 4e-3,
) -> ObjectiveEvaluation:
    """Evaluate the summed ten-point objective and its analytic gradient."""
    q = _finite_array(q_rad, (22,), "q_rad")
    y = _finite_array(y_m, (10, 3), "y_m")
    q_prev = _finite_array(q_prev_rad, (22,), "q_prev_rad")
    if not np.isfinite(alpha) or alpha != 4e-3:
        raise ValueError("this candidate objective requires alpha=4e-3")
    result = interface.forward(q)
    residual = result.landmarks_m - y
    tracking = float(np.sum(residual * residual))
    smooth_delta = q - q_prev
    smoothness = float(alpha * np.sum(smooth_delta * smooth_delta))
    jacobian = landmark_jacobian(interface, q, result)
    gradient = 2.0 * np.einsum("jci,jc->i", jacobian, residual)
    gradient += 2.0 * alpha * smooth_delta
    return ObjectiveEvaluation(
        tracking=tracking,
        smoothness=smoothness,
        total=tracking + smoothness,
        gradient=gradient,
        landmarks_m=result.landmarks_m,
        residual_m=residual,
    )


def solve_single_frame(
    interface: fk_candidate.SharpaLeftFKCandidateV1,
    y_m: np.ndarray,
    q_prev_rad: np.ndarray,
    q_init_rad: np.ndarray,
    *,
    settings: SolverSettings = SolverSettings(),
) -> dict[str, Any]:
    """Solve one bounded frame with NLopt LD_SLSQP and analytic gradients.

    The solver receives only ``y_m``, ``q_prev_rad``, ``q_init_rad``, the FK
    contract, and fixed settings.  It has no test-oracle or ``q_star`` input.
    """
    y = _finite_array(y_m, (10, 3), "y_m")
    q_prev = _finite_array(q_prev_rad, (22,), "q_prev_rad")
    q_init = _finite_array(q_init_rad, (22,), "q_init_rad")
    interface.forward(q_init)  # strict finite/shape/limit validation; no clip
    if settings != SolverSettings():
        raise ValueError("solver settings differ from the frozen candidate settings")

    import nlopt  # type: ignore[import-not-found]  # optional, checked by the audit

    optimizer = nlopt.opt(nlopt.LD_SLSQP, 22)
    optimizer.set_lower_bounds(interface.soft_limits_rad[:, 0])
    optimizer.set_upper_bounds(interface.soft_limits_rad[:, 1])
    optimizer.set_maxeval(settings.maxeval)
    optimizer.set_ftol_abs(settings.ftol_abs)
    optimizer.set_xtol_abs(np.full(22, settings.xtol_abs_rad, dtype=np.float64))
    calls = 0

    def objective(q: np.ndarray, gradient: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        evaluation = evaluate_objective(
            interface, q, y, q_prev, alpha=settings.alpha
        )
        if gradient.size:
            gradient[:] = evaluation.gradient
        return evaluation.total

    optimizer.set_min_objective(objective)
    q_final = np.asarray(optimizer.optimize(q_init.copy()), dtype=np.float64)
    return {
        "q_final_rad": q_final,
        "nlopt_return_code": int(optimizer.last_optimize_result()),
        "nlopt_reported_objective": float(optimizer.last_optimum_value()),
        "objective_call_count": calls,
    }

