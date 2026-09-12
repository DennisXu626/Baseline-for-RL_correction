"""Small, dependency-light geometry helpers for the Pour17 adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def ndarray_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_module(path: str | Path, name: str):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize(vector: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError(f"degenerate vector for {label}: {norm}")
    return vector / norm


def axis_angle_matrix(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / angle
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def quat_normalize_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm < 1e-12):
        raise ValueError("non-finite or zero quaternion")
    return value / norm


def quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    return np.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), axis=-1)


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("matrix must have shape (3,3)")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        value = np.array((0.25 * scale,
                          (matrix[2, 1] - matrix[1, 2]) / scale,
                          (matrix[0, 2] - matrix[2, 0]) / scale,
                          (matrix[1, 0] - matrix[0, 1]) / scale))
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            value = np.array(((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                              (matrix[0, 1] + matrix[1, 0]) / scale,
                              (matrix[0, 2] + matrix[2, 0]) / scale))
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            value = np.array(((matrix[0, 2] - matrix[2, 0]) / scale,
                              (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                              (matrix[1, 2] + matrix[2, 1]) / scale))
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            value = np.array(((matrix[1, 0] - matrix[0, 1]) / scale,
                              (matrix[0, 2] + matrix[2, 0]) / scale,
                              (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale))
    return quat_normalize_wxyz(value)


def quat_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize_wxyz(quaternion)
    return np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ), dtype=np.float64)


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = quat_normalize_wxyz(q0)
    q1 = quat_normalize_wxyz(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return quat_normalize_wxyz((1.0 - fraction) * q0 + fraction * q1)
    angle = np.arccos(dot)
    return (np.sin((1.0 - fraction) * angle) * q0
            + np.sin(fraction * angle) * q1) / np.sin(angle)


def resample_linear(values: np.ndarray, source_times: np.ndarray,
                    target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    result = np.column_stack([
        np.interp(target_times, source_times, flat[:, index])
        for index in range(flat.shape[1])
    ])
    return result.reshape((len(target_times),) + values.shape[1:])


def resample_pose_wxyz(poses: np.ndarray, source_times: np.ndarray,
                       target_times: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    result = np.empty((len(target_times), 7), dtype=np.float64)
    result[:, :3] = resample_linear(poses[:, :3], source_times, target_times)
    for target_index, time_s in enumerate(target_times):
        if time_s <= source_times[0]:
            result[target_index, 3:] = quat_normalize_wxyz(poses[0, 3:])
            continue
        if time_s >= source_times[-1]:
            result[target_index, 3:] = quat_normalize_wxyz(poses[-1, 3:])
            continue
        upper = int(np.searchsorted(source_times, time_s, side="right"))
        lower = upper - 1
        fraction = float((time_s - source_times[lower])
                         / (source_times[upper] - source_times[lower]))
        result[target_index, 3:] = slerp_wxyz(
            poses[lower, 3:], poses[upper, 3:], fraction
        )
    return result


def mano_landmarks(joints21_wrist_zero_m: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints21_wrist_zero_m, dtype=np.float64)
    return np.asarray((
        joints[4], (joints[2] + joints[3]) / 2.0,
        joints[8], (joints[6] + joints[7]) / 2.0,
        joints[12], (joints[10] + joints[11]) / 2.0,
        joints[16], (joints[14] + joints[15]) / 2.0,
        joints[20], (joints[18] + joints[19]) / 2.0,
    ))


def palm_basis(wrist: np.ndarray, mcp_points: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    index, middle, ring, pinky = [np.asarray(value, dtype=np.float64) for value in mcp_points]
    origin = (index + middle + ring + pinky) / 4.0
    x_axis = normalize(middle - wrist, "palm longitudinal axis")
    z_axis = normalize(np.cross(x_axis, index - pinky), "palm normal")
    y_axis = normalize(np.cross(z_axis, x_axis), "palm transverse axis")
    return origin, np.column_stack((x_axis, y_axis, z_axis))


def build_calibration(decoder, model: dict, final_sources: tuple,
                      interface, *, shapedirs_x_flip: bool, side: str) -> dict:
    decoded = decoder.decode_mano(
        model, transl=np.zeros(3), global_aa=np.zeros(3), pose45=np.zeros(45),
        betas=np.zeros(10), add_hands_mean=False,
        left_shapedirs_x_flip=shapedirs_x_flip, final_sources=final_sources,
    )
    joints = np.asarray(decoded["joints21_m"], dtype=np.float64)
    wrist = joints[0].copy()
    joints -= wrist
    mano_origin, mano_basis = palm_basis(
        np.zeros(3), (joints[5], joints[9], joints[13], joints[17])
    )
    zero = interface.forward(np.zeros(22)).transforms_by_link()
    prefix = f"{side}_"
    robot_mcp = tuple(zero[f"{prefix}{finger}_MCP_VL"][:3, 3]
                      for finger in ("index", "middle", "ring", "pinky"))
    robot_origin, robot_basis = palm_basis(np.zeros(3), robot_mcp)
    rotation = robot_basis @ mano_basis.T
    translation = robot_origin - rotation @ mano_origin
    return {
        "R_S_from_H": rotation,
        "t_S_from_H_m": translation,
        "mano_canonical_wrist_m": wrist,
        "mano_palm_origin_H_m": mano_origin,
        "robot_palm_origin_S_m": robot_origin,
        "orthogonality_fro": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
        "determinant": float(np.linalg.det(rotation)),
    }


def mano_world_to_robot_root_pose(decoded_world_joints: np.ndarray, global_aa: np.ndarray,
                                  calibration: dict) -> np.ndarray:
    world_from_h = axis_angle_matrix(global_aa)
    s_from_h = np.asarray(calibration["R_S_from_H"], dtype=np.float64)
    t_s_from_h = np.asarray(calibration["t_S_from_H_m"], dtype=np.float64)
    world_from_s = world_from_h @ s_from_h.T
    wrist_world = np.asarray(decoded_world_joints[0], dtype=np.float64)
    root_world = wrist_world - world_from_s @ t_s_from_h
    return np.r_[root_world, matrix_to_quat_wxyz(world_from_s)]


def common_yaw_translation(source_centers: np.ndarray, target_centers: np.ndarray) -> dict:
    source = np.asarray(source_centers, dtype=np.float64)
    target = np.asarray(target_centers, dtype=np.float64)
    if source.shape != (2, 3) or target.shape != (2, 3):
        raise ValueError("source_centers and target_centers must have shape (2,3)")
    source_vector = source[1, :2] - source[0, :2]
    target_vector = target[1, :2] - target[0, :2]
    yaw = float(np.arctan2(target_vector[1], target_vector[0])
                - np.arctan2(source_vector[1], source_vector[0]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    aligned = (rotation @ source.T).T + translation
    return {
        "yaw_rad": yaw,
        "R_env_from_reconstruction": rotation,
        "t_env_from_reconstruction_m": translation,
        "aligned_source_centers_m": aligned,
        "target_centers_m": target,
        "center_residual_m": np.linalg.norm(aligned - target, axis=1),
    }


def transform_pose_series(poses_wxyz: np.ndarray, rotation: np.ndarray,
                          translation: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses_wxyz, dtype=np.float64)
    result = poses.copy()
    result[:, :3] = (rotation @ poses[:, :3].T).T + translation
    left_quaternion = matrix_to_quat_wxyz(rotation)
    result[:, 3:] = quat_normalize_wxyz(np.asarray([
        quat_multiply_wxyz(left_quaternion, quaternion) for quaternion in poses[:, 3:]
    ]))
    return result


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw rotation."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array((
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    ))


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-10:
        return np.zeros(3, dtype=np.float64)
    if angle > np.pi - 1e-6:
        symmetric = (rotation + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(symmetric), 0.0, None))
        pivot = int(np.argmax(axis))
        axis = symmetric[:, pivot] / (axis[pivot] + 1e-12)
        return axis / (np.linalg.norm(axis) + 1e-12) * angle
    vector = np.array((rotation[2, 1] - rotation[1, 2],
                       rotation[0, 2] - rotation[2, 0],
                       rotation[1, 0] - rotation[0, 1]))
    return vector * (angle / (2.0 * np.sin(angle)))


class UrdfKinematics:
    """Minimal named-chain FK used only for registered-arm IK and checks."""

    def __init__(self, path: str | Path):
        root = ET.parse(path).getroot()
        self.path = str(Path(path).resolve())
        self.joints: dict[str, dict] = {}
        self.parent_joint: dict[str, str] = {}
        for element in root.findall("joint"):
            name = str(element.get("name"))
            kind = str(element.get("type"))
            origin = element.find("origin")
            xyz = np.zeros(3) if origin is None or origin.get("xyz") is None else np.fromstring(origin.get("xyz"), sep=" ")
            rpy = np.zeros(3) if origin is None or origin.get("rpy") is None else np.fromstring(origin.get("rpy"), sep=" ")
            axis_element = element.find("axis")
            axis = np.array((1.0, 0.0, 0.0)) if axis_element is None else np.fromstring(axis_element.get("xyz"), sep=" ")
            axis = axis / (np.linalg.norm(axis) + 1e-15)
            limit = element.find("limit")
            if kind == "continuous":
                lower, upper = -np.pi, np.pi
            elif limit is None:
                lower, upper = 0.0, 0.0
            else:
                lower = float(limit.get("lower", "0"))
                upper = float(limit.get("upper", "0"))
            child = str(element.find("child").get("link"))
            self.joints[name] = {
                "type": kind,
                "parent": str(element.find("parent").get("link")),
                "child": child,
                "origin": _transform(_rpy_matrix(rpy), xyz),
                "axis": axis,
                "lower": lower,
                "upper": upper,
            }
            self.parent_joint[child] = name

    def chain_to(self, link: str) -> list[str]:
        result: list[str] = []
        cursor = link
        while cursor in self.parent_joint:
            joint = self.parent_joint[cursor]
            result.append(joint)
            cursor = self.joints[joint]["parent"]
        return result[::-1]

    def chain_between(self, start_link: str | None, end_link: str) -> list[str]:
        chain = self.chain_to(end_link)
        if start_link is None:
            return chain
        children = [self.joints[name]["child"] for name in chain]
        if start_link not in children:
            raise ValueError(f"{start_link} is not on the chain to {end_link}")
        return chain[children.index(start_link) + 1:]

    def joint_transform(self, name: str, value: float) -> np.ndarray:
        joint = self.joints[name]
        if joint["type"] == "fixed":
            motion = np.eye(4)
        elif joint["type"] == "prismatic":
            motion = _transform(np.eye(3), joint["axis"] * value)
        else:
            motion = _transform(axis_angle_matrix(joint["axis"] * value), np.zeros(3))
        return joint["origin"] @ motion

    def fk_chain(self, end_link: str, q_by_name: dict[str, float],
                 anchor_link: str | None = None,
                 anchor_transform: np.ndarray | None = None) -> tuple[np.ndarray, list[tuple]]:
        pose = np.eye(4) if anchor_transform is None else np.asarray(anchor_transform, dtype=np.float64).copy()
        axes: list[tuple] = []
        for name in self.chain_between(anchor_link, end_link):
            joint = self.joints[name]
            pose = pose @ joint["origin"]
            if joint["type"] != "fixed":
                axes.append((name, pose[:3, 3].copy(), pose[:3, :3] @ joint["axis"]))
                if joint["type"] == "prismatic":
                    pose = pose @ _transform(np.eye(3), joint["axis"] * float(q_by_name.get(name, 0.0)))
                else:
                    pose = pose @ _transform(
                        axis_angle_matrix(joint["axis"] * float(q_by_name.get(name, 0.0))),
                        np.zeros(3),
                    )
        return pose, axes


class ArmIK:
    """Limit-aware LM arm IK anchored to a measured runtime link pose.

    Failed samples return their own best finite solution and error.  The caller
    must preserve the failure mask; this class never substitutes another
    trajectory or silently drops a sample.
    """

    def __init__(self, urdf: UrdfKinematics, side: str, anchor_link: str,
                 anchor_pose_wxyz: np.ndarray, arm_joint_names: Iterable[str],
                 initial_q: np.ndarray, limits: np.ndarray):
        self.urdf = urdf
        self.side = side
        self.ee_link = f"{side}_hand_C_MC"
        self.anchor_link = anchor_link
        anchor = np.asarray(anchor_pose_wxyz, dtype=np.float64)
        self.anchor_transform = _transform(quat_to_matrix_wxyz(anchor[3:]), anchor[:3])
        self.arm_joint_names = tuple(arm_joint_names)
        self.initial_q = np.asarray(initial_q, dtype=np.float64).copy()
        limits = np.asarray(limits, dtype=np.float64)
        self.lower = limits[:, 0]
        self.upper = limits[:, 1]
        if len(self.arm_joint_names) != 7 or limits.shape != (7, 2):
            raise ValueError("ArmIK requires exactly seven named arm joints")
        chain = self.urdf.chain_between(anchor_link, self.ee_link)
        missing = [name for name in self.arm_joint_names if name not in chain]
        if missing:
            raise ValueError(f"arm joints missing from {anchor_link}->{self.ee_link}: {missing}")

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose, _ = self.urdf.fk_chain(
            self.ee_link,
            {name: float(value) for name, value in zip(self.arm_joint_names, q)},
            self.anchor_link,
            self.anchor_transform,
        )
        return pose[:3, 3].copy(), pose[:3, :3].copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        pose, axes = self.urdf.fk_chain(
            self.ee_link,
            {name: float(value) for name, value in zip(self.arm_joint_names, q)},
            self.anchor_link,
            self.anchor_transform,
        )
        endpoint = pose[:3, 3]
        lookup = {name: index for index, name in enumerate(self.arm_joint_names)}
        jacobian = np.zeros((6, 7), dtype=np.float64)
        for name, origin, axis in axes:
            if name in lookup:
                index = lookup[name]
                jacobian[:3, index] = np.cross(axis, endpoint - origin)
                jacobian[3:, index] = axis
        return jacobian

    def solve(self, target_pose_wxyz: np.ndarray, initial_q: np.ndarray,
              *, iterations: int = 250, position_tolerance_m: float = 0.005,
              rotation_tolerance_rad: float = 0.05, rotation_weight: float = 0.35,
              damping: float = 0.05, step_clip_rad: float = 0.25) -> dict:
        target = np.asarray(target_pose_wxyz, dtype=np.float64)
        target_rotation = quat_to_matrix_wxyz(target[3:])
        q = np.clip(np.asarray(initial_q, dtype=np.float64), self.lower, self.upper)

        def evaluate(candidate: np.ndarray) -> tuple:
            position, rotation = self.fk(candidate)
            position_vector = target[:3] - position
            rotation_vector = _so3_log(target_rotation @ rotation.T)
            cost = float(np.dot(position_vector, position_vector)
                         + (rotation_weight ** 2) * np.dot(rotation_vector, rotation_vector))
            return cost, float(np.linalg.norm(position_vector)), float(np.linalg.norm(rotation_vector)), position_vector, rotation_vector

        current = evaluate(q)
        best = (current[0], q.copy(), current[1], current[2])
        damping_value = float(damping)
        used = 0
        for used in range(1, iterations + 1):
            if current[1] < position_tolerance_m and current[2] < rotation_tolerance_rad:
                break
            jacobian = self.jacobian(q)
            weighted = np.vstack((jacobian[:3], rotation_weight * jacobian[3:]))
            error = np.r_[current[3], rotation_weight * current[4]]
            delta = weighted.T @ np.linalg.solve(
                weighted @ weighted.T + damping_value * damping_value * np.eye(6), error
            )
            largest = float(np.max(np.abs(delta)))
            if largest > step_clip_rad:
                delta *= step_clip_rad / largest
            trial_q = np.clip(q + delta, self.lower, self.upper)
            trial = evaluate(trial_q)
            if trial[0] < current[0]:
                q, current = trial_q, trial
                damping_value = max(damping_value * 0.5, 1e-4)
                if current[0] < best[0]:
                    best = (current[0], q.copy(), current[1], current[2])
            else:
                damping_value = min(damping_value * 2.0, 10.0)
                if damping_value >= 10.0:
                    break
        _, best_q, position_error, rotation_error = best
        return {
            "q": best_q,
            "position_error_m": position_error,
            "rotation_error_rad": rotation_error,
            "ok": bool(position_error < position_tolerance_m
                       and rotation_error < rotation_tolerance_rad),
            "iterations": used,
            "at_limit": np.isclose(best_q, self.lower, atol=1e-4)
                        | np.isclose(best_q, self.upper, atol=1e-4),
        }

    def solve_trajectory(self, targets_wxyz: np.ndarray, *, seed: int,
                         restart_count: int = 6, **kwargs) -> tuple[np.ndarray, list[dict]]:
        rng = np.random.default_rng(seed)
        q = self.initial_q.copy()
        last_success = q.copy()
        output = np.empty((len(targets_wxyz), 7), dtype=np.float64)
        records: list[dict] = []
        for index, target in enumerate(targets_wxyz):
            candidates = [self.solve(target, q, **kwargs)]
            if not candidates[0]["ok"] and restart_count > 0:
                seeds = [self.initial_q, last_success]
                seeds.extend(rng.uniform(self.lower, self.upper)
                             for _ in range(max(0, restart_count - 2)))
                candidates.extend(self.solve(target, candidate, **kwargs) for candidate in seeds)
            result = min(candidates, key=lambda item: item["position_error_m"] ** 2
                         + (kwargs.get("rotation_weight", 0.35) * item["rotation_error_rad"]) ** 2)
            successful = [item for item in candidates if item["ok"]]
            if successful:
                result = min(successful, key=lambda item: np.linalg.norm(item["q"] - q))
                last_success = result["q"].copy()
            q = result["q"].copy()
            output[index] = q
            records.append({
                "index": index,
                "ok": bool(result["ok"]),
                "position_error_m": float(result["position_error_m"]),
                "rotation_error_rad": float(result["rotation_error_rad"]),
                "iterations": int(result["iterations"]),
                "at_limit_count": int(np.count_nonzero(result["at_limit"])),
            })
        return output, records
