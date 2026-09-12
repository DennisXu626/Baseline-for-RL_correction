"""Candidate left-Sharpa 22-joint FK interface for isolated M1-A diagnostics.

This is not an authoritative simulator articulation interface.  It evaluates
the registered candidate URDF using the control order and soft limits supplied
by the registered candidate manifest.  Joint values are never clipped.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class JointRecord:
    name: str
    kind: str
    parent: str
    child: str
    xyz_m: np.ndarray
    rpy_rad: np.ndarray
    axis: np.ndarray
    lower_rad: float | None
    upper_rad: float | None
    mimic: str | None


@dataclass(frozen=True)
class LandmarkRecord:
    name: str
    link: str
    local_m: np.ndarray


@dataclass(frozen=True)
class FKResult:
    link_names: tuple[str, ...]
    link_transforms: np.ndarray
    landmark_names: tuple[str, ...]
    landmarks_m: np.ndarray

    def transforms_by_link(self) -> dict[str, np.ndarray]:
        return dict(zip(self.link_names, self.link_transforms))


def read_json_from_tar(archive_path: Path, member_name: str) -> tuple[dict, bytes]:
    """Read the manifest directly from the registered archive member."""
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember(member_name)
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError(f"archive member is not a regular file: {member_name}")
        payload = stream.read()
    return json.loads(payload.decode("utf-8")), payload


def _vector(element: ET.Element | None, attribute: str, default: str) -> np.ndarray:
    value = default if element is None else element.attrib.get(attribute, default)
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid URDF vector {attribute}={value!r}")
    return result


def parse_urdf(path: Path) -> tuple[dict[str, JointRecord], tuple[str, ...]]:
    root = ET.parse(path).getroot()
    records: dict[str, JointRecord] = {}
    order = []
    for element in root.findall("joint"):
        name = element.attrib["name"]
        if name in records:
            raise RuntimeError(f"duplicate URDF joint name: {name}")
        parent_element = element.find("parent")
        child_element = element.find("child")
        if parent_element is None or child_element is None:
            raise RuntimeError(f"URDF joint lacks parent/child: {name}")
        limit = element.find("limit")
        lower = float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None
        upper = float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None
        mimic = element.find("mimic")
        records[name] = JointRecord(
            name=name,
            kind=element.attrib["type"],
            parent=parent_element.attrib["link"],
            child=child_element.attrib["link"],
            xyz_m=_vector(element.find("origin"), "xyz", "0 0 0"),
            rpy_rad=_vector(element.find("origin"), "rpy", "0 0 0"),
            axis=_vector(element.find("axis"), "xyz", "0 0 1"),
            lower_rad=lower,
            upper_rad=upper,
            mimic=None if mimic is None else mimic.attrib.get("joint"),
        )
        order.append(name)
    return records, tuple(order)


def rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw rotation, Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    return np.array(
        (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        ),
        dtype=np.float64,
    )


def axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"invalid revolute axis with norm {norm}")
    x, y, z = axis / norm
    cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle_rad) * cross + (1.0 - math.cos(angle_rad)) * (cross @ cross)


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


class SharpaLeftFKCandidateV1:
    """Strict 22-joint candidate FK rooted at ``left_hand_C_MC``."""

    def __init__(
        self,
        *,
        urdf_path: Path,
        root_link: str,
        joint_names: tuple[str, ...],
        soft_limits_rad: np.ndarray,
        landmarks: tuple[LandmarkRecord, ...],
        limit_tolerance_rad: float = 1e-6,
    ) -> None:
        if len(joint_names) != 22 or len(set(joint_names)) != 22:
            raise RuntimeError("candidate control contract must contain 22 unique joint names")
        limits = np.asarray(soft_limits_rad, dtype=np.float64)
        if limits.shape != (22, 2) or not np.isfinite(limits).all():
            raise RuntimeError("candidate soft limits must be finite with shape (22,2)")
        if not np.all(limits[:, 0] < limits[:, 1]):
            raise RuntimeError("every candidate soft limit must satisfy lower < upper")
        if not np.isfinite(limit_tolerance_rad) or limit_tolerance_rad != 1e-6:
            raise RuntimeError("candidate limit comparison tolerance must equal 1e-6 rad")

        all_joints, urdf_order = parse_urdf(urdf_path)
        children_by_parent: dict[str, list[str]] = {}
        for name in urdf_order:
            children_by_parent.setdefault(all_joints[name].parent, []).append(name)

        reachable_joint_names = []
        reachable_links = [root_link]
        pending_links = [root_link]
        seen_links = {root_link}
        while pending_links:
            parent = pending_links.pop(0)
            for name in children_by_parent.get(parent, []):
                joint = all_joints[name]
                if joint.child in seen_links:
                    raise RuntimeError(f"left subtree is not a tree at link {joint.child}")
                reachable_joint_names.append(name)
                reachable_links.append(joint.child)
                seen_links.add(joint.child)
                pending_links.append(joint.child)

        controlled = set(joint_names)
        missing = [name for name in joint_names if name not in all_joints]
        if missing:
            raise RuntimeError(f"manifest joints missing from URDF: {missing}")
        unreachable = [name for name in joint_names if name not in reachable_joint_names]
        if unreachable:
            raise RuntimeError(f"manifest joints not reachable from {root_link}: {unreachable}")
        extra_movable = [
            name for name in reachable_joint_names
            if all_joints[name].kind != "fixed" and name not in controlled
        ]
        if extra_movable:
            raise RuntimeError(f"uncontrolled movable joints in candidate left subtree: {extra_movable}")

        comparisons = []
        for index, name in enumerate(joint_names):
            joint = all_joints[name]
            if joint.kind != "revolute":
                raise RuntimeError(f"candidate controlled joint is not revolute: {name} ({joint.kind})")
            if joint.mimic is not None:
                raise RuntimeError(f"candidate controlled joint has mimic dependency: {name} -> {joint.mimic}")
            if joint.lower_rad is None or joint.upper_rad is None:
                raise RuntimeError(f"candidate controlled joint lacks finite URDF limits: {name}")
            if not np.isfinite((joint.lower_rad, joint.upper_rad)).all() or joint.lower_rad >= joint.upper_rad:
                raise RuntimeError(f"candidate controlled joint has invalid URDF limits: {name}")
            if float(np.linalg.norm(joint.axis)) <= 0.0:
                raise RuntimeError(f"candidate controlled joint has a zero axis: {name}")
            lower, upper = limits[index]
            within = bool(
                lower >= joint.lower_rad - limit_tolerance_rad
                and upper <= joint.upper_rad + limit_tolerance_rad
            )
            comparisons.append({
                "index": index,
                "name": name,
                "manifest_soft_lower_rad": float(lower),
                "manifest_soft_upper_rad": float(upper),
                "urdf_lower_rad": float(joint.lower_rad),
                "urdf_upper_rad": float(joint.upper_rad),
                "soft_minus_urdf_lower_rad": float(lower - joint.lower_rad),
                "soft_minus_urdf_upper_rad": float(upper - joint.upper_rad),
                "within_urdf_with_1e-6_rad_tolerance": within,
                "type": joint.kind,
                "axis": joint.axis.tolist(),
                "parent": joint.parent,
                "child": joint.child,
                "mimic": joint.mimic,
            })
            if not within:
                raise RuntimeError(f"manifest soft limit conflicts with URDF limit: {name}")

        for name in reachable_joint_names:
            joint = all_joints[name]
            if joint.kind not in ("fixed", "revolute"):
                raise RuntimeError(f"unsupported joint type in target subtree: {name} ({joint.kind})")
            if joint.mimic is not None:
                raise RuntimeError(f"mimic dependency in target subtree: {name} -> {joint.mimic}")

        if len(landmarks) != 10 or len({item.name for item in landmarks}) != 10:
            raise RuntimeError("candidate landmark contract must contain 10 unique ordered names")
        for item in landmarks:
            if item.link not in seen_links:
                raise RuntimeError(f"candidate landmark link is not reachable: {item.name} -> {item.link}")
            if np.asarray(item.local_m).shape != (3,) or not np.isfinite(item.local_m).all():
                raise RuntimeError(f"invalid local landmark coordinate: {item.name}")

        self.urdf_path = Path(urdf_path)
        self.root_link = root_link
        self.joint_names = joint_names
        self.soft_limits_rad = limits
        self.landmarks = landmarks
        self.limit_tolerance_rad = limit_tolerance_rad
        self.joints = all_joints
        self.reachable_joint_names = tuple(reachable_joint_names)
        self.link_names = tuple(reachable_links)
        self.limit_comparisons = tuple(comparisons)

    def forward(self, q_rad: np.ndarray) -> FKResult:
        q = np.asarray(q_rad, dtype=np.float64)
        if q.shape != (22,):
            raise ValueError(f"q_rad must have shape (22,), got {q.shape}")
        if not np.isfinite(q).all():
            raise ValueError("q_rad must contain only finite values")
        lower = self.soft_limits_rad[:, 0]
        upper = self.soft_limits_rad[:, 1]
        outside = np.flatnonzero(
            (q < lower - self.limit_tolerance_rad) | (q > upper + self.limit_tolerance_rad)
        )
        if len(outside):
            index = int(outside[0])
            raise ValueError(
                f"q_rad[{index}] ({self.joint_names[index]})={q[index]:.17g} exceeds "
                f"candidate soft limit [{lower[index]:.17g}, {upper[index]:.17g}] "
                f"with tolerance {self.limit_tolerance_rad:.1e}; input was not clipped"
            )

        q_by_name = dict(zip(self.joint_names, q))
        frames: dict[str, np.ndarray] = {self.root_link: np.eye(4, dtype=np.float64)}
        for name in self.reachable_joint_names:
            joint = self.joints[name]
            fixed_origin = transform(rpy_rotation(joint.rpy_rad), joint.xyz_m)
            motion = np.eye(4, dtype=np.float64)
            if joint.kind == "revolute":
                motion = transform(axis_angle_rotation(joint.axis, float(q_by_name[name])), np.zeros(3))
            frames[joint.child] = frames[joint.parent] @ fixed_origin @ motion

        link_transforms = np.stack([frames[name] for name in self.link_names])
        points = []
        for item in self.landmarks:
            homogeneous = np.r_[item.local_m, 1.0]
            points.append((frames[item.link] @ homogeneous)[:3])
        landmarks_m = np.asarray(points, dtype=np.float64)
        return FKResult(
            link_names=self.link_names,
            link_transforms=link_transforms,
            landmark_names=tuple(item.name for item in self.landmarks),
            landmarks_m=landmarks_m,
        )

    def descendants_of_joint(self, joint_name: str) -> set[str]:
        if joint_name not in self.reachable_joint_names:
            raise KeyError(joint_name)
        root = self.joints[joint_name].child
        descendants = {root}
        changed = True
        while changed:
            changed = False
            for name in self.reachable_joint_names:
                joint = self.joints[name]
                if joint.parent in descendants and joint.child not in descendants:
                    descendants.add(joint.child)
                    changed = True
        return descendants

    def pre_motion_frame(self, frames: dict[str, np.ndarray], joint_name: str) -> np.ndarray:
        joint = self.joints[joint_name]
        return frames[joint.parent] @ transform(rpy_rotation(joint.rpy_rad), joint.xyz_m)

