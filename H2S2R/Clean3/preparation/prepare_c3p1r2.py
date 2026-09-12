"""Prepare the reviewer-frozen C3-P1R2 palm contract without replacing R artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
BASE = WORKSPACE / "RL/Baselines/task_plate/h2s2r_clean3_staging_20260912/extracted/base/clean3_base_package_20260909"
SOURCE = WORKSPACE / "reports/H2S2R_TRAINING_PILOT_20260912/source_after"
PREPARED = HERE / "prepared"
R_PREPARED = HERE / "c3p1r_prepared"
OUTPUT = HERE / "c3p1r2_prepared"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_literal(node):
    if isinstance(node, ast.Tuple):
        return tuple(numeric_literal(value) for value in node.elts)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -numeric_literal(node.operand)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr == "pi":
        return float(np.pi)
    raise ValueError(f"unsupported controller constant expression: {ast.dump(node)}")


def seven_palm_points(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Return the origin and +/- 10 cm local axes used by the FABRICS palm task."""

    local = np.array([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, -0.1],
    ])
    return position[None, :] + local @ rotation.T


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    runtime_path = PREPARED / "clean3_c3p1_runtime_inputs.npz"
    r_sidecar_path = R_PREPARED / "clean3_action_calibration_c3p1r.npz"
    expected = {
        "runtime": "3da2b16f27d74be4898a00fc807e6993dc1915eadbaff2f9aa8539fd1a6a84ff",
        "r_sidecar": "f878f87f61f218175292583b266f6150142c95dc3366a328c516383e58507747",
        "right_pca": "31c349ea79313a504b3a809a51bbf9fbc646922ae86c3363e0688c8475df36e2",
        "left_pca": "578ff0a8a9ada51cdf0ff7a442fa04d6ed22dd51e1130366f1936863fe7fd521",
    }
    actual = {
        "runtime": sha256(runtime_path), "r_sidecar": sha256(r_sidecar_path),
        "right_pca": sha256(PREPARED / "right_synergy_pca5.npz"),
        "left_pca": sha256(PREPARED / "left_synergy_pca5.npz"),
    }
    if actual != expected:
        raise RuntimeError(f"frozen input identity changed: {actual}")

    # The R2 sidecar deliberately reuses the exact R hand-calibration bytes.
    r2_sidecar_path = OUTPUT / "clean3_action_calibration_c3p1r2.npz"
    r2_sidecar_path.write_bytes(r_sidecar_path.read_bytes())

    os.environ["VEGA_URDF"] = str((HERE / "assets/vega_1p_sharpa.urdf").resolve())
    sys.path.insert(0, str(BASE / "code"))
    from rl_rebuild.correction.kinematics import Urdf  # pylint: disable=import-outside-toplevel

    controller_path = SOURCE / "h2s2r/controller.py"
    assignments = {}
    for node in ast.parse(controller_path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"PALM_MINIMUM", "PALM_MAXIMUM"}:
                    assignments[target.id] = numeric_literal(node.value)

    runtime = np.load(runtime_path, allow_pickle=False)
    q58 = runtime["reset_q58"]
    arm_q = {"right": q58[:7], "left": q58[7:14]}
    palms = {}
    for side in ("right", "left"):
        urdf_path = SOURCE / f"h2s2r/assets/vega_sharpa_{side}_fabric.urdf"
        urdf = Urdf(str(urdf_path))
        revolute = [name for name in urdf.chain_to("h2s2r_palm") if urdf.joints[name]["type"] == "revolute"]
        transform = urdf.link_pose("h2s2r_palm", {name: float(value) for name, value in zip(revolute, arm_q[side])})
        rotation = transform[:3, :3]
        # Uppercase ZYX is the real FABRICS Rz(z) @ Ry(y) @ Rx(x) contract.
        angles = Rotation.from_matrix(rotation).as_euler("ZYX")
        reconstructed = Rotation.from_euler("ZYX", angles).as_matrix()
        low = np.asarray(assignments["PALM_MINIMUM"], dtype=float)
        high = np.asarray(assignments["PALM_MAXIMUM"], dtype=float)
        if side == "left":
            low[2] = -0.03
        pose = np.r_[transform[:3, 3], angles]
        points = seven_palm_points(transform[:3, 3], rotation)
        reconstructed_points = seven_palm_points(pose[:3], reconstructed)
        outside = (pose < low - 1e-6) | (pose > high + 1e-6)
        palms[side] = {
            "fabric_urdf": str(urdf_path.resolve()), "fabric_urdf_sha256": sha256(urdf_path),
            "joint_order": revolute, "static_fk_position_euler_ZYX": pose.tolist(),
            "minimum": low.tolist(), "maximum": high.tolist(),
            "outside_dimensions": np.flatnonzero(outside).tolist(),
            "passes_frozen_r2_bounds": not bool(outside.any()),
            "rotation_roundtrip_max_abs_error": float(np.max(np.abs(reconstructed - rotation))),
            "seven_point_roundtrip_max_abs_error_m": float(np.max(np.abs(reconstructed_points - points))),
            "seven_points_arm_center_m": points.tolist(),
        }

    status = "CPU_PREPARED" if all(v["passes_frozen_r2_bounds"] for v in palms.values()) else "BLOCKED_R2_PALM_BOUNDS"
    report = {
        "schema": "h2s2r_clean3_c3p1r2_action_calibration_v1", "status": status,
        "method_adaptation": "R hand calibration plus fixed Clean3 left-palm z minimum -0.03 m and fixed nominal FK palm centers",
        "angle_contract": "SciPy uppercase ZYX; R=Rz(z)@Ry(y)@Rx(x)",
        "left_palm_z_minimum_m": -0.03, "online_range_growth": False,
        "r2_sidecar": {"path": str(r2_sidecar_path.resolve()), "sha256": sha256(r2_sidecar_path)},
        "identical_to_r_sidecar": sha256(r2_sidecar_path) == sha256(r_sidecar_path),
        "frozen_inputs": actual, "controller_source": {"path": str(controller_path.resolve()), "sha256": sha256(controller_path)},
        "palm_static_fk": palms,
        "prohibited": ["PCA change", "static initialization change", "online reanchor", "per-frame GraspPose feedforward", "further palm-bound change"],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
