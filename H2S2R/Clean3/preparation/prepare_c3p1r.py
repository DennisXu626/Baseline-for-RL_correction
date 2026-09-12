"""Prepare and validate the C3-P1R fixed action-calibration sidecar."""

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
OUTPUT = HERE / "c3p1r_prepared"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_literal(node):
    """Read tuple constants plus the controller's explicit ``np.pi`` values."""

    if isinstance(node, ast.Tuple):
        return tuple(numeric_literal(value) for value in node.elts)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -numeric_literal(node.operand)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "np"
        and node.attr == "pi"
    ):
        return float(np.pi)
    raise ValueError(f"unsupported controller constant expression: {ast.dump(node)}")


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    runtime_path = PREPARED / "clean3_c3p1_runtime_inputs.npz"
    pca_paths = {side: PREPARED / f"{side}_synergy_pca5.npz" for side in ("right", "left")}
    expected_pca_sha = {
        "right": "31c349ea79313a504b3a809a51bbf9fbc646922ae86c3363e0688c8475df36e2",
        "left": "578ff0a8a9ada51cdf0ff7a442fa04d6ed22dd51e1130366f1936863fe7fd521",
    }
    for side, path in pca_paths.items():
        if sha256(path) != expected_pca_sha[side]:
            raise RuntimeError(f"{side} PCA identity changed")
    runtime = np.load(runtime_path, allow_pickle=False)

    sys.path.insert(0, str(HERE))
    from c3p1r.action_calibration import build_sidecar  # pylint: disable=import-outside-toplevel

    sidecar_path = OUTPUT / "clean3_action_calibration_c3p1r.npz"
    build_sidecar(
        right_pca=pca_paths["right"],
        left_pca=pca_paths["left"],
        q0_right=runtime["static_right_f_grasp"],
        q0_left=runtime["static_left_f_grasp"],
        output=sidecar_path,
    )

    # Reproduce the exact single-side FABRICS palm FK from its generated URDF.
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
    PALM_MINIMUM = assignments["PALM_MINIMUM"]
    PALM_MAXIMUM = assignments["PALM_MAXIMUM"]

    palm = {}
    q58 = runtime["reset_q58"]
    arm_q = {"right": q58[:7], "left": q58[7:14]}
    for side in ("right", "left"):
        urdf_path = SOURCE / f"h2s2r/assets/vega_sharpa_{side}_fabric.urdf"
        urdf = Urdf(str(urdf_path))
        revolute = [
            name for name in urdf.chain_to("h2s2r_palm")
            if urdf.joints[name]["type"] == "revolute"
        ]
        transform = urdf.link_pose(
            "h2s2r_palm", {name: float(value) for name, value in zip(revolute, arm_q[side])}
        )
        pose = np.r_[
            transform[:3, 3],
            Rotation.from_matrix(transform[:3, :3]).as_euler("zyx"),
        ]
        low = np.asarray(PALM_MINIMUM, dtype=float)
        high = np.asarray(PALM_MAXIMUM, dtype=float)
        outside = (pose < low - 1e-6) | (pose > high + 1e-6)
        palm[side] = {
            "fabric_urdf": str(urdf_path.resolve()),
            "fabric_urdf_sha256": sha256(urdf_path),
            "joint_order": revolute,
            "static_fk_position_euler_zyx": pose.tolist(),
            "minimum": low.tolist(),
            "maximum": high.tolist(),
            "outside_dimensions": np.flatnonzero(outside).tolist(),
            "passes_existing_bounds": not bool(outside.any()),
        }

    with np.load(sidecar_path, allow_pickle=False) as data:
        sides = {}
        for side in ("right", "left"):
            sides[side] = {
                "pca_sha256": str(data[f"{side}_pca_sha256"].item()),
                "x0": data[f"{side}_x0"].tolist(),
                "original_low": data[f"{side}_original_low"].tolist(),
                "original_high": data[f"{side}_original_high"].tolist(),
                "runtime_low": data[f"{side}_runtime_low"].tolist(),
                "runtime_high": data[f"{side}_runtime_high"].tolist(),
                "width_ratio": data[f"{side}_width_ratio"].tolist(),
                "maximum_width_ratio": float(data[f"{side}_width_ratio"].max()),
            }
    status = (
        "CPU_ACTION_CALIBRATION_PASS"
        if all(value["passes_existing_bounds"] for value in palm.values())
        else "BLOCKED_STATIC_PALM_ZERO_OUTSIDE_EXISTING_V12_BOUNDS"
    )
    report = {
        "schema": "h2s2r_clean3_c3p1r_action_calibration_v1",
        "status": status,
        "method_adaptation": "fixed shared-static-posture centered action range with 5% original-width margin",
        "not_pure_demo_percentile_runtime_range": True,
        "margin_fraction": 0.05,
        "sidecar": {"path": str(sidecar_path.resolve()), "sha256": sha256(sidecar_path)},
        "controller_source": {"path": str(controller_path.resolve()), "sha256": sha256(controller_path)},
        "original_runtime_inputs_sha256": sha256(runtime_path),
        "sides": sides,
        "palm_static_fk": palm,
        "prohibited": [
            "additional margin tuning", "online reanchor", "per-frame GraspPose feedforward",
            "PCA refit", "PCA dimension change", "static initialization change",
        ],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
