"""Targeted CPU checks for newly ported WP1 mechanisms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from .augmentation import apply_global_placement, sample_placement
from .curriculum import CurriculumState
from .geometry import ArmIK, UrdfKinematics, sha256
from .reward import object_mimic_bimanual


def test_reward() -> dict:
    torch.manual_seed(4)
    n = 7
    actual_hand = torch.randn(n, 2, 10, 3) * 0.03
    target_hand = torch.randn(n, 2, 10, 3) * 0.03
    actual_object = torch.randn(n, 2, 7) * 0.1
    target_object = torch.randn(n, 2, 7) * 0.1
    for value in (actual_object, target_object):
        value[..., 3:] /= torch.linalg.vector_norm(value[..., 3:], dim=-1, keepdim=True)
    contacts = torch.rand(n, 2, 5) > 0.5
    table = torch.rand(n, 2) > 0.5
    pregrasp = torch.tensor([True, False, True, False, True, False, True])
    initial_z = torch.zeros(n, 2)
    reward, terms = object_mimic_bimanual(
        actual_hand_points=actual_hand, target_hand_points=target_hand,
        actual_object_pose=actual_object, target_object_pose=target_object,
        fingertip_contacts=contacts, object_table_contacts=table,
        pregrasp_success=pregrasp, initial_object_z=initial_z)
    hand_error = np.linalg.norm(
        actual_hand.numpy() - target_hand.numpy(), axis=-1).mean(axis=-1)
    expected_pregrasp = 10.0 * np.exp(-10.0 * hand_error)
    max_abs = float(np.max(np.abs(
        terms["side_reward"].detach().numpy()[~pregrasp.numpy()]
        - expected_pregrasp[~pregrasp.numpy()])))
    return {"status": "PASS" if max_abs < 2e-6 and torch.isfinite(reward).all() else "FAIL",
            "pregrasp_numpy_torch_max_abs": max_abs}


def test_augmentation_curriculum(output_dir: Path) -> dict:
    base = {
        "right_wrist_pose_wxyz": np.zeros((3, 7)),
        "left_wrist_pose_wxyz": np.zeros((3, 7)),
        "object_pose_wxyz": np.zeros((3, 2, 7)),
        "right_hand_target_m": np.zeros((3, 10, 3)),
        "left_hand_target_m": np.zeros((3, 10, 3)),
    }
    sample = sample_placement(1, np.random.default_rng(2026))
    changed = apply_global_placement(base, sample)
    offset = np.asarray(sample.translation_xy_m)
    checks = [
        np.allclose(changed["right_wrist_pose_wxyz"][..., :2], offset),
        np.allclose(changed["left_wrist_pose_wxyz"][..., :2], offset),
        np.allclose(changed["object_pose_wxyz"][..., :2], offset),
        not np.allclose(offset, 0.0), sample.yaw_rad == 0.0,
    ]
    state = CurriculumState()
    no_change = not state.update([0.95] * 25)
    first_change = state.update([1.0] * 25) and state.stage == 1
    second_change = state.update([1.0] * 25) and state.stage == 2
    state_path = output_dir / "curriculum_state_test.json"
    state.save(state_path)
    restored = CurriculumState.load(state_path)
    restore_ok = restored == state
    exemplar_path = output_dir / "augmentation_stage1_exemplar.npz"
    np.savez_compressed(exemplar_path, **changed,
                        stage=np.asarray(sample.stage),
                        translation_xy_m=offset,
                        yaw_rad=np.asarray(sample.yaw_rad))
    return {
        "status": "PASS" if all(checks) and no_change and first_change and second_change and restore_ok else "FAIL",
        "sample": {"stage": sample.stage, "translation_xy_m": list(sample.translation_xy_m),
                   "yaw_rad": sample.yaw_rad},
        "strict_threshold_no_change_at_0_95": no_change,
        "stage_transitions": state.transitions,
        "save_reload_equal": restore_ok,
        "exemplar": {"path": str(exemplar_path), "sha256": sha256(exemplar_path)},
    }


def test_arm_ik(urdf_path: Path) -> dict:
    urdf = UrdfKinematics(urdf_path)
    anchor = np.array((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    rows = {}
    for side, prefix in (("right", "R"), ("left", "L")):
        names = tuple(f"{prefix}_arm_j{index}" for index in range(1, 8))
        limits = np.asarray([[urdf.joints[name]["lower"], urdf.joints[name]["upper"]]
                             for name in names])
        q = np.clip(np.array((-.4, -.2, .1, -1.0, .2, .1, -.1))
                    * (-1.0 if side == "left" else 1.0), limits[:, 0], limits[:, 1])
        ik = ArmIK(urdf, side, "arm_center", anchor, names, q, limits)
        position, rotation = ik.fk(q)
        from .geometry import matrix_to_quat_wxyz
        result = ik.solve(np.r_[position, matrix_to_quat_wxyz(rotation)], q,
                          position_tolerance_m=1e-8, rotation_tolerance_rad=1e-8)
        rows[side] = {
            "ok": result["ok"], "q_max_abs": float(np.max(np.abs(result["q"] - q))),
            "position_error_m": result["position_error_m"],
            "rotation_error_rad": result["rotation_error_rad"],
        }
    return {"status": "PASS" if all(row["ok"] for row in rows.values()) else "FAIL",
            "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "reward_port": test_reward(),
        "augmentation_curriculum": test_augmentation_curriculum(args.output_dir),
        "arm_ik": test_arm_ik(args.robot_urdf),
    }
    report["status"] = "PASS" if all(
        value["status"] == "PASS" for value in report.values()) else "FAIL"
    output = args.output_dir / "targeted_cpu_tests.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "artifact": str(output)}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

