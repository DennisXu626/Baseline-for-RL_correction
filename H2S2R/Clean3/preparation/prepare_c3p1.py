"""Build the reviewer-frozen Clean3 C3-P1 inputs without importing ours control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ElementTree

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
OLD_STAGE = WORKSPACE / "RL/Baselines/task_plate/h2s2r_clean3_staging_20260912"
BASE = OLD_STAGE / "extracted/base/clean3_base_package_20260909"
SOURCE_SNAPSHOT = WORKSPACE / "reports/H2S2R_TRAINING_PILOT_20260912/source_after"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vertices(path: Path) -> np.ndarray:
    result = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.startswith("v "):
            result.append([float(value) for value in line.split()[1:4]])
    value = np.asarray(result, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError(f"invalid OBJ vertices: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "prepared")
    parser.add_argument("--urdf", type=Path, default=HERE / "assets/vega_1p_sharpa.urdf")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    candidate_path = OLD_STAGE / "prepared/clean3_perception_candidate.npz"
    old_path = BASE / "task/Clean3/A_Design/L2_Reference/clean3_reference_v1.npz"
    data_root = BASE / "inputs/clean_tableware_3"
    rts_paths = [data_root / f"poseqa/rts_clean_tableware_3_object_{i}.npz" for i in (0, 1)]
    mesh_paths = [data_root / f"objects/object_{i}/object_mesh_scaled_final.obj" for i in (0, 1)]
    required = [candidate_path, old_path, args.urdf, *rts_paths, *mesh_paths]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    candidate = np.load(candidate_path, allow_pickle=False)
    old = np.load(old_path, allow_pickle=True)
    for side in ("right", "left"):
        if candidate[f"{side}_hand_q"].shape != (300, 22):
            raise ValueError(f"{side} hand corpus is not 300x22")
        if not bool(candidate[f"{side}_valid"].all()):
            raise ValueError(f"{side} does not have all 300 valid rows")

    # The archived RTS arrays are upstream-smoothed estimates.  The only task
    # adaptation below is the reviewer-approved plate-frame planar projection.
    rts = [np.load(path, allow_pickle=False)["object_ob_in_world_smooth"] for path in rts_paths]
    if any(value.shape != (300, 4, 4) for value in rts):
        raise ValueError("RTS object arrays must both be 300x4x4")
    relative_input = np.einsum(
        "nji,nj->ni", rts[0][:, :3, :3], rts[1][:, :3, 3] - rts[0][:, :3, 3]
    )
    canonical_from_input = Rotation.from_euler("x", 90.0, degrees=True)
    relative = canonical_from_input.apply(relative_input)

    plate_vertices = vertices(mesh_paths[0])
    sponge_vertices = canonical_from_input.apply(vertices(mesh_paths[1]))
    radius = np.linalg.norm(plate_vertices[:, [0, 2]], axis=1)
    edges = np.linspace(0.0, 0.09, 19)
    profile_r, profile_z = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        keep = (radius >= low) & (radius < high)
        if keep.any():
            profile_r.append((low + high) / 2.0)
            profile_z.append(float(plate_vertices[keep, 1].max()))
    profile_r = np.asarray(profile_r)
    profile_z = np.asarray(profile_z)
    face_offset = -float(sponge_vertices[:, 2].min())

    plate_pose = np.r_[old["obj_pos_0"][0], old["obj_quat_0"][0]].astype(np.float32)
    old_sponge_pose = np.r_[old["obj_pos_1"][0], old["obj_quat_1"][0]].astype(np.float32)
    initial_xy = (old["obj_pos_1"][0] - old["obj_pos_0"][0])[:2]
    sponge_xy = relative[:, :2] - relative[0, :2] + initial_xy
    q1 = old["obj_quat_1"][0]
    sponge_yaw = (
        Rotation.from_quat(q1[[1, 2, 3, 0]]) * canonical_from_input.inv()
    ).as_euler("xyz")[2]
    fx = np.arange(sponge_vertices[:, 0].min() + 0.0075, sponge_vertices[:, 0].max(), 0.01)
    fy = np.arange(sponge_vertices[:, 1].min() + 0.0065, sponge_vertices[:, 1].max(), 0.01)
    xx, yy = np.meshgrid(fx, fy, indexing="ij")
    footprint = Rotation.from_euler("z", sponge_yaw).apply(
        np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    )[:, :2]
    points = sponge_xy[:, None, :] + footprint[None, :, :]
    point_radius = np.linalg.norm(points, axis=2)
    point_top = np.interp(point_radius, profile_r, profile_z)
    support_z = point_top.max(axis=1)
    relative_height = support_z + face_offset
    support_gap = relative_height[:, None] - face_offset - point_top
    support_contact = (support_gap >= -0.002) & (support_gap < 0.005) & (point_radius <= 0.085)
    cell = np.floor((points + 0.09) / 0.01).astype(int).clip(0, 17)
    covered = np.zeros((18, 18), dtype=bool)
    covered[cell[..., 0][support_contact], cell[..., 1][support_contact]] = True
    grid_x, grid_y = np.meshgrid(
        np.arange(-0.085, 0.09, 0.01), np.arange(-0.085, 0.09, 0.01), indexing="ij"
    )
    disk = grid_x * grid_x + grid_y * grid_y <= 0.085**2
    contact_rows = support_contact.any(axis=1)
    planar_step = np.linalg.norm(np.diff(sponge_xy, axis=0), axis=1)
    ideal_coverage = float((covered & disk).sum() / disk.sum())
    ideal_travel = float(planar_step[contact_rows[1:] & contact_rows[:-1]].sum())
    sponge_pose = np.empty((300, 7), dtype=np.float32)
    sponge_pose[:, :2] = plate_pose[:2] + sponge_xy
    sponge_pose[:, 2] = plate_pose[2] + relative_height
    sponge_pose[:, 3:] = old_sponge_pose[3:]
    plate_reference = np.repeat(plate_pose[None], 300, axis=0)
    support_delta = float(relative_height[0] - (old["obj_pos_1"][0, 2] - old["obj_pos_0"][0, 2]))

    # One static IK correction only: lift the initial right wrist by the same
    # mesh-support delta.  No reference row or object-hand feedforward is used.
    os.environ["VEGA_URDF"] = str(args.urdf.resolve())
    sys.path.insert(0, str(BASE / "code"))
    from rl_rebuild.correction.kinematics import ArmIK  # pylint: disable=import-outside-toplevel

    right_q_original = np.asarray(old["right_q"][0], dtype=np.float64)
    ik = ArmIK("right", anchor_link="arm_center", anchor_T=old["anchor_T"])
    initial_wrist_position, initial_wrist_rotation = ik.fk(right_q_original)
    ik_result = ik.solve(
        initial_wrist_position + np.array([0.0, 0.0, support_delta]),
        initial_wrist_rotation,
        q0=right_q_original,
        iters=300,
        pos_tol=2e-4,
        rot_tol=5e-3,
    )
    if not ik_result["ok"] or bool(np.asarray(ik_result["at_limit"]).any()):
        raise RuntimeError(f"one-time static right-arm IK failed: {ik_result}")
    right_q = np.asarray(ik_result["q"], dtype=np.float32)
    left_q = np.asarray(old["left_q"][0], dtype=np.float32)
    reset_q58 = np.concatenate(
        [right_q, left_q, old["right_f_grasp"], old["left_f_grasp"]]
    ).astype(np.float32)
    if reset_q58.shape != (58,) or not np.isfinite(reset_q58).all():
        raise ValueError("reset q is not finite 58-D")

    sys.path.insert(0, str(SOURCE_SNAPSHOT))
    from h2s2r.contract import simulation_joint_names  # pylint: disable=import-outside-toplevel
    from h2s2r.synergy import fit_pca_synergy, save_synergy  # pylint: disable=import-outside-toplevel

    joint_limits = {}
    for joint in ElementTree.parse(args.urdf).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
            joint_limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    limit_tolerance_rad = 1e-6
    limit_rows, violations, boundary_rows = [], [], []
    for name, value in zip(simulation_joint_names(), reset_q58):
        if name not in joint_limits:
            violations.append({"joint": name, "reason": "missing_limit"})
            continue
        lower, upper = joint_limits[name]
        row = {"joint": name, "value_rad": float(value), "lower_rad": lower, "upper_rad": upper}
        limit_rows.append(row)
        if min(abs(float(value) - lower), abs(float(value) - upper)) <= limit_tolerance_rad:
            boundary_rows.append(row)
        if float(value) < lower - limit_tolerance_rad or float(value) > upper + limit_tolerance_rad:
            violations.append(row)

    candidate_sha = sha256(candidate_path)
    synergy_code = SOURCE_SNAPSHOT / "h2s2r/synergy.py"
    pca_report = {}
    for side in ("right", "left"):
        field = f"{side}_hand_q"
        source = (
            f"sha256={candidate_sha};field={field};rows=0:300;valid_field={side}_valid;"
            f"fit=existing_pca5_percentile_0.5_99.5;code_sha256={sha256(synergy_code)}"
        )
        synergy = fit_pca_synergy(candidate[field], source=source)
        synergy_path = args.output / f"{side}_synergy_pca5.npz"
        save_synergy(synergy_path, synergy)
        grasp = np.asarray(old[f"{side}_f_grasp"], dtype=np.float32)
        coordinate = (grasp - synergy.center) @ synergy.matrix.T
        clipped = np.clip(coordinate, synergy.minimum, synergy.maximum)
        decoded_unclipped = synergy.center + coordinate @ synergy.matrix
        decoded = synergy.center + clipped @ synergy.matrix
        error = decoded - grasp
        error_unclipped = decoded_unclipped - grasp
        pca_report[side] = {
            "source": source,
            "synergy_path": str(synergy_path.resolve()),
            "synergy_sha256": sha256(synergy_path),
            "initial_coordinate": coordinate.tolist(),
            "clipped_coordinate": clipped.tolist(),
            "minimum": synergy.minimum.tolist(),
            "maximum": synergy.maximum.tolist(),
            "joint_error_rad": error.tolist(),
            "maximum_absolute_joint_error_rad": float(np.abs(error).max()),
            "unclipped_subspace_joint_error_rad": error_unclipped.tolist(),
            "unclipped_subspace_maximum_absolute_joint_error_rad": float(
                np.abs(error_unclipped).max()
            ),
            "passes_0p25_rad_gate": bool(np.abs(error).max() <= 0.25),
        }

    runtime_path = args.output / "clean3_c3p1_runtime_inputs.npz"
    np.savez_compressed(
        runtime_path,
        plate_reference_pose_wxyz=plate_reference,
        sponge_reference_pose_wxyz=sponge_pose,
        reference_valid=np.ones(300, dtype=bool),
        reset_q58=reset_q58,
        reset_object_0_pose_wxyz=plate_pose,
        reset_object_1_pose_wxyz=sponge_pose[0],
        static_right_f_grasp=np.asarray(old["right_f_grasp"], dtype=np.float32),
        static_left_f_grasp=np.asarray(old["left_f_grasp"], dtype=np.float32),
        source_frame_ids=np.arange(300, dtype=np.int64),
        source_fps=np.float32(30.0),
        control_hz=np.float32(20.0),
        physics_hz=np.float32(240.0),
        support_height_delta_m=np.float32(support_delta),
        footprint_xy=footprint.astype(np.float32),
        plate_profile_r=profile_r.astype(np.float32),
        plate_profile_z=profile_z.astype(np.float32),
        sponge_face_offset=np.float32(face_offset),
    )
    manifest = {
        "schema": "h2s2r_clean3_c3p1_inputs_v1",
        "status": "CPU_PREPARED_PCA_GATE_PASS" if all(v["passes_0p25_rad_gate"] for v in pca_report.values()) else "BLOCKED_PCA_INITIAL_STATE_JUMP",
        "route": "H2S2R-adapted Clean3 / RTS planar reference / shared static initialization",
        "prohibited_runtime_inputs": [
            "ours policy/checkpoint", "right_f/left_f trajectory", "per-frame GraspPose feedforward",
            "squeeze", "pin/release", "settled_oh", "certification", "ours reward",
        ],
        "candidate": {"path": str(candidate_path.resolve()), "sha256": candidate_sha},
        "old_static_source": {
            "path": str(old_path.resolve()), "sha256": sha256(old_path),
            "consumed": ["right_q[0]", "left_q[0]", "obj_pos/quat_{0,1}[0]", "right_f_grasp", "left_f_grasp", "anchor_T"],
        },
        "rts_sources": [{"path": str(path.resolve()), "sha256": sha256(path), "field": "object_ob_in_world_smooth", "rows": 300} for path in rts_paths],
        "mesh_sources": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in mesh_paths],
        "reference": {
            "rows": 300, "source_fps": 30.0, "control_hz": 20.0,
            "plate_pose_constant": True, "sponge_yaw_constant": True,
            "support_height_no_press": True, "support_height_delta_m": support_delta,
            "extra_filtering": False, "cropping": False, "path_clamp": False,
            "time_stretch": False, "later_ours_reference_rows": 0,
            "ideal_geometry_only_coverage": ideal_coverage,
            "ideal_geometry_only_two_endpoint_contact_travel_m": ideal_travel,
            "ideal_geometry_only_contact_rows": int(contact_rows.sum()),
            "maximum_planar_step_m": float(planar_step.max()),
            "maximum_center_radius_m": float(np.linalg.norm(sponge_xy, axis=1).max()),
            "limitations": "sampled mesh profile/footprint only; not a physics or control pass",
        },
        "reset_joint_limits": {
            "checked": len(limit_rows), "numeric_tolerance_rad": limit_tolerance_rad,
            "boundary_rows": boundary_rows, "violations": violations, "passes": not violations,
        },
        "static_ik": {
            "side": "right", "translation_m": [0.0, 0.0, support_delta],
            "position_error_m": float(ik_result["pos_err"]),
            "rotation_error_rad": float(ik_result["rot_err"]),
            "joint_delta_rad": (right_q - right_q_original).tolist(),
            "at_limit": np.asarray(ik_result["at_limit"]).tolist(),
            "urdf": str(args.urdf.resolve()), "urdf_sha256": sha256(args.urdf),
        },
        "pca": pca_report,
        "runtime_inputs": {"path": str(runtime_path.resolve()), "sha256": sha256(runtime_path)},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
