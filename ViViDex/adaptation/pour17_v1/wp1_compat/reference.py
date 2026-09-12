"""Build the hash-bound bimanual Pour17 pre-IK reference.

The registered left raw-mean finger solution is reused.  The right hand is
decoded from MANO_RIGHT and retargeted with the same fixed ten-point/NLopt
construction on the actual right Sharpa subtree.  Arm IK and home-wrist
prefix/suffix completion are performed inside the registered Isaac scene.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

from .geometry import (
    build_calibration, common_yaw_translation, load_module, mano_landmarks,
    mano_world_to_robot_root_pose, ndarray_sha256, resample_linear,
    resample_pose_wxyz, sha256, sha256_bytes, transform_pose_series,
)


def _read_member(archive: Path, member: str) -> bytes:
    with tarfile.open(archive, "r:gz") as stream:
        item = stream.extractfile(member)
        if item is None:
            raise RuntimeError(f"bundle member is not a file: {member}")
        return item.read()


def _selector_for_side(decoder, source: Path, model: dict, side: str) -> tuple:
    contract = decoder.official_output_contract(source, model["kintree_table"])
    final_sources = list(contract["final_sources"])
    if side == "right":
        # The registered ManoLayer uses right middle fingertip vertex 444 and
        # left vertex 445; the remaining four tip vertices are identical.
        final_sources = [
            (kind, 444 if kind == "tip_vertex" and value == 445 else value)
            for kind, value in final_sources
        ]
    return tuple(final_sources)


def _interface(fk_module, config: dict, manifest: dict, side: str):
    start, stop = config["manifest"][f"{side}_finger_slice"]
    names = tuple(manifest["robot"]["controlled_joint_names_in_order"][start:stop])
    limits = np.asarray(manifest["robot"]["controlled_joint_limits_rad"][start:stop])
    landmarks = tuple(
        fk_module.LandmarkRecord(
            item["name"], item["link"].replace("left_", f"{side}_", 1),
            np.asarray(item["local_m"], dtype=np.float64),
        )
        for item in config["landmarks"]
    )
    return fk_module.SharpaLeftFKCandidateV1(
        urdf_path=Path(config["paths"]["robot_urdf"]),
        root_link=f"{side}_hand_C_MC", joint_names=names,
        soft_limits_rad=limits, landmarks=landmarks, limit_tolerance_rad=1e-6,
    )


def _decode_side(decoder, model: dict, final_sources: tuple, perception,
                 side: str, calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    targets = np.empty((142, 10, 3), dtype=np.float64)
    wrist_poses = np.empty((142, 7), dtype=np.float64)
    for frame in range(142):
        pose45 = np.asarray(perception[f"{side}_mano_pose45"][frame], dtype=np.float64)
        betas = np.asarray(perception[f"{side}_mano_betas"][frame], dtype=np.float64)
        global_aa = np.asarray(perception[f"{side}_mano_rot"][frame], dtype=np.float64)
        translation = np.asarray(perception[f"{side}_mano_trans"][frame], dtype=np.float64)
        local = decoder.decode_mano(
            model, transl=np.zeros(3), global_aa=np.zeros(3), pose45=pose45,
            betas=betas, add_hands_mean=False,
            left_shapedirs_x_flip=(side == "left"), final_sources=final_sources,
        )
        joints_local = np.asarray(local["joints21_m"], dtype=np.float64)
        joints_local -= joints_local[0]
        landmarks_h = mano_landmarks(joints_local)
        targets[frame] = (
            np.asarray(calibration["R_S_from_H"]) @ landmarks_h.T
        ).T + np.asarray(calibration["t_S_from_H_m"])
        world = decoder.decode_mano(
            model, transl=translation, global_aa=global_aa, pose45=pose45,
            betas=betas, add_hands_mean=False,
            left_shapedirs_x_flip=(side == "left"), final_sources=final_sources,
        )
        wrist_poses[frame] = mano_world_to_robot_root_pose(
            np.asarray(world["joints21_m"]), global_aa, calibration
        )
    if not np.isfinite(targets).all() or not np.isfinite(wrist_poses).all():
        raise RuntimeError(f"non-finite decoded {side} geometry")
    return targets, wrist_poses


def _solve_right(interface, solver, targets: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    q = np.empty((142, 22), dtype=np.float64)
    rows = []
    previous = np.mean(interface.soft_limits_rad, axis=1)
    for frame in range(142):
        result = solver.solve_single_frame(interface, targets[frame], previous, previous)
        q[frame] = np.asarray(result["q_final_rad"], dtype=np.float64)
        evaluation = solver.evaluate_objective(interface, q[frame], targets[frame], previous)
        code = int(result["nlopt_return_code"])
        if code <= 0 or code in (5, 6) or not np.isfinite(q[frame]).all():
            raise RuntimeError(f"right retarget failed at source frame {frame}: NLopt {code}")
        rows.append({
            "source_frame": frame, "nlopt_return_code": code,
            "objective_calls": int(result["objective_call_count"]),
            "tracking_m2": float(evaluation.tracking),
            "rms_mm": float(np.sqrt(evaluation.tracking / 10.0) * 1000.0),
        })
        previous = q[frame].copy()
    return q, rows


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=root / "configs/pour17_v1/wp1_config.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = {name: Path(value).resolve() for name, value in config["paths"].items()}
    for name, expected in config["expected_sha256"].items():
        if name == "perception_member":
            continue
        actual = sha256(paths[name])
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {name}: {actual}")

    perception_payload = _read_member(paths["bundle_archive"], config["members"]["perception"])
    if sha256_bytes(perception_payload) != config["expected_sha256"]["perception_member"]:
        raise RuntimeError("registered perception member hash mismatch")
    perception = np.load(io.BytesIO(perception_payload), allow_pickle=False)
    frame_ids = np.asarray(perception["frame_ids"], dtype=np.int64)
    if not np.array_equal(frame_ids, np.arange(142)):
        raise RuntimeError("source frames are not exactly 0..141")
    finite_fields = [
        f"{side}_mano_{field}" for side in ("right", "left")
        for field in ("trans", "rot", "pose45", "betas")
    ] + ["object_0_pose_wxyz", "object_1_pose_wxyz"]
    for name in finite_fields:
        if not np.isfinite(perception[name]).all():
            raise RuntimeError(f"non-finite registered input: {name}")

    manifest = json.loads(_read_member(
        paths["bundle_archive"], config["members"]["manifest"]
    ).decode("utf-8"))
    canonical = json.loads(_read_member(
        paths["bundle_archive"], config["members"]["canonical_reset"]
    ).decode("utf-8"))
    diagnostics = paths["decoder"].parent
    sys.path.insert(0, str(diagnostics))
    decoder = load_module(paths["decoder"], "pour17_v1_registered_mano_decoder")
    fk_module = load_module(paths["fk_interface"], "m1a_sharpa_left_fk_candidate_v1")
    solver = load_module(paths["single_frame_solver"], "pour17_v1_registered_solver")
    interfaces = {side: _interface(fk_module, config, manifest, side)
                  for side in ("right", "left")}
    models = {side: decoder.load_model(paths[f"mano_{side}"])
              for side in ("right", "left")}
    selectors = {side: _selector_for_side(
        decoder, paths["mano_layer_source"], models[side], side
    ) for side in ("right", "left")}
    calibrations = {side: build_calibration(
        decoder, models[side], selectors[side], interfaces[side],
        shapedirs_x_flip=(side == "left"), side=side,
    ) for side in ("right", "left")}
    registered_left = json.loads(paths["left_calibration"].read_text(encoding="utf-8"))
    left_rotation_error = float(np.max(np.abs(
        calibrations["left"]["R_S_from_H"]
        - np.asarray(registered_left["transform"]["R_S_from_H"])
    )))
    left_translation_error = float(np.max(np.abs(
        calibrations["left"]["t_S_from_H_m"]
        - np.asarray(registered_left["transform"]["t_S_from_H_m"])
    )))
    if max(left_rotation_error, left_translation_error) > 1e-10:
        raise RuntimeError("same-rule calibration no longer reproduces registered left A")

    targets = {}
    wrists_reconstruction = {}
    for side in ("right", "left"):
        targets[side], wrists_reconstruction[side] = _decode_side(
            decoder, models[side], selectors[side], perception, side,
            calibrations[side],
        )

    with np.load(paths["left_candidate_arrays"], allow_pickle=False) as existing:
        variants = [str(value) for value in existing["variant_names"]]
        raw_index = variants.index("proposed_raw_pose")
        left_q = np.asarray(existing["q_final_rad"][raw_index], dtype=np.float64)
        saved_targets = np.asarray(
            existing["target_landmarks_S_m"][raw_index], dtype=np.float64
        )
        completed = np.asarray(existing["completed_mask"][raw_index], dtype=bool)
        left_names = tuple(str(value) for value in existing["joint_names"])
    if not completed.all() or left_q.shape != (142, 22):
        raise RuntimeError("registered left raw candidate is incomplete")
    if left_names != interfaces["left"].joint_names:
        raise RuntimeError("registered left candidate joint order changed")
    left_target_error = float(np.max(np.abs(saved_targets - targets["left"])))
    if left_target_error > 1e-12:
        raise RuntimeError(f"left raw target reproduction changed: {left_target_error}")
    right_q, right_rows = _solve_right(interfaces["right"], solver, targets["right"])

    source_centers = np.stack((perception["object_0_pose_wxyz"][0, :3],
                               perception["object_1_pose_wxyz"][0, :3]))
    target_centers = np.stack((
        canonical["objects_initial_state"]["object_0_cup"]["pose_xyz_wxyz_env_frame"][:3],
        canonical["objects_initial_state"]["object_1_bottle"]["pose_xyz_wxyz_env_frame"][:3],
    ))
    world = common_yaw_translation(source_centers, target_centers)
    rotation = world["R_env_from_reconstruction"]
    translation = world["t_env_from_reconstruction_m"]
    wrists_env = {
        side: transform_pose_series(wrists_reconstruction[side], rotation, translation)
        for side in ("right", "left")
    }
    objects_env = np.stack([
        transform_pose_series(np.asarray(perception[f"object_{index}_pose_wxyz"]),
                              rotation, translation)
        for index in (0, 1)
    ], axis=1)

    source_time = frame_ids.astype(np.float64) / 15.0
    core_time = np.arange(189, dtype=np.float64) / 20.0
    if core_time[-1] != source_time[-1]:
        raise RuntimeError("20 Hz core does not end at source frame 141")
    core = {
        "right_finger_q_rad": resample_linear(right_q, source_time, core_time),
        "left_finger_q_rad": resample_linear(left_q, source_time, core_time),
        "right_wrist_pose_wxyz": resample_pose_wxyz(wrists_env["right"], source_time, core_time),
        "left_wrist_pose_wxyz": resample_pose_wxyz(wrists_env["left"], source_time, core_time),
        "object_pose_wxyz": np.stack([
            resample_pose_wxyz(objects_env[:, index], source_time, core_time)
            for index in range(2)
        ], axis=1),
    }
    source_index_core = np.minimum(
        np.floor(core_time * 15.0 + 1e-12).astype(np.int64), 141
    )
    output = paths["preik_reference"]
    output.parent.mkdir(parents=True, exist_ok=True)
    validity_names = np.asarray(("right_mano", "left_mano", "object_0", "object_1"))
    source_valid = np.column_stack((
        perception["right_mano_valid"], perception["left_mano_valid"],
        perception["object_0_valid"], perception["object_1_valid"],
    )).astype(bool)
    np.savez_compressed(
        output,
        schema_version=np.asarray("pour17_v1_preik"), task_id=np.asarray("pour17"),
        source_sha256=np.asarray(config["expected_sha256"]["perception_member"]),
        contract_sha256=np.asarray(sha256(config_path)),
        source_frame=frame_ids, derived_source_time_s=source_time,
        core_control_time_s=core_time,
        source_index_core=source_index_core, source_valid=source_valid,
        source_valid_names=validity_names,
        right_finger_q_rad_core=core["right_finger_q_rad"],
        left_finger_q_rad_core=core["left_finger_q_rad"],
        right_wrist_pose_wxyz_core=core["right_wrist_pose_wxyz"],
        left_wrist_pose_wxyz_core=core["left_wrist_pose_wxyz"],
        object_pose_wxyz_core=core["object_pose_wxyz"],
        joint_names=np.asarray(manifest["robot"]["controlled_joint_names_in_order"]),
        joint_lower_rad=np.asarray(manifest["robot"]["controlled_joint_limits_rad"])[:, 0],
        joint_upper_rad=np.asarray(manifest["robot"]["controlled_joint_limits_rad"])[:, 1],
        canonical_home_q_rad=np.asarray(canonical["robot_initial_state"]["joint_q_rad"]),
        right_target_landmarks_S_m=targets["right"],
        left_target_landmarks_S_m=targets["left"],
    )
    manifest_output = paths["preik_manifest"]
    record = {
        "status": "PASS_PREIK_REFERENCE__WORLD_TRANSFORM_DEVELOPMENT_CANDIDATE_NOT_FROZEN",
        "scope": "right/left MANO decode, finger retarget, common world candidate, 20Hz core; arm IK pending Isaac",
        "source": {
            "bundle": {"path": str(paths["bundle_archive"]), "sha256": sha256(paths["bundle_archive"])},
            "perception_member": {"member": config["members"]["perception"],
                                  "sha256": sha256_bytes(perception_payload)},
            "mano_right": {"path": str(paths["mano_right"]), "sha256": sha256(paths["mano_right"])},
            "mano_left": {"path": str(paths["mano_left"]), "sha256": sha256(paths["mano_left"])},
            "left_candidate_arrays": {"path": str(paths["left_candidate_arrays"]),
                                      "sha256": sha256(paths["left_candidate_arrays"]),
                                      "q_content_sha256": ndarray_sha256(left_q)},
        },
        "mean": "proposed_raw_pose; no hands_mean added",
        "calibration": {
            side: {key: value.tolist() if isinstance(value, np.ndarray) else value
                   for key, value in calibrations[side].items()}
            for side in ("right", "left")
        },
        "left_reproduction": {"target_max_abs_m": left_target_error,
                              "R_max_abs": left_rotation_error,
                              "t_max_abs_m": left_translation_error},
        "right_retarget": {
            "algorithm": "NLopt LD_SLSQP, alpha=0.004, previous solution recursion",
            "source_frames": 142, "all_success": True,
            "tracking_rms_mm": {
                "min": min(row["rms_mm"] for row in right_rows),
                "median": float(np.median([row["rms_mm"] for row in right_rows])),
                "max": max(row["rms_mm"] for row in right_rows),
            },
            "rows": right_rows,
        },
        "world_alignment": {
            "status": "DEVELOPMENT_CANDIDATE_NOT_FROZEN",
            "rule": "single gravity-preserving yaw from object center baseline plus translation from center mean; scale=1",
            **{key: value.tolist() if isinstance(value, np.ndarray) else value
               for key, value in world.items()},
        },
        "sampling": {
            "source": "142 frames at derived frame/15 seconds",
            "core": "189 samples at 20Hz from 0 through 9.4s inclusive",
            "prefix_suffix": "2s approach and 2s return are completed after measured canonical-home wrist FK and arm IK",
            "position_and_joint": "linear", "rotation": "shortest-path SLERP",
            "validity": "reported only; no row removal, fill, confidence weighting, or selection",
        },
        "outputs": {"preik_reference": {"path": str(output), "sha256": sha256(output)}},
        "prohibited_inputs_used": {
            "perception_right_hand_q": False, "perception_left_hand_q": False,
            "ours_reference": False, "confidence": False, "dexonomy": False,
        },
    }
    manifest_output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": record["status"], "preik_reference": str(output),
        "manifest": str(manifest_output), "right_frames": 142,
        "world_yaw_deg": float(np.degrees(world["yaw_rad"])),
        "world_center_residual_mm": (world["center_residual_m"] * 1000.0).tolist(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
