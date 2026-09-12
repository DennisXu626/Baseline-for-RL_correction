"""W1/S1 reference construction from saved geometry; no MANO or finger solve.

Original perception is retained, not decoded. Wrists/fingers at exact video
cut times are interpolated from the registered WP1 20 Hz own reference.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from .geometry import (sha256, quat_to_matrix_wxyz, matrix_to_quat_wxyz,
                       axis_angle_matrix, resample_pose_wxyz, resample_linear,
                       transform_pose_series, ArmIK, UrdfKinematics)
from .scene import pose_interpolation
from .schema import schema


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def put(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def transform_points(pose, points):
    return points @ quat_to_matrix_wxyz(pose[3:]).T + pose[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--archive", type=Path, default=Path("D:/UCBP/pour17_baseline_bundle_20260829.tar.gz"))
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--solve-arms", action="store_true")
    a = ap.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    base = a.baseline
    old = base / "artifacts/pour17_v1/wp1_20260912_02"
    pre = base / "artifacts/pour17_v1/wp1_20260912_01"
    pre_manifest = json.loads((pre / "reference_preik_manifest.json").read_text())
    config = json.loads((base / "configs/pour17_v1/wp1_config.json").read_text())
    if sha256(a.archive) != config["expected_sha256"]["bundle_archive"]:
        raise RuntimeError("registered bundle hash mismatch")
    if sha256(a.urdf) != config["expected_sha256"]["robot_urdf"]:
        raise RuntimeError("registered URDF hash mismatch")
    with tarfile.open(a.archive, "r:gz") as archive:
        blob = archive.extractfile("pour17/perception/pour17_perception.npz").read()
    from hashlib import sha256 as hash_bytes
    if hash_bytes(blob).hexdigest() != config["expected_sha256"]["perception_member"]:
        raise RuntimeError("perception member hash mismatch")
    perception = load(io.BytesIO(blob))
    ref, preik = load(old / "reference.npz"), load(pre / "reference_preik.npz")
    if sha256(old / "reference.npz") != "46c56cdc1b1775d57be405d4188d9616ffbb1606684c3989f1d5108e95b042ee":
        raise RuntimeError("WP1 reference hash mismatch")
    w1 = pre_manifest["world_alignment"]
    rotation = np.array(w1["R_env_from_reconstruction"])
    translation = np.array(w1["t_env_from_reconstruction_m"])
    # Whitelist: do not retain upstream DexPilot/Ours robot fields.
    originals = {k: v for k, v in perception.items() if "mano_" in k or k.startswith("object_") or k in
                 ("frame_ids", "timestamps", "object_0_pose", "object_1_pose")}
    pose_keys = [k for k in perception if "object" in k and perception[k].shape == (142, 7)]
    if len(pose_keys) != 2:
        raise RuntimeError(f"expected two named original object poses, got {pose_keys}")
    pose_keys.sort()
    objects_source = np.stack([transform_pose_series(perception[k], rotation, translation) for k in pose_keys], axis=1)
    originals.update({k: perception[k] for k in pose_keys})
    np.savez_compressed(out / "preserved_source_142.npz", **originals)
    cut_candidates = np.flatnonzero(objects_source[:, 1, 2] - objects_source[0, 1, 2] > .12)
    if len(cut_candidates) == 0:
        put(out / "s1_geometry.json", {"status": "BLOCKED_NO_CUT_GT_012", "max_lift_m": float(np.max(objects_source[:, 1, 2] - objects_source[0, 1, 2]))})
        return
    cut = int(cut_candidates[0])
    mouth_axes = np.asarray([quat_to_matrix_wxyz(q) @ np.array([0., 1., 0.]) for q in objects_source[:, 1, 3:]])
    tilts = np.arccos(np.clip(mouth_axes[:, 2], -1, 1))
    key = cut + 1 + int(np.argmax(tilts[cut + 1:]))
    cup_mouth = transform_points(objects_source[key, 0], np.array([0., .066, 0.]))
    toward = cup_mouth - objects_source[cut, 1, :3]
    toward[2] = 0
    axis = np.cross([0., 0., 1.], toward)
    if np.linalg.norm(axis) < 1e-10:
        raise RuntimeError("S1 degenerate cut-to-cup horizontal axis")
    axis /= np.linalg.norm(axis)
    core_t = preik["core_control_time_s"]
    source_t = preik["derived_source_time_s"]
    before_t = np.r_[core_t[core_t < cut / 15.], cut / 15.]
    after_t = core_t[core_t > key / 15.]
    dist = np.linalg.norm(objects_source[key, 1, :3] - objects_source[cut, 1, :3])
    if dist < 1e-10:
        raise RuntimeError("S1 zero move length: source division by num_step not defined")
    full_steps = int(dist // .02)
    if full_steps == 0:
        raise RuntimeError("source Pour rotation_step is undefined for a move shorter than0.02m")
    fractions = np.r_[np.arange(1, full_steps + 1) * .02 / dist, 1.]
    rotation_fractions = np.r_[np.arange(1, full_steps + 1) / full_steps, 1.]
    # As pinned source: full 2cm increments plus exact final endpoint (not new search).
    timeline_left = cut / 15. + np.arange(1, len(fractions) + 1) / len(fractions) * (key - cut) / 15.
    core = {}
    for oi in (0, 1):
        core[f"object_{oi}"] = resample_pose_wxyz(objects_source[:, oi], source_t, core_t)
    for side in ("right", "left"):
        core[f"{side}_wrist"] = preik[f"{side}_wrist_pose_wxyz_core"]
        core[f"{side}_fingers"] = preik[f"{side}_finger_q_rad_core"]
        local = resample_linear(preik[f"{side}_target_landmarks_S_m"], source_t, core_t)
        core[f"{side}_hand"] = np.array([transform_points(p, pts) for p, pts in zip(core[f"{side}_wrist"], local)])
    palm_local = np.array([pre_manifest["calibration"][s]["robot_palm_origin_S_m"] for s in ("right", "left")])
    for si, side in enumerate(("right", "left")):
        core[f"{side}_palm"] = np.array([transform_points(p, palm_local[si]) for p in core[f"{side}_wrist"]])
    def sample(name, t):
        value = core[name]
        fn = resample_pose_wxyz if name.startswith("object_") or name.endswith("_wrist") else resample_linear
        return fn(value, core_t, np.asarray(t))
    synthesized = {name: sample(name, timeline_left) for name in core}
    cut_bottle = objects_source[cut, 1]
    goal_bottle = objects_source[key, 1]
    right_cut = {name: sample(name, [cut / 15.])[0] for name in core if name.startswith("right_")}
    for i, f in enumerate(fractions):
        delta_r = axis_angle_matrix(axis * (2 * np.pi / 3) * rotation_fractions[i])
        position = cut_bottle[:3] + f * (goal_bottle[:3] - cut_bottle[:3])
        synthesized["object_1"][i] = np.r_[position, matrix_to_quat_wxyz(delta_r @ quat_to_matrix_wxyz(cut_bottle[3:]))]
        for name, value in right_cut.items():
            if name.endswith("fingers"):
                synthesized[name][i] = value
            elif name.endswith("wrist"):
                synthesized[name][i] = np.r_[delta_r @ (value[:3] - cut_bottle[:3]) + position,
                                                matrix_to_quat_wxyz(delta_r @ quat_to_matrix_wxyz(value[3:]))]
            else:
                synthesized[name][i] = (value - cut_bottle[:3]) @ delta_r.T + position
    operation = {}
    bridge_n, hold_n = 10, 25
    for name in core:
        endpoint = sample(name, [key / 15.])[0]
        syn = synthesized[name]
        if name.startswith("object_") or name.endswith("wrist"):
            bridge = pose_interpolation(syn[-1], endpoint, bridge_n, include_stop=True)
        else:
            f = np.arange(1, bridge_n + 1).reshape((bridge_n,) + (1,) * endpoint.ndim) / bridge_n
            bridge = syn[-1] + f * (endpoint - syn[-1])
        operation[name] = np.concatenate((sample(name, before_t), syn, np.repeat(syn[-1:], hold_n, axis=0), bridge, sample(name, after_t)))
    stages = np.array(["video_pre_cut"] * len(before_t) + ["synthetic_pour"] * len(fractions) +
                      ["synthetic_hold"] * hold_n + ["bridge_to_video"] * bridge_n + ["video_return_release"] * len(after_t))
    provenance_time = np.r_[before_t, np.full(len(fractions) + hold_n + bridge_n, np.nan), after_t]
    mouth_b = np.array([transform_points(p, np.array([0., .087, 0.])) for p in synthesized["object_1"]])
    mouth_c = np.array([transform_points(p, np.array([0., .066, 0.])) for p in synthesized["object_0"]])
    end_tilt = float(np.arccos(np.clip((quat_to_matrix_wxyz(synthesized["object_1"][-1, 3:]) @ [0., 1., 0.])[2], -1, 1)))
    mouth_distance = float(np.linalg.norm(mouth_b[-1] - mouth_c[-1]))
    geom = {"cut_source_frame": cut, "pour_keyframe": key, "source_max_tilt_deg": float(np.degrees(tilts[key])),
            "axis_world": axis.tolist(), "move_distance_m": float(dist), "move_samples": len(fractions),
            "translation_fractions": fractions.tolist(), "rotation_fractions": rotation_fractions.tolist(),
            "left_video_sample_times_s": timeline_left.tolist(),
            "hold_samples": hold_n, "bridge_samples": bridge_n, "total_samples_with_home": len(stages) + 80,
            "end_bottle_tilt_deg": float(np.degrees(end_tilt)), "end_mouth_distance_m": mouth_distance,
            "end_bottle_mouth_m": mouth_b[-1].tolist(), "end_cup_mouth_m": mouth_c[-1].tolist(),
            "nominal_pour_geometry_valid": end_tilt >= np.pi / 2 and mouth_distance <= .12,
            "minimum_object_center_z_m": float(min(operation["object_0"][:, 2].min(), operation["object_1"][:, 2].min())),
            "table_surface_clearance": "not certified by center height; requires surface rollout",
            "W1": w1, "source_cut_queries": "exact video key indices; own WP1 20Hz wrists/fingers interpolated at cut time; no MANO decoding",
            "original_reference_sha256": sha256(old / "reference.npz"), "preik_sha256": sha256(pre / "reference_preik.npz"),
            "preik_manifest_sha256": sha256(pre / "reference_preik_manifest.json"), "script_sha256": sha256(Path(__file__))}
    geom["status"] = "TARGETS_BUILT_PENDING_EXECUTION" if geom["nominal_pour_geometry_valid"] else "BLOCKED_S1_NOMINAL_POUR_GEOMETRY"
    put(out / "s1_geometry.json", geom)
    np.savez_compressed(out / "s1_operation_targets.npz", **operation, stage=stages, source_time_s=provenance_time)
    # A v2 copy of the old nominal reference is only for execution diagnosis;
    # never confuse it with a completed S1 training reference.
    ref["palm_local_m"] = palm_local
    ref["palm_target_m"] = np.stack([np.array([transform_points(p, palm_local[si]) for p in ref[f"{side}_wrist_pose_wxyz"]]) for si, side in enumerate(("right", "left"))], axis=1)
    np.savez_compressed(out / "wp1_reference_schema_v2_diagnostic.npz", **ref)
    put(out / "observation_action_schema_v2.json", schema())
    if not a.solve_arms or not geom["nominal_pour_geometry_valid"]:
        return
    probe = json.loads((old / "runtime_probe.json").read_text())
    home_q = preik["canonical_home_q_rad"]
    full = {}
    for name, value in operation.items():
        if name.startswith("object_"):
            prefix, suffix = np.repeat(value[:1], 40, axis=0), np.repeat(value[-1:], 40, axis=0)
        else:
            side = name.split("_")[0]
            si = 0 if side == "right" else 1
            home_wrist = np.array(probe["canonical_home"][f"{side}_wrist_pose_env_wxyz"])
            if name.endswith("wrist"):
                prefix = pose_interpolation(home_wrist, value[0], 40, include_stop=False)
                suffix = pose_interpolation(value[-1], home_wrist, 40, include_stop=True)
            else:
                if name.endswith("fingers"):
                    home = home_q[14:36] if si == 0 else home_q[36:58]
                elif name.endswith("palm"):
                    home = transform_points(home_wrist, palm_local[si])
                else:
                    home = np.array(probe["canonical_home"][f"{side}_landmarks_env_m"])
                shape = (40,) + (1,) * home.ndim
                prefix = home + np.arange(40).reshape(shape) / 40 * (value[0] - home)
                suffix = value[-1] + np.arange(1, 41).reshape(shape) / 40 * (home - value[-1])
        full[name] = np.concatenate((prefix, value, suffix))
    names = tuple(str(s) for s in preik["joint_names"])
    limits = np.c_[preik["joint_lower_rad"], preik["joint_upper_rad"]]
    urdf = UrdfKinematics(a.urdf)
    arms, records = {}, {}
    for si, side in enumerate(("right", "left")):
        sl = slice(si * 7, si * 7 + 7)
        ik = ArmIK(urdf, side, "arm_center", np.array(probe["canonical_home"]["arm_center_pose_env_wxyz"]), names[sl], home_q[sl], limits[sl])
        arms[side], records[side] = ik.solve_trajectory(full[f"{side}_wrist"], seed=1701+si, restart_count=0)
    count = len(full["object_0"])
    ref.update(schema_version=np.array("pour17_reference_s1_v2"), control_time_s=np.arange(count) / 20,
        joint_q_rad=np.c_[arms["right"], arms["left"], full["right_fingers"], full["left_fingers"]],
        object_pose_wxyz=np.stack((full["object_0"], full["object_1"]), axis=1),
        stage=np.r_[np.repeat("approach", 40), stages, np.repeat("return", 40)],
        source_time_s=np.r_[np.full(40, np.nan), provenance_time, np.full(40, np.nan)],
        palm_target_m=np.stack((full["right_palm"], full["left_palm"]), axis=1))
    for side in ("right", "left"):
        ref[f"{side}_wrist_pose_wxyz"] = full[f"{side}_wrist"]
        ref[f"{side}_hand_target_m"] = full[f"{side}_hand"]
        ref[f"{side}_finger_q_rad"] = full[f"{side}_fingers"]
        ref[f"{side}_ik_failure_mask"] = np.array([not r["ok"] for r in records[side]])
    # Remove WP1 timeline fields instead of mislabeling synthetic samples.
    ref.pop("source_index", None)
    ref.pop("source_valid", None)
    np.savez_compressed(out / "reference_s1_v2.npz", **ref)
    put(out / "reference_s1_manifest.json", {"geometry": geom, "ik": records, "reference_sha256": sha256(out / "reference_s1_v2.npz")})


if __name__ == "__main__":
    main()
