"""Registered Pour17 Isaac scene and final arm-reference construction.

Only asset, physics, sensor, and named-joint wiring is selectively ported from
the local H2S2R Pour17 scene.  No H2S2R reset, controller, trajectory, reward,
clock, synergy, or evaluator behavior is imported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .geometry import (
    ArmIK, UrdfKinematics, matrix_to_quat_wxyz, ndarray_sha256,
    quat_to_matrix_wxyz, resample_linear, sha256, slerp_wxyz,
)


CONTROLLED_LAYOUT = (
    ("right_arm", slice(0, 7)),
    ("left_arm", slice(7, 14)),
    ("right_fingers", slice(14, 36)),
    ("left_fingers", slice(36, 58)),
)


def pose_interpolation(start: np.ndarray, stop: np.ndarray, count: int,
                       *, include_stop: bool) -> np.ndarray:
    """Interpolate xyz+wxyz poses without inventing time samples."""
    start = np.asarray(start, dtype=np.float64)
    stop = np.asarray(stop, dtype=np.float64)
    fractions = ((np.arange(1, count + 1) / count) if include_stop
                 else (np.arange(count) / count))
    result = np.empty((count, 7), dtype=np.float64)
    result[:, :3] = start[:3] + fractions[:, None] * (stop[:3] - start[:3])
    result[:, 3:] = np.asarray([slerp_wxyz(start[3:], stop[3:], float(value))
                                for value in fractions])
    return result


def _transform_points(pose_wxyz: np.ndarray, local_points: np.ndarray) -> np.ndarray:
    rotation = quat_to_matrix_wxyz(pose_wxyz[3:])
    return (rotation @ np.asarray(local_points).T).T + pose_wxyz[:3]


def build_final_reference(*, preik_path: Path, probe_path: Path, urdf_path: Path,
                          output_path: Path, manifest_path: Path,
                          source_manifest_path: Path, config_path: Path) -> dict:
    """Complete both arm trajectories from a measured canonical runtime anchor."""
    preik_path = preik_path.resolve()
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe["status"] != "PASS_RUNTIME_PROBE":
        raise RuntimeError("runtime probe did not pass")
    if probe["inputs"]["robot_urdf_sha256"] != sha256(urdf_path):
        raise RuntimeError("runtime probe and IK URDF hash differ")
    with np.load(preik_path, allow_pickle=False) as data:
        preik = {name: np.asarray(data[name]) for name in data.files}

    names = tuple(str(value) for value in preik["joint_names"])
    if names != tuple(probe["joint_names_in_order"]):
        raise RuntimeError("runtime controlled-joint order differs from reference")
    lower = np.asarray(preik["joint_lower_rad"], dtype=np.float64)
    upper = np.asarray(preik["joint_upper_rad"], dtype=np.float64)
    limits = np.column_stack((lower, upper))
    canonical_q = np.asarray(preik["canonical_home_q_rad"], dtype=np.float64)
    core_count = len(preik["core_control_time_s"])
    prefix_count = 40
    suffix_count = 40

    core_wrist = {
        side: np.asarray(preik[f"{side}_wrist_pose_wxyz_core"], dtype=np.float64)
        for side in ("right", "left")
    }
    home_wrist = {
        side: np.asarray(probe["canonical_home"][f"{side}_wrist_pose_env_wxyz"],
                         dtype=np.float64)
        for side in ("right", "left")
    }
    wrists = {}
    for side in ("right", "left"):
        wrists[side] = np.concatenate((
            pose_interpolation(home_wrist[side], core_wrist[side][0], prefix_count,
                               include_stop=False),
            core_wrist[side],
            pose_interpolation(core_wrist[side][-1], home_wrist[side], suffix_count,
                               include_stop=True),
        ))

    urdf = UrdfKinematics(urdf_path)
    anchor_pose = np.asarray(probe["canonical_home"]["arm_center_pose_env_wxyz"],
                             dtype=np.float64)
    measured_q = np.asarray(probe["canonical_home"]["joint_q_rad"], dtype=np.float64)
    arms = {}
    ik_rows = {}
    for side, arm_slice, seed in (("right", slice(0, 7), 1701),
                                  ("left", slice(7, 14), 1702)):
        ik = ArmIK(
            urdf, side, "arm_center", anchor_pose,
            names[arm_slice], canonical_q[arm_slice], limits[arm_slice],
        )
        home_fk_position, home_fk_rotation = ik.fk(measured_q[arm_slice])
        measured_home = home_wrist[side]
        home_position_error = float(np.linalg.norm(home_fk_position - measured_home[:3]))
        home_rotation_error = float(np.linalg.norm(
            quat_to_matrix_wxyz(measured_home[3:]) - home_fk_rotation
        ))
        if home_position_error > 1e-4 or home_rotation_error > 1e-4:
            raise RuntimeError(
                f"{side} URDF/runtime home wrist mismatch: "
                f"position={home_position_error}, rotation_matrix={home_rotation_error}"
            )
        # Approved WP1 path: common home for the first frame, followed by the
        # preceding frame's iterate.  No random multi-start search is added.
        arms[side], ik_rows[side] = ik.solve_trajectory(
            wrists[side], seed=seed, restart_count=0, iterations=250,
            position_tolerance_m=0.005, rotation_tolerance_rad=0.05,
            rotation_weight=0.35, damping=0.05, step_clip_rad=0.25,
        )

    finger_core = {
        side: np.asarray(preik[f"{side}_finger_q_rad_core"], dtype=np.float64)
        for side in ("right", "left")
    }
    fingers = {}
    for side, finger_slice in (("right", slice(14, 36)),
                               ("left", slice(36, 58))):
        home = canonical_q[finger_slice]
        core = finger_core[side]
        fractions_prefix = np.arange(prefix_count, dtype=np.float64) / prefix_count
        fractions_suffix = np.arange(1, suffix_count + 1, dtype=np.float64) / suffix_count
        fingers[side] = np.concatenate((
            home + fractions_prefix[:, None] * (core[0] - home),
            core,
            core[-1] + fractions_suffix[:, None] * (home - core[-1]),
        ))

    hand_targets = {}
    source_time = np.asarray(preik["derived_source_time_s"], dtype=np.float64)
    core_time = np.asarray(preik["core_control_time_s"], dtype=np.float64)
    for side in ("right", "left"):
        local_source = np.asarray(preik[f"{side}_target_landmarks_S_m"], dtype=np.float64)
        local_core = resample_linear(local_source, source_time, core_time)
        world_core = np.asarray([
            _transform_points(pose, points)
            for pose, points in zip(core_wrist[side], local_core)
        ])
        home_points = np.asarray(
            probe["canonical_home"][f"{side}_landmarks_env_m"], dtype=np.float64
        )
        fractions_prefix = np.arange(prefix_count, dtype=np.float64) / prefix_count
        fractions_suffix = np.arange(1, suffix_count + 1, dtype=np.float64) / suffix_count
        hand_targets[side] = np.concatenate((
            home_points + fractions_prefix[:, None, None] * (world_core[0] - home_points),
            world_core,
            world_core[-1] + fractions_suffix[:, None, None] * (home_points - world_core[-1]),
        ))

    object_core = np.asarray(preik["object_pose_wxyz_core"], dtype=np.float64)
    objects = np.concatenate((
        np.repeat(object_core[:1], prefix_count, axis=0), object_core,
        np.repeat(object_core[-1:], suffix_count, axis=0),
    ))
    joint_q = np.concatenate((arms["right"], arms["left"],
                              fingers["right"], fingers["left"]), axis=1)
    control_time = np.arange(len(joint_q), dtype=np.float64) / 20.0
    source_index_core = np.asarray(preik["source_index_core"], dtype=np.int64)
    source_index = np.r_[np.repeat(0, prefix_count), source_index_core,
                         np.repeat(141, suffix_count)].astype(np.int64)
    stage = np.asarray(
        ["approach"] * prefix_count + ["video_core"] * core_count
        + ["return"] * suffix_count
    )
    source_valid = np.asarray(preik["source_valid"], dtype=bool)
    validity = source_valid[source_index]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray("pour17_v1_reference_v1"),
        task_id=np.asarray("pour17"),
        source_sha256=preik["source_sha256"],
        contract_sha256=np.asarray(sha256(config_path)),
        source_frame=np.asarray(preik["source_frame"], dtype=np.int64),
        derived_source_time_s=np.asarray(preik["derived_source_time_s"], dtype=np.float64),
        control_time_s=control_time,
        right_finger_q_rad=fingers["right"],
        left_finger_q_rad=fingers["left"],
        right_wrist_pose_wxyz=wrists["right"],
        left_wrist_pose_wxyz=wrists["left"],
        joint_q_rad=joint_q,
        object_pose_wxyz=objects,
        source_valid=validity,
        source_valid_names=preik["source_valid_names"],
        source_index=source_index,
        joint_names=preik["joint_names"],
        joint_lower_rad=lower,
        joint_upper_rad=upper,
        stage=stage,
        right_hand_target_m=hand_targets["right"],
        left_hand_target_m=hand_targets["left"],
        right_ik_failure_mask=np.asarray([not row["ok"] for row in ik_rows["right"]]),
        left_ik_failure_mask=np.asarray([not row["ok"] for row in ik_rows["left"]]),
    )

    ik_summary = {}
    for side in ("right", "left"):
        position = np.asarray([row["position_error_m"] for row in ik_rows[side]])
        rotation = np.asarray([row["rotation_error_rad"] for row in ik_rows[side]])
        failures = np.asarray([not row["ok"] for row in ik_rows[side]])
        ik_summary[side] = {
            "failure_count": int(failures.sum()),
            "failure_indices": np.flatnonzero(failures).tolist(),
            "position_error_m": {
                "median": float(np.median(position)), "p95": float(np.percentile(position, 95)),
                "max": float(position.max()),
            },
            "rotation_error_rad": {
                "median": float(np.median(rotation)), "p95": float(np.percentile(rotation, 95)),
                "max": float(rotation.max()),
            },
            "rows": ik_rows[side],
        }
    result = {
        "status": "PASS_REFERENCE_WITH_RECORDED_IK_FAILURES" if any(
            value["failure_count"] for value in ik_summary.values()
        ) else "PASS_REFERENCE_ALL_IK_WITHIN_TOLERANCE",
        "development_only": True,
        "world_alignment": "DEVELOPMENT_CANDIDATE_NOT_FROZEN",
        "inputs": {
            "preik_reference": {"path": str(preik_path), "sha256": sha256(preik_path)},
            "runtime_probe": {"path": str(probe_path.resolve()), "sha256": sha256(probe_path)},
            "robot_urdf": {"path": str(urdf_path.resolve()), "sha256": sha256(urdf_path)},
            "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
        },
        "sampling": {
            "control_hz": 20, "T": len(joint_q),
            "approach": {"samples": prefix_count, "duration_s": 2.0,
                         "rule": "home inclusive to first video sample exclusive; linear xyz/q and shortest SLERP"},
            "video_core": {"samples": core_count, "duration_s": 9.4,
                           "rule": "frame/15 resampled at 20Hz; linear xyz/q and shortest SLERP"},
            "return": {"samples": suffix_count, "duration_s": 2.0,
                       "rule": "last video sample exclusive to home inclusive; objects hold final"},
        },
        "ik": ik_summary,
        "output": {"path": str(output_path.resolve()), "sha256": sha256(output_path),
                   "joint_q_content_sha256": ndarray_sha256(joint_q)},
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    source_manifest_path.write_text(json.dumps({
        "schema": "pour17_v1_source_manifest_v1",
        "development_only": True,
        "files": result["inputs"],
        "output": result["output"],
        "selective_ports": [{
            "source": "RL-Correction-H2S2R/tasks/h2s2r_pour17/{cfg.py,env.py}",
            "used": ["runtime USD/object asset paths", "table and object physics", "contact sensor prim names", "runtime name mapping"],
            "excluded": ["phase1_g2 reset", "object clamp", "synergy", "FABRICS", "trajectory/clock", "reward", "PPO", "evaluator presets"],
        }],
    }, indent=2) + "\n", encoding="utf-8")
    return result


def isaac_scene_config(bundle_root: str | Path, *, num_envs: int, seed: int):
    """Create a fresh DirectRLEnv scene configuration from the registered bundle."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import configclass
    from isaaclab.envs import DirectRLEnvCfg

    root = Path(bundle_root).resolve()
    required = (
        root / "world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd",
        root / "world/objects/object_1.usd", root / "world/cache/object_0.usd",
        root / "world/canonical_reset_v1.json", root / "world/world_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"registered bundle incomplete: {missing}")

    @configclass
    class Pour17Cfg(DirectRLEnvCfg):
        decimation = 12
        episode_length_s = 45.15
        action_space = 58
        observation_space = 286
        state_space = 0
        is_finite_horizon = False
        sim = sim_utils.SimulationCfg(
            dt=1.0 / 240.0, render_interval=2, gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(
                solver_type=1, max_position_iteration_count=32,
                max_velocity_iteration_count=0, bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.04,
                friction_correlation_distance=0.025,
                gpu_max_rigid_contact_count=2**23,
                enable_ccd=False, enable_stabilization=False,
            ),
        )
        scene = InteractiveSceneCfg(num_envs=int(num_envs), env_spacing=2.0,
                                    replicate_physics=False)

    cfg = Pour17Cfg()
    cfg.seed = int(seed)
    kp = [140312.7 * 0.2, 124609.6 * 0.2, 108032.0 * 0.2,
          60512.1 * 0.2, 37899.2 * 0.2, 12066.2 * 0.2, 6497.3 * 0.2]
    kd = [5729.6 * 0.02] * 7
    cfg.robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(required[0]), activate_contact_sensors=True,
        ),
        actuators={
            "arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["[RL]_arm_j[12]"],
                stiffness={f"[RL]_arm_j{i}": kp[i - 1] for i in (1, 2)},
                damping={f"[RL]_arm_j{i}": kd[i - 1] for i in (1, 2)},
                effort_limit_sim=150.0),
            "arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["[RL]_arm_j[34]"],
                stiffness={f"[RL]_arm_j{i}": kp[i - 1] for i in (3, 4)},
                damping={f"[RL]_arm_j{i}": kd[i - 1] for i in (3, 4)},
                effort_limit_sim=80.0),
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["[RL]_arm_j[567]"],
                stiffness={f"[RL]_arm_j{i}": kp[i - 1] for i in (5, 6, 7)},
                damping={f"[RL]_arm_j{i}": kd[i - 1] for i in (5, 6, 7)},
                effort_limit_sim=25.0),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_(thumb|index|middle|ring|pinky).*"],
                stiffness=20.0, damping=2.0),
        },
    )
    cfg.bottle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(required[1]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8, solver_velocity_iteration_count=0,
                max_depenetration_velocity=1000.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.53),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.87)),
    )
    cfg.cup = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Aux",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(required[2])),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.87)),
    )
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    cfg.contact_sensors = [
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{side}_{finger}_elastomer",
            history_length=1,
            filter_prim_paths_expr=[f"/World/envs/env_.*/{'Object' if side == 'right' else 'Aux'}"],
        )
        for side in ("right", "left") for finger in fingers
    ]
    cfg.contact_sensors += [
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/{object_name}", history_length=1,
            filter_prim_paths_expr=["/World/envs/env_.*/Table"],
        )
        for object_name in ("Object", "Aux")
    ]
    return cfg
