"""Bimanual Clean3 curriculum with reference-centred residual 58D control."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import hashlib
import json
import time

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    create_rotation_matrix_from_view,
    quat_apply,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
)

from rl_rebuild.baselines.h2s2r.contract import (
    ControlledSide, fingertip_body_names, joint_names_for_side,
    simulation_joint_names,
)
from rl_rebuild.baselines.h2s2r.pour17.bimanual import (
    SharedReferenceClock, bimanual_tracking_reward, mean_fingertip_object_distance,
    object_keypoint_distance,
)
from rl_rebuild.baselines.h2s2r.pour17.observation import (
    build_bimanual_observation, build_side_observation,
)

from .cfg import CRITIC_STATE_DIM, Clean3H2S2REnvCfg


ACTION_DIM = 58
OBJECT_KEYPOINT_OFFSETS_M = (
    (0.2, 0.0, 0.0),
    (0.0, 0.2, 0.0),
    (0.0, 0.0, 0.2),
)


class Clean3H2S2REnv(DirectRLEnv):
    cfg: Clean3H2S2REnvCfg

    def _object_keypoints(self, pose_wxyz: torch.Tensor) -> torch.Tensor:
        """Return H2S2R's three 20 cm axis keypoints as a flat 9D block."""

        offsets = torch.tensor(
            OBJECT_KEYPOINT_OFFSETS_M, dtype=pose_wxyz.dtype, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1, -1)
        quaternion = pose_wxyz[:, 3:].unsqueeze(1).expand(-1, 3, -1)
        return (
            pose_wxyz[:, :3].unsqueeze(1) + quat_apply(quaternion, offsets)
        ).reshape(self.num_envs, 9)

    def _critic_state(
        self,
        policy_observation: torch.Tensor,
        plate: torch.Tensor,
        sponge: torch.Tensor,
        goal_plate: torch.Tensor,
        goal_sponge: torch.Tensor,
    ) -> torch.Tensor:
        """Build the disclosed bimanual analogue of H2S2R privileged state."""

        contact_forces = torch.cat(
            [
                sensor.data.net_forces_w.reshape(self.num_envs, -1, 3).sum(dim=1)
                for sensor in self._contact_sensors
            ],
            dim=-1,
        )
        filtered_forces = torch.cat(
            [
                sensor.data.force_matrix_w.reshape(self.num_envs, -1, 3).sum(dim=1)
                for sensor in self._contact_sensors
            ],
            dim=-1,
        )
        state = torch.cat(
            [
                policy_observation,
                self._object_keypoints(plate),
                self._object_keypoints(sponge),
                self._object_keypoints(goal_plate),
                self._object_keypoints(goal_sponge),
                self.object.data.root_lin_vel_w,
                self.object.data.root_ang_vel_w,
                self.aux.data.root_lin_vel_w,
                self.aux.data.root_ang_vel_w,
                self.episode_length_buf.to(torch.float32).unsqueeze(-1),
                self.hand.data.applied_torque[:, self.sim_joint_ids],
                contact_forces,
                filtered_forces,
            ],
            dim=-1,
        ).nan_to_num(0.0)
        if state.shape != (self.num_envs, CRITIC_STATE_DIM):
            raise RuntimeError(
                f"Clean3 asymmetric critic state must be {CRITIC_STATE_DIM}D, got {tuple(state.shape)}"
            )
        return state

    def _prepare_rigid_asset(self, path: str, mass: float, material_path: str) -> None:
        import omni.usd
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(path)
        if not root.IsValid():
            raise RuntimeError(f"missing rigid object root: {path}")
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        rb.CreateSleepThresholdAttr(0.005)
        rb.CreateStabilizationThresholdAttr(0.0025)
        meshes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError(f"object asset has no mesh: {path}")
        for prim in meshes:
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(0.002)
            collision.CreateRestOffsetAttr(0.0)
            # Bind to the actual collider rather than the rigid-body Xform.
            # Binding the root produced one warning per cloned /Aux instance.
            binding_started = time.monotonic()
            sim_utils.bind_physics_material(
                str(prim.GetPath()), material_path, stronger_than_descendants=True
            )
            self._object_material_binding_seconds += time.monotonic() - binding_started

    def _spawn_prepared_rigid_object(
        self, cfg, path: str, mass: float, material_path: str
    ) -> RigidObject:
        """Apply missing rigid schemas before Isaac Lab activates contact sensors."""

        from isaaclab.sim.schemas.schemas import activate_contact_sensors

        spawn_cfg = cfg.spawn.copy()
        spawn_cfg.activate_contact_sensors = False
        spawn_cfg.func(
            cfg.prim_path,
            spawn_cfg,
            translation=cfg.init_state.pos,
            orientation=cfg.init_state.rot,
        )
        self._prepare_rigid_asset(path, mass, material_path)
        activate_contact_sensors(path)
        runtime_cfg = cfg.copy()
        runtime_cfg.spawn = None
        return RigidObject(runtime_cfg)

    def _setup_scene(self) -> None:
        setup_started = time.monotonic()
        self._object_material_binding_seconds = 0.0
        object_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.object_friction, dynamic_friction=self.cfg.object_friction,
            restitution=0.0, friction_combine_mode="multiply", restitution_combine_mode="multiply",
        )
        object_material.func("/World/Materials/Clean3Object", object_material)
        body_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.body_friction, dynamic_friction=self.cfg.body_friction,
            restitution=0.0, friction_combine_mode="multiply", restitution_combine_mode="multiply",
        )
        body_material.func("/World/Materials/Clean3Body", body_material)

        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = self._spawn_prepared_rigid_object(
            self.cfg.object_cfg,
            "/World/envs/env_0/Object",
            self.cfg.plate_mass_kg,
            "/World/Materials/Clean3Object",
        )  # object_0 plate, left hand
        self.aux = self._spawn_prepared_rigid_object(
            self.cfg.aux_object_cfg,
            "/World/envs/env_0/Aux",
            self.cfg.sponge_mass_kg,
            "/World/Materials/Clean3Object",
        )  # object_1 sponge, right hand
        self._record_cameras = {}
        if self.cfg.record_cameras:
            camera_views = {
                "front": ((0.35, -0.45, 1.28), (-0.15, 0.04, 1.02)),
                "side": ((-0.68, 0.38, 1.25), (-0.15, 0.04, 1.02)),
                "top": ((-0.15, -0.01, 1.72), (-0.15, 0.04, 0.98)),
            }
            for name, (eye, target) in camera_views.items():
                eye_tensor = torch.tensor([eye], dtype=torch.float32)
                target_tensor = torch.tensor([target], dtype=torch.float32)
                rotation = quat_from_matrix(
                    create_rotation_matrix_from_view(
                        eye_tensor, target_tensor, up_axis="Z"
                    )
                )[0]
                camera_cfg = TiledCameraCfg(
                    prim_path=f"/World/envs/env_.*/Camera{name.title()}",
                    offset=TiledCameraCfg.OffsetCfg(
                        pos=eye,
                        rot=tuple(float(value) for value in rotation),
                        convention="opengl",
                    ),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=1.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)),
                    width=320, height=240,
                )
                self._record_cameras[name] = TiledCamera(camera_cfg)

        sx, sy, sz = self.cfg.table_size
        table = sim_utils.CuboidCfg(
            size=(sx, sy, sz), collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=self.cfg.table_friction, dynamic_friction=self.cfg.table_friction,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        )
        table.func("/World/envs/env_.*/Table", table, translation=(0.0, 0.0, self.cfg.table_top_z - sz / 2.0))
        # Vendored grid-world asset rather than GroundPlaneCfg's default, which
        # resolves to Omniverse/S3 and fails on an offline training host.  Same
        # plane prim and default physics material as before.
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(usd_path=self.cfg.ground_plane_usd),
        )

        # Author material relationships on the source environment before
        # cloning. Clones inherit the same bindings without an O(num_envs)
        # post-clone traversal of every robot and object mesh.
        material_binding_started = time.monotonic()
        for path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Robot"):
            sim_utils.bind_physics_material(path, "/World/Materials/Clean3Body", stronger_than_descendants=False)
        for side in (ControlledSide.RIGHT, ControlledSide.LEFT):
            for name in fingertip_body_names(side):
                for path in sim_utils.find_matching_prim_paths(f"/World/envs/env_.*/Robot/{name}"):
                    sim_utils.bind_physics_material(path, "/World/Materials/Clean3Object", stronger_than_descendants=True)
        material_binding_seconds = (
            self._object_material_binding_seconds
            + time.monotonic() - material_binding_started
        )

        clone_started = time.monotonic()
        self.scene.clone_environments(copy_from_source=False)
        clone_seconds = time.monotonic() - clone_started
        self.scene.filter_collisions()
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["aux"] = self.aux
        self._contact_sensors = []
        for index, sensor_cfg in enumerate(self.cfg.contact_sensors):
            sensor = ContactSensor(sensor_cfg)
            self._contact_sensors.append(sensor)
            self.scene.sensors[f"clean3_contact_{index}"] = sensor
        self._penetration_sensors = []
        for index, sensor_cfg in enumerate(self.cfg.penetration_sensor_cfgs):
            sensor = ContactSensor(sensor_cfg)
            self._penetration_sensors.append(sensor)
            self.scene.sensors[f"clean3_penetration_{index}"] = sensor
        for name, camera in self._record_cameras.items():
            self.scene.sensors[f"clean3_camera_{name}"] = camera
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)
        self.setup_timings = {
            "material_binding_seconds": material_binding_seconds,
            "clone_seconds": clone_seconds,
            "setup_scene_seconds": time.monotonic() - setup_started,
        }

    def __init__(self, cfg: Clean3H2S2REnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        expected_sha = {
            cfg.runtime_inputs_npz: "3da2b16f27d74be4898a00fc807e6993dc1915eadbaff2f9aa8539fd1a6a84ff",
        }
        for path, expected in expected_sha.items():
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"frozen Clean3 input identity changed: {path} {actual}")
        with np.load(cfg.runtime_inputs_npz, allow_pickle=False) as data:
            if data["reset_q58"].shape != (58,) or data["plate_reference_pose_wxyz"].shape != (300, 7):
                raise RuntimeError("Clean3 runtime input shape contradiction")
            self.reset_q58 = torch.as_tensor(data["reset_q58"], dtype=torch.float32, device=self.device)
            self.reset_qd58 = torch.zeros_like(self.reset_q58)
            self.reset_object_pose = {
                0: torch.as_tensor(data["reset_object_0_pose_wxyz"], dtype=torch.float32, device=self.device),
                1: torch.as_tensor(data["reset_object_1_pose_wxyz"], dtype=torch.float32, device=self.device),
            }
            self.reference_object_pose = {
                0: torch.as_tensor(data["plate_reference_pose_wxyz"], dtype=torch.float32, device=self.device),
                1: torch.as_tensor(data["sponge_reference_pose_wxyz"], dtype=torch.float32, device=self.device),
            }
            self.footprint_xy = torch.as_tensor(data["footprint_xy"], dtype=torch.float32, device=self.device)
            self.plate_profile_r = torch.as_tensor(data["plate_profile_r"], dtype=torch.float32, device=self.device)
            self.plate_profile_z = torch.as_tensor(data["plate_profile_z"], dtype=torch.float32, device=self.device)
            self.sponge_face_offset = float(data["sponge_face_offset"])

        names = list(self.hand.joint_names)
        self.sim_joint_ids = torch.tensor([names.index(name) for name in simulation_joint_names()], dtype=torch.long, device=self.device)
        self.side_joint_ids = {
            side: torch.tensor([names.index(name) for name in joint_names_for_side(side)], dtype=torch.long, device=self.device)
            for side in (ControlledSide.RIGHT, ControlledSide.LEFT)
        }
        contract_names = simulation_joint_names()
        self.side_contract_ids = {
            side: torch.tensor(
                [contract_names.index(name) for name in joint_names_for_side(side)],
                dtype=torch.long, device=self.device,
            )
            for side in (ControlledSide.RIGHT, ControlledSide.LEFT)
        }
        bodies = list(self.hand.body_names)
        self.tip_body_ids = {
            side: torch.tensor([bodies.index(name) for name in fingertip_body_names(side)], dtype=torch.long, device=self.device)
            for side in (ControlledSide.RIGHT, ControlledSide.LEFT)
        }
        self.wrist_body_ids = {
            ControlledSide.RIGHT: bodies.index("right_hand_C_MC"),
            ControlledSide.LEFT: bodies.index("left_hand_C_MC"),
        }
        with np.load(cfg.residual_controller_inputs_npz, allow_pickle=False) as data:
            required_shapes = {
                "arm_reference_q14": (424, 14),
                "plate_reference_pose_wxyz": (424, 7),
                "sponge_reference_pose_wxyz": (424, 7),
                "right_finger_grasp_q22": (22,),
                "left_finger_grasp_q22": (22,),
                "right_finger_squeeze_q22": (22,),
                "left_finger_squeeze_q22": (22,),
            }
            bad = {
                key: tuple(data[key].shape) if key in data else None
                for key, shape in required_shapes.items()
                if key not in data or tuple(data[key].shape) != shape
            }
            if bad:
                raise RuntimeError(f"invalid Clean3 residual-controller inputs: {bad}")
            self.arm_reference_q14 = torch.as_tensor(
                data["arm_reference_q14"], dtype=torch.float32, device=self.device
            )
            self.reference_object_pose = {
                0: torch.as_tensor(
                    data["plate_reference_pose_wxyz"], dtype=torch.float32, device=self.device
                ),
                1: torch.as_tensor(
                    data["sponge_reference_pose_wxyz"], dtype=torch.float32, device=self.device
                ),
            }
            self.reset_object_pose = {
                object_id: reference[0].clone()
                for object_id, reference in self.reference_object_pose.items()
            }
            self.finger_grasp_q44 = torch.as_tensor(
                np.concatenate([
                    data["right_finger_grasp_q22"], data["left_finger_grasp_q22"]
                ]), dtype=torch.float32, device=self.device,
            )
            self.finger_squeeze_q44 = torch.as_tensor(
                np.concatenate([
                    data["right_finger_squeeze_q22"], data["left_finger_squeeze_q22"]
                ]), dtype=torch.float32, device=self.device,
            )

        self.clock = SharedReferenceClock.create(
            self.num_envs, 423, str(self.device),
            control_dt=cfg.sim.dt * cfg.decimation,
            reference_dt=1.0 / 20.0,
        )

        # Same 58D output order as before, but the action now means a bounded
        # cumulative residual around the teammate arm/finger feed-forward.
        self.policy_center_q58 = torch.cat(
            [self.arm_reference_q14[0], self.finger_grasp_q44]
        )
        limits = self.hand.data.joint_pos_limits[0, self.sim_joint_ids]
        self.action_lower_q58 = limits[:, 0].clone()
        self.action_upper_q58 = limits[:, 1].clone()
        if not bool(torch.isfinite(limits).all()):
            raise RuntimeError("Clean3 residual-58D requires finite physical joint limits")
        center_outside = (self.policy_center_q58 < self.action_lower_q58) | (
            self.policy_center_q58 > self.action_upper_q58
        )
        if bool(center_outside.any()):
            indices = torch.nonzero(center_outside, as_tuple=False).squeeze(1)
            details = [
                {
                    "contract_index": int(index),
                    "joint": simulation_joint_names()[int(index)],
                    "center": float(self.policy_center_q58[index]),
                    "lower": float(self.action_lower_q58[index]),
                    "upper": float(self.action_upper_q58[index]),
                }
                for index in indices
            ]
            raise RuntimeError(
                f"Clean3 residual feed-forward is outside physical joint limits: {details}"
            )
        self._target_q = self.hand.data.joint_pos.clone()
        self._actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._cumulative_residual = torch.zeros_like(self._actions)
        self._residual_step = torch.tensor(
            [cfg.arm_step_rad] * 14 + [cfg.finger_step_rad] * 44,
            dtype=torch.float32, device=self.device,
        )
        self._residual_limit = torch.tensor(
            [cfg.arm_dev_rad] * 14 + [cfg.finger_dev_rad] * 44,
            dtype=torch.float32, device=self.device,
        )
        # Stage-1 allows all 58 residual coordinates, exactly as the working
        # CleanHoldEnv.  These four residuals are hard-clamped only after the
        # grasp succeeds and Stage-2 takes over (or in curriculum-off replay).
        self.stage2_clamped_joint_names = (
            "right_pinky_CMC", "right_pinky_MCP_AA",
            "left_middle_MCP_AA", "left_ring_MCP_AA",
        )
        self._stage2_clamped_contract_ids = torch.tensor(
            [contract_names.index(name) for name in self.stage2_clamped_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self._previous_target_q58 = self.policy_center_q58.unsqueeze(0).repeat(
            self.num_envs, 1
        )
        self._previous_object_pose = {0: self._object_poses()[0].clone(), 1: self._object_poses()[1].clone()}
        self._previous_previous_object_pose = {0: self._previous_object_pose[0].clone(), 1: self._previous_object_pose[1].clone()}
        self.runtime_counts = {"controls": 0, "physics_substeps": 0, "joint_target_writes": 0}
        self.substep_callback = None
        self._evaluation_joint_offsets = torch.zeros(self.num_envs, 58, device=self.device)
        self._audit_stream = None
        self._audit_policy_step = 0
        self._last_failure_audit = None
        # Stage-1 grasp-startup curriculum.  ``release_row_cur`` is the curriculum
        # level owned by the training entry; ``release_row`` is the per-episode
        # jittered value actually used.  All of this must exist before the reset
        # below, which samples the first release rows.
        # cfg.release_row_start lets a resumed run begin at the level it annealed
        # to.  It must be honoured here, before the reset below samples the first
        # release rows, or the first batch of episodes would run at the longest pin.
        self.release_row_cur = int(cfg.release_row_start or cfg.release_max_rows)
        self.release_row = torch.full(
            (self.num_envs,), self.release_row_cur, dtype=torch.long, device=self.device
        )
        self._pinned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pinned_ids = torch.empty(0, dtype=torch.long, device=self.device)
        # Clock freeze covers the pin and the post-release grasp-hold window.
        self._clock_frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Row on which each env actually released, for the settle/hold windows.
        self._release_row_actual = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._hold_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cert_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_certified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._new_certified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._new_grasp_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._new_grasp_drop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._startup_reward_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_within = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._contact_plate_score = torch.zeros(self.num_envs, device=self.device)
        self._contact_sponge_score = torch.zeros(self.num_envs, device=self.device)
        self._grasp_position_error = {
            object_id: torch.zeros(self.num_envs, device=self.device) for object_id in (0, 1)
        }
        self._grasp_rotation_error = {
            object_id: torch.zeros(self.num_envs, device=self.device) for object_id in (0, 1)
        }
        # Stage-1 always scores against the nominal/reset world pose, matching
        # the working teammate implementation.  _release_latched records only
        # whether release occurred; it must never redefine this datum.
        self._release_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_goal = {
            object_id: self.reset_object_pose[object_id].unsqueeze(0).repeat(self.num_envs, 1)
            for object_id in (0, 1)
        }
        self._last_tracking = None
        self._rate_sums = {}
        self._rate_episodes = 0
        self.arm_sag_q14 = torch.zeros(14, dtype=torch.float32, device=self.device)
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.arm_sag_q14.copy_(self._calibrate_arm_sag())
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        # Nominal adjacent-finger ordering for the teammate crossing penalty.
        self._cross_sign = {}
        for side in (ControlledSide.RIGHT, ControlledSide.LEFT):
            gap, _ = self._finger_gaps(side, signed=False)
            sign = torch.sign(gap[0]).detach()
            sign[sign == 0.0] = 1.0
            self._cross_sign[side] = sign
        self._validate_runtime_contract()

    def _calibrate_arm_sag(self) -> torch.Tensor:
        """One-time teammate PD steady-state compensation at reference row zero."""

        env_ids = torch.arange(self.num_envs, device=self.device)
        command = self.hand.data.default_joint_pos.clone()
        command[:, self.sim_joint_ids] = self.policy_center_q58
        zeros = torch.zeros_like(command)
        self.hand.write_joint_state_to_sim(command, zeros)

        def advance(steps: int) -> None:
            for _ in range(steps):
                self.hand.set_joint_position_target(command)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.cfg.sim.dt)

        for _ in range(3):
            self._write_object_reset_pose(env_ids)
            advance(8)
        reference = self.arm_reference_q14[0].expand(self.num_envs, -1)
        for _ in range(4):
            error = reference - self.hand.data.joint_pos[:, self.sim_joint_ids[:14]]
            command[:, self.sim_joint_ids[:14]] += error.clamp(-0.02, 0.02)
            self._write_object_reset_pose(env_ids)
            advance(8)
        return (command[:, self.sim_joint_ids[:14]] - reference).mean(dim=0)

    def _validate_runtime_contract(self) -> None:
        if len(self._contact_sensors) != 10:
            raise RuntimeError("Clean3 requires ten side-filtered fingertip contact sensors")
        for sensor in self._contact_sensors:
            for label, values in (
                ("net", sensor.data.net_forces_w),
                ("filtered", sensor.data.force_matrix_w),
            ):
                if values is None or values.shape[0] != self.num_envs or not bool(torch.isfinite(values).all()):
                    raise RuntimeError(f"invalid Clean3 {label} contact tensor")
        if self.cfg.warmup_clamp_steps != 0:
            raise RuntimeError("Clean3 C3-P1R2 objects must be dynamic from the first physics step")
        if self._actions.shape != (self.num_envs, ACTION_DIM):
            raise RuntimeError("Clean3 residual policy action contract must be 58")
        if not bool((self._residual_step > 0.0).all()) or not bool((self._residual_limit > 0.0).all()):
            raise RuntimeError("Clean3 Stage-1 must expose all 58 residual coordinates")
        if abs(self.cfg.sim.dt - 1.0 / 240.0) > 1e-12 or self.cfg.decimation != 12:
            raise RuntimeError("Clean3 must reproduce 240 Hz physics and 20 Hz residual actions")
        if (
            self.cfg.close_rows,
            self.cfg.release_max_rows,
            self.cfg.release_min_rows,
            self.cfg.release_step_rows,
            self.cfg.hold_rows,
        ) != (10, 50, 10, 5, 60):
            raise RuntimeError("Clean3 residual curriculum must reproduce 10/50/10/5/60 rows")
        # The proven Clean3 squeeze prior intentionally commands three finger
        # joints slightly beyond their articulation limits.  Preserve that raw
        # feed-forward (and its residual dead-zone), matching the teammate
        # controller; _pre_physics_step saturates every command at the current
        # articulation's physical limits before it is written to the simulator.
        feedforward_values = torch.cat([
            self.arm_reference_q14.flatten(),
            self.finger_grasp_q44,
            self.finger_squeeze_q44,
        ])
        if not bool(torch.isfinite(feedforward_values).all()):
            raise RuntimeError("Clean3 residual feed-forward contains non-finite values")
        if self.cfg.grasp_curriculum:
            if not 0 < self.cfg.release_min_rows <= self.cfg.release_max_rows:
                raise RuntimeError(
                    "Clean3 grasp curriculum requires 0 < release_min_rows <= release_max_rows: "
                    f"{self.cfg.release_min_rows} / {self.cfg.release_max_rows}"
                )
            if self.cfg.release_step_rows < 1 or self.cfg.release_jitter_rows < 0:
                raise RuntimeError("Clean3 grasp curriculum needs a positive step and non-negative jitter")
            if (
                self.cfg.hold_rows < 1
                or self.cfg.drop_position_m <= 0.0
                or self.cfg.drop_rotation_deg <= 0.0
            ):
                raise RuntimeError("Clean3 grasp curriculum needs a positive hold window and drop bounds")
            if self.cfg.release_settle_rows < 0:
                raise RuntimeError("Clean3 grasp curriculum needs a non-negative settle window")
            # Every episode must still contain the settle and hold windows after the
            # longest possible pin, otherwise success is unreachable at the start of
            # the curriculum.  The reference is frozen for all of it, so the task
            # portion of the episode begins only afterwards.
            # The ceiling must clear the worst-case startup plus the full task
            # budget, or the longest-pin level would be clipped by the hard timeout
            # and would not get the same task time as the annealed levels.
            startup_rows = (
                self.cfg.release_max_rows + self.cfg.release_settle_rows + self.cfg.hold_rows
            )
            if self.cfg.ours_stage1_contract:
                if self.cfg.task_rows != 0:
                    raise RuntimeError("Ours Stage-1 contract must not include a task tail")
            elif self.cfg.task_rows < 1:
                raise RuntimeError("Clean3 grasp curriculum needs a positive task_rows budget")
            if self.max_episode_length - 1 < startup_rows + self.cfg.task_rows:
                raise RuntimeError(
                    f"Clean3 episode ceiling {self.max_episode_length} rows cannot fit a "
                    f"{startup_rows}-row startup plus {self.cfg.task_rows} task rows"
                )
            if self.max_episode_length <= startup_rows:
                raise RuntimeError(
                    f"Clean3 episode of {self.max_episode_length} rows cannot hold a "
                    f"{self.cfg.release_max_rows}-row pin plus a {self.cfg.release_settle_rows}-row "
                    f"settle and a {self.cfg.hold_rows}-row hold window"
                )

    def _object_poses(self):
        origins = self.scene.env_origins
        plate = torch.cat([self.object.data.root_pos_w - origins, self.object.data.root_quat_w], dim=-1)
        sponge = torch.cat([self.aux.data.root_pos_w - origins, self.aux.data.root_quat_w], dim=-1)
        return plate, sponge

    def _tips(self, side):
        return self.hand.data.body_pos_w[:, self.tip_body_ids[side]] - self.scene.env_origins.unsqueeze(1)

    def _wrist(self, side):
        body = self.wrist_body_ids[side]
        return self.hand.data.body_pos_w[:, body] - self.scene.env_origins, self.hand.data.body_quat_w[:, body]

    def _goal_poses(self):
        indices = self.clock.indices
        return self.reference_object_pose[0][indices], self.reference_object_pose[1][indices]

    def _write_object_reset_pose(self, env_ids: torch.Tensor) -> None:
        """Place both objects at their nominal reset pose with zero velocity.

        Shared by the episode reset and by the Stage-1 pin, which re-applies the
        same pose on every physics substep while the grasp window is open.
        """

        count = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        still = torch.zeros(count, 6, device=self.device)
        for object_id, rigid in ((0, self.object), (1, self.aux)):
            pose = self.reset_object_pose[object_id].repeat(count, 1)
            pose[:, :3] += origins
            rigid.write_root_pose_to_sim(pose, env_ids=env_ids)
            rigid.write_root_velocity_to_sim(still, env_ids=env_ids)

    def _finger_feedforward(self) -> torch.Tensor:
        """Working Clean3 grasp->squeeze ramp, then constant squeeze."""

        alpha = (
            self.episode_length_buf.float() / float(self.cfg.close_rows)
        ).clamp(0.0, 1.0).unsqueeze(1)
        return (
            (1.0 - alpha) * self.finger_grasp_q44
            + alpha * self.finger_squeeze_q44
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1.0, 1.0))
        self._cumulative_residual.add_(self._actions * self._residual_step)
        self._cumulative_residual.copy_(torch.maximum(
            torch.minimum(self._cumulative_residual, self._residual_limit),
            -self._residual_limit,
        ))
        stage2 = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if not self.cfg.grasp_curriculum
            else self._grasp_success
        )
        if bool(stage2.any()):
            rows = torch.nonzero(stage2, as_tuple=False).squeeze(1)
            self._cumulative_residual[
                rows.unsqueeze(1), self._stage2_clamped_contract_ids.unsqueeze(0)
            ] = 0.0
        feedforward = torch.cat(
            [
                self.arm_reference_q14[self.clock.indices] + self.arm_sag_q14,
                self._finger_feedforward(),
            ], dim=1
        )
        target = (feedforward + self._cumulative_residual).clamp(
            self.action_lower_q58, self.action_upper_q58
        )
        self._previous_target_q58.copy_(self._target_q[:, self.sim_joint_ids])
        self._target_q[:, self.sim_joint_ids] = target
        self.runtime_counts["controls"] += 1
        # Stage-1: envs whose pin window is still open.  episode_length_buf has
        # not been incremented for this step yet, so row 0 is the first pinned row.
        if self.cfg.grasp_curriculum:
            self._pinned = self.episode_length_buf < self.release_row
            self._pinned_ids = torch.nonzero(self._pinned, as_tuple=False).squeeze(1)
            # _get_rewards consumes this in the same step; _get_dones refreshes it
            # afterwards to extend the freeze across the grasp-hold window.
            self._clock_frozen = ~self._grasp_success

    def _apply_action(self) -> None:
        self.runtime_counts["physics_substeps"] += 1
        self.hand.set_joint_position_target(self._target_q)
        self.runtime_counts["joint_target_writes"] += 1
        # Stage-1 pin: hold both objects at their reset pose for the whole grasp
        # window, re-applied every physics substep.  Collisions stay fully enabled,
        # so the closing fingers still build real contact impulses -- the object
        # simply cannot fall before the policy has had a chance to grip it.
        if self.cfg.grasp_curriculum and self._pinned_ids.numel():
            self._write_object_reset_pose(self._pinned_ids)
        if self.substep_callback is not None:
            self.substep_callback(self)

    def _tracking(self):
        plate, sponge = self._object_poses()
        goal_plate, goal_sponge = self._goal_poses()
        left_tips, right_tips = self._tips(ControlledSide.LEFT), self._tips(ControlledSide.RIGHT)
        return {
            "plate": plate, "sponge": sponge, "goal_plate": goal_plate, "goal_sponge": goal_sponge,
            "plate_error": object_keypoint_distance(plate, goal_plate),
            "sponge_error": object_keypoint_distance(sponge, goal_sponge),
            "left_tip_distance": mean_fingertip_object_distance(left_tips, plate[:, :3]),
            "right_tip_distance": mean_fingertip_object_distance(right_tips, sponge[:, :3]),
        }

    @staticmethod
    def _rotation_error(q: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
        """Shortest full quaternion angle in radians."""

        cosine = (q * q0).sum(dim=1).abs().clamp(0.0, 1.0)
        return 2.0 * torch.acos(cosine)

    def _finger_gaps(self, side: ControlledSide, *, signed: bool = True):
        """Adjacent non-thumb fingertip gaps in the local palm frame."""

        palm_p, palm_q = self._wrist(side)
        tips = self._tips(side)[:, 1:]
        local = quat_apply(
            quat_conjugate(palm_q)[:, None, :].expand(-1, 4, -1).reshape(-1, 4),
            (tips - palm_p[:, None, :]).reshape(-1, 3),
        ).reshape(self.num_envs, 4, 3)
        lateral = local[:, :-1, 1] - local[:, 1:, 1]
        distance = torch.linalg.vector_norm(tips[:, :-1] - tips[:, 1:], dim=2)
        if not signed:
            return lateral, distance
        return lateral * self._cross_sign[side].unsqueeze(0), distance

    def _cross_penalty(self) -> torch.Tensor:
        terms = []
        for side in (ControlledSide.RIGHT, ControlledSide.LEFT):
            lateral, distance = self._finger_gaps(side)
            terms.append(
                ((self.cfg.cross_gap_min_m - lateral) / self.cfg.cross_gap_min_m)
                .clamp(0.0, 1.0)
            )
            terms.append(
                ((self.cfg.overlap_min_m - distance) / self.cfg.overlap_min_m)
                .clamp(0.0, 1.0)
            )
        return torch.cat(terms, dim=1).mean(dim=1) * 2.0

    def _get_rewards(self):
        values = self._tracking()
        task_reward, terms = bimanual_tracking_reward(
            values["plate_error"], values["sponge_error"], values["left_tip_distance"], values["right_tip_distance"],
            previous_actions=self._previous_actions, actions=self._actions, smoothing_weight=0.0,
        )

        # Exact working Stage-1 objective.  It wholly replaces task tracking until
        # the grasp has survived the certification/hold window; the pinned-object
        # tracking shortcut therefore cannot reward an empty grasp.
        contact_reward = self.cfg.reward_contact_weight * (
            self._contact_plate_score + self._contact_sponge_score
        )
        hold_penalty = -(
            self.cfg.reward_hold_position_weight
            * 0.5 * (self._grasp_position_error[0] + self._grasp_position_error[1])
            / self.cfg.certify_position_m
            + self.cfg.reward_hold_rotation_weight
            * 0.5 * (self._grasp_rotation_error[0] + self._grasp_rotation_error[1])
            / np.radians(self.cfg.certify_rotation_deg)
        ).clamp(min=-3.0)
        hold_reward = torch.where(
            self._release_latched,
            hold_penalty + self.cfg.reward_within_weight * self._grasp_within.float(),
            torch.zeros_like(hold_penalty),
        )
        bonus = (
            self.cfg.reward_cert_bonus * self._new_certified.float()
            + self.cfg.reward_success_bonus * self._new_grasp_success.float()
            + self.cfg.reward_drop_bonus * self._new_grasp_drop.float()
        )
        action_penalty = -self.cfg.reward_action_weight * self._actions.square().mean(dim=1)
        cross_penalty = -self.cfg.reward_cross_weight * self._cross_penalty()
        startup_reward = contact_reward + hold_reward + bonus + action_penalty + cross_penalty
        reward = torch.where(self._startup_reward_active, startup_reward, task_reward)
        advanced = self.clock.step(
            values["plate_error"],
            values["sponge_error"],
            # A pinned object sits exactly on reference row 0, which would
            # otherwise satisfy the advance test and burn the opening rows of the
            # reference for free.  The freeze extends through the grasp-hold window
            # so that holding still is not mistaken for losing the object.
            frozen=self._clock_frozen,
        )
        self.extras["h2s2r/plate_tracking_reward"] = terms["left_object_tracking"].mean()
        self.extras["h2s2r/sponge_tracking_reward"] = terms["right_object_tracking"].mean()
        self.extras["h2s2r/shared_advance_rate"] = advanced.float().mean()
        self.extras["h2s2r/reference_index_mean"] = self.clock.indices.float().mean()
        reward_audit = {
            "reward_mean": float(reward.mean()),
            "reward_min": float(reward.min()),
            "reward_max": float(reward.max()),
            "plate_tracking_reward_mean": float(terms["left_object_tracking"].mean()),
            "sponge_tracking_reward_mean": float(terms["right_object_tracking"].mean()),
            "startup_reward_mean": float(startup_reward.mean()),
            "startup_active_count": int(self._startup_reward_active.sum()),
            "contact_plate_score_mean": float(self._contact_plate_score.mean()),
            "contact_sponge_score_mean": float(self._contact_sponge_score.mean()),
            "grasp_certified_count": int(self._grasp_certified.sum()),
            "reference_advanced_count": int(advanced.sum()),
            # Curriculum phase, so a per-step reward row can be read in context:
            # a low reward during the pin means something different from a low
            # reward once the object is unsupported.
            "release_row_cur": int(self.release_row_cur),
            "pinned_count": int(self._pinned.sum()),
            "clock_frozen_count": int(self._clock_frozen.sum()),
            "grasp_dropped_count": int(self._grasp_dropped.sum()),
            "grasp_success_count": int(self._grasp_success.sum()),
            "hold_run_mean": float(self._hold_run.float().mean()),
        }
        if self._audit_stream is not None and self._last_failure_audit is not None:
            self._audit_policy_step += 1
            row = {
                "policy_step": self._audit_policy_step,
                "transitions": self._audit_policy_step * self.num_envs,
                **reward_audit,
                "failure_counts": self._last_failure_audit,
            }
            self._audit_stream.write(json.dumps(row, allow_nan=False) + "\n")
        return reward

    def _get_dones(self):
        values = self._tracking()
        self._last_tracking = values
        plate, sponge = values["plate"], values["sponge"]
        # Stage-1: an env is released once its pin window has elapsed.  The row
        # counter was already incremented for this step, hence the strict ">".
        if self.cfg.grasp_curriculum:
            released = self.episode_length_buf > self.release_row
        else:
            released = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.external_eval:
            # External evaluation never terminates, but the curriculum state must
            # still advance: without this the clock would unfreeze at release
            # instead of after the hold window, so a milestone replay would not
            # reproduce the dynamics the checkpoint was trained under.
            if self.cfg.grasp_curriculum:
                self._score_grasp(released, values)
                self._clock_frozen = ~self._grasp_success
            never = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._last_failure_audit = {
                "external_eval": int(self.num_envs),
                "grasp_dropped": int(self._grasp_dropped.sum()),
                "grasp_success": int(self._grasp_success.sum()),
                "terminated_any": 0,
            }
            return never, never
        # A pinned object cannot fall and the hand is still searching for contact,
        # so no failure termination may fire before release.
        plate_below = (plate[:, 2] < self.cfg.table_top_z - 0.05) & released
        sponge_below = (sponge[:, 2] < self.cfg.table_top_z - 0.05) & released
        left_far = torch.zeros_like(plate_below)
        right_far = torch.zeros_like(plate_below)
        reference_complete = torch.zeros_like(plate_below)
        terminated = plate_below | sponge_below
        if self.cfg.early_reset_if_fingertips_far:
            left_far = (values["left_tip_distance"] >= self.cfg.fingertips_close_threshold_m) & released
            right_far = (values["right_tip_distance"] >= self.cfg.fingertips_close_threshold_m) & released
            terminated |= left_far | right_far
        if self.cfg.terminate_when_reference_ends:
            reference_complete = self.clock.is_complete
            terminated |= reference_complete
        grasp_drop = torch.zeros_like(plate_below)
        grasp_hold_failed = torch.zeros_like(plate_below)
        if self.cfg.grasp_curriculum:
            grasp_drop = self._score_grasp(released, values)
            # Curriculum-only failure termination: once the object has actually
            # left the hand there is nothing left to learn from the episode, and
            # the remaining ~400 rows would otherwise be spent on a dead state.
            # Armed only after release plus the settling grace, so pin-release
            # contact jitter cannot manufacture a false failure.  The evaluation
            # configuration (grasp_curriculum=False) keeps the original baseline
            # termination set untouched.
            if self.cfg.terminate_on_grasp_drop:
                terminated |= grasp_drop
            grasp_hold_failed = (
                self._release_latched
                & (
                    self.episode_length_buf >= (
                        self.cfg.release_max_rows + self.cfg.hold_rows
                        if self.cfg.ours_stage1_contract
                        else self._release_row_actual + self.cfg.hold_rows - 1
                    )
                )
                & ~self._grasp_success
                & ~self._grasp_dropped
            )
            terminated |= grasp_hold_failed
            self._clock_frozen = ~self._grasp_success
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.grasp_curriculum:
            # Every curriculum level gets exactly cfg.task_rows of post-startup
            # task time.  The startup phase shrinks as release_row anneals down, so
            # a fixed timeout would hand the policy up to 30 extra task rows at the
            # floor; time out per env against its own release row instead.
            # max_episode_length stays fixed (it sizes the PPO buffers) and remains
            # the hard ceiling, so this only ever ends an episode earlier.
            if self.cfg.ours_stage1_contract:
                level_timeout = (
                    self.episode_length_buf
                    >= self.cfg.release_max_rows + self.cfg.hold_rows
                )
            else:
                level_timeout = (
                    self.episode_length_buf
                    >= self.release_row + self.cfg.release_settle_rows
                    + self.cfg.hold_rows + self.cfg.task_rows
                )
            timeout |= level_timeout
        self._last_failure_audit = {
            "plate_below": int(plate_below.sum()),
            "sponge_below": int(sponge_below.sum()),
            "left_fingertips_far": int(left_far.sum()),
            "right_fingertips_far": int(right_far.sum()),
            "reference_complete": int(reference_complete.sum()),
            "grasp_drop": int(grasp_drop.sum()),
            "grasp_hold_failed": int(grasp_hold_failed.sum()),
            "timeout": int(timeout.sum()),
            "terminated_any": int(terminated.sum()),
        }
        return terminated, timeout

    def _score_grasp(self, released: torch.Tensor, values: dict) -> torch.Tensor:
        """Working contact-aware Clean3 Stage-1 certification and drop test."""

        self._startup_reward_active = ~self._grasp_success
        self._new_certified.zero_()
        self._new_grasp_success.zero_()
        self._new_grasp_drop.zero_()
        newly_released = released & ~self._release_latched
        if bool(newly_released.any()):
            rows = torch.nonzero(newly_released, as_tuple=False).squeeze(1)
            self._release_latched[rows] = True
            self._release_row_actual[rows] = self.episode_length_buf[rows]

        forces = self.contact_forces()
        # Sensor order is right/sponge first, then left/plate; each is
        # thumb,index,middle,ring,pinky.
        sponge_on = forces[:, :5] > self.cfg.pad_force_threshold_n
        plate_on = forces[:, 5:] > self.cfg.pad_force_threshold_n
        plate_ok = plate_on[:, 0] & (
            plate_on[:, 1:].sum(dim=1) >= self.cfg.plate_support_min
        )
        sponge_ok = sponge_on.sum(dim=1) >= self.cfg.sponge_pads_min
        self._contact_plate_score = (
            plate_on[:, 0].float()
            + plate_on[:, 1:].sum(dim=1).clamp(max=3).float() / 3.0
        )
        self._contact_sponge_score = sponge_on.sum(dim=1).float() / 5.0

        startup_open = ~self._grasp_success
        for object_id, key in ((0, "plate"), (1, "sponge")):
            position_error = torch.linalg.vector_norm(
                values[key][:, :3] - self._hold_goal[object_id][:, :3], dim=1
            )
            rotation_error = self._rotation_error(
                values[key][:, 3:], self._hold_goal[object_id][:, 3:]
            )
            self._grasp_position_error[object_id] = torch.where(
                startup_open, position_error, self._grasp_position_error[object_id]
            )
            self._grasp_rotation_error[object_id] = torch.where(
                startup_open, rotation_error, self._grasp_rotation_error[object_id]
            )
        tight_position = torch.stack([
            self._grasp_position_error[0] < self.cfg.certify_position_m,
            self._grasp_position_error[1] < self.cfg.certify_position_m,
        ], dim=1).all(dim=1)
        tight_rotation = torch.stack([
            self._grasp_rotation_error[0] < np.radians(self.cfg.certify_rotation_deg),
            self._grasp_rotation_error[1] < np.radians(self.cfg.certify_rotation_deg),
        ], dim=1).all(dim=1)
        self._grasp_within = tight_position & tight_rotation

        z_low = torch.stack([
            values["plate"][:, 2] < self.cfg.table_top_z + 0.03,
            values["sponge"][:, 2] < self.cfg.table_top_z + 0.03,
        ], dim=1).any(dim=1)
        startup_released = released & ~self._grasp_success
        drop_now = startup_released & (
            (self._grasp_position_error[0] > self.cfg.drop_position_m)
            | (self._grasp_position_error[1] > self.cfg.drop_position_m)
            | (self._grasp_rotation_error[0] > np.radians(self.cfg.drop_rotation_deg))
            | (self._grasp_rotation_error[1] > np.radians(self.cfg.drop_rotation_deg))
            | z_low
        )
        self._new_grasp_drop = drop_now & ~self._grasp_dropped
        self._grasp_dropped |= drop_now

        cert_now = (
            startup_released & self._grasp_within & plate_ok & sponge_ok
            & ~self._grasp_dropped
        )
        self._cert_run = torch.where(
            cert_now, self._cert_run + 1, torch.zeros_like(self._cert_run)
        )
        self._new_certified = (
            (self._cert_run >= self.cfg.certify_rows) & ~self._grasp_certified
        )
        self._grasp_certified |= self._new_certified
        next_hold_run = torch.where(
            cert_now, self._hold_run + 1, torch.zeros_like(self._hold_run)
        )
        self._hold_run = torch.where(
            self._grasp_success, self._hold_run, next_hold_run
        )
        hold_complete = (
            startup_released
            & (self.episode_length_buf >= (
                self.cfg.release_max_rows + self.cfg.hold_rows
                if self.cfg.ours_stage1_contract
                else self._release_row_actual + self.cfg.hold_rows - 1
            ))
        )
        self._new_grasp_success = (
            hold_complete & self._grasp_certified & self._grasp_within
            & ~self._grasp_dropped & ~self._grasp_success
        )
        self._grasp_success |= self._new_grasp_success
        self._clock_frozen = ~self._grasp_success
        return self._new_grasp_drop

    def set_audit_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_stream = path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        if self._audit_stream is not None:
            self._audit_stream.close()
            self._audit_stream = None
        super().close()

    def _get_ours_stage1_observations(self):
        """Checkpoint-compatible 367D actor and 22D privileged blocks.

        Stage-1 always exposes error to the fixed hold pose.  In the explicitly
        disclosed Stage-2 handoff, those same 6D slots switch to error against
        the current task goal only after grasp success.  The input shape and all
        pre-success semantics therefore remain checkpoint-compatible.
        """

        plate, sponge = self._object_poses()
        goal_plate, goal_sponge = self._goal_poses()
        joint_q = self.hand.data.joint_pos[:, self.sim_joint_ids]
        joint_qd = self.hand.data.joint_vel[:, self.sim_joint_ids]
        blocks = [
            joint_q[:, :14],
            joint_qd[:, :14] * 0.1,
            joint_q[:, 14:],
            joint_qd[:, 14:] * 0.1,
            self._cumulative_residual / self._residual_limit,
        ]
        deviations = []
        force_columns = []
        for side, object_pose, task_goal, rigid, object_id in (
            (ControlledSide.LEFT, plate, goal_plate, self.object, 0),
            (ControlledSide.RIGHT, sponge, goal_sponge, self.aux, 1),
        ):
            palm_position, palm_quaternion = self._wrist(side)
            relative_position = quat_apply(
                quat_conjugate(palm_quaternion),
                object_pose[:, :3] - palm_position,
            )
            relative_quaternion = quat_mul(
                quat_conjugate(palm_quaternion), object_pose[:, 3:]
            )
            nominal = self._hold_goal[object_id]
            if self.cfg.ours_stage2_contract:
                nominal = torch.where(
                    self._grasp_success.unsqueeze(1), task_goal, nominal
                )
            delta_quaternion = quat_mul(
                quat_conjugate(nominal[:, 3:]), object_pose[:, 3:]
            )
            deviation = torch.cat(
                [
                    object_pose[:, :3] - nominal[:, :3],
                    2.0
                    * delta_quaternion[:, 1:]
                    * torch.sign(delta_quaternion[:, :1]),
                ],
                dim=1,
            )
            blocks.extend(
                [
                    palm_position,
                    palm_quaternion,
                    object_pose[:, :3],
                    object_pose[:, 3:],
                    relative_position,
                    relative_quaternion,
                    deviation,
                    rigid.data.root_lin_vel_w,
                    rigid.data.root_ang_vel_w * 0.1,
                ]
            )
            deviations.append(deviation)

        # contact_forces() is right/sponge then left/plate; Ours Stage-1 is
        # left/plate then right/sponge.
        forces_right_left = self.contact_forces()
        forces = torch.cat(
            [forces_right_left[:, 5:], forces_right_left[:, :5]], dim=1
        )
        force_columns.append((forces / 10.0).clamp(max=3.0))
        force_block = torch.cat(force_columns, dim=1)
        row = self.episode_length_buf
        blocks.extend(
            [
                force_block,
                (force_block > self.cfg.pad_force_threshold_n / 10.0).float(),
                (row < self.release_row).float().unsqueeze(1),
                ((self.release_row - row).float() / 60.0)
                .clamp(-2.0, 2.0)
                .unsqueeze(1),
                ((row - self.release_row).float() / 60.0)
                .clamp(-2.0, 2.0)
                .unsqueeze(1),
                self._grasp_certified.float().unsqueeze(1),
                (self._cert_run.float() / self.cfg.certify_rows)
                .clamp(max=1.0)
                .unsqueeze(1),
                self._finger_feedforward(),
                self._actions,
            ]
        )
        policy = torch.cat(blocks, dim=1).float().clamp(-10.0, 10.0).nan_to_num(0.0)
        privileged = (
            torch.cat(deviations + [force_block], dim=1)
            .float()
            .clamp(-10.0, 10.0)
            .nan_to_num(0.0)
        )
        if policy.shape != (self.num_envs, 367):
            raise RuntimeError(f"Ours Stage-1 policy must be 367D, got {tuple(policy.shape)}")
        if privileged.shape != (self.num_envs, 22):
            raise RuntimeError(
                f"Ours Stage-1 privileged info must be 22D, got {tuple(privileged.shape)}"
            )
        return {"policy": policy, "priv_info": privileged}

    def _get_observations(self):
        if self.cfg.ours_stage1_contract or self.cfg.ours_stage2_contract:
            return self._get_ours_stage1_observations()
        plate, sponge = self._object_poses()
        goal_plate, goal_sponge = self._goal_poses()
        sides = {}
        for side, object_pose, goal_pose, object_id in (
            (ControlledSide.RIGHT, sponge, goal_sponge, 1),
            (ControlledSide.LEFT, plate, goal_plate, 0),
        ):
            ids = self.side_joint_ids[side]
            wrist_p, wrist_q = self._wrist(side)
            actual_q = self.hand.data.joint_pos[:, ids]
            actual_qd = self.hand.data.joint_vel[:, ids]
            sides[side] = build_side_observation(
                q=actual_q, qd=actual_qd,
                fingertip_positions=self._tips(side), palm_position=wrist_p, palm_quaternion_wxyz=wrist_q,
                object_pose_wxyz=object_pose, goal_object_pose_wxyz=goal_pose,
                previous_object_pose_wxyz=self._previous_object_pose[object_id],
                previous_previous_object_pose_wxyz=self._previous_previous_object_pose[object_id],
                # Reuse the existing 29+29 command slots to expose the residual
                # controller state without changing the frozen 342D policy shape.
                command_q=self._target_q[:, ids],
                command_qd=(
                    self._target_q[:, ids]
                    - self._previous_target_q58[:, self.side_contract_ids[side]]
                ) / (self.cfg.sim.dt * self.cfg.decimation),
            )
        obs = build_bimanual_observation(sides[ControlledSide.RIGHT], sides[ControlledSide.LEFT]).clamp(-self.cfg.clip_obs, self.cfg.clip_obs).nan_to_num(0.0)
        critic = self._critic_state(obs, plate, sponge, goal_plate, goal_sponge)
        self._previous_previous_object_pose[0].copy_(self._previous_object_pose[0])
        self._previous_previous_object_pose[1].copy_(self._previous_object_pose[1])
        self._previous_object_pose[0].copy_(plate)
        self._previous_object_pose[1].copy_(sponge)
        return {
            "policy": obs,
            "critic": critic,
            "priv_info": torch.empty(self.num_envs, 0, device=self.device),
        }

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return
        # Book the finished episodes before DirectRLEnv zeroes the row counter.
        # _last_tracking is None until the first real step, so the reset issued
        # from __init__ and from env.reset() contributes nothing.
        if self.cfg.grasp_curriculum and self._last_tracking is not None:
            self._book_grasp_rates(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)
        n = len(env_ids)
        q = self.hand.data.default_joint_pos[env_ids].clone()
        qd = torch.zeros_like(q)
        reset_q = self.policy_center_q58.unsqueeze(0) + self._evaluation_joint_offsets[env_ids]
        q[:, self.sim_joint_ids] = reset_q
        self.hand.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        target_q = q.clone()
        target_q[:, self.sim_joint_ids[:14]] += self.arm_sag_q14
        self.hand.set_joint_position_target(target_q, env_ids=env_ids)
        self._target_q[env_ids] = target_q
        self._write_object_reset_pose(env_ids)
        self.clock.reset(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._cumulative_residual[env_ids] = 0.0
        self._previous_target_q58[env_ids] = target_q[:, self.sim_joint_ids]
        for object_id in (0, 1):
            pose = self.reset_object_pose[object_id].repeat(n, 1)
            self._previous_object_pose[object_id][env_ids] = pose
            self._previous_previous_object_pose[object_id][env_ids] = pose
        if self.cfg.grasp_curriculum:
            # Release 0..jitter rows early so the policy cannot simply count beats
            # to the release; the curriculum floor is never crossed.
            lowest = max(
                int(self.cfg.release_min_rows),
                self.release_row_cur - int(self.cfg.release_jitter_rows),
            )
            self.release_row[env_ids] = torch.randint(
                lowest, self.release_row_cur + 1, (n,), device=self.device
            )
            self._hold_run[env_ids] = 0
            self._cert_run[env_ids] = 0
            self._grasp_certified[env_ids] = False
            self._grasp_dropped[env_ids] = False
            self._grasp_success[env_ids] = False
            self._new_certified[env_ids] = False
            self._new_grasp_success[env_ids] = False
            self._new_grasp_drop[env_ids] = False
            self._startup_reward_active[env_ids] = True
            self._grasp_within[env_ids] = False
            self._contact_plate_score[env_ids] = 0.0
            self._contact_sponge_score[env_ids] = 0.0
            for object_id in (0, 1):
                self._grasp_position_error[object_id][env_ids] = 0.0
                self._grasp_rotation_error[object_id][env_ids] = 0.0
            self._release_latched[env_ids] = False
            self._release_row_actual[env_ids] = 0
            self._pinned[env_ids] = True
            self._clock_frozen[env_ids] = True
            for object_id in (0, 1):
                self._hold_goal[object_id][env_ids] = self.reset_object_pose[object_id]

    def _book_grasp_rates(self, env_ids: torch.Tensor) -> None:
        """Accumulate finished-episode curriculum statistics for pop_rates().

        Sums stay on device; the single host transfer happens in pop_rates().
        """

        values = self._last_tracking
        finished = {
            "sr/grasp_success": self._grasp_success[env_ids].float(),
            "sr/grasp_certified": self._grasp_certified[env_ids].float(),
            "sr/grasp_drop": self._grasp_dropped[env_ids].float(),
            "curr/release_row": self.release_row[env_ids].float(),
            # Deviation from the latched release pose -- the same datum _score_grasp
            # uses -- not from the moving current reference row.
            "hold/plate_keypoint_m": object_keypoint_distance(
                values["plate"][env_ids], self._hold_goal[0][env_ids]
            ),
            "hold/sponge_keypoint_m": object_keypoint_distance(
                values["sponge"][env_ids], self._hold_goal[1][env_ids]
            ),
            "hold/released_before_end": self._release_latched[env_ids].float(),
            "hold/max_run_rows": self._hold_run[env_ids].float(),
            # Sanity invariant: success and drop are mutually exclusive by
            # construction, so this must stay flat at zero in the logs.
            "sr/success_and_drop": (
                self._grasp_success[env_ids] & self._grasp_dropped[env_ids]
            ).float(),
            "ep/length_rows": self.episode_length_buf[env_ids].float(),
            # Per-episode failure breakdown, so a stalled curriculum can be read
            # off the logs without re-running anything.
            "fail/never_released": (~self._release_latched[env_ids]).float(),
            "fail/dropped_after_release": self._grasp_dropped[env_ids].float(),
            "fail/held_but_short": (
                ~self._grasp_dropped[env_ids] & ~self._grasp_success[env_ids]
                & self._release_latched[env_ids]
            ).float(),
        }
        for key, value in finished.items():
            total = value.sum()
            self._rate_sums[key] = self._rate_sums[key] + total if key in self._rate_sums else total
        self._rate_episodes += len(env_ids)

    def pop_rates(self) -> dict[str, float]:
        """Mean per-episode curriculum statistics since the previous call.

        The training entry consumes ``sr/grasp_success`` to drive the release_row
        annealer.  Accumulators are cleared, so each call covers exactly one
        window; an empty dict means no episode finished in that window.
        """

        if not self._rate_episodes:
            return {}
        episodes = float(self._rate_episodes)
        rates = {key: float(total) / episodes for key, total in self._rate_sums.items()}
        rates["curr/episodes"] = episodes
        self._rate_sums = {}
        self._rate_episodes = 0
        return rates

    def contact_forces(self) -> torch.Tensor:
        # Working teammate Stage-1 uses each pad sensor's total net force, not
        # its filtered force_matrix_w.  Keep that exact signal for contact
        # reward and certification; the critic may still observe both blocks.
        values = [
            sensor.data.net_forces_w.reshape(self.num_envs, -1, 3)
            .norm(dim=-1).max(dim=-1).values
            for sensor in self._contact_sensors
        ]
        return torch.stack(values, dim=1)

    def runtime_snapshot(self) -> dict[str, torch.Tensor]:
        plate, sponge = self._object_poses()
        return {
            "joint_q": self.hand.data.joint_pos[:, self.sim_joint_ids],
            "joint_qd": self.hand.data.joint_vel[:, self.sim_joint_ids],
            "target_q": self._target_q[:, self.sim_joint_ids],
            "plate_pose": plate, "sponge_pose": sponge,
            "left_tips": self._tips(ControlledSide.LEFT), "right_tips": self._tips(ControlledSide.RIGHT),
            "contact_force_n": self.contact_forces(),
            "penetration_depth_m": self.penetration_depth(),
            "reference_index": self.clock.indices,
        }

    def set_evaluation_joint_offsets(self, offsets: torch.Tensor) -> None:
        if offsets.shape != (self.num_envs, 58) or not bool(torch.isfinite(offsets).all()):
            raise ValueError(f"evaluation offsets must be finite ({self.num_envs}, 58)")
        if bool((offsets.abs() > 0.0050001).any()):
            raise ValueError("evaluation joint offsets exceed frozen +/-0.005 rad")
        self._evaluation_joint_offsets.copy_(offsets)

    def camera_frames(self) -> dict[str, torch.Tensor]:
        if not self._record_cameras:
            raise RuntimeError("recording cameras were not enabled")
        return {name: camera.data.output["rgb"][..., :3] for name, camera in self._record_cameras.items()}

    def penetration_depth(self) -> torch.Tensor:
        """Maximum PhysX contact penetration across hand/object and object/object pairs."""

        maximum = torch.zeros(self.num_envs, device=self.device)
        for sensor in (*self._contact_sensors, *self._penetration_sensors):
            _, _, _, separations, counts, starts = sensor.contact_physx_view.get_contact_data(dt=self.cfg.sim.dt)
            rows = counts.shape[0]
            per_row = torch.zeros(rows, device=self.device)
            for row in range(rows):
                for filt in range(counts.shape[1]):
                    count = int(counts[row, filt])
                    if count:
                        start = int(starts[row, filt])
                        stop = min(start + count, separations.numel())
                        if start < stop:
                            depth = (-separations[start:stop]).clamp(min=0.0).max()
                            per_row[row] = torch.maximum(per_row[row], depth)
            per_env = per_row.view(self.num_envs, -1).max(dim=1).values
            maximum = torch.maximum(maximum, per_env)
        return maximum

    def plate_sponge_contact_force(self) -> torch.Tensor:
        matrix = self._penetration_sensors[0].data.force_matrix_w
        return matrix.reshape(self.num_envs, -1, 3).norm(dim=-1).max(dim=-1).values
