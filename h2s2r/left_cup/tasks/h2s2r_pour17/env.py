"""Synchronized bimanual H2S2R training environment for Pour17.

The simulator, robot, assets, reset and external evaluation quantities match the
frozen Pour17 bundle.  The policy-facing action, observation, reward and FABRICS
controller remain H2S2R-derived and do not import ours priors or reward code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate

from rl_rebuild.baselines.h2s2r.contract import (
    ACTION_DIM,
    ControlledSide,
    fingertip_body_names,
    joint_names_for_side,
    simulation_joint_names,
)
from rl_rebuild.baselines.h2s2r.controller import BimanualH2S2RFabricController
from rl_rebuild.baselines.h2s2r.pour17.bimanual import (
    SharedReferenceClock,
    bimanual_tracking_reward,
    joint_progress_success_reward,
    mean_fingertip_object_distance,
    object_keypoint_distance,
)
from rl_rebuild.baselines.h2s2r.pour17.inputs import load_inputs
from rl_rebuild.baselines.h2s2r.pour17.observation import (
    build_bimanual_observation,
    build_side_observation,
)
from rl_rebuild.baselines.h2s2r.pour17.reset import load_phase1_g2_reset
from rl_rebuild.baselines.h2s2r.pour17.trajectory import build_reference_trajectory
from rl_rebuild.baselines.h2s2r.synergy import load_synergy

from .cfg import Pour17H2S2REnvCfg


class Pour17H2S2REnv(DirectRLEnv):
    cfg: Pour17H2S2REnvCfg

    def _setup_scene(self) -> None:
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)       # object_1, bottle
        self.aux = RigidObject(self.cfg.aux_object_cfg)      # object_0, cup

        # object_1.usd is the visual asset used by ours at runtime.  Reproduce
        # the same runtime rigid-body/collision construction before cloning.
        import omni.usd
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        bottle_root = stage.GetPrimAtPath("/World/envs/env_0/Object")
        UsdPhysics.RigidBodyAPI.Apply(bottle_root)
        UsdPhysics.MassAPI.Apply(bottle_root).CreateMassAttr(self.cfg.bottle_mass_kg)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(bottle_root)
        rb.CreateSleepThresholdAttr(0.005)
        rb.CreateStabilizationThresholdAttr(0.0025)
        for prim in Usd.PrimRange(bottle_root):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                    "convexDecomposition"
                )
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr(0.002)
                collision.CreateRestOffsetAttr(0.0)

        sx, sy, sz = self.cfg.table_size
        table = sim_utils.CuboidCfg(
            size=(sx, sy, sz),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=self.cfg.table_friction,
                dynamic_friction=self.cfg.table_friction,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                # Visual-only sync to ours e6b08cf; collision/material physics unchanged.
                diffuse_color=(0.0, 0.0, 0.0)
            ),
        )
        table.func(
            "/World/envs/env_.*/Table",
            table,
            translation=(0.0, 0.0, self.cfg.table_top_z - sz / 2.0),
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        super_grip = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.bottle_friction,
            dynamic_friction=self.cfg.bottle_friction,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        super_grip.func("/World/Materials/SuperGrip", super_grip)
        low_grip = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.body_friction,
            dynamic_friction=self.cfg.body_friction,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        low_grip.func("/World/Materials/LowGrip", low_grip)
        for path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Object"):
            sim_utils.bind_physics_material(
                path,
                "/World/Materials/SuperGrip",
                stronger_than_descendants=True,
            )
        for path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Robot"):
            sim_utils.bind_physics_material(
                path,
                "/World/Materials/LowGrip",
                stronger_than_descendants=False,
            )
        for side in (ControlledSide.RIGHT, ControlledSide.LEFT):
            for name in fingertip_body_names(side):
                for path in sim_utils.find_matching_prim_paths(
                    f"/World/envs/env_.*/Robot/{name}"
                ):
                    sim_utils.bind_physics_material(
                        path,
                        "/World/Materials/SuperGrip",
                        stronger_than_descendants=True,
                    )

        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["aux"] = self.aux
        self._contact_sensors: list[ContactSensor] = []
        for index, sensor_cfg in enumerate(self.cfg.contact_sensors):
            sensor = ContactSensor(sensor_cfg)
            self._contact_sensors.append(sensor)
            self.scene.sensors[f"h2s2r_contact_{index}"] = sensor
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def __init__(
        self,
        cfg: Pour17H2S2REnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        root = Path(cfg.bundle_root)
        canonical = json.loads(
            (root / "world/canonical_reset_v1.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (root / "world/world_manifest.json").read_text(encoding="utf-8")
        )
        self._verify_frozen_world(canonical, manifest)

        names = list(self.hand.joint_names)
        self.sim_joint_ids = torch.tensor(
            [names.index(name) for name in simulation_joint_names()],
            dtype=torch.long,
            device=self.device,
        )
        self.side_joint_ids = {
            side: torch.tensor(
                [names.index(name) for name in joint_names_for_side(side)],
                dtype=torch.long,
                device=self.device,
            )
            for side in (ControlledSide.RIGHT, ControlledSide.LEFT)
        }
        body_names = list(self.hand.body_names)
        self.tip_body_ids = {
            side: torch.tensor(
                [body_names.index(name) for name in fingertip_body_names(side)],
                dtype=torch.long,
                device=self.device,
            )
            for side in (ControlledSide.RIGHT, ControlledSide.LEFT)
        }
        self.wrist_body_ids = {
            ControlledSide.RIGHT: body_names.index("right_hand_C_MC"),
            ControlledSide.LEFT: body_names.index("left_hand_C_MC"),
        }
        self.arm_center_body_id = body_names.index("arm_center")

        if cfg.reset_regime == "phase1_g2":
            reset = load_phase1_g2_reset(root)
            reset_q = reset.joint_q_rad
            reset_qd = reset.joint_qd_rad_s
            object_0_reset = reset.object_0_pose_wxyz
            object_1_reset = reset.object_1_pose_wxyz
            self.reset_metadata = {
                "regime": "phase1_g2",
                "ours_entry": "g2",
                "gates_preset": ["G1", "G2"],
                "interaction_row": reset.interaction_row,
                "reference_path": str(reset.reference_path),
                "reference_sha256": reset.reference_sha256,
            }
        else:
            reset_q = canonical["robot_initial_state"]["joint_q_rad"]
            reset_qd = canonical["robot_initial_state"]["joint_qd_rad_s"]
            object_0_reset = canonical["objects_initial_state"]["object_0_cup"][
                "pose_xyz_wxyz_env_frame"
            ]
            object_1_reset = canonical["objects_initial_state"]["object_1_bottle"][
                "pose_xyz_wxyz_env_frame"
            ]
            self.reset_metadata = {
                "regime": "canonical_t0",
                "ours_entry": "t0",
                "gates_preset": [],
            }
        q58 = torch.tensor(reset_q, dtype=torch.float32, device=self.device)
        qd58 = torch.tensor(reset_qd, dtype=torch.float32, device=self.device)
        self.reset_q58 = q58
        self.reset_qd58 = qd58
        self.reset_object_pose = {
            0: torch.tensor(object_0_reset, dtype=torch.float32, device=self.device),
            1: torch.tensor(object_1_reset, dtype=torch.float32, device=self.device),
        }

        provenance = cfg.provenance_manifest or None
        inputs = load_inputs(
            cfg.perception_npz,
            regime=cfg.input_regime,
            provenance_manifest=provenance,
        )
        reference = build_reference_trajectory(
            inputs,
            object_0_reset_pose_wxyz=self.reset_object_pose[0].cpu().numpy(),
            object_1_reset_pose_wxyz=self.reset_object_pose[1].cpu().numpy(),
            # Preserve the released H2S2R 30 Hz reference.  The floating clock
            # below performs the control-rate conversion and speed warping.
            control_hz=30.0,
            source_start_index=cfg.reference_start_index,
        )
        self.reference_object_pose = {
            0: torch.tensor(reference.object_0_pose_wxyz, device=self.device),
            1: torch.tensor(reference.object_1_pose_wxyz, device=self.device),
        }
        self.reference_valid = {
            0: torch.tensor(reference.object_0_source_valid, device=self.device),
            1: torch.tensor(reference.object_1_source_valid, device=self.device),
        }
        self.input_provenance = inputs.provenance
        self.reference_metadata = reference
        self.clock = SharedReferenceClock.create(
            self.num_envs,
            reference.length - 1,
            str(self.device),
            control_dt=cfg.sim.dt * cfg.decimation,
            reference_dt=1.0 / 30.0,
        )

        initial_q_right = q58[
            torch.tensor(
                list(range(0, 7)) + list(range(14, 36)), device=self.device
            )
        ].repeat(self.num_envs, 1)
        initial_q_left = q58[
            torch.tensor(
                list(range(7, 14)) + list(range(36, 58)), device=self.device
            )
        ].repeat(self.num_envs, 1)
        initial_qd_right = qd58[
            torch.tensor(
                list(range(0, 7)) + list(range(14, 36)), device=self.device
            )
        ].repeat(self.num_envs, 1)
        initial_qd_left = qd58[
            torch.tensor(
                list(range(7, 14)) + list(range(36, 58)), device=self.device
            )
        ].repeat(self.num_envs, 1)

        from fabrics_sim.utils.utils import initialize_warp

        synergy_paths = (cfg.right_synergy_npz, cfg.left_synergy_npz)
        if not all(synergy_paths) and not cfg.allow_analytic_synergy:
            raise ValueError(
                "formal H2S2R runs require frozen right and left PCA synergy files; "
                "the analytic fallback is allowed only for explicit smoke tests"
            )
        for synergy_path in filter(None, synergy_paths):
            synergy = load_synergy(synergy_path)
            if inputs.provenance.sha256 not in synergy.source:
                raise ValueError(
                    f"synergy {synergy_path} was not fitted from the declared input "
                    f"archive {inputs.provenance.sha256}"
                )

        initialize_warp(warp_cache_name=str(self.device))
        world_dict = self._fabric_world_dict()
        self.controller = BimanualH2S2RFabricController(
            num_envs=self.num_envs,
            device=str(self.device),
            timestep=cfg.sim.dt * cfg.decimation,
            right_urdf_path=cfg.right_fabric_urdf,
            left_urdf_path=cfg.left_fabric_urdf,
            initial_q_right=initial_q_right,
            initial_q_left=initial_q_left,
            initial_qd_right=initial_qd_right,
            initial_qd_left=initial_qd_left,
            right_synergy_path=cfg.right_synergy_npz or None,
            left_synergy_path=cfg.left_synergy_npz or None,
            world_dict=world_dict,
        )

        self._target_q = self.hand.data.joint_pos.clone()
        self._actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._certification_alpha = torch.zeros(self.num_envs, device=self.device)
        self._success_run = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._success_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._warmup_clamp_left = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        current_0, current_1 = self._object_poses()
        self._previous_object_pose = {0: current_0.clone(), 1: current_1.clone()}
        self._previous_previous_object_pose = {
            0: current_0.clone(),
            1: current_1.clone(),
        }
        self._last_distances = {
            0: torch.zeros(self.num_envs, device=self.device),
            1: torch.zeros(self.num_envs, device=self.device),
        }
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._validate_runtime_contract()

    def _validate_runtime_contract(self) -> None:
        """Fail fast if shared physics assets or consumed contact tensors are invalid."""

        if len(self._contact_sensors) != 10:
            raise RuntimeError(
                f"expected 10 fingertip-object contact sensors, got {len(self._contact_sensors)}"
            )
        for index, sensor in enumerate(self._contact_sensors):
            forces = sensor.data.force_matrix_w
            if forces is None:
                raise RuntimeError(f"contact sensor {index} has no filtered force tensor")
            if (
                forces.shape[0] != self.num_envs
                or forces.shape[-1] != 3
                or forces.numel() == 0
                or forces.numel() % (self.num_envs * 3) != 0
            ):
                raise RuntimeError(
                    f"contact sensor {index} shape {tuple(forces.shape)} is incompatible "
                    f"with {self.num_envs} environments"
                )
            if not bool(torch.isfinite(forces).all()):
                raise RuntimeError(f"contact sensor {index} contains non-finite forces")

        for name, asset in (("bottle", self.object), ("cup", self.aux)):
            positions = asset.data.root_pos_w
            if positions.shape != (self.num_envs, 3):
                raise RuntimeError(
                    f"{name} root position shape {tuple(positions.shape)} is invalid"
                )
            if not bool(torch.isfinite(positions).all()):
                raise RuntimeError(f"{name} root positions contain non-finite values")

        import omni.usd
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        bottle_root = stage.GetPrimAtPath("/World/envs/env_0/Object")
        if not bottle_root.IsValid():
            raise RuntimeError("bottle root prim is missing after scene construction")
        required_root_apis = (
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.MassAPI,
            PhysxSchema.PhysxRigidBodyAPI,
        )
        missing = [api.__name__ for api in required_root_apis if not bottle_root.HasAPI(api)]
        if missing:
            raise RuntimeError(f"bottle root is missing physics APIs: {missing}")
        mass = UsdPhysics.MassAPI(bottle_root).GetMassAttr().Get()
        if mass is None or abs(float(mass) - self.cfg.bottle_mass_kg) > 1e-6:
            raise RuntimeError(
                f"bottle mass is {mass}, expected {self.cfg.bottle_mass_kg}"
            )

        meshes = [prim for prim in Usd.PrimRange(bottle_root) if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError("bottle asset contains no mesh prims")
        for prim in meshes:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(f"{prim.GetPath()} is missing CollisionAPI")
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                raise RuntimeError(f"{prim.GetPath()} is missing MeshCollisionAPI")
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if str(approximation) != "convexDecomposition":
                raise RuntimeError(
                    f"{prim.GetPath()} collision approximation is {approximation}"
                )
        print(
            "[H2S2R] runtime contract verified: 10 contact tensors; "
            f"bottle rigid/mass/collision APIs; {len(meshes)} collision meshes"
        )

    def _verify_frozen_world(self, canonical: dict, manifest: dict) -> None:
        expected_names = list(simulation_joint_names())
        if canonical["robot_initial_state"]["joint_names_in_order"] != expected_names:
            raise ValueError("canonical reset joint order differs from the adapter contract")
        if manifest["robot"]["controlled_joint_names_in_order"] != expected_names:
            raise ValueError("world manifest joint order differs from the adapter contract")
        if canonical["warmup_clamp"]["evaluation_protocol"]["POUR_HOLD_K"] != 0:
            raise ValueError("canonical evaluation warmup must be POUR_HOLD_K=0")
        training_hold = int(canonical["warmup_clamp"]["ours_training_default"]["POUR_HOLD_K"])
        if self.cfg.external_evaluator_controls_termination:
            if self.cfg.warmup_clamp_steps != 0:
                raise ValueError("formal evaluation must use POUR_HOLD_K=0")
        elif self.cfg.reset_regime == "phase1_g2":
            if self.cfg.warmup_clamp_steps != training_hold:
                raise ValueError(
                    "phase-one training warmup must match ours POUR_HOLD_K="
                    f"{training_hold}"
                )
        if abs(float(manifest["time"]["control_dt_s"]) - 0.05) > 1e-9:
            raise ValueError("world manifest control_dt is not 0.05 s")
        if int(canonical["episode_horizon"]["max_control_steps"]) != 903:
            raise ValueError("canonical episode horizon is not 903")

    def _fabric_world_dict(self) -> dict:
        """Express the shared table cuboid in the arm-center kinematic frame."""

        position = (
            self.hand.data.body_pos_w[0, self.arm_center_body_id]
            - self.scene.env_origins[0]
        )
        quaternion = self.hand.data.body_quat_w[0, self.arm_center_body_id]
        table_world = torch.tensor(
            [0.0, 0.0, self.cfg.table_top_z - self.cfg.table_size[2] / 2.0],
            dtype=torch.float32,
            device=self.device,
        )
        table_root = quat_apply(
            quat_conjugate(quaternion).unsqueeze(0),
            (table_world - position).unsqueeze(0),
        )[0]
        # WorldMeshesModel consumes x y z qx qy qz qw.
        q_xyzw = quaternion[[1, 2, 3, 0]]
        inverse_q_xyzw = torch.cat([-q_xyzw[:3], q_xyzw[3:4]])
        transform = torch.cat([table_root, inverse_q_xyzw]).cpu().tolist()
        return {
            "table": {
                "env_index": "all",
                "type": "box",
                "scaling": " ".join(str(value) for value in self.cfg.table_size),
                "transform": " ".join(str(float(value)) for value in transform),
            }
        }

    def _object_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        origins = self.scene.env_origins
        cup = torch.cat(
            [self.aux.data.root_pos_w - origins, self.aux.data.root_quat_w], dim=-1
        )
        bottle = torch.cat(
            [self.object.data.root_pos_w - origins, self.object.data.root_quat_w],
            dim=-1,
        )
        return cup, bottle

    def _goal_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.clock.indices
        return self.reference_object_pose[0][indices], self.reference_object_pose[1][indices]

    def _tips(self, side: ControlledSide) -> torch.Tensor:
        return (
            self.hand.data.body_pos_w[:, self.tip_body_ids[side]]
            - self.scene.env_origins.unsqueeze(1)
        )

    def _wrist(self, side: ControlledSide) -> tuple[torch.Tensor, torch.Tensor]:
        body_id = self.wrist_body_ids[side]
        return (
            self.hand.data.body_pos_w[:, body_id] - self.scene.env_origins,
            self.hand.data.body_quat_w[:, body_id],
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1.0, 1.0))

        # The evaluator override is one-shot.  If an older evaluator omits the
        # final alpha=0 callback, the offset still cannot leak into later steps.
        alpha = self._certification_alpha.clone()
        self._certification_alpha.zero_()
        arm_center_q = self.hand.data.body_quat_w[:, self.arm_center_body_id]
        world_z = torch.zeros(self.num_envs, 3, device=self.device)
        world_z[:, 2] = self.cfg.certification_lift_m
        root_z = quat_apply(quat_conjugate(arm_center_q), world_z)
        offsets = (alpha.unsqueeze(1) * root_z).unsqueeze(1).expand(-1, 2, -1)
        right_q, left_q = self.controller.step(
            self._actions, palm_position_offsets=offsets
        )
        self._target_q.copy_(self.hand.data.joint_pos)
        self._target_q[:, self.side_joint_ids[ControlledSide.RIGHT]] = right_q
        self._target_q[:, self.side_joint_ids[ControlledSide.LEFT]] = left_q

    def _apply_action(self) -> None:
        self.hand.set_joint_position_target(self._target_q)

    def _tracking_quantities(self) -> dict[str, torch.Tensor]:
        cup, bottle = self._object_poses()
        goal_cup, goal_bottle = self._goal_poses()
        left_tips = self._tips(ControlledSide.LEFT)
        right_tips = self._tips(ControlledSide.RIGHT)
        return {
            "cup": cup,
            "bottle": bottle,
            "goal_cup": goal_cup,
            "goal_bottle": goal_bottle,
            "left_tips": left_tips,
            "right_tips": right_tips,
            "cup_error": object_keypoint_distance(cup, goal_cup),
            "bottle_error": object_keypoint_distance(bottle, goal_bottle),
            "left_tip_distance": mean_fingertip_object_distance(left_tips, cup[:, :3]),
            "right_tip_distance": mean_fingertip_object_distance(
                right_tips, bottle[:, :3]
            ),
        }

    def _get_rewards(self) -> torch.Tensor:
        warmup_active = self._warmup_clamp_left > 0
        self._apply_warmup_clamp()
        values = self._tracking_quantities()
        reward, terms = bimanual_tracking_reward(
            values["cup_error"],
            values["bottle_error"],
            values["left_tip_distance"],
            values["right_tip_distance"],
            previous_actions=self._previous_actions,
            actions=self._actions,
            smoothing_weight=self.cfg.action_smoothing_weight,
        )
        self._last_distances[0] = values["cup_error"]
        self._last_distances[1] = values["bottle_error"]
        shared_advanced = self.clock.step(
            values["cup_error"], values["bottle_error"], frozen=warmup_active
        )
        at_final = self.clock.is_complete
        in_success = at_final & (
            values["cup_error"] < self.cfg.success_region_radius_m
        ) & (values["bottle_error"] < self.cfg.success_region_radius_m)
        was_successful = self._success_latched.clone()
        self._success_run = torch.where(
            in_success, self._success_run + 1, torch.zeros_like(self._success_run)
        )
        self._success_latched |= self._success_run >= 60  # Original H2S2R: 3 s.
        new_joint_success = self._success_latched & ~was_successful
        joint_reward, joint_terms = joint_progress_success_reward(
            shared_advanced,
            new_joint_success,
            progress_weight=self.cfg.joint_progress_reward_weight,
            success_weight=self.cfg.joint_success_reward_weight,
        )
        reward = reward + joint_reward
        self.extras["success_rate"] = self._success_latched.float().mean()
        self.extras["h2s2r/reference_index_mean"] = self.clock.indices.float().mean()
        self.extras["h2s2r/reference_float_index_mean"] = self.clock.float_indices.mean()
        self.extras["h2s2r/reference_speed_factor_mean"] = self.clock.speed_factors.mean()
        self.extras["h2s2r/warmup_frozen_rate"] = warmup_active.float().mean()
        self.extras["h2s2r/cup_tracking_reward"] = terms[
            "left_object_tracking"
        ].mean()
        self.extras["h2s2r/bottle_tracking_reward"] = terms[
            "right_object_tracking"
        ].mean()
        self.extras["h2s2r/shared_advance_rate"] = shared_advanced.float().mean()
        self.extras["h2s2r/joint_progress_reward"] = joint_terms[
            "joint_progress"
        ].mean()
        self.extras["h2s2r/joint_success_reward"] = joint_terms[
            "joint_success"
        ].mean()
        return reward

    def _apply_warmup_clamp(self) -> None:
        """Mirror ours' phase-one training-only object settling clamp."""

        holding = self._warmup_clamp_left > 0
        if not holding.any():
            return
        env_ids = holding.nonzero(as_tuple=False).flatten()
        origins = self.scene.env_origins[env_ids]
        for object_id, rigid_object in ((0, self.aux), (1, self.object)):
            pose = self.reset_object_pose[object_id].repeat(len(env_ids), 1)
            pose[:, :3] += origins
            rigid_object.write_root_pose_to_sim(pose, env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(
                torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
            )
        self._warmup_clamp_left[holding] -= 1

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.external_evaluator_controls_termination:
            # The frozen Pour17 evaluator owns every deadline and the 903-step
            # cap.  Suppressing DirectRLEnv auto-reset lets it inspect the
            # terminal physical state before deciding and attributing failure.
            never = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            return never, never
        cup, bottle = self._object_poses()
        dropped = (cup[:, 2] < self.cfg.table_top_z - 0.05) | (
            bottle[:, 2] < self.cfg.table_top_z - 0.05
        )
        if self.cfg.early_reset_if_fingertips_far:
            values = self._tracking_quantities()
            dropped |= (
                values["left_tip_distance"] >= self.cfg.fingertips_close_threshold_m
            ) | (
                values["right_tip_distance"] >= self.cfg.fingertips_close_threshold_m
            )
        if self.cfg.terminate_when_reference_ends:
            dropped |= self.clock.is_complete
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dropped, time_out

    def _get_observations(self) -> dict[str, torch.Tensor]:
        cup, bottle = self._object_poses()
        goal_cup, goal_bottle = self._goal_poses()
        observations = {}
        for side, object_pose, goal_pose, object_id in (
            (ControlledSide.RIGHT, bottle, goal_bottle, 1),
            (ControlledSide.LEFT, cup, goal_cup, 0),
        ):
            ids = self.side_joint_ids[side]
            wrist_position, wrist_quaternion = self._wrist(side)
            controller = (
                self.controller.right if side is ControlledSide.RIGHT else self.controller.left
            )
            observations[side] = build_side_observation(
                q=self.hand.data.joint_pos[:, ids],
                qd=self.hand.data.joint_vel[:, ids],
                fingertip_positions=self._tips(side),
                palm_position=wrist_position,
                palm_quaternion_wxyz=wrist_quaternion,
                object_pose_wxyz=object_pose,
                goal_object_pose_wxyz=goal_pose,
                previous_object_pose_wxyz=self._previous_object_pose[object_id],
                previous_previous_object_pose_wxyz=(
                    self._previous_previous_object_pose[object_id]
                ),
                fabric_q=controller.q,
                fabric_qd=controller.qd,
            )
        result = build_bimanual_observation(
            observations[ControlledSide.RIGHT], observations[ControlledSide.LEFT]
        ).clamp(-self.cfg.clip_obs, self.cfg.clip_obs).nan_to_num(0.0)
        self._previous_previous_object_pose[0].copy_(self._previous_object_pose[0])
        self._previous_previous_object_pose[1].copy_(self._previous_object_pose[1])
        self._previous_object_pose[0].copy_(cup)
        self._previous_object_pose[1].copy_(bottle)
        # The repository PPO always indexes ``priv_info`` even when the model
        # is configured not to use privileged information.  Width zero keeps
        # H2S2R's observation contract unchanged while satisfying that API.
        return {
            "policy": result,
            "priv_info": torch.empty(self.num_envs, 0, device=self.device),
        }

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return
        DirectRLEnv._reset_idx(self, env_ids)
        n = len(env_ids)

        q = self.hand.data.default_joint_pos[env_ids].clone()
        qd = torch.zeros_like(q)
        q[:, self.sim_joint_ids] = self.reset_q58
        qd[:, self.sim_joint_ids] = self.reset_qd58
        self.hand.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        self.hand.set_joint_position_target(q, env_ids=env_ids)

        origins = self.scene.env_origins[env_ids]
        for object_id, rigid_object in ((0, self.aux), (1, self.object)):
            pose = self.reset_object_pose[object_id].repeat(n, 1)
            pose[:, :3] += origins
            rigid_object.write_root_pose_to_sim(pose, env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(
                torch.zeros(n, 6, device=self.device), env_ids=env_ids
            )

        q_right = self.reset_q58[
            torch.tensor(
                list(range(0, 7)) + list(range(14, 36)), device=self.device
            )
        ].repeat(n, 1)
        q_left = self.reset_q58[
            torch.tensor(
                list(range(7, 14)) + list(range(36, 58)), device=self.device
            )
        ].repeat(n, 1)
        self.controller.reset(
            env_ids,
            q_right=q_right,
            q_left=q_left,
            qd_right=torch.zeros_like(q_right),
            qd_left=torch.zeros_like(q_left),
        )
        self.clock.reset(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._certification_alpha[env_ids] = 0.0
        self._success_run[env_ids] = 0
        self._success_latched[env_ids] = False
        self._warmup_clamp_left[env_ids] = int(self.cfg.warmup_clamp_steps)
        for object_id in (0, 1):
            pose = self.reset_object_pose[object_id].repeat(n, 1)
            self._previous_object_pose[object_id][env_ids] = pose
            self._previous_previous_object_pose[object_id][env_ids] = pose

    def apply_certification_offset(
        self,
        alpha: float | torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Schedule one step of the evaluator's world-+Z 15 mm wrist override."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        value = torch.as_tensor(alpha, dtype=torch.float32, device=self.device)
        if value.ndim == 0:
            value = value.expand(len(env_ids))
        if value.shape != (len(env_ids),):
            raise ValueError(f"alpha must be scalar or ({len(env_ids)},), got {value.shape}")
        self._certification_alpha[env_ids] = value.clamp(0.0, 1.0)

    def evaluator_info(self) -> dict[str, torch.Tensor]:
        """Method-agnostic measured quantities consumed by the shared evaluator."""

        cup, bottle = self._object_poses()
        q = self.hand.data.joint_pos
        right_arm = self.side_joint_ids[ControlledSide.RIGHT][:7]
        left_arm = self.side_joint_ids[ControlledSide.LEFT][:7]
        forces = torch.stack(
            [
                sensor.data.force_matrix_w.reshape(self.num_envs, -1, 3).norm(dim=-1).max(dim=-1).values
                for sensor in self._contact_sensors[:10]
            ],
            dim=1,
        )
        pads3 = ((forces[:, :5] > 0.5).sum(dim=1) >= 3) & (
            (forces[:, 5:] > 0.5).sum(dim=1) >= 3
        )
        wrist_right, _ = self._wrist(ControlledSide.RIGHT)
        wrist_left, _ = self._wrist(ControlledSide.LEFT)
        return {
            "object_0_pose": cup,
            "object_1_pose": bottle,
            "arm_q_right": q[:, right_arm],
            "arm_q_left": q[:, left_arm],
            "pads3": pads3,
            "wrist_right": wrist_right,
            "wrist_left": wrist_left,
        }

    def reset_state(self) -> dict:
        return {
            "joint_names": list(simulation_joint_names()),
            "joint_q_rad": self.reset_q58.cpu().tolist(),
            "joint_qd_rad_s": self.reset_qd58.cpu().tolist(),
            "object_0_pose_wxyz": self.reset_object_pose[0].cpu().tolist(),
            "object_1_pose_wxyz": self.reset_object_pose[1].cpu().tolist(),
            "warmup_clamp_steps": int(self.cfg.warmup_clamp_steps),
            "reset": self.reset_metadata,
            "input_regime": self.input_provenance.regime.value,
            "task_supervision_kind": self.input_provenance.task_supervision_kind.value,
            "input_sha256": self.input_provenance.sha256,
        }
