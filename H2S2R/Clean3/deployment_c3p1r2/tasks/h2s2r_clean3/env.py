"""Real bimanual H2S2R environment for Clean3 C3-P1R2."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import hashlib

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate

from c3p1r2.action_calibration import install_bimanual
from clean3_adapter.cadence import advance_bimanual, set_bimanual_targets
from rl_rebuild.baselines.h2s2r.contract import (
    ACTION_DIM, ControlledSide, fingertip_body_names, joint_names_for_side,
    simulation_joint_names,
)
from rl_rebuild.baselines.h2s2r.controller import BimanualH2S2RFabricController
from rl_rebuild.baselines.h2s2r.pour17.bimanual import (
    SharedReferenceClock, bimanual_tracking_reward, mean_fingertip_object_distance,
    object_keypoint_distance,
)
from rl_rebuild.baselines.h2s2r.pour17.observation import (
    build_bimanual_observation, build_side_observation,
)

from .cfg import Clean3H2S2REnvCfg


class Clean3H2S2REnv(DirectRLEnv):
    cfg: Clean3H2S2REnvCfg

    def _prepare_rigid_asset(self, path: str, mass: float) -> None:
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

    def _setup_scene(self) -> None:
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)  # object_0 plate, left hand
        self.aux = RigidObject(self.cfg.aux_object_cfg)  # object_1 sponge, right hand
        self._prepare_rigid_asset("/World/envs/env_0/Object", self.cfg.plate_mass_kg)
        self._prepare_rigid_asset("/World/envs/env_0/Aux", self.cfg.sponge_mass_kg)
        self._record_cameras = {}
        if self.cfg.record_cameras:
            for name in ("front", "side", "top"):
                camera_cfg = TiledCameraCfg(
                    prim_path=f"/World/envs/env_.*/Camera{name.title()}",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.5), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
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
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

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
        for root in ("Object", "Aux"):
            for path in sim_utils.find_matching_prim_paths(f"/World/envs/env_.*/{root}"):
                sim_utils.bind_physics_material(path, "/World/Materials/Clean3Object", stronger_than_descendants=True)
        for path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Robot"):
            sim_utils.bind_physics_material(path, "/World/Materials/Clean3Body", stronger_than_descendants=False)
        for side in (ControlledSide.RIGHT, ControlledSide.LEFT):
            for name in fingertip_body_names(side):
                for path in sim_utils.find_matching_prim_paths(f"/World/envs/env_.*/Robot/{name}"):
                    sim_utils.bind_physics_material(path, "/World/Materials/Clean3Object", stronger_than_descendants=True)

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

    def __init__(self, cfg: Clean3H2S2REnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        expected_sha = {
            cfg.runtime_inputs_npz: "3da2b16f27d74be4898a00fc807e6993dc1915eadbaff2f9aa8539fd1a6a84ff",
            cfg.action_sidecar_npz: "f878f87f61f218175292583b266f6150142c95dc3366a328c516383e58507747",
            cfg.right_synergy_npz: "31c349ea79313a504b3a809a51bbf9fbc646922ae86c3363e0688c8475df36e2",
            cfg.left_synergy_npz: "578ff0a8a9ada51cdf0ff7a442fa04d6ed22dd51e1130366f1936863fe7fd521",
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
        bodies = list(self.hand.body_names)
        self.tip_body_ids = {
            side: torch.tensor([bodies.index(name) for name in fingertip_body_names(side)], dtype=torch.long, device=self.device)
            for side in (ControlledSide.RIGHT, ControlledSide.LEFT)
        }
        self.wrist_body_ids = {
            ControlledSide.RIGHT: bodies.index("right_hand_C_MC"),
            ControlledSide.LEFT: bodies.index("left_hand_C_MC"),
        }
        self.arm_center_body_id = bodies.index("arm_center")
        self.clock = SharedReferenceClock.create(
            self.num_envs, 299, str(self.device), control_dt=cfg.sim.dt * cfg.decimation, reference_dt=1.0 / 30.0,
        )
        right_ids = torch.tensor(list(range(0, 7)) + list(range(14, 36)), device=self.device)
        left_ids = torch.tensor(list(range(7, 14)) + list(range(36, 58)), device=self.device)
        q_right = self.reset_q58[right_ids].repeat(self.num_envs, 1)
        q_left = self.reset_q58[left_ids].repeat(self.num_envs, 1)
        from fabrics_sim.utils.utils import initialize_warp
        initialize_warp(warp_cache_name=str(self.device))
        self.controller = BimanualH2S2RFabricController(
            num_envs=self.num_envs, device=str(self.device), timestep=cfg.sim.dt, right_timestep=cfg.sim.dt,
            right_urdf_path=cfg.right_fabric_urdf, left_urdf_path=cfg.left_fabric_urdf,
            initial_q_right=q_right, initial_q_left=q_left,
            initial_qd_right=torch.zeros_like(q_right), initial_qd_left=torch.zeros_like(q_left),
            right_synergy_path=cfg.right_synergy_npz, left_synergy_path=cfg.left_synergy_npz,
            world_dict=self._fabric_world_dict(),
        )
        install_bimanual(self.controller, cfg.action_sidecar_npz)
        zero = torch.zeros(self.num_envs, 22, device=self.device)
        nominal_centers = {
            "right": self.controller.right.palm_policy_center.clone(),
            "left": self.controller.left.palm_policy_center.clone(),
        }
        set_bimanual_targets(self.controller, zero)
        self.palm_zero_errors_m = {
            "right": float((self.controller.right.palm_target[:, :3] - nominal_centers["right"][:, :3]).abs().max()),
            "left": float((self.controller.left.palm_target[:, :3] - nominal_centers["left"][:, :3]).abs().max()),
        }
        if max(self.palm_zero_errors_m.values()) > 1e-6:
            raise RuntimeError(f"actual set_targets palm zero mismatch: {self.palm_zero_errors_m}")
        if abs(float(self.controller.left.palm_minimum[2]) + 0.03) > 1e-8:
            raise RuntimeError("Clean3 left palm z minimum is not frozen at -0.03 m")
        self._target_q = self.hand.data.joint_pos.clone()
        self._actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._previous_object_pose = {0: self._object_poses()[0].clone(), 1: self._object_poses()[1].clone()}
        self._previous_previous_object_pose = {0: self._previous_object_pose[0].clone(), 1: self._previous_object_pose[1].clone()}
        self.runtime_counts = {"controls": 0, "physics_substeps": 0, "right_set_targets": 0, "left_set_targets": 0, "right_advance": 0, "left_advance": 0}
        self.substep_callback = None
        self._evaluation_joint_offsets = torch.zeros(self.num_envs, 58, device=self.device)
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        if self._record_cameras:
            local_views = {
                "front": ((0.35, -0.45, 1.28), (-0.15, 0.04, 1.02)),
                "side": ((-0.68, 0.38, 1.25), (-0.15, 0.04, 1.02)),
                "top": ((-0.15, 0.04, 1.72), (-0.15, 0.04, 0.98)),
            }
            origins = self.scene.env_origins
            for name, (eye, target) in local_views.items():
                eyes = origins + torch.tensor(eye, device=self.device)
                targets = origins + torch.tensor(target, device=self.device)
                self._record_cameras[name].set_world_poses_from_view(eyes, targets)
        self._validate_runtime_contract()

    def _validate_runtime_contract(self) -> None:
        if len(self._contact_sensors) != 10:
            raise RuntimeError("Clean3 requires ten side-filtered fingertip contact sensors")
        for sensor in self._contact_sensors:
            values = sensor.data.force_matrix_w
            if values is None or values.shape[0] != self.num_envs or not bool(torch.isfinite(values).all()):
                raise RuntimeError("invalid Clean3 filtered contact tensor")
        if self.cfg.warmup_clamp_steps != 0:
            raise RuntimeError("Clean3 C3-P1R2 objects must be dynamic from the first physics step")
        if self._actions.shape != (self.num_envs, 22):
            raise RuntimeError("Clean3 policy action contract must be 22")

    def _fabric_world_dict(self) -> dict:
        position = self.hand.data.body_pos_w[0, self.arm_center_body_id] - self.scene.env_origins[0]
        quaternion = self.hand.data.body_quat_w[0, self.arm_center_body_id]
        table_world = torch.tensor([0.0, 0.0, self.cfg.table_top_z - self.cfg.table_size[2] / 2.0], dtype=torch.float32, device=self.device)
        table_root = quat_apply(quat_conjugate(quaternion).unsqueeze(0), (table_world - position).unsqueeze(0))[0]
        q_xyzw = quaternion[[1, 2, 3, 0]]
        inverse = torch.cat([-q_xyzw[:3], q_xyzw[3:4]])
        values = torch.cat([table_root, inverse]).cpu().tolist()
        return {"table": {"env_index": "all", "type": "box", "scaling": " ".join(map(str, self.cfg.table_size)), "transform": " ".join(str(float(v)) for v in values)}}

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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1.0, 1.0))
        set_bimanual_targets(self.controller, self._actions)
        self.runtime_counts["controls"] += 1
        self.runtime_counts["right_set_targets"] += 1
        self.runtime_counts["left_set_targets"] += 1
        self._target_q.copy_(self.hand.data.joint_pos)

    def _apply_action(self) -> None:
        right_q, left_q = advance_bimanual(self.controller)
        self.runtime_counts["physics_substeps"] += 1
        self.runtime_counts["right_advance"] += 1
        self.runtime_counts["left_advance"] += 1
        self._target_q[:, self.side_joint_ids[ControlledSide.RIGHT]] = right_q
        self._target_q[:, self.side_joint_ids[ControlledSide.LEFT]] = left_q
        self.hand.set_joint_position_target(self._target_q)
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

    def _get_rewards(self):
        values = self._tracking()
        reward, terms = bimanual_tracking_reward(
            values["plate_error"], values["sponge_error"], values["left_tip_distance"], values["right_tip_distance"],
            previous_actions=self._previous_actions, actions=self._actions, smoothing_weight=0.0,
        )
        advanced = self.clock.step(values["plate_error"], values["sponge_error"])
        self.extras["h2s2r/plate_tracking_reward"] = terms["left_object_tracking"].mean()
        self.extras["h2s2r/sponge_tracking_reward"] = terms["right_object_tracking"].mean()
        self.extras["h2s2r/shared_advance_rate"] = advanced.float().mean()
        self.extras["h2s2r/reference_index_mean"] = self.clock.indices.float().mean()
        return reward

    def _get_dones(self):
        if self.cfg.external_eval:
            never = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return never, never
        plate, sponge = self._object_poses()
        terminated = (plate[:, 2] < self.cfg.table_top_z - 0.05) | (sponge[:, 2] < self.cfg.table_top_z - 0.05)
        if self.cfg.early_reset_if_fingertips_far:
            values = self._tracking()
            terminated |= (values["left_tip_distance"] >= self.cfg.fingertips_close_threshold_m) | (values["right_tip_distance"] >= self.cfg.fingertips_close_threshold_m)
        if self.cfg.terminate_when_reference_ends:
            terminated |= self.clock.is_complete
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timeout

    def _get_observations(self):
        plate, sponge = self._object_poses()
        goal_plate, goal_sponge = self._goal_poses()
        sides = {}
        for side, object_pose, goal_pose, object_id in (
            (ControlledSide.RIGHT, sponge, goal_sponge, 1),
            (ControlledSide.LEFT, plate, goal_plate, 0),
        ):
            ids = self.side_joint_ids[side]
            wrist_p, wrist_q = self._wrist(side)
            ctrl = self.controller.right if side is ControlledSide.RIGHT else self.controller.left
            sides[side] = build_side_observation(
                q=self.hand.data.joint_pos[:, ids], qd=self.hand.data.joint_vel[:, ids],
                fingertip_positions=self._tips(side), palm_position=wrist_p, palm_quaternion_wxyz=wrist_q,
                object_pose_wxyz=object_pose, goal_object_pose_wxyz=goal_pose,
                previous_object_pose_wxyz=self._previous_object_pose[object_id],
                previous_previous_object_pose_wxyz=self._previous_previous_object_pose[object_id],
                fabric_q=ctrl.q, fabric_qd=ctrl.qd,
            )
        obs = build_bimanual_observation(sides[ControlledSide.RIGHT], sides[ControlledSide.LEFT]).clamp(-self.cfg.clip_obs, self.cfg.clip_obs).nan_to_num(0.0)
        self._previous_previous_object_pose[0].copy_(self._previous_object_pose[0])
        self._previous_previous_object_pose[1].copy_(self._previous_object_pose[1])
        self._previous_object_pose[0].copy_(plate)
        self._previous_object_pose[1].copy_(sponge)
        return {"policy": obs, "priv_info": torch.empty(self.num_envs, 0, device=self.device)}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return
        DirectRLEnv._reset_idx(self, env_ids)
        n = len(env_ids)
        q = self.hand.data.default_joint_pos[env_ids].clone()
        qd = torch.zeros_like(q)
        reset_q = self.reset_q58.unsqueeze(0) + self._evaluation_joint_offsets[env_ids]
        q[:, self.sim_joint_ids] = reset_q
        self.hand.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        self.hand.set_joint_position_target(q, env_ids=env_ids)
        origins = self.scene.env_origins[env_ids]
        for object_id, rigid in ((0, self.object), (1, self.aux)):
            pose = self.reset_object_pose[object_id].repeat(n, 1)
            pose[:, :3] += origins
            rigid.write_root_pose_to_sim(pose, env_ids=env_ids)
            rigid.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
        right_contract_ids = torch.tensor(list(range(0, 7)) + list(range(14, 36)), device=self.device)
        left_contract_ids = torch.tensor(list(range(7, 14)) + list(range(36, 58)), device=self.device)
        q_right = reset_q[:, right_contract_ids]
        q_left = reset_q[:, left_contract_ids]
        self.controller.reset(env_ids, q_right=q_right, q_left=q_left, qd_right=torch.zeros_like(q_right), qd_left=torch.zeros_like(q_left))
        self.clock.reset(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        for object_id in (0, 1):
            pose = self.reset_object_pose[object_id].repeat(n, 1)
            self._previous_object_pose[object_id][env_ids] = pose
            self._previous_previous_object_pose[object_id][env_ids] = pose

    def contact_forces(self) -> torch.Tensor:
        values = [sensor.data.force_matrix_w.reshape(self.num_envs, -1, 3).norm(dim=-1).max(dim=-1).values for sensor in self._contact_sensors]
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
        for sensor in self._penetration_sensors:
            _, _, _, separations, counts, starts = sensor.contact_physx_view.get_contact_data(dt=self.cfg.sim.dt)
            rows = counts.shape[0]
            per_row = torch.zeros(rows, device=self.device)
            for row in range(rows):
                for filt in range(counts.shape[1]):
                    count = int(counts[row, filt])
                    if count:
                        start = int(starts[row, filt])
                        depth = (-separations[start:start + count]).clamp(min=0.0).max()
                        per_row[row] = torch.maximum(per_row[row], depth)
            per_env = per_row.view(self.num_envs, -1).max(dim=1).values
            maximum = torch.maximum(maximum, per_env)
        return maximum

    def plate_sponge_contact_force(self) -> torch.Tensor:
        matrix = self._penetration_sensors[2].data.force_matrix_w
        return matrix.reshape(self.num_envs, -1, 3).norm(dim=-1).max(dim=-1).values
