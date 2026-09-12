"""Joint 58-D state-RL environment for the isolated Pour17 WP1 adapter."""

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
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from .augmentation import sample_placement
from .geometry import ArmIK, UrdfKinematics, ndarray_sha256, sha256
from .reward import RewardConfig, object_mimic_bimanual
from .tensor_contact_probe import GpuPairContacts
from .schema import OBSERVATION_DIM


LANDMARKS = (
    ("thumb_tip", "thumb_fingertip", (0.0, 0.0, 0.0)),
    ("thumb_middle", "thumb_PP", (0.0195, 0.0, 0.0)),
    ("index_tip", "index_fingertip", (0.0, 0.0, 0.0)),
    ("index_middle", "index_MP", (0.01575, 0.0, 0.0)),
    ("middle_tip", "middle_fingertip", (0.0, 0.0, 0.0)),
    ("middle_middle", "middle_MP", (0.01575, 0.0, 0.0)),
    ("ring_tip", "ring_fingertip", (0.0, 0.0, 0.0)),
    ("ring_middle", "ring_MP", (0.01575, 0.0, 0.0)),
    ("pinky_tip", "pinky_fingertip", (0.0, 0.0, 0.0)),
    ("pinky_middle", "pinky_MP", (0.01575, 0.0, 0.0)),
)


def _relative_pose(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pose of target in source coordinates, both xyz+wxyz."""
    inverse = quat_conjugate(source[..., 3:])
    position = quat_apply(inverse, target[..., :3] - source[..., :3])
    quaternion = quat_mul(inverse, target[..., 3:])
    return torch.cat((position, quaternion), dim=-1)


class Pour17JointEnv(DirectRLEnv):
    """One simulator process with N vectorized, physically measured environments."""

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)
        self.bottle = RigidObject(self.cfg.bottle)
        self.cup = RigidObject(self.cfg.cup)

        # The registered bottle is a visual USD; reproduce the recorded runtime
        # rigid-body and convex-decomposition construction before cloning.
        import omni.usd
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        bottle_root = stage.GetPrimAtPath("/World/envs/env_0/Object")
        UsdPhysics.RigidBodyAPI.Apply(bottle_root)
        UsdPhysics.MassAPI.Apply(bottle_root).CreateMassAttr(0.53)
        rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(bottle_root)
        rigid.CreateSleepThresholdAttr(0.005)
        rigid.CreateStabilizationThresholdAttr(0.0025)
        PhysxSchema.PhysxContactReportAPI.Apply(bottle_root).CreateThresholdAttr(0.0)
        cup_root = stage.GetPrimAtPath("/World/envs/env_0/Aux")
        if cup_root.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(cup_root).CreateThresholdAttr(0.0)
        else:
            raise RuntimeError("registered cup USD root is not a rigid body")
        for prim in Usd.PrimRange(bottle_root):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                    "convexDecomposition"
                )
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr(0.002)
                collision.CreateRestOffsetAttr(0.0)

        table = sim_utils.CuboidCfg(
            size=(1.2192, 1.8288, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        )
        table.func("/World/envs/env_.*/Table", table, translation=(0.0, 0.0, 0.85))
        # Contact-report metadata only: the table remains a static collider.
        # Do not add RigidBodyAPI, change mass, or use a kinematic substitute.
        for path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Table"):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        strong = sim_utils.RigidBodyMaterialCfg(
            static_friction=3.0, dynamic_friction=3.0, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply")
        weak = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.2, dynamic_friction=0.2, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply")
        strong.func("/World/Materials/Pour17Strong", strong)
        weak.func("/World/Materials/Pour17Weak", weak)
        for path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Object"):
            sim_utils.bind_physics_material(
                path, "/World/Materials/Pour17Strong", stronger_than_descendants=True)
        for path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Robot"):
            sim_utils.bind_physics_material(
                path, "/World/Materials/Pour17Weak", stronger_than_descendants=False)
        for side in ("right", "left"):
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                for path in sim_utils.find_matching_prim_paths(
                    f"/World/envs/env_.*/Robot/{side}_{finger}_elastomer"
                ):
                    sim_utils.bind_physics_material(
                        path, "/World/Materials/Pour17Strong", stronger_than_descendants=True)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["bottle"] = self.bottle
        self.scene.rigid_objects["cup"] = self.cup
        self.contact_sensors: list[ContactSensor] = []
        for index, sensor_cfg in enumerate(self.cfg.contact_sensors):
            sensor = ContactSensor(sensor_cfg)
            self.contact_sensors.append(sensor)
            self.scene.sensors[f"pour17_tip_{index}"] = sensor
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        self.pair_contacts = GpuPairContacts(self)
        self.bundle_root = Path(cfg.bundle_root).resolve()
        self.reference_path = Path(cfg.reference_path).resolve()
        canonical = json.loads((self.bundle_root / "world/canonical_reset_v1.json").read_text())
        world_manifest = json.loads((self.bundle_root / "world/world_manifest.json").read_text())
        with np.load(self.reference_path, allow_pickle=False) as data:
            self.reference_np = {name: np.asarray(data[name]) for name in data.files}
        probe_only = bool(getattr(cfg, "probe_only", False))
        self.joint_names = tuple(world_manifest["robot"]["controlled_joint_names_in_order"])
        if tuple(str(value) for value in self.reference_np["joint_names"]) != self.joint_names:
            raise RuntimeError("reference/manifest joint order mismatch")
        if canonical["robot_initial_state"]["joint_names_in_order"] != list(self.joint_names):
            raise RuntimeError("canonical/manifest joint order mismatch")
        runtime_names = list(self.robot.joint_names)
        if set(runtime_names) != set(self.joint_names) or len(runtime_names) != 58:
            raise RuntimeError("runtime articulation is not the registered 58-joint robot")
        self.joint_ids = torch.tensor(
            [runtime_names.index(name) for name in self.joint_names],
            dtype=torch.long, device=self.device)
        self.arm_ids = {
            "right": self.joint_ids[:7], "left": self.joint_ids[7:14],
        }
        body_names = list(self.robot.body_names)
        self.wrist_body_ids = {
            side: body_names.index(f"{side}_hand_C_MC") for side in ("right", "left")
        }
        self.arm_center_body_id = body_names.index("arm_center")
        self.landmark_body_ids = {}
        for side in ("right", "left"):
            self.landmark_body_ids[side] = torch.tensor(
                [body_names.index(f"{side}_{suffix}") for _, suffix, _ in LANDMARKS],
                dtype=torch.long, device=self.device)
        self.landmark_local = torch.tensor(
            [offset for _, _, offset in LANDMARKS], dtype=torch.float32,
            device=self.device).unsqueeze(0)

        self.lower = torch.tensor(self.reference_np["joint_lower_rad"],
                                  dtype=torch.float32, device=self.device)
        self.upper = torch.tensor(self.reference_np["joint_upper_rad"],
                                  dtype=torch.float32, device=self.device)
        self.reset_q = torch.tensor(canonical["robot_initial_state"]["joint_q_rad"],
                                    dtype=torch.float32, device=self.device)
        self.reset_qd = torch.tensor(canonical["robot_initial_state"]["joint_qd_rad_s"],
                                     dtype=torch.float32, device=self.device)
        self.reset_object_pose = torch.tensor([
            canonical["objects_initial_state"]["object_1_bottle"]["pose_xyz_wxyz_env_frame"],
            canonical["objects_initial_state"]["object_0_cup"]["pose_xyz_wxyz_env_frame"],
        ], dtype=torch.float32, device=self.device)
        self.initial_object_z = self.reset_object_pose[:, 2].unsqueeze(0)
        if probe_only:
            self.reference_length = 1
            self.reference_q = self.reset_q.unsqueeze(0)
            self.reference_wrist = torch.zeros(1, 2, 7, device=self.device)
            self.reference_wrist[..., 3] = 1.0
            self.reference_object = self.reset_object_pose.unsqueeze(0)
            self.reference_hand = torch.zeros(1, 2, 10, 3, device=self.device)
            self.reference_stage = np.asarray(["approach"])
            self.pregrasp_end_step = 0
        else:
            self.reference_length = len(self.reference_np["control_time_s"])
            self.reference_q = torch.tensor(self.reference_np["joint_q_rad"],
                                            dtype=torch.float32, device=self.device)
            self.reference_wrist = torch.tensor(np.stack((
                self.reference_np["right_wrist_pose_wxyz"],
                self.reference_np["left_wrist_pose_wxyz"]), axis=1),
                dtype=torch.float32, device=self.device)
            self.reference_object = torch.tensor(
                self.reference_np["object_pose_wxyz"][:, [1, 0]],
                dtype=torch.float32, device=self.device)
            self.reference_hand = torch.tensor(np.stack((
                self.reference_np["right_hand_target_m"],
                self.reference_np["left_hand_target_m"]), axis=1),
                dtype=torch.float32, device=self.device)
            self.reference_stage = np.asarray(self.reference_np["stage"]).astype(str)
            self.pregrasp_end_step = int(np.flatnonzero(self.reference_stage != "approach")[0])
        self.palm_local = torch.tensor(self.reference_np["palm_local_m"],
                                       dtype=torch.float32, device=self.device)
        self.reference_palm = torch.tensor(self.reference_np["palm_target_m"],
                                           dtype=torch.float32, device=self.device)
        self.final_object_goal = torch.tensor(self.reference_np["final_object_goal_m"][[1, 0]],
                                               dtype=torch.float32, device=self.device)
        self.stage = int(getattr(cfg, "curriculum_stage", 0))
        self.rng = np.random.default_rng(int(cfg.seed))
        self.env_rng = [np.random.default_rng(int(cfg.seed) + i) for i in range(self.num_envs)]
        self.reference_xy_offset = torch.zeros(self.num_envs, 2, device=self.device)
        self.actions = torch.zeros(self.num_envs, 58, device=self.device)
        self.target_q = self.robot.data.joint_pos.clone()
        self.pregrasp_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pregrasp_decided = torch.zeros_like(self.pregrasp_success)
        self.external_evaluator_controls_termination = bool(
            getattr(cfg, "external_evaluator_controls_termination", False))
        self.certification_alpha = torch.zeros(self.num_envs, device=self.device)
        self.certification_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.certification_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.certification_start_q = torch.zeros(self.num_envs, 2, 7, device=self.device)
        self.certification_goal_q = torch.zeros_like(self.certification_start_q)
        self.certification_trace: list[dict] = []
        self.terminal_observation_by_env: dict[int, torch.Tensor] = {}
        self.terminal_stats_by_env: dict[int, dict] = {}
        self.boundary_trace = []
        self.reset_records = []
        self.episode_sample_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._arm_ik: dict[str, ArmIK] | None = None
        self._reset_idx(torch.arange(self.num_envs, device=self.device))

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            self.env_rng = [np.random.default_rng(int(seed) + i) for i in range(self.num_envs)]
        self.terminal_observation_by_env.clear()
        return super().reset(seed=seed, options=options)

    def _reference_index(self) -> torch.Tensor:
        return torch.clamp(self.episode_length_buf, max=self.reference_length - 1)

    def _object_poses(self) -> torch.Tensor:
        origins = self.scene.env_origins
        bottle = torch.cat((self.bottle.data.root_pos_w - origins,
                            self.bottle.data.root_quat_w), dim=-1)
        cup = torch.cat((self.cup.data.root_pos_w - origins,
                         self.cup.data.root_quat_w), dim=-1)
        return torch.stack((bottle, cup), dim=1)

    def _object_velocities(self) -> torch.Tensor:
        return torch.stack((
            torch.cat((self.bottle.data.root_lin_vel_w, self.bottle.data.root_ang_vel_w), dim=-1),
            torch.cat((self.cup.data.root_lin_vel_w, self.cup.data.root_ang_vel_w), dim=-1),
        ), dim=1)

    def _wrist_poses(self) -> torch.Tensor:
        origins = self.scene.env_origins
        values = []
        for side in ("right", "left"):
            body = self.wrist_body_ids[side]
            values.append(torch.cat((self.robot.data.body_pos_w[:, body] - origins,
                                     self.robot.data.body_quat_w[:, body]), dim=-1))
        return torch.stack(values, dim=1)

    def _wrist_velocities(self) -> torch.Tensor:
        return torch.stack([
            torch.cat((self.robot.data.body_lin_vel_w[:, self.wrist_body_ids[side]],
                       self.robot.data.body_ang_vel_w[:, self.wrist_body_ids[side]]), dim=-1)
            for side in ("right", "left")
        ], dim=1)

    def _landmarks(self) -> torch.Tensor:
        origins = self.scene.env_origins[:, None, :]
        sides = []
        for side in ("right", "left"):
            ids = self.landmark_body_ids[side]
            positions = self.robot.data.body_pos_w[:, ids] - origins
            quaternions = self.robot.data.body_quat_w[:, ids]
            local = self.landmark_local.expand(self.num_envs, -1, -1)
            sides.append(positions + quat_apply(quaternions, local))
        return torch.stack(sides, dim=1)

    def _tip_contacts(self) -> torch.Tensor:
        forces = []
        for sensor in self.contact_sensors[:10]:
            matrix = sensor.data.force_matrix_w
            if matrix is None:
                raise RuntimeError("missing fingertip force matrix: sensor unavailable, not no-contact")
            else:
                forces.append(matrix.reshape(self.num_envs, -1, 3).norm(dim=-1).amax(dim=-1))
        return torch.stack(forces, dim=1).reshape(self.num_envs, 2, 5) > 0.5

    def _object_table_contacts(self) -> torch.Tensor:
        return self._semantic_contacts()[2]

    def _semantic_contacts(self):
        return tuple(torch.as_tensor(v, device=self.device) for v in self.pair_contacts.read())

    def _hand_six(self):
        wrists = self._wrist_poses()
        palm = wrists[..., :3] + quat_apply(wrists[..., 3:], self.palm_local.expand(self.num_envs, -1, -1))
        return torch.cat((palm[:, :, None], self._landmarks()[:, :, ::2]), dim=2)

    def _hand_six_rot_velocity(self):
        result = []
        for si, side in enumerate(("right", "left")):
            ids = torch.cat((torch.tensor([self.wrist_body_ids[side]], device=self.device),
                             self.landmark_body_ids[side][::2]))
            quat = self.robot.data.body_quat_w[:, ids]
            angular = self.robot.data.body_link_ang_vel_w[:, ids]
            linear = self.robot.data.body_link_lin_vel_w[:, ids].clone()
            offset = quat_apply(quat[:, 0], self.palm_local[si].expand(self.num_envs, -1))
            linear[:, 0] += torch.cross(angular[:, 0], offset, dim=-1)
            result.append(torch.cat((quat, linear, angular), dim=-1))
        return torch.stack(result, dim=1)

    def _placement_weight(self, index):
        # Operation reference moves; canonical home remains fixed at both ends.
        start = self.pregrasp_end_step
        weight = torch.minimum(index.float() / start,
                               (self.reference_length - 1 - index).float() / 40)
        return weight.clamp(0, 1)

    def _target_hand_six(self):
        index = self._reference_index()
        palm = self.reference_palm[index].clone()
        palm[..., :2] += self.reference_xy_offset[:, None] * self._placement_weight(index)[:, None, None]
        return torch.cat((palm[:, :, None], self._targets()[2][:, :, ::2]), dim=2)

    def _targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = self._reference_index()
        wrist = self.reference_wrist[index].clone()
        objects = self.reference_object[index].clone()
        hands = self.reference_hand[index].clone()
        offset = self.reference_xy_offset
        weight = self._placement_weight(index)
        wrist[..., :2] += offset[:, None, :] * weight[:, None, None]
        objects[..., :2] += offset[:, None, :]
        hands[..., :2] += offset[:, None, None, :] * weight[:, None, None, None]
        return wrist, objects, hands

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions.copy_(actions.clamp(-1.0, 1.0))
        normalized = (self.actions + 1.0) * 0.5
        command = self.lower + normalized * (self.upper - self.lower)
        self.target_q.copy_(self.robot.data.joint_pos)
        self.target_q[:, self.joint_ids] = command
        active = self.certification_pending.clone()
        if active.any():
            alpha = self.certification_alpha[active, None, None]
            arm = self.certification_start_q[active] + alpha * (
                self.certification_goal_q[active] - self.certification_start_q[active])
            active_ids = torch.nonzero(active, as_tuple=False).flatten()
            for side_index, side in enumerate(("right", "left")):
                self.target_q[active_ids[:, None], self.arm_ids[side][None, :]] = arm[:, side_index]
        self.certification_pending.zero_()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.target_q)

    def _get_rewards(self) -> torch.Tensor:
        actual_hand = self._hand_six()
        actual_object = self._object_poses()
        _, target_object, _ = self._targets()
        target_hand = self._target_hand_six()
        hand_contacts, contacts, table_contact = self._semantic_contacts()
        reward, terms = object_mimic_bimanual(
            actual_hand_points=actual_hand, target_hand_points=target_hand,
            actual_object_pose=actual_object, target_object_pose=target_object,
            fingertip_contacts=contacts, object_table_contacts=table_contact,
            hand_object_contacts=hand_contacts,
            pregrasp_success=self._reward_pregrasp_success,
            initial_object_z=self.initial_object_z.expand(self.num_envs, -1),
            config=RewardConfig(),
        )
        self.extras["pregrasp_success_rate"] = self.pregrasp_success.float().mean()
        self.extras["hand_error_m_mean"] = terms["hand_error_m"].mean()
        self.extras["object_error_m_mean"] = terms["object_position_error_m"].mean()
        self.extras["tip_contact_rate"] = contacts.float().mean()
        if not bool(torch.isfinite(reward).all()):
            raise FloatingPointError("non-finite simulator-derived reward")
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Source rewards use the pre-transition phase on the boundary step.
        self._reward_pregrasp_success = self.pregrasp_success.clone()
        if self.external_evaluator_controls_termination:
            never = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return never, never
        if hasattr(self, "policy_arm_table_probe"):
            current = torch.as_tensor(self.policy_arm_table_probe.read(), device=self.device)
            self.policy_arm_table_contact_latch |= current
        deciding = (~self.pregrasp_decided) & (
            self.episode_length_buf >= self.pregrasp_end_step)
        if deciding.any():
            target_hand = self._target_hand_six()
            hand_error = torch.linalg.vector_norm(
                self._hand_six() - target_hand, dim=-1).mean(dim=-1)
            both_close = (hand_error < 0.05).all(dim=1)
            self.pregrasp_success[deciding] = both_close[deciding]
            self.pregrasp_decided[deciding] = True
            for env_id in torch.nonzero(deciding).flatten().cpu().tolist():
                self.boundary_trace.append({"env": env_id, "completed_control_steps": int(self.episode_length_buf[env_id]),
                    "reference_index": int(self._reference_index()[env_id]), "hand_six_error_m": hand_error[env_id].cpu().tolist(),
                    "both_success": bool(both_close[env_id])})
        contacts = self._semantic_contacts()[0].any(dim=1)
        before_gate = ~self.pregrasp_success
        failed_pregrasp = before_gate & (
            (self.episode_length_buf >= self.pregrasp_end_step) | contacts)
        actual_object = self._object_poses()
        _, target_object, _ = self._targets()
        object_error = torch.linalg.vector_norm(
            actual_object[..., :3] - target_object[..., :3], dim=-1)
        failed_tracking = self.pregrasp_success & (object_error >= 0.25).any(dim=1)
        reference_done = self.pregrasp_success & (
            self.episode_length_buf >= self.reference_length - 1)
        terminated = failed_pregrasp | failed_tracking | reference_done
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_observations(self) -> dict[str, torch.Tensor]:
        q = self.robot.data.joint_pos[:, self.joint_ids]
        qd = self.robot.data.joint_vel[:, self.joint_ids]
        wrist = self._wrist_poses()
        wrist_velocity = self._wrist_velocities()
        objects = self._object_poses()
        object_velocity = self._object_velocities()
        target_wrist, target_objects, target_hand = self._targets()
        relative_actual = _relative_pose(wrist, objects)
        relative_target = _relative_pose(target_wrist, target_objects)
        phase = self._reference_index().float() / max(self.reference_length, 1)
        phase = phase[:, None] * torch.tensor([1., 4., 6., 8.], device=self.device)
        phase_features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        hand_six = self._hand_six()
        final_goal = self.final_object_goal.expand(self.num_envs, -1, -1).clone()
        final_goal[..., :2] += self.reference_xy_offset[:, None]
        future = []
        for offset in (1, 5, 10):
            pose = self.reference_object[(self._reference_index() + offset).clamp(max=self.reference_length - 1)].clone()
            pose[..., :2] += self.reference_xy_offset[:, None]
            future.append(torch.cat((pose[..., 3:], pose[..., :3]), dim=-1).flatten(1))
        observation = torch.cat((
            q, qd,
            torch.cat((wrist, wrist_velocity), dim=-1).flatten(1),
            torch.cat((objects, object_velocity), dim=-1).flatten(1),
            target_wrist.flatten(1), target_objects.flatten(1), target_hand.flatten(1),
            relative_actual.flatten(1), relative_target.flatten(1), hand_six.flatten(1),
            self._hand_six_rot_velocity().flatten(1),
            (hand_six - objects[:, :, None, :3]).flatten(1), final_goal.flatten(1),
            (hand_six[:, :, 0] - final_goal).flatten(1), (objects[..., :3] - final_goal).flatten(1),
            *future, phase_features,
        ), dim=-1)
        if observation.shape[1] != OBSERVATION_DIM:
            raise RuntimeError(f"observation width {observation.shape[1]} != {OBSERVATION_DIM}")
        if not bool(torch.isfinite(observation).all()):
            raise FloatingPointError("non-finite simulator-derived observation")
        return {"policy": observation}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return
        if hasattr(self, "terminal_observation_by_env") and hasattr(self, "reset_buf"):
            terminal_ids = env_ids[self.reset_buf[env_ids]]
            if len(terminal_ids) > 0:
                terminal = self._get_observations()["policy"][terminal_ids].detach().clone()
                errors = torch.linalg.vector_norm(self._hand_six() - self._target_hand_six(), dim=-1).mean(dim=-1)
                for row, env_id in enumerate(terminal_ids.cpu().tolist()):
                    self.terminal_observation_by_env[int(env_id)] = terminal[row]
                    self.terminal_stats_by_env[int(env_id)] = {
                        "pregrasp_success": bool(self.pregrasp_success[env_id]),
                        "pregrasp_decided": bool(self.pregrasp_decided[env_id]), "curriculum_stage": self.stage,
                        "hand_six_error_m": errors[env_id].cpu().tolist(),
                        "sample_id": int(self.episode_sample_id[env_id]),
                        "failure_or_end": ("pregrasp_threshold" if bool(self.pregrasp_decided[env_id]) else "pregrasp_early_hand_contact")
                            if not bool(self.pregrasp_success[env_id]) else "tracking_or_reference_end"}
                    if hasattr(self, "policy_arm_table_contact_latch"):
                        self.terminal_stats_by_env[int(env_id)].update(
                            any_arm_table_contact=bool(self.policy_arm_table_contact_latch[env_id]),
                            arm_table_active_at_terminal=self.policy_arm_table_probe.latest_by_env[int(env_id)])
                    if getattr(self, "policy_capture_terminal", False) and int(env_id) < 2:
                        self.terminal_stats_by_env[int(env_id)]["terminal_detail"] = {
                            "q_rad": self.robot.data.joint_pos[env_id, self.joint_ids].cpu().tolist(),
                            "joint_position_target_rad": self.target_q[env_id, self.joint_ids].cpu().tolist(),
                            "actual_hand_six_m": self._hand_six()[env_id].cpu().tolist(),
                            "target_hand_six_m": self._target_hand_six()[env_id].cpu().tolist()}
        DirectRLEnv._reset_idx(self, env_ids)
        q = self.robot.data.default_joint_pos[env_ids].clone()
        qd = torch.zeros_like(q)
        q[:, self.joint_ids] = self.reset_q
        qd[:, self.joint_ids] = self.reset_qd
        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        origins = self.scene.env_origins[env_ids]
        for env_id in env_ids.cpu().tolist():
            sample = sample_placement(self.stage, self.env_rng[env_id])
            self.reference_xy_offset[env_id] = torch.tensor(
                sample.translation_xy_m, dtype=torch.float32, device=self.device)
            table_rows = {}
            # A geometric warning only, not a rejection sampler or a new gate.
            # Table cuboid is the registered x/y footprint and z=[0.83,0.87].
            weight = np.minimum(np.arange(self.reference_length) / self.pregrasp_end_step,
                                (self.reference_length - 1 - np.arange(self.reference_length)) / 40).clip(0, 1)
            for side in ("right", "left"):
                points = self.reference_np[f"{side}_hand_target_m"].copy()
                points[..., :2] += np.asarray(sample.translation_xy_m)[None, None] * weight[:, None, None]
                inside = ((np.abs(points[..., 0]) <= .6096) & (np.abs(points[..., 1]) <= .9144)
                          & (points[..., 2] >= .83) & (points[..., 2] <= .87))
                table_rows[side] = int(inside.any(axis=-1).sum())
            self.episode_sample_id[env_id] = len(self.reset_records)
            self.reset_records.append({"sample_id": len(self.reset_records), "env_id": env_id,
                "stage": self.stage, "translation_xy_m": list(sample.translation_xy_m),
                "target_table_volume_warning_frames": table_rows,
                "resampled_or_replaced": False, "global_reachability": "not established"})
        for object_index, object_asset in ((0, self.bottle), (1, self.cup)):
            pose = self.reset_object_pose[object_index].repeat(len(env_ids), 1)
            pose[:, :3] += origins
            pose[:, :2] += self.reference_xy_offset[env_ids]
            object_asset.write_root_pose_to_sim(pose, env_ids=env_ids)
            object_asset.write_root_velocity_to_sim(
                torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)
        self.pair_contacts.clear(env_ids.cpu().tolist())
        self.actions[env_ids] = 0.0
        self.pregrasp_success[env_ids] = False
        self.pregrasp_decided[env_ids] = False
        self.certification_alpha[env_ids] = 0.0
        self.certification_active[env_ids] = False
        self.certification_pending[env_ids] = False
        if hasattr(self, "policy_arm_table_contact_latch"):
            self.policy_arm_table_contact_latch[env_ids] = False

    def set_curriculum_stage(self, stage: int) -> None:
        if int(stage) not in (0, 1, 2):
            raise ValueError("curriculum stage must be 0, 1, or 2")
        self.stage = int(stage)

    def runtime_probe(self) -> dict:
        """Return measured canonical quantities and hash-bound runtime names."""
        wrists = self._wrist_poses()[0].detach().cpu().numpy()
        landmarks = self._landmarks()[0].detach().cpu().numpy()
        arm_center = np.r_[
            (self.robot.data.body_pos_w[0, self.arm_center_body_id]
             - self.scene.env_origins[0]).detach().cpu().numpy(),
            self.robot.data.body_quat_w[0, self.arm_center_body_id].detach().cpu().numpy(),
        ]
        runtime_names = list(self.robot.joint_names)
        order = [runtime_names[index] for index in self.joint_ids.cpu().tolist()]
        return {
            "status": "PASS_RUNTIME_PROBE",
            "joint_names_in_order": order,
            "runtime_joint_names": runtime_names,
            "runtime_body_names": list(self.robot.body_names),
            "canonical_home": {
                "joint_q_rad": self.robot.data.joint_pos[0, self.joint_ids].detach().cpu().tolist(),
                "arm_center_pose_env_wxyz": arm_center.tolist(),
                "right_wrist_pose_env_wxyz": wrists[0].tolist(),
                "left_wrist_pose_env_wxyz": wrists[1].tolist(),
                "right_landmarks_env_m": landmarks[0].tolist(),
                "left_landmarks_env_m": landmarks[1].tolist(),
            },
            "inputs": {
                "reference_sha256": sha256(self.reference_path),
                "robot_usd_sha256": sha256(self.bundle_root / "world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd"),
                "robot_urdf_sha256": sha256(Path(self.cfg.robot_urdf)),
            },
        }

    def _ensure_arm_ik(self) -> None:
        if self._arm_ik is not None:
            return
        arm_center = np.r_[
            (self.robot.data.body_pos_w[0, self.arm_center_body_id]
             - self.scene.env_origins[0]).detach().cpu().numpy(),
            self.robot.data.body_quat_w[0, self.arm_center_body_id].detach().cpu().numpy(),
        ]
        urdf = UrdfKinematics(self.cfg.robot_urdf)
        limits = np.column_stack((self.lower.cpu().numpy(), self.upper.cpu().numpy()))
        self._arm_ik = {
            "right": ArmIK(urdf, "right", "arm_center", arm_center,
                           self.joint_names[:7], self.reset_q[:7].cpu().numpy(), limits[:7]),
            "left": ArmIK(urdf, "left", "arm_center", arm_center,
                          self.joint_names[7:14], self.reset_q[7:14].cpu().numpy(), limits[7:14]),
        }

    def apply_certification_offset(self, alpha: float, env_id: int = 0) -> None:
        """Schedule evaluator-only common +Z lift from latched measured wrists."""
        slot = int(env_id)
        if slot < 0 or slot >= self.num_envs:
            raise IndexError("certification environment slot out of range")
        value = float(np.clip(alpha, 0.0, 1.0))
        if value == 0.:
            self.certification_active[slot] = False
            self.certification_pending[slot] = False
            return
        if value > 0.0 and not bool(self.certification_active[slot]):
            self._ensure_arm_ik()
            actual_q = self.robot.data.joint_pos[slot]
            wrists = self._wrist_poses()[slot].detach().cpu().numpy()
            self.certification_start_q[slot, 0] = actual_q[self.arm_ids["right"]]
            self.certification_start_q[slot, 1] = actual_q[self.arm_ids["left"]]
            for side_index, side in enumerate(("right", "left")):
                target = wrists[side_index].copy()
                target[2] += 0.015
                solved = self._arm_ik[side].solve(
                    target, self.certification_start_q[slot, side_index].cpu().numpy(),
                    iterations=250, position_tolerance_m=0.005,
                    rotation_tolerance_rad=0.05)
                self.certification_goal_q[slot, side_index] = torch.tensor(
                    solved["q"], dtype=torch.float32, device=self.device)
                self.certification_trace.append({
                    "event": "latch", "side": side, "env_id": slot,
                    "measured_wrist_pose_env_wxyz": wrists[side_index].tolist(),
                    "ik_position_error_m": solved["position_error_m"],
                    "ik_rotation_error_rad": solved["rotation_error_rad"],
                    "ik_ok": solved["ok"],
                })
            self.certification_active[slot] = True
        self.certification_alpha[slot] = value
        self.certification_pending[slot] = True
        self.certification_trace.append({"event": "alpha", "alpha": value, "env_id": slot})

    def evaluator_infos(self):
        # One physical-state readback per batch, not N repeated GPU snapshots.
        objects = self._object_poses().detach().cpu().numpy()
        wrists = self._wrist_poses().detach().cpu().numpy()
        q = self.robot.data.joint_pos[:, self.joint_ids].detach().cpu().numpy()
        contacts = self._tip_contacts().detach().cpu().numpy()
        return [{"object_0_pose": objects[i, 1], "object_1_pose": objects[i, 0],
            "arm_q_right": q[i, :7], "arm_q_left": q[i, 7:14],
            "pads3": bool((contacts[i, 0].sum() >= 3) and (contacts[i, 1].sum() >= 3)),
            "wrist_right": wrists[i, 0, :3], "wrist_left": wrists[i, 1, :3]}
            for i in range(self.num_envs)]

    def evaluator_info(self, env_id: int = 0) -> dict[str, np.ndarray | bool]:
        return self.evaluator_infos()[env_id]

    def reset_state(self) -> dict:
        # Explicit PhysX readback, not cfg constants or the write-side caches.
        poses = []
        for asset in (self.bottle, self.cup):
            transform = asset.root_physx_view.get_transforms().clone()
            poses.append(torch.cat((transform[:, :3] - self.scene.env_origins,
                                    transform[:, 6:7], transform[:, 3:6]), dim=-1))
        objects = torch.stack(poses, dim=1).detach().cpu().numpy()
        joints = self.robot.root_physx_view.get_dof_positions()[:, self.joint_ids].detach().cpu().numpy()
        return {
            "readback_backend": "PhysX get_dof_positions/get_transforms; xyzw converted to wxyz",
            "regime": "canonical_t0", "gates_preset": [],
            "warmup_clamp_steps": 0,
            "joint_names": list(self.joint_names),
            "joint_q_rad": joints.tolist(), "state_joint_sha256": ndarray_sha256(joints),
            "object_0_pose_wxyz": objects[:, 1].tolist(),
            "object_1_pose_wxyz": objects[:, 0].tolist(),
            "state_object_sha256": ndarray_sha256(objects),
            "placement_xy_m": self.reference_xy_offset.cpu().tolist(), "stage": self.stage,
            "reference_sha256": sha256(self.reference_path),
            "reference_joint_q_content_sha256": ndarray_sha256(self.reference_np["joint_q_rad"]),
        }
