# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is a non-commercial NVIDIA-processor-only derivative adapter of
# NVlabs/FABRICS' KukaAllegroPoseFabric. The upstream NVIDIA License is retained
# in THIRD_PARTY_NOTICES.md and the external FABRICS checkout remains unchanged.
"""Vega + Sharpa embodiment adapter for the public H2S2R FABRICS controller."""

from __future__ import annotations

from pathlib import Path

import torch
from urdfpy import URDF

from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.taskmaps.linear_taskmap import LinearMap
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap

from .contract import FABRIC_DOF_DIM
from .synergy import HandSynergy
from .tools.generate_single_side_urdf import PALM_FRAME, PALM_HELPERS


class VegaSharpaPoseFabric(KukaAllegroPoseFabric):
    """Keep H2S2R's pose-fabric method while replacing its robot kinematics."""

    def __init__(
        self,
        *,
        batch_size: int,
        device: str,
        timestep: float,
        urdf_path: str | Path,
        default_config: torch.Tensor,
        synergy: HandSynergy,
        fabric_params: dict,
        graph_capturable: bool = True,
    ) -> None:
        synergy.validate()
        BaseFabric.__init__(
            self,
            device=device,
            batch_size=batch_size,
            timestep=timestep,
            fabric_params_filename="kuka_allegro_pose_params.yaml",
            fabric_params=fabric_params,
            graph_capturable=graph_capturable,
        )
        self.urdf_path = str(Path(urdf_path).resolve())
        self._load_kinematic_robot()

        default_config = default_config.to(device=device, dtype=torch.float32)
        if default_config.shape == (FABRIC_DOF_DIM,):
            default_config = default_config.unsqueeze(0).repeat(batch_size, 1)
        if default_config.shape != (batch_size, FABRIC_DOF_DIM):
            raise ValueError(
                f"default_config must be (29,) or ({batch_size}, 29), "
                f"got {tuple(default_config.shape)}"
            )
        self.default_config = default_config.clone()
        self._synergy_matrix = torch.as_tensor(
            synergy.matrix, device=device, dtype=torch.float32
        )
        self._pca_matrix = self._synergy_matrix.clone()
        self.synergy_center_projection = torch.as_tensor(
            synergy.matrix @ synergy.center, device=device, dtype=torch.float32
        )
        self.synergy_minimum = (
            torch.as_tensor(synergy.minimum, device=device, dtype=torch.float32)
            + self.synergy_center_projection
        )
        self.synergy_maximum = (
            torch.as_tensor(synergy.maximum, device=device, dtype=torch.float32)
            + self.synergy_center_projection
        )

        self.construct_fabric()
        self._palm_pose_target = torch.zeros(batch_size, 12, device=device)
        self._native_palm_pose_target = None

    def _load_kinematic_robot(self) -> None:
        """Load joint names/limits without adding the URDF to the Isaac scene."""

        self.urdfpy_robot = URDF.load(self.urdf_path)
        self.joint_names = [
            joint.name
            for joint in self.urdfpy_robot.joints
            if joint.joint_type == "revolute"
        ]
        self._num_joints = len(self.joint_names)
        if self._num_joints != FABRIC_DOF_DIM:
            raise ValueError(
                f"FABRICS URDF must contain 29 revolute joints, got {self._num_joints}"
            )
        # BaseFabric's remaining allocations do not depend on its optional Warp
        # simulation model. Isaac Lab owns physics; FABRICS owns kinematics/control.
        BaseFabric.load_robot(self)

    def add_hand_fabric(self) -> None:
        matrix = torch.cat(
            [
                torch.zeros(5, 7, device=self.device),
                self._synergy_matrix,
            ],
            dim=1,
        )
        taskmap_name = "pca_hand"
        self.add_taskmap(
            taskmap_name,
            LinearMap(matrix, self.device),
            graph_capturable=self.graph_capturable,
        )
        self.add_fabric(
            taskmap_name,
            "hand_attractor",
            Attractor(
                True,
                self.fabric_params["hand_attractor"],
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )

    def add_palm_points_attractor(self) -> None:
        frames = [
            PALM_FRAME,
            "h2s2r_palm_x",
            "h2s2r_palm_x_neg",
            "h2s2r_palm_y",
            "h2s2r_palm_y_neg",
            "h2s2r_palm_z",
            "h2s2r_palm_z_neg",
        ]
        if set(frames[1:]) != set(PALM_HELPERS):
            raise AssertionError("palm helper-frame contract changed")
        taskmap_name = "palm"
        self.add_taskmap(
            taskmap_name,
            RobotFrameOriginsTaskMap(
                self.urdf_path, frames, self.batch_size, self.device
            ),
            graph_capturable=self.graph_capturable,
        )
        self.add_fabric(
            taskmap_name,
            "palm_attractor",
            Attractor(
                True,
                self.fabric_params["palm_attractor"],
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )

