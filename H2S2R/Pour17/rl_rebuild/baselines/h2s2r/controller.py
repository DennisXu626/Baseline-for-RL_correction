"""Stateful H2S2R FABRICS controller wrapper used by the Isaac Lab adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contract import (
    BIMANUAL_ACTION_DIM,
    FABRIC_DOF_DIM,
    SIDE_ACTION_DIM,
    ControlledSide,
    hand_joint_names,
)
from .fabric_params import fabric_params_for_side
from .synergy import HandSynergy, analytic_finger_synergy, load_synergy


PALM_MINIMUM = (0.0, -0.7, 0.0, -np.pi, -np.pi, -np.pi)
PALM_MAXIMUM = (1.0, 0.7, 1.0, np.pi, np.pi, np.pi)


def rescale_from_policy(
    values: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor
) -> torch.Tensor:
    return minimum + 0.5 * (values.clamp(-1.0, 1.0) + 1.0) * (maximum - minimum)


def rescale_from_policy_centered(
    values: torch.Tensor,
    minimum: torch.Tensor,
    center: torch.Tensor,
    maximum: torch.Tensor,
) -> torch.Tensor:
    """Map -1/0/+1 to the audited minimum/near-grasp/maximum targets.

    This keeps H2S2R's absolute target action and the existing physical bounds,
    while calibrating normalized zero independently for each embodiment side.
    """

    clipped = values.clamp(-1.0, 1.0)
    lower = center + clipped * (center - minimum)
    upper = center + clipped * (maximum - center)
    return torch.where(clipped <= 0.0, lower, upper)


class H2S2RFabricController:
    """Run the official displacement-integrated fabric behind an 11-D action."""

    def __init__(
        self,
        *,
        side: ControlledSide | str,
        num_envs: int,
        device: str,
        timestep: float,
        urdf_path: str | Path,
        initial_q: torch.Tensor,
        initial_qd: torch.Tensor | None = None,
        synergy_path: str | Path | None = None,
        world_dict: dict[str, Any] | None = None,
        graph_capturable: bool = True,
    ) -> None:
        side = side if isinstance(side, ControlledSide) else ControlledSide.parse(side)
        if initial_q.shape != (num_envs, FABRIC_DOF_DIM):
            raise ValueError(f"initial_q must be ({num_envs}, 29), got {initial_q.shape}")
        if initial_qd is not None and initial_qd.shape != initial_q.shape:
            raise ValueError("initial_qd must have the same shape as initial_q")

        # Imports are intentionally delayed: pure contract/reward/evaluator tests do
        # not require the GPU-only FABRICS dependency.
        from fabrics_sim.integrator.integrators import DisplacementIntegrator
        from fabrics_sim.utils.utils import capture_fabric
        from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

        from .vega_sharpa_fabric import VegaSharpaPoseFabric

        self.side = side
        self.num_envs = num_envs
        self.device = device
        self.timestep = float(timestep)
        self.synergy: HandSynergy = (
            load_synergy(synergy_path)
            if synergy_path is not None
            else analytic_finger_synergy(hand_joint_names(side))
        )
        self.synergy.validate()

        self.q = initial_q.to(device=device, dtype=torch.float32).clone()
        self.qd = (
            torch.zeros_like(self.q)
            if initial_qd is None
            else initial_qd.to(device=device, dtype=torch.float32).clone()
        )
        self.qdd = torch.zeros_like(self.q)
        self.palm_minimum = torch.tensor(PALM_MINIMUM, device=device)
        self.palm_maximum = torch.tensor(PALM_MAXIMUM, device=device)

        self.fabric = VegaSharpaPoseFabric(
            batch_size=num_envs,
            device=device,
            timestep=self.timestep,
            urdf_path=urdf_path,
            default_config=self.q,
            synergy=self.synergy,
            fabric_params=fabric_params_for_side(side),
            graph_capturable=graph_capturable,
        )
        self.hand_minimum = self.fabric.synergy_minimum
        self.hand_maximum = self.fabric.synergy_maximum
        self.palm_target = torch.zeros(num_envs, 6, device=device)
        self.hand_target = torch.zeros(num_envs, 5, device=device)
        self.palm_policy_center: torch.Tensor | None = None
        self.hand_policy_center: torch.Tensor | None = None

        palm_pose = self.fabric.get_palm_pose(self.q, "euler_zyx")
        self.palm_target.copy_(palm_pose)
        hand_pose = self.q[:, 7:] @ self.fabric.pca_matrix.T
        self.hand_target.copy_(hand_pose)

        self.world = WorldMeshesModel(
            batch_size=num_envs,
            max_objects_per_env=max(1, len(world_dict or {})),
            device=device,
            world_dict=world_dict,
        )
        self.object_ids, self.object_indicator = self.world.get_object_ids()
        self.integrator = DisplacementIntegrator(self.fabric)
        inputs = [
            self.hand_target,
            self.palm_target,
            "euler_zyx",
            self.q.detach(),
            self.qd.detach(),
            self.object_ids,
            self.object_indicator,
        ]
        self.graph, self._q_new, self._qd_new, self._qdd_new = capture_fabric(
            fabric=self.fabric,
            q=self.q,
            qd=self.qd,
            qdd=self.qdd,
            timestep=self.timestep,
            fabric_integrator=self.integrator,
            inputs=inputs,
            device=device,
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.fabric.get_joint_names())

    def reset(
        self,
        env_ids: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor | None = None,
    ) -> None:
        self.q[env_ids] = q
        self.qd[env_ids] = 0.0 if qd is None else qd
        self.qdd[env_ids] = 0.0
        self.fabric.default_config[env_ids] = q
        # FABRICS' Warp palm task-map is allocated at construction time for
        # ``num_envs`` and therefore requires a full-batch FK input even when
        # Isaac Lab resets only a subset of environments.
        palm_pose = self.fabric.get_palm_pose(self.q, "euler_zyx")
        if palm_pose.shape != self.palm_target.shape:
            raise RuntimeError(
                "FABRICS palm FK must return the controller's full batch: "
                f"expected {tuple(self.palm_target.shape)}, got {tuple(palm_pose.shape)}"
            )
        self.palm_target[env_ids] = palm_pose[env_ids]
        self.hand_target[env_ids] = q[:, 7:] @ self.fabric.pca_matrix.T

    def set_targets(
        self,
        actions: torch.Tensor,
        *,
        palm_position_offset: torch.Tensor | None = None,
    ) -> None:
        """Decode and persist one side's targets without advancing FABRICS.

        ``palm_position_offset`` is expressed in this controller's kinematic
        root frame.  It is applied after decoding the policy action so the G2
        evaluator override cannot be overwritten by the policy command.
        """

        if actions.shape != (self.num_envs, SIDE_ACTION_DIM):
            raise ValueError(
                f"actions must be ({self.num_envs}, {SIDE_ACTION_DIM}), "
                f"got {actions.shape}"
            )
        palm_target = (
            rescale_from_policy(actions[:, :6], self.palm_minimum, self.palm_maximum)
            if self.palm_policy_center is None
            else rescale_from_policy_centered(
                actions[:, :6],
                self.palm_minimum,
                self.palm_policy_center,
                self.palm_maximum,
            )
        )
        self.palm_target.copy_(palm_target)
        if palm_position_offset is not None:
            if palm_position_offset.shape != (self.num_envs, 3):
                raise ValueError(
                    "palm_position_offset must be "
                    f"({self.num_envs}, 3), got {palm_position_offset.shape}"
                )
            self.palm_target[:, :3].add_(palm_position_offset)
        hand_target = (
            rescale_from_policy(actions[:, 6:], self.hand_minimum, self.hand_maximum)
            if self.hand_policy_center is None
            else rescale_from_policy_centered(
                actions[:, 6:],
                self.hand_minimum,
                self.hand_policy_center,
                self.hand_maximum,
            )
        )
        self.hand_target.copy_(hand_target)

    def advance(self) -> torch.Tensor:
        """Advance FABRICS once using the currently persisted targets."""

        self.graph.replay()
        self.q.copy_(self._q_new)
        self.qd.copy_(self._qd_new)
        self.qdd.copy_(self._qdd_new)
        return self.q

    def step(
        self,
        actions: torch.Tensor,
        *,
        palm_position_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set one side's targets and advance FABRICS once."""

        self.set_targets(actions, palm_position_offset=palm_position_offset)
        return self.advance()


class BimanualH2S2RFabricController:
    """Two independent H2S2R fabrics behind one synchronized 22-D action."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str,
        timestep: float,
        right_timestep: float,
        right_urdf_path: str | Path,
        left_urdf_path: str | Path,
        initial_q_right: torch.Tensor,
        initial_q_left: torch.Tensor,
        initial_qd_right: torch.Tensor | None = None,
        initial_qd_left: torch.Tensor | None = None,
        right_synergy_path: str | Path | None = None,
        left_synergy_path: str | Path | None = None,
        world_dict: dict[str, Any] | None = None,
        graph_capturable: bool = True,
    ) -> None:
        common = dict(
            num_envs=num_envs,
            device=device,
            world_dict=world_dict,
            graph_capturable=graph_capturable,
        )
        self.right = H2S2RFabricController(
            side=ControlledSide.RIGHT,
            timestep=right_timestep,
            urdf_path=right_urdf_path,
            initial_q=initial_q_right,
            initial_qd=initial_qd_right,
            synergy_path=right_synergy_path,
            **common,
        )
        self.left = H2S2RFabricController(
            side=ControlledSide.LEFT,
            timestep=timestep,
            urdf_path=left_urdf_path,
            initial_q=initial_q_left,
            initial_qd=initial_qd_left,
            synergy_path=left_synergy_path,
            **common,
        )
        self.num_envs = num_envs

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        q_right: torch.Tensor,
        q_left: torch.Tensor,
        qd_right: torch.Tensor | None = None,
        qd_left: torch.Tensor | None = None,
    ) -> None:
        self.right.reset(env_ids, q_right, qd_right)
        self.left.reset(env_ids, q_left, qd_left)

    def step(
        self,
        actions: torch.Tensor,
        *,
        palm_position_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return right and left 29-DoF commands in fabric joint order."""

        if actions.shape != (self.num_envs, BIMANUAL_ACTION_DIM):
            raise ValueError(
                f"actions must be ({self.num_envs}, {BIMANUAL_ACTION_DIM}), "
                f"got {actions.shape}"
            )
        right_offset = left_offset = None
        if palm_position_offsets is not None:
            if palm_position_offsets.shape != (self.num_envs, 2, 3):
                raise ValueError(
                    "palm_position_offsets must be "
                    f"({self.num_envs}, 2, 3), got {palm_position_offsets.shape}"
                )
            right_offset = palm_position_offsets[:, 0]
            left_offset = palm_position_offsets[:, 1]
        q_right = self.right.step(
            actions[:, :SIDE_ACTION_DIM], palm_position_offset=right_offset
        )
        q_left = self.left.step(
            actions[:, SIDE_ACTION_DIM:], palm_position_offset=left_offset
        )
        return q_right, q_left
