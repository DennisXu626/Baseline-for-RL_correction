"""Isolated Sharpa-right joint-limit assembly for the Stage-B diagnostic.

The right arm retains the inherited ten-degree numerical limits.  Only the 22
right-hand joints use their URDF limits directly.  Left-side construction is
delegated unchanged to the verified adapter/base implementation.
"""

from __future__ import annotations

import numpy as np

from fabrics_sim.fabric_terms.joint_limit_repulsion import JointLimitRepulsion
from fabrics_sim.taskmaps.lower_joint_limit import LowerJointLimitMap
from fabrics_sim.taskmaps.upper_joint_limit import UpperJointLimitMap

from rl_rebuild.baselines.h2s2r.contract import ControlledSide, joint_names_for_side
from rl_rebuild.baselines.h2s2r.vega_sharpa_fabric import (
    VegaSharpaPoseFabric as _VerifiedVegaSharpaPoseFabric,
)


class SharpaRightZeroMarginPoseFabric(_VerifiedVegaSharpaPoseFabric):
    """Preserve assembly semantics while removing hand-only limit contraction."""

    def add_joint_limit_repulsion(self) -> None:
        names = tuple(self.joint_names)
        right_names = joint_names_for_side(ControlledSide.RIGHT)
        left_names = joint_names_for_side(ControlledSide.LEFT)
        if names == left_names:
            super().add_joint_limit_repulsion()
            return
        if names != right_names:
            raise ValueError(f"unexpected FABRICS joint order: {names}")

        revolute = [
            joint for joint in self.urdfpy_robot.joints if joint.joint_type == "revolute"
        ]
        if tuple(joint.name for joint in revolute) != right_names:
            raise ValueError("right URDF revolute order differs from the adapter contract")

        margin = np.deg2rad(10.0)
        lower = []
        upper = []
        for index, joint in enumerate(revolute):
            if index < 7:
                lower.append(joint.limit.lower + margin)
                upper.append(joint.limit.upper - margin)
            else:
                lower.append(joint.limit.lower)
                upper.append(joint.limit.upper)

        self.add_taskmap(
            "upper_joint_limit",
            UpperJointLimitMap(upper, self.batch_size, self.device),
            graph_capturable=self.graph_capturable,
        )
        self.add_fabric(
            "upper_joint_limit",
            "joint_limit_repulsion",
            JointLimitRepulsion(
                True,
                self.fabric_params["joint_limit_repulsion"],
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )
        self.add_taskmap(
            "lower_joint_limit",
            LowerJointLimitMap(lower, self.batch_size, self.device),
            graph_capturable=self.graph_capturable,
        )
        self.add_fabric(
            "lower_joint_limit",
            "joint_limit_repulsion",
            JointLimitRepulsion(
                True,
                self.fabric_params["joint_limit_repulsion"],
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )


def install_right_only_zero_margin_patch() -> None:
    """Install the isolated class before controller construction in this process."""

    import rl_rebuild.baselines.h2s2r.vega_sharpa_fabric as runtime_module

    runtime_module.VegaSharpaPoseFabric = SharpaRightZeroMarginPoseFabric

