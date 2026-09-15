"""Policy contract for direct bimanual Sharpa joint control.

The centralized policy commands all physical joints in simulator order: right
arm 7, left arm 7, right hand 22, left hand 22.  PCA and FABRICS are outside
this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


ARM_DOF_DIM = 7
HAND_DOF_DIM = 22
SIDE_JOINT_DIM = ARM_DOF_DIM + HAND_DOF_DIM
SIDE_ACTION_DIM = SIDE_JOINT_DIM
NUM_FINGERTIPS = 5
NUM_SIDES = 2
BIMANUAL_JOINT_DIM = NUM_SIDES * SIDE_JOINT_DIM
BIMANUAL_ACTION_DIM = NUM_SIDES * SIDE_ACTION_DIM


class ControlledSide(str, Enum):
    """One side of the synchronized dual-arm embodiment."""

    LEFT = "left"
    RIGHT = "right"

    @classmethod
    def parse(cls, value: str) -> "ControlledSide":
        try:
            return cls(value.lower())
        except ValueError as exc:
            raise ValueError(f"side must be 'left' or 'right', got {value!r}") from exc

    @property
    def arm_prefix(self) -> str:
        return "L" if self is ControlledSide.LEFT else "R"


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def arm_joint_names(side: ControlledSide | str) -> tuple[str, ...]:
    side = side if isinstance(side, ControlledSide) else ControlledSide.parse(side)
    return tuple(f"{side.arm_prefix}_arm_j{i}" for i in range(1, 8))


def hand_joint_names(side: ControlledSide | str) -> tuple[str, ...]:
    side = side if isinstance(side, ControlledSide) else ControlledSide.parse(side)
    p = side.value
    return (
        f"{p}_thumb_CMC_FE",
        f"{p}_thumb_CMC_AA",
        f"{p}_thumb_MCP_FE",
        f"{p}_thumb_MCP_AA",
        f"{p}_thumb_IP",
        f"{p}_index_MCP_FE",
        f"{p}_index_MCP_AA",
        f"{p}_index_PIP",
        f"{p}_index_DIP",
        f"{p}_middle_MCP_FE",
        f"{p}_middle_MCP_AA",
        f"{p}_middle_PIP",
        f"{p}_middle_DIP",
        f"{p}_ring_MCP_FE",
        f"{p}_ring_MCP_AA",
        f"{p}_ring_PIP",
        f"{p}_ring_DIP",
        f"{p}_pinky_CMC",
        f"{p}_pinky_MCP_FE",
        f"{p}_pinky_MCP_AA",
        f"{p}_pinky_PIP",
        f"{p}_pinky_DIP",
    )


def joint_names_for_side(side: ControlledSide | str) -> tuple[str, ...]:
    """Per-side action/command joint order."""

    return arm_joint_names(side) + hand_joint_names(side)


def simulation_joint_names() -> tuple[str, ...]:
    """Frozen 58-DoF order used by the Pour17 simulator and reset manifest."""

    return (
        arm_joint_names(ControlledSide.RIGHT)
        + arm_joint_names(ControlledSide.LEFT)
        + hand_joint_names(ControlledSide.RIGHT)
        + hand_joint_names(ControlledSide.LEFT)
    )


def fingertip_body_names(side: ControlledSide | str) -> tuple[str, ...]:
    side = side if isinstance(side, ControlledSide) else ControlledSide.parse(side)
    return tuple(f"{side.value}_{finger}_elastomer" for finger in FINGERS)


@dataclass(frozen=True)
class H2S2RObservationLayout:
    """Observation widths, preserving the official H2S2R field order.

    H2S2R uses four Allegro fingertips and 23 c-space DoFs (144 values). The
    embodiment adapter uses five Sharpa fingertips and 29 c-space DoFs, giving
    171 values without adding project-specific priors.
    """

    q: int = SIDE_JOINT_DIM
    qd: int = SIDE_JOINT_DIM
    fingertip_positions: int = NUM_FINGERTIPS * 3
    palm_points: int = 4 * 3  # palm origin and +x/+y/+z helper points
    object_pose: int = 7
    goal_object_pose: int = 7
    previous_object_pose: int = 7
    previous_previous_object_pose: int = 7
    command_q: int = SIDE_JOINT_DIM
    command_qd: int = SIDE_JOINT_DIM

    @property
    def total(self) -> int:
        return sum(self.__dict__.values())


SIDE_OBSERVATION_DIM = H2S2RObservationLayout().total
BIMANUAL_OBSERVATION_DIM = NUM_SIDES * SIDE_OBSERVATION_DIM

# Public task-level aliases.  Side-level components use the explicit SIDE_ names.
ACTION_DIM = BIMANUAL_ACTION_DIM
OBSERVATION_DIM = BIMANUAL_OBSERVATION_DIM

assert SIDE_ACTION_DIM == 29
assert ACTION_DIM == 58
assert SIDE_JOINT_DIM == 29
assert BIMANUAL_JOINT_DIM == 58
assert SIDE_OBSERVATION_DIM == 171
assert OBSERVATION_DIM == 342
