"""Human2Sim2Robot adapter for the Vega + dual-Sharpa platform."""

from .contract import (
    ACTION_DIM,
    BIMANUAL_ACTION_DIM,
    BIMANUAL_JOINT_DIM,
    BIMANUAL_OBSERVATION_DIM,
    OBSERVATION_DIM,
    SIDE_ACTION_DIM,
    SIDE_JOINT_DIM,
    SIDE_OBSERVATION_DIM,
    ControlledSide,
    H2S2RObservationLayout,
    fingertip_body_names,
    joint_names_for_side,
    simulation_joint_names,
)

__all__ = [
    "ACTION_DIM",
    "BIMANUAL_ACTION_DIM",
    "BIMANUAL_JOINT_DIM",
    "BIMANUAL_OBSERVATION_DIM",
    "OBSERVATION_DIM",
    "SIDE_ACTION_DIM",
    "SIDE_JOINT_DIM",
    "SIDE_OBSERVATION_DIM",
    "ControlledSide",
    "H2S2RObservationLayout",
    "fingertip_body_names",
    "joint_names_for_side",
    "simulation_joint_names",
]
