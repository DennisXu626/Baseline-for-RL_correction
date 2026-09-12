"""FABRICS parameters for the 29-DoF Vega + Sharpa kinematic adapter."""

from __future__ import annotations

from copy import deepcopy

from .contract import ControlledSide, FABRIC_DOF_DIM, fingertip_body_names


_COMMON_PARAMS = {
    "cspace_attractor": {
        "min_isotropic_mass": 1.0,
        "max_isotropic_mass": 1.0,
        "mass_sharpness": 20.0,
        "mass_switch_offset": 1.0,
        "conical_sharpness": 20.0,
        "conical_gain": 1.0,
    },
    "joint_limit_repulsion": {
        "velocity_gate": True,
        "breakaway_distance": 0.0,
        "breakaway_velocity": 0.0,
        "soft_relu_gain": 8.0,
        "metric_scalar": 0.02,
        "metric_exploder_offset": 0.1,
        "max_metric": 100.0,
        "damping_gain": 8.0,
    },
    "hand_attractor": {
        "min_isotropic_mass": 8.0,
        "max_isotropic_mass": 8.0,
        "mass_sharpness": 20.0,
        "mass_switch_offset": 0.5,
        "conical_sharpness": 40.0,
        "conical_gain": 50.0,
        "damping": 50.0,
        "damping_sharpness": 10.0,
        "damping_radius": 0.2,
    },
    "palm_attractor": {
        "min_isotropic_mass": 1.0,
        "max_isotropic_mass": 1.0,
        "mass_sharpness": 10.0,
        "mass_switch_offset": 0.5,
        "conical_sharpness": 40.0,
        "conical_gain": 50.0,
        "damping": 50.0,
        "damping_sharpness": 10.0,
        "damping_radius": 0.2,
    },
    "body_repulsion": {
        "max_depth": 2.0,
        "min_depth": 0.01,
        "engage_depth": 0.5,
        "breakaway_depth": 0.0,
        "breakaway_velocity": 0.0,
        "metric_scalar": 1.0,
        "velocity_gate": True,
        "velocity_gate_sharpness": 1000.0,
        "velocity_gate_offset": 0.002,
        "rescaled_min_dist": 0.01,
        "forcing_metric_scalar": 0.01,
        "geom_metric_scalar": 0.01,
        "constant_accel": 1.0,
        "constant_accel_geom": 5.0,
        "damping_gain": 0.0,
    },
    "cspace_damping": {"gain": 65.0, "hand_gain": 5.0, "arm_dof_count": 7},
    "speed_control": {"active": False, "energy_target": 10.0, "damping": 100.0},
    "joint_limits": {
        "active": True,
        "acceleration": [7.5, 7.5, 10.0, 10.0, 10.0, 20.0, 20.0]
        + [22.5] * 22,
        "jerk": [3750.0, 3750.0, 5000.0, 5000.0, 5000.0, 10000.0, 10000.0]
        + [2250.0] * 22,
    },
}


def fabric_params_for_side(side: ControlledSide | str) -> dict:
    """Return an isolated parameter dictionary for one controlled side."""

    side = side if isinstance(side, ControlledSide) else ControlledSide.parse(side)
    p = side.arm_prefix
    arm_frames = (
        f"vega_1p_{p}_arm_l1",
        f"{p}_arm_l2",
        f"{p}_arm_l3",
        f"{p}_arm_l4",
        f"{p}_arm_l5",
        f"vega_1p_{p}_arm_l6",
        f"{p}_arm_l7",
    )
    tip_frames = fingertip_body_names(side)
    params = deepcopy(_COMMON_PARAMS)
    body = params["body_repulsion"]
    body["collision_sphere_frames"] = [*arm_frames, "h2s2r_palm", *tip_frames]
    body["collision_sphere_radii"] = [0.07] * 7 + [0.06] + [0.018] * 5
    body["collision_sphere_pairs"] = []
    # Match the official method's limited self-collision pattern: distal hand
    # bodies avoid a proximal arm link; world collision applies to every sphere.
    body["collision_link_prefix_pairs"] = [
        [tip, arm_frames[1]] for tip in tip_frames
    ]
    if len(params["joint_limits"]["acceleration"]) != FABRIC_DOF_DIM:
        raise AssertionError("acceleration parameter length must match 29 DoFs")
    return params

