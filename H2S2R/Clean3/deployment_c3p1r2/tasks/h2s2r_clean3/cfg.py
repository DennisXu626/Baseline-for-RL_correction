"""Isaac Lab configuration for the frozen Clean3 C3-P1R2 route."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from rl_rebuild.baselines.h2s2r.contract import ACTION_DIM, OBSERVATION_DIM
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg


ROBOT_ASSET_PATH = Path(
    "/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17/world/"
    "vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd"
)


@configclass
class Clean3H2S2REnvCfg(DexmateCorrectionEnvCfg):
    action_space = ACTION_DIM
    observation_space = OBSERVATION_DIM
    state_space = 0
    priv_info_dim = 0
    enable_pointcloud = False
    enable_world_model = False
    prop_hist_len = 1
    clip_actions = 1.0
    clip_obs = 5.0
    decimation = 12
    episode_length_s = 20.0
    is_finite_horizon = False

    runtime_root = ""
    runtime_inputs_npz = ""
    action_sidecar_npz = ""
    right_synergy_npz = ""
    left_synergy_npz = ""
    robot_usd = ""
    right_fabric_urdf = ""
    left_fabric_urdf = ""
    external_eval = False
    record_cameras = False
    penetration_sensor_cfgs = ()

    table_size = (1.2192, 1.8288, 0.04)
    table_top_z = 0.87
    table_friction = 0.5
    object_friction = 1.0
    body_friction = 0.2
    plate_mass_kg = 0.30
    sponge_mass_kg = 0.05
    warmup_clamp_steps = 0
    early_reset_if_fingertips_far = True
    fingertips_close_threshold_m = 0.3
    terminate_when_reference_ends = True
    action_smoothing_weight = 0.0

    aux_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Aux",
        spawn=sim_utils.UsdFileCfg(usd_path="__SET_BY_BUILD_CFG__.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, table_top_z)),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=2.0, replicate_physics=False)


def _sensors() -> list[ContactSensorCfg]:
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    result = [
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/right_{finger}_elastomer",
            history_length=1,
            filter_prim_paths_expr=["/World/envs/env_.*/Aux"],
        ) for finger in fingers
    ]
    result += [
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/left_{finger}_elastomer",
            history_length=1,
            filter_prim_paths_expr=["/World/envs/env_.*/Object"],
        ) for finger in fingers
    ]
    return result


def build_cfg(*, runtime_root: str | Path, v12_root: str | Path, num_envs: int, seed: int, external_eval: bool = False, record_cameras: bool = False) -> Clean3H2S2REnvCfg:
    root = Path(runtime_root).resolve()
    v12 = Path(v12_root).resolve()
    required = {
        "inputs": root / "data/clean3_c3p1_runtime_inputs.npz",
        "sidecar": root / "data/clean3_action_calibration_c3p1r2.npz",
        "right_pca": root / "data/right_synergy_pca5.npz",
        "left_pca": root / "data/left_synergy_pca5.npz",
        "plate": root / "assets/retarget/object_0_textured.usd",
        "sponge": root / "assets/retarget/object_1_textured.usd",
        "robot": ROBOT_ASSET_PATH,
        "right_urdf": v12 / "rl_rebuild/baselines/h2s2r/assets/vega_sharpa_right_fabric.urdf",
        "left_urdf": v12 / "rl_rebuild/baselines/h2s2r/assets/vega_sharpa_left_fabric.urdf",
    }
    missing = [f"{name}:{path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Clean3 C3-P1R2 deployment incomplete: {missing}")
    cfg = Clean3H2S2REnvCfg()
    cfg.runtime_root = str(root)
    cfg.runtime_inputs_npz = str(required["inputs"])
    cfg.action_sidecar_npz = str(required["sidecar"])
    cfg.right_synergy_npz = str(required["right_pca"])
    cfg.left_synergy_npz = str(required["left_pca"])
    cfg.robot_usd = str(required["robot"])
    cfg.right_fabric_urdf = str(required["right_urdf"])
    cfg.left_fabric_urdf = str(required["left_urdf"])
    cfg.external_eval = bool(external_eval)
    cfg.record_cameras = bool(record_cameras)
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = int(seed)
    cfg.robot_cfg.spawn.usd_path = cfg.robot_usd
    cfg.object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=str(required["plate"])),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, cfg.table_top_z)),
    )
    cfg.aux_object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Aux",
        spawn=sim_utils.UsdFileCfg(usd_path=str(required["sponge"])),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, cfg.table_top_z)),
    )
    cfg.contact_sensors = _sensors()
    cfg.penetration_sensor_cfgs = (
        ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=1, filter_prim_paths_expr=["/World/envs/env_.*/Object"]),
        ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=1, filter_prim_paths_expr=["/World/envs/env_.*/Aux"]),
        ContactSensorCfg(prim_path="/World/envs/env_.*/Object", history_length=1, filter_prim_paths_expr=["/World/envs/env_.*/Aux"]),
    )
    cfg.sim.dt = 1.0 / 240.0
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.gravity = (0.0, 0.0, -9.81)
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.max_position_iteration_count = 32
    cfg.sim.physx.max_velocity_iteration_count = 0
    cfg.sim.physx.bounce_threshold_velocity = 0.2
    cfg.sim.physx.friction_offset_threshold = 0.04
    cfg.sim.physx.friction_correlation_distance = 0.025
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**23
    cfg.sim.physx.enable_ccd = False
    cfg.sim.physx.enable_stabilization = False
    cfg.episode_length_s = 400 * cfg.sim.dt * cfg.decimation
    return cfg
