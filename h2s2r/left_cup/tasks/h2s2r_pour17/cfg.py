"""Isaac Lab configuration for independent single-hand H2S2R baselines."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from rl_rebuild.baselines.h2s2r.contract import ACTION_DIM, OBSERVATION_DIM
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg


@configclass
class Pour17H2S2REnvCfg(DexmateCorrectionEnvCfg):
    """Same robot/world as Pour17, with H2S2R's policy-facing contract."""

    action_space = ACTION_DIM
    observation_space = OBSERVATION_DIM
    state_space = 0
    priv_info_dim = 0
    # The inherited correction task enables ours PointNet branch.  H2S2R's
    # policy consumes only its proprioception/object/reference observation.
    enable_pointcloud = False
    enable_world_model = False
    prop_hist_len = 1
    clip_actions = 1.0
    clip_obs = 5.0
    decimation = 12
    episode_length_s = 45.15  # 903 control steps at 20 Hz.
    is_finite_horizon = False

    # These paths are filled by build_cfg; keeping them explicit makes every run
    # record which frozen bundle supplied data and assets.
    bundle_root = ""
    perception_npz = ""
    input_regime = "estimated"
    reference_start_index = 0
    provenance_manifest = ""
    right_synergy_npz = ""
    left_synergy_npz = ""
    allow_analytic_synergy = False
    external_evaluator_controls_termination = False
    reset_regime = "phase1_g2"

    right_fabric_urdf = str(
        Path(__file__).resolve().parents[2]
        / "rl_rebuild/baselines/h2s2r/assets/vega_sharpa_right_fabric.urdf"
    )
    left_fabric_urdf = str(
        Path(__file__).resolve().parents[2]
        / "rl_rebuild/baselines/h2s2r/assets/vega_sharpa_left_fabric.urdf"
    )

    # Original H2S2R forced-reference reward/clock constants.
    fingertips_close_threshold_m = 0.3
    stop_reference_threshold_m = 0.2
    success_region_radius_m = 0.05
    action_smoothing_weight = 0.0
    joint_progress_reward_weight = 0.0
    joint_success_reward_weight = 0.0

    # The phase-one comparison starts at ours' g2 pre-manipulation entry, so the
    # original H2S2R 30 cm far-fingertip termination remains valid.
    early_reset_if_fingertips_far = True
    terminate_when_reference_ends = True

    table_size = (1.2192, 1.8288, 0.04)
    table_top_z = 0.87
    table_friction = 0.5
    bottle_mass_kg = 0.53
    bottle_friction = 3.0
    cup_mass_kg = 0.15
    cup_friction = 0.5
    body_friction = 0.2
    # Ours clamps both objects for 15 training steps after a g2 RSI reset so the
    # fingers can settle.  Formal evaluation always overrides this to zero.
    warmup_clamp_steps = 15
    certification_lift_m = 0.015

    aux_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Aux",
        # Replaced by build_cfg() before scene construction.  A non-empty
        # placeholder keeps Isaac Lab config validation deterministic.
        spawn=sim_utils.UsdFileCfg(usd_path="__SET_BY_BUILD_CFG__.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, table_top_z)),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=2.0,
        replicate_physics=False,
    )


def _sensor_cfgs() -> list[ContactSensorCfg]:
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    sensors = [
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/right_{finger}_elastomer",
            history_length=1,
            filter_prim_paths_expr=["/World/envs/env_.*/Object"],
        )
        for finger in fingers
    ]
    sensors += [
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/left_{finger}_elastomer",
            history_length=1,
            filter_prim_paths_expr=["/World/envs/env_.*/Aux"],
        )
        for finger in fingers
    ]
    return sensors


def build_cfg(
    *,
    bundle_root: str | Path,
    num_envs: int,
    seed: int,
    input_regime: str = "estimated",
    input_archive: str | Path | None = None,
    provenance_manifest: str | Path | None = None,
    right_synergy_npz: str | Path | None = None,
    left_synergy_npz: str | Path | None = None,
    allow_analytic_synergy: bool = False,
    external_evaluator_controls_termination: bool = False,
    reset_regime: str = "phase1_g2",
    reference_start_index: int = 0,
    joint_progress_reward_weight: float = 0.0,
    joint_success_reward_weight: float = 0.0,
) -> Pour17H2S2REnvCfg:
    """Create a run config from the frozen bundle without importing ours rewards."""

    root = Path(bundle_root).resolve()
    required = (
        root / "world/canonical_reset_v1.json",
        root / "world/world_manifest.json",
        root / "world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd",
        root / "world/objects/object_1.usd",
        root / "world/cache/object_0.usd",
        root / "perception/pour17_perception.npz",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Pour17 bundle is incomplete: {missing}")

    cfg = Pour17H2S2REnvCfg()
    cfg.bundle_root = str(root)
    cfg.perception_npz = str(
        (root / "perception/pour17_perception.npz")
        if input_archive is None
        else Path(input_archive).resolve()
    )
    if not Path(cfg.perception_npz).is_file():
        raise FileNotFoundError(f"task input archive not found: {cfg.perception_npz}")
    cfg.input_regime = input_regime
    if not isinstance(reference_start_index, int) or reference_start_index < 0:
        raise ValueError("reference_start_index must be a non-negative integer")
    cfg.reference_start_index = reference_start_index
    if joint_progress_reward_weight or joint_success_reward_weight:
        raise ValueError(
            "v6 permits only the released single-hand H2S2R reward; the earlier "
            "bimanual joint reward is archived with v5"
        )
    cfg.joint_progress_reward_weight = 0.0
    cfg.joint_success_reward_weight = 0.0
    cfg.provenance_manifest = "" if provenance_manifest is None else str(provenance_manifest)
    cfg.right_synergy_npz = "" if right_synergy_npz is None else str(right_synergy_npz)
    cfg.left_synergy_npz = "" if left_synergy_npz is None else str(left_synergy_npz)
    cfg.allow_analytic_synergy = bool(allow_analytic_synergy)
    cfg.external_evaluator_controls_termination = bool(
        external_evaluator_controls_termination
    )
    if reset_regime not in ("phase1_g2", "canonical_t0"):
        raise ValueError("reset_regime must be 'phase1_g2' or 'canonical_t0'")
    cfg.reset_regime = reset_regime
    cfg.warmup_clamp_steps = 0 if external_evaluator_controls_termination else 15
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = int(seed)

    cfg.robot_cfg.spawn.usd_path = str(
        root / "world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd"
    )
    cfg.object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(root / "world/objects/object_1.usd"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, cfg.table_top_z)),
    )
    cfg.aux_object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Aux",
        spawn=sim_utils.UsdFileCfg(usd_path=str(root / "world/cache/object_0.usd")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, cfg.table_top_z)),
    )
    cfg.contact_sensors = _sensor_cfgs()

    # Match the runtime manifest rather than the older base-task defaults.
    cfg.sim.dt = 1.0 / 240.0
    cfg.sim.render_interval = 2
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
    return cfg
