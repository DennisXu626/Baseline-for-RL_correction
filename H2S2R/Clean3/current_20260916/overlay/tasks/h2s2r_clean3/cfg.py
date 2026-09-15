"""Isaac Lab configuration for the Clean3 residual-58D curriculum route."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from rl_rebuild.baselines.h2s2r.contract import OBSERVATION_DIM
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg


ROBOT_ASSET_PATH = Path(
    "/ssd/sy/kailang/pour17/bundle/pour17/world/"
    "vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd"
)
# The NVIDIA grid-world asset vendored beside the robot in the frozen bundle.
# GroundPlaneCfg's default resolves to Omniverse/S3, which fails on an offline
# training host; using the vendored copy keeps the same plane prim and default
# physics material with no network dependency.
GROUND_PLANE_ASSET_PATH = ROBOT_ASSET_PATH.parent / "default_environment.usd"
# Welded to fixed joints in the bundled asset, so they are not articulated joints.
FIXED_TORSO_JOINTS = frozenset({"torso_j1", "torso_j2", "torso_j3"})

# H2S2R-style task state for the asymmetric critic only:
# actor obs 342 + two-object keypoints 18 + goal keypoints 18 + object
# velocities 12 + clock 1 + joint efforts 58 + two 10-pad force blocks 60.
CRITIC_STATE_DIM = 509
OURS_STAGE1_OBSERVATION_DIM = 367
OURS_STAGE1_PRIVILEGED_DIM = 22


@configclass
class Clean3H2S2REnvCfg(DexmateCorrectionEnvCfg):
    action_space = 58
    observation_space = OBSERVATION_DIM
    state_space = CRITIC_STATE_DIM
    priv_info_dim = 0
    enable_pointcloud = False
    enable_world_model = False
    prop_hist_len = 1
    clip_actions = 1.0
    clip_obs = 5.0
    # Exact teammate Stage-1 physics/control cadence: 240 Hz physics, 20 Hz policy.
    decimation = 12
    episode_length_s = 20.0
    is_finite_horizon = False

    runtime_root = ""
    runtime_inputs_npz = ""
    robot_usd = ""
    external_eval = False
    record_cameras = False
    penetration_sensor_cfgs = ()
    # Isolated exact Ours Stage-1 learning contract.  The default stays the
    # H2S2R-derived 342D actor + 509D critic route.
    ours_stage1_contract = False

    table_size = (1.2192, 1.8288, 0.04)
    table_top_z = 0.87
    table_friction = 0.5
    object_friction = 1.0
    # The working teammate route explicitly sets both object and fingertip-pad
    # friction to 1.0.  The previous 0.2 value was inherited from the failed
    # direct-58D baseline and makes released grasps five times more slippery.
    body_friction = 1.0
    plate_mass_kg = 0.30
    sponge_mass_kg = 0.05
    warmup_clamp_steps = 0
    early_reset_if_fingertips_far = True
    fingertips_close_threshold_m = 0.3
    terminate_when_reference_ends = True
    action_smoothing_weight = 0.0

    # Proven Clean3 controller: policy actions are bounded cumulative residuals
    # around the arm reference and grasp->squeeze finger feed-forward.
    residual_controller_inputs_npz = ""
    arm_step_rad = 0.004
    arm_dev_rad = 0.08
    finger_step_rad = 0.03
    finger_dev_rad = 0.60

    # Stage-1 reward/certification constants from the working teammate branch.
    pad_force_threshold_n = 0.5
    plate_support_min = 2
    sponge_pads_min = 3
    certify_position_m = 0.01
    certify_rotation_deg = 5.0
    certify_rows = 10
    drop_position_m = 0.03
    drop_rotation_deg = 20.0
    reward_contact_weight = 0.1
    reward_hold_position_weight = 0.5
    reward_hold_rotation_weight = 0.5
    reward_within_weight = 0.2
    reward_cert_bonus = 5.0
    reward_success_bonus = 20.0
    reward_drop_bonus = -10.0
    reward_action_weight = 0.001
    reward_cross_weight = 1.0
    cross_gap_min_m = 0.010
    overlap_min_m = 0.015

    # ---- Stage-1 grasp-startup curriculum (object pin + release_row annealing) ----
    # Both objects start inside the hands, so under an untrained policy they fall
    # within 0.33 s and the episode never recovers.  Pinning them to their reset
    # pose for an initial window gives the hand time to build contact; the window
    # is annealed away as the success rate rises.
    #
    # The schedule is authored in seconds and resolves to the teammate's exact
    # row counts at the configured 20 Hz control rate.
    # Vendored grid-world ground plane; set by build_cfg from the bundle.
    ground_plane_usd = ""

    grasp_curriculum = True
    close_seconds = 0.5            # nominal finger-closing window
    pin_seconds = 2.5              # curriculum start: total pin duration
    release_step_seconds = 0.25    # one anneal notch
    hold_seconds = 3.0             # post-release survival window scored as success
    release_jitter_rows = 4        # exact working 20 Hz curriculum value
    release_settle_rows = 0        # working Stage-1 scores from release onward
    # Working Stage-2 clock budget: round(424 reference rows * 1.3 slack).
    # Every curriculum level receives the same post-startup task budget.
    task_rows = 551
    # Curriculum-only failure termination.  A dropped object leaves ~400 dead rows
    # that teach nothing, so during curriculum training the episode ends instead.
    # Never enabled for the grasp_curriculum=False evaluation configuration, which
    # must keep the original baseline termination set.
    terminate_on_grasp_drop = True
    # Curriculum level to start at, in rows.  0 means "use release_max_rows".  Set
    # this to resume a run at the level it had annealed to, or to replay a
    # checkpoint at the pin length it was actually trained with.  It must be known
    # before the environment's first reset, hence a cfg field rather than an
    # attribute assigned afterwards.
    release_row_start = 0
    # Derived from the seconds above by build_cfg(); never set these by hand.
    close_rows = 0
    release_max_rows = 0
    release_min_rows = 0
    release_step_rows = 0
    hold_rows = 0

    aux_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Aux",
        spawn=sim_utils.UsdFileCfg(
            usd_path="__SET_BY_BUILD_CFG__.usd", activate_contact_sensors=True
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, table_top_z)),
    )
    # Replicate the identical physics topology so Isaac Lab also installs
    # collision filtering between cloned environments.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)


def _sensors() -> list[ContactSensorCfg]:
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    result = [
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/right_{finger}_elastomer",
            history_length=1,
            filter_prim_paths_expr=["/World/envs/env_.*/Aux"],
            max_contact_data_count_per_prim=64,
        ) for finger in fingers
    ]
    result += [
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/left_{finger}_elastomer",
            history_length=1,
            filter_prim_paths_expr=["/World/envs/env_.*/Object"],
            max_contact_data_count_per_prim=64,
        ) for finger in fingers
    ]
    return result


def build_cfg(
    *,
    runtime_root: str | Path,
    v12_root: str | Path,
    num_envs: int,
    seed: int,
    external_eval: bool = False,
    record_cameras: bool = False,
    grasp_curriculum: bool | None = None,
    release_row_start: int | None = None,
    terminate_on_grasp_drop: bool | None = None,
    ours_stage1_contract: bool = False,
) -> Clean3H2S2REnvCfg:
    root = Path(runtime_root).resolve()
    v12 = Path(v12_root).resolve()
    required = {
        "inputs": root / "data/clean3_c3p1_runtime_inputs.npz",
        "residual_inputs": root / "data/clean3_residual_controller_inputs.npz",
        "plate": root / "assets/retarget/object_0_textured.usd",
        "sponge": root / "assets/retarget/object_1_textured.usd",
        "robot": ROBOT_ASSET_PATH,
        # Checked up front so an offline host fails here with a path, rather than
        # deep inside spawn_ground_plane with an S3 URL.
        "ground_plane": GROUND_PLANE_ASSET_PATH,
    }
    missing = [f"{name}:{path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Clean3 C3-P1R2 deployment incomplete: {missing}")
    cfg = Clean3H2S2REnvCfg()
    cfg.ground_plane_usd = str(GROUND_PLANE_ASSET_PATH)
    cfg.runtime_root = str(root)
    cfg.runtime_inputs_npz = str(required["inputs"])
    cfg.residual_controller_inputs_npz = str(required["residual_inputs"])
    cfg.robot_usd = str(required["robot"])
    cfg.external_eval = bool(external_eval)
    cfg.record_cameras = bool(record_cameras)
    if grasp_curriculum is not None:
        cfg.grasp_curriculum = bool(grasp_curriculum)
    cfg.ours_stage1_contract = bool(ours_stage1_contract)
    if cfg.ours_stage1_contract:
        if not cfg.grasp_curriculum:
            raise ValueError("Ours Stage-1 contract requires the grasp curriculum")
        cfg.observation_space = OURS_STAGE1_OBSERVATION_DIM
        cfg.state_space = 0
        cfg.priv_info_dim = OURS_STAGE1_PRIVILEGED_DIM
        cfg.clip_obs = 10.0
        # Exact Stage-1 has no moving-reference task tail.  It terminates at the
        # fixed RELEASE_MAX + HOLD_ROWS boundary used by the teammate run.
        cfg.task_rows = 0
        cfg.early_reset_if_fingertips_far = False
        cfg.terminate_when_reference_ends = False
    if terminate_on_grasp_drop is not None:
        cfg.terminate_on_grasp_drop = bool(terminate_on_grasp_drop)
    if release_row_start is not None:
        cfg.release_row_start = int(release_row_start)
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = int(seed)
    cfg.robot_cfg.spawn.usd_path = cfg.robot_usd
    # The bundled robot has 58 movable joints: its three torso joints were replaced
    # by fixed joints.  The inherited Dexmate config still carries initial-state
    # entries for them, and Isaac's name resolution raises ValueError on a pattern
    # that matches nothing, so drop them when binding the fixed-torso asset.
    # Pour17 direct-58D does the same.
    cfg.robot_cfg.init_state.joint_pos = {
        name: value
        for name, value in cfg.robot_cfg.init_state.joint_pos.items()
        if name not in FIXED_TORSO_JOINTS
    }
    cfg.object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(required["plate"]), activate_contact_sensors=True
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, cfg.table_top_z)),
    )
    cfg.aux_object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Aux",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(required["sponge"]), activate_contact_sensors=True
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, cfg.table_top_z)),
    )
    cfg.contact_sensors = _sensors()
    cfg.penetration_sensor_cfgs = (
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Object",
            history_length=1,
            filter_prim_paths_expr=["/World/envs/env_.*/Aux"],
            max_contact_data_count_per_prim=64,
        ),
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

    # At the teammate controller's 20 Hz these evaluate exactly to 10, 50, 5, 60.
    control_hz = 1.0 / (cfg.sim.dt * cfg.decimation)
    cfg.close_rows = round(cfg.close_seconds * control_hz)
    cfg.release_max_rows = round(cfg.pin_seconds * control_hz)
    cfg.release_step_rows = round(cfg.release_step_seconds * control_hz)
    cfg.hold_rows = round(cfg.hold_seconds * control_hz)
    # Curriculum floor: release the instant the closing window ends.
    cfg.release_min_rows = cfg.close_rows

    if cfg.release_row_start and not (
        cfg.release_min_rows <= cfg.release_row_start <= cfg.release_max_rows
    ):
        raise ValueError(
            f"release_row_start {cfg.release_row_start} outside the curriculum "
            f"[{cfg.release_min_rows}, {cfg.release_max_rows}]"
        )
    # The drop termination belongs to curriculum training only; with the curriculum
    # off, the original baseline termination set must stand unmodified.
    if not cfg.grasp_curriculum:
        cfg.terminate_on_grasp_drop = False

    # episode_length_s sizes the PPO rollout buffers, so it is held constant at the
    # worst case: the longest pin plus settle plus hold, then the task rows.  It is
    # the hard ceiling.  Each env additionally times out against its own release row
    # (see _get_dones), which is what keeps the task budget identical at every
    # curriculum level.
    startup_rows = cfg.release_max_rows + cfg.release_settle_rows + cfg.hold_rows
    # +1 row of headroom: the base timeout fires at max_episode_length - 1, so
    # without it the longest-pin/no-jitter case would lose one task row.
    episode_rows = cfg.task_rows + (startup_rows + 1 if cfg.grasp_curriculum else 0)
    cfg.episode_length_s = episode_rows * cfg.sim.dt * cfg.decimation
    return cfg
