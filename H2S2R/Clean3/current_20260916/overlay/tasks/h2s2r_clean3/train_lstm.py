"""Clean3 reference-centred residual-58D PPO/LSTM training entry."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher


HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--runtime_root", type=Path, required=True)
parser.add_argument("--v12_root", type=Path, required=True)
parser.add_argument("--output_root", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--smoke_iterations", type=int, default=0,
    help="Run exactly this many complete PPO iterations; zero keeps the full configured budget.",
)
parser.add_argument("--resume_checkpoint", type=Path)
parser.add_argument("--resume_checkpoint_sha256")
parser.add_argument(
    "--release_row", type=int,
    help="Stage-1 curriculum start in rows; defaults to the configured maximum. "
         "When resuming, pass release_row_cur from the previous run's curriculum_state.json "
         "so the pin window does not restart at its longest setting.",
)
parser.add_argument(
    "--anneal_ema", type=float, default=0.7,
    help="Shorten the pin window once the grasp success-rate EMA holds at or above this value.",
)
parser.add_argument(
    "--anneal_sustain", type=int, default=10,
    help="Consecutive epoch windows the EMA must hold before one anneal step is taken.",
)
parser.add_argument(
    "--resume_curriculum_state", type=Path,
    help="curriculum_state.json from the run being resumed; restores the annealer's "
         "EMA and sustain counter, which are not stored in the PPO checkpoint.",
)
parser.add_argument(
    "--max_transitions", type=int,
    help="Override the transition budget for this run. Use for bounded tests "
         f"(e.g. 4000000); defaults to the configured {150_000_000 // 1_000_000}M.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or args.seed != 42 or args.smoke_iterations not in (0, 3):
    raise SystemExit("Clean3 residual-58D training requires a positive env count and seed42")
if bool(args.resume_checkpoint) != bool(args.resume_checkpoint_sha256):
    raise SystemExit("resume checkpoint and its expected SHA256 must be provided together")
if args.resume_checkpoint and args.smoke_iterations:
    raise SystemExit("resume training cannot be combined with the three-iteration smoke mode")
if not 0.0 < args.anneal_ema <= 1.0 or args.anneal_sustain < 1:
    raise SystemExit("release_row annealing needs 0 < anneal_ema <= 1 and a positive sustain window")
if args.max_transitions is not None and args.max_transitions < 1:
    raise SystemExit("--max_transitions must be positive")
if args.resume_curriculum_state and not args.resume_checkpoint:
    raise SystemExit("--resume_curriculum_state only applies when resuming a checkpoint")
if args.resume_curriculum_state and args.release_row is None:
    # The level must be known before the environment's first reset, so read it here
    # rather than making the operator repeat it on the command line.
    args.release_row = int(
        json.loads(args.resume_curriculum_state.read_text(encoding="utf-8"))["release_row_cur"]
    )
app = AppLauncher(args).app
OFFICIAL = args.v12_root.resolve() / "third_party/h2s2r_official"
MAX_TRANSITIONS = args.max_transitions or 150_000_000
MILESTONE_STEP = 5_000_000          # checkpoint cadence
VIDEO_EVERY_N_MILESTONES = 2        # record video every second milestone, i.e. 10M
HORIZON = 16

sys.path.insert(0, str(args.v12_root.resolve()))
sys.path.insert(0, str(args.runtime_root.resolve()))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(OFFICIAL))
from human2sim2robot.ppo.ppo_agent import PpoAgent, PpoConfig  # noqa: E402
from human2sim2robot.ppo.utils.dict_to_dataclass import dict_to_dataclass  # noqa: E402
from human2sim2robot.ppo.utils.network import NetworkConfig  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.h2s2r_clean3.cfg import build_cfg  # noqa: E402
from tasks.h2s2r_clean3.env import Clean3H2S2REnv  # noqa: E402
from tasks.h2s2r_clean3.official_ppo_adapter import OfficialPpoEnvAdapter  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_safe(value):
    """Represent non-finite configuration sentinels in standards-compliant JSON."""

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


class AuditedAgent(PpoAgent):
    """Preserve the latest completed optimizer state and milestone counts, and
    drive the Stage-1 release_row curriculum."""

    def __init__(self, *positional, raw_env, **keyword):
        super().__init__(*positional, **keyword)
        self._raw_env = raw_env
        self._release_ema = 0.0
        self._release_sustain = 0

    def _anneal_release_row(self, transitions: int) -> dict:
        """Success-rate EMA controller for the Stage-1 pin window.

        One PPO epoch is one window: ema = 0.95*ema + 0.05*success_rate.  Once the
        EMA holds at or above --anneal_ema for --anneal_sustain consecutive windows
        the pin shortens by one step and both counters restart, so every notch has
        to be earned from scratch.  The curriculum only ever moves down.
        """

        env_cfg = self._raw_env.cfg
        rates = self._raw_env.pop_rates()
        success = rates.get("sr/grasp_success")
        annealed = False
        if success is not None:
            self._release_ema = 0.95 * self._release_ema + 0.05 * success
            self._release_sustain = (
                self._release_sustain + 1 if self._release_ema >= args.anneal_ema else 0
            )
            if (
                self._release_sustain >= args.anneal_sustain
                and self._raw_env.release_row_cur > env_cfg.release_min_rows
            ):
                self._raw_env.release_row_cur = max(
                    int(env_cfg.release_min_rows),
                    self._raw_env.release_row_cur - int(env_cfg.release_step_rows),
                )
                self._release_ema = 0.0
                self._release_sustain = 0
                annealed = True
                print(
                    f"[CLEAN3] curriculum release_row -> {self._raw_env.release_row_cur} rows "
                    f"@ epoch {int(self.epoch_num)} transitions {transitions}",
                    flush=True,
                )
        if self.writer is not None:
            for tag, value in (
                ("curr/release_row_cur", float(self._raw_env.release_row_cur)),
                ("curr/ema_success", self._release_ema),
                ("curr/sustain_windows", float(self._release_sustain)),
                *rates.items(),
            ):
                self.writer.add_scalar(tag=tag, scalar_value=value, global_step=transitions)
            # The vendored PPO never flushes, and tensorboardX's default
            # flush_secs=120 never elapses on a short run, so curriculum scalars
            # would be lost on exactly the smoke runs that verify them.
            self.writer.flush()
        return {
            "completed_epochs": int(self.epoch_num),
            "cumulative_transitions": transitions,
            "release_row_cur": int(self._raw_env.release_row_cur),
            "release_min_rows": int(env_cfg.release_min_rows),
            "release_step_rows": int(env_cfg.release_step_rows),
            "ema_success": self._release_ema,
            "sustain_windows": self._release_sustain,
            "annealed_this_epoch": annealed,
            "window_rates": rates,
        }

    def restore_curriculum(self, state: dict) -> None:
        """Restore annealer state saved by a previous run.

        The EMA and sustain counter are not part of the PPO checkpoint.  The
        release level is validated against this run's configured ladder, and
        ``release_row_start`` should already have applied it before the first
        reset -- this only re-asserts it and brings the counters back.
        """

        level = int(state["release_row_cur"])
        env_cfg = self._raw_env.cfg
        if not env_cfg.release_min_rows <= level <= env_cfg.release_max_rows:
            raise SystemExit(
                f"restored release_row_cur {level} outside this run's curriculum "
                f"[{env_cfg.release_min_rows}, {env_cfg.release_max_rows}]"
            )
        if level != self._raw_env.release_row_cur:
            raise SystemExit(
                f"restored release_row_cur {level} disagrees with the level applied "
                f"before reset ({self._raw_env.release_row_cur}); pass --release_row {level}"
            )
        self._release_ema = float(state["ema_success"])
        self._release_sustain = int(state["sustain_windows"])

    def train(self):
        """Join the TensorBoard writer thread before Isaac tears the process down.

        The vendored PPO returns from train() without closing its SummaryWriter, so
        the last queued scalar is dropped when app.close() runs.  Diagnostics only:
        no PPO update, reward, physics or reset behaviour changes.
        """

        try:
            return super().train()
        finally:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

    def train_epoch(self):
        value = super().train_epoch()
        completed = int(self.epoch_num)
        transitions = completed * args.num_envs * HORIZON
        if self._raw_env.cfg.grasp_curriculum:
            atomic_json(
                Path(self.experiment_dir) / "curriculum_state.json",
                self._anneal_release_row(transitions),
            )
        if completed == 1 or completed % 8 == 0 or completed == int(self.cfg.max_epochs):
            self.save(self.nn_dir / "latest_complete.pth")
        while self._next_milestone <= min(transitions, self._run_end_transitions):
            milestone_m = self._next_milestone // 1_000_000
            checkpoint = self.nn_dir / f"milestone_{milestone_m:03d}M.pth"
            self.save(checkpoint)
            atomic_json(checkpoint.with_suffix(".json"), {
                "requested_transition": self._next_milestone,
                "completed_transitions": transitions,
                "completed_epochs": completed,
                "checkpoint_sha256": sha256(checkpoint),
                # The curriculum level this checkpoint was trained at.  Milestone
                # replay MUST run at this pin length; replaying an annealed policy
                # at the initial 38-row pin does not evaluate what was trained.
                "release_row_cur": int(self._raw_env.release_row_cur),
                "grasp_curriculum": bool(self._raw_env.cfg.grasp_curriculum),
                # Video is recorded on every Nth milestone (10M with 5M checkpoints).
                "record_video": milestone_m % (
                    VIDEO_EVERY_N_MILESTONES * (MILESTONE_STEP // 1_000_000)
                ) == 0,
            })
            self._next_milestone += MILESTONE_STEP
        atomic_json(
            Path(self.experiment_dir) / "training_progress.json",
            {
                "completed_epochs": completed,
                "cumulative_transitions": transitions,
                "segment_transitions": transitions - self._run_start_transitions,
                "optimizer_updates_lower_bound": completed - self._run_start_epoch,
            },
        )
        return value


def main() -> None:
    started = time.monotonic()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with (HERE / "ppo_h2s2r_lstm.yaml").open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    max_epochs = MAX_TRANSITIONS // (args.num_envs * HORIZON)
    if max_epochs < 1:
        raise RuntimeError("environment count exceeds the training transition budget")
    values["ppo"]["num_actors"] = args.num_envs
    values["ppo"]["device"] = args.device
    configured_epochs = args.smoke_iterations or max_epochs
    values["ppo"]["max_epochs"] = configured_epochs
    network_cfg = dict_to_dataclass(values["network"], NetworkConfig)
    ppo_cfg = dict_to_dataclass(values["ppo"], PpoConfig)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_manifest.json").exists() or (output / "nn").exists():
        raise FileExistsError(f"training output already contains a run: {output}")
    cfg = build_cfg(
        runtime_root=args.runtime_root, v12_root=args.v12_root, num_envs=args.num_envs,
        seed=42, external_eval=False,
        # Applied before the environment's first reset so that a resumed run's very
        # first episodes already use the annealed pin length.
        release_row_start=args.release_row,
    )
    cfg.sim.device = args.device
    print("[CLEAN3] stage=construct_env begin", flush=True)
    env_raw = Clean3H2S2REnv(cfg)
    env_raw.set_audit_log(output / "policy_step_audit.jsonl")
    print(
        f"[CLEAN3] curriculum release_row start={env_raw.release_row_cur} "
        f"min={cfg.release_min_rows} step={cfg.release_step_rows} "
        f"jitter={cfg.release_jitter_rows} hold={cfg.hold_rows} "
        f"enabled={cfg.grasp_curriculum}",
        flush=True,
    )
    print("[CLEAN3] stage=construct_env complete", flush=True)
    env = OfficialPpoEnvAdapter(GymStyleEnvWrapper(env_raw, clip_actions=cfg.clip_actions))
    agent = AuditedAgent(
        experiment_dir=output, ppo_config=ppo_cfg, network_config=network_cfg, env=env, raw_env=env_raw
    )
    print("[CLEAN3] stage=ppo_agent complete", flush=True)
    start_epoch = 0
    resume_path = None
    resume_sha = None
    if args.resume_checkpoint:
        resume_path = args.resume_checkpoint.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume_sha = sha256(resume_path)
        if resume_sha != args.resume_checkpoint_sha256.lower():
            raise RuntimeError(
                f"resume checkpoint SHA256 mismatch: expected {args.resume_checkpoint_sha256.lower()}, got {resume_sha}"
            )
        agent.restore(resume_path, set_epoch=True)
        start_epoch = int(agent.epoch_num)
        agent.cfg.max_epochs = start_epoch + max_epochs
        atomic_json(output / "resume_checkpoint.json", {
            "path": str(resume_path), "sha256": resume_sha,
            "restored_epoch": start_epoch, "restored_frame": int(agent.frame),
            "optimizer_state_restored": True,
        })
        # The annealer's EMA and sustain counter live outside the PPO checkpoint, so
        # restore them from the previous run's curriculum_state.json.  Without this
        # a resumed run re-accumulates the EMA from zero and delays the next notch
        # by ~24 epochs even though the policy already earned it.
        if cfg.grasp_curriculum and args.resume_curriculum_state:
            state = json.loads(args.resume_curriculum_state.read_text(encoding="utf-8"))
            agent.restore_curriculum(state)
            print(
                f"[CLEAN3] curriculum restored release_row={env_raw.release_row_cur} "
                f"ema={agent._release_ema:.4f} sustain={agent._release_sustain}",
                flush=True,
            )
        print(f"[CLEAN3] stage=resume complete epoch={start_epoch}", flush=True)
    else:
        initial_path = agent.nn_dir / "initial_random.pth"
        agent.save(initial_path)
        atomic_json(output / "initial_checkpoint.json", {"path": str(initial_path), "sha256": sha256(initial_path)})
    agent._run_start_epoch = start_epoch
    agent._run_start_transitions = start_epoch * args.num_envs * HORIZON
    agent._run_end_transitions = agent.cfg.max_epochs * args.num_envs * HORIZON
    agent._next_milestone = (
        agent._run_start_transitions // MILESTONE_STEP + 1
    ) * MILESTONE_STEP
    manifest = {
        "schema": "h2s2r_clean3_ours_grasp_bootstrap_adapter_ppo_v4",
        "method": (
            "Local bimanual 342D H2S2R-derived actor adapter using the upstream PPO/LSTM "
            "implementation and a disclosed bimanual H2S2R-style asymmetric critic; with explicitly "
            "disclosed Ours-derived Clean3 grasp-start residual curriculum and Stage-1 reward"
        ),
        "upstream_fidelity_warning": (
            "Official default plate training is 144D actor observation plus 453D asymmetric-critic "
            "state at upstream commit 894eae2. This adapter intentionally excludes Fabrics/PCA, uses "
            "a bimanual 342D actor and 509D critic state, and is not the upstream policy contract."
        ),
        "seed": 42, "num_envs": args.num_envs, "observation_dim": 342,
        "critic_state_dim": 509, "action_dim": 58,
        "rollout": HORIZON, "minibatch": 16384,
        "start_epoch": start_epoch, "end_epoch": int(agent.cfg.max_epochs),
        "start_cumulative_transitions": agent._run_start_transitions,
        "end_cumulative_transitions": agent._run_end_transitions,
        "scheduled_segment_transitions": agent._run_end_transitions - agent._run_start_transitions,
        "requested_segment_transition_cap": MAX_TRANSITIONS,
        "from_scratch": resume_path is None,
        "run_mode": "resume_20m" if resume_path else ("three_iteration_smoke" if args.smoke_iterations else "full_training"),
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "resume_checkpoint_sha256": resume_sha,
        "reward": (
            "Ours-derived Clean3 Stage-1 contact/hold/certification reward before grasp success; "
            "mean H2S2R object tracking reward after grasp success"
        ),
        "asymmetric_critic": {
            "source": "task-adapted analogue of official H2S2R privileged state",
            "fields": {
                "actor_observation": 342,
                "two_object_keypoints": 18,
                "two_goal_object_keypoints": 18,
                "two_object_linear_angular_velocity": 12,
                "episode_clock": 1,
                "joint_applied_torque": 58,
                "ten_fingertip_net_contact_forces": 30,
                "ten_object_filtered_contact_forces": 30,
            },
            "difference_from_upstream": (
                "The Isaac Lab adapter exposes the ten task fingertip sensors rather than the "
                "official single-hand simulator's all-rigid-body contact and force-sensor tensors."
            ),
        },
        "controller": {
            "source": "Ours Clean3 grasp-start adapter (not original H2S2R)",
            "action_semantics": "bounded cumulative residual around reference feed-forward",
            "inputs": cfg.residual_controller_inputs_npz,
            "inputs_sha256": sha256(Path(cfg.residual_controller_inputs_npz)),
            "arm_step_rad": cfg.arm_step_rad,
            "arm_dev_rad": cfg.arm_dev_rad,
            "finger_step_rad": cfg.finger_step_rad,
            "finger_dev_rad": cfg.finger_dev_rad,
            "finger_feedforward": "grasp_to_squeeze_then_hold",
            "arm_feedforward": "reference_row_selected_by_shared_task_clock",
            "stage1_residual_coordinates": 58,
            "stage2_clamped_residual_joints": list(env_raw.stage2_clamped_joint_names),
            "arm_sag_rad": [float(value) for value in env_raw.arm_sag_q14.cpu()],
        },
        "grasp_curriculum": {
            "enabled": bool(cfg.grasp_curriculum),
            "control_hz": 1.0 / (cfg.sim.dt * cfg.decimation),
            "close_seconds": cfg.close_seconds,
            "pin_seconds": cfg.pin_seconds,
            "release_step_seconds": cfg.release_step_seconds,
            "hold_seconds": cfg.hold_seconds,
            "release_max_rows": int(cfg.release_max_rows),
            "release_min_rows": int(cfg.release_min_rows),
            "release_step_rows": int(cfg.release_step_rows),
            "release_jitter_rows": int(cfg.release_jitter_rows),
            "hold_rows": int(cfg.hold_rows),
            "certify_position_m": cfg.certify_position_m,
            "certify_rotation_deg": cfg.certify_rotation_deg,
            "certify_rows": cfg.certify_rows,
            "drop_position_m": cfg.drop_position_m,
            "drop_rotation_deg": cfg.drop_rotation_deg,
            "pad_force_threshold_n": cfg.pad_force_threshold_n,
            "contact_signal": "fingertip sensor net_forces_w, matching teammate Stage-1",
            "hold_datum": "fixed reset world pose, matching teammate Stage-1",
            "plate_support_min": cfg.plate_support_min,
            "sponge_pads_min": cfg.sponge_pads_min,
            "release_row_start": int(env_raw.release_row_cur),
            "anneal_ema": args.anneal_ema,
            "anneal_sustain": args.anneal_sustain,
            "episode_length_rows": int(env_raw.max_episode_length),
        },
        "ppo_config": json_safe(dataclasses.asdict(agent.cfg)),
        "network_config": json_safe(dataclasses.asdict(network_cfg)),
        "config_sha256": sha256(HERE / "ppo_h2s2r_lstm.yaml"),
    }
    atomic_json(output / "run_manifest.json", manifest)
    try:
        agent.train()
    finally:
        latest = agent.nn_dir / "latest_complete.pth"
        atomic_json(output / "training_exit.json", {
            "elapsed_seconds": time.monotonic() - started,
            "completed_epochs": int(getattr(agent, "epoch_num", 0)),
            "cumulative_transitions": int(getattr(agent, "epoch_num", 0)) * args.num_envs * HORIZON,
            "segment_transitions": (
                int(getattr(agent, "epoch_num", 0)) - agent._run_start_epoch
            ) * args.num_envs * HORIZON,
            "latest_complete": str(latest) if latest.is_file() else None,
            "latest_complete_sha256": sha256(latest) if latest.is_file() else None,
        })
        env.close()


try:
    main()
    print("[CLEAN3] stage=training complete", flush=True)
except BaseException as error:
    try:
        atomic_json(args.output_root / "entry_failure.json", {
            "stage": "training_entry", "type": type(error).__name__,
            "message": repr(error), "traceback": traceback.format_exc(),
        })
    except BaseException:
        traceback.print_exc()
    print("[CLEAN3-ENTRY-FAILURE] training", flush=True)
    print(f"[CLEAN3] training failed {type(error).__name__}: {error!r}", flush=True)
    traceback.print_exc()
    raise
finally:
    app.close()
