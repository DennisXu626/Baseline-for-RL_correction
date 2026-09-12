"""Train the Pour17 adaptation with H2S2R's released recurrent PPO."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from runtime.training_pilot_20260912.resume_observation import from_environment
observation = from_environment()

from isaaclab.app import AppLauncher


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_OFFICIAL_ROOT = _REPO_ROOT / "third_party/h2s2r_official"
_DEFAULT_SYNERGY_DIR = _HERE / "artifacts/synergies"
_UPSTREAM_COMMIT = "894eae2ec3ae39a573b81bd1860d14cc6bdfa6df"

parser = argparse.ArgumentParser()
parser.add_argument("--bundle_root", type=Path, required=True)
parser.add_argument("--name", default="H2S2R_Pour17_RightOnly_Hold0_11")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--reference_start_index", type=int, required=True)
parser.add_argument("--joint_progress_reward_weight", type=float, default=0.0)
parser.add_argument("--joint_success_reward_weight", type=float, default=0.0)
parser.add_argument("--max_agent_steps", type=int, default=None)
parser.add_argument("--max_epochs", type=int, default=None)
parser.add_argument("--diagnostic_stop_epoch", type=int, default=None,
                    help="Stop at a saved boundary without shortening the PPO learning-rate schedule.")
parser.add_argument(
    "--training_pilot", action="store_true",
    help="Run the fixed 1500-epoch/6-hour bounded training pilot.",
)
parser.add_argument(
    "--pilot_output_root", type=Path, default=None,
    help="Existing large-capacity output root for the bounded pilot.",
)
parser.add_argument("--load_path", type=Path, default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--input_regime", choices=("estimated",), default="estimated")
parser.add_argument("--input_archive", type=Path, default=None)
parser.add_argument("--provenance_manifest", type=Path, default=None)
parser.add_argument(
    "--right_synergy_npz",
    type=Path,
    default=_DEFAULT_SYNERGY_DIR / "right_synergy_pca5.npz",
)
parser.add_argument(
    "--left_synergy_npz",
    type=Path,
    default=_DEFAULT_SYNERGY_DIR / "left_synergy_pca5.npz",
)
parser.add_argument(
    "--allow_analytic_synergy",
    action="store_true",
    help="Bring-up only; forbidden for formal results.",
)
parser.add_argument(
    "--smoke_test",
    action="store_true",
    help="One epoch with one full-rollout minibatch; never report as a formal run.",
)
parser.add_argument(
    "--preflight",
    action="store_true",
    help="One approved V13 256-env PPO epoch for memory/integration validation.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.num_envs <= 0:
    raise SystemExit("--num_envs must be positive")
if args.reference_start_index != 14:
    raise SystemExit("Right-only Run11 requires --reference_start_index 14")
if args.joint_progress_reward_weight < 0 or args.joint_success_reward_weight < 0:
    raise SystemExit("joint reward weights must be non-negative")
if args.joint_progress_reward_weight or args.joint_success_reward_weight:
    raise SystemExit("Right-only diagnostic uses the original single-side reward only")
if args.max_agent_steps is not None and args.max_epochs is not None:
    raise SystemExit("choose either --max_agent_steps or --max_epochs")
if args.diagnostic_stop_epoch is not None:
    if args.diagnostic_stop_epoch <= 0:
        raise SystemExit("diagnostic_stop_epoch must be positive")
    if args.max_agent_steps is not None or args.max_epochs is not None:
        raise SystemExit("diagnostic_stop_epoch preserves the configured PPO schedule")
if args.training_pilot:
    if args.diagnostic_stop_epoch is not None or args.max_agent_steps is not None or args.max_epochs is not None:
        raise SystemExit("training_pilot uses only its fixed independent stop hook")
    if args.pilot_output_root is None:
        raise SystemExit("training_pilot requires --pilot_output_root")
elif args.pilot_output_root is not None:
    raise SystemExit("--pilot_output_root requires --training_pilot")
if args.smoke_test and args.preflight:
    raise SystemExit("--smoke_test and --preflight are mutually exclusive")
if args.preflight and (args.max_agent_steps is not None or args.max_epochs is not None):
    raise SystemExit("--preflight fixes its own one-epoch limit")
if args.preflight and args.num_envs != 256:
    raise SystemExit("--preflight requires --num_envs 256")
if args.resume and args.load_path is None:
    raise SystemExit("--resume requires --load_path")
if args.resume or args.load_path is not None:
    raise SystemExit("Run11 starts fresh: checkpoint/optimizer/normalizer reuse is forbidden")
if args.input_regime == "oracle" and args.provenance_manifest is None:
    raise SystemExit("Oracle runs require --provenance_manifest")
if args.input_regime == "estimated" and args.provenance_manifest is not None:
    raise SystemExit("Estimated runs must not receive an Oracle provenance manifest")
if args.training_pilot:
    args.enable_cameras = True

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("h2s2r-pour17-lstm-train")
app = observation.call('AppLauncher', AppLauncher, args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(_OFFICIAL_ROOT))

from human2sim2robot.ppo.ppo_agent import PpoAgent, PpoConfig  # noqa: E402
from human2sim2robot.ppo.utils.dict_to_dataclass import dict_to_dataclass  # noqa: E402
from human2sim2robot.ppo.utils.network import NetworkConfig  # noqa: E402
from rl_rebuild.baselines.h2s2r.official_ppo_adapter import (  # noqa: E402
    OfficialPpoEnvAdapter,
)
from rl_rebuild.baselines.h2s2r.pour17.inputs import load_inputs  # noqa: E402
from rl_rebuild.baselines.h2s2r.contract import ControlledSide  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.h2s2r_pour17.right_zero_margin_install import (  # noqa: E402
    install_right_training_patch, verify_installed_right_fabric, verify_limit_dispatch,
)

_EXPECTED_RIGHT_FABRIC_CLASS, _PATCH_RECORD = install_right_training_patch()

from tasks.h2s2r_pour17.right_env import (  # noqa: E402
    build_cfg, RightBottleH2S2REnv as Pour17H2S2REnv,
)
from tasks.h2s2r_pour17.training_pilot_audit import AuditedRightBottleEnv, atomic_json  # noqa: E402
from tasks.h2s2r_pour17.right_workspace import workspace_record  # noqa: E402
from tasks.h2s2r_pour17.ours_physics import physics_record  # noqa: E402

observation.install_initialization()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


class DiagnosticComplete(Exception):
    """A completed saved boundary, not a trainer failure."""


class PilotComplete(Exception):
    """The bounded pilot reached its fixed epoch or wall-clock boundary."""


PILOT_MAX_EPOCHS = 1500
PILOT_WALL_SECONDS = 6 * 60 * 60
PILOT_SAVE_RESERVE_SECONDS = 5 * 60
PILOT_NUM_ENVS = 256
PILOT_HORIZON = 16
PILOT_MINIBATCH = 4096


def validate_v13_batch(num_envs: int, ppo: dict) -> dict:
    actual = {
        'num_envs': int(num_envs),
        'horizon_length': int(ppo['horizon_length']),
        'minibatch_size': int(ppo['minibatch_size']),
        'mini_epochs': int(ppo['mini_epochs']),
        'seq_length': int(ppo['seq_length']),
    }
    expected = {'num_envs': 256, 'horizon_length': 16, 'minibatch_size': 4096,
                'mini_epochs': 4, 'seq_length': 16}
    if actual != expected:
        raise ValueError(f'V13 batch protocol mismatch: {actual} != {expected}')
    return dict(actual, environment_transitions=actual['num_envs'] * actual['horizon_length'])


def pilot_stop_reason(epoch: int, elapsed_seconds: float) -> str | None:
    if epoch >= PILOT_MAX_EPOCHS:
        return "epoch_limit"
    if elapsed_seconds >= PILOT_WALL_SECONDS - PILOT_SAVE_RESERVE_SECONDS:
        return "wall_clock_save_reserve"
    return None


class LoggedPpoAgent(PpoAgent):
    """Upstream PPO with scalar diagnostics and the shared GPU pause hook only."""

    def train_epoch(self):
        _slot.yield_if_paused()
        if getattr(self, 'pilot_enabled', False):
            self.pilot_raw_env.left_delivery.preview()
            self.pilot_raw_env.delivery.preview()
        result = super().train_epoch()
        if getattr(self, "pilot_enabled", False):
            finite_values = []
            for value in result:
                if torch.is_tensor(value):
                    finite_values.append(bool(torch.isfinite(value).all()))
                elif isinstance(value, list):
                    finite_values.extend(
                        bool(torch.isfinite(item).all()) for item in value if torch.is_tensor(item)
                    )
                elif isinstance(value, (int, float)):
                    finite_values.append(bool(np.isfinite(value)))
            if not all(finite_values):
                raise RuntimeError("non-finite PPO epoch output")
            self.pilot_raw_env.finish_epoch(self.epoch_num)
            self.pilot_completed_epochs = self.epoch_num
            if self.epoch_num == 1:
                self.save(self.nn_dir / 'first_pilot_epoch_1.pth')
            record = dict(epoch=self.epoch_num, completed_epochs=self.pilot_completed_epochs,
                          transitions=self.pilot_raw_env.audit_counts['controls'] * self.pilot_raw_env.num_envs,
                          optimizer_updates=self.pilot_optimizer_updates,
                          elapsed_seconds=time.monotonic() - self.pilot_started_monotonic,
                          learning_rate=float(result[-2]), scalars={
                              key: float(value.detach().cpu()) if torch.is_tensor(value) else value
                              for key, value in self.env.last_extras.items()
                              if isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1)})
            with (self.pilot_status_path.parent / 'learning_progress.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            atomic_json(self.pilot_status_path.parent / 'learning_progress.json', record)
        return result

    def save(self, filename):
        super().save(filename)
        if getattr(self, 'pilot_enabled', False):
            path = Path(filename)
            normalizer = path.with_suffix('.normalizer.pth')
            torch.save({
                'running_mean_std': self.model.running_mean_std.state_dict(),
                'value_mean_std': self.model.value_mean_std.state_dict(),
            }, normalizer)
            atomic_json(self.pilot_status_path.parent / 'latest_checkpoint.json',
                dict(path=str(path.resolve()), exists=path.is_file(),
                     sha256=_sha256(path) if path.is_file() else None,
                     normalizer_path=str(normalizer.resolve()),
                     normalizer_sha256=_sha256(normalizer),
                     optimizer_updates=self.pilot_optimizer_updates,
                     completed_epochs=getattr(self, 'pilot_completed_epochs', 0)))
            print(f'PILOT_CHECKPOINT {path}', flush=True)

    def install_update_observation(self):
        original = self.optimizer.step
        self.pilot_optimizer_updates = 0
        self.pilot_completed_epochs = 0
        def observed_step(*positional, **keyword):
            if self.pilot_optimizer_updates == 0:
                gate = json.loads((self.pilot_raw_env.audit_root / 'first_rollout.json').read_text())
                if gate.get('state') != 'PASS_BEFORE_OPTIMIZER':
                    raise RuntimeError('optimizer reached without first-rollout signed evidence')
                before = [p.detach().clone() for p in self.model.parameters()]
            value = original(*positional, **keyword)
            self.pilot_optimizer_updates += 1
            if self.pilot_optimizer_updates == 1:
                changed = any(not torch.equal(old, new.detach()) for old, new in zip(before, self.model.parameters()))
                atomic_json(self.pilot_status_path.parent / 'first_optimizer_update.json',
                    dict(optimizer_updates=1, parameters_changed=changed, epoch=self.epoch_num,
                         sampled_transitions=self.pilot_raw_env.audit_counts['controls'] * self.pilot_raw_env.num_envs))
                print(f'PILOT_FIRST_OPTIMIZER_UPDATE parameters_changed={changed}', flush=True)
                if not changed:
                    raise RuntimeError('first optimizer step did not change model parameters')
            return value
        self.optimizer.step = observed_step

    def update_epoch(self):
        if getattr(self, "pilot_enabled", False):
            elapsed = time.monotonic() - self.pilot_started_monotonic
            reason = pilot_stop_reason(self.epoch_num, elapsed)
            if (self.pilot_status_path.parent / 'stop_and_save.json').is_file():
                reason = 'support_delivery_capacity_stop'
            if reason is not None:
                checkpoint = self.nn_dir / f"last_pilot_epoch_{self.epoch_num}.pth"
                self.pilot_raw_env.close_audit()
                self.save(checkpoint)
                atomic_json(
                    self.pilot_status_path,
                    {
                        "state": "completed",
                        "reason": reason,
                        "epoch": self.epoch_num,
                        "transitions": self.epoch_num * PILOT_NUM_ENVS * PILOT_HORIZON,
                        "wall_seconds": elapsed,
                        "checkpoint": str(checkpoint.resolve()),
                    },
                )
                raise PilotComplete
        if args.diagnostic_stop_epoch is not None and self.epoch_num >= args.diagnostic_stop_epoch:
            saved = list(self.nn_dir.glob(f"ep_{self.epoch_num}_rew_*.pth"))
            if not saved:
                raise RuntimeError("Refusing to finish diagnostic without its boundary checkpoint")
            if self.writer is not None:
                self.writer.flush()
            print(f"RIGHT_ONLY_DIAGNOSTIC_COMPLETE epoch={self.epoch_num} checkpoint={saved[0]}", flush=True)
            raise DiagnosticComplete
        epoch = super().update_epoch()
        if getattr(self, "pilot_enabled", False):
            self.pilot_raw_env.audit_epoch = epoch
        return epoch

    def write_stats(self, *positional, **keyword) -> None:
        super().write_stats(*positional, **keyword)
        if self.writer is None:
            return
        frame = int(keyword["frame"])
        for name, value in self.env.last_extras.items():
            if torch.is_tensor(value) and value.numel() == 1:
                self.writer.add_scalar(name, float(value.item()), frame)
            elif isinstance(value, (int, float)):
                self.writer.add_scalar(name, float(value), frame)


def _load_official_configs() -> tuple[dict, dict]:
    config_path = _HERE / "ppo_h2s2r_lstm.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    network = values["network"]
    ppo = values["ppo"]
    ppo["num_actors"] = args.num_envs
    ppo["device"] = args.device

    rollout_size = args.num_envs * int(ppo["horizon_length"])
    official_minibatch = int(ppo["minibatch_size"])
    if args.smoke_test:
        ppo["minibatch_size"] = rollout_size
        ppo["max_epochs"] = 1
        ppo["max_frames"] = -1
        ppo["save_frequency"] = 1
        ppo["save_best_after"] = 0
    elif rollout_size < official_minibatch or rollout_size % official_minibatch:
        raise ValueError(
            "V13 formal runs require minibatch_size=4096; "
            "num_envs*horizon_length must be a positive multiple of 4096 "
            f"(got {args.num_envs}*{ppo['horizon_length']}={rollout_size})"
        )

    if args.preflight:
        ppo["max_epochs"] = 1
        ppo["max_frames"] = -1
        ppo["save_frequency"] = 1
        ppo["save_best_after"] = 0

    if args.training_pilot:
        validate_v13_batch(args.num_envs, ppo)
        if int(ppo["max_epochs"]) != 2_000_000:
            raise ValueError("training_pilot must preserve the original max_epochs schedule")
        ppo["save_frequency"] = 100

    if args.max_agent_steps is not None:
        ppo["max_frames"] = int(args.max_agent_steps)
        ppo["max_epochs"] = -1
    elif args.max_epochs is not None:
        ppo["max_epochs"] = int(args.max_epochs)
        ppo["max_frames"] = -1
    return network, ppo


def _main() -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    network_values, ppo_values = _load_official_configs()
    network_config = dict_to_dataclass(network_values, NetworkConfig)
    ppo_config = dict_to_dataclass(ppo_values, PpoConfig)
    if args.diagnostic_stop_epoch is not None and args.diagnostic_stop_epoch % ppo_config.save_frequency:
        raise ValueError("diagnostic_stop_epoch must be a checkpoint boundary")

    env_cfg = build_cfg(
        bundle_root=args.bundle_root,
        num_envs=args.num_envs,
        seed=args.seed,
        input_regime=args.input_regime,
        input_archive=args.input_archive,
        provenance_manifest=args.provenance_manifest,
        right_synergy_npz=args.right_synergy_npz,
        left_synergy_npz=args.left_synergy_npz,
        allow_analytic_synergy=args.allow_analytic_synergy,
        reference_start_index=args.reference_start_index,
        joint_progress_reward_weight=args.joint_progress_reward_weight,
        joint_success_reward_weight=args.joint_success_reward_weight,
    )
    if env_cfg.warmup_clamp_steps != 0:
        raise RuntimeError("Run11 requires object hold 0 from the first training step")
    env_cfg.sim.device = args.device

    log_root = args.pilot_output_root / "logs" if args.training_pilot else Path("logs")
    log_dir = log_root / args.name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=False)
    input_path = Path(env_cfg.perception_npz)
    declared_inputs = load_inputs(
        input_path,
        regime=args.input_regime,
        provenance_manifest=args.provenance_manifest,
    )
    selected_source_time_s = float(
        declared_inputs.source_times_s[args.reference_start_index]
    )
    manifest = {
        "schema": "h2s2r_pour17_right_only_hold0_v13_256",
        "formal_run": bool(args.training_pilot),
        "run_kind": (
            "training_pilot" if args.training_pilot else "smoke" if args.smoke_test else "preflight" if args.preflight else "right_only_diagnostic"
        ),
        "method": "H2S2R right-bottle-only embodiment diagnostic",
        "parent_source": "rl_correction_h2s2r_adapter_v8_centered_workspace",
        "train_from_scratch": True,
        "workspace_bounds": workspace_record(),
        "physics": physics_record(),
        "world_version": "Run11 black table + uniform 0.5kg/mu1/PD + right-centered workspace + hold0",
        "training_warmup_clamp_steps": env_cfg.warmup_clamp_steps,
        "right_zero_margin_patch": _PATCH_RECORD,
        "training_pilot_budget": (
            {
                "max_epochs": PILOT_MAX_EPOCHS,
                "max_transitions": PILOT_MAX_EPOCHS * PILOT_NUM_ENVS * PILOT_HORIZON,
                "wall_seconds": PILOT_WALL_SECONDS,
                "save_reserve_seconds": PILOT_SAVE_RESERVE_SECONDS,
                "configured_ppo_max_epochs_preserved": ppo_config.max_epochs,
                "checkpoint_frequency_epochs": ppo_config.save_frequency,
                "num_envs": PILOT_NUM_ENVS,
                "horizon_length": PILOT_HORIZON,
                "minibatch_size": PILOT_MINIBATCH,
                "mini_epochs": 4,
            }
            if args.training_pilot else None
        ),
        "diagnostic_stop_epoch": args.diagnostic_stop_epoch,
        "diagnostic_source_sha256": {
            name: _sha256(_HERE / name)
            for name in (
                "env.py", "cfg.py", "right_env.py", "train_right_lstm.py",
                "right_zero_margin_install.py", "training_pilot_audit.py",
                "test_right_only.py", "right_workspace.py", "right_workspace_bounds.json",
                "test_workspace_centering.py",
                "ours_physics.py", "ours_physics_spec.json",
            )
        },
        "isolation": {
            "controlled_side": "right", "controlled_object_id": 1,
            "policy_action_dim": 11, "policy_observation_dim": 171,
            "left": "fixed reset joint position targets; no policy/controller step",
            "cup": "unchanged dynamic physics; excluded from reward/clock/success/reset",
            "success": "reference-end final-target error <0.05m; diagnostic only",
            "clock": (
                "released H2S2R floating clock: 30Hz reference, per-episode "
                "speed U[0.5,1.0], update iff bottle error<=0.2m; no warmup freeze"
            ),
            "not_a_bimanual_benchmark": True,
        },
        "termination": {
            "reference_end": True,
            "fingertips_far_m": env_cfg.fingertips_close_threshold_m,
            "object_center_below_table_m": 0.05,
            "training_horizon_steps": env_cfg.training_horizon_steps,
            "training_horizon_rule": "int(reference_length_frames / 0.5)",
            "reference_length_frames": env_cfg.reference_length_frames,
            "evaluation_horizon_steps": env_cfg.evaluation_horizon_steps,
        },
        "diagnostics": [
            "reset reason rates", "episode step mean", "float reference index",
            "reference speed factor", "warmup frozen rate", "tracking reward/error",
            "fingertip distance", "object height from reset",
        ],
        "official_ppo": {
            "upstream": "https://github.com/tylerlum/human2sim2robot",
            "commit": _UPSTREAM_COMMIT,
            "config_sha256": _sha256(_HERE / "ppo_h2s2r_lstm.yaml"),
            "network": dataclasses.asdict(network_config),
            "ppo": dataclasses.asdict(ppo_config),
        },
        "input_regime": args.input_regime,
        "input_archive": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "input_timebase": {
            "fps": declared_inputs.reconstruction_fps,
            "row_count": len(declared_inputs.frame_ids),
            "first_frame_id": int(declared_inputs.frame_ids[0]),
            "last_frame_id": int(declared_inputs.frame_ids[-1]),
            "source_last_frame_timestamp_s": float(
                declared_inputs.source_times_s[-1]
            ),
            "selected_start_timestamp_s": selected_source_time_s,
            "selected_reference_duration_s": float(
                declared_inputs.source_times_s[-1] - selected_source_time_s
            ),
            "frame_time_mapping": declared_inputs.metadata.get(
                "frame_time_mapping", "unspecified"
            ),
            "reference_hz": 30.0,
            "control_hz": 20.0,
            "clock_step": "speed_factor * control_dt / reference_dt",
            "speed_factor_range": [0.5, 1.0],
        },
        "reference_start_index": args.reference_start_index,
        "reference_start_selection": "right-specific manual interaction start",
        "reset": {
            "regime": env_cfg.reset_regime,
            "ours_equivalent_entry": "g2",
            "randomization": "none",
        },
        "reward": {
            "dense": "unscaled original H2S2R right-only forced-reference tracking term",
            "joint_progress": {
                "condition": "disabled",
                "weight": args.joint_progress_reward_weight,
            },
            "joint_success": {
                "condition": "disabled; right success remains diagnostic only",
                "weight": args.joint_success_reward_weight,
            },
        },
        "excluded_from_method": [
            "ours object confidence reward",
            "ours GraspPose prior",
            "ours hand-trajectory reward",
            "affordance/contact supervision",
        ],
        "seed": args.seed,
        "num_envs": args.num_envs,
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty_files": _git_value("status", "--short").splitlines(),
        },
    }
    (log_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "[H2S2R-LSTM] "
        f"log_dir={log_dir} envs={args.num_envs} start_idx={args.reference_start_index} "
        f"train_horizon={env_cfg.training_horizon_steps} "
        f"run_kind={manifest['run_kind']} formal={manifest['formal_run']}"
    )

    print("[H2S2R-LSTM] stage=construct_env begin", flush=True)
    try:
        if args.training_pilot:
            env_cfg.sim.render_interval = env_cfg.decimation
            env_raw = AuditedRightBottleEnv(
                env_cfg,
                render_mode="rgb_array",
                audit_root=log_dir / "pilot_audit",
                controls_per_chunk=160,
                video_epochs=[1, *range(100, PILOT_MAX_EPOCHS + 1, 100)],
            )
        else:
            env_raw = Pour17H2S2REnv(env_cfg)
    except BaseException as error:
        print(
            "[H2S2R-LSTM] stage=construct_env failed "
            f"type={type(error).__name__} repr={error!r}",
            flush=True,
        )
        traceback.print_exc()
        raise
    print("[H2S2R-LSTM] stage=construct_env complete", flush=True)
    verify_installed_right_fabric(
        env_raw.controller.right.fabric, _EXPECTED_RIGHT_FABRIC_CLASS
    )
    limit_dispatch = verify_limit_dispatch(env_raw.controller.right.fabric, env_raw.controller.left.fabric, _EXPECTED_RIGHT_FABRIC_CLASS)
    atomic_json(log_dir / 'limit_dispatch.json', limit_dispatch)
    (log_dir / "runtime_physics.json").write_text(
        json.dumps(env_raw.physics_runtime_record, indent=2) + "\n",
        encoding="utf-8",
    )
    assert env_raw.physics_runtime_record["spec"] == manifest["physics"]
    assert env_raw.controller.right.workspace_bounds_record == manifest['workspace_bounds']
    print('[RUN08] right-centered workspace + uniform physics + official clock verified', flush=True)
    assert env_raw._actions.shape == (args.num_envs, 11)
    initial_left_fabric_q = env_raw.controller.left.q.clone()
    gymnasium_env = GymStyleEnvWrapper(
        env_raw, clip_actions=env_cfg.clip_actions
    )
    print("[H2S2R-LSTM] stage=gymnasium_wrapper complete", flush=True)
    env = OfficialPpoEnvAdapter(gymnasium_env)
    print("[H2S2R-LSTM] stage=official_adapter complete", flush=True)
    agent = observation.call('policy.construct', LoggedPpoAgent,
        experiment_dir=log_dir,
        ppo_config=ppo_config,
        network_config=network_config,
        env=env,
    )
    if args.training_pilot:
        agent.pilot_enabled = True
        agent.pilot_raw_env = env_raw
        prior_elapsed = max(0.0, time.time() - float(os.environ.get('H2S2R_TRAIN_STARTED_UNIX', time.time())))
        agent.pilot_started_monotonic = time.monotonic() - prior_elapsed
        agent.pilot_status_path = log_dir / "pilot_status.json"
        agent.install_update_observation()
        atomic_json(
            agent.pilot_status_path,
            {"state": "running", "epoch": 0, "transitions": 0},
        )
    print("[H2S2R-LSTM] stage=official_agent complete", flush=True)
    if args.resume:
        agent.restore(args.load_path.resolve())
    try:
        print("[H2S2R-LSTM] stage=train begin", flush=True)
        try:
            agent.train()
        except DiagnosticComplete:
            pass
        except PilotComplete:
            pass
        if args.smoke_test or args.preflight:
            assert torch.equal(initial_left_fabric_q, env_raw.controller.left.q)
            assert torch.equal(
                env_raw._target_q[:, env_raw.side_joint_ids[ControlledSide.LEFT]],
                env_raw.controller.left.q,
            )
            print("[RIGHT-ONLY] isolation verified: left fabric unchanged; fixed targets; action_dim=11", flush=True)
    except BaseException as error:
        if args.training_pilot:
            try:
                env_raw.close_audit()
                atomic_json(
                    log_dir / "pilot_status.json",
                    {
                        "state": "failed",
                        "epoch": int(getattr(agent, "epoch_num", 0)),
                        "transitions": env_raw.audit_counts["controls"] * env_raw.num_envs,
                        "completed_epochs": getattr(agent, "pilot_completed_epochs", 0),
                        "optimizer_updates": getattr(agent, "pilot_optimizer_updates", 0),
                        "error": repr(error),
                    },
                )
            except BaseException as preservation_error:
                print(f"PILOT_EVIDENCE_PRESERVATION_FAILED {preservation_error!r}", flush=True)
        raise
    finally:
        if args.training_pilot:
            env_raw.close_audit()
        env.close()


observation.run(_main, app.close)
