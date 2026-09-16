"""Time-limited Clean3 Stage-2 handoff from the selected Ours Stage-1 policy.

The Stage-1 checkpoint is restored into the same 367D actor / 22D privileged
critic architecture.  Before grasp certification the environment uses the
disclosed Ours startup objective; after certification it switches to the H2S2R
bimanual tracking reward already implemented by ``Clean3H2S2REnv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--runtime_root", required=True)
parser.add_argument("--v12_root", required=True)
parser.add_argument("--output_root", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--checkpoint_sha256", required=True)
parser.add_argument("--allowed_root", default="/ssd/sy/kailang")
parser.add_argument("--robot_usd")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--release_row", type=int, default=50)
parser.add_argument("--max_agent_steps", type=int, default=150_000_000)
parser.add_argument("--smoke_epochs", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.headless:
    raise ValueError("Clean3 Stage-2 training must be headless")

runtime_root = Path(args.runtime_root).resolve()
output_root = Path(args.output_root).resolve()
checkpoint = Path(args.checkpoint).resolve()
allowed_root = Path(args.allowed_root).resolve()
if not allowed_root.is_dir():
    raise FileNotFoundError(f"allowed root does not exist: {allowed_root}")
for path in (runtime_root, output_root, checkpoint):
    if allowed_root not in (path, *path.parents):
        raise ValueError(f"path outside /ssd/sy/kailang: {path}")
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
output_root.mkdir(parents=True, exist_ok=False)

app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(args.v12_root).resolve()))
sys.path.insert(0, str(runtime_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ours_stage1_overlay"))

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.h2s2r_clean3.cfg import build_cfg  # noqa: E402
from tasks.h2s2r_clean3.env import Clean3H2S2REnv  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Stage2PPO(PPO):
    def __init__(self, *ppo_args, raw_env, **ppo_kwargs):
        super().__init__(*ppo_args, **ppo_kwargs)
        self.raw_env = raw_env
        # The vendored PPO names these folders stage1_nn/stage1_tb.  Override the
        # checkpoint target without changing shared PPO code.
        self.nn_dir = str(output_root / "stage2_nn")
        Path(self.nn_dir).mkdir(parents=True, exist_ok=True)

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls)
        rates = self.raw_env.pop_rates()
        for name, value in rates.items():
            if value == value:
                self.writer.add_scalar(name, value, self.agent_steps)
        write_json(
            output_root / "training_progress.json",
            {
                "completed_epochs": self.epoch_num,
                "agent_steps": int(self.agent_steps),
                "release_row": int(self.raw_env.release_row_cur),
                "window_rates": rates,
            },
        )


started = time.monotonic()
try:
    expected_sha = args.checkpoint_sha256.lower()
    actual_sha = sha256(checkpoint)
    if actual_sha != expected_sha:
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    os.environ.setdefault("SHARPA_WANDB", "0")
    config_path = Path(__file__).with_name("ppo_ours_stage1.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["seed"] = args.seed
    config["device"] = "cuda:0"
    algorithm = config["algorithm"]
    algorithm["experiment_name"] = "Clean3_ours_best_to_h2s2r_stage2"
    algorithm["num_actors"] = args.num_envs
    algorithm["minibatch_size"] = min(
        int(algorithm["minibatch_size"]), 32 * args.num_envs
    )
    batch_size = 32 * args.num_envs
    algorithm["max_agent_steps"] = (
        batch_size * (args.smoke_epochs + 1)
        if args.smoke_epochs
        else args.max_agent_steps
    )

    cfg = build_cfg(
        runtime_root=runtime_root,
        v12_root=args.v12_root,
        num_envs=args.num_envs,
        seed=args.seed,
        grasp_curriculum=True,
        release_row_start=args.release_row,
        terminate_on_grasp_drop=True,
        ours_stage2_contract=True,
        robot_usd=args.robot_usd,
    )
    raw_env = Clean3H2S2REnv(cfg)
    env = GymStyleEnvWrapper(raw_env, clip_actions=1.0)
    observation = env.reset()
    if observation["obs"].shape != (args.num_envs, 367):
        raise RuntimeError(f"invalid Ours policy shape: {observation['obs'].shape}")
    if observation["priv_info"].shape != (args.num_envs, 22):
        raise RuntimeError(
            f"invalid Ours privileged shape: {observation['priv_info'].shape}"
        )
    write_json(
        output_root / "run_manifest.json",
        {
            "schema": "clean3_time_limited_best_to_stage2_v1",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_sha,
            "policy_observation_dim": 367,
            "privileged_dim": 22,
            "action_dim": 58,
            "controller": "bounded cumulative residual",
            "reward_switch": "Ours grasp startup then H2S2R bimanual tracking",
            "release_row": args.release_row,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "max_agent_steps": int(algorithm["max_agent_steps"]),
            "optimizer_state_restored": False,
            "strict_stage1_gate_passed": False,
            "allowed_root": str(allowed_root),
            "robot_usd": str(Path(args.robot_usd).resolve()) if args.robot_usd else None,
        },
    )
    agent = Stage2PPO(
        env,
        output_dir=str(output_root),
        full_config=ConfigWrapper(config, {}),
        raw_env=raw_env,
    )
    agent.restore_train(str(checkpoint))
    agent.train()
    latest = Path(agent.nn_dir) / "latest_complete.pth"
    agent.save(str(latest.with_suffix("")))
    write_json(
        output_root / "training_exit.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "completed_epochs": agent.epoch_num,
            "agent_steps": int(agent.agent_steps),
            "latest_complete": str(latest),
            "latest_complete_sha256": sha256(latest),
        },
    )
except Exception as error:
    write_json(
        output_root / "entry_failure.json",
        {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )
    raise
finally:
    app.close()
