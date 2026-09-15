"""Train the isolated, working Ours Clean3 Stage-1 initializer contract."""

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
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_agent_steps", type=int, default=5_000_000)
parser.add_argument("--smoke_epochs", type=int, default=0)
parser.add_argument("--anneal_ema", type=float, default=0.7)
parser.add_argument("--anneal_sustain", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.headless:
    raise ValueError("Ours Stage-1 training must be headless")

runtime_root = Path(args.runtime_root).resolve()
output_root = Path(args.output_root).resolve()
allowed_root = Path("/ssd/sy/kailang").resolve()
for path in (runtime_root, output_root):
    if allowed_root not in (path, *path.parents):
        raise ValueError(f"path outside /ssd/sy/kailang: {path}")
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


class AnnealingPPO(PPO):
    def __init__(self, *ppo_args, raw_env, **ppo_kwargs):
        super().__init__(*ppo_args, **ppo_kwargs)
        self.raw_env = raw_env
        self.success_ema = 0.0
        self.sustain_windows = 0

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls)
        rates = self.raw_env.pop_rates()
        for name, value in rates.items():
            if value == value:
                self.writer.add_scalar(name, value, self.agent_steps)
        success_rate = rates.get("sr/grasp_success", float("nan"))
        if success_rate == success_rate:
            self.success_ema = 0.95 * self.success_ema + 0.05 * success_rate
        self.sustain_windows = (
            self.sustain_windows + 1
            if self.success_ema >= args.anneal_ema
            else 0
        )
        annealed = False
        if (
            self.sustain_windows >= args.anneal_sustain
            and self.raw_env.release_row_cur > self.raw_env.cfg.release_min_rows
        ):
            self.raw_env.release_row_cur = max(
                self.raw_env.cfg.release_min_rows,
                self.raw_env.release_row_cur - self.raw_env.cfg.release_step_rows,
            )
            self.sustain_windows = 0
            self.success_ema = 0.0
            annealed = True
            print(
                f"[OURS_STAGE1] release_row -> {self.raw_env.release_row_cur} "
                f"@ {self.agent_steps / 1e6:.2f}M",
                flush=True,
            )
        self.writer.add_scalar("curr/ema_success", self.success_ema, self.agent_steps)
        self.writer.add_scalar(
            "curr/release_row_cur", float(self.raw_env.release_row_cur), self.agent_steps
        )
        write_json(
            output_root / "training_progress.json",
            {
                "completed_epochs": self.epoch_num,
                "agent_steps": int(self.agent_steps),
                "release_row_cur": int(self.raw_env.release_row_cur),
                "success_ema": self.success_ema,
                "sustain_windows": self.sustain_windows,
                "annealed_this_epoch": annealed,
                "window_rates": rates,
            },
        )


started = time.monotonic()
try:
    os.environ.setdefault("SHARPA_WANDB", "0")
    config_path = Path(__file__).with_name("ppo_ours_stage1.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["seed"] = args.seed
    config["device"] = "cuda:0"
    algorithm = config["algorithm"]
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
        terminate_on_grasp_drop=True,
        ours_stage1_contract=True,
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
            "schema": "clean3_ours_stage1_initializer_v1",
            "source_commit": "43d5747587619d5538ec4503089da30ff9ccaafc",
            "policy_observation_dim": 367,
            "privileged_dim": 22,
            "action_dim": 58,
            "controller": "bounded cumulative residual",
            "reward": "Ours Clean3 Stage-1 contact/hold/certification",
            "ppo_config_sha256": sha256(config_path),
            "num_envs": args.num_envs,
            "seed": args.seed,
            "max_agent_steps": int(algorithm["max_agent_steps"]),
            "stage2_policy_loaded": False,
        },
    )
    agent = AnnealingPPO(
        env,
        output_dir=str(output_root),
        full_config=ConfigWrapper(config, {}),
        raw_env=raw_env,
    )
    agent.train()
    latest = output_root / "stage1_nn" / "latest_complete.pth"
    agent.save(str(latest.with_suffix("")))
    write_json(
        output_root / "training_exit.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "completed_epochs": agent.epoch_num,
            "agent_steps": int(agent.agent_steps),
            "release_row_cur": int(raw_env.release_row_cur),
            "success_ema": agent.success_ema,
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
