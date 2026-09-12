"""Single frozen Clean3 C3-P1R2 PPO/LSTM pilot entry."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
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
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_epochs", type=int, default=512)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.num_envs, args.seed, args.max_epochs) != (256, 42, 512):
    raise SystemExit("C3-P1R2 pilot is frozen at 256 envs, seed42, 512 epochs")
app = AppLauncher(args).app
OFFICIAL = args.v12_root.resolve() / "third_party/h2s2r_official"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(OFFICIAL))
from human2sim2robot.ppo.ppo_agent import PpoAgent, PpoConfig  # noqa: E402
from human2sim2robot.ppo.utils.dict_to_dataclass import dict_to_dataclass  # noqa: E402
from human2sim2robot.ppo.utils.network import NetworkConfig  # noqa: E402
from rl_rebuild.baselines.h2s2r.official_ppo_adapter import OfficialPpoEnvAdapter  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.h2s2r_clean3.cfg import build_cfg  # noqa: E402
from tasks.h2s2r_clean3.env import Clean3H2S2REnv  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class AuditedAgent(PpoAgent):
    """Preserve the latest completed optimizer state and milestone counts."""

    def train_epoch(self):
        value = super().train_epoch()
        completed = int(self.epoch_num)
        if completed == 1 or completed % 8 == 0:
            self.save(self.nn_dir / "latest_complete.pth")
        atomic_json(
            Path(self.experiment_dir) / "training_progress.json",
            {"completed_epochs": completed, "transitions": completed * 256 * 16, "optimizer_updates_lower_bound": completed},
        )
        return value


def main() -> None:
    started = time.monotonic()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with (HERE / "ppo_h2s2r_lstm.yaml").open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    values["ppo"]["num_actors"] = 256
    values["ppo"]["device"] = args.device
    network_cfg = dict_to_dataclass(values["network"], NetworkConfig)
    ppo_cfg = dict_to_dataclass(values["ppo"], PpoConfig)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "h2s2r_clean3_c3p1r2_ppo_v1", "method": "H2S2R-adapted Clean3",
        "seed": 42, "num_envs": 256, "observation_dim": 342, "action_dim": 22,
        "rollout": 16, "minibatch": 4096, "maximum_epochs": 512,
        "maximum_transitions": 2097152, "from_scratch": True,
        "reward": "mean of two unchanged H2S2R tracking rewards",
        "ppo_config": dataclasses.asdict(ppo_cfg), "network_config": dataclasses.asdict(network_cfg),
        "config_sha256": sha256(HERE / "ppo_h2s2r_lstm.yaml"),
    }
    atomic_json(output / "run_manifest.json", manifest)
    cfg = build_cfg(runtime_root=args.runtime_root, v12_root=args.v12_root, num_envs=256, seed=42, external_eval=False)
    cfg.sim.device = args.device
    print("[CLEAN3] stage=construct_env begin", flush=True)
    env_raw = Clean3H2S2REnv(cfg)
    print("[CLEAN3] stage=construct_env complete", flush=True)
    env = OfficialPpoEnvAdapter(GymStyleEnvWrapper(env_raw, clip_actions=cfg.clip_actions))
    agent = AuditedAgent(experiment_dir=output, ppo_config=ppo_cfg, network_config=network_cfg, env=env)
    initial_path = agent.nn_dir / "initial_random.pth"
    agent.save(initial_path)
    atomic_json(output / "initial_checkpoint.json", {"path": str(initial_path), "sha256": sha256(initial_path)})
    try:
        agent.train()
    finally:
        latest = agent.nn_dir / "latest_complete.pth"
        atomic_json(output / "training_exit.json", {
            "elapsed_seconds": time.monotonic() - started,
            "completed_epochs": int(getattr(agent, "epoch_num", 0)),
            "transitions": int(getattr(agent, "epoch_num", 0)) * 256 * 16,
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
