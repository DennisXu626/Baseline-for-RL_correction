"""Development-only SB3 PPO closed-loop runner for Pour17 WP1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from isaaclab.app import AppLauncher


PROCESS_STARTED = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument("--bundle-root", type=Path, required=True)
parser.add_argument("--robot-urdf", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--resolved-config", type=Path, required=True)
parser.add_argument("--schema", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=1701)
parser.add_argument("--n-steps", type=int, default=128)
parser.add_argument("--total-timesteps", type=int, default=8192)
parser.add_argument("--resume", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .env import Pour17JointEnv
from .geometry import ndarray_sha256, sha256
from .sb3_vecenv import IsaacSB3VecEnv
from .scene import isaac_scene_config


class RolloutAudit(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rollouts: list[dict] = []
        self._rollout_started = 0.0

    def _on_rollout_start(self) -> None:
        self._rollout_started = time.perf_counter()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        rewards = self.model.rollout_buffer.rewards
        self.rollouts.append({
            "index": len(self.rollouts),
            "global_transitions": int(rewards.size),
            "reward_min": float(np.min(rewards)),
            "reward_mean": float(np.mean(rewards)),
            "reward_max": float(np.max(rewards)),
            "reward_finite": bool(np.isfinite(rewards).all()),
            "observation_finite": bool(np.isfinite(
                self.model.rollout_buffer.observations).all()),
            "collection_s": time.perf_counter() - self._rollout_started,
        })


class GpuMemoryMonitor:
    """Poll this process's nvidia-smi allocation without touching other jobs."""
    def __init__(self):
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            try:
                output = subprocess.check_output([
                    "nvidia-smi", "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ], text=True, timeout=5)
                for line in output.splitlines():
                    pid, memory = (part.strip() for part in line.split(",", 1))
                    if int(pid) == os.getpid():
                        self.peak_mib = max(self.peak_mib, int(memory))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6.0)


def parameter_vector(model: PPO) -> torch.Tensor:
    return torch.cat([value.detach().flatten().cpu()
                      for value in model.policy.parameters()])


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def build_env() -> tuple[Pour17JointEnv, IsaacSB3VecEnv]:
    cfg = isaac_scene_config(args.bundle_root, num_envs=args.num_envs, seed=args.seed)
    cfg.sim.device = args.device
    cfg.sim.log_dir = str((args.output_dir / "isaaclab_logs").resolve())
    cfg.bundle_root = str(args.bundle_root.resolve())
    cfg.reference_path = str(args.reference.resolve())
    cfg.robot_urdf = str(args.robot_urdf.resolve())
    cfg.probe_only = False
    cfg.external_evaluator_controls_termination = False
    cfg.curriculum_stage = 0
    raw = Pour17JointEnv(cfg)
    return raw, IsaacSB3VecEnv(raw)


def main() -> int:
    if args.total_timesteps > 2_000_000:
        raise ValueError("WP1 development cap is 2,000,000 transitions")
    if args.total_timesteps % (args.num_envs * args.n_steps):
        raise ValueError("total timesteps must contain complete PPO rollouts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.resolved_config, args.output_dir / "resolved_config.yaml")
    shutil.copy2(args.schema, args.output_dir / "observation_action_schema.json")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    monitor = GpuMemoryMonitor()
    monitor.start()
    environment_start = time.perf_counter()
    raw, env = build_env()
    startup_complete = time.perf_counter()
    try:
        if args.resume is not None:
            model = PPO.load(args.resume, env=env, device=args.device)
            resumed_from = {"path": str(args.resume.resolve()), "sha256": sha256(args.resume)}
        else:
            model = PPO(
                "MlpPolicy", env, seed=args.seed, device=args.device,
                gamma=0.95, gae_lambda=0.95, learning_rate=1e-5,
                ent_coef=0.001, vf_coef=0.5, clip_range=0.2,
                n_steps=args.n_steps, batch_size=256, n_epochs=5,
                policy_kwargs={
                    "net_arch": {"pi": [256, 128], "vf": [256, 128]},
                    "log_std_init": -1.60,
                    "activation_fn": torch.nn.Tanh,
                },
                verbose=1,
            )
            # Upstream warm_start_mean: begin near the canonical joint target.
            home = raw.reset_q.cpu().numpy()
            lower = raw.lower.cpu().numpy()
            upper = raw.upper.cpu().numpy()
            home_action = np.clip(2.0 * (home - lower) / (upper - lower) - 1.0, -1.0, 1.0)
            with torch.no_grad():
                model.policy.action_net.bias.copy_(
                    torch.tensor(home_action, dtype=torch.float32, device=model.device))
            resumed_from = None
        before = parameter_vector(model)
        reset_start = time.perf_counter()
        fixed_observation = env.reset().copy()
        explicit_reset_s = time.perf_counter() - reset_start
        initial_action, _ = model.predict(fixed_observation, deterministic=True)
        callback = RolloutAudit()
        training_start = time.perf_counter()
        model.learn(
            total_timesteps=args.total_timesteps, callback=callback,
            reset_num_timesteps=(args.resume is None), progress_bar=False)
        training_elapsed = time.perf_counter() - training_start
        after = parameter_vector(model)
        parameter_change = float(torch.linalg.vector_norm(after - before))
        model_path = args.output_dir / "final_model.zip"
        model.save(model_path)
        completed_steps = int(model.num_timesteps)
        (args.output_dir / "completed_steps.txt").write_text(
            f"{completed_steps}\n", encoding="utf-8")
        np.save(args.output_dir / "reload_fixed_observation.npy", fixed_observation)
        torch.save({
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "seed": args.seed, "completed_steps": completed_steps,
            "curriculum_stage": raw.stage,
        }, args.output_dir / "rng_state.pt")

        reloaded = PPO.load(model_path, env=env, device=args.device)
        reloaded_action, _ = reloaded.predict(fixed_observation, deterministic=True)
        reload_action_max_abs = float(np.max(np.abs(reloaded_action - model.predict(
            fixed_observation, deterministic=True)[0])))
        episode_start = time.perf_counter()
        episode_observation = env.reset()
        episode_reward = np.zeros(args.num_envs, dtype=np.float64)
        episode_steps = 0
        episode_done = np.zeros(args.num_envs, dtype=bool)
        while episode_steps < 903 and not episode_done[0]:
            action, _ = reloaded.predict(episode_observation, deterministic=True)
            episode_observation, reward, done, infos = env.step(action)
            episode_reward += reward
            episode_done |= done
            episode_steps += 1
        reload_episode_s = time.perf_counter() - episode_start
        monitor.stop()
        logger_values = {
            key: float(value) for key, value in model.logger.name_to_value.items()
            if key.startswith("train/") and isinstance(value, (int, float, np.number))
        }
        losses_finite = bool(logger_values) and all(
            np.isfinite(value) for value in logger_values.values())
        result = {
            "status": "PASS" if (
                len(callback.rollouts) >= 2
                and all(row["reward_finite"] and row["observation_finite"]
                        for row in callback.rollouts)
                and losses_finite
                and parameter_change > 0.0 and reload_action_max_abs <= 1e-6
            ) else "FAIL",
            "development_only": True,
            "seed": args.seed, "num_envs": args.num_envs,
            "n_steps_per_env": args.n_steps,
            "requested_global_transitions": args.total_timesteps,
            "completed_steps": completed_steps,
            "rollouts": callback.rollouts,
            "parameter": {
                "before_sha256": tensor_sha256(before),
                "after_sha256": tensor_sha256(after),
                "change_l2": parameter_change,
            },
            "losses": logger_values,
            "losses_finite": losses_finite,
            "reload": {
                "model": {"path": str(model_path), "sha256": sha256(model_path)},
                "deterministic_action_max_abs": reload_action_max_abs,
                "tolerance": 1e-6,
                "real_physics_episode_steps": episode_steps,
                "real_physics_episode_return_env0": float(episode_reward[0]),
                "real_physics_episode_completed": bool(episode_done[0]),
                "simulator_bit_state_restored": False,
            },
            "timing": {
                "kit_and_import_startup_s": environment_start - PROCESS_STARTED,
                "environment_build_s": startup_complete - environment_start,
                "training_s": training_elapsed,
                "rollout_collection_s": float(sum(
                    row["collection_s"] for row in callback.rollouts)),
                "ppo_update_and_overhead_s": float(training_elapsed - sum(
                    row["collection_s"] for row in callback.rollouts)),
                "explicit_reset_s": explicit_reset_s,
                "reload_physics_episode_s": reload_episode_s,
                "global_transitions_per_s": args.total_timesteps / training_elapsed,
                "process_peak_gpu_memory_mib": monitor.peak_mib,
            },
            "inputs": {
                "reference": {"path": str(args.reference.resolve()), "sha256": sha256(args.reference)},
                "robot_urdf": {"path": str(args.robot_urdf.resolve()), "sha256": sha256(args.robot_urdf)},
                "resolved_config": {"path": str(args.resolved_config.resolve()), "sha256": sha256(args.resolved_config)},
                "observation_action_schema": {"path": str(args.schema.resolve()), "sha256": sha256(args.schema)},
            },
            "resume": resumed_from,
        }
        result_path = args.output_dir / "ppo_update_result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "artifact": str(result_path),
                          "steps": completed_steps,
                          "reload_action_max_abs": reload_action_max_abs}, indent=2))
        return 0 if result["status"] == "PASS" else 1
    finally:
        monitor.stop()
        env.close()


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
