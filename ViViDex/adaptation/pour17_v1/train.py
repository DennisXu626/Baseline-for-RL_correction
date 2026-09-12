"""WP2 SB3 PPO runner; formal capacity requires separate reviewer approval."""

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
import sys
import random
import yaml

from isaaclab.app import AppLauncher


PROCESS_STARTED = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument("--bundle-root", type=Path, required=True)
parser.add_argument("--robot-urdf", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--root-overlay", type=Path)
parser.add_argument("--resolved-config", type=Path, required=True)
parser.add_argument("--schema", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=1701)
parser.add_argument("--n-steps", type=int, default=128)
parser.add_argument("--total-timesteps", type=int, default=8192)
parser.add_argument("--resume", type=Path)
parser.add_argument("--development-extension-approved", action="store_true")
parser.add_argument("--development-curriculum-approved", action="store_true")
parser.add_argument("--formal-approved", action="store_true", help="Only supply after reviewer launch approval; never used by WP2 package")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (not args.formal_approved and not args.development_extension_approved
        and not args.development_curriculum_approved and args.total_timesteps > 65_536):
    parser.error("WP2 development total cannot exceed65,536 transitions")
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
from .curriculum import CurriculumState
from .train_state import capture as capture_state, restore as restore_state


class RolloutAudit(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rollouts: list[dict] = []
        self._rollout_started = 0.0

    def _on_rollout_start(self) -> None:
        self._rollout_started = time.perf_counter()
        self.episodes = []

    def _on_step(self) -> bool:
        self.episodes.extend({k: v for k, v in info.items()
                              if k != "terminal_observation"}
                             for info in self.locals["infos"] if "episode" in info)
        elapsed = (time.perf_counter() - self.model.wp2_started
                   - getattr(self.model, "wp2_excluded_eval_s", 0.0))
        return args.formal_approved or elapsed < self.model.wp2_wall_allowance

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
            "completed_collection_steps": int(self.num_timesteps),
            "episodes": self.episodes,
        })
        (args.output_dir / "rollout_windows.json").write_text(json.dumps(self.rollouts, indent=2) + "\n")


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
    if args.root_overlay:
        from .wp4_world import apply_root_overlay
        apply_root_overlay(cfg, args.bundle_root, args.root_overlay)
    raw = Pour17JointEnv(cfg)
    return raw, IsaacSB3VecEnv(raw)


def main() -> int:
    if sum((args.formal_approved, args.development_extension_approved,
            args.development_curriculum_approved)) > 1:
        raise ValueError("formal and development authorization modes are mutually exclusive")
    if args.development_extension_approved and (
            args.total_timesteps != 262_144 or args.resume is None or args.seed != 1701
            or args.num_envs != 32 or args.n_steps != 128):
        raise ValueError("DEV_EXTEND_262144_V1 requires resume, seed1701,32env,n_steps128,target262144")
    if args.development_curriculum_approved and (
            args.total_timesteps != 1_003_520 or args.resume is None or args.seed != 1701
            or args.num_envs != 32 or args.n_steps != 128):
        raise ValueError("DEV_TO_FIRST_CURRICULUM_V1 requires resume, seed1701,32env,n_steps128,target1003520")
    cap = (3_002_368 if args.formal_approved else
           1_003_520 if args.development_curriculum_approved else
           262_144 if args.development_extension_approved else 65_536)
    if args.total_timesteps > cap:
        raise ValueError("WP2 development cap 65,536; formal budget 3,002,368 only after reviewer approval")
    if args.formal_approved and (args.seed, args.num_envs, args.n_steps, args.total_timesteps) != (0, 32, 128, 3_002_368):
        raise ValueError("formal seed0/32x128/3,002,368 frozen capacity contract mismatch")
    if args.total_timesteps % (args.num_envs * args.n_steps):
        raise ValueError("total timesteps must contain complete PPO rollouts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_dir.parent / "development_training_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    reserved = sum(r["reserved_steps"] for r in ledger)
    used_wall = sum(r.get("training_wall_s", 0.) for r in ledger)
    reserved_steps = (741_376 if args.development_curriculum_approved else
                      204_800 if args.development_extension_approved else args.total_timesteps)
    if args.development_extension_approved and reserved != 65_536:
        raise RuntimeError(f"extension requires preserved prior ledger of65,536 steps, got{reserved}")
    if args.development_curriculum_approved and reserved != 270_336:
        raise RuntimeError(f"curriculum extension requires preserved prior ledger of270,336 steps, got{reserved}")
    if (not args.formal_approved and not args.development_extension_approved
            and not args.development_curriculum_approved and (
            reserved + reserved_steps > 65_536 or used_wall >= 3600)):
        raise RuntimeError("WP2 cumulative development training budget exhausted")
    authorization = ("DEV_TO_FIRST_CURRICULUM_V1" if args.development_curriculum_approved else
                     "DEV_EXTEND_262144_V1" if args.development_extension_approved else
                     "original_development")
    ledger_row = {"output": str(args.output_dir), "reserved_steps": reserved_steps,
                  "authorization": authorization,
                  "state": "RESERVED_COUNTED_UNTIL_MEASURED", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if not args.formal_approved:
        ledger.append(ledger_row)
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    resolved = yaml.safe_load(args.resolved_config.read_text())
    if resolved["num_envs"] != args.num_envs or resolved["n_steps"] != args.n_steps:
        raise ValueError("run env/rollout size does not match registered WP2 config")
    from .schema import OBSERVATION_DIM
    if resolved["observation_dim"] != OBSERVATION_DIM or json.loads(args.schema.read_text())["observation_dim"] != OBSERVATION_DIM:
        raise ValueError("resolved schema width mismatch")
    fixed_ppo = dict(gamma=.95, gae_lambda=.95, learning_rate=1e-5, ent_coef=.001,
                     vf_coef=.5, clip_range=.2, batch_size=256, n_epochs=5, log_std_init=-1.60)
    if any(resolved[k] != value for k, value in fixed_ppo.items()):
        raise ValueError("WP2 config differs from the frozen PPO parameters")
    resolved.update(seed=args.seed, total_timesteps=args.total_timesteps, reference=str(args.reference.resolve()),
                    formal_approved=args.formal_approved, input_config_sha256=sha256(args.resolved_config))
    (args.output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    shutil.copy2(args.schema, args.output_dir / "observation_action_schema.json")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    monitor = GpuMemoryMonitor()
    monitor.start()
    environment_start = time.perf_counter()
    raw, env = build_env()
    binding = None
    if args.root_overlay:
        from .development_binding import identity, readback
        binding = identity(args.bundle_root, args.reference, args.robot_urdf, args.root_overlay)
    startup_complete = time.perf_counter()
    curriculum = CurriculumState()
    next_validation = args.num_envs * args.n_steps
    curriculum_events = []
    fixed_diagnostic_events = []
    resume_start_steps = 0
    class UpdatedPPO(PPO):
        def train(self):
            nonlocal next_validation
            super().train()
            self.last_complete_update_steps = int(self.num_timesteps)
            # Preserve a complete update even if the following validation fails.
            self.save(args.output_dir / "latest_updated_model.zip")
            torch.save(capture_state(raw, curriculum, self.num_timesteps, next_validation), args.output_dir / "latest_training_state.pt")
            if self.num_timesteps == 8192:
                self.save(args.output_dir / "checkpoint_8192.zip")
                torch.save(capture_state(raw, curriculum, self.num_timesteps, next_validation), args.output_dir / "checkpoint_8192_state.pt")
            (args.output_dir / "progress.json").write_text(json.dumps({
                "completed_steps": int(self.num_timesteps), "stage": curriculum.stage,
                "training_wall_s": time.perf_counter() - self.wp2_started - self.wp2_excluded_eval_s,
                "curriculum_events": curriculum_events,
                "fixed_diagnostic_events": fixed_diagnostic_events}, indent=2) + "\n")
            if args.development_curriculum_approved and self.num_timesteps == 524_288:
                checkpoint = args.output_dir / "checkpoint_524288.zip"
                self.save(checkpoint)
                event = {"at_completed_update_steps": int(self.num_timesteps), "commands": []}
                for mode in ("stochastic", "deterministic"):
                    directory = args.output_dir.parent / f"eval_midpoint_{mode}"
                    command = [sys.executable, "-m", "adaptation.pour17_v1.policy_diagnostic_eval",
                        "--bundle-root", str(args.bundle_root), "--robot-urdf", str(args.robot_urdf),
                        "--reference", str(args.reference), "--root-overlay", str(args.root_overlay),
                        "--model", str(checkpoint), "--output-dir", str(directory),
                        "--mode", mode, "--expected-steps", str(self.num_timesteps),
                        "--device", args.device, "--headless"]
                    started = time.perf_counter()
                    with (args.output_dir.parent / f"eval_midpoint_{mode}.log").open("w") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       check=True, timeout=1200)
                    elapsed = time.perf_counter() - started
                    self.wp2_excluded_eval_s += elapsed
                    event["commands"].append({"mode": mode, "wall_s": elapsed,
                                               "output": str(directory)})
                event["wall_s"] = sum(row["wall_s"] for row in event["commands"])
                if event["wall_s"] > 1200:
                    raise RuntimeError(f"midpoint fixed diagnostic wall cap exceeded: {event['wall_s']}")
                fixed_diagnostic_events.append(event)
                (args.output_dir.parent / "midpoint_diagnostic_event.json").write_text(
                    json.dumps(event, indent=2) + "\n")
            if self.num_timesteps < next_validation:
                return
            directory = args.output_dir / f"internal_eval_{self.num_timesteps}"
            directory.mkdir(exist_ok=False)
            checkpoint = directory / "updated_model.zip"
            self.save(checkpoint)
            command = [sys.executable, "-m", "adaptation.pour17_v1.internal_eval",
                "--bundle-root", str(args.bundle_root), "--robot-urdf", str(args.robot_urdf),
                "--reference", str(args.reference), "--model", str(checkpoint),
                "--output-dir", str(directory), "--stage", str(curriculum.stage),
                "--device", args.device, "--headless"]
            if args.root_overlay:
                command += ["--root-overlay", str(args.root_overlay)]
            with (directory / "console.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            measured = json.loads((directory / "internal_eval.json").read_text())
            prior = curriculum.stage
            changed = curriculum.update(measured["pregrasp_successes"])
            raw.set_curriculum_stage(curriculum.stage)
            million = (self.num_timesteps // 1_000_000 + 1) * 1_000_000
            rollout = args.num_envs * args.n_steps
            next_validation = int(np.ceil(million / rollout)) * rollout
            curriculum_events.append({"at_completed_update_steps": self.num_timesteps,
                "before_stage": prior, "after_stage": curriculum.stage, "changed": changed,
                "mean": float(np.mean(measured["pregrasp_successes"])), "next_validation": next_validation,
                "evidence": str(directory / "internal_eval.json")})
            # Both stage settings change together; next child gets this stage.
            self.save(args.output_dir / "latest_updated_model.zip")
            torch.save(capture_state(raw, curriculum, self.num_timesteps, next_validation), args.output_dir / "latest_training_state.pt")
    try:
        if args.resume is not None:
            model = UpdatedPPO.load(args.resume, env=env, device=args.device)
            if (getattr(model, "wp2_reference_sha256", None) != sha256(args.reference)
                    or getattr(model, "wp2_schema_sha256", None) != sha256(args.schema)):
                raise RuntimeError("resume reference/schema identity mismatch; WP1 is incompatible")
            if int(model.seed) != args.seed:
                raise RuntimeError("resume seed differs; development checkpoints cannot seed a formal run")
            state_name = "latest_training_state.pt" if args.resume.name == "latest_updated_model.zip" else "training_state.pt"
            state = torch.load(args.resume.with_name(state_name), weights_only=False, map_location="cpu")
            curriculum, next_validation = restore_state(raw, state)
            if int(model.num_timesteps) != state["completed_steps"]:
                raise RuntimeError("checkpoint/state transition count mismatch")
            resume_start_steps = int(model.num_timesteps)
            if args.development_extension_approved and (resume_start_steps != 57_344
                    or curriculum.stage != 0 or next_validation != 1_003_520):
                raise RuntimeError("DEV_EXTEND_262144_V1 resume state mismatch")
            if args.development_curriculum_approved and (resume_start_steps != 262_144
                    or curriculum.stage != 0 or next_validation != 1_003_520):
                raise RuntimeError("DEV_TO_FIRST_CURRICULUM_V1 resume state mismatch")
            old_binding = getattr(model, "development_binding", None)
            from .development_binding import compatible_resume
            compatibility = compatible_resume(old_binding, binding)
            if not compatibility["compatible"]:
                raise RuntimeError(f"resume world/method identity mismatch: {compatibility}")
            resumed_from = {"path": str(args.resume.resolve()), "sha256": sha256(args.resume)}
        else:
            model = UpdatedPPO(
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
            warm_action = home_action.copy()
            warm_action[14:] = np.clip(warm_action[14:] + .2, -1., 1.)
            with torch.no_grad():
                model.policy.action_net.bias.copy_(
                    torch.tensor(warm_action, dtype=torch.float32, device=model.device))
            resumed_from = None
        before = parameter_vector(model)
        initial_bias = model.policy.action_net.bias.detach().cpu().tolist()
        model.wp2_reference_sha256 = sha256(args.reference)
        model.wp2_schema_sha256 = sha256(args.schema)
        model.development_binding = binding
        reset_start = time.perf_counter()
        fixed_observation = env.reset().copy()
        if args.resume is not None:
            # PPO.load has no live simulator observation. Bind its next rollout
            # to this explicit new-simulator reset; simulator bit-state is not restored.
            model._last_obs = fixed_observation
            model._last_episode_starts = np.ones(args.num_envs, dtype=bool)
        if binding:
            (args.output_dir / "world_binding.json").write_text(json.dumps({
                "identity": binding, "live": readback(raw)}, indent=2) + "\n")
        explicit_reset_s = time.perf_counter() - reset_start
        initial_action, _ = model.predict(fixed_observation, deterministic=True)
        callback = RolloutAudit()
        training_start = time.perf_counter()
        model.wp2_started = training_start
        model.wp2_excluded_eval_s = 0.0
        model.wp2_wall_allowance = (14_400. if args.development_curriculum_approved else
                                    3_600. if args.development_extension_approved else
                                    min(3_500., 3_600. - used_wall))
        model.learn(
            total_timesteps=args.total_timesteps - (int(model.num_timesteps) if args.resume else 0), callback=callback,
            reset_num_timesteps=(args.resume is None), progress_bar=False)
        training_process_elapsed = time.perf_counter() - training_start
        training_elapsed = training_process_elapsed - model.wp2_excluded_eval_s
        consumed_steps = int(model.num_timesteps)
        consumed_new_steps = consumed_steps - resume_start_steps
        model.num_timesteps = int(getattr(model, "last_complete_update_steps", 0))
        if not args.formal_approved:
            ledger_row.update(state="MEASURED", completed_steps=int(model.num_timesteps),
                consumed_steps=consumed_new_steps, training_wall_s=training_elapsed)
            ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
        after = parameter_vector(model)
        parameter_change = float(torch.linalg.vector_norm(after - before))
        model_path = args.output_dir / "final_model.zip"
        model.save(model_path)
        completed_steps = int(model.num_timesteps)
        (args.output_dir / "completed_steps.txt").write_text(
            f"{completed_steps}\n", encoding="utf-8")
        np.save(args.output_dir / "reload_fixed_observation.npy", fixed_observation)
        state = capture_state(raw, curriculum, completed_steps, next_validation)
        torch.save(state, args.output_dir / "training_state.pt")
        # Exercise restoration, compare the NEXT random draws, then restore again.
        expected_draws = [r.uniform() for r in raw.env_rng]
        restore_state(raw, torch.load(args.output_dir / "training_state.pt", weights_only=False, map_location="cpu"))
        restored_draws = [r.uniform() for r in raw.env_rng]
        restore_state(raw, state)
        restore_equal = expected_draws == restored_draws

        reloaded = PPO.load(model_path, env=env, device=args.device)
        if getattr(reloaded, "development_binding", None) != binding:
            raise RuntimeError("save/load world/source/reference identity mismatch")
        reloaded_action, _ = reloaded.predict(fixed_observation, deterministic=True)
        reload_action_max_abs = float(np.max(np.abs(reloaded_action - model.predict(
            fixed_observation, deterministic=True)[0])))
        episode_start = time.perf_counter()
        episode_observation = env.reset()
        episode_reward = np.zeros(args.num_envs, dtype=np.float64)
        episode_steps = 0
        episode_done = np.zeros(args.num_envs, dtype=bool)
        # New-world delivery reuses the passed reload smoke; no extra fixed-action probe.
        while not args.root_overlay and episode_steps < 903 and not episode_done[0]:
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
        core_checks = (
                len(callback.rollouts) >= 2
                and all(row["reward_finite"] and row["observation_finite"]
                        for row in callback.rollouts)
                and losses_finite
                and parameter_change > 0.0 and reload_action_max_abs <= 1e-6
                and restore_equal)
        target_reached = completed_steps == args.total_timesteps
        curriculum_condition = (bool(curriculum_events) or args.development_extension_approved
                                or not target_reached)
        wall_cap_reached = (args.development_curriculum_approved and not target_reached
                            and training_elapsed >= model.wp2_wall_allowance)
        status = ("PASS" if core_checks and curriculum_condition and target_reached else
                  "WALL_CAP_REACHED" if core_checks and wall_cap_reached else "FAIL")
        result = {
            "status": status,
            "development_only": True,
            "authorization": authorization,
            "curriculum_events": curriculum_events,
            "fixed_diagnostic_events": fixed_diagnostic_events,
            "training_state_restore_next_rng_draw_equal": restore_equal,
            "next_validation": next_validation,
            "warm_start": {"rule": "canonical normalized home; +0.2 on44 fingers and clip; arms unchanged",
                           "initial_actual_action_env0": initial_action[0].tolist(),
                           "initial_action_net_bias": initial_bias},
            "seed": args.seed, "num_envs": args.num_envs,
            "n_steps_per_env": args.n_steps,
            "requested_global_transitions": args.total_timesteps,
            "resume_start_steps": resume_start_steps,
            "completed_steps": completed_steps,
            "consumed_global_transitions": consumed_new_steps,
            "resume_boundary": {"explicit_new_sim_reset": args.resume is not None,
                "sb3_last_observation_rebound": args.resume is not None,
                "simulator_bit_state_restored": False,
                "identity_compatibility": compatibility if args.resume is not None else None},
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
                "training_process_s_including_fixed_diagnostics": training_process_elapsed,
                "fixed_diagnostic_s_excluded_from_training_cap": model.wp2_excluded_eval_s,
                "rollout_collection_s": float(sum(
                    row["collection_s"] for row in callback.rollouts)),
                "ppo_update_and_overhead_s": float(training_elapsed - sum(
                    row["collection_s"] for row in callback.rollouts)),
                "explicit_reset_s": explicit_reset_s,
                "reload_physics_episode_s": reload_episode_s,
                "global_transitions_per_s": consumed_new_steps / training_elapsed,
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
        (args.output_dir / "training_reset_samples.json").write_text(json.dumps(raw.reset_records, indent=2) + "\n")
        (args.output_dir / "pregrasp_boundary_trace.json").write_text(json.dumps(raw.boundary_trace, indent=2) + "\n")
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "artifact": str(result_path),
                          "steps": completed_steps,
                          "reload_action_max_abs": reload_action_max_abs}, indent=2))
        return 0 if result["status"] in ("PASS", "WALL_CAP_REACHED") else 1
    except BaseException:
        import traceback
        (args.output_dir / "failure.txt").write_text(traceback.format_exc())
        if not args.formal_approved:
            ledger_row.update(state="FAILED_RESERVED_STEPS_RETAINED",
                consumed_steps=(int(model.num_timesteps) - resume_start_steps) if "model" in locals() else 0,
                completed_steps=int(getattr(model, "last_complete_update_steps", 0)) if "model" in locals() else 0,
                training_wall_s=time.perf_counter() - training_start if "training_start" in locals() else 0.)
            ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
        raise
    finally:
        monitor.stop()
        env.close()


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
