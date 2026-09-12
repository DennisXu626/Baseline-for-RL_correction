"""Fixed-seed stage0 policy diagnostic; it never updates curriculum or policy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
for key in ("bundle-root", "robot-urdf", "reference", "root-overlay", "model", "output-dir"):
    parser.add_argument("--" + key, type=Path, required=True)
parser.add_argument("--mode", choices=("stochastic", "deterministic"), required=True)
parser.add_argument("--expected-steps", type=int, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np
import torch
from stable_baselines3 import PPO
from .development_binding import identity, readback, compatible_resume
from .env import Pour17JointEnv
from .geometry import sha256
from .scene import isaac_scene_config
from .wp4_world import apply_root_overlay
from .robot_table_contact import RobotTableContacts


def main():
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cfg = isaac_scene_config(args.bundle_root, num_envs=25, seed=100000)
    cfg.sim.device, cfg.sim.log_dir = args.device, str(args.output_dir / "isaaclab_logs")
    cfg.bundle_root, cfg.robot_urdf, cfg.reference_path = map(str, (args.bundle_root, args.robot_urdf, args.reference))
    cfg.curriculum_stage, cfg.probe_only = 0, False
    cfg.external_evaluator_controls_termination = False
    apply_root_overlay(cfg, args.bundle_root, args.root_overlay)
    raw = Pour17JointEnv(cfg)
    try:
        model = PPO.load(args.model, device=args.device)
        if int(model.num_timesteps) != args.expected_steps:
            raise RuntimeError("policy step count differs from requested endpoint")
        binding = identity(args.bundle_root, args.reference, args.robot_urdf, args.root_overlay)
        compatibility = compatible_resume(getattr(model, "development_binding", None), binding)
        if not compatibility["compatible"]:
            raise RuntimeError(f"policy diagnostic identity mismatch: {compatibility}")
        np.random.seed(200000)
        torch.manual_seed(200000)
        raw.policy_arm_table_probe = RobotTableContacts(raw)
        raw.policy_arm_table_contact_latch = torch.zeros(25, dtype=torch.bool, device=raw.device)
        raw.policy_capture_terminal = True
        observation, _ = raw.reset(seed=100000)
        live = readback(raw)

        records, traces = [None] * 25, {0: [], 1: []}
        first_arm_table_contacts = [None] * 25
        for step in range(903):
            action, _ = model.predict(observation["policy"].cpu().numpy(),
                                      deterministic=args.mode == "deterministic")
            observation, _, terminated, truncated, _ = raw.step(torch.as_tensor(action, device=raw.device))
            for env_id, active in enumerate(raw.policy_arm_table_probe.latest_by_env):
                if active and first_arm_table_contacts[env_id] is None:
                    first_arm_table_contacts[env_id] = {
                        "step": step + 1, "links": dict(active)}
            for env_id in (0, 1):
                if records[env_id] is None:
                    q = raw.robot.root_physx_view.get_dof_positions()[env_id, raw.joint_ids]
                    q_target = raw.robot.root_physx_view.get_dof_position_targets()[env_id, raw.joint_ids]
                    traces[env_id].append({"step": step + 1, "q_rad": q.cpu().tolist(),
                        "joint_position_target_rad": q_target.cpu().tolist(),
                        "actual_hand_six_m": raw._hand_six()[env_id].cpu().tolist(),
                        "target_hand_six_m": raw._target_hand_six()[env_id].cpu().tolist(),
                        "state_timing": "post-step; terminal step state is reset and replaced by terminal_detail",
                        "active_arm_table_counts": raw.policy_arm_table_probe.latest_by_env[env_id]})
            for env_id in torch.nonzero(terminated | truncated).flatten().cpu().tolist():
                if records[env_id] is None:
                    records[env_id] = dict(raw.terminal_stats_by_env[env_id],
                        seed=100000 + env_id, steps=step + 1,
                        first_arm_table_contact=first_arm_table_contacts[env_id])
            if all(record is not None for record in records):
                break
        if any(record is None for record in records):
            raise RuntimeError("fixed25 diagnostic did not finish under903 steps")
        result = {"kind": "DEVELOPMENT_STAGE0_POLICY_DIAGNOSTIC_NOT_TASK_SR",
            "mode": args.mode, "episode_seeds": list(range(100000, 100025)),
            "policy_sampling_seed": 200000, "records": records,
            "first_two_step_traces": traces, "policy_steps": int(model.num_timesteps),
            "model_sha256": sha256(args.model), "binding": binding,
            "binding_compatibility": compatibility, "live_readback": live,
            "arm_body_names": raw.policy_arm_table_probe.arm_names,
            "contact_backend": "GPU detailed rigid contact view; arm links versus actual static table collider",
            "contact_evidence": raw.policy_arm_table_probe.evidence(),
            "wall_s_including_kit": time.perf_counter() - started,
            "curriculum_updated": False, "normal_termination": True}
        (args.output_dir / "policy_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        raw.close()


try:
    main()
finally:
    simulation_app.close()
