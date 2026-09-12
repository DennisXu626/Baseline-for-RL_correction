"""Runtime probe, final reference build, and physical wiring smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=(
        "probe-finalize", "wiring", "openloop", "eval-callbacks", "long-horizon",
    ),
    required=True,
)
parser.add_argument("--bundle-root", type=Path, required=True)
parser.add_argument("--robot-urdf", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--preik-reference", type=Path)
parser.add_argument("--reference", type=Path)
parser.add_argument("--model", type=Path)
parser.add_argument("--evaluator", type=Path)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1701)
parser.add_argument("--record-video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app
print("[pour17] Isaac application launched", flush=True)

import numpy as np
import torch

from .env import Pour17JointEnv
from .geometry import sha256
from .scene import build_final_reference, isaac_scene_config
print("[pour17] WP1 modules imported", flush=True)


def _cfg(reference: Path, *, probe_only: bool, external: bool):
    cfg = isaac_scene_config(args.bundle_root, num_envs=args.num_envs, seed=args.seed)
    cfg.sim.device = args.device
    cfg.sim.log_dir = str((args.output_dir / "isaaclab_logs").resolve())
    cfg.bundle_root = str(args.bundle_root.resolve())
    cfg.reference_path = str(reference.resolve())
    cfg.robot_urdf = str(args.robot_urdf.resolve())
    cfg.probe_only = bool(probe_only)
    cfg.external_evaluator_controls_termination = bool(external)
    cfg.curriculum_stage = 0
    cfg.viewer.eye = (2.2, 2.2, 1.8)
    cfg.viewer.lookat = (-0.14, 0.08, 1.0)
    cfg.viewer.resolution = (960, 720)
    return cfg


def _normalized(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (q - lower) / (upper - lower) - 1.0, -1.0, 1.0)


def probe_finalize() -> dict:
    print("[pour17] constructing probe environment", flush=True)
    if args.preik_reference is None:
        raise ValueError("--preik-reference is required")
    try:
        config = _cfg(args.preik_reference, probe_only=True, external=True)
        print("[pour17] scene config constructed", flush=True)
        env = Pour17JointEnv(config)
        print("[pour17] probe environment constructed", flush=True)
    except BaseException as error:
        import traceback
        print(f"[pour17] probe construction failed: {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        raise
    try:
        env.reset(seed=args.seed)
        home = env.reset_q.detach().cpu().numpy()
        action = torch.tensor(
            np.repeat(_normalized(home, env.lower.cpu().numpy(), env.upper.cpu().numpy())[None],
                      args.num_envs, axis=0),
            dtype=torch.float32, device=env.device)
        for _ in range(2):
            env.step(action)
        probe = env.runtime_probe()
        probe["settling"] = {"control_steps": 2, "physics_steps": 24}
        probe_path = args.output_dir / "runtime_probe.json"
        probe_path.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()
    reference_path = args.output_dir / "reference.npz"
    manifest_path = args.output_dir / "reference_manifest.json"
    source_manifest = args.output_dir / "source_manifest.json"
    reference = build_final_reference(
        preik_path=args.preik_reference, probe_path=probe_path,
        urdf_path=args.robot_urdf, output_path=reference_path,
        manifest_path=manifest_path, source_manifest_path=source_manifest,
        config_path=args.config,
    )
    return {"probe": probe, "reference": reference}


def wiring() -> dict:
    if args.reference is None:
        raise ValueError("--reference is required")
    env = Pour17JointEnv(_cfg(args.reference, probe_only=False, external=True))
    try:
        obs, _ = env.reset(seed=args.seed)
        initial_q = env.robot.data.joint_pos[:, env.joint_ids].detach().cpu().numpy()
        sample_indices = (0, 40, min(141, env.reference_length - 1))
        rows = []
        for index in sample_indices:
            q_target = env.reference_q[index].detach().cpu().numpy()
            action_np = _normalized(q_target, env.lower.cpu().numpy(), env.upper.cpu().numpy())
            action = torch.tensor(np.repeat(action_np[None], args.num_envs, axis=0),
                                  dtype=torch.float32, device=env.device)
            before = env.robot.data.joint_pos[:, env.joint_ids].detach().cpu().numpy().copy()
            for _ in range(3):
                obs, reward, terminated, truncated, info = env.step(action)
            after = env.robot.data.joint_pos[:, env.joint_ids].detach().cpu().numpy().copy()
            rows.append({
                "reference_index": index,
                "finite_observation": bool(torch.isfinite(obs["policy"]).all()),
                "finite_reward": bool(torch.isfinite(reward).all()),
                "right_arm_response_l2": float(np.linalg.norm(after[0, :7] - before[0, :7])),
                "left_arm_response_l2": float(np.linalg.norm(after[0, 7:14] - before[0, 7:14])),
                "right_finger_response_l2": float(np.linalg.norm(after[0, 14:36] - before[0, 14:36])),
                "left_finger_response_l2": float(np.linalg.norm(after[0, 36:58] - before[0, 36:58])),
                "tip_contact_count": int(env._tip_contacts()[0].sum().item()),
            })
        actual_q = env.robot.data.joint_pos[:, env.joint_ids].detach().cpu().numpy()
        response_fields = (
            "right_arm_response_l2", "left_arm_response_l2",
            "right_finger_response_l2", "left_finger_response_l2",
        )
        response_ok = {
            field: max(row[field] for row in rows) > 1e-6 for field in response_fields
        }
        names_match = env.runtime_probe()["joint_names_in_order"] == list(env.joint_names)
        return {
            "status": "PASS" if (
                names_match
                and all(row["finite_observation"] and row["finite_reward"] for row in rows)
                and all(response_ok.values())
            ) else "FAIL",
            "num_envs": args.num_envs, "seed": args.seed,
            "observation_shape": list(obs["policy"].shape),
            "joint_names_match": names_match,
            "controlled_group_response_gt_1e-6": response_ok,
            "initial_q_sha256": sha256_array(initial_q),
            "actual_q_sha256": sha256_array(actual_q),
            "representative_pose_rows": rows,
            "reset_state": env.reset_state(),
        }
    finally:
        env.close()


def sha256_array(array: np.ndarray) -> str:
    from .geometry import ndarray_sha256
    return ndarray_sha256(np.asarray(array))


def openloop() -> dict:
    if args.reference is None:
        raise ValueError("--reference is required")
    env = Pour17JointEnv(_cfg(args.reference, probe_only=False, external=True),
                         render_mode="rgb_array" if args.record_video else None)
    frames = []
    trace = {name: [] for name in (
        "input_target_q_rad", "actual_q_rad", "object_pose_wxyz", "wrist_pose_wxyz",
        "landmarks_m", "tip_contacts", "reward", "hand_error_m", "object_error_m")}
    start = time.perf_counter()
    try:
        env.reset(seed=args.seed)
        lower = env.lower.cpu().numpy()
        upper = env.upper.cpu().numpy()
        for index in range(env.reference_length):
            target = env.reference_q[index].cpu().numpy()
            action = torch.tensor(
                np.repeat(_normalized(target, lower, upper)[None], args.num_envs, axis=0),
                dtype=torch.float32, device=env.device)
            obs, reward, _, _, _ = env.step(action)
            actual_hand = env._landmarks()
            actual_object = env._object_poses()
            _, target_object, target_hand = env._targets()
            trace["input_target_q_rad"].append(target.copy())
            trace["actual_q_rad"].append(env.robot.data.joint_pos[0, env.joint_ids].cpu().numpy().copy())
            trace["object_pose_wxyz"].append(actual_object[0].cpu().numpy().copy())
            trace["wrist_pose_wxyz"].append(env._wrist_poses()[0].cpu().numpy().copy())
            trace["landmarks_m"].append(actual_hand[0].cpu().numpy().copy())
            trace["tip_contacts"].append(env._tip_contacts()[0].cpu().numpy().copy())
            trace["reward"].append(float(reward[0]))
            trace["hand_error_m"].append(
                torch.linalg.vector_norm(actual_hand[0] - target_hand[0], dim=-1).mean(dim=-1).cpu().numpy())
            trace["object_error_m"].append(
                torch.linalg.vector_norm(actual_object[0, :, :3] - target_object[0, :, :3], dim=-1).cpu().numpy())
            if args.record_video:
                frames.append(env.render(recompute=True))
        elapsed = time.perf_counter() - start
    finally:
        env.close()
    trace_np = {name: np.asarray(values) for name, values in trace.items()}
    trace_path = args.output_dir / "real_state_trace.npz"
    np.savez_compressed(trace_path, **trace_np,
                        reference_sha256=np.asarray(sha256(args.reference)))
    video_status = "not_requested"
    video_path = args.output_dir / "openloop_physical.mp4"
    if frames:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(video_path, frames, fps=20, macro_block_size=1)
            video_status = "written"
        except Exception as error:
            video_status = f"failed: {type(error).__name__}: {error}"
    return {
        "status": "PASS" if all(np.isfinite(value).all() for value in trace_np.values()) else "FAIL",
        "steps": env.reference_length, "elapsed_s": elapsed,
        "control_steps_per_s": env.reference_length * args.num_envs / elapsed,
        "trace": {"path": str(trace_path), "sha256": sha256(trace_path)},
        "video": {"status": video_status, "path": str(video_path) if video_path.is_file() else None,
                  "sha256": sha256(video_path) if video_path.is_file() else None},
        "tracking": {
            "joint_abs_error_rad_median": float(np.median(np.abs(
                trace_np["actual_q_rad"] - trace_np["input_target_q_rad"]))),
            "hand_error_m_median_by_side": np.median(trace_np["hand_error_m"], axis=0).tolist(),
            "object_error_m_max_by_side": np.max(trace_np["object_error_m"], axis=0).tolist(),
        },
    }


def eval_callbacks() -> dict:
    """Exercise evaluator-only G2 callbacks without counting task success."""
    if args.reference is None or args.model is None or args.evaluator is None:
        raise ValueError("--reference, --model, and --evaluator are required")
    import runpy
    from stable_baselines3 import PPO

    contract = runpy.run_path(str(args.evaluator), run_name="wp1_eval_contract")
    cert_alpha = contract["cert_alpha"]
    horizon = int(contract["MAX_CONTROL_STEPS"])
    env = Pour17JointEnv(_cfg(args.reference, probe_only=False, external=True))
    rows = []
    try:
        observation, _ = env.reset(seed=args.seed)
        model = PPO.load(args.model, device=args.device)
        schedule = ((1, 0), (1, 7), (2, 0), (3, 0), (3, 6))
        for phase, tick in schedule:
            alpha = float(cert_alpha(phase, tick))
            before = env._wrist_poses()[0].detach().cpu().numpy().copy()
            if alpha > 0.0:
                env.apply_certification_offset(alpha)
            action, _ = model.predict(
                observation["policy"].detach().cpu().numpy(), deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(
                torch.as_tensor(action, dtype=torch.float32, device=env.device))
            after = env._wrist_poses()[0].detach().cpu().numpy().copy()
            rows.append({
                "phase": phase, "tick": tick, "alpha": alpha,
                "wrist_z_before_m": before[:, 2].tolist(),
                "wrist_z_after_m": after[:, 2].tolist(),
                "observation_finite": bool(torch.isfinite(observation["policy"]).all()),
                "reward_finite": bool(torch.isfinite(reward).all()),
                "env_terminated": bool(terminated[0]),
                "env_truncated": bool(truncated[0]),
            })
        latch_events = [row for row in env.certification_trace
                        if row.get("event") == "latch"]
        alpha_events = [row for row in env.certification_trace
                        if row.get("event") == "alpha"]
        boundary = [
            {"steps": horizon - 1, "evaluator_hit_cap": False},
            {"steps": horizon, "evaluator_hit_cap": True},
        ]
        ok = (
            len(latch_events) == 2 and len(alpha_events) == len(schedule)
            and all(row["observation_finite"] and row["reward_finite"] for row in rows)
            and horizon == 903
        )
        return {
            "status": "PASS_TRACE_ONLY_NOT_SUCCESS_RESULT" if ok else "FAIL",
            "seed": args.seed,
            "evaluator": {"path": str(args.evaluator.resolve()),
                          "sha256": sha256(args.evaluator)},
            "model": {"path": str(args.model.resolve()), "sha256": sha256(args.model)},
            "schedule_rows": rows,
            "certification_events": env.certification_trace,
            "termination_boundary_from_registered_evaluator": boundary,
            "note": "Callback coverage only; excluded from task success rate.",
        }
    finally:
        env.close()


def long_horizon() -> dict:
    """Measure 903 physical control steps without training-env early resets."""
    if args.reference is None or args.model is None or args.evaluator is None:
        raise ValueError("--reference, --model, and --evaluator are required")
    import runpy
    from stable_baselines3 import PPO

    contract = runpy.run_path(str(args.evaluator), run_name="wp1_eval_contract")
    horizon = int(contract["MAX_CONTROL_STEPS"])
    env = Pour17JointEnv(_cfg(args.reference, probe_only=False, external=True))
    checkpoints = []
    try:
        observation, _ = env.reset(seed=args.seed)
        model = PPO.load(args.model, device=args.device)
        rewards = []
        started = time.perf_counter()
        for step in range(1, horizon + 1):
            action, _ = model.predict(
                observation["policy"].detach().cpu().numpy(), deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(
                torch.as_tensor(action, dtype=torch.float32, device=env.device))
            rewards.append(reward.detach().cpu().numpy())
            if step in {1, 269, horizon}:
                checkpoints.append({
                    "step": step,
                    "observation_finite": bool(torch.isfinite(observation["policy"]).all()),
                    "reward_finite": bool(torch.isfinite(reward).all()),
                    "env_terminated_count": int(terminated.sum()),
                    "env_truncated_count": int(truncated.sum()),
                })
        elapsed = time.perf_counter() - started
        reward_array = np.asarray(rewards)
        return {
            "status": "PASS" if (
                np.isfinite(reward_array).all()
                and all(row["observation_finite"] for row in checkpoints)
            ) else "FAIL",
            "num_envs": args.num_envs, "seed": args.seed,
            "control_steps_per_env": horizon,
            "global_transitions": horizon * args.num_envs,
            "elapsed_s": elapsed,
            "global_transitions_per_s": horizon * args.num_envs / elapsed,
            "reward_min_mean_max": [float(reward_array.min()),
                                    float(reward_array.mean()),
                                    float(reward_array.max())],
            "checkpoints": checkpoints,
            "note": "Physical execution throughput only; no PPO updates or task-success claim.",
        }
    finally:
        env.close()


def main() -> int:
    print(f"[pour17] main mode={args.mode}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.mode == "probe-finalize":
        result = probe_finalize()
    elif args.mode == "wiring":
        result = wiring()
    elif args.mode == "openloop":
        result = openloop()
    elif args.mode == "eval-callbacks":
        result = eval_callbacks()
    else:
        result = long_horizon()
    result["command_mode"] = args.mode
    result["wall_time_s"] = time.perf_counter() - started
    suffix = (f"_{args.num_envs}env"
              if args.mode in {"wiring", "openloop", "long-horizon"} else "")
    path = args.output_dir / f"{args.mode}{suffix}_result.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result.get("status", result.get("reference", {}).get("status")),
                      "artifact": str(path)}, indent=2))
    return 0


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
