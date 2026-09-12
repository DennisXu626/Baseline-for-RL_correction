"""25 genuine stochastic episodes in a separate Isaac process.

No parent training episode is reset or advanced while this evaluation runs.
These are curriculum pregrasp statistics, never external task success rates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time


def main():
    from isaaclab.app import AppLauncher
    ap = argparse.ArgumentParser()
    for key in ("bundle-root", "robot-urdf", "reference", "model", "output-dir"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--root-overlay", type=Path)
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    app = AppLauncher(args).app
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from .env import Pour17JointEnv
    from .scene import isaac_scene_config
    cfg = isaac_scene_config(args.bundle_root, num_envs=25, seed=100000)
    cfg.sim.device, cfg.sim.log_dir = args.device, str(args.output_dir / "isaaclab_logs")
    cfg.bundle_root, cfg.robot_urdf, cfg.reference_path = map(str, (args.bundle_root, args.robot_urdf, args.reference))
    cfg.curriculum_stage = args.stage
    cfg.external_evaluator_controls_termination = False
    cfg.probe_only = False
    if args.root_overlay:
        from .wp4_world import apply_root_overlay
        apply_root_overlay(cfg, args.bundle_root, args.root_overlay)
    raw = Pour17JointEnv(cfg)
    try:
        np.random.seed(100000)
        torch.manual_seed(100000)
        model = PPO.load(args.model, device=args.device)
        obs, _ = raw.reset(seed=100000)
        if args.root_overlay:
            from .development_binding import identity, readback
            binding = identity(args.bundle_root, args.reference, args.robot_urdf, args.root_overlay)
            if getattr(model, "development_binding", None) != binding:
                raise RuntimeError("internal evaluation world/source/reference identity mismatch")
            (args.output_dir / "world_binding.json").write_text(json.dumps({
                "identity": binding, "live": readback(raw)}, indent=2) + "\n")
        records = [None] * 25
        initial = raw.reset_state()
        for step in range(903):
            action, _ = model.predict(obs["policy"].cpu().numpy(), deterministic=False)
            obs, _, term, trunc, _ = raw.step(torch.as_tensor(action, device=raw.device))
            for i in torch.nonzero(term | trunc).flatten().cpu().tolist():
                if records[i] is None:
                    records[i] = dict(raw.terminal_stats_by_env[i], seed=100000+i, steps=step+1)
            if all(r is not None for r in records):
                break
        if any(r is None for r in records):
            raise RuntimeError("internal 25-episode evaluation did not finish under903 steps")
        result = {"kind": "REAL_STOCHASTIC_CURRICULUM_NOT_TASK_SR", "stage": args.stage,
            "seeds": list(range(100000, 100025)), "records": records,
            "pregrasp_successes": [float(r["pregrasp_success"]) for r in records],
            "wall_s_including_kit": time.perf_counter() - started, "initial_state": initial,
            "pregrasp_boundary_trace": raw.boundary_trace}
        (args.output_dir / "internal_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        raw.close()
        app.close()


if __name__ == "__main__":
    main()
