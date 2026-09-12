"""Registered scalar-evaluator entry and launcher for Pour17 WP1."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


_RAW_ENV = None
_MODEL = None
_CERTIFICATION_RUNS: list[dict[str, Any]] = []


class ScalarEvaluatorEnv:
    def __init__(self, raw, seed: int):
        self.raw = raw
        self.seed = int(seed)

    def reset(self):
        observation, _ = self.raw.reset(seed=self.seed)
        return observation["policy"][0].detach().cpu().numpy().astype(np.float32)

    def step(self, action):
        import torch
        value = np.asarray(action, dtype=np.float32).reshape(1, 58)
        observation, reward, terminated, truncated, extras = self.raw.step(
            torch.tensor(value, dtype=torch.float32, device=self.raw.device))
        info = self.raw.evaluator_info()
        return (
            observation["policy"][0].detach().cpu().numpy().astype(np.float32),
            bool(terminated[0]), bool(truncated[0]), info,
        )

    def reset_state(self) -> dict:
        return self.raw.reset_state()

    def apply_certification_offset(self, alpha: float) -> None:
        self.raw.apply_certification_offset(alpha)


def _record_current_trace() -> None:
    if _RAW_ENV is not None and getattr(_RAW_ENV, "wp1_eval_seed", None) is not None:
        _CERTIFICATION_RUNS.append({
            "seed": getattr(_RAW_ENV, "wp1_eval_seed", None),
            "events": list(_RAW_ENV.certification_trace),
        })
        _RAW_ENV.wp1_eval_seed = None
        output = os.environ.get("VIVIDEX_POUR17_EVAL_OUT")
        if output:
            trace_path = Path(output).with_name("certification_trace.json")
            trace_path.write_text(json.dumps({
                "status": "TRACE_ONLY_NOT_SUCCESS_RESULT",
                "runs": _CERTIFICATION_RUNS,
            }, indent=2) + "\n", encoding="utf-8")


def _close_raw() -> None:
    global _RAW_ENV
    if _RAW_ENV is not None:
        _record_current_trace()
        _RAW_ENV.close()
        _RAW_ENV = None


atexit.register(_close_raw)


def make_env(num_envs: int, seed: int, device: str):
    """Evaluator callback: create exactly one canonical-t0 physical env."""
    global _RAW_ENV
    if int(num_envs) != 1:
        raise ValueError("registered run_eval.py is scalar; num_envs must be 1")
    if _RAW_ENV is not None:
        # The scalar evaluator asks for one environment per episode.  Reuse the
        # same Isaac SimulationContext and perform a real seeded reset instead
        # of destroying/recreating Kit inside one process.
        _record_current_trace()
        _RAW_ENV.certification_trace.clear()
        _RAW_ENV.wp1_eval_seed = int(seed)
        return ScalarEvaluatorEnv(_RAW_ENV, seed)
    from .env import Pour17JointEnv
    from .scene import isaac_scene_config

    bundle = Path(os.environ["VIVIDEX_POUR17_BUNDLE"])
    reference = Path(os.environ["VIVIDEX_POUR17_REFERENCE"])
    robot_urdf = Path(os.environ["VIVIDEX_POUR17_URDF"])
    cfg = isaac_scene_config(bundle, num_envs=1, seed=int(seed))
    cfg.sim.device = device
    cfg.sim.log_dir = str((Path(os.environ["VIVIDEX_POUR17_EVAL_OUT"]).parent / "isaaclab_logs").resolve())
    cfg.bundle_root = str(bundle.resolve())
    cfg.reference_path = str(reference.resolve())
    cfg.robot_urdf = str(robot_urdf.resolve())
    cfg.probe_only = False
    cfg.external_evaluator_controls_termination = True
    cfg.curriculum_stage = 0
    _RAW_ENV = Pour17JointEnv(cfg)
    _RAW_ENV.wp1_eval_seed = int(seed)
    return ScalarEvaluatorEnv(_RAW_ENV, seed)


def policy(observation):
    """Evaluator callback: deterministic SB3 mean with no exploration noise."""
    global _MODEL
    if _MODEL is None:
        from stable_baselines3 import PPO
        _MODEL = PPO.load(os.environ["VIVIDEX_POUR17_MODEL"], device=os.environ.get(
            "VIVIDEX_POUR17_DEVICE", "cuda:0"))
    action, _ = _MODEL.predict(np.asarray(observation, dtype=np.float32), deterministic=True)
    return action


def _driver() -> int:
    import argparse
    import runpy
    import sys
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=20260829)
    parser.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    os.environ.update({
        "VIVIDEX_POUR17_BUNDLE": str(args.bundle_root.resolve()),
        "VIVIDEX_POUR17_REFERENCE": str(args.reference.resolve()),
        "VIVIDEX_POUR17_URDF": str(args.robot_urdf.resolve()),
        "VIVIDEX_POUR17_MODEL": str(args.model.resolve()),
        "VIVIDEX_POUR17_DEVICE": args.device,
        "VIVIDEX_POUR17_EVAL_OUT": str(args.out.resolve()),
    })
    sys.argv = [str(args.evaluator), "--episodes", str(args.episodes),
                "--seed-base", str(args.seed_base),
                "--entry", "adaptation.pour17_v1.eval_entry:make_env,policy",
                "--out", str(args.out), "--device", args.device]
    try:
        runpy.run_path(str(args.evaluator), run_name="__main__")
    except SystemExit as exit_value:
        return int(exit_value.code or 0)
    finally:
        _close_raw()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_driver())
