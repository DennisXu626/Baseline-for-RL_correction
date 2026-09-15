"""One-environment gate for the isolated 367D/22D Ours Stage-1 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--runtime_root", type=Path, required=True)
parser.add_argument("--v12_root", type=Path, required=True)
parser.add_argument("--output_root", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, str(args.v12_root.resolve()))
sys.path.insert(0, str(args.runtime_root.resolve()))

from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.h2s2r_clean3.cfg import build_cfg  # noqa: E402
from tasks.h2s2r_clean3.env import Clean3H2S2REnv  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = build_cfg(
        runtime_root=args.runtime_root,
        v12_root=args.v12_root,
        num_envs=1,
        seed=42,
        external_eval=False,
        record_cameras=False,
        ours_stage1_contract=True,
    )
    cfg.sim.device = args.device
    raw = Clean3H2S2REnv(cfg)
    wrapped = GymStyleEnvWrapper(raw, clip_actions=cfg.clip_actions)
    try:
        observation = wrapped.reset()
        policy = observation["obs"]
        privileged = observation["priv_info"]
        if policy.shape != (1, 367) or privileged.shape != (1, 22):
            raise RuntimeError(
                f"invalid Ours observation shapes: {policy.shape} / {privileged.shape}"
            )
        if not bool(torch.isfinite(policy).all() and torch.isfinite(privileged).all()):
            raise RuntimeError("Ours observation contains non-finite values")
        if cfg.task_rows != 0 or raw.max_episode_length - 1 != 110:
            raise RuntimeError(
                f"invalid fixed Stage-1 horizon: task_rows={cfg.task_rows}, "
                f"deadline={raw.max_episode_length - 1}"
            )

        probe = torch.linspace(-0.5, 0.5, 58, device=raw.device).unsqueeze(0)
        next_observation, _, _, _ = wrapped.step(probe)
        next_policy = next_observation["obs"]
        expected_residual = raw._cumulative_residual / raw._residual_limit
        if not torch.allclose(next_policy[:, 116:174], expected_residual, atol=1e-6):
            raise RuntimeError("367D observation does not expose cumulative residual")
        if not torch.allclose(next_policy[:, 309:367], probe, atol=1e-6):
            raise RuntimeError("367D observation does not expose last action")
        expected_force = torch.cat(
            [raw.contact_forces()[:, 5:], raw.contact_forces()[:, :5]], dim=1
        )
        expected_force = (expected_force / 10.0).clamp(max=3.0)
        if not torch.allclose(next_policy[:, 240:250], expected_force, atol=1e-6):
            raise RuntimeError("367D observation force block/order mismatch")
        if not torch.allclose(next_observation["priv_info"][:, 12:], expected_force, atol=1e-6):
            raise RuntimeError("22D privileged force block/order mismatch")

        write_json(
            output / "result.json",
            {
                "policy_observation_dim": policy.shape[1],
                "privileged_dim": privileged.shape[1],
                "action_dim": probe.shape[1],
                "cumulative_residual_observed": True,
                "last_action_observed": True,
                "contact_force_order": "left_plate_then_right_sponge",
                "contact_force_matches_net_forces_w": True,
                "fixed_success_deadline_row": 110,
                "task_tail_rows": cfg.task_rows,
                "default_h2s2r_contract_unchanged": True,
            },
        )
    finally:
        wrapped.close()


try:
    main()
except BaseException as error:
    try:
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output_root / "entry_failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        app.close()
    raise
else:
    app.close()
