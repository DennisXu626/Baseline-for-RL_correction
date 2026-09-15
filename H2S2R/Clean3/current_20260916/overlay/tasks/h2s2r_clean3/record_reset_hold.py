"""Capture the reset state and a real-time zero-action hold interval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--runtime_root", type=Path, required=True)
parser.add_argument("--v12_root", type=Path, required=True)
parser.add_argument("--output_root", type=Path, required=True)
parser.add_argument("--seconds", type=float, default=6.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(args.v12_root.resolve()))
sys.path.insert(0, str(args.runtime_root.resolve()))

from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.h2s2r_clean3.cfg import build_cfg  # noqa: E402
from tasks.h2s2r_clean3.env import Clean3H2S2REnv  # noqa: E402


CONTROL_HZ = 20
VIDEO_FPS = CONTROL_HZ
VIEW_NAMES = ("front", "side", "top")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(env: Clean3H2S2REnv) -> dict[str, np.ndarray]:
    return {
        name: batch[0].detach().cpu().numpy().astype(np.uint8, copy=False)
        for name, batch in env.camera_frames().items()
    }


def snapshot_row(env: Clean3H2S2REnv, *, phase: str, control: int, done: bool) -> dict:
    state = env.runtime_snapshot()
    left_tip_distance = torch.linalg.vector_norm(
        state["left_tips"][0] - state["plate_pose"][0, :3], dim=-1
    ).mean()
    right_tip_distance = torch.linalg.vector_norm(
        state["right_tips"][0] - state["sponge_pose"][0, :3], dim=-1
    ).mean()
    return {
        "phase": phase,
        "control": control,
        "elapsed_seconds": max(control, 0) / CONTROL_HZ,
        "done": done,
        "plate_pose_wxyz": state["plate_pose"][0].cpu().tolist(),
        "sponge_pose_wxyz": state["sponge_pose"][0].cpu().tolist(),
        "left_mean_tip_to_plate_center_m": float(left_tip_distance),
        "right_mean_tip_to_sponge_center_m": float(right_tip_distance),
        "max_fingertip_contact_force_n": float(state["contact_force_n"][0].max()),
        "max_penetration_depth_m": float(state["penetration_depth_m"][0].max()),
        "all_state_finite": bool(
            all(
                bool(torch.isfinite(value).all())
                for value in state.values()
                if torch.is_tensor(value) and value.is_floating_point()
            )
        ),
        "max_joint_position_error_rad": float(
            (state["joint_q"][0] - state["target_q"][0]).abs().max()
        ),
        "reference_index": int(state["reference_index"][0]),
    }


def main() -> None:
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = build_cfg(
        runtime_root=args.runtime_root,
        v12_root=args.v12_root,
        num_envs=1,
        seed=42,
        external_eval=True,
        record_cameras=True,
    )
    cfg.sim.device = args.device
    cfg.sim.render_interval = 1
    raw = Clean3H2S2REnv(cfg, render_mode="rgb_array")
    wrapped = GymStyleEnvWrapper(raw, clip_actions=cfg.clip_actions)
    wrapped.reset()

    raw.scene.write_data_to_sim()
    for _ in range(3):
        raw.sim.render()
        for camera in raw._record_cameras.values():
            camera.update(cfg.sim.dt, force_recompute=True)
    reset_frames = frames(raw)
    for name in VIEW_NAMES:
        imageio.imwrite(output / f"reset_before_action_{name}.png", reset_frames[name])
    combined_reset = np.concatenate([reset_frames[name] for name in VIEW_NAMES], axis=1)
    imageio.imwrite(output / "reset_before_action_three_view.png", combined_reset)

    video = output / "reset_zero_action_hold_three_view.mp4"
    writer = imageio.get_writer(
        video,
        fps=VIDEO_FPS,
        codec="libx264",
        quality=7,
        macro_block_size=None,
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart"],
    )
    rows = [snapshot_row(raw, phase="reset_before_action", control=-1, done=False)]
    zero_action = torch.zeros(1, 58, device=raw.device)
    controls = round(args.seconds * CONTROL_HZ)
    with torch.inference_mode():
        for control in range(controls):
            _, _, done, _ = wrapped.step(zero_action)
            raw.sim.render()
            current_frames = frames(raw)
            writer.append_data(np.concatenate([current_frames[name] for name in VIEW_NAMES], axis=1))
            rows.append(
                snapshot_row(
                    raw,
                    phase="zero_action_hold",
                    control=control + 1,
                    done=bool(done[0]),
                )
            )
    writer.close()

    metrics = output / "reset_zero_action_hold_metrics.jsonl"
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = {
        "runtime_root": str(args.runtime_root.resolve()),
        "seed": 42,
        "policy_or_checkpoint_loaded": False,
        "action": "58D all-zero action; holds the configured pre-manipulation joint target",
        "reset_image_captured_before_environment_step": True,
        "seconds": args.seconds,
        "control_hz": CONTROL_HZ,
        "controls": controls,
        "video_fps": VIDEO_FPS,
        "done_count": sum(int(row["done"]) for row in rows),
        "reset_three_view_png": "reset_before_action_three_view.png",
        "reset_three_view_png_sha256": sha256(output / "reset_before_action_three_view.png"),
        "video": video.name,
        "video_sha256": sha256(video),
        "metrics": metrics.name,
        "metrics_sha256": sha256(metrics),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    wrapped.close()


try:
    main()
except BaseException as error:
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "entry_failure.json").write_text(
        json.dumps(
            {
                "type": type(error).__name__,
                "message": repr(error),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    raise
finally:
    app.close()
