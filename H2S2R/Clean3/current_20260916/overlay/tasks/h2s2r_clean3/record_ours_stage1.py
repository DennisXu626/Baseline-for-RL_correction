"""Record one deterministic, true-episode three-view Ours replay."""

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
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output_root", type=Path, required=True)
parser.add_argument("--release_row", type=int, required=True)
parser.add_argument("--controls", type=int, default=300)
parser.add_argument("--stage2", action="store_true")
parser.add_argument("--allowed_root", default="/ssd/sy/kailang")
parser.add_argument("--robot_usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

allowed_root = Path(args.allowed_root).resolve()
for candidate in (
    args.runtime_root.resolve(),
    args.checkpoint.resolve(),
    args.output_root.resolve(),
    *(tuple([Path(args.robot_usd).resolve()]) if args.robot_usd else ()),
):
    if allowed_root not in (candidate, *candidate.parents):
        raise ValueError(f"path outside allowed root {allowed_root}: {candidate}")

output = args.output_root.resolve()
output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app
writer = None
env = None

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

runtime_root = args.runtime_root.resolve()
sys.path.insert(0, str(args.v12_root.resolve()))
sys.path.insert(0, str(runtime_root))
sys.path.insert(0, str(runtime_root / "tasks"))
sys.path.insert(0, str(runtime_root / "ours_stage1_overlay"))

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from h2s2r_clean3.cfg import build_cfg  # noqa: E402
from h2s2r_clean3.env import Clean3H2S2REnv  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


try:
    config_path = Path(__file__).with_name("ppo_ours_stage1.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["device"] = "cuda:0"
    config["algorithm"]["num_actors"] = 1
    config["algorithm"]["minibatch_size"] = 32

    cfg = build_cfg(
        runtime_root=runtime_root,
        v12_root=args.v12_root,
        num_envs=1,
        seed=42,
        external_eval=False,
        record_cameras=True,
        grasp_curriculum=True,
        release_row_start=args.release_row,
        terminate_on_grasp_drop=True,
        ours_stage1_contract=not args.stage2,
        ours_stage2_contract=args.stage2,
        robot_usd=args.robot_usd,
    )
    cfg.sim.device = args.device
    cfg.sim.render_interval = 1
    cfg.sim.log_dir = str(output / "isaaclab_logs")
    raw = Clean3H2S2REnv(cfg, render_mode="rgb_array")
    env = GymStyleEnvWrapper(raw, clip_actions=1.0)
    agent = PPO(
        env,
        output_dir=str(output),
        full_config=ConfigWrapper(config, {}, test=True),
        create_output_dir=False,
    )
    agent.restore_test(str(args.checkpoint.resolve()))
    agent.set_eval()
    obs = env.reset()
    if obs["obs"].shape != (1, 367) or obs["priv_info"].shape != (1, 22):
        raise RuntimeError(
            f"invalid Ours replay shapes: obs={obs['obs'].shape}, "
            f"priv={obs['priv_info'].shape}"
        )

    video = output / "three_view.mp4"
    # One frame per 20 Hz policy control: playback duration equals simulation time.
    writer = imageio.get_writer(
        video,
        fps=20,
        codec="libx264",
        quality=7,
        macro_block_size=None,
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart"],
    )
    rows = []
    with torch.inference_mode():
        for control in range(args.controls):
            snapshot = raw.runtime_snapshot()
            raw.sim.render()
            frames = raw.camera_frames()
            writer.append_data(
                np.concatenate(
                    [
                        frames[name][0].cpu().numpy().astype(np.uint8)
                        for name in ("front", "side", "top")
                    ],
                    axis=1,
                )
            )
            input_dict = {
                "obs": agent.running_mean_std(obs["obs"]),
                "priv_info": obs["priv_info"],
            }
            action = agent.model.act_inference(input_dict).clamp(-1.0, 1.0)
            obs, reward, done, _ = env.step(action)
            rows.append(
                {
                    "control": control,
                    "reward": float(reward[0]),
                    "done": bool(done[0]),
                    "action_abs_max": float(action.abs().max()),
                    "plate_z": float(snapshot["plate_pose"][0, 2]),
                    "sponge_z": float(snapshot["sponge_pose"][0, 2]),
                    "reference_index": int(snapshot["reference_index"][0]),
                    "release_row": int(raw.release_row[0]),
                    "pinned": bool(raw._pinned[0]),
                    "grasp_certified": bool(raw._grasp_certified[0]),
                    "grasp_success": bool(raw._grasp_success[0]),
                    "grasp_dropped": bool(raw._grasp_dropped[0]),
                    "cert_run": int(raw._cert_run[0]),
                    "hold_run": int(raw._hold_run[0]),
                    "snapshot_before_action": True,
                }
            )
            if bool(done[0]):
                break
    writer.close()
    writer = None
    metrics = output / "step_metrics.jsonl"
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "policy_contract": (
            "Ours Stage-2 367D actor + 22D privileged critic"
            if args.stage2
            else "Ours Stage-1 367D actor + 22D privileged critic"
        ),
        "deterministic": True,
        "requested_max_controls": args.controls,
        "controls": len(rows),
        "control_hz": 20,
        "video_fps": 20,
        "stopped_at_first_done": bool(rows and rows[-1]["done"]),
        "replay_release_row": args.release_row,
        "video": video.name,
        "video_sha256": sha256(video),
        "metrics": metrics.name,
        "metrics_sha256": sha256(metrics),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
except Exception as error:
    (output / "entry_failure.json").write_text(
        json.dumps(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    raise
finally:
    if writer is not None:
        writer.close()
    if env is not None:
        env.close()
    app.close()
