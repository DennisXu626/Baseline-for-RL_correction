"""Record one deterministic three-view episode for a training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--runtime_root", type=Path, required=True)
parser.add_argument("--v12_root", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output_root", type=Path, required=True)
parser.add_argument("--controls", type=int, default=300)
parser.add_argument(
    "--release_row", type=int,
    help="Stage-1 pin length to replay at. Defaults to the value recorded in the "
         "checkpoint's sidecar JSON, which is the level the policy was trained at. "
         "Replaying an annealed policy at the initial pin length does not evaluate it.",
)
parser.add_argument(
    "--no_grasp_curriculum", action="store_true",
    help="Replay with the curriculum off, i.e. the original baseline: no pin at all. "
         "This is the final-evaluation configuration.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
official = args.v12_root.resolve() / "third_party/h2s2r_official"

sys.path.insert(0, str(args.v12_root.resolve()))
sys.path.insert(0, str(args.runtime_root.resolve()))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(official))
from human2sim2robot.ppo.ppo_player import PlayerConfig, PpoPlayer, PpoPlayerConfig  # noqa: E402
from human2sim2robot.ppo.utils.dict_to_dataclass import dict_to_dataclass  # noqa: E402
from human2sim2robot.ppo.utils.network import NetworkConfig  # noqa: E402
from rl_rebuild.baselines.h2s2r.official_ppo_adapter import OfficialPpoEnvAdapter  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.h2s2r_clean3.cfg import build_cfg  # noqa: E402
from tasks.h2s2r_clean3.env import Clean3H2S2REnv  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


output = args.output_root.resolve()
output.mkdir(parents=True, exist_ok=False)

# Replay at the curriculum level this checkpoint was trained at.  The trainer
# records it in the sidecar JSON next to the .pth; an explicit --release_row wins,
# and --no_grasp_curriculum replays the unpinned baseline instead.
sidecar = args.checkpoint.resolve().with_suffix(".json")
sidecar_release_row = None
if sidecar.is_file():
    sidecar_release_row = json.loads(sidecar.read_text(encoding="utf-8")).get("release_row_cur")
replay_release_row = args.release_row if args.release_row is not None else sidecar_release_row
if not args.no_grasp_curriculum and replay_release_row is None:
    raise SystemExit(
        f"no release_row for this checkpoint: {sidecar} is missing or has no "
        "release_row_cur. Pass --release_row explicitly, or --no_grasp_curriculum "
        "to replay the unpinned baseline."
    )
cfg = build_cfg(
    runtime_root=args.runtime_root,
    v12_root=args.v12_root,
    num_envs=1,
    seed=42,
    # Reproduce training termination.  external_eval=True forces every done
    # flag to false, contradicting this recorder's one-episode contract.
    external_eval=False,
    record_cameras=True,
    grasp_curriculum=not args.no_grasp_curriculum,
    release_row_start=None if args.no_grasp_curriculum else int(replay_release_row),
)
print(
    f"[CLEAN3] milestone replay grasp_curriculum={cfg.grasp_curriculum} "
    f"release_row={replay_release_row if not args.no_grasp_curriculum else 'n/a'} "
    f"(sidecar={sidecar_release_row}, override={args.release_row})",
    flush=True,
)
cfg.sim.device = args.device
cfg.sim.render_interval = 1
raw = Clean3H2S2REnv(cfg, render_mode="rgb_array")
wrapped = GymStyleEnvWrapper(raw, clip_actions=cfg.clip_actions)
official_env = OfficialPpoEnvAdapter(wrapped)
with (HERE / "ppo_h2s2r_lstm.yaml").open("r", encoding="utf-8") as stream:
    network = dict_to_dataclass(yaml.safe_load(stream)["network"], NetworkConfig)
player = PpoPlayer(
    ppo_player_config=PpoPlayerConfig(normalize_input=True, normalize_value=True, clip_actions=True, device=args.device),
    player_config=PlayerConfig(deterministic=True, print_stats=False),
    network_config=network,
    env=official_env,
)
player.restore(args.checkpoint.resolve())
obs = wrapped.reset()
player.get_batch_size(obs["obs"], batch_size=1)
player.init_rnn()
video = output / "three_view.mp4"
# One frame is written for every 20 Hz policy control.  Encode at the same
# cadence so playback time equals simulated time.
writer = imageio.get_writer(video, fps=20, codec="libx264", quality=7, macro_block_size=None, pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])
rows = []
with torch.inference_mode():
    for control in range(args.controls):
        # Capture the current episode state before step().  DirectRLEnv resets a
        # finished environment inside step(), so rendering afterwards would add
        # one frame from the next episode to this video.
        snapshot = raw.runtime_snapshot()
        raw.sim.render()
        frames = raw.camera_frames()
        writer.append_data(np.concatenate([frames[name][0].cpu().numpy().astype(np.uint8) for name in ("front", "side", "top")], axis=1))
        pre_action_state = {
            "pre_action_release_row": int(raw.release_row[0]),
            "pre_action_pinned": bool(raw._pinned[0]),
            "pre_action_grasp_certified": bool(raw._grasp_certified[0]),
            "pre_action_grasp_success": bool(raw._grasp_success[0]),
            "pre_action_grasp_dropped": bool(raw._grasp_dropped[0]),
            "pre_action_cert_run": int(raw._cert_run[0]),
            "pre_action_hold_run": int(raw._hold_run[0]),
        }
        actions = player.get_action(obs["obs"], is_deterministic=True).clamp(-1.0, 1.0)
        obs, reward, done, _ = wrapped.step(actions)
        if player.is_rnn and bool(done.any()):
            done_ids = done.nonzero(as_tuple=False).squeeze(1)
            for state in player.states:
                state[:, done_ids, :] = 0.0
        rows.append({
            "control": control,
            "reward": float(reward[0]),
            "done": bool(done[0]),
            "action_abs_max": float(actions.abs().max()),
            "plate_z": float(snapshot["plate_pose"][0, 2]),
            "sponge_z": float(snapshot["sponge_pose"][0, 2]),
            "reference_index": int(snapshot["reference_index"][0]),
            "snapshot_before_action": True,
            **pre_action_state,
        })
        # A reset has already happened inside wrapped.step() when done is true.
        # The frame above was captured before that step, so the next episode
        # cannot contaminate this video's final frame.
        if bool(done[0]):
            break
writer.close()
(output / "step_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
(output / "result.json").write_text(json.dumps({
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_sha256": digest(args.checkpoint.resolve()),
    "requested_max_controls": args.controls,
    "controls": len(rows),
    "stopped_at_first_done": bool(rows and rows[-1]["done"]),
    "video": video.name,
    "video_sha256": digest(video),
    # Replay configuration, so a video can never be misread as evaluating a
    # curriculum level it was not recorded at.
    "grasp_curriculum": bool(cfg.grasp_curriculum),
    "replay_release_row": None if args.no_grasp_curriculum else int(replay_release_row),
    "release_row_source": (
        "disabled" if args.no_grasp_curriculum
        else ("explicit_flag" if args.release_row is not None else "checkpoint_sidecar")
    ),
}, indent=2) + "\n", encoding="utf-8")
wrapped.close()
app.close()
