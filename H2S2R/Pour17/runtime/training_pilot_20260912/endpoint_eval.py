"""One frozen 16-episode endpoint evaluation for the bounded V13 256-env pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from runtime.training_pilot_20260912.resume_observation import from_environment
observation = from_environment()

from isaaclab.app import AppLauncher


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--bundle_root", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if args.out.exists():
    raise FileExistsError(f"refusing to overwrite {args.out}")

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

slot = isaac_slot("h2s2r-v12-endpoint-eval")
app = observation.call('AppLauncher', AppLauncher, args).app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402

sys.path.insert(0, str(ROOT / "third_party/h2s2r_official"))
from human2sim2robot.ppo.ppo_player import PpoPlayer, PpoPlayerConfig, PlayerConfig  # noqa: E402
from human2sim2robot.ppo.utils.dict_to_dataclass import dict_to_dataclass  # noqa: E402
from human2sim2robot.ppo.utils.network import NetworkConfig  # noqa: E402
from rl_rebuild.baselines.h2s2r.official_ppo_adapter import OfficialPpoEnvAdapter  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from rl_rebuild.baselines.h2s2r.contract import ControlledSide  # noqa: E402
from tasks.h2s2r_pour17.right_zero_margin_install import (  # noqa: E402
    install_right_training_patch, verify_installed_right_fabric, verify_limit_dispatch,
)

expected_patch_class, patch_record = install_right_training_patch()

from tasks.h2s2r_pour17.right_env import build_cfg  # noqa: E402
from tasks.h2s2r_pour17.training_pilot_audit import (  # noqa: E402
    AuditedRightBottleEnv, CAMERAS, atomic_json,
)


NUM_ENVS = 16
observation.install_initialization()
MAX_CONTROLS = 903
SEEDS = list(range(1000, 1016))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quaternion_angle(quaternions: np.ndarray, initial: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(quaternions * initial[None, :], axis=1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(dots)


def auxiliary_grasp_metric(bottle: np.ndarray, palm: np.ndarray) -> dict:
    bottle_initial = bottle[0].copy()
    palm_initial = palm[0].copy()
    relative = bottle[:, :3] - palm[:, :3]
    angles = quaternion_angle(bottle[:, 3:7], bottle_initial[3:7])
    for end in range(4, len(bottle)):
        window = slice(end - 4, end + 1)
        bottle_rise = bottle[window, 2] - bottle_initial[2]
        palm_rise = palm[window, 2] - palm_initial[2]
        points = relative[window]
        pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        pre = slice(0, end + 1)
        checks = {
            "bottle_and_palm_rise_10mm_for_window": bool(
                np.all(bottle_rise >= 0.010) and np.all(palm_rise >= 0.010)
            ),
            "relative_drift_max_5mm": bool(pairwise.max() <= 0.005),
            "pre_window_bottle_angle_max_0p4rad": bool(angles[pre].max() <= 0.4),
            "pre_window_bottle_drop_max_5mm": bool(
                bottle[pre, 2].min() >= bottle_initial[2] - 0.005
            ),
        }
        if all(checks.values()):
            return {
                "numeric_candidate": True,
                "first_window_end_control": end,
                "checks": checks,
                "window_relative_drift_m": float(pairwise.max()),
                "pre_window_max_angle_rad": float(angles[pre].max()),
                "pre_window_min_bottle_delta_z_m": float(
                    bottle[pre, 2].min() - bottle_initial[2]
                ),
                "video_exclusion_review": "PENDING",
            }
    return {
        "numeric_candidate": False,
        "first_window_end_control": None,
        "video_exclusion_review": "NOT_APPLICABLE_UNLESS_NUMERIC_CANDIDATE",
    }


def load_all_chunks(chunk_root: Path) -> dict[str, np.ndarray]:
    files = sorted(chunk_root.glob("controls_*.npz"))
    if not files:
        raise RuntimeError("endpoint evaluator produced no raw trajectory chunks")
    parts: dict[str, list[np.ndarray]] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                parts.setdefault(key, []).append(archive[key])
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()}


def validate_v13_manifest(manifest):
    if not manifest.get("formal_run") or manifest.get("run_kind") != "training_pilot":
        raise RuntimeError("endpoint evaluation accepts only this V13 formal pilot")
    budget = manifest.get('training_pilot_budget', {})
    ppo = manifest.get('official_ppo', {}).get('ppo', {})
    if (manifest.get('num_envs'), budget.get('num_envs'), budget.get('horizon_length'),
            budget.get('minibatch_size'), budget.get('mini_epochs'), ppo.get('seq_length')) != (256, 256, 16, 4096, 4, 16):
        raise RuntimeError("checkpoint is not from the approved V13 256-env batch protocol")
    return True


def main() -> None:
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out / "status.json", {"state": "starting"})
    run_dir = args.checkpoint.resolve().parent.parent
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_v13_manifest(manifest)
    if manifest["reference_start_index"] != 14 or manifest["input_regime"] != "estimated":
        raise RuntimeError("checkpoint protocol differs from the frozen endpoint protocol")
    if manifest["isolation"]["policy_action_dim"] != 11:
        raise RuntimeError("checkpoint action interface is not 11-dimensional")

    np.random.seed(1000)
    torch.manual_seed(1000)
    synergy = ROOT / "tasks/h2s2r_pour17/artifacts/synergies"
    cfg = build_cfg(
        bundle_root=args.bundle_root,
        num_envs=NUM_ENVS,
        seed=1000,
        input_regime="estimated",
        reference_start_index=14,
        right_synergy_npz=synergy / "right_synergy_pca5.npz",
        left_synergy_npz=synergy / "left_synergy_pca5.npz",
        external_evaluator_controls_termination=False,
    )
    if cfg.warmup_clamp_steps != 0:
        raise RuntimeError("endpoint evaluation requires hold=0")
    cfg.sim.device = args.device
    cfg.sim.render_interval = cfg.decimation
    cfg.viewer = ViewerCfg(
        eye=CAMERAS["front"][0], lookat=CAMERAS["front"][1],
        origin_type="env", env_index=0, resolution=(640, 480),
    )
    raw = AuditedRightBottleEnv(
        cfg,
        render_mode="rgb_array",
        audit_root=args.out / "audit",
        controls_per_chunk=64,
        suppress_auto_reset=True,
    )
    verify_installed_right_fabric(raw.controller.right.fabric, expected_patch_class)
    limit_dispatch = verify_limit_dispatch(raw.controller.right.fabric, raw.controller.left.fabric, expected_patch_class)
    atomic_json(args.out / 'limit_dispatch.json', limit_dispatch)
    env = GymStyleEnvWrapper(raw, clip_actions=cfg.clip_actions)
    player = observation.call('policy.construct', PpoPlayer,
        ppo_player_config=PpoPlayerConfig(
            normalize_input=True, normalize_value=True,
            clip_actions=True, device=args.device,
        ),
        player_config=PlayerConfig(deterministic=True, print_stats=False),
        network_config=dict_to_dataclass(manifest["official_ppo"]["network"], NetworkConfig),
        env=OfficialPpoEnvAdapter(env),
    )
    player.restore(args.checkpoint.resolve())
    env.seed(1000)
    obs = observation.call('policy.reset', env.reset)
    player.get_batch_size(obs["obs"], batch_size=NUM_ENVS)
    player.init_rnn()
    if not player.is_rnn or player.states is None:
        raise RuntimeError("endpoint checkpoint did not initialize recurrent state")
    if player.actions_num != 11 or player.obs_shape != (171,):
        raise RuntimeError("endpoint policy interface differs from 171/11")

    # The reset has no randomized pose.  The only randomized training-state term
    # is reference speed; draw it independently from the unchanged distribution.
    speed_factors = np.asarray(
        [np.random.default_rng(seed).uniform(0.5, 1.0) for seed in SEEDS],
        dtype=np.float32,
    )
    raw.clock.speed_factors.copy_(torch.as_tensor(speed_factors, device=raw.device))
    initial_bottle = raw._object_poses()[1].detach().cpu().numpy().copy()
    palm_xyz, palm_quat = raw._wrist(ControlledSide.RIGHT)
    initial_palm = torch.cat((palm_xyz, palm_quat), dim=1).detach().cpu().numpy().copy()

    media_root = args.out / "media"
    media_root.mkdir()
    raw.delivery.preview(tuple(range(NUM_ENVS)))
    raw.left_delivery.preview((0,))
    frame_counts = {(env_id, view): 0 for env_id in range(NUM_ENVS) for view in CAMERAS}
    for env_id in range(NUM_ENVS):
        episode_dir = media_root / f"episode_{env_id:02d}_seed_{SEEDS[env_id]}"
        episode_dir.mkdir()

    active = torch.ones(NUM_ENVS, dtype=torch.bool, device=raw.device)
    end_controls = np.full(NUM_ENVS, -1, dtype=np.int64)
    try:
        with torch.no_grad():
            for control in range(MAX_CONTROLS):
                action = player.get_action(obs["obs"], is_deterministic=True)
                action[~active] = 0.0
                obs, reward, done, info = env.step(action)
                just_done = active & done.to(dtype=torch.bool)
                for env_id in range(NUM_ENVS):
                    if not bool(active[env_id]):
                        continue
                    frames = raw.delivery.capture(env_id, int(raw._media_episode_ids[env_id]),
                        control, 'terminal_reset_suppressed' if bool(just_done[env_id]) else 'post_control',
                        reset=bool(just_done[env_id]))
                    episode_dir = media_root / f"episode_{env_id:02d}_seed_{SEEDS[env_id]}"
                    raw.delivery.append(episode_dir, frames, prefix='')
                    if env_id == 0:
                        left_frames = raw.left_delivery.capture(0, int(raw._media_episode_ids[0]),
                            control, 'terminal_reset_suppressed' if bool(just_done[0]) else 'post_control',
                            reset=bool(just_done[0]))
                        raw.left_delivery.append(episode_dir / 'left_cup', left_frames, prefix='')
                    for view in CAMERAS:
                        frame_counts[(env_id, view)] += 1
                if just_done.any():
                    indices = just_done.nonzero(as_tuple=False).flatten()
                    end_controls[indices.detach().cpu().numpy()] = control
                    for state in player.states:
                        state[:, indices, :] = 0.0
                    active &= ~just_done
                atomic_json(
                    args.out / "status.json",
                    {"state": "running", "control": control + 1, "active": int(active.sum())},
                )
                if not bool(active.any()):
                    break
        end_controls[end_controls < 0] = control
        raw.close_audit()
    finally:
        raw.delivery.close()
        raw.left_delivery.close()

    arrays = load_all_chunks(args.out / "audit/raw_chunks")
    episodes = []
    valid = 0
    numeric_candidates = 0
    for env_id, seed in enumerate(SEEDS):
        end = int(end_controls[env_id])
        episode_dir = media_root / f"episode_{env_id:02d}_seed_{seed}"
        trajectory_path = episode_dir / "trajectory.npz"
        payload = {
            key: value[: end + 1, env_id]
            for key, value in arrays.items()
            if value.ndim >= 2 and value.shape[1] == NUM_ENVS
        }
        payload["initial_bottle_pose_env_xyz_wxyz"] = initial_bottle[env_id]
        payload["initial_right_palm_pose_env_xyz_wxyz"] = initial_palm[env_id]
        np.savez_compressed(trajectory_path, **payload)
        metric = auxiliary_grasp_metric(
            np.vstack((initial_bottle[env_id], payload["bottle_pose_env_xyz_wxyz"])),
            np.vstack((initial_palm[env_id], payload["right_palm_pose_env_xyz_wxyz"])),
        )
        numeric_candidates += int(metric["numeric_candidate"])
        media_complete = all(frame_counts[(env_id, view)] == end + 1 for view in CAMERAS)
        finite = all(
            np.isfinite(value).all() for value in payload.values() if value.dtype.kind in "fcu"
        )
        episode_valid = bool(media_complete and finite)
        valid += int(episode_valid)
        episodes.append({
            "episode": env_id,
            "seed": seed,
            "reference_speed_factor": float(speed_factors[env_id]),
            "controls": end + 1,
            "valid": episode_valid,
            "termination": "normal_done" if end < MAX_CONTROLS - 1 else "max_903",
            "trajectory": str(trajectory_path.resolve()),
            "trajectory_sha256": sha256(trajectory_path),
            "video_frames": {view: frame_counts[(env_id, view)] for view in CAMERAS},
            "auxiliary_grasp": metric,
        })

    result = {
        "schema": "h2s2r_v13_256_endpoint_eval_v1",
        "status": "COMPLETE_PENDING_VISUAL_REVIEW" if valid == NUM_ENVS else "INVALID",
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        "run_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
        "patch": patch_record,
        "limit_dispatch": limit_dispatch,
        "left_fixed_command_isolation": observation.trace.state['left_fixed_command_isolation'],
        "left_advance_calls": observation.trace.state['left_advance_calls'],
        "policy": {"official_lstm": True, "deterministic": True, "autonomous": True,
                   "source_commands": False, "normalizer_restored": True},
        "seeds": SEEDS,
        "valid_n": valid,
        "invalid_n": NUM_ENVS - valid,
        "numeric_candidates_pending_video_exclusion": numeric_candidates,
        "episodes": episodes,
        "counts": raw.audit_counts,
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(args.out / "evaluation.json", result)
    atomic_json(args.out / "status.json", {"state": "completed", "status": result["status"]})
    env.close()


observation.run(main, app.close)
