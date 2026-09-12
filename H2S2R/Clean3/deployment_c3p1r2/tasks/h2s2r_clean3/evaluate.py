"""Frozen initial/final 16-episode Clean3 external evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--runtime_root", type=Path, required=True)
parser.add_argument("--v12_root", type=Path, required=True)
parser.add_argument("--initial_checkpoint", type=Path, required=True)
parser.add_argument("--final_checkpoint", type=Path, required=True)
parser.add_argument("--output_root", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
OFFICIAL = args.v12_root.resolve() / "third_party/h2s2r_official"

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(OFFICIAL))
from clean3_adapter.external_eval import Clean3ExternalEvaluatorV1  # noqa: E402
from human2sim2robot.ppo.ppo_player import PlayerConfig, PpoPlayer, PpoPlayerConfig  # noqa: E402
from human2sim2robot.ppo.utils.dict_to_dataclass import dict_to_dataclass  # noqa: E402
from human2sim2robot.ppo.utils.network import NetworkConfig  # noqa: E402
from rl_rebuild.baselines.h2s2r.official_ppo_adapter import OfficialPpoEnvAdapter  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.h2s2r_clean3.cfg import build_cfg  # noqa: E402
from tasks.h2s2r_clean3.env import Clean3H2S2REnv  # noqa: E402


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_vertices(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                values.append([float(v) for v in line.split()[1:4]])
    result = np.asarray(values)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise RuntimeError(f"invalid mesh vertices: {path}")
    return result


def matrices(quat_wxyz: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()


def mosaic(batch: np.ndarray) -> np.ndarray:
    canvas = Image.new("RGB", (4 * batch.shape[2], 4 * batch.shape[1]))
    draw = ImageDraw.Draw(canvas)
    for index, frame in enumerate(batch):
        x, y = (index % 4) * batch.shape[2], (index // 4) * batch.shape[1]
        canvas.paste(Image.fromarray(frame), (x, y))
        draw.rectangle((x, y, x + 70, y + 18), fill=(0, 0, 0))
        draw.text((x + 3, y + 2), f"episode {42 + index}", fill=(255, 255, 255))
    return np.asarray(canvas)


def main() -> None:
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = build_cfg(runtime_root=args.runtime_root, v12_root=args.v12_root, num_envs=16, seed=42, external_eval=True, record_cameras=True)
    cfg.sim.device = args.device
    cfg.sim.render_interval = 1
    print("[CLEAN3-EVAL] stage=construct_env begin", flush=True)
    raw = Clean3H2S2REnv(cfg, render_mode="rgb_array")
    print("[CLEAN3-EVAL] stage=construct_env complete", flush=True)
    env = GymStyleEnvWrapper(raw, clip_actions=cfg.clip_actions)
    official_env = OfficialPpoEnvAdapter(env)
    with (HERE / "ppo_h2s2r_lstm.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    network = dict_to_dataclass(config["network"], NetworkConfig)
    perturb = np.stack([np.random.RandomState(seed).uniform(-0.005, 0.005, 58) for seed in range(42, 58)]).astype(np.float32)
    np.save(output / "frozen_joint_perturbations_seed42_57.npy", perturb)
    raw.set_evaluation_joint_offsets(torch.as_tensor(perturb, device=raw.device))

    plate_vertices = load_vertices(args.runtime_root / "assets/objects/object_0/object_mesh_scaled_final.obj")
    sponge_vertices = load_vertices(args.runtime_root / "assets/objects/object_1/object_mesh_scaled_final.obj")
    plate_tree, sponge_tree = cKDTree(plate_vertices), cKDTree(sponge_vertices)
    rx90 = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
    reset = np.load(args.runtime_root / "data/clean3_c3p1_runtime_inputs.npz", allow_pickle=False)
    reset_plate_r = matrices(reset["reset_object_0_pose_wxyz"][None])[0]
    reset_sponge_r = matrices(reset["reset_object_1_pose_wxyz"][None])[0]
    relative_initial = rx90 @ reset_plate_r.T @ reset_sponge_r @ rx90.T
    footprint_vectors_initial = np.c_[reset["footprint_xy"], -float(reset["sponge_face_offset"]) * np.ones(len(reset["footprint_xy"]))]
    profile_r, profile_z = reset["plate_profile_r"], reset["plate_profile_z"]
    evaluations = {}

    for label, checkpoint in (("initial", args.initial_checkpoint), ("final", args.final_checkpoint)):
        checkpoint = checkpoint.resolve()
        player = PpoPlayer(
            ppo_player_config=PpoPlayerConfig(normalize_input=True, normalize_value=True, clip_actions=True, device=args.device),
            player_config=PlayerConfig(deterministic=True, print_stats=False), network_config=network, env=official_env,
        )
        player.restore(checkpoint)
        obs = env.reset()
        player.get_batch_size(obs["obs"], batch_size=1)
        player.init_rnn()
        evaluators = [Clean3ExternalEvaluatorV1() for _ in range(16)]
        writers = {
            view: imageio.get_writer(output / f"{label}_episodes42_57_{view}.mp4", fps=20, codec="libx264", quality=7, macro_block_size=None, pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])
            for view in ("front", "side", "top")
        }
        rows = []
        substep_depths = []

        def substep_callback(current):
            depth = current.penetration_depth().detach().cpu().numpy()
            substep_depths.append(depth.copy())
            for index, evaluator in enumerate(evaluators):
                evaluator.update_physics(depth[index:index + 1])

        raw.substep_callback = substep_callback
        with torch.inference_mode():
            for control in range(400):
                before = len(substep_depths)
                actions = player.get_action(obs["obs"], is_deterministic=True).clamp(-1.0, 1.0)
                obs, _, simulator_done, _ = env.step(actions)
                if bool(simulator_done.any()):
                    raise RuntimeError("external evaluation simulator unexpectedly terminated")
                snapshot = raw.runtime_snapshot()
                plate = snapshot["plate_pose"].detach().cpu().numpy()
                sponge = snapshot["sponge_pose"].detach().cpu().numpy()
                left_tips = snapshot["left_tips"].detach().cpu().numpy()
                right_tips = snapshot["right_tips"].detach().cpu().numpy()
                contact = raw.plate_sponge_contact_force().detach().cpu().numpy()
                rp, rs = matrices(plate[:, 3:]), matrices(sponge[:, 3:])
                plate_normal = rp[:, :, 1]
                centers = np.einsum("ij,njk,nk->ni", rx90, rp.transpose(0, 2, 1), sponge[:, :3] - plate[:, :3])
                footprint_xy, gaps = [], []
                pad_distances = np.empty((16, 2, 5), dtype=np.float32)
                for index in range(16):
                    relative = rx90 @ rp[index].T @ rs[index] @ rx90.T
                    delta = relative @ relative_initial.T
                    points = centers[index] + footprint_vectors_initial @ delta.T
                    footprint_xy.append(points[:, :2])
                    top = np.interp(np.linalg.norm(points[:, :2], axis=1), profile_r, profile_z)
                    gaps.append(points[:, 2] - top)
                    left_local = (rp[index].T @ (left_tips[index] - plate[index, :3]).T).T
                    right_local = (rs[index].T @ (right_tips[index] - sponge[index, :3]).T).T
                    pad_distances[index, 0] = plate_tree.query(left_local)[0]
                    pad_distances[index, 1] = sponge_tree.query(right_local)[0]
                    evaluators[index].update_control(
                        plate_center=plate[index, :3], plate_normal=plate_normal[index],
                        sponge_center_plate_xy=centers[index, :2], footprint_points_plate_xy=footprint_xy[-1],
                        footprint_gap_m=gaps[-1], plate_sponge_contact_force_n=float(contact[index]),
                        object_centers_z=np.array([plate[index, 2], sponge[index, 2]]),
                        all_pad_surface_distances_m=pad_distances[index],
                    )
                raw.sim.render()
                for view, batch in raw.camera_frames().items():
                    writers[view].append_data(mosaic(batch.detach().cpu().numpy().astype(np.uint8)))
                rows.append({
                    "control": np.asarray(control, np.int64), "actions": actions.detach().cpu().numpy().astype(np.float32),
                    "joint_q": snapshot["joint_q"].detach().cpu().numpy().astype(np.float32),
                    "target_q": snapshot["target_q"].detach().cpu().numpy().astype(np.float32),
                    "plate_pose": plate.astype(np.float32), "sponge_pose": sponge.astype(np.float32),
                    "contact_force_n": snapshot["contact_force_n"].detach().cpu().numpy().astype(np.float32),
                    "substep_penetration_depth_m": np.stack(substep_depths[before:]),
                })
        raw.substep_callback = None
        for writer in writers.values():
            writer.close()
        results = [evaluator.result() for evaluator in evaluators]
        valid = len(results)
        success = sum(bool(value["success"]) for value in results)
        evaluations[label] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "attempted": 16, "valid": valid, "invalid": 0, "successes": success,
            "success_rate": success / valid if valid else None, "episodes": results,
        }
        keys = tuple(rows[0])
        np.savez_compressed(output / f"{label}_raw_trajectories.npz", **{key: np.stack([row[key] for row in rows]) for key in keys})
    result = {"schema": "h2s2r_clean3_c3p1r2_external_eval_v1", "seeds": list(range(42, 58)), "same_perturbations": True, "deterministic": True, "evaluations": evaluations}
    atomic_json(output / "evaluation.json", result)
    index = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(output.iterdir()) if path.is_file()}
    atomic_json(output / "artifact_index.json", index)
    print(json.dumps(result, indent=2), flush=True)
    env.close()


try:
    main()
    print("[CLEAN3-EVAL] stage=evaluation complete", flush=True)
except BaseException as error:
    try:
        atomic_json(args.output_root / "entry_failure.json", {
            "stage": "evaluation_entry", "type": type(error).__name__,
            "message": repr(error), "traceback": traceback.format_exc(),
        })
    except BaseException:
        traceback.print_exc()
    print("[CLEAN3-ENTRY-FAILURE] evaluation", flush=True)
    traceback.print_exc()
    raise
finally:
    app.close()
