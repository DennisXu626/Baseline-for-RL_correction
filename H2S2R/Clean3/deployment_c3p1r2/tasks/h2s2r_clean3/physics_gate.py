"""One bounded four-environment physical gate for Clean3 C3-P1R2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--runtime_root", type=Path, required=True)
parser.add_argument("--v12_root", type=Path, required=True)
parser.add_argument("--output_root", type=Path, required=True)
parser.add_argument("--controls", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.controls != 400:
    raise SystemExit("C3-P1R2 physical gate is frozen at 400 controls")
app = AppLauncher(args).app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.baselines.h2s2r.contract import ControlledSide  # noqa: E402
from tasks.h2s2r_clean3.cfg import build_cfg  # noqa: E402
from tasks.h2s2r_clean3.env import Clean3H2S2REnv  # noqa: E402


VIEWS = ("front", "side", "top")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu(value, dtype=np.float32):
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def main() -> None:
    started = time.monotonic()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = build_cfg(runtime_root=args.runtime_root, v12_root=args.v12_root, num_envs=4, seed=42, external_eval=True, record_cameras=True)
    cfg.sim.device = args.device
    cfg.sim.render_interval = 1
    print("[CLEAN3-GATE] stage=construct_env begin", flush=True)
    env = Clean3H2S2REnv(cfg, render_mode="rgb_array")
    print("[CLEAN3-GATE] stage=construct_env complete", flush=True)
    rows = []
    frames = {(env_id, name): [] for env_id in range(4) for name in VIEWS}
    phase = "fixed_pd"
    control_index = 0
    substep_index = 0
    consecutive_deep = torch.zeros(4, dtype=torch.long, device=env.device)
    sustained_deep = torch.zeros(4, dtype=torch.bool, device=env.device)
    reset_q = env.reset_q58.repeat(4, 1)

    def record_substep(current: Clean3H2S2REnv) -> None:
        nonlocal substep_index
        snapshot = current.runtime_snapshot()
        depth = snapshot["penetration_depth_m"]
        consecutive_deep.copy_(torch.where(depth > 0.002, consecutive_deep + 1, torch.zeros_like(consecutive_deep)))
        sustained_deep.logical_or_(consecutive_deep >= 2)
        rows.append({
            "phase": np.asarray(phase), "control_index": np.asarray(control_index, np.int64),
            "substep_index": np.asarray(substep_index, np.int64),
            "joint_q": cpu(snapshot["joint_q"]), "joint_qd": cpu(snapshot["joint_qd"]),
            "target_q": cpu(snapshot["target_q"]), "plate_pose": cpu(snapshot["plate_pose"]),
            "sponge_pose": cpu(snapshot["sponge_pose"]), "contact_force_n": cpu(snapshot["contact_force_n"]),
            "penetration_depth_m": cpu(depth), "reference_index": cpu(snapshot["reference_index"], np.int64),
        })
        substep_index += 1

    def capture() -> None:
        env.sim.render()
        for name, values in env.camera_frames().items():
            batch = values.detach().cpu().numpy().astype(np.uint8, copy=False)
            for env_id in range(4):
                frames[(env_id, name)].append(batch[env_id].copy())

    initial = env.runtime_snapshot()
    initial_penetration = cpu(initial["penetration_depth_m"])
    # Phase 1: fixed articulation PD target; objects remain fully dynamic.
    fixed_initial_plate = cpu(initial["plate_pose"])
    fixed_initial_sponge = cpu(initial["sponge_pose"])
    fixed_target = env.hand.data.joint_pos.clone()
    for control_index in range(20):
        phase = "fixed_pd"
        for _ in range(12):
            env.hand.set_joint_position_target(fixed_target)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(cfg.sim.dt)
            record_substep(env)
        capture()
    fixed_final = env.runtime_snapshot()

    # Reset once, then keep one unbroken 20-control zero + 360-control small-random FABRICS episode.
    env._reset_idx(torch.arange(4, device=env.device))
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(cfg.sim.dt)
    fabric_initial = env.runtime_snapshot()
    fabric_initial_plate = cpu(fabric_initial["plate_pose"])
    fabric_initial_sponge = cpu(fabric_initial["sponge_pose"])
    env.substep_callback = record_substep
    generator = torch.Generator(device=env.device)
    generator.manual_seed(42)
    fabric_results = []
    for local_control in range(380):
        control_index = 20 + local_control
        phase = "fabric_zero" if local_control < 20 else "fabric_small_random"
        actions = torch.zeros(4, 22, device=env.device)
        if local_control >= 20:
            actions.uniform_(-0.05, 0.05, generator=generator)
        result = env.step(actions)
        fabric_results.append((cpu(result[2], np.bool_), cpu(result[3], np.bool_)))
        capture()
        if bool(sustained_deep.any()):
            break
    env.substep_callback = None
    fabric_zero_final_row = next(row for row in reversed(rows) if int(row["control_index"]) == 39)

    fixed_final_plate = cpu(fixed_final["plate_pose"])
    fixed_final_sponge = cpu(fixed_final["sponge_pose"])
    fixed_forces = cpu(fixed_final["contact_force_n"])
    zero_plate = fabric_zero_final_row["plate_pose"]
    zero_sponge = fabric_zero_final_row["sponge_pose"]
    zero_forces = fabric_zero_final_row["contact_force_n"]
    fixed_held = (
        (fixed_final_plate[:, 2] > cfg.table_top_z) & (fixed_final_sponge[:, 2] > cfg.table_top_z)
        & (np.linalg.norm(fixed_final_plate[:, :3] - fixed_initial_plate[:, :3], axis=1) < 0.03)
        & (np.linalg.norm(fixed_final_sponge[:, :3] - fixed_initial_sponge[:, :3], axis=1) < 0.03)
        & (fixed_forces[:, :5].max(axis=1) > 0.5) & (fixed_forces[:, 5:].max(axis=1) > 0.5)
    )
    zero_held = (
        (zero_plate[:, 2] > cfg.table_top_z) & (zero_sponge[:, 2] > cfg.table_top_z)
        & (np.linalg.norm(zero_plate[:, :3] - fabric_initial_plate[:, :3], axis=1) < 0.03)
        & (np.linalg.norm(zero_sponge[:, :3] - fabric_initial_sponge[:, :3], axis=1) < 0.03)
        & (zero_forces[:, :5].max(axis=1) > 0.5) & (zero_forces[:, 5:].max(axis=1) > 0.5)
    )
    zero_rows = [row for row in rows if str(row["phase"]) == "fabric_zero"]
    first_step_delta = float(np.max(np.abs(zero_rows[11]["joint_q"] - reset_q.cpu().numpy())))
    first_second_delta = float(max(np.max(np.abs(row["joint_q"] - reset_q.cpu().numpy())) for row in zero_rows))
    first_second_target_delta = float(max(np.max(np.abs(row["target_q"] - reset_q.cpu().numpy())) for row in zero_rows))
    limits = cpu(env.hand.data.soft_joint_pos_limits[:, env.sim_joint_ids])
    all_q = np.stack([row["joint_q"] for row in rows])
    finite = bool(np.isfinite(all_q).all())
    within_limits = bool((all_q >= limits[None, :, :, 0] - 1e-6).all() and (all_q <= limits[None, :, :, 1] + 1e-6).all())
    cadence = dict(env.runtime_counts)
    fabric_controls = int(cadence["controls"])
    cadence_pass = cadence == {
        "controls": fabric_controls, "physics_substeps": fabric_controls * 12,
        "right_set_targets": fabric_controls, "left_set_targets": fabric_controls,
        "right_advance": fabric_controls * 12, "left_advance": fabric_controls * 12,
    }
    result = {
        "schema": "h2s2r_clean3_c3p1r2_physics_gate_v1",
        "controls_completed": len(frames[(0, "front")]), "substeps_recorded": len(rows),
        "initial_penetration_depth_m": initial_penetration.tolist(),
        "sustained_deep_penetration": sustained_deep.cpu().tolist(),
        "fixed_pd_held_both_count": int(fixed_held.sum()), "fixed_pd_held_both": fixed_held.tolist(),
        "fabric_zero_held_both_count": int(zero_held.sum()), "fabric_zero_held_both": zero_held.tolist(),
        "first_zero_control_max_joint_delta_rad": first_step_delta,
        "first_zero_second_max_joint_delta_rad": first_second_delta,
        "first_zero_second_max_target_delta_rad": first_second_target_delta,
        "all_joint_values_finite": finite, "all_joint_values_within_limits": within_limits,
        "cadence": cadence, "cadence_pass": cadence_pass, "hold_zero": 0,
    }
    result["components"] = {
        "complete_400_controls": result["controls_completed"] == 400,
        "no_initial_or_sustained_deep_penetration": bool((initial_penetration <= 0.002).all() and not sustained_deep.any()),
        "fixed_pd_dynamic_hold_2_of_4": int(fixed_held.sum()) >= 2,
        "fabric_zero_dynamic_hold_2_of_4": int(zero_held.sum()) >= 2,
        "zero_joint_change_le_0p25": max(first_step_delta, first_second_delta, first_second_target_delta) <= 0.25,
        "finite_and_within_limits": finite and within_limits,
        "bilateral_cadence": cadence_pass,
    }
    result["physics_gate_pass"] = bool(all(result["components"].values()))
    result["elapsed_seconds"] = time.monotonic() - started
    atomic_json(output / "physics_gate.json", result)
    keys = tuple(rows[0])
    np.savez_compressed(output / "raw_substeps.npz", **{key: np.stack([row[key] for row in rows]) for key in keys})
    for (env_id, name), values in frames.items():
        with imageio.get_writer(output / f"clean3_gate_env{env_id:02d}_{name}.mp4", fps=20, codec="libx264", quality=7, macro_block_size=None, pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"]) as writer:
            for frame in values:
                writer.append_data(frame)
    index = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(output.iterdir()) if path.is_file()}
    atomic_json(output / "artifact_index.json", index)
    print(json.dumps(result, indent=2), flush=True)
    env.close()


try:
    main()
    print("[CLEAN3-GATE] stage=run complete", flush=True)
except BaseException as error:
    failure = {
        "stage": "physics_gate_entry",
        "type": type(error).__name__,
        "message": repr(error),
        "traceback": traceback.format_exc(),
    }
    try:
        atomic_json(args.output_root / "entry_failure.json", failure)
    except BaseException:
        traceback.print_exc()
    print("[CLEAN3-ENTRY-FAILURE] physics_gate", flush=True)
    traceback.print_exc()
    raise
finally:
    app.close()
