"""Independent scalar PourProgress objects sharing only batched physics.

No alternate/vectorized criteria implementation and no H2S2R evaluator import.
Only a whitelisted common-criteria tape overlay is passed to PourProgress; its
arm stance comes from this adapter's canonical home, never Ours robot q.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from .geometry import sha256


def load_evaluator(bundle):
    path = Path(bundle) / "evaluator/run_eval.py"
    expected = "40e6b157e836cb3ea3cd754791c1231ced7e9ef633c184d09a6ebe125f7d6d6b"
    if sha256(path) != expected:
        raise RuntimeError("registered evaluator hash mismatch")
    progress = path.with_name("progress.py")
    if sha256(progress) != "abdbfad4cda8c9f15ea96ec5feb781c27ee958ad575454f068272f433b6fb03e":
        raise RuntimeError("registered progress hash mismatch")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("wp2_registered_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def criteria_overlay(bundle, reference, output):
    e = load_evaluator(bundle)
    tape = Path(bundle) / "reference/pour17_reference_v2__2ed81358__653rows.npz"
    if sha256(tape) != "a164d34e3ad0f86ca7aaae24f425a0f065474fe438fa2d2779c55edcaaa5865f":
        raise RuntimeError("common criteria tape hash mismatch")
    allowed = ("source", "obj_pos_0", "obj_pos_1", "obj_quat_0", "obj_quat_1",
               "conf_pos_0", "conf_pos_1", "conf_rot_0", "conf_rot_1")
    with np.load(tape, allow_pickle=False) as data:
        values = {k: data[k] for k in allowed}
    with np.load(reference, allow_pickle=False) as data:
        names = list(data["joint_names"].astype(str))
        q = data["joint_q_rad"][-1]
    for side, letter in (("right", "R"), ("left", "L")):
        indices = [names.index(f"{letter}_arm_j{i}") for i in range(1, 8)]
        values[side + "_q"] = np.repeat(q[indices][None], len(values["source"]), axis=0)
    rows = np.flatnonzero(values["source"] == 1)
    mb = e.mouth_local(values, rows, 1, e.MOUTH_HALF_BOTTLE)
    mc = e.mouth_local(values, rows, 0, e.MOUTH_HALF_CUP)
    target = Path(output) / "criteria_only_overlay.npz"
    np.savez_compressed(target, **values)
    manifest = {"original_tape_sha256": sha256(tape), "read_fields": list(allowed),
                "never_read": ["right_q", "left_q", "finger_q", "human_q", "cert_arm7"],
                "stance_source": "own reference final canonical home arm joints by name",
                "reference_sha256": sha256(reference), "overlay_sha256": sha256(target),
                "mouth_b": mb.tolist(), "mouth_c": mc.tolist(), "actor_access_to_overlay": False}
    (Path(output) / "criteria_overlay_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target, mb, mc


class Slots:
    def __init__(self, evaluator, tape, mb, mc, seeds):
        self.e = evaluator
        self.seeds = list(seeds)
        self.progress = [evaluator.PourProgress(str(tape), mouth_local_bot=mb, mouth_local_cup=mc,
                        mouth_gate=evaluator.MOUTH_GATE) for _ in seeds]
        self.steps = np.zeros(len(seeds), dtype=np.int64)
        self.finished = np.zeros(len(seeds), dtype=bool)
        self.deadlines = [set() for _ in seeds]
        self.results = [None for _ in seeds]

    def alphas(self):
        return [0. if self.finished[i] else self.e.cert_alpha(p.cert_phase, p.cert_t)
                for i, p in enumerate(self.progress)]

    def step(self, infos, terminated=None):
        outputs = []
        for i, (p, info) in enumerate(zip(self.progress, infos)):
            if self.finished[i]:
                outputs.append(None)
                continue
            out = p.step(info["object_0_pose"], info["object_1_pose"], info["arm_q_right"], info["arm_q_left"],
                         info["pads3"], info["wrist_right"], info["wrist_left"])
            self.steps[i] += 1
            self.deadlines[i].update(self.e.classify_failure(p, info["object_0_pose"], info["object_1_pose"]))
            term = bool(terminated[i]) if terminated is not None else False
            cap = self.steps[i] >= self.e.MAX_CONTROL_STEPS
            if out["done"] or term or cap:
                self.finished[i] = True
                self.results[i] = {"seed": self.seeds[i], "steps": int(self.steps[i]), "success": bool(p.g[4]),
                    "gates": dict(p.g), "placed": bool(p.placed), "deadlines_fired": sorted(self.deadlines[i]),
                    "fail_reason_progress": out["fail"] or ("timeout" if cap and not(out["done"] or term) else None)}
            outputs.append(out)
        return outputs


def main():
    import argparse
    from isaaclab.app import AppLauncher
    ap = argparse.ArgumentParser()
    for key in ("bundle-root", "robot-urdf", "reference", "model", "output-dir"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--seed-base", type=int, default=20260829)
    ap.add_argument("--long-horizon", action="store_true", help="continue physical slots to 903, freeze already terminal criteria")
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    app = AppLauncher(args).app
    import torch
    from stable_baselines3 import PPO
    from .env import Pour17JointEnv
    from .scene import isaac_scene_config
    e = load_evaluator(args.bundle_root)
    tape, mb, mc = criteria_overlay(args.bundle_root, args.reference, args.output_dir)
    cfg = isaac_scene_config(args.bundle_root, num_envs=args.num_envs, seed=args.seed_base)
    cfg.sim.device, cfg.sim.log_dir = args.device, str(args.output_dir / "isaaclab_logs")
    cfg.bundle_root, cfg.robot_urdf, cfg.reference_path = map(str, (args.bundle_root, args.robot_urdf, args.reference))
    cfg.external_evaluator_controls_termination, cfg.curriculum_stage = True, 0
    raw = Pour17JointEnv(cfg)
    model = PPO.load(args.model, device=args.device)
    if model.observation_space.shape != (int(cfg.observation_space),):
        raise RuntimeError("WP2 evaluator checkpoint schema mismatch")
    if getattr(model, "wp2_reference_sha256", None) != sha256(args.reference):
        raise RuntimeError("WP2 checkpoint/reference hash mismatch")
    resets, records, timing = [], [], {"reset_s": 0., "inference_s": 0., "physics_s": 0., "criteria_s": 0., "certification_s": 0.}
    init_s = time.perf_counter() - started
    actual_control_steps = 0
    try:
        for offset in range(0, args.episodes, args.num_envs):
            size = min(args.num_envs, args.episodes - offset)
            seed = args.seed_base + offset
            now = time.perf_counter()
            obs, _ = raw.reset(seed=seed)
            resets.append(raw.reset_state())
            timing["reset_s"] += time.perf_counter() - now
            slots = Slots(e, tape, mb, mc, range(seed, seed + size))
            for k in range(903):
                now = time.perf_counter()
                for slot, alpha in enumerate(slots.alphas()):
                    raw.apply_certification_offset(alpha, env_id=slot)
                timing["certification_s"] += time.perf_counter() - now
                now = time.perf_counter()
                actions, _ = model.predict(obs["policy"].cpu().numpy(), deterministic=True)
                timing["inference_s"] += time.perf_counter() - now
                now = time.perf_counter()
                obs, _, term, _, _ = raw.step(torch.as_tensor(actions, device=raw.device))
                actual_control_steps += 1
                timing["physics_s"] += time.perf_counter() - now
                now = time.perf_counter()
                slots.step(raw.evaluator_infos()[:size], term.cpu().numpy())
                timing["criteria_s"] += time.perf_counter() - now
                if slots.finished.all() and not args.long_horizon:
                    break
            records.extend(slots.results)
            print(f"[WP2 eval] finished {len(records)}/{args.episodes}", flush=True)
        wall = time.perf_counter() - started
        result = {"status": "COMPLETED_DEVELOPMENT_BATCH", "episodes": records, "num_slots": args.num_envs,
                  "timing": timing, "startup_s": init_s, "wall_s": wall,
                  "success_rate": sum(r["success"] for r in records) / len(records), "denominator": len(records),
                  "long_horizon_cost_measurement": args.long_horizon,
                  "actual_batched_control_steps": actual_control_steps,
                  "actual_physical_slot_transitions": actual_control_steps * args.num_envs,
                  "projected_512_s_with_25pct_margin": init_s + (wall - init_s) * 512 / args.episodes * 1.25,
                  "model_sha256": sha256(args.model), "reference_sha256": sha256(args.reference),
                  "script_sha256": sha256(Path(__file__)), "certification_trace": raw.certification_trace}
        (args.output_dir / "batch_eval_result.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output_dir / "reset_manifest.json").write_text(json.dumps(resets, indent=2) + "\n")
    except BaseException:
        import traceback
        (args.output_dir / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        raw.close()
        app.close()


if __name__ == "__main__":
    main()
