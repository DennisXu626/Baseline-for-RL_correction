"""Explicit review camera; images never enter a policy observation.

WP1 compatibility is selected before importing its environment. Recordings
are new executions, not recovery or bit-exact playback of the old episodes.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


def camera_environment(base):
    class RecordedEnv(base):
        def _setup_scene(self):
            super()._setup_scene()
            from isaaclab.sensors import Camera, CameraCfg
            import isaaclab.sim as sim_utils
            cfg = CameraCfg(prim_path="/World/WP2ReviewCamera", width=960, height=720,
                data_types=["rgb"], update_period=0.,
                spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=36.,
                                               clipping_range=(0.05, 20.)))
            self.review_camera = Camera(cfg)
            self.scene.sensors["review_camera"] = self.review_camera
    return RecordedEnv


class Capture:
    def __init__(self, env, output: Path):
        import cv2
        import torch
        self.env, self.output = env, output
        self.writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 20., (960, 720))
        if not self.writer.isOpened():
            raise RuntimeError("MP4 writer unavailable")
        self.env.review_camera.set_world_poses_from_view(
            eyes=torch.tensor([[1.8, 1.6, 2.4]], device=env.device),
            targets=torch.tensor([[-0.14, 0.08, 1.0]], device=env.device))
        self.rows, self.last = [], None
        self.target_markers = []
        if hasattr(env, "palm_local"):
            import omni.usd
            from pxr import UsdGeom, Gf
            stage = omni.usd.get_context().get_stage()
            for i in range(20):
                sphere = UsdGeom.Sphere.Define(stage, f"/World/WP2ReferenceMarkers/point_{i}")
                sphere.CreateRadiusAttr(.004)
                sphere.CreateDisplayColorAttr([Gf.Vec3f(.15, 1., .25)])
                self.target_markers.append(UsdGeom.Xformable(sphere).AddTranslateOp())
        # Render-only initialization: does not advance a control or physics step.
        for _ in range(8):
            env.sim.render()
        env.review_camera.update(0., force_recompute=True)

    def frame(self, label):
        import cv2
        if self.target_markers:
            from pxr import Gf
            points = self.env._targets()[2][0].detach().cpu().numpy().reshape(-1, 3)
            for op, point in zip(self.target_markers, points):
                op.Set(Gf.Vec3d(*point.tolist()))
            label += " | green dots=reference only"
        self.env.sim.render()
        self.env.review_camera.update(0., force_recompute=True)
        rgb = self.env.review_camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        rgb = np.asarray(rgb, dtype=np.uint8)
        delta = None if self.last is None else float(np.abs(rgb.astype(float) - self.last.astype(float)).mean())
        self.rows.append({"frame": len(self.rows), "mean": float(rgb.mean()), "std": float(rgb.std()),
                          "delta_from_previous": delta})
        if len(self.rows) <= 6:
            cv2.imwrite(str(self.output.with_name(self.output.stem + f"_probe_{len(self.rows):02}.png")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.last = rgb.copy()
        if rgb.std() <= 1. or rgb.max() == 0:
            raise RuntimeError("explicit camera image has no visible scene; full recording stopped")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1)
        self.writer.write(bgr)

    def close(self):
        self.writer.release()
        result = {"label": "REEXECUTED_DIAGNOSTIC_NOT_RECOVERED_VIDEO", "fps": 20,
                  "camera_eye": [1.8, 1.6, 2.4], "camera_target": [-.14, .08, 1.],
                  "frames": len(self.rows), "frame_statistics": self.rows,
                  "nonblack": bool(self.rows) and all(r["std"] > 1 for r in self.rows),
                  "moving": any((r["delta_from_previous"] or 0) > .01 for r in self.rows)}
        self.output.with_suffix(".camera.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    for key in ("bundle-root", "robot-urdf", "reference", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--compat", action="store_true")
    parser.add_argument("--mode", choices=("probe", "openloop", "eval"), default="probe")
    parser.add_argument("--seed", type=int, default=20260829)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.enable_cameras:
        raise ValueError("--enable_cameras is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    import torch
    from .geometry import sha256
    package = "adaptation.pour17_v1" + (".wp1_compat" if args.compat else "")
    base = importlib.import_module(package + ".env").Pour17JointEnv
    cfg = importlib.import_module(package + ".scene").isaac_scene_config(args.bundle_root, num_envs=1, seed=args.seed)
    cfg.sim.device = args.device
    cfg.sim.log_dir = str(args.output_dir / "isaaclab_logs")
    cfg.bundle_root, cfg.robot_urdf, cfg.reference_path = map(str, (args.bundle_root, args.robot_urdf, args.reference))
    cfg.probe_only = False
    cfg.curriculum_stage = 0
    cfg.external_evaluator_controls_termination = True
    env = camera_environment(base)(cfg)
    recorder = None
    try:
        obs, _ = env.reset(seed=args.seed)
        reset = env.reset_state()
        recorder = Capture(env, args.output_dir / (args.mode + ".mp4"))
        evaluator = None
        if args.mode == "eval":
            from .batch_eval import load_evaluator, criteria_overlay
            evaluator = load_evaluator(args.bundle_root)
            tape, mb, mc = criteria_overlay(args.bundle_root, args.reference, args.output_dir)
            prog = evaluator.PourProgress(str(tape), mouth_local_bot=mb, mouth_local_cup=mc, mouth_gate=evaluator.MOUTH_GATE)
            from stable_baselines3 import PPO
            model = PPO.load(args.model, device=args.device)
            if model.observation_space.shape != (int(cfg.observation_space),):
                raise RuntimeError("checkpoint observation schema mismatch")
        count = 8 if args.mode == "probe" else (903 if args.mode == "eval" else env.reference_length - 1)
        recorder.frame(f"{'WP1 compat' if args.compat else 'WP2'} | {args.mode} | seed {args.seed} | reset step0")
        traces, progress_rows = [], []
        started = time.perf_counter()
        for step in range(count):
            alpha = 0.
            if evaluator:
                alpha = evaluator.cert_alpha(prog.cert_phase, prog.cert_t)
                if alpha > 0 or not args.compat:
                    env.apply_certification_offset(alpha)
                action, _ = model.predict(obs["policy"].cpu().numpy(), deterministic=True)
            else:
                q = env.reference_q[min(step + 1, env.reference_length - 1)]
                action = (2 * (q - env.lower) / (env.upper - env.lower) - 1).clamp(-1, 1)[None].cpu().numpy()
            obs, reward, terminated, truncated, extras = env.step(torch.as_tensor(action, device=env.device))
            info = env.evaluator_info()
            state = {"step": step + 1, "action": action[0].tolist(),
                     "q": env.robot.data.joint_pos[0, env.joint_ids].cpu().tolist(),
                     "wrist": env._wrist_poses()[0].cpu().tolist(), "objects": env._object_poses()[0].cpu().tolist(),
                     "hand_ten": env._landmarks()[0].cpu().tolist(), "reference_index": int(env._reference_index()[0]),
                     "desired_wrist": env._targets()[0][0].cpu().tolist(), "target_hand": env._targets()[2][0].cpu().tolist(),
                     "pads": env._tip_contacts()[0].cpu().tolist(), "alpha": alpha}
            if not args.compat:
                state["hand_six_error_m"] = torch.linalg.vector_norm(env._hand_six() - env._target_hand_six(), dim=-1).mean(-1)[0].cpu().tolist()
                state["contact_reports"] = env.pair_contacts.evidence()
                if step + 1 in (39, 40, 41, 141):
                    from .geometry import matrix_to_quat_wxyz
                    env._ensure_arm_ik()
                    reference_index = state["reference_index"]
                    reference_q = env.reference_q[reference_index].cpu().numpy()
                    fk_rows = {}
                    for si, side in enumerate(("right", "left")):
                        sl = slice(7 * si, 7 * si + 7)
                        p, r = env._arm_ik[side].fk(reference_q[sl])
                        actual_p, actual_r = env._arm_ik[side].fk(np.asarray(state["q"])[sl])
                        fk_rows[side] = {"reference_ik_fk_wrist": np.r_[p, matrix_to_quat_wxyz(r)].tolist(),
                            "actual_q_fk_wrist": np.r_[actual_p, matrix_to_quat_wxyz(actual_r)].tolist(),
                            "actual_q_fk_vs_runtime_position_m": float(np.linalg.norm(actual_p - np.asarray(state["wrist"])[si, :3])),
                            "reference_arm_q": reference_q[sl].tolist(),
                            "actual_arm_q": state["q"][sl],
                            "lower": env.lower[sl].cpu().tolist(), "upper": env.upper[sl].cpu().tolist()}
                    state["boundary_execution_layers"] = fk_rows
            traces.append(state)
            stop = False
            if evaluator:
                out = prog.step(info["object_0_pose"], info["object_1_pose"], info["arm_q_right"], info["arm_q_left"],
                                info["pads3"], info["wrist_right"], info["wrist_left"])
                fired = evaluator.classify_failure(prog, info["object_0_pose"], info["object_1_pose"])
                progress_rows.append({"step": step + 1, "out": out, "gates": dict(prog.g), "deadlines": fired})
                stop = out["done"] or bool(terminated[0])
            recorder.frame(f"{'WP1 compat' if args.compat else 'WP2'} | {args.mode} | seed {args.seed} | step {step+1}")
            if stop:
                break
        result = {"status": "RECORDED_REEXECUTION", "compat_schema_286": args.compat, "seed": args.seed,
                  "steps": len(traces), "wall_s": time.perf_counter() - started, "reset": reset,
                  "camera": recorder.close(), "progress": progress_rows,
                  "reference_sha256": sha256(args.reference), "model_sha256": sha256(args.model) if args.model else None,
                  "script_sha256": sha256(Path(__file__))}
        recorder = None
        if evaluator:
            result.update(success=bool(prog.g[4]), placed=bool(prog.placed), gates=dict(prog.g))
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x) + "\n")
        (args.output_dir / "trace.json").write_text(json.dumps(traces) + "\n")
        print(json.dumps({k: result[k] for k in ("status", "steps", "wall_s")}), flush=True)
    except BaseException:
        import traceback
        (args.output_dir / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if recorder:
            recorder.close()
        env.close()
        app.close()


if __name__ == "__main__":
    main()
