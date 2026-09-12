"""Constructed contact controls and live nonidentity reset; never task SR."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from isaaclab.app import AppLauncher
    ap = argparse.ArgumentParser()
    for key in ("bundle-root", "robot-urdf", "reference", "output-dir"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--enable-contact-readback", action="store_true")
    ap.add_argument("--tensor-pair-probe", action="store_true")
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    import torch
    from .env import Pour17JointEnv
    from .scene import isaac_scene_config
    cfg = isaac_scene_config(args.bundle_root, num_envs=1, seed=1701)
    cfg.sim.device, cfg.sim.log_dir = args.device, str(args.output_dir / "isaaclab_logs")
    cfg.bundle_root, cfg.robot_urdf, cfg.reference_path = map(str, (args.bundle_root, args.robot_urdf, args.reference))
    cfg.curriculum_stage, cfg.external_evaluator_controls_termination = 0, True
    cfg.contact_report_readback = args.enable_contact_readback
    cfg.tensor_pair_probe = args.tensor_pair_probe
    raw = Pour17JointEnv(cfg)
    try:
        if args.enable_contact_readback:
            # Keep initialized GPU tensors; changing this before construction
            # makes Isaac allocate CPU force-composer buffers (v4 failed).
            import carb
            carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
        from isaacsim.core.simulation_manager import SimulationManager
        device_record = {"configured_device": str(raw.device), "tensor_device": str(raw.robot.data.joint_pos.device),
                         "gpu_dynamics": SimulationManager.is_gpu_dynamics_enabled(),
                         "broadphase": SimulationManager.get_broadphase_type(),
                         "report_readback_requested": args.enable_contact_readback}
        (args.output_dir / "physics_device.json").write_text(json.dumps(device_record, indent=2) + "\n")
        assert device_record["gpu_dynamics"] and device_record["broadphase"] == "GPU"
        assert device_record["tensor_device"].startswith("cuda")
        raw.reset(seed=1701)
        if args.tensor_pair_probe:
            from .tensor_contact_probe import GpuPairContacts
            contact_reader = GpuPairContacts(raw)
        else:
            contact_reader = raw.pair_contacts
        nominal = raw.reset_state()
        initial_home_targets = raw._targets()[0].cpu().numpy().copy()
        raw.set_curriculum_stage(1)
        raw.reset(seed=100000)
        augmented = raw.reset_state()
        augmented_home_targets = raw._targets()[0].cpu().numpy().copy()
        raw.episode_length_buf[:] = 40
        operation_targets = {"wrist": raw._targets()[0].cpu().tolist(), "objects": raw._targets()[1].cpu().tolist(),
                             "hands": raw._targets()[2].cpu().tolist()}
        delta = np.array(augmented["placement_xy_m"])[0]
        assert np.linalg.norm(delta) > 0
        for oi in (0, 1):
            diff = np.array(augmented[f"object_{oi}_pose_wxyz"])[0, :3] - np.array(nominal[f"object_{oi}_pose_wxyz"])[0, :3]
            assert np.max(np.abs(diff - np.r_[delta, 0.])) < 1e-6
        assert np.max(np.abs(augmented_home_targets - initial_home_targets)) < 1e-6
        assert augmented["joint_q_rad"] == nominal["joint_q_rad"]
        (args.output_dir / "live_augmentation.json").write_text(json.dumps({"status": "PASS_MEASURED_RESET_READBACK",
            "nominal": nominal, "augmented": augmented, "operation_at_index40": operation_targets,
            "canonical_home_target_max_abs": float(np.max(np.abs(augmented_home_targets-initial_home_targets))),
            "unusable_samples_not_replaced": True}, indent=2) + "\n")
        raw.set_curriculum_stage(0)
        controls = []
        for kind in ("separated", "table", "right_hand", "left_hand", "right_tip", "left_tip"):
            raw.reset(seed=1701)
            poses = raw.reset_object_pose.clone()
            if kind != "table":
                poses[:, :3] = torch.tensor([[1., -.6, 1.8], [1., .6, 1.8]], device=raw.device)
            if kind.endswith("hand") or kind.endswith("tip"):
                si = 0 if kind.startswith("right") else 1
                side = ("right", "left")[si]
                body_name = f"{side}_index_elastomer" if kind.endswith("tip") else f"{side}_hand_C_MC"
                bi = list(raw.robot.body_names).index(body_name)
                # Deliberate local overlap constructs a positive control only.
                # It never changes a formal reset, reference, gate or score.
                poses[si, :3] = raw.robot.data.body_pos_w[0, bi] - raw.scene.env_origins[0]
            for oi, asset in enumerate((raw.bottle, raw.cup)):
                asset.write_root_pose_to_sim(poses[oi:oi+1])
                asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=raw.device))
            samples = []
            for step in range(36):
                raw.scene.write_data_to_sim()
                raw.sim.step(render=False)
                raw.scene.update(raw.physics_dt)
                hand, tips, table = contact_reader.read()
                samples.append({"physics_step": step + 1, "hand": hand.tolist(), "tips": tips.tolist(),
                                "table": table.tolist(), "G1_force_gt05": raw._tip_contacts().cpu().tolist()})
            controls.append({"control": kind, "samples": samples, "events": contact_reader.evidence()})
        hits = {}
        for row in controls:
            hits[row["control"]] = {field: np.any(np.array([s[field] for s in row["samples"]]), axis=(0, 1)).tolist()
                                    for field in ("hand", "tips", "table", "G1_force_gt05")}
        passed = (not np.any(hits["separated"]["hand"]) and not np.any(hits["separated"]["table"])
                  and all(hits["table"]["table"]))
        for si, side in enumerate(("right", "left")):
            passed &= bool(hits[side + "_hand"]["hand"][si])
            passed &= bool(np.any(hits[side + "_tip"]["tips"][si]))
            passed &= bool(np.any(hits[side + "_tip"]["G1_force_gt05"][si]))
        result = {"status": "PASS" if passed else "FAIL_CONTACT_BACKEND_CONTROLS", "constructed_only_not_SR": True,
                  "hits": hits, "controls": controls,
                  "note": "actual paired contact reports + filtered G1 normal forces; no inferred contact from motion"}
        (args.output_dir / "contact_validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result["status"], flush=True)
        if not passed:
            raise RuntimeError("contact positive/negative controls did not all pass; training not released")
    except BaseException:
        import traceback
        (args.output_dir / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        raw.close()
        app.close()


if __name__ == "__main__":
    main()
