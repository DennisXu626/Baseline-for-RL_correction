"""Assemble WP2 delivery from completed evidence; never launches a job."""
from pathlib import Path
import json
import hashlib
import cv2


BASE = Path(__file__).resolve().parents[2]
ROOT = BASE / "artifacts/pour17_v1/wp2_20260912_01"


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(name, value):
    path = ROOT / name
    if path.exists():
        raise RuntimeError(f"delivery does not overwrite: {path}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def main():
    ppo = read("output/ppo_v1/ppo_update_result.json")
    batch = read("output/batch_v1/batch_eval_result.json")
    internal = read("output/ppo_v1/internal_eval_4096/internal_eval.json")
    contacts = read("output/contacts_v9_dynamic_sensor/contact_validation.json")
    scalar = read("output/cpu_tests_v4/batch_eval_validation.json")
    reward = read("output/cpu_runtime_v1/runtime_torch_reward_comparison.json")
    augmentation = read("output/contacts_v9_dynamic_sensor/live_augmentation.json")
    ledger = read("output/development_training_ledger.json")
    assert ppo["status"] == "PASS" and ppo["completed_steps"] == 8192
    assert batch["actual_physical_slot_transitions"] == 32*903
    assert len(batch["episodes"]) == 32 and contacts["status"] == "PASS"
    assert sum(r["reserved_steps"] for r in ledger) <= 65536
    assert sum(r.get("training_wall_s", 0) for r in ledger) <= 3600
    camera_rows = []
    for job, mode, expected in (("camera_black_v1", "probe", 9),
                                ("old_seed_0_black_v1", "eval", 904),
                                ("old_seed_1_black_v1", "eval", 904),
                                ("openloop_black_v2", "openloop", 271)):
        result = read(f"output/{job}/result.json")
        video = ROOT / f"output/{job}/{mode}.mp4"
        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        decoded = []
        for index in (0, count//2, count-1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            assert ok and frame.std() > 1, (job, index)
            decoded.append(index)
        cap.release()
        assert count == expected == result["camera"]["frames"] and fps == 20
        assert result["camera"]["nonblack"] and result["camera"]["moving"]
        camera_rows.append(dict(job=job, path=str(video), sha256=digest(video),
            steps=result["steps"], frames=count, fps=fps, sampled_decoded_frames=decoded,
            all_recorded_frames_nonblack=True, moving=True,
            task_success=result.get("success"), gates=result.get("gates"),
            label="NEW_EXECUTION_NOT_RECOVERED_OLD_VIDEO", camera=result["camera"]["camera_eye"]))
    put("camera_validation.json", dict(status="PASS_RECORDING_CHECKS_NOT_TASK_SUCCESS", recordings=camera_rows,
        visual_observations=["Black table, both hands and cup/bottle visible in the fixed camera.",
            "Open-loop completed step40: visible separation between robot hand and green reference markers.",
            "Green markers are reference targets, not physical objects or contact evidence."],
        limitations="One camera cannot certify all occluded geometry, penetration or physical safety; pixels never enter policy."))
    put("training_mechanism_report.json", dict(status="PASS_INTEGRATION_NOT_REFERENCE_READINESS",
        source_to_port="source_to_port.md", observation_dim=544, action_dim=58,
        source_reward_cases=len(reward["rows"]), max_torch_source_error=max(x["abs_error"] for x in reward["rows"]),
        contact_controls=list(contacts["hits"]), contact_status=contacts["status"],
        live_augmentation_status=augmentation["status"],
        live_augmentation_xy_m=augmentation["augmented"]["placement_xy_m"],
        canonical_home_target_max_abs=augmentation["canonical_home_target_max_abs"],
        curriculum_actual_episodes=len(internal["records"]), curriculum_events=ppo["curriculum_events"],
        no_actual_stage_promotion=True, stage_branch_fixture="output/cpu_tests_v4/curriculum_branch_test.json",
        warm_start=ppo["warm_start"], complete_ppo_rollouts=2, completed_global_transitions=8192,
        parameter_change_l2=ppo["parameter"]["change_l2"], losses_finite=ppo["losses_finite"],
        reload=ppo["reload"], rng_restore_next_draw_equal=ppo["training_state_restore_next_rng_draw_equal"],
        resume_limitations="No interrupted PPO was resumed. Tested load/action equivalence and RNG restoration; new physical episodes, not simulator bit-state replay. WP1 cannot resume WP2.",
        development_ledger=ledger, inputs=ppo["inputs"]))
    put("batch_eval_validation.json", dict(status="PASS_BATCH_INTEGRATION_NOT_FORMAL_EVALUATION",
        fixed_trace_status=scalar["status"], scalar_step_comparisons=scalar["scalar_step_comparisons"],
        synthetic_trace_not_SR=True, live_result=batch,
        limitation="Development seed1701 model,32 episodes; not the formal512 result. Positive G2/G3/G4 only covered by fixed trace when absent from live run."))
    # A conservative direct extrapolation includes the frequent first callback.
    train_s = 3002368 / ppo["timing"]["global_transitions_per_s"]
    eval_s = batch["projected_512_s_with_25pct_margin"]
    put("throughput.json", dict(status="MEASURED_DEVELOPMENT_ONLY",
        training=ppo["timing"], internal_25_wall_s=internal["wall_s_including_kit"], batch=batch["timing"],
        formal_transitions=3002368, direct_training_extrapolation_s=train_s,
        training_with_25pct_margin_s=train_s*1.25,
        projected_512_s_with_25pct_margin=eval_s,
        combined_with_margins_s=train_s*1.25+eval_s,
        limitations=["Measured episodes terminate at40 steps and curriculum stayed0. Later-stage and successful long-episode PPO throughput is not measured.",
            "The direct extrapolation includes first-callback overhead every8192 steps; not a timing guarantee.",
            "Positive live certification and recovery overhead remain unmeasured. Formal start is not authorized."]))
    put("wp2_report.json", dict(status="INTEGRATION_DELIVERED_REFERENCE_GATE_NOT_READY", formal_launch=False,
        repaired=["Black diffuse_color only; snapshot/diff/hashes preserved.",
            "GPU detailed contacts: dynamic sensors and own-object/static-collider filters; missing data fails explicitly.",
            "544-dimensional state interface, source reward/warm-start, physical XY reset, completed-step40 curriculum boundary.",
            "PPO callback/checkpoint/RNG restore and per-slot batch evaluator."],
        measured_passed=["Both old903-step failures re-recorded with904 real frames each.",
            "Six contact controls, live nonidentity reset, pinned reward cases and989 scalar/batch trace comparisons.",
            "Fresh8192 PPO:2 complete rollouts; finite losses, changed parameters, exact reload actions, true RNG restore check.",
            "25 genuine stochastic callback episodes;0 successes, no stage promotion.",
            "32 physical evaluator slots each executed903 steps; recorded result/actual reset/timing."],
        not_completed=["Nominal reference does not enter pregrasp threshold: step40 right0.12955m/left0.17663m vs0.05m.",
            "Original left141 (S1 index143) has0.23942m reference IK position residual with three arm limits active.",
            "No demonstrated successful live stage promotion or positive live G2–G4; synthetic coverage is not physical success.",
            "Colleague alternate-root interpretation overlay not performed; no alternate reference imported.",
            "Formal training readiness is not established; no formal3M or final512 launched."],
        method_decisions_pending=["Reviewer must decide how to address the residual reference/physical mismatch; no target projection, reset relocation, limit/PD/A/mean change authorized or applied."],
        pregrasp_evidence="reference_execution_layers_v1.json",
        budget=dict(actual_training_transitions=8192, reserved_training_transitions=sum(x["reserved_steps"] for x in ledger),
            training_wall_s=sum(x.get("training_wall_s", 0) for x in ledger), cap_transitions=65536, cap_wall_s=3600,
            budget_reset=False, no_additional_training_for_low_score=True),
        unchanged=["registered physical parameters/control/reset", "W1/S1 rules", "A/mean/finger limits/alpha", "G1–G4", "M0 BLOCKED", "M1-A NOT PASSED"],
        checkpoint=ppo["reload"]["model"], handoff="Stop at reviewer milestone. Commands are review-only and must not be run without release."))
    # Source identity is separate from immutable result provenance.
    paths = sorted(set(list((BASE/"adaptation/pour17_v1").rglob("*.py"))
        + list((BASE/"configs/pour17_v1").glob("*"))
        + [p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in str(p)
           and p.suffix in (".json", ".npz", ".zip", ".pt", ".mp4", ".md", ".gz", ".txt")]))
    put("wp2_delivery_manifest.json", dict(schema="actual_file_sha256_v1", script_sha256=digest(Path(__file__)),
        files=[dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in paths if p.is_file()],
        note="Manifest excludes itself; immutable old artifacts remain intact. Remote source hash list is independently captured."))
    print(json.dumps({"status": "DELIVERED_NOT_READY", "root": str(ROOT)}))


if __name__ == "__main__":
    main()
