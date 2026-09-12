"""Summarize WP4 measured evidence without rerunning physics or optimization."""
from pathlib import Path
import json
import tarfile
import difflib
import numpy as np
from .geometry import sha256, quat_to_matrix_wxyz


def main():
    base = Path(__file__).resolve().parents[2]
    out = base/"artifacts/pour17_v1/wp4_20260912_01"
    read = lambda p: json.loads(p.read_text())
    def put(name, value):
        p = out/name
        assert not p.exists(), p
        p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    run = out/"output/reference_v1"
    trace = read(run/"trace.json")
    result = read(run/"result.json")
    assert len(trace)==270 and result["video_frames"]==271
    assert all(r["reference_index"]==[r["step"]]*2 for r in trace)
    static = read(out/"static_separation_gate.json")
    rebuild = read(out/"reference_build_v1/reference_rebuild_manifest.json")
    points = ("thumb_tip","thumb_middle","index_tip","index_middle","middle_tip","middle_middle","ring_tip","ring_middle","pinky_tip","pinky_middle")
    refs = read(run/"reset_world_readback.json")
    roots = np.array([r["root_transforms_xyzw"] for r in refs])
    root_drift = float(np.max(np.abs(roots-roots[0])))
    assert root_drift < 1e-6
    by_step = {r["completed_global_steps"]:r for r in refs if r["label"]=="reference"}
    command_error = max(float(np.max(np.abs(np.array(by_step[r["step"]]["physx_joint_targets"])-np.array(r["desired_joint_q"])))) for r in trace)
    assert command_error < 2e-6, "PhysX target differs from own commanded reference"
    joint_tracking = []
    for r in trace:
        if r["step"] in (29,40,143,270):
            actual = np.array(r["actual_q"])
            target = np.array(r["desired_joint_q"])
            joint_tracking.append(dict(step=r["step"],joint_names=refs[0]["joint_names"],
                target_rad=target.tolist(),actual_rad=actual.tolist(),error_rad=(actual-target).tolist(),
                actual_qd=by_step[r["step"]]["qd"]))
    put("command_following.json",dict(physx_target_vs_command_max_abs_rad=command_error,
        keyframes=joint_tracking,conclusion="Target wiring checked; following deviations do not identify a unique dynamics/contact cause. PD unchanged."))
    keyframes = []
    for r in trace:
        if r["step"] in (39,40,41,143,270):
            keyframes.append(r)
            for l in r["wrist_layers"]:
                angle = lambda a,b: float(np.arccos(np.clip((np.trace(quat_to_matrix_wxyz(np.array(a)[3:]) @ quat_to_matrix_wxyz(np.array(b)[3:]).T)-1)/2,-1,1)))
                l["ik_target_rotation_rad"] = angle(l["reference_ik_fk"],l["target"])
                l["actual_target_rotation_rad"] = angle(l["runtime_wrist"],l["target"])
                l["actual_q_fk_runtime_rotation_rad"] = angle(l["actual_q_fk"],l["runtime_wrist"])
    put("reference_execution_layers.json",dict(keyframes=keyframes,
        max_actual_q_fk_runtime_position_m=max(l["fk_runtime_position_m"] for r in trace for l in r["wrist_layers"]),
        error_definition="six point mean Euclidean distance: palm plus5tips, same WP2 definition; not ten-point RMS",
        reference_index_alignment="reset index0 then next-index commands1..270; each trace after completed control step"))
    contacts = {}
    for slot in range(2):
        for si, side in enumerate(("right","left")):
            hand = [r["step"] for r in trace if r["semantic_hand"][slot][si]]
            table = [r["step"] for r in trace if r["semantic_object_table"][slot][si]]
            contacts[f"slot{slot}_{side}"] = dict(hand_own_object_positive_steps=hand,
                hand_own_object_negative_steps=[r["step"] for r in trace if not r["semantic_hand"][slot][si]],
                own_object_table_positive_steps=table,
                own_object_table_negative_steps=[r["step"] for r in trace if not r["semantic_object_table"][slot][si]])
    table_contact = {}
    for r in trace:
        for name, counts in r["robot_table_active_counts"].items():
            table_contact.setdefault(name,[]).append(dict(step=r["step"],count=sum(counts)))
    geometric = []
    for g in read(run/"scene_geometry.json"):
        for slot in g["slots"]:
            positive = [dict(path=m["path"],body=m["body"],surface_triangles=m["surface_triangle_intersections"],witness=m["witness"])
                for m in slot["meshes"] if m["surface_triangle_intersections"]]
            geometric.append(dict(label=g["label"],slot=slot["slot"],surface_intersections=positive))
    put("dynamic_contact_geometry.json",dict(semantic_contacts=contacts,robot_table_contacts=table_contact,
        live_geometry=geometric,dynamic_scene_queries_used_for_verdict=False,
        caveat="Surface clipping uses live PhysX body transforms; positive source collision mesh is distinguished from cooked collision. Paired contacts support actual interaction, not certified collision-free motion."))
    # Snapshot only approved adaptation/config differences; old baseline files remain.
    with tarfile.open(out/"source_before.tar.gz") as archive:
        previous = {m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    current = {str(p.relative_to(base)).replace('\\','/'):p.read_bytes()
        for folder in ("adaptation/pour17_v1","configs/pour17_v1") for p in (base/folder).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts}
    changed = sorted(k for k in set(previous)|set(current) if previous.get(k)!=current.get(k))
    assert all(Path(k).name.startswith('wp4_') for k in changed), changed
    diff = ''.join(''.join(difflib.unified_diff(previous.get(k,b'').decode().splitlines(True),current.get(k,b'').decode().splitlines(True),fromfile='before/'+k,tofile='after/'+k)) for k in changed)
    (out/"source.diff").write_text(diff,encoding="utf-8")
    put("source_changes.json",dict(changed=changed,production_env_scene_unchanged=True,
        source_before_sha256=sha256(out/"source_before.tar.gz"),diff_sha256=sha256(out/"source.diff")))
    overlay = read(base/"configs/pour17_v1/wp4_root_overlay.json")
    put("common_world_interface.json",dict(overlay=overlay,overlay_sha256=sha256(base/"configs/pour17_v1/wp4_root_overlay.json"),
        base_bundle_archive_sha256="77edc03ca5be028fcbd8b6d978fd1a4072582b5c9904d603f5c86b29a70210f6",
        canonical_rule="base canonical q/qd and objects unchanged; root now spawn(-.75,0,0), not reset-write; old bundle immutable",
        required_other_method_fields=["world version/base hashes/root overlay", "env origins", "actual fixed root and arm_center env/world poses", "first/repeat q/qd/targets", "table/objects readback", "static cooked separation and pair contact", "own reference provenance"],
        vividex_effective_anchor_world_xyz=static["effective_fixed_anchor_world_xyz"],other_method="PENDING_NOT_COMMON_RUNTIME_FROZEN",
        prohibited="No cross-method robot q/reference sharing; no old checkpoint automatic continuation"))
    table_targets = {s:[points[i] for i,b in enumerate(rebuild["table_inside_targets"][s]["index40_inside_mask"]) if b] for s in ("right","left")}
    summary = dict(status="STATIC_WORLD_PASS_REFERENCE_REVIEW_REQUIRED_NOT_TRAINING_RELEASED",
        world_version=overlay["world_version"],root_reference_execution_drift_max_abs=root_drift,
        control_steps=40+270,slot_transitions=2*(40+270),max_allowed_slot_transitions=2048,new_ppo_steps=0,
        reference_samples=271,failed_ik={s:len(rebuild["ik"][s]["failure_indices"]) for s in ("right","left")},
        ik_wrist_error_m={s:rebuild["ik"][s]["wrist_error_m"] for s in ("right","left")},
        index40_six_point_mean_m=trace[39]["hand_six_error_m"],index143_six_point_mean_m=trace[142]["hand_six_error_m"],
        index40_targets_inside_table=table_targets,robot_table_contact_body_count=len(table_contact),
        contact_sign_evidence={k:{n:len(v) for n,v in value.items()} for k,value in contacts.items()},
        wall_s=dict(reset=read(out/"reset_hold_v1_process.json")["wall_s"],
                    reference=read(out/"reference_v1_process.json")["wall_s"],arm_rebuild=rebuild["wall_s"]),
        boundaries=["root separation does not remove unchanged table-inside targets", "IK failures do not prove global unreachability", "contact and execution gaps not explained by one cause", "other method pending; no common runtime freeze", "no automatic old checkpoint resume, PPO, formal3M or512evaluation"],
        gates=dict(formal3M="NOT_RELEASED",M0="BLOCKED",M1A="NOT_PASSED"))
    put("wp4_report.json",summary)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
