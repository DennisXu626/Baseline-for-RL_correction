"""Summarize completed WP3 evidence and one unexecuted common-world proposal."""
from pathlib import Path
import json
import tarfile
import difflib
import numpy as np
from .geometry import sha256


def main():
    base=Path(__file__).resolve().parents[2]
    root=base/"artifacts/pour17_v1/wp3_20260912_01"
    read=lambda name:json.loads((root/name).read_text())
    def put(name,value):
        p=root/name
        if p.exists():
            raise RuntimeError(f"existing versioned report: {p}")
        p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    states=read("output/reset_hold_v2/reset_world_readback.json")
    geometries=read("output/geometry_query_v3/scene_geometry.json")
    before=next(s for s in states if s["label"]=="first_reset_after_write_data")
    again=next(s for s in states if s["label"]=="repeat_reset_after_write_data")
    origins=np.array(before["env_origins"])
    controlled=np.array(before["q"])
    reset_checks={key:float(np.max(np.abs(np.array(before[key])-np.array(again[key]))))
        for key in ("q","qd","physx_joint_targets","root_transforms_xyzw","body_transforms_world_xyzw")}
    assert max(reset_checks.values()) < 1e-6
    root_env=np.array(before["root_transforms_xyzw"])[:,:3]-origins
    assert np.max(np.abs(root_env)) < 1e-6
    held=[s for s in states if s["label"]=="hold"]
    names=before["robot_body_names"]
    fixed_names=["vega_1p_mobile","vega_1p_torso_l1","vega_1p_torso_l2","vega_1p_torso_l3","arm_center"]
    fixed_rows={n:np.array(before["body_transforms_world_xyzw"])[:,names.index(n)].tolist() for n in fixed_names}
    reset_summary=dict(status="RESET_WIRING_CONSISTENT_NOT_SCENE_GEOMETRY_PASS",
        env_origins=origins.tolist(),fixed_body_world_xyzw=fixed_rows,
        robot_root_env_xyz=root_env.tolist(),first_vs_repeat_max_abs=reset_checks,
        q_vs_target_at_reset_max_abs=float(np.max(np.abs(controlled-np.array(before["physx_joint_targets"])))),
        hold_q_vs_home_max_rad=max(float(np.max(np.abs(np.array(s["q"])-controlled))) for s in held),
        constructor_target_pending_write_max_rad=float(np.max(np.abs(np.array(states[0]["q"])-np.array(states[0]["physx_joint_targets"])))),
        constructor_target_note="set_joint_position_target updates a pending buffer; public reset and scene write synchronize before the first action. Constructor-only readback is not a completed reset transaction.",
        first_vs_repeat_home_hold_identical=max(float(np.max(np.abs(np.array(held[i]["q"])-np.array(held[i+20]["q"])))) for i in range(20)),
        root_usd="authored defaultPrim translate0; root_joint fixed to world with localPos0/localPos1=0; slot origin applied once",
        canonical_home="registered neutral joint start, not Ours near-grasp",
        no_root_reset_write=True,full_readback="output/reset_hold_v2/reset_world_readback.json")
    put("reset_world_summary.json",reset_summary)
    geometry_rows=[]
    for slot,s in enumerate(geometries[-1]["slots"]):
        shapes={c["path"]:c for c in s["all_collision_shapes"]}
        fixed_hits=sorted(set(h["collision"] for h in s["physx_cooked_overlap_box"]["0.001"]["hits"]
                             if "/vega_1p_torso_" in h["rigid_body"]))
        surfaces=[dict(path=m["path"],body=m["body"],triangles=m["surface_triangle_intersections"],witness=m["witness"])
            for m in s["meshes"] if m["surface_triangle_intersections"] and "/visuals/" in m["path"]]
        counts={p:v for p,v in held[-1]["robot_table_contact_counts"].items()
                if p.startswith(f"/World/envs/env_{slot}/") and sum(v)>0}
        assert len(fixed_hits)==3 and len(surfaces)==2
        geometry_rows.append(dict(slot=slot,table_env_bounds=s["table_env_bounds"],
            classification="VISUAL_AND_ENABLED_COOKED_COLLISION_INTERSECTION_AT_FIXED_TORSO",
            visual_triangle_box_witnesses=surfaces,
            fixed_cooked_shapes=[shapes[p] for p in fixed_hits],
            query_box_inset_m=.001,contact_counts_after20_home_steps=counts,
            query_limitation="Moving arm/hand query hits do not match live geometry/contacts: GPU scene-query moving poses are not trusted. Only fixed torso/table, whose poses do not change, are used here."))
    put("scene_geometry_summary.json",dict(status="REGISTERED_WORLD_HAS_FIXED_TORSO_TABLE_INTERSECTION",
        rows=geometry_rows,full_geometry="output/geometry_query_v3/scene_geometry.json",
        no_aabb_only_positive_claim=True,
        earlier_query_disabled="reset_hold_v2 had query supportFalse; all-zero query output is discarded, not separation evidence.",
        surface_vs_shape="CollisionAPI lives on ancestor Xform for torso meshes; testing only child Mesh.HasAPI incorrectly misses active convexHull shapes.",
        causality_limit="Fixed torso overlap is established; its contribution to arm reference error or training outcome is not isolated."))
    # One conservative geometric proposal, not a fitted/optimized placement.
    shift=np.array([-.75,0.,0.])
    candidate_rows=[]
    for slot,s in enumerate(geometries[-1]["slots"]):
        lo,hi=np.array(s["table_world_bounds"])
        candidates=[]
        torso_gap=[]
        for mesh in s["meshes"]:
            minimum,maximum=np.array(mesh["world_aabb"])+shift
            if np.all(maximum>lo) and np.all(minimum<hi):
                candidates.append(mesh["path"])
            if mesh["body"] in ("vega_1p_torso_l2","vega_1p_torso_l3"):
                torso_gap.append(float(lo[0]-maximum[0]))
        candidate_rows.append(dict(slot=slot,mesh_aabb_remaining_candidates=candidates,
            fixed_torso_min_x_separation_m=min(torso_gap)))
    assert all(not r["mesh_aabb_remaining_candidates"] for r in candidate_rows)
    put("common_world_revision_proposal.json",dict(status="PROPOSAL_ONLY_REQUIRES_COMMON_REVIEWER_APPROVAL",
        candidate_id="CW_ROOT_NEGATIVE_X_075_V1",executed=False,
        objective="Place the complete fixed robot behind the unchanged table; remove the proven static torso/table intersection, not fit video error or optimize score.",
        fields=[dict(field="common robot spawn/fixed-root pose in env coordinates",before=[0,0,0,1,0,0,0],after=[-.75,0,0,1,0,0,0]),
            dict(field="scene ArticulationCfg.init_state.pos",before="implicit default(0,0,0)",after=[-.75,0,0]),
            dict(field="canonical robot_initial_state.root_pose rule",before="USD-defined root; no explicit placement",after="spawn fixed root at(-.75,0,0); never write root during episode reset; add env origin exactly once")],
        fixed_joint_contract="New composed root_joint world anchor must equal env_origin+(-.75,0,0). Keep local joint frames unchanged; verify parser/clone placement before any rollout, do not compensate by arbitrary double offsets.",
        unchanged_fields=["table center/size/top", "object poses and masses", "all58 canonical q/qd", "gravity/dt/PD/contact filters", "A/mean/finger references", "W1/S1 operation targets", "G1–G4"],
        selection_rule="Conservative negative-X separation from full fixed-torso visual/collision-source AABBs; choose one rounded offset giving>20mm clearance. No search, IK retry or physical trial.",
        static_checks=candidate_rows,
        impacts={"both_methods":"same new common-world version and root pose; both must revalidate and retrain, no old comparison score reuse",
            "reset":"joint values unchanged but world home wrists translate-.75m; preserve neutral start semantics",
            "reference":"recompute each method's own arm IK with the shifted runtime anchor and reconnect own home prefixes/suffixes; finger-local q and world W1/S1 targets unchanged. Reachability is NOT established by this proposal",
            "old_checkpoints":"retain as old-world development evidence; no automatic continuation or formal warm start",
            "hashes":"new common manifest/canonical/scene/reference version and actual hashes; registered old bundle remains immutable"},
        required_after_approval=["two-slot fixed-anchor/readback and geometric separation", "bounded nominal approach with existing IK/PD rules", "review reference feasibility before formal training"],
        estimated_effort_after_approval="1–2h for one agreed common-world implementation/short recheck plus own arm-reference regeneration; not a promise of reference feasibility"))
    layers=base/"artifacts/pour17_v1/wp2_20260912_01/reference_execution_layers_v1.json"
    interpretation=read("reference_interpretation.json")
    put("step40_reference_evidence.json",dict(status="REUSED_WP2_PHYSICAL_EVIDENCE_NEW_PHYSICS_BLOCKED_BY_WORLD",
        source=str(layers),sha256=sha256(layers),rows=json.loads(layers.read_text())["rows"],
        colleague_overlay="left_root_interpretation.png",comparison=interpretation,
        not_run="No WP3 reference-to40/143 physical execution after the world conflict was established."))
    put("wp3_report.json",dict(status="COMMON_WORLD_DECISION_REQUIRED",training_added_steps=0,
        completed=["Two-slot40-control-step home hold including first/repeated reset; plus one-step query-enabled diagnostic.",
            "Root/env-origin/q/qd/target/link/table readback and static authored constraints compared.",
            "Visual triangle intersection, enabled ancestor convexHull shapes, fixed-table query and persistent torso contact confirm actual collision overlap.",
            "Left saved21-point colleague root comparison at40/143; no decoder/retarget/IK execution."],
        implementation_findings="No deviation from registered root/table/reset found. Only the new diagnostic camera indexing and query registration were corrected; production env/scene unchanged.",
        remaining=["Shared world change needs approval; dependent physical reference rollout paused.",
            "Right21-point saved input unavailable for colleague overlay; no decoder rerun.",
            "No claim that torso overlap uniquely causes existing IK or following errors.",
            "No new-reference reachability or task success certification."],
        proposal="common_world_revision_proposal.json",source_before="source_before.tar.gz",
        actual_control_steps=41,physical_slot_transitions=82,
        failed_diagnostic="reset_hold_v1: global camera1-entry buffer indexed by2envs; failed in constructor before controlled steps; preserved logs",
        checkpoints="WP1 and WP2 unchanged; WP2 training cumulative8192 steps/144.061s remains",
        gates={"formal3M":"NOT RELEASED","M0":"BLOCKED","M1-A":"NOT PASSED"}))
    with tarfile.open(root/"source_before.tar.gz") as tar:
        old={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile() and m.name.endswith(".py")}
    patch=[]
    changed=[]
    for path in sorted((base/"adaptation/pour17_v1").rglob("*.py")):
        name=path.relative_to(base).as_posix()
        prior=old.get(name,b"")
        if path.read_bytes()!=prior:
            changed.append(name)
            patch.extend(difflib.unified_diff(prior.decode().splitlines(True),path.read_text(encoding="utf-8").splitlines(True),fromfile="before/"+name,tofile="after/"+name))
    (root/"diagnostic_source.diff").write_text("".join(patch),encoding="utf-8")
    assert all(Path(p).name.startswith("wp3_") for p in changed), changed
    put("source_change_summary.json",dict(changed_or_added=changed,production_environment_changed=False,
        source_before_sha256=sha256(root/"source_before.tar.gz"),diff_sha256=sha256(root/"diagnostic_source.diff")))
    print(json.dumps({"status":"WORLD_DECISION_REQUIRED","root":str(root),"reset_checks":reset_checks,"proposal":candidate_rows}))


if __name__=="__main__":
    main()
