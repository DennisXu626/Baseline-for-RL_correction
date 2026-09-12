"""Gate reference rebuilding on measured two-slot WP4 root and separation."""
import json
from pathlib import Path
import numpy as np
from .geometry import sha256


def main():
    base = Path(__file__).resolve().parents[2]
    out = base/"artifacts/pour17_v1/wp4_20260912_01"
    previous = base/"artifacts/pour17_v1/wp3_20260912_01/output/reset_hold_v2"
    current = out/"output/reset_hold_v1"
    read = lambda p: json.loads(p.read_text())
    rows, old = read(current/"reset_world_readback.json"), read(previous/"reset_world_readback.json")
    first = next(r for r in rows if r["label"] == "first_reset_after_write_data")
    repeat = next(r for r in rows if r["label"] == "repeat_reset_after_write_data")
    before = next(r for r in old if r["label"] == "first_reset_after_write_data")
    diff = lambda x,y: float(np.max(np.abs(np.array(x)-np.array(y))))
    reset = {k:diff(first[k],repeat[k]) for k in ("q","qd","physx_joint_targets","root_transforms_xyzw","body_transforms_world_xyzw")}
    assert max(reset.values()) < 1e-6
    fixed = ["vega_1p_mobile","vega_1p_torso_l1","vega_1p_torso_l2","vega_1p_torso_l3","arm_center"]
    ids = [first["robot_body_names"].index(n) for n in fixed]
    changed = np.array(first["body_transforms_world_xyzw"])[:,ids] - np.array(before["body_transforms_world_xyzw"])[:,ids]
    expected = np.zeros_like(changed); expected[:,:,0] = -.75
    fixed_error = diff(changed,expected)
    assert fixed_error < 1e-6
    for key in ("q","qd","physx_joint_targets"):
        assert diff(first[key],before[key]) < 1e-6
    objects = {k:diff(first["canonical"][k],before["canonical"][k]) for k in ("object_0_pose_wxyz","object_1_pose_wxyz")}
    assert max(objects.values()) < 1e-6
    root_drift = max(diff(r["root_transforms_xyzw"],first["root_transforms_xyzw"]) for r in rows)
    assert root_drift < 1e-6
    geometry = read(current/"scene_geometry.json")
    old_geo = read(previous/"scene_geometry.json")[0]["slots"]
    checks = []
    for g in geometry:
        for s in g["slots"]:
            slot = s["slot"]
            assert diff(s["table_world_bounds"],old_geo[slot]["table_world_bounds"]) < 1e-6
            fixed_mesh = [m for m in s["meshes"] if m["body"] in fixed]
            gap = min(s["table_world_bounds"][0][0] - m["world_aabb"][1][0] for m in fixed_mesh)
            assert gap > .02
            assert not any(m["surface_triangle_intersections"] for m in fixed_mesh)
            hits = [h for h in s["physx_cooked_overlap_box"]["0.0"]["hits"] if any('/'+n in h["rigid_body"] for n in fixed)]
            assert not hits
            checks.append(dict(label=g["label"],slot=slot,fixed_mesh_x_gap_m=gap,fixed_cooked_hits=hits))
    counts = [{p:v for p,v in r["robot_table_contact_counts"].items() if "torso_" in p and sum(v)>0} for r in rows if r["label"]=="hold"]
    assert not any(counts)
    result = dict(status="PASS_FIXED_ROOT_AND_STATIC_TORSO_SEPARATION_ONLY",first_repeat_reset_max_abs=reset,
        fixed_body_translation_error_m=fixed_error,object_reset_max_abs=objects,root_hold_drift_max_abs=root_drift,
        effective_fixed_anchor_world_xyz=np.array(first["root_transforms_xyzw"])[:,:3].tolist(),
        fixed_anchor_evidence="USD enabled fixed articulation + complete PhysX root remains at requested world anchor during40 controls; no runtime root writes; local joint attributes retain authored zeros",
        checks=checks,torso_contact_positive_hold_frames=0,slot_transitions=80,
        not_certified=["dynamic arm/hand safety","reference reachability","other method runtime"],
        inputs={str(p):sha256(p) for p in (current/"reset_world_readback.json",current/"scene_geometry.json",previous/"reset_world_readback.json",Path(__file__))})
    path = out/"static_separation_gate.json"
    assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
