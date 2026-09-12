"""Rebuild only own arm IK and home bridges from the approved live root.

Operation wrist/hand/object goals and all finger samples remain byte-identical
as arrays to the bound WP2 reference. No synthesis or MANO layer is rerun.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .geometry import ArmIK, UrdfKinematics, sha256, ndarray_sha256
from .scene import pose_interpolation, _transform_points


def main():
    ap = argparse.ArgumentParser()
    for name in ("old-reference", "probe", "urdf", "overlay", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise RuntimeError("preserve prior output")
    a.output.mkdir(parents=True)
    started = time.perf_counter()
    assert sha256(a.old_reference) == "c478d989e0175e831f63080e702f9d7e995205f1e558c042ee77a4b84b2189c5"
    probe = json.loads(a.probe.read_text())
    assert probe["inputs"]["robot_urdf_sha256"] == sha256(a.urdf)
    home = probe["canonical_home"]
    with np.load(a.old_reference, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    ref = {k: v.copy() for k, v in old.items()}
    assert len(ref["joint_q_rad"]) == 271
    assert list(ref["joint_names"]) == probe["joint_names_in_order"]
    home_q = np.array(home["joint_q_rad"])
    urdf = UrdfKinematics(a.urdf)
    records, checks = {}, {}
    for si, side in enumerate(("right", "left")):
        sl = slice(si*7, (si+1)*7)
        wrist = np.array(home[f"{side}_wrist_pose_env_wxyz"])
        key = f"{side}_wrist_pose_wxyz"
        ref[key][:40] = pose_interpolation(wrist, old[key][40], 40, include_stop=False)
        ref[key][-40:] = pose_interpolation(old[key][-41], wrist, 40, include_stop=True)
        for key, point in ((f"{side}_hand_target_m", np.array(home[f"{side}_landmarks_env_m"])),
                           ("palm_target_m", _transform_points(wrist, ref["palm_local_m"][si]))):
            values = ref[key] if key != "palm_target_m" else ref[key][:, si]
            shape = (40,) + (1,)*point.ndim
            values[:40] = point + np.arange(40).reshape(shape)/40*(values[40]-point)
            values[-40:] = values[-41] + np.arange(1,41).reshape(shape)/40*(point-values[-41])
        ik = ArmIK(urdf, side, "arm_center", np.array(home["arm_center_pose_env_wxyz"]),
                   ref["joint_names"][sl], home_q[sl],
                   np.c_[ref["joint_lower_rad"][sl], ref["joint_upper_rad"][sl]])
        p, _ = ik.fk(home_q[sl])
        assert np.linalg.norm(p-wrist[:3]) < 1e-4
        # Same WP2 defaults, sequential iterate, no random restarts.
        q, records[side] = ik.solve_trajectory(ref[f"{side}_wrist_pose_wxyz"], seed=1701+si, restart_count=0)
        ref["joint_q_rad"][:, sl] = q
        ref[f"{side}_ik_failure_mask"] = np.array([not r["ok"] for r in records[side]])
    for key in ("right_wrist_pose_wxyz", "left_wrist_pose_wxyz", "right_hand_target_m", "left_hand_target_m", "palm_target_m"):
        assert np.array_equal(old[key][40:-40], ref[key][40:-40]), key
        checks[key+"_operation"] = ndarray_sha256(ref[key][40:-40])
    for key in ("right_finger_q_rad", "left_finger_q_rad", "object_pose_wxyz", "final_object_goal_m", "stage", "control_time_s"):
        assert np.array_equal(old[key], ref[key]), key
        checks[key] = ndarray_sha256(ref[key])
    assert np.array_equal(old["joint_q_rad"][:,14:], ref["joint_q_rad"][:,14:])
    ref["world_version"] = np.array("CW_ROOT_NEGATIVE_X_075_V1")
    ref["root_overlay_sha256"] = np.array(sha256(a.overlay))
    ref["runtime_probe_sha256"] = np.array(sha256(a.probe))
    output = a.output / "reference_wp4.npz"
    np.savez_compressed(output, **ref)
    summary = {}
    for side, rows in records.items():
        errors = np.array([r["position_error_m"] for r in rows])
        summary[side] = dict(failure_indices=[r["index"] for r in rows if not r["ok"]],
            at_limit_frames=sum(r["at_limit_count"]>0 for r in rows),
            wrist_error_m=dict(median=float(np.median(errors)), p95=float(np.percentile(errors,95)),max=float(errors.max())),rows=rows)
    table_inside = {}
    for side in ("right", "left"):
        pts = ref[f"{side}_hand_target_m"]
        mask = ((pts > [-.6096,-.9144,.83]) & (pts < [.6096,.9144,.87])).all(-1)
        table_inside[side] = dict(frame_point_indices=np.argwhere(mask).tolist(),
            index40_points_m=pts[40].tolist(), index40_inside_mask=mask[40].tolist(),
            note="unchanged operation targets; root shift cannot remove this conflict")
    report = dict(status="REFERENCE_REBUILT_WITH_RECORDED_FAILURES_NOT_REACHABILITY_PASS",world_version="CW_ROOT_NEGATIVE_X_075_V1",
        count=271,approach_samples=40,return_samples=40,operation_samples=191,
        unchanged_content_hashes=checks,ik=summary,table_inside_targets=table_inside,
        wall_s=time.perf_counter()-started, inputs={str(p):sha256(p) for p in (a.old_reference,a.probe,a.urdf,a.overlay,Path(__file__))},
        output_sha256=sha256(output),no_finger_solve=True,no_new_method=True)
    (a.output/"reference_rebuild_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(output),"wall_s":report["wall_s"],"failure_counts":{s:len(v["failure_indices"]) for s,v in summary.items()}}),flush=True)


if __name__ == "__main__":
    main()
