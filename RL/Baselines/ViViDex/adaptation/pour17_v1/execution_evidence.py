"""Read existing physical rollout and separate target / IK / execution errors.

No solve, model change, or new trajectory. Original WP1 index141 is mapped by
its saved desired wrist, not by assuming the S1 timeline has the same index.
"""
from pathlib import Path
import json
import numpy as np
from .geometry import ArmIK, UrdfKinematics, sha256, quat_to_matrix_wxyz


def main():
    base = Path(__file__).resolve().parents[2]
    root = base / "artifacts/pour17_v1/wp2_20260912_01"
    build = root / "reference_build_v2"
    old_root = base / "artifacts/pour17_v1/wp1_20260912_02"
    urdf_path = base.parents[1] / "Worktrees/RL-Correction-Step4-Publish/datasets/vega_urdf/vega_1p_sharpa/vega_1p_sharpa_fix.urdf"
    ref_path = build / "reference_s1_v2_bound.npz"
    old_path = build / "wp1_reference_schema_v2_bound_diagnostic.npz"
    trace_path = root / "output/openloop_black_v2/trace.json"
    probe_path = old_root / "runtime_probe.json"
    ref, old = np.load(ref_path), np.load(old_path)
    trace = {r["step"]: r for r in json.loads(trace_path.read_text())}
    probe = json.loads(probe_path.read_text())["canonical_home"]
    distance = np.linalg.norm(ref["left_wrist_pose_wxyz"] - old["left_wrist_pose_wxyz"][141], axis=1)
    matches = np.flatnonzero(distance < 1e-12)
    if len(matches) != 1:
        raise RuntimeError(f"original left141 does not have one exact S1 wrist match: {matches}")
    mapped = int(matches[0])
    rows = []
    urdf = UrdfKinematics(urdf_path)
    for si, side in enumerate(("right", "left")):
        sl = slice(si * 7, (si + 1) * 7)
        lower, upper = ref["joint_lower_rad"][sl], ref["joint_upper_rad"][sl]
        ik = ArmIK(urdf, side, "arm_center", np.array(probe["arm_center_pose_env_wxyz"]),
                   ref["joint_names"][sl], ref["joint_q_rad"][0, sl], np.c_[lower, upper])
        for step in sorted({39, 40, 41, 141, mapped}):
            r = trace[step]
            index = r["reference_index"]
            target = np.array(r["desired_wrist"])[si]
            actual = np.array(r["wrist"])[si]
            q = ref["joint_q_rad"][index, sl]
            aq = np.array(r["q"])[sl]
            p, rot = ik.fk(q)
            ap, ar = ik.fk(aq)
            target_rot = quat_to_matrix_wxyz(target[3:])
            angle = lambda a, b: float(np.arccos(np.clip((np.trace(a @ b.T)-1)/2, -1, 1)))
            rows.append(dict(side=side, completed_step=step, reference_index=index,
                mapped_original_left141=(side == "left" and step == mapped),
                target_wrist=target.tolist(), reference_ik_fk_position=p.tolist(), actual_wrist=actual.tolist(),
                ik_target_position_m=float(np.linalg.norm(p-target[:3])),
                ik_target_rotation_rad=angle(rot, target_rot),
                actual_target_position_m=float(np.linalg.norm(actual[:3]-target[:3])),
                actual_reference_fk_position_m=float(np.linalg.norm(actual[:3]-p)),
                actual_q_fk_runtime_position_m=float(np.linalg.norm(ap-actual[:3])),
                actual_q_fk_runtime_rotation_rad=angle(ar, quat_to_matrix_wxyz(actual[3:])),
                hand_six_error_m=r["hand_six_error_m"][si],
                reference_arm_q=q.tolist(), actual_arm_q=aq.tolist(), lower=lower.tolist(), upper=upper.tolist(),
                reference_min_limit_distance_rad=np.minimum(q-lower, upper-q).tolist(),
                contacts=r["contact_reports"], tip_forces=r["pads"][si]))
    files = [ref_path, old_path, trace_path, probe_path, urdf_path, Path(__file__),
             base / "adaptation/pour17_v1/geometry.py", root / "reference_contact_evidence_v1.tar.gz"]
    report = dict(status="REFERENCE_PREGRASP_THRESHOLD_NOT_MET_NOT_FORMAL_READY",
        original_left141_mapping=dict(new_index=mapped, exact_wrist_max_abs=float(distance[mapped]),
            original_source_index=int(old["source_index"][141]),
            s1_source_time_s=float(ref["source_time_s"][mapped])),
        rows=rows,
        conclusions=["At completed step40 both six-point mean errors exceed0.05m; no success or training readiness claimed.",
            "FK of actual arm q agrees with runtime wrist; observed discrepancy is not a world/link-position mismatch at checked states.",
            "Reference IK residual and dynamic following discrepancy are separate measured contributions, not causal percentages.",
            "Limit proximity and IK failure do not prove global unreachability. No solver retry, target projection or physical change performed.",
            "The first8192 PPO is independent mechanism integration only; no budget expansion or formal release."],
        inputs=[dict(path=str(p), sha256=sha256(p)) for p in files])
    output = root / "reference_execution_layers_v1.json"
    if output.exists():
        raise RuntimeError("versioned evidence already exists")
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output), "mapped_index": mapped}))


if __name__ == "__main__":
    main()
