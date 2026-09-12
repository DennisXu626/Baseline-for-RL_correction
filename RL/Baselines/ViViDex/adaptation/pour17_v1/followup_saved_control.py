"""Normal left control from saved decoded geometry, no additional decoder call."""
from pathlib import Path
import io
import json
import tarfile
import numpy as np
from .geometry import axis_angle_matrix, mano_landmarks, sha256


def main():
    base=Path(__file__).resolve().parents[2]
    out=base/"artifacts/pour17_v1/wp4_followup_20260912_01"
    config=json.loads((base/"configs/pour17_v1/wp1_config.json").read_text())
    with tarfile.open(config["paths"]["bundle_archive"]) as archive:
        source=np.load(io.BytesIO(archive.extractfile(config["members"]["perception"]).read()))
    saved_path=base/"artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_geometry_input_v1.npz"
    saved=np.load(saved_path)
    vi=list(saved["variant_names"]).index("proposed_raw_pose")
    manifest=json.loads(Path(config["paths"]["preik_manifest"]).read_text())
    w1=manifest["world_alignment"]
    rw,tw=np.array(w1["R_env_from_reconstruction"]),np.array(w1["t_env_from_reconstruction_m"])
    ref_path=base/"artifacts/pour17_v1/wp4_20260912_01/reference_build_v1/reference_wp4.npz"
    ref=np.load(ref_path)
    trace_path=base/"artifacts/pour17_v1/wp4_20260912_01/output/reference_v1/trace.json"
    trace={r["step"]:r for r in json.loads(trace_path.read_text())}
    rows=[]
    # Source0 cross-checks reconstruction against the independent world decode
    # already performed; source81 is the saved, above-table normal left case.
    existing=json.loads((out/"source_chain_report.json").read_text())
    for frame,index in ((0,40),(81,150)):
        assert np.array_equal(saved["pose45"][frame],source["left_mano_pose45"][frame])
        assert np.array_equal(saved["betas"][frame],source["left_mano_betas"][frame])
        assert np.isclose(ref["source_time_s"][index],frame/15,atol=1e-12)
        raw=saved["joints21_H_m"][vi,frame]@axis_angle_matrix(source["left_mano_rot"][frame]).T
        raw+=saved["decoded_wrist_m"][vi,frame]+source["left_mano_trans"][frame]
        ten=mano_landmarks(raw)
        direct=ten@rw.T+tw
        if frame==0:
            independent=next(r["raw_MANO_world"] for r in existing["rows"] if r["side"]=="left" and r["source_frame"]==0)
            assert np.max(np.abs(ten-independent)) < 1e-12
        target=ref["left_hand_target_m"][index]
        runtime=np.array(trace[index]["target_hand_ten"])[0,1]
        inside=((direct>[-.6096,-.9144,.83])&(direct<[.6096,.9144,.87])).all(-1)
        rows.append(dict(side="left",source_frame=frame,control_index=index,raw_MANO_world=ten.tolist(),
            direct_W1=direct.tolist(),final_target=target.tolist(),max_abs_m=float(np.max(np.abs(direct-target))),
            runtime_target=runtime.tolist(),runtime_target_max_abs_m=float(np.max(np.abs(runtime-target))),
            reward_six_target=np.r_[ref["palm_target_m"][index,1][None],target[::2]].tolist(),
            inside_table_mask=inside.tolist()))
    assert rows[1]["max_abs_m"]<1e-12 and not any(rows[1]["inside_table_mask"])
    (out/"saved_normal_control.json").write_text(json.dumps(dict(rows=rows,additional_decoder_calls=0,
        reason="frame30 left pinky still inside; preserve that observation and use saved source81 as normal left control",
        inputs={str(p):sha256(p) for p in (saved_path,ref_path,Path(__file__))}),indent=2)+"\n")
    print([(r["source_frame"],r["max_abs_m"]) for r in rows])


if __name__=="__main__":
    main()
