"""Two source frames only: direct MANO world versus saved W1/reward targets."""
import json
import io
import sys
import tarfile
from pathlib import Path
import numpy as np
from .geometry import load_module, sha256, sha256_bytes, mano_landmarks, quat_to_matrix_wxyz
from .reference import _selector_for_side


def main():
    base = Path(__file__).resolve().parents[2]
    out = base/"artifacts/pour17_v1/wp4_followup_20260912_01"
    config = json.loads((base/"configs/pour17_v1/wp1_config.json").read_text())
    paths = {k:Path(v) for k,v in config["paths"].items()}
    for key in ("decoder","mano_left","mano_right","mano_layer_source"):
        assert sha256(paths[key]) == config["expected_sha256"][key]
    with tarfile.open(paths["bundle_archive"]) as archive:
        blob = archive.extractfile(config["members"]["perception"]).read()
    assert sha256_bytes(blob) == config["expected_sha256"]["perception_member"]
    perception = np.load(io.BytesIO(blob))
    pre = np.load(paths["preik_reference"])
    manifest = json.loads(paths["preik_manifest"].read_text())
    w1 = manifest["world_alignment"]
    rw, tw = np.array(w1["R_env_from_reconstruction"]),np.array(w1["t_env_from_reconstruction_m"])
    refpath=base/"artifacts/pour17_v1/wp4_20260912_01/reference_build_v1/reference_wp4.npz"
    ref = np.load(refpath)
    tracepath=base/"artifacts/pour17_v1/wp4_20260912_01/output/reference_v1/trace.json"
    trace = {r["step"]:r for r in json.loads(tracepath.read_text())}
    sys.path.insert(0,str(paths["decoder"].parent))
    decoder=load_module(paths["decoder"],"followup_registered_decoder")
    rows=[]; arrays={}
    for si,side in enumerate(("right","left")):
        model=decoder.load_model(paths["mano_"+side])
        selector=_selector_for_side(decoder,paths["mano_layer_source"],model,side)
        cal=manifest["calibration"][side]
        ra,ta=np.array(cal["R_S_from_H"]),np.array(cal["t_S_from_H_m"])
        for frame in (0,30):
            index=40+frame*4//3
            assert float(ref["source_time_s"][index]) == frame/15
            decoded=decoder.decode_mano(model,transl=perception[f"{side}_mano_trans"][frame],
                global_aa=perception[f"{side}_mano_rot"][frame],pose45=perception[f"{side}_mano_pose45"][frame],
                betas=perception[f"{side}_mano_betas"][frame],add_hands_mean=False,
                left_shapedirs_x_flip=side=="left",final_sources=selector)
            world21=decoded["joints21_m"]
            direct=mano_landmarks(world21)@rw.T+tw
            local=pre[f"{side}_target_landmarks_S_m"][frame]
            root=pre[f"{side}_wrist_pose_wxyz_core"][frame*4//3]
            recomposed=local@quat_to_matrix_wxyz(root[3:]).T+root[:3]
            final=ref[f"{side}_hand_target_m"][index]
            live=np.array(trace[index]["target_hand_ten"])[0,si]
            actual=np.array(trace[index]["actual_hand_ten"])[0,si]
            wrist=next(l["runtime_wrist"] for l in trace[index]["wrist_layers"] if l["slot"]==0 and l["side"]==side)
            palm_local=ref["palm_local_m"][si]
            actual_palm=quat_to_matrix_wxyz(np.array(wrist)[3:])@palm_local+np.array(wrist)[:3]
            actual_six=np.r_[actual_palm[None],actual[::2]]
            target_six=np.r_[ref["palm_target_m"][index,si][None],final[::2]]
            error=float(np.linalg.norm(actual_six-target_six,axis=-1).mean())
            measured=trace[index]["hand_six_error_m"][0][si]
            diffs=dict(direct_W1_to_recomposed_m=float(np.max(np.abs(direct-recomposed))),
                recomposed_to_final_m=float(np.max(np.abs(recomposed-final))),
                final_to_runtime_target_m=float(np.max(np.abs(final-live))),
                recomputed_six_to_logged_m=abs(error-measured))
            inside=((direct>[-.6096,-.9144,.83])&(direct<[.6096,.9144,.87])).all(-1)
            objects=[]
            for oi in (0,1):
                pose=perception[f"object_{oi}_pose_wxyz"][frame]
                mouth=quat_to_matrix_wxyz(pose[3:])@np.array([0.,.066 if oi==0 else .087,0.])+pose[:3]
                objects.append(dict(object=oi,raw_center=pose[:3].tolist(),W1_center=(pose[:3]@rw.T+tw).tolist(),W1_mouth=(mouth@rw.T+tw).tolist()))
            rows.append(dict(side=side,source_frame=frame,control_index=index,source_time_s=frame/15,
                point_names=[v["name"] for v in config["landmarks"]],raw_MANO_world=mano_landmarks(world21).tolist(),
                direct_W1=direct.tolist(),recomposed=recomposed.tolist(),final_target=final.tolist(),
                runtime_target=live.tolist(),actual_reward_six_target=target_six.tolist(),
                differences=diffs,direct_inside_table_mask=inside.tolist(),object_references=objects,
                reward_palm_note="robot canonical palm_local transformed by desired wrist, not asserted equal to animated MANO MCP mean"))
            arrays[f"{side}_{frame}_raw_joints21_world_m"]=world21
    np.savez_compressed(out/"two_frame_world_arrays.npz",**arrays)
    result=dict(status="NO_SOURCE_WIRING_DISCREPANCY" if all(max(r["differences"].values())<1e-6 for r in rows) else "SOURCE_DISCREPANCY",
        selected_frames=[0,30],normal_control_rule="source30 (2s), exact existing non-synthetic sample before cut46; both hands inspected, not chosen by policy score",
        decoder_calls=4,no_retarget=True,no_reference_write=True,W1=w1,rows=rows,
        inputs={str(p):sha256(p) for p in [paths["decoder"],paths["mano_left"],paths["mano_right"],paths["preik_reference"],paths["preik_manifest"],refpath,tracepath,Path(__file__)]},
        perception_member_sha256=sha256_bytes(blob))
    (out/"source_chain_report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"rows":[{k:r[k] for k in ("side","source_frame","differences","direct_inside_table_mask")} for r in rows]}))


if __name__=="__main__":
    main()
