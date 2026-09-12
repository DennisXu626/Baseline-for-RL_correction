"""Non-physical left wrist-frame comparison from existing decoded arrays.

No MANO invocation, retarget, IK, q write or new reference. The colleague
world-root formula is read from its archive; its exporter is never executed.
"""
from pathlib import Path
import json
import tarfile
import hashlib
import numpy as np
from .geometry import quat_to_matrix_wxyz, resample_linear, sha256


def main():
    base=Path(__file__).resolve().parents[2]
    out=base/"artifacts/pour17_v1/wp3_20260912_01"
    geometry=base/"artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_geometry_input_v1.npz"
    refpath=base/"artifacts/pour17_v1/wp2_20260912_01/reference_build_v2/reference_s1_v2_bound.npz"
    config=base/"configs/m1a_wrist_zero_hand_local_calibration_candidate_v2.json"
    package=Path("D:/xwechat_files/wxid_u1adbhrwjm7822_fb04/msg/file/2026-09/mano2sharpa_retarget_20260911.tar.gz")
    archive_hash=sha256(package)
    assert archive_hash=="e0adf8032f86d1455ef90f6029485508e195457f12091a5fb26cf524c08a2cea"
    with tarfile.open(package) as tar:
        members={name:hashlib.sha256(tar.extractfile("mano2sharpa_retarget_20260911/"+name).read()).hexdigest()
                 for name in ("retarget/frames.py","production_example/make_ref_qpos.py")}
    data=np.load(geometry)
    variant=list(data["variant_names"]).index("proposed_raw_pose")
    joints=data["joints21_H_m"][variant]
    ref=np.load(refpath)
    c=json.loads(config.read_text(encoding="utf-8"))["transform"]
    a,b=np.array(c["R_S_from_H"]),np.array(c["t_S_from_H_m"])
    rows=[]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,6))
    for panel, index in enumerate((40,143)):
        t=float(ref["source_time_s"][index])
        h=resample_linear(joints,np.arange(142)/15,np.array([t]))[0]
        root=ref["left_wrist_pose_wxyz"][index]
        r=quat_to_matrix_wxyz(root[3:])
        world=(h @ a.T+b) @ r.T+root[:3]
        ten=np.array([world[4],(world[2]+world[3])/2,world[8],(world[6]+world[7])/2,
            world[12],(world[10]+world[11])/2,world[16],(world[14]+world[15])/2,
            world[20],(world[18]+world[19])/2])
        reconstruction_error=float(np.max(np.abs(ten-ref["left_hand_target_m"][index])))
        assert reconstruction_error < 1e-6, (index,reconstruction_error)
        # Exact displayed arithmetic of the colleague world-root construction;
        # the local to_mano path is a DIFFERENT transform, documented below.
        z=world[[5,9,13,17]].mean(0)-world[0]
        z=z/(np.linalg.norm(z)+1e-9)
        radial=world[5]-world[17]
        y=radial-(radial@z)*z
        y=y/(np.linalg.norm(y)+1e-9)
        x=np.cross(y,z)
        colleague=np.stack((-x,-y,z),axis=1)  # explicit left flip in exporter
        angle=float(np.degrees(np.arccos(np.clip((np.trace(colleague@r.T)-1)/2,-1,1))))
        rows.append(dict(side="left",index=index,source_time_s=t,current_root=root.tolist(),
            colleague_root_position=world[0].tolist(),colleague_root_rotation=colleague.tolist(),
            relative_rotation_deg=angle,root_offset_m=(world[0]-root[:3]).tolist(),
            reconstructed_ten_max_abs_m=reconstruction_error,world_joints21=world.tolist(),
            label="NONPHYSICAL_INTERPRETATION_ONLY_NOT_NEW_REFERENCE"))
        ax=fig.add_subplot(1,2,panel+1,projection="3d")
        ax.scatter(*world.T,s=12,c="gray",label="existing MANO21 targets")
        for p,rotation,style,label in ((root[:3],r,"-","current fixed-A robot root"),
                                      (world[0],colleague,"--","colleague J0/root interpretation")):
            for k,color in enumerate(("red","green","blue")):
                end=p+.065*rotation[:,k]
                ax.plot(*np.array([p,end]).T,color=color,ls=style,label=label if k==0 else None)
                ax.text(*end,"XYZ"[k],fontsize=8)
        center=world.mean(0)
        for dim,setter in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):
            setter(center[dim]-.12,center[dim]+.12)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=25,azim=30)
        ax.set(xlabel="world X (m)",ylabel="world Y (m)",zlabel="world Z (m)",
               title=f"Left S1 index{index}, source {t:.2f}s\nroot interpretation difference {angle:.1f}deg")
        ax.legend(fontsize=7,loc="upper left")
    fig.suptitle("NONPHYSICAL ONLY: same saved points/q/reference; no orientation selected")
    fig.tight_layout()
    fig.savefig(out/"left_root_interpretation.png",dpi=150)
    plt.close(fig)
    result=dict(status="LEFT_ONLY_SAVED_INPUT_COMPARISON",rows=rows,
        conventions={"current":"column p_world=R_root*(R_A*p_H+t_A)+t_root; robot root differs from MANO J0",
            "colleague_local_normalization":"centered row points @ operator_frame @ OPERATOR2MANO[hand]; frame from J0/J5/J9. Not itself the world root rotation.",
            "colleague_world_root":"J0 position; z=mean(J5,J9,J13,J17)-J0, y=orthogonalized J5-J17, x=y cross z; negate x,y for left. This is a proper two-axis rotation, not a reflection.",
            "not_imported":"No OPERATOR2MANO constant guessed, exporter executed, q imported, vector/DexPilot/scale1.07/filter/clip applied."},
        limitation="Only left has registered saved21-point decoded arrays. Right comparison would need missing21-point input; no decoder rerun performed. Orientation differences between conventions are not proof of a bug or an approved replacement.",
        inputs={str(p):sha256(p) for p in (geometry,refpath,config,package,Path(__file__))},package_members=members)
    (out/"reference_interpretation.json").write_text(json.dumps(result,indent=2)+"\n")
    print([(r["index"],r["relative_rotation_deg"],r["reconstructed_ten_max_abs_m"]) for r in rows])


if __name__=="__main__":
    main()
