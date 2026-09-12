"""Two fixed diagnostic action schedules, same40-step stage; not new reference."""
from pathlib import Path
import argparse
import json
import time
import numpy as np


def main():
    from isaaclab.app import AppLauncher
    ap=argparse.ArgumentParser()
    for name in ("bundle-root","robot-urdf","reference","root-overlay","output"):
        ap.add_argument('--'+name,type=Path,required=True)
    AppLauncher.add_app_launcher_args(ap)
    a=ap.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    app=AppLauncher(a).app
    import torch
    import omni.physics.tensors as tensors
    from .env import Pour17JointEnv
    from .scene import isaac_scene_config
    from .wp4_world import apply_root_overlay
    from .geometry import sha256
    cfg=isaac_scene_config(a.bundle_root,num_envs=2,seed=1701)
    apply_root_overlay(cfg,a.bundle_root,a.root_overlay)
    cfg.sim.device=a.device
    cfg.sim.log_dir=str(a.output/'isaaclab_logs')
    cfg.bundle_root,cfg.robot_urdf,cfg.reference_path=map(str,(a.bundle_root,a.robot_urdf,a.reference))
    cfg.external_evaluator_controls_termination=True
    cfg.curriculum_stage=0
    raw=None; records=[]; start=time.perf_counter()
    def save(name,value):
        (a.output/name).write_text(json.dumps(value,indent=2)+"\n")
    try:
        raw=Pour17JointEnv(cfg)
        raw.reset(seed=1701); raw.scene.write_data_to_sim()
        root=raw.robot.root_physx_view.get_root_transforms().cpu().numpy()
        assert np.max(np.abs(root[:,:3]-raw.scene.env_origins.cpu().numpy()-[-.75,0,0]))<1e-6
        save('initial.json',raw.reset_state())
        sim=tensors.create_simulation_view('torch')
        names=raw.robot.body_names
        paths=[f'/World/envs/env_{i}/Robot/{n}' for i in range(2) for n in names]
        filters=[[f'/World/envs/env_{i}/Table/geometry/mesh'] for i in range(2) for n in names]
        view=sim.create_rigid_contact_view(paths,filters,max_contact_data_count=8192)
        expected=dict(zip(paths,filters))
        assert set(view.sensor_paths)==set(paths)
        assert list(view.filter_paths)==[expected[p] for p in view.sensor_paths]
        home=raw.reset_q.clone()
        goal=raw.reference_q[40].clone()
        for step in range(1,41):
            command=raw.reference_q[step][None].repeat(2,1)
            command[0,:14]=goal[:14]
            command[1,:14]=home[:14]+step/40*(goal[:14]-home[:14])
            action=2*(command-raw.lower)/(raw.upper-raw.lower)-1
            assert bool(((action>=-1.000001)&(action<=1.000001)).all())
            obs,reward,terminated,truncated,extras=raw.step(action.clamp(-1,1))
            assert bool(torch.isfinite(reward).all()) and bool(torch.isfinite(obs['policy']).all())
            hand,tips,table=raw.pair_contacts.read()
            counts=view.get_contact_data(raw.physics_dt)[4].cpu().numpy()
            error=torch.linalg.vector_norm(raw._hand_six()-raw._target_hand_six(),dim=-1).mean(-1)
            records.append(dict(step=step,reference_index=raw._reference_index().cpu().tolist(),
                action=action.cpu().tolist(),command=command.cpu().tolist(),
                physx_target=raw.robot.root_physx_view.get_dof_position_targets()[:,raw.joint_ids].cpu().tolist(),
                actual_q=raw.robot.data.joint_pos[:,raw.joint_ids].cpu().tolist(),
                wrist=raw._wrist_poses().cpu().tolist(),actual_six=raw._hand_six().cpu().tolist(),
                target_six=raw._target_hand_six().cpu().tolist(),six_error_m=error.cpu().tolist(),
                hand_own_object=hand.tolist(),object_table=table.tolist(),reward=reward.cpu().tolist(),
                robot_table={p:c.tolist() for p,c in zip(view.sensor_paths,counts) if c.sum()>0},
                root=raw.robot.root_physx_view.get_root_transforms().cpu().tolist()))
        save('trace.json',records)
        save('result.json',dict(status='COMPLETED_FIXED_ACTION_DIAGNOSTIC_NOT_REFERENCE_CHANGE',
            schedules=['constant own IK index40 arm target','linear canonical-to-own-IK40 arm target in40steps'],
            fingers='unchanged own approach reference per index',slot_transitions=80,
            wall_s=time.perf_counter()-start,final_six_error_m=records[-1]['six_error_m'],
            early_hand_contact=[any(any(r['hand_own_object'][s]) for r in records[:-1]) for s in range(2)],
            inputs={str(p):sha256(p) for p in (a.reference,a.root_overlay,a.robot_urdf,Path(__file__))},
            observation_reward_finite=True,no_reference_write=True))
    except BaseException:
        import traceback
        save('failure.json',dict(error=traceback.format_exc(),completed_steps=len(records)))
        save('partial_trace.json',records)
        raise
    finally:
        if raw is not None: raw.close()
        app.close()


if __name__=='__main__':
    main()
