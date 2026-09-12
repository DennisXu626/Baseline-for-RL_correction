"""Summarize the bounded approach obstruction; propose, do not implement, IK change."""
from pathlib import Path
import json
import numpy as np
from .geometry import UrdfKinematics,quat_to_matrix_wxyz,sha256


def main():
    base=Path(__file__).resolve().parents[2]
    out=base/"artifacts/pour17_v1/wp4_followup_20260912_01"
    prior=base/"artifacts/pour17_v1/wp4_20260912_01"
    ref_path=prior/"reference_build_v1/reference_wp4.npz"
    z=np.load(ref_path)
    probe=json.loads((prior/"output/reset_hold_v1/runtime_probe.json").read_text())
    p=np.array(probe['canonical_home']['arm_center_pose_env_wxyz'])
    T=np.eye(4); T[:3,:3]=quat_to_matrix_wxyz(p[3:]); T[:3,3]=p[:3]
    urdf_path=Path(json.loads((base/'configs/pour17_v1/wp1_config.json').read_text())['paths']['robot_urdf'])
    u=UrdfKinematics(urdf_path)
    vals=dict(zip(z['joint_names'],z['joint_q_rad'][40]))
    kinematics={}
    for si,s in enumerate(('right','left')):
        wrist=u.fk_chain(s+'_hand_C_MC',vals,'arm_center',T)[0]
        points=[wrist[:3,:3]@z['palm_local_m'][si]+wrist[:3,3]]
        for f in ('thumb','index','middle','ring','pinky'):
            points.append(u.fk_chain(s+'_'+f+'_fingertip',vals,'arm_center',T)[0][:3,3])
        target=np.r_[z['palm_target_m'][40,si][None],z[s+'_hand_target_m'][40,::2]]
        kinematics[s]=dict(six_error_m=float(np.linalg.norm(np.array(points)-target,axis=-1).mean()),
            points=np.array(points).tolist(),target=target.tolist(),collision_feasibility_not_certified=True)
    trace=json.loads((out/'approach_v1/trace.json').read_text())
    result=json.loads((out/'approach_v1/result.json').read_text())
    command_error=max(float(np.max(np.abs(np.array(r['command'])-np.array(r['physx_target'])))) for r in trace)
    assert command_error<2e-6
    slots=[]
    for slot in range(2):
        contacts=[dict(step=r['step'],pairs={p:c for p,c in r['robot_table'].items() if f'/env_{slot}/' in p}) for r in trace]
        contacts=[r for r in contacts if r['pairs']]
        errors=np.array([r['six_error_m'][slot] for r in trace])
        actual=np.array(trace[-1]['actual_q'][slot]);command=np.array(trace[-1]['command'][slot])
        recent=np.array([r['actual_q'][slot][:14] for r in trace[-6:]])
        slots.append(dict(slot=slot,schedule=result['schedules'][slot],first_table_contact_step=contacts[0]['step'],
            positive_contact_controls=len(contacts),contacts=contacts,
            final_six_error_m=errors[-1].tolist(),early_hand_contact=result['early_hand_contact'][slot],
            final_arm_target_error_rad=(actual[:14]-command[:14]).tolist(),arm_joint_names=z['joint_names'][:14].tolist(),
            last5steps_max_arm_increment_rad=float(np.max(np.abs(np.diff(recent,axis=0)))),
            final_contact_pairs=contacts[-1]['pairs']))
    report=dict(status='BLOCKED_CURRENT_OWN_APPROACH_TABLE_CONTACT_REQUIRES_METHOD_DECISION',
        source_chain='No demonstrated unit/side/index/transform/reward-time bug; no production fix applied',
        kinematic_terminal_state=kinematics,actual_slots=slots,command_target_max_abs_rad=command_error,
        learning_gate=dict(static_world=True,source_wiring=True,observation_reward_finite=True,
            actual_pregrasp_path_established=False,reason='Both fixed own-action schedules retain forearm-table contacts and miss50mm at40steps'),
        limits=['Does not prove every possible policy/path is infeasible','Does not prove collision uniquely explains every dynamic error',
            'No rejection merely from0success or full-horizon IK failures','No nonfinite or simulator explosion observed'],
        budgets=dict(prior_slot_transitions=620,new_slot_transitions=80,total_slot_transitions=700,remaining=1348,
            new_ppo_transitions=0,remaining_ppo_transitions=57344,remaining_training_wall_s=3455.939,
            gpu_queue_wait_s=0,physical_probe_wall_s=result['wall_s']),
        inputs={str(p):sha256(p) for p in (ref_path,urdf_path,out/'approach_v1/trace.json',Path(__file__))})
    (out/'approach_blocker.json').write_text(json.dumps(report,indent=2)+"\n")
    proposal=dict(id='OWN_APPROACH_TABLE_CLEARANCE_V1',status='PROPOSAL_ONLY_NOT_IMPLEMENTED',
        decision_requested='Permit one ViViDex-owned collision-constrained arm approach IK/path adaptation; keep common world and stage contract',
        scope='Only arm reference samples0..40, endpoint/index40 included; own fingers and MANO/reward targets unchanged',
        exact_change='Add hard arm-collider/table separation constraints to the own wrist-objective arm approach IK; keep the existing wrist objective and own sequential initialization; no foreign q seeds',
        proposed_constraint='Signed geometric separation of every own arm collision shape from the registered table cuboid >=0.025m at20Hz samples and swept segments; use registered shapes/origins, not visual mesh substitutes',
        continuity='Keep q0 exactly canonical and existing previous-iterate initialization; no additional smoothness weight, random restarts or offset search proposed',
        unchanged=['root(-.75,0,0)','canonical reset','2s/40control approach','PD/physics/material/filters','W1/S1','A/mean','finger q','hand/reward targets','50mm stage gate','PPO/curriculum'],
        validation_if_approved='One rebuilt own approach, within remaining1348 slot transitions; require finite commands, physical table clearance and actual dual-hand50mm by40 without early hand-object termination; if infeasible stop, no alternate root or relaxed target',
        concern='May be infeasible with fixed wrist orientation/targets and2s; no feasible result promised. Do not execute until reviewer selects this method change.',
        comparison='Method-specific own reference only, no common reset/physical change, no other method trajectory shared',
        training='Not started; old checkpoints untouched')
    (out/'minimal_revision_proposal.json').write_text(json.dumps(proposal,indent=2)+"\n")
    print(json.dumps({"slots":[{k:r[k] for k in ('slot','first_table_contact_step','final_six_error_m','last5steps_max_arm_increment_rad')} for r in slots],"command_error":command_error}))


if __name__=='__main__':
    main()
