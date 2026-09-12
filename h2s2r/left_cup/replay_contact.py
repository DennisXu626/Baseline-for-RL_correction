"""D fixed-target left-cup replay with per-physics-substep all-body contacts."""
from __future__ import annotations
import argparse,faulthandler,json,signal,time,traceback
from pathlib import Path
from runtime_support import (AtomicSubstepJournal, StateWriteRecorder, atomic_json,
                             close_media_then_app, jsonable, quat_angle,
                             register_observation_sensor, route_fixed_target, utc_now)
from isaaclab.app import AppLauncher

P=argparse.ArgumentParser();P.add_argument('--bundle_root',type=Path,required=True);P.add_argument('--commands',type=Path,required=True);P.add_argument('--out',type=Path,required=True);P.add_argument('--seed',type=int,default=20260829);AppLauncher.add_app_launcher_args(P);args=P.parse_args();args.enable_cameras=True
if args.out.exists():raise FileExistsError(f'refusing output reuse: {args.out}')
args.out.mkdir(parents=True);(args.out/'stacks.log').touch();stack=(args.out/'stacks.log').open('a',buffering=1);faulthandler.enable(stack,all_threads=True);faulthandler.dump_traceback_later(30,repeat=True,file=stack)
if hasattr(signal,'SIGUSR1'):faulthandler.register(signal.SIGUSR1,file=stack,all_threads=True)
status=args.out/'status.json';atomic_json(status,dict(state='APP_LAUNCH',mode='D',started=utc_now()))
app=AppLauncher(args).app

import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab.envs import ViewerCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from pxr import PhysxSchema,Usd,UsdGeom,UsdPhysics
from rl_rebuild.baselines.h2s2r.contract import ControlledSide,joint_names_for_side
from tasks.h2s2r_pour17.left_env import build_cfg,LeftCupH2S2REnv
from camera_delivery import ThreeView
from contact_capture import unpack_six,float_close

LEFT_BODIES=('left_hand_C_MC','left_thumb_CMC_VL','left_thumb_MC','left_thumb_MCP_VL','left_thumb_PP','left_thumb_DP','left_thumb_fingertip','left_thumb_elastomer','left_index_MCP_VL','left_index_PP','left_index_MP','left_index_DP','left_index_fingertip','left_index_elastomer','left_middle_MCP_VL','left_middle_PP','left_middle_MP','left_middle_DP','left_middle_fingertip','left_middle_elastomer','left_ring_MCP_VL','left_ring_PP','left_ring_MP','left_ring_DP','left_ring_fingertip','left_ring_elastomer','left_pinky_MC','left_pinky_MCP_VL','left_pinky_PP','left_pinky_MP','left_pinky_DP','left_pinky_fingertip','left_pinky_elastomer')

class FixedReplay(LeftCupH2S2REnv):
 def _setup_scene(self):
  super()._setup_scene();stage=self.sim.stage
  cup=stage.GetPrimAtPath('/World/envs/env_0/Aux');PhysxSchema.PhysxContactReportAPI.Apply(cup)
  self.full_contact=[];self.full_contact_names=[]
  for name in LEFT_BODIES:
   cfg=ContactSensorCfg(prim_path=f'/World/envs/env_.*/Robot/{name}',history_length=0,track_contact_points=True,max_contact_data_count_per_prim=4096,filter_prim_paths_expr=['/World/envs/env_.*/Aux'])
   register_observation_sensor(sensor_factory=ContactSensor,cfg=cfg,collection=self.full_contact,names=self.full_contact_names,scene_sensors=self.scene.sensors,label=name,key='full_'+name)
  cfg=ContactSensorCfg(prim_path='/World/envs/env_.*/Aux',history_length=0,track_contact_points=True,max_contact_data_count_per_prim=4096,filter_prim_paths_expr=['/World/envs/env_.*/Table'])
  register_observation_sensor(sensor_factory=ContactSensor,cfg=cfg,collection=self.full_contact,names=self.full_contact_names,scene_sensors=self.scene.sensors,label='cup_to_table',key='full_cup_to_table')
 def _pre_physics_step(self,target):
  self.current_request=target.detach().clone();self._target_q.copy_(self.hand.data.joint_pos)
  self._target_q=route_fixed_target(current_q=self._target_q,left_target=target,
   left_ids=self.side_joint_ids[ControlledSide.LEFT],right_ids=self.side_joint_ids[ControlledSide.RIGHT],
   right_reset_q=self.controller.right.q,joint_names=self.hand.joint_names,
   expected_left_names=self.expected_left_names)
 def _apply_action(self):self.hand.set_joint_position_target(self._target_q)
 def _get_dones(self):
  never=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device);return never,never

class Recorder:
 def __init__(self,raw,out):
  self.raw=raw;self.out=Path(out);self.phase='post_construct';self.started_control=0;self.completed_controls=0;self.within=-1;self.rows=[];self.contacts=[];self.counts=[];self.guard_calls={'controller.step':0,'left.step':0,'right.step':0,'left.advance':0,'right.advance':0}
  self.early=AtomicSubstepJournal(self.out/'early_substeps',atomic_until=12)
  self.write_recorder=StateWriteRecorder(self.out/'state_writes.jsonl',self._context);self._install_writes()
  self.body_ids=np.array([raw.hand.body_names.index(n) for n in LEFT_BODIES]);self.names=list(raw.full_contact_names)
  self.original_update=raw.scene.update
  def update(dt):
   result=self.original_update(dt)
   if self.phase=='control':self.capture_substep()
   return result
  raw.scene.update=update
 def _context(self):return dict(phase=self.phase,started_control=self.started_control,completed_controls=self.completed_controls,within_substep=self.within)
 def _snapshot(self,owner):
  row={}
  for name in ('root_pos_w','root_quat_w','root_vel_w','joint_pos','joint_vel'):
   value=getattr(owner.data,name,None)
   if value is not None:row[name]=value.detach().cpu().numpy().copy()
  return row
 def _install_writes(self):
  methods=[(self.raw.hand,'write_joint_state_to_sim'),(self.raw.hand,'write_joint_position_to_sim'),(self.raw.hand,'write_joint_velocity_to_sim'),(self.raw.aux,'write_root_pose_to_sim'),(self.raw.aux,'write_root_velocity_to_sim'),(self.raw.object,'write_root_pose_to_sim'),(self.raw.object,'write_root_velocity_to_sim')]
  for owner,name in methods:
   original=getattr(owner,name)
   asset_path=getattr(owner.cfg,'prim_path',f'<unknown:{type(owner).__name__}>').replace('env_.*','env_0')
   setattr(owner,name,self.write_recorder.wrap(asset_path=asset_path,method_name=name,original=original,snapshot=lambda __o=owner:self._snapshot(__o)))
 def install_controller_guards(self):
  for label,owner,name in [('controller.step',self.raw.controller,'step'),('left.step',self.raw.controller.left,'step'),('right.step',self.raw.controller.right,'step'),('left.advance',self.raw.controller.left,'advance'),('right.advance',self.raw.controller.right,'advance')]:
   def forbidden(*a,__l=label,**k):self.guard_calls[__l]+=1;raise RuntimeError(f'forbidden FABRICS call during fixed replay: {__l}')
   setattr(owner,name,forbidden)
 def capture_state(self):
  r=self.raw;origin=r.scene.env_origins[0];drive=None;getter=getattr(r.hand.root_physx_view,'get_dof_position_targets',None)
  if getter is not None:drive=getter()[0,r.sim_joint_ids].detach().cpu().numpy().copy()
  def pose(asset):return torch.cat((asset.data.root_pos_w[0]-origin,asset.data.root_quat_w[0])).detach().cpu().numpy().copy()
  row=dict(control=self.started_control,within_substep=self.within,global_substep=len(self.rows),sim_step_counter=int(r._sim_step_counter),request_left=r.current_request[0].detach().cpu().numpy().copy(),q=r.hand.data.joint_pos[0].detach().cpu().numpy().copy(),qd=r.hand.data.joint_vel[0].detach().cpu().numpy().copy(),target=r.hand.data.joint_pos_target[0].detach().cpu().numpy().copy(),processed=r.hand._joint_pos_target_sim[0].detach().cpu().numpy().copy(),drive=drive,effort=r.hand.data.applied_torque[0].detach().cpu().numpy().copy(),body_pos=r.hand.data.body_pos_w[0,self.body_ids].detach().cpu().numpy().copy(),body_quat=r.hand.data.body_quat_w[0,self.body_ids].detach().cpu().numpy().copy(),cup=pose(r.aux),cup_vel=r.aux.data.root_vel_w[0].detach().cpu().numpy().copy(),bottle=pose(r.object),bottle_vel=r.object.data.root_vel_w[0].detach().cpu().numpy().copy(),root=torch.cat((r.hand.data.root_pos_w[0]-origin,r.hand.data.root_quat_w[0])).detach().cpu().numpy().copy())
  return row
 def capture_substep(self):
  index=len(self.rows);self.within=(index%self.raw.cfg.decimation);row=self.capture_state();view_counts=[];step_contacts=[]
  for si,sensor in enumerate(self.raw.full_contact):
   result=sensor.contact_physx_view.get_contact_data(dt=float(self.raw.physics_dt));count_size=int(result[4].numel());snap=unpack_six(result,dt=float(self.raw.physics_dt),expected_pairs=count_size)
   if count_size!=1:raise RuntimeError(f'contact view {self.names[si]} resolved {count_size} pairs, expected one')
   if snap['capacity_hit']:raise OverflowError(f'contact capacity hit for {self.names[si]}')
   view_counts.append(int(snap['count'][0]))
   for rr in snap['ranges']:
    for j in range(rr['count']):step_contacts.append((index,si,rr['pair'],rr['force'][j].copy(),rr['point'][j].copy(),rr['normal'][j].copy(),np.asarray(rr['separation'][j]).copy()))
  payload={k:(np.empty(0) if v is None else v) for k,v in row.items()};payload['view_counts']=np.asarray(view_counts,np.int32)
  payload.update(self._contact_arrays(step_contacts))
  self.early.save(index,payload)
  self.counts.append(view_counts);self.rows.append(row);self.contacts.extend(step_contacts)
 @staticmethod
 def _contact_arrays(items):
  if items:return dict(contact_substep=np.asarray([x[0] for x in items],np.int32),contact_sensor=np.asarray([x[1] for x in items],np.int16),contact_pair=np.asarray([x[2] for x in items],np.int16),contact_force=np.stack([x[3] for x in items]),contact_point=np.stack([x[4] for x in items]),contact_normal=np.stack([x[5] for x in items]),contact_separation=np.stack([x[6] for x in items]))
  return dict(contact_substep=np.empty(0,np.int32),contact_sensor=np.empty(0,np.int16),contact_pair=np.empty(0,np.int16),contact_force=np.empty((0,1),np.float32),contact_point=np.empty((0,3),np.float32),contact_normal=np.empty((0,3),np.float32),contact_separation=np.empty((0,1),np.float32))
 def flush(self,complete=False):
  if not self.rows:return
  keys=('control','within_substep','global_substep','sim_step_counter','request_left','q','qd','target','processed','effort','body_pos','body_quat','cup','cup_vel','bottle','bottle_vel','root')
  arr={k:np.stack([x[k] for x in self.rows]) for k in keys};arr['drive']=np.stack([x['drive'] for x in self.rows]) if self.rows[0]['drive'] is not None else np.empty((len(self.rows),0));arr['view_counts']=np.asarray(self.counts,np.int32);arr['left_body_names']=np.asarray(LEFT_BODIES);arr['contact_view_names']=np.asarray(self.names);arr['joint_names']=np.asarray(self.raw.hand.joint_names)
  arr['drive_joint_names']=np.asarray([self.raw.hand.joint_names[int(i)] for i in self.raw.sim_joint_ids.detach().cpu().tolist()]) if self.rows[0]['drive'] is not None else np.asarray([],dtype=str)
  if arr['drive'].shape[1]!=len(arr['drive_joint_names']):raise RuntimeError('drive target width/name mismatch')
  tmp=self.out/'substeps.tmp';final=self.out/'substeps.npz'
  with tmp.open('wb') as f:np.savez_compressed(f,**arr)
  tmp.replace(final)
  ca=self._contact_arrays(self.contacts)
  tmp=self.out/'contacts.tmp';final=self.out/'contacts.npz'
  with tmp.open('wb') as f:np.savez_compressed(f,**ca)
  tmp.replace(final);atomic_json(self.out/'raw_status.json',dict(complete=complete,substeps=len(self.rows),contacts=len(self.contacts),started_control=self.started_control,completed_controls=self.completed_controls,guard_calls=self.guard_calls))

def runtime_metadata(raw):
 stage=raw.sim.stage;rows=[]
 roots=['/World/envs/env_0/Aux']+[f'/World/envs/env_0/Robot/{n}' for n in LEFT_BODIES]
 for root in roots:
  prim=stage.GetPrimAtPath(root)
  for p in Usd.PrimRange(prim):
   if not p.HasAPI(UsdPhysics.CollisionAPI):continue
   col=UsdPhysics.CollisionAPI(p);mc=UsdPhysics.MeshCollisionAPI(p);pc=PhysxSchema.PhysxCollisionAPI(p);stack=[]
   for spec in p.GetPrimStack():stack.append(getattr(spec.layer,'identifier',''))
   rows.append(dict(path=str(p.GetPath()),body=root,enabled=col.GetCollisionEnabledAttr().Get(),approximation=mc.GetApproximationAttr().Get() if p.HasAPI(UsdPhysics.MeshCollisionAPI) else None,contact_offset=pc.GetContactOffsetAttr().Get() if p.HasAPI(PhysxSchema.PhysxCollisionAPI) else None,rest_offset=pc.GetRestOffsetAttr().Get() if p.HasAPI(PhysxSchema.PhysxCollisionAPI) else None,source_layers=stack))
 return rows

media=None;rec=None;out=args.out;run_error=None;cleanup_errors=[]
try:
 atomic_json(status,dict(state='BUILD_ENV',mode='D'));z=np.load(args.commands);commands=z['targets_rad'];expected_names=z['joint_names'].tolist();assert commands.shape==(100,29)
 cfg=build_cfg(bundle_root=args.bundle_root,num_envs=1,seed=args.seed,input_archive=args.bundle_root/'perception/pour17_perception.npz',input_regime='estimated',reference_start_index=32,left_synergy_npz=Path(__file__).parent/'tasks/h2s2r_pour17/artifacts/synergies/left_synergy_pca5.npz',right_synergy_npz=Path(__file__).parent/'tasks/h2s2r_pour17/artifacts/synergies/right_synergy_pca5.npz',external_evaluator_controls_termination=True)
 cfg.sim.device=args.device;cfg.sim.dt=1/240;cfg.decimation=12;cfg.sim.render_interval=cfg.decimation;cfg.viewer=ViewerCfg(eye=(.42,-.43,1.30),lookat=(-.14,.24,1.0),origin_type='env',env_index=0,resolution=(640,480))
 raw=FixedReplay(cfg,render_mode='rgb_array');raw.expected_left_names=expected_names;rec=Recorder(raw,out)
 atomic_json(out/'observation_scope.json',dict(recorder_installed_before_explicit_reset=True,constructor_state_writes_observed=False,reason='Recorder requires the constructed asset objects; constructor activity is not claimed as observed.'))
 rec.phase='explicit_reset';raw.reset(seed=args.seed);rec.phase='post_reset'
 actual_names=[raw.hand.joint_names[int(i)] for i in raw.side_joint_ids[ControlledSide.LEFT].cpu()]
 if actual_names!=expected_names:raise RuntimeError(f'left joint order mismatch {actual_names}')
 def envpose(asset):return torch.cat((asset.data.root_pos_w[0]-raw.scene.env_origins[0],asset.data.root_quat_w[0])).detach().cpu().numpy()
 q=raw.hand.data.joint_pos[0,raw.side_joint_ids[ControlledSide.LEFT]].detach().cpu().numpy();cup=envpose(raw.aux);wpos,wquat=raw._wrist(ControlledSide.LEFT);wrist=torch.cat((wpos[0],wquat[0])).detach().cpu().numpy()
 checks={};checks['left_q'],qtol=float_close(q,z['historical_initial_q_rad']);checks['cup_pos']=bool(np.linalg.norm(cup[:3]-z['historical_cup_pose_wxyz'][:3])<=1e-5);checks['cup_rot']=quat_angle(cup[3:],z['historical_cup_pose_wxyz'][3:])<=1e-4;checks['wrist_pos']=bool(np.linalg.norm(wrist[:3]-z['historical_wrist_pose_wxyz'][:3])<=1e-5);checks['wrist_rot']=quat_angle(wrist[3:],z['historical_wrist_pose_wxyz'][3:])<=1e-4;checks['hold_zero']=int(raw._warmup_clamp_left[0])==0
 atomic_json(out/'reset_validation.json',dict(checks=checks,actual=dict(left_q=q.tolist(),cup=cup.tolist(),wrist=wrist.tolist()),expected=dict(left_q=z['historical_initial_q_rad'].tolist(),cup=z['historical_cup_pose_wxyz'].tolist(),wrist=z['historical_wrist_pose_wxyz'].tolist()),dtype_aware_q_tolerance_max=qtol))
 if not all(checks.values()):raise RuntimeError(f'reset geometry mismatch: {checks}')
 for s,name in zip(raw.full_contact,raw.full_contact_names):
  view=s.contact_physx_view;data=view.get_contact_data(dt=float(raw.physics_dt));unpack_six(data,dt=float(raw.physics_dt),expected_pairs=int(data[4].numel()))
  if int(data[4].numel())!=1:raise RuntimeError(f'{name} view pair mapping not one: {tuple(data[4].shape)}')
 atomic_json(out/'runtime_collision.json',dict(cooked_geometry='COOKED_GEOMETRY_UNAVAILABLE',contact_dt=float(raw.physics_dt),contact_views=raw.full_contact_names,capacity_each=4096,collision_prims=runtime_metadata(raw)))
 rec.install_controller_guards();media=ThreeView(raw,out/'media');media.preview();media.append(0,'post_reset_initial')
 atomic_json(status,dict(state='CONTROL',started_control=0,completed_controls=0,substeps=0));started=time.monotonic();rec.phase='control'
 with torch.no_grad():
  for i,target in enumerate(commands,1):
   rec.started_control=i;atomic_json(status,dict(state='CONTROL',started_control=i,completed_controls=rec.completed_controls,substeps=len(rec.rows)))
   raw.step(torch.as_tensor(target,device=raw.device).unsqueeze(0));rec.completed_controls=i;rec.flush(False);media.append(i,'fixed_target_replay')
   atomic_json(status,dict(state='CONTROL',started_control=i,completed_controls=i,substeps=len(rec.rows)))
 rec.flush(True);media.close_media();media=None
 atomic_json(out/'completion.json',dict(status='COMPLETE',mode='D',started_control=100,completed_controls=100,explicit_substeps=len(rec.rows),initialization_internal_steps=int(raw._sim_step_counter)-len(rec.rows),wall_seconds=time.monotonic()-started,media_counts={'front':101,'top':101,'side':101},policy=False,ppo=False,fabrics_calls=rec.guard_calls))
 atomic_json(status,dict(state='COMPLETE',started_control=100,completed_controls=100,substeps=len(rec.rows)))
except BaseException as e:
 run_error=e
 if rec:
  try:rec.flush(False)
  except BaseException:pass
 atomic_json(status,dict(state='FAILED',error=repr(e),traceback=traceback.format_exc(),started_control=0 if rec is None else rec.started_control,completed_controls=0 if rec is None else rec.completed_controls,substeps=0 if rec is None else len(rec.rows)))
finally:
 cleanup_errors=close_media_then_app(media,app)
 if cleanup_errors:
  prior=json.loads(status.read_text()) if status.exists() else {}
  atomic_json(status,dict(prior,state='FAILED',cleanup_errors=cleanup_errors))
if run_error is not None:raise run_error
if cleanup_errors:raise RuntimeError(f'cleanup failed: {cleanup_errors}')
