"""Fixed three-view media for the left-hand/cup region; rendering never steps physics."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from runtime_support import atomic_json

CAMERAS={
 'front':((.42,-.43,1.30),(-.14,.24,1.00)),
 'top':((-.11,.24,1.75),(-.14,.24,.99)),
 'side':((-.72,.45,1.26),(-.14,.24,1.00)),
}
def camera_config():
 value={'resolution':[640,480],'fps':20,'cameras':{k:{'eye':list(v[0]),'target':list(v[1])} for k,v in CAMERAS.items()}}
 value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return value

def validate_camera_config(value=None):
 value=camera_config() if value is None else value
 if value.get('resolution')!=[640,480] or value.get('fps')!=20 or set(value.get('cameras',{}))!=set(CAMERAS):
  raise ValueError('camera configuration does not match frozen three-view contract')
 return value

class ThreeView:
 def __init__(self,raw,out):
  import imageio.v2 as imageio
  self.raw=raw;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.imageio=imageio;self.writers={};self.counts={v:0 for v in CAMERAS};self.closed=False;self.config=validate_camera_config()
  atomic_json(self.out/'camera_config.json',self.config)
 def _frame(self,view,refreshes=2):
  eye,target=CAMERAS[view];origin=self.raw.scene.env_origins[0].detach().cpu().numpy();eye=np.asarray(eye)+origin;target=np.asarray(target)+origin
  before=int(self.raw._sim_step_counter);self.raw.sim.set_camera_view(tuple(eye),tuple(target),camera_prim_path=self.raw.cfg.viewer.cam_prim_path)
  self.raw.render(recompute=True)
  for _ in range(refreshes):self.raw.sim.render()
  frame=np.asarray(self.raw.render(recompute=True))[:,:,:3].copy()
  if int(self.raw._sim_step_counter)!=before:raise RuntimeError('render-only capture advanced physics counter')
  if frame.size==0 or not frame.any():raise RuntimeError(f'{view} returned empty/black frame')
  return frame,dict(view=view,eye_world=eye.tolist(),target_world=target.tolist(),physics_counter=before,sha256=hashlib.sha256(frame.tobytes()).hexdigest())
 def preview(self):
  p=self.out/'preview';p.mkdir(parents=True,exist_ok=True);records={}
  for view in CAMERAS:
   frame,meta=self._frame(view,8);self.imageio.imwrite(p/f'{view}.png',frame);records[view]=meta
  atomic_json(p/'preview_ready.json',dict(camera_config_sha256=self.config['sha256'],frames=records))
  return records
 def append(self,control,phase):
  records={}
  try:
   for view in CAMERAS:
    frame,meta=self._frame(view,2);meta.update(control=int(control),phase=phase,frame_index=self.counts[view])
    journal=self.out/'frames'/view;journal.mkdir(parents=True,exist_ok=True);self.imageio.imwrite(journal/f'{self.counts[view]:06d}.png',frame)
    with (self.out/f'{view}.jsonl').open('a') as f:f.write(json.dumps(meta)+'\n');f.flush()
    if view not in self.writers:self.writers[view]=self.imageio.get_writer(self.out/f'{view}.mp4',fps=20,codec='libx264',quality=7,macro_block_size=None,pixelformat='yuv420p',ffmpeg_params=['-movflags','+faststart'])
    self.writers[view].append_data(frame);self.counts[view]+=1;records[view]=meta
  except BaseException as exc:
   atomic_json(self.out/'media_status.json',dict(state='FAILED',counts=self.counts,error=repr(exc)))
   raise
  atomic_json(self.out/'media_status.json',dict(state='RECORDING',counts=self.counts))
  return records
 def close_media(self):
  if self.closed:return
  errors=[]
  for w in self.writers.values():
   try:w.close()
   except BaseException as e:errors.append(repr(e))
  self.closed=True;atomic_json(self.out/'media_status.json',dict(state='FAILED' if errors else 'CLOSED',counts=self.counts,errors=errors))
  if errors:raise RuntimeError('video close failed: '+repr(errors))
