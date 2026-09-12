"""Shared observation-only camera delivery. No simulator construction or stepping."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

CAMERAS = {
    'front': ((0.35, -0.45, 1.28), (-0.13, 0.06, 1.00)),
    'top': ((-0.13, 0.06, 1.72), (-0.13, 0.06, 0.98)),
    'side': ((-0.65, 0.40, 1.25), (-0.13, 0.06, 1.00)),
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def camera_world(origin, bottle_local, palm_local, view, round_index=0):
    """Track measured interaction midpoint; translate local coordinates exactly once."""
    if round_index not in (0, 1):
        raise ValueError('at most two camera configurations per process')
    origin = np.asarray(origin, dtype=float)
    target_local = (np.asarray(bottle_local)[:3] + np.asarray(palm_local)[:3]) / 2
    eye, target = (np.asarray(v) for v in CAMERAS[view])
    offset = (eye - target) * (0.65 if round_index == 0 else 0.85)
    if view == 'top':
        offset[0] = 0.025  # Avoid the look-at/up collinearity, camera only.
    world_target = origin + target_local
    return world_target + offset, world_target


def project_points(points, view_matrix, projection_matrix, width, height):
    """USD row-vector matrices; retain both matrices for independently reproducible pixels."""
    homogeneous = np.column_stack((np.asarray(points), np.ones(len(points))))
    clip = homogeneous @ np.asarray(view_matrix) @ np.asarray(projection_matrix)
    ndc = clip[:, :3] / clip[:, 3:4]
    pixels = np.column_stack(((ndc[:, 0] + 1) * width / 2,
                             (1 - ndc[:, 1]) * height / 2))
    visible = (clip[:, 3] > 0) & (np.abs(ndc[:, :2]) < 0.95).all(axis=1)
    return pixels, visible


def refresh_rgb(raw, eye, target, refreshes):
    """Source-verified DirectRLEnv.render(recompute=True) reads only.

    On this deployed IsaacLab version recompute=True suppresses its implicit
    sim.render despite the docstring. Call sim.render explicitly, then read.
    SimulationContext.render disables playSimulations during app.update.
    """
    if not 1 <= refreshes <= 12:
        raise ValueError('render refresh budget exceeded')
    before = int(raw._sim_step_counter)
    raw.sim.set_camera_view(eye=tuple(eye), target=tuple(target),
                            camera_prim_path=raw.cfg.viewer.cam_prim_path)
    raw.render(recompute=True)  # Create annotator before the first refresh.
    for _ in range(refreshes):
        raw.sim.render()
    frame = np.asarray(raw.render(recompute=True))[:, :, :3].copy()
    if int(raw._sim_step_counter) != before:
        raise RuntimeError('render-only support advanced the control physics counter')
    if frame.size == 0 or not frame.any():
        raise RuntimeError('media support returned an empty/black frame')
    return frame


class CameraDelivery:
    """Lossless frame journal plus streaming encoders, with fail-closed preview gates."""
    def __init__(self, raw, root, *, side="right"):
        self.side = side
        self.raw = raw
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.round_index = 0
        self.ready = False
        self.writers = {}
        self.counts = {}
        self.failed = False
        self.last = {}

    def snapshot(self, env_id):
        from rl_rebuild.baselines.h2s2r.contract import ControlledSide
        bottle = self.raw._object_poses()[0 if self.side == "left" else 1][env_id].detach().cpu().numpy().copy()
        position, quaternion = self.raw._wrist(ControlledSide.LEFT if self.side == "left" else ControlledSide.RIGHT)
        palm = np.concatenate((position[env_id].detach().cpu().numpy(),
                               quaternion[env_id].detach().cpu().numpy()))
        origin = self.raw.scene.env_origins[env_id].detach().cpu().numpy().copy()
        return origin, bottle, palm

    def capture(self, env_id, episode_id, control, phase, *, reset=False, preview=False):
        from pxr import UsdGeom
        origin, bottle, palm = self.snapshot(env_id)
        results = {}
        for view in CAMERAS:
            eye, target = camera_world(origin, bottle, palm, view, self.round_index)
            frame = refresh_rgb(self.raw, eye, target, 8 if preview else 3)
            camera = UsdGeom.Camera(self.raw.sim.stage.GetPrimAtPath(self.raw.cfg.viewer.cam_prim_path))
            frustum = camera.GetCamera().frustum
            view_matrix = np.asarray(frustum.ComputeViewMatrix(), dtype=float)
            projection = np.asarray(frustum.ComputeProjectionMatrix(), dtype=float)
            points = np.stack((bottle[:3] + origin, palm[:3] + origin))
            pixels, visible = project_points(points, view_matrix, projection, frame.shape[1], frame.shape[0])
            meta = dict(observed_side=self.side, object_role='cup' if self.side == 'left' else 'bottle', env_id=int(env_id), episode_id=int(episode_id), control_index=int(control),
                        substep_index=11 if control >= 0 else None, view=view, phase=phase,
                        reset=bool(reset), round_index=self.round_index,
                        eye_world=eye.tolist(), target_world=target.tolist(), env_origin=origin.tolist(),
                        bottle_pose_env_xyz_wxyz=bottle.tolist(), palm_pose_env_xyz_wxyz=palm.tolist(),
                        bottle_palm_world_xyz=points.tolist(), projected_pixels=pixels.tolist(),
                        centers_in_frame=visible.tolist(), view_matrix=view_matrix.tolist(),
                        projection_matrix=projection.tolist(),
                        pure_render_refreshes=8 if preview else 3,
                        physics_counter=int(self.raw._sim_step_counter),
                        frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                        raw_control_terminal_override=bool(reset and not self.raw._suppress_auto_reset))
            if self.side == 'left':
                meta['cup_pose_env_xyz_wxyz'] = meta.pop('bottle_pose_env_xyz_wxyz')
                meta['cup_palm_world_xyz'] = meta.pop('bottle_palm_world_xyz')
            results[view] = (frame, meta)
        self.last[env_id] = results
        return results

    def save_frames(self, directory, frames, *, annotate=False):
        import imageio.v2 as imageio
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {}
        for view, (frame, meta) in frames.items():
            imageio.imwrite(directory / f'{view}.png', frame)
            metadata[view] = meta
            if annotate:
                from PIL import Image, ImageDraw
                picture = Image.fromarray(frame)
                draw = ImageDraw.Draw(picture)
                draw.text((10, 10), f"env{meta['env_id']} {view} {meta['phase']}", fill='yellow')
                for name, xy in zip((meta.get('object_role', 'bottle'), 'palm'), meta['projected_pixels']):
                    x, y = xy
                    if np.isfinite([x, y]).all():
                        draw.ellipse((x-5, y-5, x+5, y+5), outline='red', width=2)
                        draw.text((x+7, y), name, fill='yellow')
                picture.save(directory / f'{view}_annotated.png')
        atomic_json(directory / 'frames.json', metadata)
        return metadata

    def wait_signoff(self, name, frames_by_env):
        gate = self.root / 'gates' / name
        records = {}
        for env_id, frames in frames_by_env.items():
            records[str(env_id)] = self.save_frames(gate / f'env{env_id}', frames, annotate=True)
        geometry_ok = all(all(m['centers_in_frame']) for views in records.values() for m in views.values())
        digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        request = dict(state='WAIT_VISUAL_READY', gate=name, pid=os.getpid(), digest=digest,
                       geometry_ok=geometry_ok, env_ids=list(frames_by_env), round_index=self.round_index)
        atomic_json(gate / 'request.json', request)
        atomic_json(self.root / 'stage.json', request)
        wait_started = time.monotonic()
        while True:
            if time.monotonic() - wait_started > 900:
                raise RuntimeError('bounded visual review wait exceeded 15 minutes')
            # Transfer approval.upload first; publish approval.ready.json with
            # an atomic rename only after JSON/digest validation. Never read SCP's
            # partially-created destination (attempt01 evidence is preserved).
            response = gate / 'approval.ready.json'
            if response.is_file():
                approval = json.loads(response.read_text())
                if approval.get('digest') != digest:
                    raise RuntimeError('visual approval does not match the saved frames')
                action = approval.get('action')
                if action == 'approve' and geometry_ok and approval.get('visual_checked') is True:
                    atomic_json(gate / 'accepted.json', approval)
                    atomic_json(self.root / 'stage.json', dict(state='VISUAL_READY', gate=name, pid=os.getpid()))
                    return 'approve'
                if action == 'reframe' and name.startswith('preview') and self.round_index == 0:
                    return 'reframe'
                raise RuntimeError('visual signoff rejected or geometry failed')
            if (self.root / 'stop.json').is_file():
                raise RuntimeError('executor requested evidence stop')
            time.sleep(0.5)  # External launcher deadline remains active, including this pause.

    def preview(self, env_ids=(0,)):
        if self.ready:
            return
        while True:
            frames = {i: self.capture(i, int(self.raw._media_episode_ids[i]), -1,
                                      'initial_post_reset_preview', preview=True) for i in env_ids}
            action = self.wait_signoff(f'preview_round{self.round_index}', frames)
            if action == 'approve':
                self.ready = True
                return
            self.round_index += 1

    def append(self, directory, frames, *, prefix='env0_'):
        import imageio.v2 as imageio
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for view, (frame, meta) in frames.items():
                key = (str(directory), view)
                index = self.counts.get(key, 0)
                # Exact PNG written before encoder; an interrupted MP4 is recoverable offline.
                journal = directory / 'frames' / view
                journal.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(journal / f'{index:06d}.png', frame)
                with (directory / f'{view}.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(meta, video_frame_index=index), allow_nan=False) + '\n')
                    stream.flush()
                if key not in self.writers:
                    self.writers[key] = imageio.get_writer(directory / f'{prefix}{view}.mp4', fps=20,
                        codec='libx264', quality=7, macro_block_size=None, pixelformat='yuv420p',
                        ffmpeg_params=['-movflags', '+faststart'])
                self.writers[key].append_data(frame)
                self.counts[key] = index + 1
            atomic_json(directory / 'media_status.json', dict(state='RECORDING',
                        counts={v: self.counts[(str(directory), v)] for v in frames}))
        except BaseException as error:
            self.failed = True
            atomic_json(directory / 'media_failure.json', dict(error=repr(error), state='FAILED'))
            raise

    def close(self, directory=None):
        errors = []
        for key in list(self.writers):
            if directory is not None and key[0] != str(directory):
                continue
            try:
                self.writers.pop(key).close()
            except BaseException as error:
                self.failed = True
                errors.append(repr(error))
        atomic_json(self.root / 'encoder_status.json', dict(state='FAILED' if self.failed else 'CLOSED',
                    errors=errors, counts={f'{d}/{v}': n for (d, v), n in self.counts.items()}))
        if errors:
            raise RuntimeError('encoder close failed: ' + '; '.join(errors))

