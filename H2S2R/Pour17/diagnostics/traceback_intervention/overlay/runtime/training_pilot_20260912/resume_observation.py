"""Recording and read-only isolation support; never constructs an App or policy."""
from __future__ import annotations

import faulthandler
import functools
import hashlib
import importlib
import os
from pathlib import Path
import signal
import sys
import traceback

from runtime.training_pilot_20260912.init_trace import Trace, atomic_json


def preserve_before_close(trace, work, close):
    """Persist outcome even when close raises or terminates the interpreter."""
    try:
        result = work()
    except BaseException:
        trace.state['status'] = 'FAILED'
        trace.emit('exception', 'entry', traceback=traceback.format_exc())
        raise
    else:
        trace.state['status'] = 'ENTRY_RETURNED_REQUIRES_NATIVE_ACCEPTANCE'
        trace.emit('complete', 'entry')
        return result
    finally:
        trace.call('app.close', close)


class Observation:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.trace = Trace(self.out)
        self.trace.state.update(environment_returned=False, left_advance_calls=0,
                                left_fixed_command_isolation='NOT_YET_CHECKED')
        self.raw = None
        self.saved_frames = {}
        self.saved_substeps = 0
        self.flushing = False
        self.stacks = (self.out / 'stacks.log').open('w', buffering=1)
        faulthandler.enable(file=self.stacks, all_threads=True)
        if os.environ.get('H2S2R_DISABLE_PERIODIC_TRACEBACK') != '1':
            faulthandler.dump_traceback_later(30, repeat=True, file=self.stacks)
        if hasattr(signal, 'SIGUSR1'):
            faulthandler.register(signal.SIGUSR1, file=self.stacks, all_threads=True)
            signal.signal(signal.SIGUSR2, self.timeout)
        sys.excepthook = self.exception_hook
        self.trace.emit('enter', 'entry')

    def exception_hook(self, kind, value, tb):
        self.trace.state['status'] = 'FAILED'
        self.trace.emit('exception', 'uncaught', traceback=''.join(traceback.format_exception(kind, value, tb)))
        sys.__excepthook__(kind, value, tb)

    def timeout(self, *_):
        self.trace.state['status'] = 'TIMEOUT'
        self.trace.emit('timeout', 'deadline')
        faulthandler.dump_traceback(file=self.stacks, all_threads=True)
        self.flush_partial()
        raise TimeoutError('authorized initialization/process deadline; no retry')

    def call(self, name, function, *args, **kwargs):
        return self.trace.call(name, function, *args, **kwargs)

    def identities(self):
        records = {}
        for name, module in tuple(sys.modules.items()):
            if name.startswith(('tasks.h2s2r_pour17', 'rl_rebuild.baselines.h2s2r',
                                'human2sim2robot.ppo', 'runtime.training_pilot_20260912')):
                file = getattr(module, '__file__', None)
                if file and Path(file).is_file():
                    records[name] = {'file': str(Path(file).resolve()),
                                     'sha256': hashlib.sha256(Path(file).read_bytes()).hexdigest()}
        atomic_json(self.out / 'actual_imports.json', records)

    def install_initialization(self):
        """Reuse the already verified boundaries in the real entry process."""
        from isaaclab.envs import DirectRLEnv
        from isaaclab.sim import SimulationContext
        utils = importlib.import_module('fabrics_sim.utils.utils')
        world = importlib.import_module('fabrics_sim.worlds.world_mesh_model')
        controller = importlib.import_module('rl_rebuild.baselines.h2s2r.controller')
        vega = importlib.import_module('rl_rebuild.baselines.h2s2r.vega_sharpa_fabric')
        env = importlib.import_module('tasks.h2s2r_pour17.env')
        right = importlib.import_module('tasks.h2s2r_pour17.right_env')
        audit = importlib.import_module('tasks.h2s2r_pour17.training_pilot_audit')
        original_step = SimulationContext.step

        @functools.wraps(original_step)
        def counted_step(*args, **kwargs):
            initializing = not self.trace.state['environment_returned']
            if initializing:
                self.trace.state['initialization_step_entered'] += 1
                self.trace.emit('enter', 'initialization.SimulationContext.step')
            result = original_step(*args, **kwargs)
            if initializing:
                self.trace.state['initialization_step_returned'] += 1
                self.trace.emit('exit', 'initialization.SimulationContext.step')
            return result

        SimulationContext.step = counted_step
        for owner, name, label in (
            (SimulationContext, 'reset', 'SimulationContext.reset'),
            (DirectRLEnv, '__init__', 'DirectRLEnv.__init__'),
            (env.Pour17H2S2REnv, '__init__', 'Pour17H2S2REnv.__init__'),
            (right.RightBottleH2S2REnv, '__init__', 'RightBottleH2S2REnv.__init__'),
            (utils, 'initialize_warp', 'initialize_warp'),
            (controller.BimanualH2S2RFabricController, '__init__', 'BimanualController.__init__'),
            (controller.H2S2RFabricController, '__init__', 'Controller.__init__'),
            (vega.VegaSharpaPoseFabric, '__init__', 'VegaSharpaPoseFabric.__init__'),
            (world.WorldMeshesModel, '__init__', 'WorldMeshesModel.__init__'),
            (utils, 'capture_fabric', 'capture_fabric'),
            (env.Pour17H2S2REnv, '_verify_frozen_world', 'verify_frozen_world'),
            (env.Pour17H2S2REnv, '_validate_runtime_contract', 'validate_runtime_contract'),
        ):
            self.trace.wrap(owner, name, label, side=label == 'Controller.__init__')
        original = audit.AuditedRightBottleEnv.__init__

        @functools.wraps(original)
        def constructed(raw, *args, **kwargs):
            self.call('AuditedRightBottleEnv.__init__', original, raw, *args, **kwargs)
            self.raw = raw
            self.trace.state['environment_returned'] = True
            self.trace.emit('exit', 'construction_complete')
            self.attach(raw)
            self.identities()

        audit.AuditedRightBottleEnv.__init__ = constructed

    def attach(self, raw):
        """Observe commands, not measured joints; resets may update only their rows."""
        import torch
        from rl_rebuild.baselines.h2s2r.contract import ControlledSide
        left = raw.controller.left
        baseline = left.q.detach().clone()
        reset_original = raw._reset_idx
        submit_original = raw.hand.set_joint_position_target
        step_original = raw.step
        self.raw = raw

        def reset(*args, **kwargs):
            nonlocal baseline
            ids = args[0] if args else kwargs.get('env_ids')
            before = left.q.detach().clone()
            result = self.call('reset', reset_original, *args, **kwargs)
            untouched = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
            if ids is None:
                untouched[:] = False
            else:
                untouched[ids] = False
            if not torch.equal(before[untouched], left.q[untouched]):
                raise RuntimeError('left reset changed unselected environment commands')
            baseline = left.q.detach().clone()
            return result

        def advance(*args, **kwargs):
            self.trace.state['left_advance_calls'] += 1
            raise RuntimeError('left controller advanced during right-only pilot')

        def submit(*args, **kwargs):
            if not raw._audit_in_reset:
                targets = args[0] if args else kwargs['target']
                ids = raw.side_joint_ids[ControlledSide.LEFT]
                if not torch.equal(left.q, baseline) or not torch.equal(targets[:, ids], baseline):
                    raise RuntimeError('left fixed command isolation failed')
                self.trace.state['left_fixed_command_isolation'] = 'PASS'
            return submit_original(*args, **kwargs)

        def step(*args, **kwargs):
            try:
                result = step_original(*args, **kwargs)
                count = int(raw.audit_counts['controls'])
                self.trace.state['policy_steps'] = count
                if count == 1 or count % 16 == 0:
                    self.trace.emit('progress', 'control', controls=count,
                                    left_fixed_command_isolation=self.trace.state['left_fixed_command_isolation'])
                    if os.environ.get('H2S2R_STAGE') == 'short':
                        self.flush_partial()
                return result
            except BaseException:
                self.trace.state['status'] = 'FAILED'
                self.trace.emit('exception', 'control', traceback=traceback.format_exc())
                self.flush_partial()
                raise

        raw._reset_idx = reset
        left.advance = advance
        raw.hand.set_joint_position_target = submit
        raw.step = step

    def flush_partial(self):
        """Persist CPU buffers only; do not re-render, clear originals or step GPU."""
        if self.raw is None or self.flushing:
            return
        self.flushing = True
        try:
            import numpy as np
            import imageio.v2 as imageio
            raw = self.raw
            raw.raw_writer.flush()
            partial = self.out / 'partial'
            partial.mkdir(exist_ok=True)
            rows = raw._substep_rows[self.saved_substeps:]
            if rows:
                np.savez_compressed(partial / f'substeps_{self.saved_substeps:06d}_{self.saved_substeps+len(rows)-1:06d}.npz',
                                    **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
                self.saved_substeps += len(rows)
            for epoch, views in raw._frames.items():
                for view, frames in views.items():
                    key = (epoch, view)
                    start = self.saved_frames.get(key, 0)
                    if len(frames) > start:
                        # Lossless PNG blocks survive even an interrupted encoder/close.
                        for index in range(start, len(frames)):
                            imageio.imwrite(partial / f'epoch_{epoch:04d}_{view}_{index:06d}.png', frames[index])
                        self.saved_frames[key] = len(frames)
            atomic_json(partial / 'manifest.json', {
                'complete_media': False, 'controls': raw.audit_counts['controls'],
                'substeps_saved': self.saved_substeps,
                'frame_counts': {f'{epoch}/{view}': n for (epoch, view), n in self.saved_frames.items()},
                'counts': raw.audit_counts,
                'left_fixed_command_isolation': self.trace.state['left_fixed_command_isolation'],
                'left_advance_calls': self.trace.state['left_advance_calls']})
            if hasattr(raw, 'delivery'):
                atomic_json(partial / 'streaming_media.json', {
                    'frame_journal_root': str(raw.audit_root / 'milestones'),
                    'counts': {f'{directory}/{view}': count for (directory, view), count in raw.delivery.counts.items()},
                    'failed': raw.delivery.failed,
                    'complete_media': False})
        finally:
            self.flushing = False

    def run(self, main, app_close):
        def work():
            try:
                return main()
            except BaseException:
                self.flush_partial()
                raise
            finally:
                self.identities()
        return preserve_before_close(self.trace, work, app_close)


def from_environment():
    path = os.environ.get('H2S2R_OBSERVATION_DIR')
    if not path:
        raise RuntimeError('bounded V13 entries require the recording-aware launcher')
    return Observation(path)
