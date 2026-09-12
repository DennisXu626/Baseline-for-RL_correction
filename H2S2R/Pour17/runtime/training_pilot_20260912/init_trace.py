"""One unchanged 16-env initialization, with observational boundary tracing only."""
from __future__ import annotations
import argparse
import faulthandler
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Trace:
    def __init__(self, out):
        self.out = Path(out)
        self.started = time.monotonic()
        self.events = (self.out / 'events.jsonl').open('a', buffering=1)
        self.state = {'status': 'STARTING', 'active': [], 'last_returned': None,
                      'initialization_step_entered': 0, 'initialization_step_returned': 0,
                      'policy_steps': 0, 'ppo_updates': 0, 'snapshot': 'UNAVAILABLE_CONSTRUCTION_NOT_RETURNED'}
        self.identities = []
        self.seq = 0

    def emit(self, event, name, **fields):
        self.seq += 1
        record = {'seq': self.seq, 'monotonic_seconds': time.monotonic() - self.started,
                  'event': event, 'name': name, **fields}
        self.events.write(json.dumps(record, allow_nan=False) + '\n')
        self.events.flush()
        os.fsync(self.events.fileno())
        self.state['last_event'] = record
        atomic_json(self.out / 'state.json', self.state)

    def call(self, name, function, *args, **kwargs):
        started = time.monotonic()
        frame = {'name': name, 'started_seconds': started - self.started}
        self.state['active'].append(frame)
        self.emit('enter', name)
        try:
            value = function(*args, **kwargs)
        except BaseException:
            self.emit('exception', name, duration_seconds=time.monotonic()-started,
                      traceback=traceback.format_exc())
            raise
        else:
            self.state['last_returned'] = name
            self.emit('exit', name, duration_seconds=time.monotonic()-started)
            return value
        finally:
            self.state['active'].remove(frame)
            atomic_json(self.out / 'state.json', self.state)

    def wrap(self, owner, attribute, name, side=False, step=False):
        original = getattr(owner, attribute)
        path = inspect.getsourcefile(original)
        self.identities.append({'name': name, 'file': path,
                               'line': inspect.getsourcelines(original)[1],
                               'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
        atomic_json(self.out / 'instrumented_identity.json', self.identities)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            label = name + (':' + str(kwargs.get('side', 'unspecified')) if side else '')
            if step:
                self.state['initialization_step_entered'] += 1
            value = self.call(label, original, *args, **kwargs)
            if step:
                self.state['initialization_step_returned'] += 1
                atomic_json(self.out / 'state.json', self.state)
            return value
        setattr(owner, attribute, observed)


def main():
    # Start stack recording before importing AppLauncher or constructing anything.
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--out', type=Path, required=True)
    early_args, _ = early.parse_known_args()
    early_args.out.mkdir(parents=True, exist_ok=False)
    trace = Trace(early_args.out)
    stacks = (early_args.out / 'stacks.log').open('w', buffering=1)
    faulthandler.enable(file=stacks, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=stacks, all_threads=True)
    faulthandler.dump_traceback_later(30, repeat=True, file=stacks)
    app = None
    raw = None
    trace.emit('enter', 'entry')
    try:
        from isaaclab.app import AppLauncher
        parser = argparse.ArgumentParser()
        parser.add_argument('--out', type=Path, required=True)
        parser.add_argument('--bundle_root', type=Path, required=True)
        parser.add_argument('--num_envs', type=int, default=16)
        parser.add_argument('--seed', type=int, default=42)
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        if (args.num_envs, args.seed) != (16, 42):
            raise RuntimeError('Initialization trace frozen to 16 envs, seed42')
        args.enable_cameras = True
        from rl_rebuild.utils.gpu_guard import isaac_slot
        slot = isaac_slot('h2s2r-v12-init-trace', timeout=0)
        app = trace.call('AppLauncher', AppLauncher, args).app
        import numpy as np
        import torch
        from isaaclab.envs import DirectRLEnv, ViewerCfg
        from isaaclab.sim import SimulationContext
        from tasks.h2s2r_pour17.right_zero_margin_install import install_right_training_patch, verify_installed_right_fabric
        expected, patch = trace.call('install_patch', install_right_training_patch)
        atomic_json(args.out / 'patch.json', patch)
        utils = importlib.import_module('fabrics_sim.utils.utils')
        world = importlib.import_module('fabrics_sim.worlds.world_mesh_model')
        controller = importlib.import_module('rl_rebuild.baselines.h2s2r.controller')
        vega = importlib.import_module('rl_rebuild.baselines.h2s2r.vega_sharpa_fabric')
        env = importlib.import_module('tasks.h2s2r_pour17.env')
        right = importlib.import_module('tasks.h2s2r_pour17.right_env')
        audit = importlib.import_module('tasks.h2s2r_pour17.training_pilot_audit')
        # In-process observational wrappers only. Frozen source files stay untouched.
        trace.wrap(SimulationContext, 'step', 'SimulationContext.step', step=True)
        trace.wrap(SimulationContext, 'reset', 'SimulationContext.reset')
        trace.wrap(DirectRLEnv, '__init__', 'DirectRLEnv.__init__')
        trace.wrap(env.Pour17H2S2REnv, '__init__', 'Pour17H2S2REnv.__init__')
        trace.wrap(right.RightBottleH2S2REnv, '__init__', 'RightBottleH2S2REnv.__init__')
        trace.wrap(audit.AuditedRightBottleEnv, '__init__', 'AuditedRightBottleEnv.__init__')
        trace.wrap(utils, 'initialize_warp', 'initialize_warp')
        trace.wrap(controller.BimanualH2S2RFabricController, '__init__', 'BimanualController.__init__')
        trace.wrap(controller.H2S2RFabricController, '__init__', 'Controller.__init__', side=True)
        trace.wrap(vega.VegaSharpaPoseFabric, '__init__', 'VegaSharpaPoseFabric.__init__')
        trace.wrap(world.WorldMeshesModel, '__init__', 'WorldMeshesModel.__init__')
        trace.wrap(utils, 'capture_fabric', 'capture_fabric')
        trace.wrap(env.Pour17H2S2REnv, '_reset_idx', 'Pour17H2S2REnv._reset_idx')
        trace.wrap(env.Pour17H2S2REnv, '_verify_frozen_world', 'verify_frozen_world')
        trace.wrap(env.Pour17H2S2REnv, '_validate_runtime_contract', 'validate_runtime_contract')
        trace.wrap(right, 'apply_centered_bounds', 'apply_centered_bounds')
        trace.wrap(right, 'verify_runtime', 'verify_runtime')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        synergy = ROOT / 'tasks/h2s2r_pour17/artifacts/synergies'
        cfg = trace.call('build_cfg', right.build_cfg, bundle_root=args.bundle_root,
                         num_envs=16, seed=42, input_regime='estimated', reference_start_index=14,
                         right_synergy_npz=synergy/'right_synergy_pca5.npz',
                         left_synergy_npz=synergy/'left_synergy_pca5.npz')
        if cfg.warmup_clamp_steps != 0:
            raise RuntimeError('hold must remain zero')
        cfg.sim.device = args.device
        cfg.sim.render_interval = cfg.decimation
        cfg.viewer = ViewerCfg(eye=audit.CAMERAS['front'][0], lookat=audit.CAMERAS['front'][1],
                               origin_type='env', env_index=0, resolution=(960, 720))
        atomic_json(args.out / 'configuration.json', {'num_envs':16, 'seed':42, 'hold':0,
                    'device':args.device, 'dt':cfg.sim.dt, 'decimation':cfg.decimation,
                    'render_interval':cfg.sim.render_interval, 'bundle_root':str(args.bundle_root),
                    'input_regime':'estimated', 'reference_start_index':14,
                    'render_mode':'rgb_array', 'enable_cameras':True})
        raw = audit.AuditedRightBottleEnv(cfg, render_mode='rgb_array', audit_root=args.out/'audit',
                      controls_per_chunk=64, capture_every_control=True, capture_env0_substeps=True)
        trace.state['status'] = 'CONSTRUCTION_RETURNED'
        trace.emit('exit', 'construction_complete')
        trace.call('verify_exact_right_class', verify_installed_right_fabric, raw.controller.right.fabric, expected)
        # Read-only snapshots only after construction, without a render or policy step.
        np.savez_compressed(args.out/'constructed_snapshot.npz',
                            joint_q=raw.hand.data.joint_pos.detach().cpu().numpy(),
                            joint_qd=raw.hand.data.joint_vel.detach().cpu().numpy(),
                            object_state=raw.object.data.root_state_w.detach().cpu().numpy())
        trace.state['snapshot'] = 'constructed_snapshot.npz'
        trace.state['status'] = 'INITIALIZATION_COMPLETE_DIAGNOSTIC_ONLY'
        trace.emit('complete', 'initialization')
    except BaseException:
        trace.state['status'] = 'FAILED'
        trace.emit('exception', 'entry', traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        trace.emit('enter', 'cleanup')
        if raw is not None:
            trace.call('env.close', raw.close)
        if app is not None:
            trace.call('app.close', app.close)
        trace.emit('exit', 'cleanup')
        # Stack handle intentionally remains open until process exit, including cleanup.
    return 0 if trace.state['status'] == 'INITIALIZATION_COMPLETE_DIAGNOSTIC_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
