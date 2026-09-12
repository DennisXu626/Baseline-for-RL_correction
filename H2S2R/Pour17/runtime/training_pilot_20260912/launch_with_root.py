"""Launch the bounded V13 256-env training and endpoint entries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time
import threading


ROOT = Path(__file__).resolve().parents[2]
TRAIN_INITIALIZATION_TIMEOUT_SECONDS = 900
EVAL_INITIALIZATION_TIMEOUT_SECONDS = 480
TRAIN_PROCESS_SECONDS = 21600.0

def initialization_timeout(stage):
    return TRAIN_INITIALIZATION_TIMEOUT_SECONDS if stage == 'train' else EVAL_INITIALIZATION_TIMEOUT_SECONDS

def required_process_timeout(stage, environment=None):
    environment = environment or {}
    if stage == 'train':
        return float(environment.get('H2S2R_TRAIN_REMAINING_SECONDS', TRAIN_PROCESS_SECONDS))
    return 1800.0

def deadline_choice(stage, init_pending, total, cutoff=None):
    choices = [(total, 'process')]
    if init_pending:
        choices.append((initialization_timeout(stage), 'initialization'))
    if cutoff is not None:
        choices.append(cutoff)
    return min(choices)

def parse_kib_fields(text):
    return {line.split(':', 1)[0]: int(line.split(':', 1)[1].strip().split()[0]) * 1024
            for line in text.splitlines() if ':' in line and line.split(':', 1)[1].strip().endswith('kB')}

def resource_sample(pid, gpu):
    """Only this child /proc plus host available RAM and selected GPU; bounded command."""
    sample = {'utc': utc_now(), 'pid': pid, 'errors': []}
    try:
        status = parse_kib_fields(Path(f'/proc/{pid}/status').read_text())
        sample.update(rss_bytes=status.get('VmRSS'), swap_bytes=status.get('VmSwap'))
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        ticks = os.sysconf('SC_CLK_TCK')
        sample.update(user_cpu_seconds=int(fields[11]) / ticks, system_cpu_seconds=int(fields[12]) / ticks)
    except Exception as error:
        sample['errors'].append('child_proc: ' + repr(error))
    try:
        sample['mem_available_bytes'] = parse_kib_fields(Path('/proc/meminfo').read_text()).get('MemAvailable')
    except Exception as error:
        sample['errors'].append('meminfo: ' + repr(error))
    try:
        result = subprocess.run(['nvidia-smi', '--id=' + gpu,
            '--query-gpu=index,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True, capture_output=True, timeout=2, check=True)
        sample['gpu_csv'] = result.stdout.strip()
    except Exception as error:
        sample['errors'].append('gpu_sample: ' + repr(error))
    return sample

class ResourceMonitor:
    """Daemon sampling cannot delay the independent parent deadline."""
    def __init__(self, pid, gpu, log_dir):
        self.pid, self.gpu, self.log_dir = pid, gpu, Path(log_dir)
        self.stop_event = threading.Event()
        self.stop_reason = None
        self.low_available_count = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def consume(self, sample):
        available = sample.get('mem_available_bytes')
        self.low_available_count = self.low_available_count + 1 if available is not None and available < 8 * 1024**3 else 0
        sample['consecutive_low_available'] = self.low_available_count
        if self.low_available_count >= 3:
            self.stop_reason = 'resource_memavailable_below8GiB_three_samples'

    def run(self):
        while not self.stop_event.is_set():
            sample = resource_sample(self.pid, self.gpu)
            self.consume(sample)
            for filename in ('stdout.log', 'stderr.log'):
                try:
                    with (self.log_dir / filename).open('rb') as stream:
                        stream.seek(0, 2)
                        stream.seek(max(0, stream.tell() - 65536))
                        tail = stream.read().decode(errors='replace')
                    markers = ('CUDA out of memory', 'OutOfMemoryError', 'CUDA error:', 'Fatal Python error:', '[Fatal]')
                    found = next((marker for marker in markers if marker in tail), None)
                    if found:
                        sample['fatal_marker'] = found
                        self.stop_reason = 'confirmed_native_cuda_or_oom_failure'
                except Exception as error:
                    sample['errors'].append('log_sample: ' + repr(error))
            try:
                with (self.log_dir / 'resource_samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(sample) + '\n')
                    stream.flush()
            except Exception as error:
                self.stop_reason = 'resource_record_write_failure'
                print('RESOURCE_RECORD_WRITE_FAILURE ' + repr(error), flush=True)
                return
            self.stop_event.wait(30)

FIXED_PYTHON = Path(
    "/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python"
)
ALLOWED_ENTRIES = {
    "runtime_import": Path("runtime/training_pilot_20260912/runtime_import_probe.py"),
    "short": Path("runtime/training_pilot_20260912/short_physics_validity.py"),
    "train": Path("tasks/h2s2r_pour17/train_right_lstm.py"),
    "eval": Path("runtime/training_pilot_20260912/endpoint_eval.py"),
}
PROBE_PACKAGES = (
    "rl_rebuild",
    "tasks.h2s2r_pour17",
    "rl_rebuild.utils.gpu_guard",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_launch_inputs(root: Path, python: Path, entry_name: str) -> Path:
    root = Path(root).resolve()
    python = Path(python).resolve()
    if not root.is_dir() or not (root / "rl_rebuild").is_dir() or not (root / "tasks").is_dir():
        raise RuntimeError(f"invalid V13 ROOT: {root}")
    if not python.is_file():
        raise RuntimeError(f"fixed Python does not exist: {python}")
    if entry_name not in ALLOWED_ENTRIES:
        raise RuntimeError(f"entry is not one of {sorted(ALLOWED_ENTRIES)}")
    entry = (root / ALLOWED_ENTRIES[entry_name]).resolve()
    if not entry.is_file() or root not in entry.parents:
        raise RuntimeError(f"invalid V13 entry: {entry}")
    return entry


def launch_environment(root: Path, inherited: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    env = dict(inherited)
    paths = [str(Path(root).resolve()),
             "/home/kailang/experiments/baselines/fabrics_h2s2r_reference/src",
             str(Path(root).resolve() / "third_party/h2s2r_official")]
    if not all(Path(path).is_dir() for path in paths):
        raise RuntimeError("A required frozen dependency root is absent")
    env["PYTHONPATH"] = os.pathsep.join(paths)
    temporary = Path(root).resolve() / "runtime/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(temporary)
    return env, paths


def task_verdict(return_code, stdout, stderr, record, expected):
    """Judge structured evidence, independently of Isaac's exit code."""
    if return_code != 0 or "Traceback (most recent call last):" in stdout + stderr:
        return "FAIL"
    if not isinstance(record, dict):
        return "INCOMPLETE"
    if record.get("attempt_id") != expected["attempt_id"] or record.get("input_identity") != expected["input_identity"]:
        return "INCOMPLETE"
    if record.get("status") == "RUNTIME_IMPORT_PASS" and record.get("scene_steps") == 0:
        return "PASS"
    if record.get("status") == "SHORT_AUTOMATED_COMPLETE" and record.get("counts_verified") is True:
        return "PENDING_VISUAL_REVIEW"
    if record.get("status") == "TRAIN_COMPLETE" and record.get("counts_verified") is True:
        return "PENDING_VALIDITY_REVIEW"
    if record.get("status") == "EVAL_AUTOMATED_COMPLETE" and record.get("counts_verified") is True:
        return "PENDING_VISUAL_REVIEW"
    return "FAIL"


def collect_existing_result(stage, child_args, expected):
    """Bind structured native outputs in a fresh attempt directory to this launch."""
    def arg(name):
        return child_args[child_args.index(name) + 1]
    def read(path):
        return json.loads(path.read_text()) if path.is_file() else None
    record = {"attempt_id": expected["attempt_id"], "input_identity": expected["input_identity"]}
    if stage == "short":
        output = Path(arg('--out'))
        gate = read(output / 'gate.json')
        counts = read(output / 'audit/runtime_counts.json')
        if not gate or not counts: return None
        wanted={'controls':256,'physics_substeps':3072,'right_set_targets':256,'right_advance':3072,'position_target_submit':3072}
        ok=all(counts.get(k)==v for k,v in wanted.items()) and all(gate.get('checks',{}).values())
        ok=ok and (output/'audit/env0_substeps.npz').is_file() and bool(list((output/'audit/raw_chunks').glob('*.npz')))
        record.update(status='SHORT_AUTOMATED_COMPLETE',counts_verified=ok,native_gate=gate)
    elif stage == 'train':
        output=Path(arg('--pilot_output_root'))/'logs'/arg('--name')
        matches=list(output.glob('*/pilot_status.json'))
        if len(matches)!=1: return None
        status=read(matches[0]); counts=read(matches[0].parent/'pilot_audit/runtime_counts.json')
        if not status or not counts: return None
        epoch=status.get('epoch',0)
        ok=status.get('state')=='completed' and 0<epoch<=1500 and Path(status.get('checkpoint','')).is_file() and counts.get('controls')==epoch*16
        record.update(status='TRAIN_COMPLETE',counts_verified=ok,native_status=status,counts=counts)
    elif stage == 'eval':
        output=Path(arg('--out')); result=read(output/'evaluation.json')
        if not result: return None
        ok=result.get('valid_n')==16 and len(result.get('episodes',[]))==16
        record.update(status='EVAL_AUTOMATED_COMPLETE',counts_verified=ok,native_result=result)
    else: return None
    return record


def import_probe(python: Path, root: Path, env: dict[str, str], cwd: Path) -> dict:
    code = (
        "import importlib, importlib.util, json, pathlib, sys; "
        "names=" + repr(PROBE_PACKAGES) + "; "
        "gpu=importlib.import_module('rl_rebuild.utils.gpu_guard'); "
        "out={}; "
        "exec(\"for n in names:\\n s=importlib.util.find_spec(n)\\n "
        "out[n]={'origin': None if s is None else s.origin, "
        "'locations': [] if s is None or s.submodule_search_locations is None "
        "else list(s.submodule_search_locations)}\"); "
        "out['gpu_guard_imported_file']=gpu.__file__; "
        "out['sys_path']=sys.path; print(json.dumps(out))"
    )
    result = subprocess.run(
        [str(python), "-c", code], cwd=str(cwd), env=env,
        text=True, capture_output=True, timeout=30, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"zero-physics import probe failed: {result.stderr.strip()}")
    value = json.loads(result.stdout)
    root_text = str(Path(root).resolve())
    for name in PROBE_PACKAGES:
        record = value[name]
        candidates = [record["origin"], *record["locations"]]
        if not any(item and str(Path(item).resolve()).startswith(root_text + os.sep) for item in candidates):
            raise RuntimeError(f"{name} did not resolve inside V13 ROOT: {record}")
    return value


def package_timeout(env, requested, now=None):
    """A retry cannot reset the user package's absolute wall-clock deadline."""
    if env.get('H2S2R_PACKAGE') == 'right_training_256_01':
        deadline = datetime.fromisoformat(env['H2S2R_STAGE_DEADLINE_UTC'])
        remaining = (deadline - (now or datetime.now(timezone.utc))).total_seconds()
        if remaining <= 15:
            raise RuntimeError('cumulative stage budget exhausted')
        return min(requested, remaining)
    if env.get('H2S2R_PACKAGE') != 'batch_delivery_01':
        return requested
    if env.get('H2S2R_STAGE') != 'short':
        raise RuntimeError('batch_delivery_01 forbids PPO and endpoint launches')
    deadline = datetime.fromisoformat(env['H2S2R_PACKAGE_DEADLINE_UTC'])
    remaining = (deadline - (now or datetime.now(timezone.utc))).total_seconds()
    if remaining <= 15:
        raise RuntimeError('package deadline exhausted; no new process')
    return min(requested, remaining)


def run_process(
    argv: list[str], *, cwd: Path, env: dict[str, str], log_dir: Path,
    timeout_seconds: float | None,
) -> dict:
    timeout_seconds = package_timeout(env, timeout_seconds)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    started = time.monotonic()
    timed_out = False
    timeout_reason = None
    soft_sent = False
    cutoff_at = None
    usage_before = None
    if os.name == 'posix':
        import resource
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=stdout, stderr=stderr)
        atomic_json(log_dir / 'process_started.json', {
            'pid': process.pid, 'started_utc': utc_now(), 'argv': argv,
            'input_identity': env.get('H2S2R_INPUT_IDENTITY'),
            'attempt_id': env.get('H2S2R_ATTEMPT_ID'),
            'package': env.get('H2S2R_PACKAGE'),
            'package_deadline_utc': env.get('H2S2R_PACKAGE_DEADLINE_UTC'),
            'stage_deadline_utc': env.get('H2S2R_STAGE_DEADLINE_UTC'),
            'prior_train_process_seconds': float(env.get('H2S2R_PRIOR_TRAIN_PROCESS_SECONDS', 0.0)) if env.get('H2S2R_STAGE') == 'train' else None,
            'timeout_seconds': timeout_seconds, 'init_timeout_seconds': initialization_timeout(env.get('H2S2R_STAGE'))})
        monitor = ResourceMonitor(process.pid, env.get('CUDA_VISIBLE_DEVICES', '0'), log_dir)
        monitor.thread.start()
        while process.poll() is None:
            elapsed = time.monotonic() - started
            state_path = Path(env.get('H2S2R_OBSERVATION_DIR', log_dir / 'observation')) / 'state.json'
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
            init_pending = not state.get('environment_returned', False)
            deadlines = [(timeout_seconds, 'process')] if timeout_seconds is not None else []
            if init_pending and env.get('H2S2R_STAGE') in ('short', 'train', 'eval'):
                deadlines.append((initialization_timeout(env.get('H2S2R_STAGE')), 'initialization'))
            if monitor.stop_reason and cutoff_at is None:
                timeout_reason = monitor.stop_reason
                cutoff_at = elapsed + 15
            if cutoff_at is not None:
                deadlines.append((cutoff_at, timeout_reason))
            if deadlines:
                deadline, reason = min(deadlines)
                if elapsed >= deadline - 15 and not soft_sent:
                    soft_sent = True
                    timed_out = True
                    timeout_reason = reason
                    cutoff_at = deadline
                    atomic_json(log_dir / 'timeout.json', {'reason': reason,
                        'soft_signal_at_seconds': elapsed, 'hard_deadline_seconds': deadline,
                        'state': state})
                    # Best-effort CPU buffer flush; hard cutoff does not depend on Python GIL.
                    if hasattr(signal, 'SIGUSR2') and state_path.is_file():
                        process.send_signal(signal.SIGUSR2)
                if elapsed >= deadline:
                    timed_out = True
                    timeout_reason = reason
                    cutoff_at = deadline
                    process.kill()
                    break
            time.sleep(0.25)
        return_code = process.wait(timeout=30)
        monitor.stop_event.set()  # Never join a sampler on the hard-deadline path.
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN) if usage_before is not None else None
    return {
        "pid": process.pid,
        "return_code": return_code,
        "timed_out": timed_out,
        "timeout_reason": timeout_reason,
        "cleanup_over_deadline_seconds": max(0., time.monotonic() - started - cutoff_at) if cutoff_at is not None else 0.,
        "child_user_cpu_seconds": usage_after.ru_utime - usage_before.ru_utime if usage_after is not None else None,
        "child_system_cpu_seconds": usage_after.ru_stime - usage_before.ru_stime if usage_after is not None else None,
        "child_maxrss_kib": usage_after.ru_maxrss if usage_after is not None else None,
        "maxrss_note": "host RAM high-water, not GPU memory; wrapper owns one child",
        "wall_seconds": time.monotonic() - started,
        "stdout": {"path": str(stdout_path.resolve()), "sha256": sha256(stdout_path)},
        "stderr": {"path": str(stderr_path.resolve()), "sha256": sha256(stderr_path)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", choices=tuple(ALLOWED_ENTRIES), required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--completion-record", type=Path, required=True)
    parser.add_argument("child_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entry = validate_launch_inputs(ROOT, FIXED_PYTHON, args.entry)
    if Path(sys.executable).resolve() != FIXED_PYTHON.resolve():
        raise RuntimeError(
            f"wrapper must run under fixed Python {FIXED_PYTHON}, got {sys.executable}"
        )
    if args.log_dir.exists():
        raise FileExistsError(f"refusing to overwrite launch log directory: {args.log_dir}")
    env, public_paths = launch_environment(ROOT, os.environ)
    if args.probe_only or args.entry == 'runtime_import':
        raise RuntimeError('resume plan forbids new import or initialization probes')
    probe = {'status': 'NOT_RERUN', 'basis': 'accepted init_trace identity; actual imports recorded in child'}
    child_args = list(args.child_args)
    if child_args and child_args[0] == "--":
        child_args.pop(0)
    if '--out' in child_args and Path(child_args[child_args.index('--out')+1]).exists():
        raise FileExistsError('Attempt output must be fresh')
    argv = [str(FIXED_PYTHON), str(entry), *child_args]
    identity = {"entry_sha256": sha256(entry), "argv": child_args}
    if args.completion_record.exists():
        raise FileExistsError("Completion record must be new for this attempt")
    env["H2S2R_ATTEMPT_ID"] = args.attempt_id
    env["H2S2R_INPUT_IDENTITY"] = json.dumps(identity, sort_keys=True)
    env['H2S2R_STAGE'] = args.entry
    env['H2S2R_OBSERVATION_DIR'] = str(args.log_dir.resolve() / 'observation')
    expected_timeout = required_process_timeout(args.entry, env)
    if args.timeout_seconds is None or abs(args.timeout_seconds - expected_timeout) > 1e-6:
        raise RuntimeError(f'entry process deadline must be {expected_timeout} seconds')
    manifest = {
        "schema": "h2s2r_v13_256_launch_with_root_v1",
        "started_utc": utc_now(),
        "host": socket.gethostname(),
        "wrapper": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "entry_name": args.entry,
        "entry": {"path": str(entry), "sha256": sha256(entry)},
        "python": {"path": str(FIXED_PYTHON), "resolved": str(FIXED_PYTHON.resolve())},
        "cwd": str(ROOT),
        "argv": argv,
        "public_environment": {
            "PYTHONPATH": public_paths,
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "TMPDIR": env.get("TMPDIR"),
        },
        "resolved_imports": probe,
        "probe_only": bool(args.probe_only),
        "attempt_id": args.attempt_id,
        "input_identity": identity,
    }
    if args.probe_only:
        args.log_dir.mkdir(parents=True, exist_ok=False)
        manifest.update({"status": "PROBE_PASS", "ended_utc": utc_now(), "return_code": 0})
        atomic_json(args.log_dir / "launch_manifest.json", manifest)
        print("LAUNCH_ENV_PROBE_PASS", flush=True)
        return 0
    args.log_dir.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.log_dir.parent / f'{args.attempt_id}_dispatch.json', manifest)
    result = run_process(
        argv, cwd=ROOT, env=env, log_dir=args.log_dir,
        timeout_seconds=args.timeout_seconds,
    )
    manifest.update(result)
    completion = None
    if args.completion_record.is_file():
        completion = json.loads(args.completion_record.read_text(encoding="utf-8"))
    elif args.entry != 'runtime_import':
        completion=collect_existing_result(args.entry,child_args,manifest)
        if completion:
            args.completion_record.parent.mkdir(parents=True,exist_ok=True)
            atomic_json(args.completion_record,completion)
    verdict = task_verdict(result["return_code"],
                           Path(result["stdout"]["path"]).read_text(errors="replace"),
                           Path(result["stderr"]["path"]).read_text(errors="replace"),
                           completion, manifest)
    observation_state_path = args.log_dir / 'observation/state.json'
    observation_state = json.loads(observation_state_path.read_text()) if observation_state_path.is_file() else {}
    if result['timed_out'] or observation_state.get('status') in ('FAILED', 'TIMEOUT'):
        verdict = 'FAIL'
    if not observation_state.get('environment_returned') or observation_state.get('left_fixed_command_isolation') != 'PASS' or observation_state.get('left_advance_calls') != 0:
        verdict = 'INCOMPLETE' if verdict != 'FAIL' else verdict
    manifest.update({
        "status": verdict,
        "task_status": verdict,
        "process_return_code": result["return_code"],
        "completion_record": completion,
        "ended_utc": utc_now(),
    })
    atomic_json(args.log_dir / "launch_manifest.json", manifest)
    print(
        f"LAUNCH_WITH_ROOT_{manifest['status']} pid={result['pid']} "
        f"return_code={result['return_code']} wall={result['wall_seconds']:.3f}s",
        flush=True,
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

