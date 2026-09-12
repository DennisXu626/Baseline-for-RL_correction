"""App-independent support used by both the real replay and behavior tests."""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    """Serialize the actual nested torch/numpy argument shapes used by write wrappers."""
    try:
        import torch
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().tolist()
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def quat_angle(a, b):
    """Sign-invariant angle for xyzw/wxyz-agnostic paired arrays."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("quaternions must be finite four-vectors")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= np.finfo(np.float64).tiny or nb <= np.finfo(np.float64).tiny:
        raise ValueError("zero quaternion is invalid")
    dot = abs(float(np.dot(a / na, b / nb)))
    return float(2.0 * math.acos(np.clip(dot, 0.0, 1.0)))


class JsonlJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, value):
        line = json.dumps(jsonable(value), allow_nan=False, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


class StateWriteRecorder:
    """Wrap real state-write methods while preserving return values and exceptions."""
    def __init__(self, path, context):
        self.journal = JsonlJournal(path)
        self.context = context

    def wrap(self, *, asset_path, method_name, original, snapshot=None):
        def wrapped(*args, **kwargs):
            call_id = f"{time.time_ns()}-{os.getpid()}"
            base = dict(self.context(), timestamp=utc_now(), call_id=call_id,
                        asset_path=str(asset_path), method=method_name,
                        args=jsonable(args), kwargs=jsonable(kwargs),
                        before=jsonable(snapshot()) if snapshot is not None else None,
                        stack=traceback.format_stack(limit=6)[:-1])
            self.journal.append(dict(base, event="CALL"))
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                self.journal.append(dict(base, event="RAISED", error=repr(exc),
                                         after=jsonable(snapshot()) if snapshot is not None else None))
                raise
            self.journal.append(dict(base, event="RETURNED", result=jsonable(result),
                                     after=jsonable(snapshot()) if snapshot is not None else None))
            return result
        return wrapped


class AtomicSubstepJournal:
    """One immutable NPZ per early substep; a failed publish cannot erase prior files."""
    def __init__(self, root, *, atomic_until=12, before_publish=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.atomic_until = int(atomic_until)
        self.before_publish = before_publish

    def save(self, index, payload):
        index = int(index)
        if index >= self.atomic_until:
            return None
        final = self.root / f"substep_{index:06d}.npz"
        if final.exists():
            raise FileExistsError(final)
        tmp = self.root / f"substep_{index:06d}.tmp"
        arrays = {str(key): np.asarray(value) for key, value in payload.items()}
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if self.before_publish is not None:
            self.before_publish(index, tmp, final)
        tmp.replace(final)
        return final


def route_fixed_target(*, current_q, left_target, left_ids, right_ids,
                       right_reset_q, joint_names, expected_left_names):
    """The production route: measured other joints, fixed left target, fixed reset right."""
    if tuple(left_target.shape) != (1, 29):
        raise ValueError(f"left target must be (1,29), got {tuple(left_target.shape)}")
    left_index = [int(x) for x in np.asarray(left_ids.detach().cpu() if hasattr(left_ids, "detach") else left_ids).reshape(-1)]
    right_index = [int(x) for x in np.asarray(right_ids.detach().cpu() if hasattr(right_ids, "detach") else right_ids).reshape(-1)]
    if [joint_names[i] for i in left_index] != list(expected_left_names):
        raise ValueError("left joint names/order mismatch")
    if current_q.ndim != 2 or current_q.shape[0] != 1 or current_q.shape[1] != len(joint_names):
        raise ValueError("full joint names do not match current_q width")
    if tuple(right_reset_q.shape) not in ((len(right_index),), (1, len(right_index))):
        raise ValueError("right reset target width does not match right joint ids")
    routed = current_q.clone() if hasattr(current_q, "clone") else np.array(current_q, copy=True)
    routed[:, left_ids] = left_target
    routed[:, right_ids] = right_reset_q
    return routed


def register_observation_sensor(*, sensor_factory, cfg, collection, names,
                                scene_sensors, label, key):
    """Create and register one observation-only sensor in all runtime indexes."""
    sensor = sensor_factory(cfg)
    collection.append(sensor)
    names.append(label)
    scene_sensors[key] = sensor
    return sensor


def close_media_then_app(media, app):
    """Always close media first, then App; preserve both failures for the caller."""
    errors = []
    if media is not None:
        try:
            media.close_media()
        except BaseException as exc:
            errors.append({"stage": "media.close_media", "error": repr(exc)})
    try:
        app.close()
    except BaseException as exc:
        errors.append({"stage": "app.close", "error": repr(exc)})
    return errors


def process_verdict(*, return_code, timed_out, stderr_text, child_status, completion):
    reasons = []
    if timed_out:
        reasons.append("timeout")
    if return_code != 0:
        reasons.append(f"return_code={return_code}")
    if "Traceback (most recent call last)" in stderr_text:
        reasons.append("traceback")
    if not isinstance(child_status, dict) or child_status.get("state") != "COMPLETE":
        reasons.append("child_status_not_complete")
    if not isinstance(completion, dict) or completion.get("status") != "COMPLETE":
        reasons.append("completion_missing_or_not_complete")
    return {"status": "FAILED" if reasons else "COMPLETE", "reasons": reasons}


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def supervise_process(*, cmd, cwd, launch_dir, output_dir, timeout_s, env=None, metadata=None):
    """Run the real child, wait for log closure, then compute the final evidence hashes."""
    cwd = Path(cwd)
    launch_dir = Path(launch_dir)
    output_dir = Path(output_dir)
    launch_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = launch_dir / "stdout.log"
    stderr_path = launch_dir / "stderr.log"
    manifest = dict(metadata or {}, started=utc_now(), cmd=[str(x) for x in cmd], cwd=str(cwd),
                    timeout_s=float(timeout_s), pid=None)
    atomic_json(launch_dir / "launch.json", manifest)
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        child = subprocess.Popen(cmd, cwd=cwd, stdout=stdout, stderr=stderr,
                                 env=env, start_new_session=True)
        manifest["pid"] = child.pid
        atomic_json(launch_dir / "launch.json", manifest)
        try:
            child.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
                child.wait()
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    child_status = _load_json(output_dir / "status.json")
    completion = _load_json(output_dir / "completion.json")
    verdict = process_verdict(return_code=child.returncode, timed_out=timed_out,
                              stderr_text=stderr_text, child_status=child_status,
                              completion=completion)
    result = dict(manifest, ended=utc_now(), return_code=child.returncode,
                  timed_out=timed_out, wall_seconds=time.monotonic() - started,
                  child_status=child_status, completion=completion, **verdict,
                  stdout_sha256=sha256_file(stdout_path), stderr_sha256=sha256_file(stderr_path))
    atomic_json(launch_dir / "result.json", result)
    return result
