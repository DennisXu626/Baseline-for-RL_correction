"""Run one bounded Isaac process and preserve logs/status on every exit path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def save_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_log_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _classify_state(state: str, return_code: int | None, ready_seen: bool, success_seen: bool, failure_seen: bool) -> str:
    if state != "running":
        return "failed" if state in {"guard_failed_before_exit", "start_failed", "killed_by_timeout"} else state
    if return_code is None:
        return "failed"
    if failure_seen:
        return "failed"
    if return_code != 0:
        return "failed"
    if ready_seen and not success_seen:
        return "incomplete_without_success_marker"
    if success_seen:
        return "completed"
    return "incomplete_without_ready_marker"


parser = argparse.ArgumentParser()
parser.add_argument("--log", type=Path, required=True)
parser.add_argument("--status", type=Path, required=True)
parser.add_argument("--ready_marker", required=True)
parser.add_argument("--success_marker", required=True)
parser.add_argument("--failure_marker", default="[CLEAN3-ENTRY-FAILURE]")
parser.add_argument("--init_seconds", type=float, required=True)
parser.add_argument("--total_seconds", type=float, required=True)
parser.add_argument("--poll_seconds", type=float, default=5.0)
parser.add_argument("--status_poll_limit", type=int, default=0)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
if not args.command:
    raise SystemExit("missing guarded command")


started = time.monotonic()
start_utc = datetime.now(timezone.utc).isoformat()
args.log.parent.mkdir(parents=True, exist_ok=True)
process = None
return_code = None
state = "start_failed"
ready_at = None
ready_seen = False
success_seen = False
failure_seen = False
poll_count = 0
last_console_bytes = 0
log_read_errors = []
try:
    with args.log.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(args.command, stdout=log, stderr=subprocess.STDOUT, text=True)
        state = "running"
        while process.poll() is None:
            elapsed = time.monotonic() - started
            contents = read_log_text(args.log)
            last_console_bytes = len(contents)
            ready_seen = args.ready_marker in contents
            success_seen = args.success_marker in contents
            failure_seen = args.failure_marker in contents
            if ready_at is None and ready_seen:
                ready_at = elapsed
            if ready_at is None and elapsed > args.init_seconds:
                state = "initialization_timeout"
                process.terminate()
                break
            if elapsed > args.total_seconds:
                state = "total_timeout"
                process.terminate()
                break
            poll_count += 1
            if args.status_poll_limit > 0 and poll_count > args.status_poll_limit:
                state = "status_poll_limit"
                process.terminate()
                break
            save_status(args.status, {
                "state": state, "pid": process.pid, "start_utc": start_utc,
                "elapsed_seconds": elapsed, "ready_at_seconds": ready_at, "command": args.command,
                "ready_marker": args.ready_marker, "success_marker": args.success_marker, "failure_marker": args.failure_marker,
                "init_seconds": args.init_seconds, "total_seconds": args.total_seconds,
                "poll_count": poll_count,
                "ready_marker_seen": ready_seen,
                "success_marker_seen": success_seen,
                "failure_marker_seen": failure_seen,
            })
            time.sleep(args.poll_seconds)
        if process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = 1
                state = "killed_by_timeout"
        if return_code is None:
            return_code = process.wait()
        contents = read_log_text(args.log)
        last_console_bytes = len(contents)
        ready_seen = args.ready_marker in contents
        success_seen = args.success_marker in contents
        failure_seen = args.failure_marker in contents
        if ready_at is None and ready_seen:
            ready_at = time.monotonic() - started
        state = _classify_state(state, return_code, ready_seen, success_seen, failure_seen)
except BaseException as error:
    if state == "running":
        state = "guard_failed_before_exit"
    if state == "start_failed" and process is None:
        failure_seen = True
    log_read_errors.append(repr(error))
    if process is None:
        return_code = 1
        if ready_seen and not failure_seen:
            failure_seen = True
        if not hasattr(error, "__traceback__"):
            failure_seen = True
    else:
        return_code = process.returncode if process.returncode is not None else 1
finally:
    save_status(args.status, {
        "state": state, "pid": process.pid if process is not None else None,
        "start_utc": start_utc, "end_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started, "ready_at_seconds": ready_at,
        "return_code": return_code, "ready_marker_seen": ready_seen,
        "success_marker_seen": success_seen, "failure_marker_seen": failure_seen,
        "ready_marker": args.ready_marker, "success_marker": args.success_marker, "failure_marker": args.failure_marker,
        "poll_count": poll_count, "last_console_bytes": last_console_bytes, "status_poll_limit": args.status_poll_limit,
        "command": args.command, "log_read_errors": log_read_errors,
        "schema": "clean3_launch_guard_status_v1",
    })

if return_code is None:
    return_code = 1
if state != "completed":
    return_code = 1
raise SystemExit(return_code)
