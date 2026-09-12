"""One-shot GDB initialization validation with periodic traceback disabled."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from shared_gpu_preflight import REQUIRED_FREE_MIB, atomic_json, gpu_snapshot, mem_available_bytes


V13 = Path("/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v13_256_pilot_20260912")
ROOT = Path("/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/traceback_intervention_01")
SUPPORT = ROOT / "support"
OVERLAY = ROOT / "overlay"
OLD_GUARD = Path("/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/native_debug_01/support")
PREFLIGHT = ROOT / "preflight_10s.json"
STATUS = ROOT / "launch_status.json"
MARKER = ROOT / "INIT_RETURNED.json"
GDB_LOG = ROOT / "gdb_native.txt"
CORE = ROOT / "core.v13_256"
RESOURCES = ROOT / "resource_samples.jsonl"
ENTRY = V13 / "tasks/h2s2r_pour17/train_right_lstm.py"
PYTHON = Path("/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def snapshot(selected: int) -> dict:
    gpus, processes = gpu_snapshot()
    return {"utc": utc_now(), "gpus": gpus, "compute_processes": processes, "mem_available_bytes": mem_available_bytes(), "selected_physical_gpu": selected}


def main() -> int:
    if STATUS.exists() or MARKER.exists() or GDB_LOG.exists():
        raise RuntimeError("intervention evidence already exists; refusing another launch")
    overlay_identity = json.loads((ROOT / "overlay_identity.json").read_text(encoding="utf-8"))
    overlay_file = Path(overlay_identity["target"])
    if sha(overlay_file) != overlay_identity["target_sha256"]:
        raise RuntimeError("overlay identity changed")
    cpu_gdb = json.loads((ROOT / "cpu_full_gdb_validation.json").read_text(encoding="utf-8"))
    if not all(cpu_gdb[key] for key in ("signal", "full_backtrace", "registers", "shared_libraries", "all_threads", "continued_after_failure", "capture_complete")):
        raise RuntimeError("full deployable GDB script did not pass CPU validation")
    cpu_periodic = json.loads((ROOT / "cpu_periodic_disabled.json").read_text(encoding="utf-8"))
    if cpu_periodic["stacks_bytes_after_wait"] != 0:
        raise RuntimeError("periodic traceback disable CPU test failed")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if not preflight.get("resource_pass"):
        raise RuntimeError("shared GPU admission failed")
    selected = int(preflight["selected_physical_gpu"])
    old_processes = {(row["gpu_uuid"], row["pid"]) for row in preflight["samples"][-1]["compute_processes"]}
    immediate = snapshot(selected)
    gpu = immediate["gpus"][selected]
    new_processes = sorted({(row["gpu_uuid"], row["pid"]) for row in immediate["compute_processes"]} - old_processes)
    passed = gpu["free_mib"] >= REQUIRED_FREE_MIB and gpu["utilization_percent"] <= 90 and immediate["mem_available_bytes"] >= 8 * 1024**3 and not new_processes
    immediate.update({"required_free_mib": REQUIRED_FREE_MIB, "new_compute_processes_since_admission": new_processes, "pass": passed})
    atomic_json(ROOT / "immediate_prelaunch.json", immediate)
    if not passed:
        atomic_json(STATUS, {"schema": "h2s2r_traceback_intervention_launch_v1", "status": "BLOCKED_IMMEDIATE_RESOURCE_CHECK", "isaac_start_used": False, "immediate": immediate})
        return 3

    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join([str(OLD_GUARD), str(OVERLAY), str(V13), "/home/kailang/experiments/baselines/fabrics_h2s2r_reference/src", str(V13 / "third_party/h2s2r_official")]),
        "CUDA_VISIBLE_DEVICES": str(selected),
        "RL_ISAAC_NO_GUARD": "1",
        "H2S2R_DISABLE_PERIODIC_TRACEBACK": "1",
        "TMPDIR": str(V13 / "runtime/tmp"),
        "H2S2R_NATIVE_DEBUG_GUARD_TARGET": str(ENTRY),
        "H2S2R_NATIVE_DEBUG_GUARD_LINE": "585",
        "H2S2R_NATIVE_DEBUG_GUARD_MARKER": str(MARKER),
        "H2S2R_OBSERVATION_DIR": str(ROOT / "observation"),
        "H2S2R_STAGE": "train",
        "H2S2R_ATTEMPT_ID": "traceback_intervention01",
        "H2S2R_GDB_LOG": str(GDB_LOG),
        "H2S2R_GDB_CORE": str(CORE),
    })
    argv = [
        "/usr/bin/gdb", "-q", "-nx", "-batch", "-x", str(SUPPORT / "gdb_capture_full.gdb"), "--args",
        str(PYTHON), str(ENTRY), "--bundle_root", "/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17",
        "--name", "V13_RIGHT_256_PILOT", "--num_envs", "256", "--seed", "42", "--reference_start_index", "14",
        "--training_pilot", "--pilot_output_root", "/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/right_training_256_01/train01",
        "--device", "cuda:0", "--headless",
    ]
    started_utc = utc_now()
    started = time.monotonic()
    console_path = ROOT / "gdb_console.txt"
    with console_path.open("w", encoding="utf-8") as console:
        process = subprocess.Popen(argv, cwd=V13, env=env, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(STATUS, {"schema": "h2s2r_traceback_intervention_launch_v1", "status": "RUNNING", "isaac_start_used": True, "pid": process.pid, "started_utc": started_utc, "selected_physical_gpu": selected, "selected_gpu_uuid": gpu["uuid"], "argv": argv, "operational_environment": {"CUDA_VISIBLE_DEVICES": str(selected), "RL_ISAAC_NO_GUARD": "1", "H2S2R_DISABLE_PERIODIC_TRACEBACK": "1"}})
        reason = None
        low_gpu = low_ram = high_util = 0
        interrupted = None
        while process.poll() is None:
            row = snapshot(selected)
            selected_row = row["gpus"][selected]
            low_gpu = low_gpu + 1 if selected_row["free_mib"] < 2048 else 0
            low_ram = low_ram + 1 if row["mem_available_bytes"] < 8 * 1024**3 else 0
            high_util = high_util + 1 if selected_row["utilization_percent"] >= 95 else 0
            row.update({"consecutive_gpu_free_below_2gib": low_gpu, "consecutive_mem_available_below_8gib": low_ram, "consecutive_utilization_at_least_95": high_util})
            append_jsonl(RESOURCES, row)
            elapsed = time.monotonic() - started
            if low_gpu >= 3:
                reason = "GPU_FREE_BELOW_2GIB_THREE_SAMPLES"
            elif low_ram >= 3:
                reason = "MEMAVAILABLE_BELOW_8GIB_THREE_SAMPLES"
            elif high_util >= 6:
                reason = "SHARED_UTILIZATION_AT_LEAST_95_FOR_60S"
            elif elapsed >= 900:
                reason = "INITIALIZATION_TIMEOUT_900S"
            if reason and interrupted is None:
                interrupted = time.monotonic()
                os.killpg(process.pid, signal.SIGINT)
            if interrupted is not None and time.monotonic() - interrupted >= 300 and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                reason += "_FORCED_AFTER_300S_CAPTURE_WINDOW"
                break
            time.sleep(10)
        return_code = process.wait(timeout=30)

    text = GDB_LOG.read_text(encoding="utf-8", errors="replace") if GDB_LOG.is_file() else ""
    marker = json.loads(MARKER.read_text(encoding="utf-8")) if MARKER.is_file() else None
    signal_seen = "received signal SIGSEGV" in text or "received signal SIGABRT" in text
    sections = all(name in text for name in ("===== FULL BACKTRACE =====", "===== REGISTERS =====", "===== SHARED LIBRARIES =====", "===== ALL THREADS =====", "===== NATIVE DEBUG CAPTURE COMPLETE ====="))
    if marker and marker.get("status") == "INIT_RETURNED_CRASH_NOT_REPRODUCED":
        outcome = "INIT_RETURNED_WITH_PERIODIC_DUMP_DISABLED"
    elif signal_seen and sections:
        outcome = "NATIVE_CAPTURE_PASS"
    elif signal_seen:
        outcome = "NATIVE_SIGNAL_CAPTURE_INCOMPLETE"
    elif reason == "INITIALIZATION_TIMEOUT_900S" and sections:
        outcome = "TIMEOUT_WITH_STACK"
    elif reason:
        outcome = "STOP_" + reason
    else:
        outcome = "DEBUG_PROCESS_ENDED_WITHOUT_CLASSIFIED_EVIDENCE"
    final = {"schema": "h2s2r_traceback_intervention_launch_v1", "status": outcome, "isaac_start_used": True, "pid": process.pid, "started_utc": started_utc, "ended_utc": utc_now(), "wall_seconds": time.monotonic() - started, "gdb_return_code": return_code, "selected_physical_gpu": selected, "selected_gpu_uuid": gpu["uuid"], "operational_environment": {"CUDA_VISIBLE_DEVICES": str(selected), "RL_ISAAC_NO_GUARD": "1", "H2S2R_DISABLE_PERIODIC_TRACEBACK": "1"}, "stop_reason": reason, "init_return_marker": marker, "native_signal_seen": signal_seen, "full_capture_sections": sections, "policy_controls": 0 if marker else None, "ppo_updates": 0}
    atomic_json(STATUS, final)
    print(json.dumps(final, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
