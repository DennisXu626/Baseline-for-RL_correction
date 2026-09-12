"""Zero-Isaac shared-GPU admission for the one native-debug resume."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time


V13 = Path("/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v13_256_pilot_20260912")
EXPECTED = Path("/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/right_training_256_01/expected_v13_identity.json")
MANIFEST = Path("/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/right_training_256_01/train01_launch/launch_manifest.json")
REMOTE_ROOT = Path("/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/native_debug_01/shared_gpu_resume_01")

# Two prior real launches of this exact 256-env entry reached 2,073 MiB total from
# a 476 MiB baseline (1,597 MiB task delta).  The debugger exits at environment
# return, before policy/rollout/render delivery/PPO.  Use a 4,096 MiB task peak,
# over 2.5x that observed delta, while explicitly retaining the uncertainty.
OBSERVED_TASK_DELTA_MIB = 1597
CONSERVATIVE_TASK_PEAK_MIB = 4096
PUBLIC_RESERVE_MIB = 4096
REQUIRED_FREE_MIB = math.ceil(CONSERVATIVE_TASK_PEAK_MIB * 1.2 + PUBLIC_RESERVE_MIB)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str]) -> str:
    return subprocess.run(argv, text=True, capture_output=True, check=True, timeout=20).stdout


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def gpu_snapshot() -> tuple[dict[int, dict], list[dict]]:
    rows = run([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    gpus: dict[int, dict] = {}
    for line in rows.strip().splitlines():
        index, uuid, total, used, free, util = [part.strip() for part in line.split(",")]
        gpus[int(index)] = {
            "uuid": uuid,
            "total_mib": int(total),
            "used_mib": int(used),
            "free_mib": int(free),
            "utilization_percent": int(util),
        }
    processes_text = run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    processes = []
    for line in processes_text.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, name, used = [part.strip() for part in line.split(",", 3)]
        processes.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": name, "used_memory_mib": int(used)})
    return gpus, processes


def assess(samples: list[dict]) -> tuple[dict[int, dict], int | None]:
    assessments: dict[int, dict] = {}
    for index in (0, 1):
        rows = [sample["gpus"][index] for sample in samples]
        avg_util = sum(row["utilization_percent"] for row in rows) / len(rows)
        max_util = max(row["utilization_percent"] for row in rows)
        min_free = min(row["free_mib"] for row in rows)
        assessments[index] = {
            "uuid": rows[-1]["uuid"],
            "average_utilization_percent": avg_util,
            "maximum_utilization_percent": max_util,
            "minimum_free_mib": min_free,
            "admission_free_required_mib": REQUIRED_FREE_MIB,
            "memory_pass": min_free >= REQUIRED_FREE_MIB,
            "compute_pass": avg_util <= 80.0 and max_util <= 90,
        }
        assessments[index]["pass"] = assessments[index]["memory_pass"] and assessments[index]["compute_pass"]
    candidates = [index for index, value in assessments.items() if value["pass"]]
    selected = min(candidates, key=lambda index: (assessments[index]["average_utilization_percent"], -assessments[index]["minimum_free_mib"])) if candidates else None
    return assessments, selected


def identity_and_command() -> tuple[dict, dict, dict]:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    observed = {path: sha256(Path(path)) for path in expected}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    argv = manifest["argv"]
    entry = str(V13 / "tasks/h2s2r_pour17/train_right_lstm.py")
    checks = {
        "entry_train": manifest["entry_name"] == "train",
        "entry_path": manifest["entry"]["path"] == entry,
        "entry_sha": manifest["entry"]["sha256"] == "2f8c347ebd3e445d802b8dea25238ed5ac190223ed63a190a10fe9f52199bce3",
        "cwd": manifest["cwd"] == str(V13),
        "python": manifest["python"]["resolved"].endswith("/python3.11"),
        "num_envs_256": argv[argv.index("--num_envs") + 1] == "256",
        "seed_42": argv[argv.index("--seed") + 1] == "42",
        "device_cuda0": argv[argv.index("--device") + 1] == "cuda:0",
        "headless": "--headless" in argv,
        "training_pilot": "--training_pilot" in argv,
    }
    identity = {"status": "PASS" if observed == expected else "FAIL", "count": len(observed), "observed": observed}
    return identity, checks, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REMOTE_ROOT / "preflight_10s.json")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.samples != 10 or abs(args.interval_seconds - 1.0) > 1e-9:
        raise RuntimeError("released admission requires exactly ten one-second samples")
    identity, command_checks, manifest = identity_and_command()
    samples = []
    for number in range(args.samples):
        gpus, processes = gpu_snapshot()
        samples.append({"utc": utc_now(), "gpus": gpus, "compute_processes": processes, "mem_available_bytes": mem_available_bytes()})
        if number + 1 < args.samples:
            time.sleep(args.interval_seconds)
    assessments, selected = assess(samples)
    minimum_mem = min(sample["mem_available_bytes"] for sample in samples)
    hdd_free = shutil.disk_usage(REMOTE_ROOT.parent).free
    record = {
        "schema": "h2s2r_native_debug_shared_gpu_preflight_v1",
        "started_utc": samples[0]["utc"],
        "ended_utc": samples[-1]["utc"],
        "samples": samples,
        "identity": identity,
        "scientific_command_checks": command_checks,
        "gpu_admission": assessments,
        "selected_physical_gpu": selected,
        "minimum_mem_available_bytes": minimum_mem,
        "hdd_free_bytes": hdd_free,
        "gpu_memory_budget": {
            "basis": "same V13/256 entry measured 1597 MiB task delta before its prior native stop; debugger exits at environment return",
            "observed_task_delta_mib": OBSERVED_TASK_DELTA_MIB,
            "conservative_task_peak_mib": CONSERVATIVE_TASK_PEAK_MIB,
            "uncertainty": "no successful 256-env construction peak exists; 4096 MiB is over 2.5x the two prior sampled deltas",
            "multiplier": 1.2,
            "public_reserve_mib": PUBLIC_RESERVE_MIB,
            "required_free_mib": REQUIRED_FREE_MIB,
        },
        "operational_diff": {
            "CUDA_VISIBLE_DEVICES": {"old_manifest": manifest["public_environment"]["CUDA_VISIBLE_DEVICES"], "new": None if selected is None else str(selected), "reason": "GPU_POLICY shared admission"},
            "RL_ISAAC_NO_GUARD": {"old_manifest": None, "new": "1", "reason": "disable obsolete machine-wide lock for this child only"},
        },
    }
    record["resource_pass"] = bool(
        identity["status"] == "PASS" and all(command_checks.values()) and selected is not None
        and minimum_mem >= 8 * 1024**3 and hdd_free >= 20 * 1024**3
    )
    atomic_json(args.out, record)
    print(json.dumps({"resource_pass": record["resource_pass"], "selected_physical_gpu": selected, "gpu_admission": assessments, "minimum_mem_available_bytes": minimum_mem, "hdd_free_bytes": hdd_free}, sort_keys=True))
    return 0 if record["resource_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
