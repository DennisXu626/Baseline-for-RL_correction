"""Run the approved DEV_TO_FIRST_CURRICULUM_V1 package once."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


TARGET_STEPS = 1_003_520
EXPECTED_COMBINED_PEAK_MIB = 2 * 2621
REQUIRED_FREE_MIB = EXPECTED_COMBINED_PEAK_MIB * 1.2 + 4096


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_rows():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    rows = []
    for line in output.splitlines():
        index, uuid, total, used, free, utilization = [part.strip() for part in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "memory_total_mib": int(total),
                     "memory_used_mib": int(used), "memory_free_mib": int(free),
                     "utilization_percent": int(utilization)})
    return rows


def memory_available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def gpu_processes(uuid):
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0] == uuid:
            rows.append({"pid": int(parts[1]), "used_memory_mib": int(parts[2])})
    return rows


def descendants(root_pid):
    parents = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            values = path.read_text().split()
            parents[int(values[0])] = int(values[3])
        except (OSError, ValueError, IndexError):
            pass
    result, frontier = {root_pid}, [root_pid]
    while frontier:
        parent = frontier.pop()
        children = [pid for pid, ppid in parents.items() if ppid == parent and pid not in result]
        result.update(children)
        frontier.extend(children)
    return result


def admission(output_dir):
    samples = []
    for index in range(6):
        samples.append({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "gpus": gpu_rows(), "mem_available_bytes": memory_available_bytes()})
        if index != 5:
            time.sleep(2)
    candidates = []
    for gpu_index in (0, 1):
        rows = [next(row for row in sample["gpus"] if row["index"] == gpu_index)
                for sample in samples]
        avg_util = sum(row["utilization_percent"] for row in rows) / len(rows)
        maximum_util = max(row["utilization_percent"] for row in rows)
        minimum_free = min(row["memory_free_mib"] for row in rows)
        if (avg_util <= 80 and maximum_util <= 90 and minimum_free >= REQUIRED_FREE_MIB
                and min(sample["mem_available_bytes"] for sample in samples) >= 8 * 1024 ** 3):
            candidates.append((avg_util, -minimum_free, gpu_index, rows[-1]["uuid"]))
    if not candidates:
        raise RuntimeError("no GPU meets the recorded shared-admission thresholds")
    _, _, selected, uuid = min(candidates)
    record = {"policy": "D:/UCBP/GPU_POLICY.md", "samples": samples,
              "task_combined_peak_budget_mib": EXPECTED_COMBINED_PEAK_MIB,
              "required_free_mib": REQUIRED_FREE_MIB, "selected_physical_gpu": selected,
              "selected_gpu_uuid": uuid, "process_cuda_device": "cuda:0", "status": "PASS"}
    (output_dir / "resource_admission.json").write_text(json.dumps(record, indent=2) + "\n")
    return selected, uuid


def run(command, environment, log_path, resource_path, physical_gpu, gpu_uuid):
    started = time.perf_counter()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        low_free, low_ram, shared_hot = 0, 0, 0
        while process.poll() is None:
            row = next(item for item in gpu_rows() if item["index"] == physical_gpu)
            processes = gpu_processes(gpu_uuid)
            task_pids = descendants(process.pid)
            shared = [item for item in processes if item["pid"] not in task_pids]
            sample = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **row,
                      "mem_available_bytes": memory_available_bytes(), "gpu_processes": processes,
                      "task_descendant_pids": sorted(task_pids), "other_gpu_processes": shared}
            with resource_path.open("a") as stream:
                stream.write(json.dumps(sample) + "\n")
            low_free = low_free + 1 if row["memory_free_mib"] < 2048 else 0
            low_ram = low_ram + 1 if sample["mem_available_bytes"] < 8 * 1024 ** 3 else 0
            shared_hot = shared_hot + 1 if shared and row["utilization_percent"] >= 95 else 0
            if low_free >= 3 or low_ram >= 3 or shared_hot >= 6:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
                raise RuntimeError("resource supervisor stopped this task at the approved safety boundary")
            time.sleep(10)
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}): {command}; see {log_path}")
    return time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-id", default="development_curriculum_20260913_01")
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[2]
    wp4_root = source_root.parent
    base_root = wp4_root.parent
    old = (wp4_root / "extension_runtime_20260912_01/artifacts/pour17_v1/"
           "development_extension_20260912_01/training")
    out = source_root / "artifacts/pour17_v1" / args.output_id
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(old.parent / "development_training_ledger.json",
                 out / "development_training_ledger.json")
    model_before, state_before = old / "final_model.zip", old / "training_state.pt"
    expected = {"model": "92868d6525087ae36da2acb09982432af074d4a6d2c5208a40de7f4039b19483",
                "state": "6de5b4073d0b9ea9effb0b7323fc5cb8fda8d7ac8e9fd77479437d3ca38c08c5",
                "reference": "34c8f30eaf37a674d5eabf3943218cdba5dd1612014d87b8f9055c45978fe649"}
    if digest(model_before) != expected["model"] or digest(state_before) != expected["state"]:
        raise RuntimeError("registered262,144 checkpoint/state unavailable")
    reference = wp4_root / "artifacts/pour17_v1/wp4_20260912_01/reference_build_v1/reference_wp4.npz"
    if digest(reference) != expected["reference"]:
        raise RuntimeError("registered WP4 reference mismatch")
    physical_gpu, gpu_uuid = admission(out)
    common = ["--bundle-root", "/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17",
        "--robot-urdf", str(base_root / "assets/vega_1p_sharpa_fix.urdf"),
        "--reference", str(reference), "--root-overlay", str(source_root / "configs/pour17_v1/wp4_root_overlay.json"),
        "--device", "cuda:0", "--headless"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(physical_gpu), RL_ISAAC_NO_GUARD="1",
        TMPDIR=str(base_root / "tmp"), XDG_CACHE_HOME=str(base_root / "cache"),
        PYTHONPATH=str(source_root), PYTHONUNBUFFERED="1")
    record = {"authorization": "DEV_TO_FIRST_CURRICULUM_V1", "status": "RUNNING",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "physical_gpu": physical_gpu,
        "gpu_uuid": gpu_uuid, "process_cuda_device": "cuda:0", "rl_isaac_no_guard": "1",
        "before_model": {"path": str(model_before), "sha256": digest(model_before)},
        "before_state": {"path": str(state_before), "sha256": digest(state_before)}, "commands": []}
    record_path, resource_path = out / "curriculum_job.json", out / "resource_monitor.jsonl"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        training = out / "training"
        command = [sys.executable, "-m", "adaptation.pour17_v1.train", *common,
            "--resolved-config", str(source_root / "configs/pour17_v1/wp2_train.yaml"),
            "--schema", str(source_root / "configs/pour17_v1/development_schema.json"),
            "--output-dir", str(training), "--seed", "1701", "--num-envs", "32", "--n-steps", "128",
            "--total-timesteps", str(TARGET_STEPS), "--resume", str(model_before),
            "--development-curriculum-approved"]
        elapsed = run(command, environment, out / "training.log", resource_path, physical_gpu, gpu_uuid)
        record["commands"].append({"kind": "resume_training", "command": command, "wall_s": elapsed})
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        result = json.loads((training / "ppo_update_result.json").read_text())
        model_after = training / "final_model.zip"
        for mode in ("stochastic", "deterministic"):
            destination = out / f"eval_final_{mode}"
            command = [sys.executable, "-m", "adaptation.pour17_v1.policy_diagnostic_eval",
                *common, "--model", str(model_after), "--output-dir", str(destination),
                "--mode", mode, "--expected-steps", str(result["completed_steps"])]
            elapsed = run(command, environment, out / f"eval_final_{mode}.log",
                          resource_path, physical_gpu, gpu_uuid)
            record["commands"].append({"kind": f"eval_final_{mode}", "command": command,
                                       "wall_s": elapsed})
            record_path.write_text(json.dumps(record, indent=2) + "\n")
        midpoint_wall = sum(event["wall_s"] for event in result["fixed_diagnostic_events"])
        fixed_wall = midpoint_wall + sum(row["wall_s"] for row in record["commands"]
                                         if row["kind"].startswith("eval_final"))
        if fixed_wall > 1200:
            raise RuntimeError(f"fixed diagnostic wall cap exceeded: {fixed_wall}")
        record.update(status="COMPLETED", finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            fixed_diagnostic_wall_s=fixed_wall, training_status=result["status"],
            completed_steps=result["completed_steps"], final_model_sha256=digest(model_after),
            final_state_sha256=digest(training / "training_state.pt"))
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    except BaseException as exc:
        record.update(status="FAILED", error=repr(exc), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
