"""One bounded WP2 GPU job, with live memory checks and complete timing.

No queue, automatic retries, kills, formal training, or host setting changes.
Call again only after inspecting a failed result and its artifacts.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import re

from .geometry import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=("camera", "contacts", "old_seed_0", "old_seed_1", "openloop", "ppo", "batch"), required=True)
    ap.add_argument("--name", required=True, help="new output directory name, e.g.camera_v1")
    ap.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    ap.add_argument("--allow-shared", action="store_true", help="user approved sharing2026-09-12; never changes other jobs")
    ap.add_argument("--enable-contact-readback", action="store_true", help="contact probe only; GPU physics must remain enabled")
    ap.add_argument("--tensor-pair-probe", action="store_true", help="contact probe only; static table as GPU sensor")
    a = ap.parse_args()
    if re.fullmatch(r"[a-z0-9_]+", a.name) is None:
        ap.error("name must be a simple lowercase job directory name")
    root = Path.cwd().resolve()
    parent = root.parent
    output = root / "output" / a.name
    if output.exists():
        raise RuntimeError(f"will not overwrite an existing job directory: {output}")
    gpu_lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    gpu_state = {int(parts[0].strip()): [p.strip() for p in parts[1:]] for parts in (line.split(",") for line in gpu_lines.splitlines())}
    uuid, free_mib, utilization = gpu_state[a.gpu]
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name", "--format=csv,noheader"], text=True)
    occupied = [line for line in processes.splitlines() if uuid in line]
    if (occupied and not a.allow_shared) or int(free_mib) < 24576:
        print(json.dumps({"status": "WAIT_INSUFFICIENT_RESERVED_MEMORY_OR_SHARING_NOT_ALLOWED", "processes": occupied,
                          "gpu": a.gpu, "free_mib": int(free_mib), "required_free_mib": 24576}), flush=True)
        return 75
    bundle = Path("/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17")
    urdf = parent / "assets/vega_1p_sharpa_fix.urdf"
    old = parent / "artifacts/wp1_20260912_02"
    inputs = root / "artifacts/pour17_v1/wp2_20260912_01/reference_build_v2"
    reference = inputs / "reference_s1_v2_bound.npz"
    interpreter = parent / "venv/bin/python"
    base = ["--bundle-root", str(bundle), "--robot-urdf", str(urdf), "--output-dir", str(output), "--device", "cuda:0", "--headless"]
    if a.job in ("camera", "old_seed_0", "old_seed_1", "openloop"):
        command = [str(interpreter), "-m", "adaptation.pour17_v1.render_capture", *base, "--enable_cameras"]
        if a.job == "openloop":
            command += ["--mode", "openloop", "--reference", str(reference)]
        else:
            if sha256(old / "final_model.zip") != "f8cd6f2a79e92c9028f903e7ac3198cf3b365677aaea6e95027ed698b750dae1":
                raise RuntimeError("WP1 8192 checkpoint hash mismatch")
            if sha256(old / "reference.npz") != "46c56cdc1b1775d57be405d4188d9616ffbb1606684c3989f1d5108e95b042ee":
                raise RuntimeError("WP1 reference hash mismatch")
            command += ["--compat", "--mode", "probe" if a.job == "camera" else "eval", "--reference", str(old / "reference.npz"),
                        "--model", str(old / "final_model.zip"), "--seed", str(20260830 if a.job == "old_seed_1" else 20260829)]
    elif a.job == "contacts":
        command = [str(interpreter), "-m", "adaptation.pour17_v1.runtime_checks", *base, "--reference", str(reference)]
        if a.enable_contact_readback:
            command += ["--enable-contact-readback"]
        if a.tensor_pair_probe:
            command += ["--tensor-pair-probe"]
    elif a.job == "ppo":
        # The first8192 update integration only. No implicit budget expansion.
        command = [str(interpreter), "-m", "adaptation.pour17_v1.train", *base, "--reference", str(reference),
                   "--resolved-config", str(root / "configs/pour17_v1/wp2_train.yaml"),
                   "--schema", str(inputs / "observation_action_schema_v2.json"),
                   "--seed", "1701", "--num-envs", "32", "--n-steps", "128", "--total-timesteps", "8192"]
    else:
        model = root / "output/ppo_v1/final_model.zip"
        command = [str(interpreter), "-m", "adaptation.pour17_v1.batch_eval", *base, "--reference", str(reference),
                   "--model", str(model), "--num-envs", "32", "--episodes", "32", "--long-horizon"]
    if a.job not in ("camera", "old_seed_0", "old_seed_1"):
        binding = json.loads((inputs / "wp2_input_binding.json").read_text())
        expected = next(x["output_sha256"] for x in binding["bindings"] if x["output"] == reference.name)
        if sha256(reference) != expected:
            raise RuntimeError("WP2 reference hash mismatch, possibly incomplete upload")
    output.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(parent / "tmp"),
               XDG_CACHE_HOME=str(parent / "cache"), PYTHONPATH=str(root), PYTHONUNBUFFERED="1")
    record = {"job": a.job, "command": command, "gpu_uuid": uuid, "physical_gpu": a.gpu,
              "free_mib_before": int(free_mib), "shared_execution_authorized": a.allow_shared,
              "other_processes_untouched": processes,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "RUNNING"}
    record_path = output / "job_process.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    started = time.perf_counter()
    with (output / "console.log").open("w") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    # Kit shutdown can mask a raised exception with exit0. Require the named
    # result too; successful process exit alone is never an acceptance check.
    result_name = ("contact_validation.json" if a.job == "contacts" else
                   "ppo_update_result.json" if a.job == "ppo" else
                   "batch_eval_result.json" if a.job == "batch" else "result.json")
    result_path = output / result_name
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    evidence_ok = result.get("status") in ("PASS", "RECORDED_REEXECUTION", "COMPLETED_DEVELOPMENT_BATCH")
    record.update(returncode=completed.returncode, process_wall_s=time.perf_counter() - started,
                  required_result=result_name, evidence_status=result.get("status", "MISSING"),
                  status="PROCESS_AND_RESULT_COMPLETED" if completed.returncode == 0 and evidence_ok else "FAILED_OR_MISSING_EVIDENCE")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return completed.returncode or (0 if evidence_ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
