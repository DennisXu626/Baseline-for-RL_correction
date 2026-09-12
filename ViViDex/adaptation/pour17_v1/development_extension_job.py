"""Run DEV_EXTEND_262144_V1 once: endpoint diagnostics, resume, diagnostics."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, environment, log_path):
    started = time.perf_counter()
    with log_path.open("w") as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}; see {log_path}")
    return time.perf_counter() - started


def main():
    source_root = Path(__file__).resolve().parents[2]
    wp4_root = source_root.parent
    base_root = wp4_root.parent
    old = wp4_root / "development_runtime_20260912_01/artifacts/pour17_v1/development_20260912_01"
    out = source_root / "artifacts/pour17_v1/development_extension_20260912_01"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(old.parent / "development_training_ledger.json", out / "development_training_ledger.json")
    model_before = old / "final_model.zip"
    state_before = old / "training_state.pt"
    expected = {
        "model": "8225439c081d09a7c5d83c1852e5dec17c7a51220a216f6e3621f514f5ba9699",
        "reference": "34c8f30eaf37a674d5eabf3943218cdba5dd1612014d87b8f9055c45978fe649"}
    if digest(model_before) != expected["model"] or not state_before.is_file():
        raise RuntimeError("registered57,344 checkpoint/state unavailable")
    reference = wp4_root / "artifacts/pour17_v1/wp4_20260912_01/reference_build_v1/reference_wp4.npz"
    if digest(reference) != expected["reference"]:
        raise RuntimeError("registered WP4 reference mismatch")
    common = ["--bundle-root", "/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17",
        "--robot-urdf", str(base_root / "assets/vega_1p_sharpa_fix.urdf"),
        "--reference", str(reference), "--root-overlay", str(source_root / "configs/pour17_v1/wp4_root_overlay.json"),
        "--device", "cuda:0", "--headless"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="1", TMPDIR=str(base_root / "tmp"),
        XDG_CACHE_HOME=str(base_root / "cache"), PYTHONPATH=str(source_root), PYTHONUNBUFFERED="1")
    record = {"authorization": "DEV_EXTEND_262144_V1", "physical_gpu": 1,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "RUNNING",
        "before_model": {"path": str(model_before), "sha256": digest(model_before)},
        "before_state": {"path": str(state_before), "sha256": digest(state_before)}, "commands": []}
    record_path = out / "extension_job.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        for endpoint, model, steps in (("before", model_before, 57_344),):
            for mode in ("stochastic", "deterministic"):
                destination = out / f"eval_{endpoint}_{mode}"
                command = [sys.executable, "-m", "adaptation.pour17_v1.policy_diagnostic_eval",
                    *common, "--model", str(model), "--output-dir", str(destination),
                    "--mode", mode, "--expected-steps", str(steps)]
                elapsed = run(command, environment, out / f"eval_{endpoint}_{mode}.log")
                record["commands"].append({"kind": f"eval_{endpoint}_{mode}", "command": command, "wall_s": elapsed})
                record_path.write_text(json.dumps(record, indent=2) + "\n")
        training = out / "training"
        command = [sys.executable, "-m", "adaptation.pour17_v1.train", *common,
            "--resolved-config", str(source_root / "configs/pour17_v1/wp2_train.yaml"),
            "--schema", str(source_root / "configs/pour17_v1/development_schema.json"),
            "--output-dir", str(training), "--seed", "1701", "--num-envs", "32", "--n-steps", "128",
            "--total-timesteps", "262144", "--resume", str(model_before),
            "--development-extension-approved"]
        elapsed = run(command, environment, out / "training.log")
        record["commands"].append({"kind": "resume_training", "command": command, "wall_s": elapsed})
        model_after = training / "final_model.zip"
        for mode in ("stochastic", "deterministic"):
            destination = out / f"eval_after_{mode}"
            command = [sys.executable, "-m", "adaptation.pour17_v1.policy_diagnostic_eval",
                *common, "--model", str(model_after), "--output-dir", str(destination),
                "--mode", mode, "--expected-steps", "262144"]
            elapsed = run(command, environment, out / f"eval_after_{mode}.log")
            record["commands"].append({"kind": f"eval_after_{mode}", "command": command, "wall_s": elapsed})
            record_path.write_text(json.dumps(record, indent=2) + "\n")
        evaluation_wall = sum(row["wall_s"] for row in record["commands"] if row["kind"].startswith("eval_"))
        if evaluation_wall > 1200:
            raise RuntimeError(f"evaluation/startup wall cap exceeded: {evaluation_wall}")
        record.update(status="COMPLETED", finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            evaluation_wall_s=evaluation_wall, final_model_sha256=digest(model_after))
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    except BaseException as exc:
        record.update(status="FAILED", error=repr(exc), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
