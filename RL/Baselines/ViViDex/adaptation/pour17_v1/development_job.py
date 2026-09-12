"""One authorized continuous WP4 development run; no retry or formal launch."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path(__file__).resolve().parents[2]
    wp4 = root.parent
    parent = wp4.parent
    out = root / "artifacts/pour17_v1/development_20260912_01"
    out.mkdir(parents=True, exist_ok=False)
    ledger = out.parent / "development_training_ledger.json"
    prior = parent / "wp2_20260912_01/output/development_training_ledger.json"
    ledger.write_bytes(prior.read_bytes())
    reference = wp4 / "artifacts/pour17_v1/wp4_20260912_01/reference_build_v1/reference_wp4.npz"
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == "34c8f30eaf37a674d5eabf3943218cdba5dd1612014d87b8f9055c45978fe649"
    command = [str(parent / "venv/bin/python"), "-m", "adaptation.pour17_v1.train",
        "--bundle-root", "/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17",
        "--robot-urdf", str(parent / "assets/vega_1p_sharpa_fix.urdf"),
        "--reference", str(reference), "--root-overlay", str(root / "configs/pour17_v1/wp4_root_overlay.json"),
        "--resolved-config", str(root / "configs/pour17_v1/wp2_train.yaml"),
        "--schema", str(root / "configs/pour17_v1/development_schema.json"),
        "--output-dir", str(out), "--seed", "1701", "--num-envs", "32", "--n-steps", "128",
        "--total-timesteps", "57344", "--device", "cuda:0", "--headless"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="1", TMPDIR=str(parent / "tmp"),
        XDG_CACHE_HOME=str(parent / "cache"), PYTHONPATH=str(root), PYTHONUNBUFFERED="1")
    record = {"command": command, "physical_gpu": 1, "pid": os.getpid(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "RUNNING"}
    path = out / "job_process.json"
    path.write_text(json.dumps(record, indent=2))
    with (out / "console.log").open("w") as log:
        result = subprocess.run(command, env=environment, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    record.update(status="EXITED_CHECK_RESULT", returncode=result.returncode)
    path.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
