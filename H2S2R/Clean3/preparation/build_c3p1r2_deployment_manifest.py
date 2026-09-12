"""Build the local review/upload manifest for the independent R2 overlay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
DEPLOY = ROOT / "deployment_c3p1r2"
REMOTE = PurePosixPath("/media/msc-auto/HDD/users/kailang/h2s2r_clean3_c3p1r2_20260912/deployment_v3")
V12 = PurePosixPath("/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v12_training_pilot_20260912")
PYTHON = PurePosixPath("/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python")
FABRICS = PurePosixPath("/home/kailang/experiments/baselines/fabrics_h2s2r_reference/src")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    files = {}
    for path in sorted(DEPLOY.rglob("*")):
        if path.is_file() and path.name not in {"deployment_manifest.json"} and "__pycache__" not in path.parts:
            files[path.relative_to(DEPLOY).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    prefix = f"PYTHONPATH={REMOTE}:{V12}:{FABRICS} CUDA_VISIBLE_DEVICES=0 RL_ISAAC_NO_GUARD=1"
    def preflight_for(artifacts_dir: PurePosixPath) -> str:
        return (
            f"{PYTHON} -u {REMOTE}/run_with_preflight.py --runtime_root {REMOTE} --v12_root {V12} "
            f"--preflight_output {artifacts_dir}/preflight.json --"
        )
    commands = {
        "physics_gate": (
            f"{prefix} {PYTHON} -u {REMOTE}/launch_guard.py --log {REMOTE.parent}/physics_gate_02/console.log "
            f"--status {REMOTE.parent}/physics_gate_02/process_status.json --ready_marker '[CLEAN3-GATE] stage=construct_env complete' "
            f"--success_marker '[CLEAN3-GATE] stage=run complete' "
            f"--init_seconds 300 --total_seconds 2700 "
            f"{preflight_for(REMOTE.parent / 'physics_gate_02')} {PYTHON} -u {REMOTE}/tasks/h2s2r_clean3/physics_gate.py "
            f"--runtime_root {REMOTE} --v12_root {V12} --output_root {REMOTE.parent}/physics_gate_02/artifacts "
            "--device cuda:0 --headless --enable_cameras"
        ),
        "training": (
            f"{prefix} {PYTHON} -u {REMOTE}/launch_guard.py --log {REMOTE.parent}/training_01/console.log "
            f"--status {REMOTE.parent}/training_01/process_status.json --ready_marker '[CLEAN3] stage=construct_env complete' "
            f"--success_marker '[CLEAN3] stage=training complete' "
            f"--init_seconds 300 --total_seconds 7200 "
            f"{preflight_for(REMOTE.parent / 'training_01')} {PYTHON} -u {REMOTE}/tasks/h2s2r_clean3/train_lstm.py "
            f"--runtime_root {REMOTE} --v12_root {V12} --output_root {REMOTE.parent}/training_01/run "
            "--num_envs 256 --seed 42 --max_epochs 512 --device cuda:0 --headless"
        ),
        "evaluation_template": (
            f"{prefix} {PYTHON} -u {REMOTE}/launch_guard.py --log {REMOTE.parent}/evaluation_01/console.log "
            f"--status {REMOTE.parent}/evaluation_01/process_status.json --ready_marker '[CLEAN3-EVAL] stage=construct_env complete' "
            f"--success_marker '[CLEAN3-EVAL] stage=evaluation complete' "
            f"--init_seconds 300 --total_seconds 2700 "
            f"{preflight_for(REMOTE.parent / 'evaluation_01')} {PYTHON} -u {REMOTE}/tasks/h2s2r_clean3/evaluate.py "
            f"--runtime_root {REMOTE} --v12_root {V12} --initial_checkpoint INITIAL_CHECKPOINT "
            f"--final_checkpoint FINAL_CHECKPOINT --output_root {REMOTE.parent}/evaluation_01/artifacts "
            "--device cuda:0 --headless --enable_cameras"
        ),
    }
    payload = {
        "schema": "h2s2r_clean3_c3p1r2_deployment_manifest_v1",
        "local_root": str(DEPLOY.resolve()), "remote_target": str(REMOTE),
        "upload_scope": "only files listed below; no Pour17/shared-upstream modification",
        "read_only_dependencies": {"v12": str(V12), "fabrics": str(FABRICS), "python": str(PYTHON)},
        "files": files, "total_files": len(files), "total_bytes": sum(v["bytes"] for v in files.values()),
        "launch_commands": commands,
        "previous_rejection": {
            "action": "scp of the earlier Clean3 implementation source payload to the remote HDD target",
            "transferred_bytes": 0,
            "reason_verbatim": "This uploads internal Clean3 implementation source code to an unverified remote destination; the user authorized remote storage for large artifacts and trajectories, but did not specifically authorize exporting this source payload to that destination. The agent must not attempt to work around that approval requirement. Proceed only if the user explicitly approves after being informed.",
            "current_authorization": "The C3-P1R2 user request explicitly authorizes normal approval handling for a local reviewable deployment package, its SHA manifest, and an independent target path; no bypass is permitted.",
        },
    }
    (DEPLOY / "launch_commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    # Recompute because launch_commands.json is part of the upload payload.
    path = DEPLOY / "launch_commands.json"
    payload["files"][path.relative_to(DEPLOY).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    payload["total_files"] = len(payload["files"])
    payload["total_bytes"] = sum(v["bytes"] for v in payload["files"].values())
    (DEPLOY / "deployment_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
