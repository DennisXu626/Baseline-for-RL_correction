"""One approved two-slot270-control reference execution; no policy entry."""
from pathlib import Path
import json
import os
import subprocess
import time
from .geometry import sha256


def main():
    root = Path.cwd(); parent = root.parent
    artifact = root/"artifacts/pour17_v1/wp4_20260912_01"
    assert json.loads((artifact/"static_separation_gate.json").read_text())["status"] == "PASS_FIXED_ROOT_AND_STATIC_TORSO_SEPARATION_ONLY"
    reference = artifact/"reference_build_v1/reference_wp4.npz"
    manifest = json.loads((artifact/"reference_build_v1/reference_rebuild_manifest.json").read_text())
    assert sha256(reference) == manifest["output_sha256"]
    output = root/"output/reference_v1"
    record_path = root/"reference_v1_process.json"
    assert not output.exists() and not record_path.exists()
    gpu = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    assert int(next(l.split(',')[2] for l in gpu.splitlines() if l.split(',')[0].strip()=='0')) >= 24576
    others = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv"],text=True)
    command = [str(parent/"venv/bin/python"),"-m","adaptation.pour17_v1.wp4_rollout",
        "--bundle-root","/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17",
        "--robot-urdf",str(parent/"assets/vega_1p_sharpa_fix.urdf"),"--reference",str(reference),
        "--root-overlay",str(root/"configs/pour17_v1/wp4_root_overlay.json"),
        "--output-dir",str(output),"--device","cuda:0","--headless","--enable_cameras"]
    record = dict(status="RUNNING",command=command,gpu=gpu,other_processes_untouched=others,
        started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),reference_sha256=sha256(reference),
        script_sha256=sha256(Path(__file__)),rollout_sha256=sha256(Path(__file__).with_name("wp4_rollout.py")))
    record_path.write_text(json.dumps(record,indent=2)+"\n")
    start = time.perf_counter()
    with (root/"reference_v1_console.log").open('w') as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONPATH=str(root),TMPDIR=str(parent/"tmp"),XDG_CACHE_HOME=str(parent/"cache")))
    evidence = json.loads((output/"result.json").read_text()) if (output/"result.json").exists() else {}
    record.update(status=evidence.get("status","FAILED"),returncode=result.returncode,wall_s=time.perf_counter()-start)
    record_path.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
