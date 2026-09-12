"""Single bounded WP4 scene probe; no queue, retry or training entry."""
from pathlib import Path
import json
import os
import subprocess
import time
import argparse
import re
from .geometry import sha256


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--name", default="reset_hold_v1")
    ap.add_argument("--control-steps",type=int,choices=(1,40),default=40)
    args=ap.parse_args()
    if re.fullmatch(r"[a-z0-9_]+",args.name) is None:
        ap.error("simple output name required")
    root = Path.cwd()
    parent = root.parent
    output = root/"output"/args.name
    record_path = root/(args.name+"_process.json")
    if output.exists() or record_path.exists():
        raise RuntimeError("existing attempt retained; inspect before any versioned repair")
    gpu = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free,utilization.gpu","--format=csv,noheader,nounits"], text=True)
    selected = next(line.split(",") for line in gpu.splitlines() if line.split(",")[0].strip()=="0")
    if int(selected[2]) < 24576:
        raise RuntimeError("insufficient free memory on GPU0")
    processes = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv"],text=True)
    reference = parent/"wp2_20260912_01/artifacts/pour17_v1/wp2_20260912_01/reference_build_v2/reference_s1_v2_bound.npz"
    assert sha256(reference)=="c478d989e0175e831f63080e702f9d7e995205f1e558c042ee77a4b84b2189c5"
    bundle = Path("/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17")
    command=[str(parent/"venv/bin/python"),"-m","adaptation.pour17_v1.wp4_scene_probe",
        "--bundle-root",str(bundle),"--robot-urdf",str(parent/"assets/vega_1p_sharpa_fix.urdf"),
        "--reference",str(reference),"--root-overlay",str(root/"configs/pour17_v1/wp4_root_overlay.json"),"--output-dir",str(output),"--control-steps",str(args.control_steps),"--device","cuda:0","--headless","--enable_cameras"]
    record=dict(command=command,started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),gpu=gpu,
        other_processes_untouched=processes,status="RUNNING",source_sha256={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name("wp4_scene_probe.py"))})
    record_path.write_text(json.dumps(record,indent=2)+"\n")
    start=time.perf_counter()
    with (root/(args.name+"_console.log")).open("w") as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONPATH=str(root),TMPDIR=str(parent/"tmp"),XDG_CACHE_HOME=str(parent/"cache")))
    evidence=json.loads((output/"result.json").read_text()) if (output/"result.json").exists() else {}
    record.update(returncode=result.returncode,wall_s=time.perf_counter()-start,
        status=evidence.get("status","FAILED_MISSING_RESULT"))
    record_path.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record))


if __name__=="__main__":
    main()
