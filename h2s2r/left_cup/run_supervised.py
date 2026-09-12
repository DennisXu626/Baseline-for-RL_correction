"""One-child deadline supervisor using the same function covered by CPU tests."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from runtime_support import sha256_file, supervise_process


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)

root = Path(__file__).resolve().parent
python = Path("/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python")
bundle = Path("/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17")
command = [str(python), str(root / "replay_contact.py"),
           "--bundle_root", str(bundle), "--commands", str(root / "command_inputs.npz"),
           "--out", str(args.out), "--device", args.device, "--headless"]
metadata = {
    "mode": "D",
    "entry_sha256": sha256_file(root / "replay_contact.py"),
    "runtime_support_sha256": sha256_file(root / "runtime_support.py"),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "PYTHONPATH": os.environ.get("PYTHONPATH"),
    "TMPDIR": os.environ.get("TMPDIR"),
}
result = supervise_process(cmd=command, cwd=root,
                           launch_dir=args.out.parent / (args.out.name + "_launch"),
                           output_dir=args.out, timeout_s=900,
                           env=os.environ.copy(), metadata=metadata)
if result["status"] != "COMPLETE":
    raise SystemExit(1)
