"""CPU-only deployed-overlay check: no periodic dump may reappear after construction."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from runtime.training_pilot_20260912 import resume_observation


ROOT = Path("/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/traceback_intervention_01")
OUT = ROOT / "cpu_periodic_disabled"
identity = json.loads((ROOT / "overlay_identity.json").read_text(encoding="utf-8"))
loaded = Path(resume_observation.__file__).resolve()
expected = Path(identity["target"]).resolve()
if loaded != expected:
    raise RuntimeError(f"wrong resume_observation import: {loaded}")
if hashlib.sha256(loaded.read_bytes()).hexdigest() != identity["target_sha256"]:
    raise RuntimeError("loaded overlay hash mismatch")
if os.environ.get("H2S2R_DISABLE_PERIODIC_TRACEBACK") != "1":
    raise RuntimeError("disable flag absent")
observation = resume_observation.Observation(OUT)
time.sleep(31.5)
observation.stacks.flush()
stack_size = (OUT / "stacks.log").stat().st_size
if stack_size != 0:
    raise RuntimeError(f"periodic traceback was emitted: {stack_size} bytes")
source = loaded.read_text(encoding="utf-8")
checks = {
    "loaded_from_overlay": True,
    "loaded_sha256": identity["target_sha256"],
    "disable_flag": "1",
    "wait_seconds": 31.5,
    "stacks_bytes_after_wait": stack_size,
    "faulthandler_enable_preserved": "faulthandler.enable(file=self.stacks, all_threads=True)" in source,
    "sigusr1_registration_preserved": "faulthandler.register(signal.SIGUSR1" in source,
    "default_periodic_call_preserved_behind_gate": source.count("dump_traceback_later(30, repeat=True, file=self.stacks)") == 1,
}
if not all(value is True for key, value in checks.items() if key.endswith("preserved") or key == "loaded_from_overlay"):
    raise RuntimeError(f"preservation check failed: {checks}")
observation.stacks.close()
(ROOT / "cpu_periodic_disabled.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(checks, sort_keys=True))
