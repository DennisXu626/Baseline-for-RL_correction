"""Create one isolated resume_observation.py with the authorized one-line gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SOURCE = Path("/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v13_256_pilot_20260912/runtime/training_pilot_20260912/resume_observation.py")
ROOT = Path("/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/traceback_intervention_01")
TARGET = ROOT / "overlay/runtime/training_pilot_20260912/resume_observation.py"
BEFORE_SHA = "7fe7b1371a4c2aa493454ac74e27a84c8292f6f78e084673e674744f3030b9ab"
OLD = "        faulthandler.dump_traceback_later(30, repeat=True, file=self.stacks)\n"
NEW = (
    "        if os.environ.get('H2S2R_DISABLE_PERIODIC_TRACEBACK') != '1':\n"
    "            faulthandler.dump_traceback_later(30, repeat=True, file=self.stacks)\n"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if TARGET.exists():
    raise FileExistsError(f"refusing to overwrite {TARGET}")
if sha(SOURCE) != BEFORE_SHA:
    raise RuntimeError("frozen source identity mismatch")
text = SOURCE.read_text(encoding="utf-8")
if text.count(OLD) != 1:
    raise RuntimeError("periodic traceback call is not unique")
after = text.replace(OLD, NEW)
if after.count("dump_traceback_later(") != 1 or after.count("H2S2R_DISABLE_PERIODIC_TRACEBACK") != 1:
    raise RuntimeError("unexpected overlay edit")
TARGET.parent.mkdir(parents=True, exist_ok=False)
TARGET.write_text(after, encoding="utf-8")
record = {
    "schema": "h2s2r_traceback_overlay_identity_v1",
    "source": str(SOURCE),
    "source_sha256": BEFORE_SHA,
    "target": str(TARGET),
    "target_sha256": sha(TARGET),
    "changed_behavior": "skip only dump_traceback_later when H2S2R_DISABLE_PERIODIC_TRACEBACK=1",
    "default_behavior_preserved": True,
    "faulthandler_enable_preserved": True,
    "sigusr1_registration_preserved": True,
}
(ROOT / "overlay_identity.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True))
