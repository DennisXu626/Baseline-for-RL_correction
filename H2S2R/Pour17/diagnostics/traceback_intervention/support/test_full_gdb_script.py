"""Run the exact deployable GDB script against a real CPU SIGSEGV."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/traceback_intervention_01")
SUPPORT = ROOT / "support"
LOG = ROOT / "cpu_full_gdb_02.txt"
CORE = ROOT / "cpu_full_gdb_02.core"
env = dict(os.environ)
env.update({
    "H2S2R_GDB_LOG": str(LOG),
    "H2S2R_GDB_CORE": str(CORE),
    "H2S2R_GDB_INJECT_COMMAND_FAILURE": "1",
})
result = subprocess.run([
    "/usr/bin/gdb", "-q", "-nx", "-batch", "-x", str(SUPPORT / "gdb_capture_full.gdb"),
    "--args", "/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python",
    str(SUPPORT / "cpu_sigsegv_target.py"),
], env=env, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
text = LOG.read_text(encoding="utf-8", errors="replace")
combined = result.stdout + "\n" + text
checks = {
    "actual_deployed_script": str(SUPPORT / "gdb_capture_full.gdb"),
    "signal": "received signal SIGSEGV" in combined,
    "full_backtrace": "===== FULL BACKTRACE =====" in text and "#0" in text,
    "registers": "===== REGISTERS =====" in text and "rip" in text,
    "shared_libraries": "===== SHARED LIBRARIES =====" in text,
    "all_threads": "===== ALL THREADS =====" in text and "Thread" in text,
    "injected_failure_observed": "SECTION_ERROR INJECTED SINGLE COMMAND FAILURE" in text,
    "continued_after_failure": text.index("INJECTED SINGLE COMMAND FAILURE") < text.index("CURRENT INSTRUCTIONS") < text.index("ALL THREADS"),
    "capture_complete": "===== NATIVE DEBUG CAPTURE COMPLETE =====" in text,
    "core_exists": CORE.is_file() and CORE.stat().st_size > 0,
    "gdb_return_code": result.returncode,
}
required = [value for key, value in checks.items() if key not in ("actual_deployed_script", "gdb_return_code")]
if not all(required):
    raise RuntimeError(f"full GDB CPU test failed: {checks}\n{result.stdout}")
(ROOT / "cpu_full_gdb_validation.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(checks, sort_keys=True))
