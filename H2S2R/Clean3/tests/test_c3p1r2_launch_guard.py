from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "deployment_c3p1r2" / "launch_guard.py"
READY = "READY_FOR_TEST"
SUCCESS = "SUCCESS_FOR_TEST"


def run_case(source: str):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        command = [
            sys.executable, str(GUARD),
            "--log", str(root / "console.log"),
            "--status", str(root / "status.json"),
            "--ready_marker", READY,
            "--success_marker", SUCCESS,
            "--init_seconds", "5", "--total_seconds", "5",
            "--poll_seconds", "0.05",
            sys.executable, "-c", source,
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
        console = (root / "console.log").read_text(encoding="utf-8")
        return completed.returncode, status, console


def test_normal_completion_requires_both_markers():
    code, status, _ = run_case(f"print('{READY}', flush=True); print('{SUCCESS}', flush=True)")
    assert code == 0
    assert status["state"] == "completed"
    assert status["ready_marker_seen"] and status["success_marker_seen"]


def test_exception_is_failed_and_evidence_remains():
    code, status, console = run_case(f"print('{READY}', flush=True); raise RuntimeError('fixture')")
    assert code != 0
    assert status["state"] == "failed"
    assert "RuntimeError: fixture" in console


def test_direct_zero_exit_without_success_is_incomplete():
    source = f"import os; print('{READY}', flush=True); os._exit(0)"
    code, status, _ = run_case(source)
    assert code != 0
    assert status["state"] == "incomplete_without_success_marker"
    assert status["ready_marker_seen"] and not status["success_marker_seen"]


def test_failure_marker_forces_failed_state():
    source = f"import os; print('{READY}', flush=True); print('{SUCCESS}', flush=True); print('[CLEAN3-ENTRY-FAILURE] failure', flush=True); os._exit(0)"
    code, status, _ = run_case(source)
    assert code != 0
    assert status["state"] == "failed"
    assert status["failure_marker_seen"]


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} tests")
