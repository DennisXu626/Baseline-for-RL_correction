"""One-file/one-line pre-policy exit guard for the native-debug reproduction."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import sys


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


target_text = os.environ.get("H2S2R_NATIVE_DEBUG_GUARD_TARGET")
line_text = os.environ.get("H2S2R_NATIVE_DEBUG_GUARD_LINE")
marker_text = os.environ.get("H2S2R_NATIVE_DEBUG_GUARD_MARKER")

if target_text or line_text or marker_text:
    if not (target_text and line_text and marker_text):
        raise RuntimeError("native-debug guard requires target, line, and marker")
    target = Path(target_text).resolve()
    target_line = int(line_text)
    marker = Path(marker_text).resolve()
    if not target.is_absolute() or target_line <= 0 or not marker.is_absolute():
        raise RuntimeError("native-debug guard inputs must be absolute and positive")

    def _guard(frame, event, arg):
        if event == "line" and frame.f_lineno == target_line:
            try:
                current = Path(frame.f_code.co_filename).resolve()
            except OSError:
                return _guard
            if current == target:
                _atomic_json(
                    marker,
                    {
                        "schema": "h2s2r_native_debug_init_return_guard_v1",
                        "status": "INIT_RETURNED_CRASH_NOT_REPRODUCED",
                        "timestamp_utc": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                        "pid": os.getpid(),
                        "target": str(target),
                        "line_before_execution": target_line,
                        "policy_controls": 0,
                        "ppo_updates": 0,
                        "exit_code": 86,
                    },
                )
                os._exit(86)
        return _guard

    sys.settrace(_guard)
