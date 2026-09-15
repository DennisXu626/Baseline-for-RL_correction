#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: status.sh OUTPUT PID}"; PID="${2:?missing pid}"
if test -f "$OUT/entry_failure.json"; then echo TRAINING_FAILED; cat "$OUT/entry_failure.json"
elif test -f "$OUT/training_exit.json"; then echo TRAINING_COMPLETE; cat "$OUT/training_exit.json"; test -f "$OUT/training_progress.json" && cat "$OUT/training_progress.json"
elif kill -0 "$PID" 2>/dev/null; then echo TRAINING_RUNNING; test -f "$OUT/training_progress.json" && cat "$OUT/training_progress.json"; tail -30 "$OUT.log"
else echo TRAINING_EXITED_WITHOUT_TERMINAL; tail -120 "$OUT.log"; fi
