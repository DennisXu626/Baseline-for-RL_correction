"""Run preflight checks, then launch the requested command."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from preflight_checks import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C3-P2 preflight then launch command.")
    parser.add_argument("--runtime_root", type=Path, required=True)
    parser.add_argument("--v12_root", type=Path, required=True)
    parser.add_argument("--preflight_output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("missing command")
    status = run(args.runtime_root, args.v12_root)
    args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_output.write_text(__import__("json").dumps(status, indent=2) + "\n", encoding="utf-8")
    if not status["passed"]:
        raise SystemExit(1)
    completed = subprocess.run(args.command, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
