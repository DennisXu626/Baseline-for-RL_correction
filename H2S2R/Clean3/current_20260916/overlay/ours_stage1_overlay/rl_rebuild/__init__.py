"""Isolated Ours PPO overlay with the Clean3 runtime package as fallback."""

from pathlib import Path


__path__ = [
    str(Path(__file__).resolve().parent),
    "/ssd/sy/kailang/pour17/direct58d/rl_rebuild",
]

