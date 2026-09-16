"""Isolated Ours PPO overlay with the selected direct-58D package as fallback."""

import os
from pathlib import Path


__path__ = [
    str(Path(__file__).resolve().parent),
    os.environ.get(
        "H2S2R_BASE_RL_REBUILD",
        "/ssd/sy/kailang/pour17/direct58d/rl_rebuild",
    ),
]

