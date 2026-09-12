"""Serializable training RNG/curriculum state; simulator bit state excluded."""
from __future__ import annotations
from dataclasses import asdict
import copy
import random

import numpy as np
import torch

from .curriculum import CurriculumState


def capture(raw, curriculum, completed_steps, next_validation):
    return {"schema": "wp2_training_state_v2", "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "env_rng": [copy.deepcopy(r.bit_generator.state) for r in raw.env_rng],
        "global_env_rng": copy.deepcopy(raw.rng.bit_generator.state),
        "curriculum": asdict(curriculum), "completed_steps": int(completed_steps),
        "next_validation": int(next_validation), "simulator_bit_state_restored": False,
        "resume_policy": "new physical canonical episodes; restore algorithm and RNG, not simulator bit-state"}


def restore(raw, state):
    if state["schema"] != "wp2_training_state_v2" or len(state["env_rng"]) != raw.num_envs:
        raise ValueError("resume state/schema/num_envs mismatch")
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    for r, s in zip(raw.env_rng, state["env_rng"]):
        r.bit_generator.state = copy.deepcopy(s)
    raw.rng.bit_generator.state = copy.deepcopy(state["global_env_rng"])
    curriculum = CurriculumState(**state["curriculum"])
    raw.set_curriculum_stage(curriculum.stage)
    return curriculum, int(state["next_validation"])
