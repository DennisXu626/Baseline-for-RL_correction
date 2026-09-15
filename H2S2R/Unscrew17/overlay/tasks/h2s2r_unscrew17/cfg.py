"""Clip17 Unscrew configuration for the direct-58D H2S2R baseline."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ACTION_DIM = 58
OBSERVATION_DIM = 342
CRITIC_STATE_DIM = 513

# Reproduce the recorded V82 task world.  These values select physics, reward,
# reset and curriculum behavior only; policy I/O is replaced below.
UNSCREW17_ENV = {
    "UNSCREW_CLIP": "17",
    "UNSCREW_CAP_GRASP": "user_thumb_index",
    "UNSCREW_CAP_PINCH": "8",
    "UNSCREW_CONTACT_TIMING": "5",
    "UNSCREW_CLOCK_HOLD_SEP": "1",
    "UNSCREW_CAP_REBASE": "1",
    "UNSCREW_SUCCESS_MODE": "separate5cm",
    "UNSCREW_GRIP_REWARD": "sustained_pair",
    "UNSCREW_CONTACT_GEOMETRY": "cap_local",
    "UNSCREW_SLIP_RULE": "contact_loss",
    "UNSCREW_ADV_REWARD": "pre_grasp",
    "UNSCREW_SUCCESS_CONTACT_FILTER": "two_step",
    "UNSCREW_SEPARATION_CREDIT": "physical_interval",
    "UNSCREW_TWIST_GAIN": "10",
    "UNSCREW_RWIDE": "3",
    "UNSCREW_CHAIN": "1",
    "UNSCREW_CERT_WAGE": "3.0",
    "UNSCREW_CAP_WAGE": "3.0",
    "UNSCREW_PINCH_WAGE": "0.05",
    "UNSCREW_CAP_APPROACH": "3.0",
    "UNSCREW_MAX_TILT": "80",
    "UNSCREW_STAGE_C_FRAC": "0.25",
    "UNSCREW_STAGE_C_SAMPLE_FRAC": "0",
    "UNSCREW_MAX_RSI_GATE": "2",
    "POUR_VARIANT": "HYB",
    "POUR_BONUS_DIST": "1",
    "POUR_BONUS_NOW": "1",
    "POUR_UNLOCK": "1,2",
    "POUR_SQUEEZE_FF": "1",
    "POUR_PAD_FRIC": "6.0",
    "SHARPA_WANDB": "0",
}


def _load_unscrew_task_module():
    """Import the frozen Clip17 task only after fixing its recorded switches."""

    for name, value in UNSCREW17_ENV.items():
        os.environ[name] = value
    ground_usd = Path(__file__).resolve().parents[2] / "assets/default_environment.usd"
    os.environ["RL_LOCAL_GROUND_USD"] = str(ground_usd)
    wiring = Path(__file__).resolve().parents[1] / "Unscrew/part4/C_Wiring"
    if not wiring.is_dir():
        raise FileNotFoundError(f"Clip17 task wiring not found: {wiring}")
    path = str(wiring)
    if path not in sys.path:
        sys.path.insert(0, path)
    print(f"[H2S2R-UNSCREW17] importing frozen task_env from {wiring}", flush=True)
    import task_env
    print("[H2S2R-UNSCREW17] frozen task_env import complete", flush=True)

    return task_env


def build_cfg(*, num_envs: int, seed: int):
    """Build the V82 world with the approved H2S2R policy/state contract."""

    task_env = _load_unscrew_task_module()
    cfg = task_env.build_cfg(num_envs=num_envs)
    cfg.seed = int(seed)
    cfg.action_space = ACTION_DIM
    cfg.observation_space = OBSERVATION_DIM
    cfg.state_space = CRITIC_STATE_DIM
    cfg.priv_info_dim = 0
    return cfg


def unscrew_task_module():
    return _load_unscrew_task_module()
