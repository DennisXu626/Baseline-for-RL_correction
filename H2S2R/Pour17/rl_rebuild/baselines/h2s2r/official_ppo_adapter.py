"""Compatibility layer for H2S2R's released recurrent PPO implementation.

The simulator continues to expose the same 342-D observation and 22-D action
contract.  This adapter only translates Isaac Lab's Gymnasium vector wrapper to
the legacy ``gym`` spaces and small environment API expected by upstream PPO.
"""

from __future__ import annotations

from typing import Any

import gym
import numpy as np


class OfficialPpoEnvAdapter:
    """Expose the current H2S2R environment through upstream PPO's API."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.last_extras: dict[str, Any] = {}
        self._observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(env.num_obs),),
            dtype=np.float32,
        )
        self._action_space = gym.spaces.Box(
            low=-float(env.clip_actions),
            high=float(env.clip_actions),
            shape=(int(env.num_actions),),
            dtype=np.float32,
        )

    def get_env_info(self) -> dict[str, Any]:
        return {
            "observation_space": self._observation_space,
            "action_space": self._action_space,
            "agents": 1,
            "value_size": 1,
        }

    def reset(self) -> dict[str, Any]:
        return self.env.reset()

    def step(self, actions):
        observations, rewards, dones, extras = self.env.step(actions)
        self.last_extras = extras
        return observations, rewards, dones, extras

    def set_train_info(self, frame: int, agent: Any) -> None:
        """Upstream callback; this environment has no frame-dependent schedule."""

    def get_env_state(self) -> None:
        """The deterministic simulator reset is reconstructed from the run config."""
        return None

    def set_env_state(self, state: None) -> None:
        if state is not None:
            raise ValueError("H2S2R Pour17 checkpoints do not contain simulator state")

    def close(self):
        return self.env.close()

