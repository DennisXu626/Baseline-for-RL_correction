"""Clean3 bridge from Isaac Lab observations to H2S2R's released PPO API."""

from __future__ import annotations

from typing import Any

import gym
import numpy as np


class OfficialPpoEnvAdapter:
    """Expose actor observations and privileged state with legacy gym spaces."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.last_extras: dict[str, Any] = {}
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(env.num_obs),), dtype=np.float32
        )
        self._action_space = gym.spaces.Box(
            low=-float(env.clip_actions), high=float(env.clip_actions),
            shape=(int(env.num_actions),), dtype=np.float32,
        )
        # The vendored wrapper checks the deprecated ``num_states`` attribute
        # before consulting the critic space.  Isaac Lab 0.54 exposes
        # ``state_space`` instead, so obtain the configured dimension from the
        # unwrapped DirectRLEnv without changing the shared wrapper used by
        # other tracks. Reset and step outputs are independently shape-checked.
        configured_state = env.unwrapped.cfg.state_space
        if not isinstance(configured_state, int):
            raise RuntimeError(
                f"Clean3 critic state_space must be an integer, got {configured_state!r}"
            )
        state_dim = configured_state
        if state_dim <= 0:
            raise RuntimeError("H2S2R asymmetric critic requires a non-empty critic state")
        self._state_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32,
        )

    @staticmethod
    def _map_states(observations: dict[str, Any]) -> dict[str, Any]:
        if "critic" not in observations:
            raise RuntimeError("Clean3 environment omitted the asymmetric critic state")
        observations["states"] = observations["critic"]
        return observations

    def get_env_info(self) -> dict[str, Any]:
        return {
            "observation_space": self._observation_space,
            "action_space": self._action_space,
            "state_space": self._state_space,
            "agents": 1,
            "value_size": 1,
        }

    def reset(self) -> dict[str, Any]:
        return self._map_states(self.env.reset())

    def step(self, actions):
        observations, rewards, dones, extras = self.env.step(actions)
        self.last_extras = extras
        return self._map_states(observations), rewards, dones, extras

    def set_train_info(self, frame: int, agent: Any) -> None:
        """Upstream callback; this environment has no frame-dependent schedule."""

    def get_env_state(self) -> None:
        return None

    def set_env_state(self, state: None) -> None:
        if state is not None:
            raise ValueError("Clean3 checkpoints do not contain simulator state")

    def close(self):
        return self.env.close()
