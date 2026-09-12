"""Thin Stable-Baselines3 VecEnv bridge preserving Isaac terminal states."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


class IsaacSB3VecEnv(VecEnv):
    def __init__(self, env):
        self.env = env
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(env.cfg.observation_space),), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(58,), dtype=np.float32)
        super().__init__(env.num_envs, observation_space, action_space)
        self._pending_actions: np.ndarray | None = None
        self._episode_returns = np.zeros(env.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(env.num_envs, dtype=np.int64)

    @staticmethod
    def _numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    def reset(self) -> np.ndarray:
        seed = self._seeds[0] if self._seeds and self._seeds[0] is not None else None
        observation, extras = self.env.reset(seed=seed)
        self.reset_infos = [dict(extras) for _ in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        return self._numpy(observation["policy"]).astype(np.float32, copy=False)

    def step_async(self, actions: np.ndarray) -> None:
        value = np.asarray(actions, dtype=np.float32)
        if value.shape != (self.num_envs, 58):
            raise ValueError(f"actions must have shape {(self.num_envs, 58)}, got {value.shape}")
        self._pending_actions = value

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_wait called before step_async")
        actions = torch.as_tensor(
            self._pending_actions, dtype=torch.float32, device=self.env.device)
        self._pending_actions = None
        observation, reward, terminated, truncated, extras = self.env.step(actions)
        obs_np = self._numpy(observation["policy"]).astype(np.float32, copy=False)
        reward_np = self._numpy(reward).astype(np.float32, copy=False)
        terminated_np = self._numpy(terminated).astype(bool, copy=False)
        truncated_np = self._numpy(truncated).astype(bool, copy=False)
        done = terminated_np | truncated_np
        self._episode_returns += reward_np
        self._episode_lengths += 1
        infos: list[dict[str, Any]] = []
        for index in range(self.num_envs):
            info = {
                "terminated": bool(terminated_np[index]),
                "truncated": bool(truncated_np[index]),
                "TimeLimit.truncated": bool(truncated_np[index] and not terminated_np[index]),
            }
            if done[index]:
                terminal = self.env.terminal_observation_by_env.get(index)
                if terminal is None:
                    raise RuntimeError(
                        f"Isaac auto-reset lost terminal observation for environment {index}")
                info["terminal_observation"] = self._numpy(terminal).astype(np.float32)
                info["episode"] = {
                    "r": float(self._episode_returns[index]),
                    "l": int(self._episode_lengths[index]),
                }
                info.update(self.env.terminal_stats_by_env[index])
                self._episode_returns[index] = 0.0
                self._episode_lengths[index] = 0
            infos.append(info)
        return obs_np, reward_np, done, infos

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices=None):
        target = getattr(self.env, attr_name)
        return [target for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        method = getattr(self.env, method_name)
        result = method(*method_args, **method_kwargs)
        return [result for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def seed(self, seed: int | None = None):
        seeds = super().seed(seed)
        return seeds
