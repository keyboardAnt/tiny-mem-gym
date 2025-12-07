"""Tiny memory bandit environments.

The core idea is that an initial cue reveals the rewarding arm, and the agent
must remember this information over several timesteps to obtain reward.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TinyMemoryBanditEnv(gym.Env):
    """A tiny bandit environment with an initial cue.

    Observations are low-dimensional vectors containing:

    - a one-hot cue over arms during the cue phase (all zeros afterwards)
    - the last reward
    - a normalized timestep
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_arms: int = 2,
        max_episode_steps: int = 10,
        cue_duration: int = 1,
    ) -> None:
        super().__init__()

        if n_arms < 2:
            raise ValueError("n_arms must be at least 2.")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")
        if cue_duration <= 0:
            raise ValueError("cue_duration must be positive.")

        self.n_arms = n_arms
        self.max_episode_steps = max_episode_steps
        self.cue_duration = cue_duration

        obs_dim = n_arms + 2  # cue one-hot, last_reward, normalized timestep
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_arms)

        self._rewarding_arm: int | None = None
        self._timestep: int = 0
        self._last_reward: float = 0.0
        self._cumulative_reward: float = 0.0

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        del options  # unused

        self._rewarding_arm = int(self.np_random.integers(self.n_arms))
        self._timestep = 0
        self._last_reward = 0.0
        self._cumulative_reward = 0.0

        obs = self._get_obs()
        info = {
            "timestep": self._timestep,
            "cumulative_reward": self._cumulative_reward,
            "success": False,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._rewarding_arm is None:
            raise RuntimeError("Environment must be reset before stepping.")

        terminated = False

        reward = 1.0 if int(action) == int(self._rewarding_arm) else 0.0
        self._last_reward = reward
        self._cumulative_reward += reward
        self._timestep += 1

        truncated = self._timestep >= self.max_episode_steps

        obs = self._get_obs()
        info = {
            "timestep": self._timestep,
            "cumulative_reward": self._cumulative_reward,
            "success": bool(self._cumulative_reward > 0.0),
        }
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct the current observation vector."""
        cue = np.zeros(self.n_arms, dtype=np.float32)
        if self._timestep < self.cue_duration and self._rewarding_arm is not None:
            cue[self._rewarding_arm] = 1.0

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[: self.n_arms] = cue
        obs[self.n_arms] = np.float32(self._last_reward)
        obs[self.n_arms + 1] = np.float32(
            0.0 if self.max_episode_steps <= 1 else self._timestep / (self.max_episode_steps - 1)
        )
        return obs


