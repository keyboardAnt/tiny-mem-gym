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

    metadata = {"render_modes": ["rgb_array"]}

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

    # Rendering -----------------------------------------------------------------

    def render(self, mode: str = "rgb_array", **kwargs: Any) -> np.ndarray:
        """Render a tiny visualization of the current observation.

        The visualization is intentionally simple and derived only from the
        current observation vector, so it does not reveal any hidden state.
        """
        del kwargs  # unused
        if mode != "rgb_array":
            raise ValueError("Only 'rgb_array' render mode is supported.")

        obs = self._get_obs()
        cue = obs[: self.n_arms]
        last_reward = float(obs[self.n_arms])
        progress = float(obs[self.n_arms + 1])

        height = 64
        width_per_arm = 32
        width = self.n_arms * width_per_arm
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw arms as vertical bars whose brightness encodes the cue.
        for i in range(self.n_arms):
            x0 = i * width_per_arm
            x1 = x0 + width_per_arm
            intensity = int(255 * float(cue[i]))
            frame[:, x0:x1, 1] = intensity  # green channel

        # Draw a horizontal progress bar at the bottom.
        prog_width = int(width * max(0.0, min(1.0, progress)))
        if prog_width > 0:
            frame[-5:, :prog_width, :] = np.array([0, 0, 255], dtype=np.uint8)

        # Flash red at the top when the last reward was positive.
        if last_reward > 0.0:
            frame[:5, :, :] = np.array([255, 0, 0], dtype=np.uint8)

        return frame

    def _get_obs(self) -> np.ndarray:
        """Construct the current observation vector."""
        cue = np.zeros(self.n_arms, dtype=np.float32)
        if self._timestep < self.cue_duration and self._rewarding_arm is not None:
            cue[self._rewarding_arm] = 1.0

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[: self.n_arms] = cue
        obs[self.n_arms] = np.float32(self._last_reward)
        if self.max_episode_steps <= 0:
            progress = 0.0
        else:
            # Clamp progress to [0, 1] so observations always stay within the
            # declared Box bounds.
            clamped_t = min(self._timestep, self.max_episode_steps)
            progress = clamped_t / float(self.max_episode_steps)
        obs[self.n_arms + 1] = np.float32(progress)
        return obs


