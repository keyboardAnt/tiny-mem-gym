"""Tiny partially observable gridworld environment.

The agent moves on a small grid, but only observes a local window around its
current position. The goal is to reach a target cell.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TinyPOGridworldEnv(gym.Env):
    """Tiny partially observable gridworld.

    Observation is a flattened local window around the agent, with integer
    codes for empty, wall, agent, and goal cells.
    """

    metadata = {"render_modes": ["rgb_array"]}

    EMPTY = 0
    WALL = 1
    GOAL = 2
    AGENT = 3

    def __init__(
        self,
        grid_size: int = 5,
        window_size: int = 3,
        max_episode_steps: int = 50,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        invalid_penalty: float = -0.05,
    ) -> None:
        super().__init__()

        if grid_size < 3:
            raise ValueError("grid_size must be at least 3.")
        if window_size % 2 == 0 or window_size < 3:
            raise ValueError("window_size must be odd and at least 3.")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")

        self.grid_size = grid_size
        self.window_size = window_size
        self.max_episode_steps = max_episode_steps
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.invalid_penalty = invalid_penalty

        # Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        self.action_space = spaces.Discrete(5)

        # Observation: flattened window of integer codes normalized to [0, 1].
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(window_size * window_size,),
            dtype=np.float32,
        )

        self._agent_pos: Tuple[int, int] | None = None
        self._goal_pos: Tuple[int, int] | None = None
        self._timestep: int = 0
        self._cumulative_reward: float = 0.0

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        del options  # unused

        # Simple layout: empty interior, no walls; random start and goal.
        self._agent_pos = (
            int(self.np_random.integers(0, self.grid_size)),
            int(self.np_random.integers(0, self.grid_size)),
        )
        self._goal_pos = self._agent_pos
        while self._goal_pos == self._agent_pos:
            self._goal_pos = (
                int(self.np_random.integers(0, self.grid_size)),
                int(self.np_random.integers(0, self.grid_size)),
            )

        self._timestep = 0
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
        if self._agent_pos is None or self._goal_pos is None:
            raise RuntimeError("Environment must be reset before stepping.")

        terminated = False
        reward = self.step_penalty

        ax, ay = self._agent_pos
        dx, dy = 0, 0
        if action == 0:  # up
            dx, dy = -1, 0
        elif action == 1:  # right
            dx, dy = 0, 1
        elif action == 2:  # down
            dx, dy = 1, 0
        elif action == 3:  # left
            dx, dy = 0, -1
        elif action == 4:  # stay
            dx, dy = 0, 0

        nx, ny = ax + dx, ay + dy
        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            self._agent_pos = (nx, ny)
        else:
            # Bumping into the border incurs an additional penalty.
            reward += self.invalid_penalty

        if self._agent_pos == self._goal_pos:
            reward += self.goal_reward
            terminated = True

        self._timestep += 1
        self._cumulative_reward += reward
        truncated = self._timestep >= self.max_episode_steps

        obs = self._get_obs()
        info = {
            "timestep": self._timestep,
            "cumulative_reward": self._cumulative_reward,
            "success": bool(terminated),
        }
        return obs, float(reward), terminated, truncated, info

    # Rendering -----------------------------------------------------------------

    def render(self, mode: str = "rgb_array", **kwargs: Any) -> np.ndarray:
        """Render the grid as a small RGB image using only NumPy."""
        del kwargs  # unused
        if mode != "rgb_array":
            raise ValueError("Only 'rgb_array' render mode is supported.")

        grid = np.full((self.grid_size, self.grid_size), self.EMPTY, dtype=np.int32)
        if self._goal_pos is not None:
            gx, gy = self._goal_pos
            grid[gx, gy] = self.GOAL
        if self._agent_pos is not None:
            ax, ay = self._agent_pos
            grid[ax, ay] = self.AGENT

        # Map codes to colors.
        colors = {
            self.EMPTY: np.array([0, 0, 0], dtype=np.uint8),        # black
            self.WALL: np.array([128, 128, 128], dtype=np.uint8),  # gray (unused)
            self.GOAL: np.array([0, 255, 0], dtype=np.uint8),      # green
            self.AGENT: np.array([255, 0, 0], dtype=np.uint8),     # red
        }

        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for code, color in colors.items():
            img[grid == code] = color
        return img

    # Helpers -------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return the flattened local window around the agent."""
        if self._agent_pos is None or self._goal_pos is None:
            # Safe default.
            window = np.zeros((self.window_size, self.window_size), dtype=np.int32)
        else:
            ax, ay = self._agent_pos
            half = self.window_size // 2
            window = np.full(
                (self.window_size, self.window_size), self.EMPTY, dtype=np.int32
            )

            for i in range(-half, half + 1):
                for j in range(-half, half + 1):
                    gx, gy = ax + i, ay + j
                    wx, wy = i + half, j + half
                    if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                        code = self.EMPTY
                        if (gx, gy) == self._goal_pos:
                            code = self.GOAL
                        if (gx, gy) == self._agent_pos:
                            code = self.AGENT
                        window[wx, wy] = code

        # Normalize to [0, 1] based on max code.
        max_code = float(self.AGENT)
        obs = window.astype(np.float32).reshape(-1) / max_code
        return obs


