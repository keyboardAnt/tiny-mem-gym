"""Sequence recall / N-back environments.

These environments present a sequence of discrete symbols and then ask the
agent to recall or perform an N-back style comparison during a query phase.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SequenceRecallEnv(gym.Env):
    """Tiny sequence recall / N-back environment.

    The episode consists of:

    - A presentation phase where symbols from a small vocabulary are shown.
    - A query phase where the agent must either:
      - Predict the symbol from ``n_back`` steps ago, or
      - Answer yes/no whether the current symbol matches the one ``n_back``
        steps ago.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        n_symbols: int = 4,
        sequence_length: int = 6,
        n_back: int = 1,
        query_length: int = 2,
        max_episode_steps: int | None = None,
        query_mode: str = "predict",  # "predict" or "yesno"
    ) -> None:
        super().__init__()

        if n_symbols < 2:
            raise ValueError("n_symbols must be at least 2.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if n_back <= 0:
            raise ValueError("n_back must be positive.")
        if query_length <= 0:
            raise ValueError("query_length must be positive.")
        if query_mode not in ("predict", "yesno"):
            raise ValueError("query_mode must be 'predict' or 'yesno'.")

        self.n_symbols = n_symbols
        self.sequence_length = sequence_length
        self.n_back = n_back
        self.query_length = query_length
        self.query_mode = query_mode
        self.max_episode_steps = max_episode_steps or (sequence_length + query_length)

        # Observation: one-hot symbol plus phase bit (0 = presentation, 1 = query)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_symbols + 1,),
            dtype=np.float32,
        )

        if query_mode == "predict":
            self.action_space = spaces.Discrete(n_symbols)
        else:  # yes/no
            self.action_space = spaces.Discrete(2)

        self._sequence: np.ndarray | None = None
        self._timestep: int = 0
        self._cumulative_reward: float = 0.0
        self._last_reward: float = 0.0

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        del options  # unused

        self._sequence = self.np_random.integers(
            low=0, high=self.n_symbols, size=self.sequence_length, dtype=int
        )
        self._timestep = 0
        self._cumulative_reward = 0.0
        self._last_reward = 0.0

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
        if self._sequence is None:
            raise RuntimeError("Environment must be reset before stepping.")

        terminated = False

        # Determine whether we're in presentation or query phase.
        in_presentation = self._timestep < self.sequence_length
        reward = 0.0

        if in_presentation:
            # Ignore action; this phase is purely for observation.
            reward = 0.0
        else:
            # Query phase.
            idx = min(self._timestep - self.sequence_length, self.query_length - 1)
            # Map query index to source timestep in the sequence.
            query_t = self.sequence_length - self.query_length + idx
            target_symbol = int(self._sequence[query_t])

            if self.query_mode == "predict":
                reward = 1.0 if int(action) == target_symbol else 0.0
            else:  # yes/no mode
                current_symbol = int(self._sequence[query_t])
                match_symbol = int(self._sequence[max(0, query_t - self.n_back)])
                is_match = current_symbol == match_symbol
                predicted_match = bool(int(action) == 1)
                reward = 1.0 if predicted_match == is_match else 0.0

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
        """Render the current symbol and phase as a tiny RGB image.

        The rendering is derived from the current observation only.
        """
        del kwargs  # unused
        if mode != "rgb_array":
            raise ValueError("Only 'rgb_array' render mode is supported.")

        obs = self._get_obs()
        symbol_probs = obs[:-1]
        phase_flag = float(obs[-1])
        symbol_idx = int(np.argmax(symbol_probs))

        height = 64
        width_per_symbol = 32
        width = self.n_symbols * width_per_symbol
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Background color encodes phase: blue for presentation, yellow for query.
        if phase_flag < 0.5:
            frame[:, :, :] = np.array([0, 0, 60], dtype=np.uint8)
        else:
            frame[:, :, :] = np.array([80, 80, 0], dtype=np.uint8)

        # Color palette for symbols (cycled).
        palette = [
            np.array([255, 0, 0], dtype=np.uint8),
            np.array([0, 255, 0], dtype=np.uint8),
            np.array([0, 0, 255], dtype=np.uint8),
            np.array([255, 255, 0], dtype=np.uint8),
        ]

        for i in range(self.n_symbols):
            x0 = i * width_per_symbol
            x1 = x0 + width_per_symbol
            color = palette[i % len(palette)]
            # Draw all symbols as faint bars so humans know how many options
            # there are, and highlight the current one strongly.
            frame[12:-12, x0 + 6 : x1 - 6, :] = (color * 0.2).astype(np.uint8)
            if i == symbol_idx:
                frame[8:-8, x0 + 4 : x1 - 4, :] = color

        # Add a feedback strip at the bottom: bright green when last answer was
        # correct, dark red otherwise.
        if self._last_reward > 0.0:
            frame[-6:, :, :] = np.array([0, 200, 0], dtype=np.uint8)
        else:
            frame[-6:, :, :] = np.array([80, 0, 0], dtype=np.uint8)

        # Simple progress indicator at the top based on timestep.
        progress = min(1.0, self._timestep / float(self.max_episode_steps or 1))
        prog_width = int(width * progress)
        if prog_width > 0:
            frame[:4, :prog_width, :] = np.array([255, 255, 255], dtype=np.uint8)

        return frame

    def _get_obs(self) -> np.ndarray:
        """Return the current observation (symbol one-hot + phase bit)."""
        if self._sequence is None:
            # This should not occur during normal use, but keep a safe default.
            symbol = 0
            in_presentation = True
        else:
            in_presentation = self._timestep < self.sequence_length
            if in_presentation:
                symbol = int(self._sequence[self._timestep])
            else:
                # During query phase, we reuse the last symbol from presentation
                symbol = int(self._sequence[-1])

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[symbol] = 1.0
        obs[-1] = 1.0 if not in_presentation else 0.0
        return obs


