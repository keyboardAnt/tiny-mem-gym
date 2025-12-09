"""Cyber Hacking Environment.

Sequence Recall Memory Game.
Watch the pattern on the grid, then repeat it back.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from tiny_mem_gym.drawing import (
    RetroRenderer,
    COLOR_GRID,
    COLOR_NEON_GREEN,
    COLOR_NEON_PINK,
    COLOR_NEON_CYAN,
    COLOR_NEON_AMBER,
    COLOR_WHITE,
)

class CyberHackingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        grid_size: int = 3,
        sequence_length: int = 4,
        display_time: int = 60,  # Steps to show sequence
        max_steps: int = 200,    # Input steps allowed (excludes watch phase)
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        
        self.grid_size = grid_size
        self.sequence_length = sequence_length
        self.display_time = display_time
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        self.n_nodes = grid_size * grid_size
        self.action_space = spaces.Discrete(self.n_nodes)
        
        # Observation:
        # Grid channels:
        # 0: Highlighted Node (Sequence or Input)
        # 1: Completed Nodes
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size, grid_size, 2),
            dtype=np.float32,
        )
        
        self.renderer = None
        self.window = None
        self.clock = None
        
        self.sequence = []
        self.current_step = 0  # In sequence
        self.watch_timer = 0
        self.input_timer = 0
        self.phase = "watch"  # "watch", "input"
        self.game_over = False
        self.message = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            if "grid_size" in options and options["grid_size"] != self.grid_size:
                raise ValueError(
                    "grid_size is fixed after initialization; create a new env for a different size."
                )
            if "sequence_length" in options:
                self.sequence_length = options["sequence_length"]
            if "display_time" in options:
                self.display_time = options["display_time"]
            if "max_steps" in options:
                self.max_steps = options["max_steps"]

        # Generate Sequence
        # Allow repeats? Yes, why not.
        self.sequence = [
            self.np_random.integers(0, self.n_nodes) 
            for _ in range(self.sequence_length)
        ]
        
        self.current_step = 0
        self.watch_timer = 0
        self.input_timer = 0
        self.phase = "watch"
        self.game_over = False
        self.message = ""
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        
        if self.phase == "watch":
            self.watch_timer += 1
            if self.watch_timer >= self.display_time:
                self.phase = "input"
            # Ignore actions during watch; no truncation here
        else:
            self.input_timer += 1
            if self.input_timer >= self.max_steps:
                truncated = True
                self.game_over = True
                self.message = "TIMEOUT"
            
            if not truncated:
                expected = self.sequence[self.current_step]
                if action == expected:
                    reward = 1.0
                    self.current_step += 1
                    if self.current_step >= len(self.sequence):
                        terminated = True
                        reward = 10.0  # Bonus
                        self.game_over = True
                        self.message = "HACKED!"
                else:
                    reward = -1.0
                    terminated = True
                    self.game_over = True
                    self.message = "DENIED"

        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, {"success": reward > 5.0}

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
        
        # Channel 0: Active Node
        active_node = -1
        if self.phase == "watch":
            # Show sequence based on time
            # Divide display_time into slots
            steps_per_node = self.display_time // len(self.sequence)
            if steps_per_node < 1:
                steps_per_node = 1
            idx = self.watch_timer // steps_per_node
            
            if 0 <= idx < len(self.sequence):
                active_node = self.sequence[idx]
        
        if active_node != -1:
            r, c = divmod(active_node, self.grid_size)
            obs[r, c, 0] = 1.0
            
        # Channel 1: Completed
        for i in range(self.current_step):
            node = self.sequence[i]
            r, c = divmod(node, self.grid_size)
            obs[r, c, 1] = 1.0
            
        return obs

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def render(self):
        if self.render_mode is None: return
        
        game_size = 512
        sidebar_width = 250
        total_width = game_size + sidebar_width
        
        if not self.renderer:
            self.renderer = RetroRenderer(total_width, game_size, "Cyber Breach")
            
        if self.render_mode == "human" and not self.window:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((total_width, game_size))
            pygame.display.set_caption("Cyber Breach")
            self.clock = pygame.time.Clock()
            
        self.renderer.clear()
        
        # Layout Hex Grid or Square Grid? Square for now.
        margin = 100
        available = game_size - 2 * margin
        cell_size = available // (self.grid_size - 1) if self.grid_size > 1 else available
        
        node_positions = {}
        
        # Draw connections
        # Just simple grid lines
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = margin + c * cell_size
                y = margin + r * cell_size
                node_positions[r * self.grid_size + c] = (x, y)
                
                # Draw lines to neighbors
                if c < self.grid_size - 1:
                    nx = margin + (c+1) * cell_size
                    pygame.draw.line(self.renderer.surface, (40, 40, 60), (x, y), (nx, y), 2)
                if r < self.grid_size - 1:
                    ny = margin + (r+1) * cell_size
                    pygame.draw.line(self.renderer.surface, (40, 40, 60), (x, y), (x, ny), 2)

        # Draw Nodes
        for idx, pos in node_positions.items():
            color = COLOR_NEON_CYAN
            radius = 15
            
            # Logic for highlighting
            is_active = False
            
            if self.phase == "watch":
                steps_per_node = self.display_time // len(self.sequence)
                if steps_per_node < 1:
                    steps_per_node = 1
                seq_idx = self.watch_timer // steps_per_node
                
                if seq_idx < len(self.sequence) and self.sequence[seq_idx] == idx:
                    is_active = True
                    color = COLOR_NEON_GREEN
                    radius = 25
            
            elif self.phase == "input":
                # Show already completed nodes?
                if idx in self.sequence[:self.current_step]:
                    color = COLOR_NEON_GREEN
                    # filled
            
            if is_active:
                pygame.draw.circle(self.renderer.surface, color, pos, radius)
                # Glow
                pygame.draw.circle(self.renderer.surface, color, pos, radius + 5, 2)
            else:
                pygame.draw.circle(self.renderer.surface, color, pos, radius, 2)
                pygame.draw.circle(self.renderer.surface, (0, 0, 0), pos, radius - 2)
                
            # Draw number hint
            # Map 0..8 to 1..9
            label = str(idx + 1)
            self.renderer.draw_text(label, pos[0], pos[1], COLOR_WHITE, center=True, size=16)

        # Draw "Connection" line for completed part
        if self.phase == "input" and self.current_step > 0:
            for i in range(self.current_step - 1):
                u = self.sequence[i]
                v = self.sequence[i+1]
                pygame.draw.line(self.renderer.surface, COLOR_NEON_GREEN, node_positions[u], node_positions[v], 4)
            # Line to current target? No, that's a hint.

        # UI
        status = "BREACHING..." if self.phase == "watch" else "EXECUTE"
        color = COLOR_NEON_PINK if self.phase == "watch" else COLOR_NEON_AMBER
        self.renderer.draw_text(status, game_size//2, 50, color, center=True, size=30)
        # Beat/progress bar for rhythm and clarity
        if self.phase == "watch":
            ratio = min(1.0, self.watch_timer / max(1, self.display_time))
        else:
            ratio = 0.0
        bar_w = int(game_size * 0.6)
        bar_h = 8
        bar_x = game_size//2 - bar_w//2
        bar_y = 70
        pygame.draw.rect(self.renderer.surface, COLOR_GRID, (bar_x, bar_y, bar_w, bar_h), 1)
        if ratio > 0:
            pygame.draw.rect(self.renderer.surface, COLOR_NEON_PINK, (bar_x, bar_y, int(bar_w * ratio), bar_h))
        # Step counter for input phase
        if self.phase == "input":
            hint = f"{self.current_step}/{len(self.sequence)}"
            self.renderer.draw_text(hint, game_size//2, 90, COLOR_NEON_GREEN, center=True, size=18)
        
        # Sidebar
        instructions = [
            "1. Watch sequence",
            "   (Green Nodes)",
            "2. Wait for EXECUTE",
            "3. Repeat Pattern",
            "   Exactly!",
            "4. Don't Miss"
        ]
        controls = [
            ("1-9", "Select Node"),
            ("QWE/ASD/ZXC", "Grid Input")
        ]
        
        # Infer level from sequence length?
        # Base length is 4. Level = len - 3?
        level = self.sequence_length - 3
        self.renderer.draw_sidebar(game_size, sidebar_width, instructions, controls, level=level)
        
        self.renderer.draw_scanlines()
        
        if self.render_mode == "human":
            self.window.blit(self.renderer.surface, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(30)
            
        return self.renderer.get_frame()
