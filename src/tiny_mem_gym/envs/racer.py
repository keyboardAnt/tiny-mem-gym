"""Memory Racer Environment.

Lane-based obstacle avoidance with a twist.
Obstacles are visible at the horizon but disappear as they get closer (Memory Zone).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

# pygame lacks complete stubs; silence missing-member complaints for init/quit/SRCALPHA
# pylint: disable=no-member

from tiny_mem_gym.drawing import (
    RetroRenderer,
    COLOR_NEON_PINK,
    COLOR_NEON_CYAN,
    COLOR_NEON_AMBER,
    COLOR_WHITE,
    COLOR_RED,
)

class MemoryRacerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        n_lanes: int = 3,
        obs_depth: int = 100, # Depth in arbitrary units (distance)
        speed: float = 1.0,   # Speed of obstacles
        max_steps: int = 1000,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        
        self.n_lanes = n_lanes
        self.obs_depth = obs_depth
        self.speed = speed
        self.spawn_prob = 0.1
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Actions: 0=Left, 1=Stay, 2=Right
        self.action_space = spaces.Discrete(3)
        
        # Observation:
        # Two-channel grid: warnings (far) + memory/hidden zone markers
        self.grid_height = 20
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_height, self.n_lanes, 2),
            dtype=np.float32,
        )
        
        self.renderer = None
        self.window = None
        self.clock = None
        
        self.car_lane = 1 
        self.obstacles = [] # List of [lane, distance]
        self.memory_marks: list[tuple[int, float]] = [] # Obstacles remembered in hidden zone
        self.score = 0
        self.timer = 0
        self.game_over = False
        self._gap_distance = 15  # ensure at least one lane is free within this range
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            if "n_lanes" in options and options["n_lanes"] != self.n_lanes:
                raise ValueError(
                    "n_lanes is fixed after initialization; create a new env for a different lane count."
                )
            if "obs_depth" in options:
                self.obs_depth = options["obs_depth"]
            if "speed" in options:
                self.speed = options["speed"]
            if "spawn_prob" in options:
                self.spawn_prob = options["spawn_prob"]
        
        self.car_lane = self.n_lanes // 2
        self.obstacles = []
        self.memory_marks = []
        self.score = 0
        self.timer = 0
        self.game_over = False
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        outcome = "running"
        
        if self.timer >= self.max_steps:
            truncated = True
            self.game_over = True
            outcome = "timeout"
        
        if not (terminated or truncated):
            self.timer += 1
            self.score += 1 # Survival reward
            new_memory_marks = []
            for lane, mark_dist in self.memory_marks:
                md = mark_dist - self.speed
                if md > 0:
                    new_memory_marks.append((lane, md))
            
            # Move Car
            if action == 0: # Left
                self.car_lane = max(0, self.car_lane - 1)
            elif action == 2: # Right
                self.car_lane = min(self.n_lanes - 1, self.car_lane + 1)
            
            # Move Obstacles
            next_obstacles = []
            for lane, dist in self.obstacles:
                new_dist = dist - self.speed
                if new_dist > 0:
                    next_obstacles.append([lane, new_dist])
                    
                    # Collision Check
                    # Car is effectively at bottom (dist approx 0)
                    if lane == self.car_lane and new_dist < 5:
                        reward = -10.0
                        terminated = True
                        self.game_over = True
                        outcome = "lose"
                    
                    # Entering memory zone? Capture ghost mark for the lane
                    visible_threshold = self.obs_depth // 2
                    if dist > visible_threshold >= new_dist:
                        new_memory_marks.append((lane, new_dist))
            
            self.obstacles = next_obstacles
            self.memory_marks = new_memory_marks
            
            # Spawn new obstacles
            if self.np_random.random() < self.spawn_prob:
                lane = self.np_random.integers(0, self.n_lanes)
                dist = self.obs_depth
                # Check collision at spawn
                free = True
                for lane_existing, dist_existing in self.obstacles:
                    if lane_existing == lane and dist_existing > self.obs_depth - 10:
                        free = False
                        break
                if free:
                    self.obstacles.append([lane, dist])
            
            # Ensure solvable: always leave at least one lane free within the near window
            if not terminated:
                self._ensure_escape_gap()
            
            if not terminated:
                reward = 0.1
                outcome = "running"
        
        if self.render_mode == "human":
            self.render()
            
        info = {
            "success": outcome == "timeout" and not terminated,
            "distance": self.timer,
            "collided": terminated,
            "outcome": outcome,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs = np.zeros((self.grid_height, self.n_lanes, 2), dtype=np.float32)
        bin_size = self.obs_depth / self.grid_height
        visible_threshold = self.obs_depth // 2
        
        # Channel 0: far warnings (visible zone)
        # Channel 1: remembered / hidden zone markers (including near obstacles)
        for lane, dist in self.obstacles:
            row = int(dist / bin_size)
            row = max(0, min(self.grid_height - 1, row))
            if dist > visible_threshold:
                obs[row, lane, 0] = 1.0
            else:
                obs[row, lane, 1] = 1.0
        
        for lane, dist in self.memory_marks:
            row = int(dist / bin_size)
            row = max(0, min(self.grid_height - 1, row))
            obs[row, lane, 1] = max(obs[row, lane, 1], 0.6)  # ghost hint
        
        return obs

    def _ensure_escape_gap(self):
        """Prevent unsolvable states: keep at least one lane free in the near zone."""
        near_indices = [
            (idx, lane, dist)
            for idx, (lane, dist) in enumerate(self.obstacles)
            if dist < self._gap_distance
        ]
        if len({lane for _, lane, _ in near_indices}) == self.n_lanes:
            # Remove the farthest obstacle among the near ones to open a lane
            remove_idx, remove_lane, _ = max(near_indices, key=lambda x: x[2])
            self.obstacles.pop(remove_idx)
            # Drop matching memory marks in the near zone for that lane
            self.memory_marks = [
                (lane, dist) for (lane, dist) in self.memory_marks
                if not (lane == remove_lane and dist < self._gap_distance)
            ]

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()  # type: ignore[attr-defined]
            self.window = None

    def render(self):
        if self.render_mode is None:
            return
        
        game_width = 400
        game_height = 600
        sidebar_width = 250
        total_width = game_width + sidebar_width
        
        if not self.renderer:
            self.renderer = RetroRenderer(total_width, game_height, "Memory Racer")
            
        if self.render_mode == "human" and not self.window:
            pygame.init()  # type: ignore[attr-defined]
            pygame.display.init()
            self.window = pygame.display.set_mode((total_width, game_height))
            pygame.display.set_caption("Memory Racer")
            self.clock = pygame.time.Clock()
            
        self.renderer.clear()
        
        # Draw Lanes
        lane_width = game_width // self.n_lanes
        for i in range(1, self.n_lanes):
            x = i * lane_width
            pygame.draw.line(self.renderer.surface, (50, 50, 50), (x, 0), (x, game_height), 2)
            
        # Draw Car
        car_x = self.car_lane * lane_width + lane_width // 2
        car_y = game_height - 50
        # Triangle Car
        pts = [(car_x, car_y - 20), (car_x - 15, car_y + 20), (car_x + 15, car_y + 20)]
        pygame.draw.polygon(self.renderer.surface, COLOR_NEON_PINK, pts)
        
        # Draw Obstacles / Signs
        scale_y = game_height / self.obs_depth
        
        for lane, dist in self.obstacles:
            # Render logic for human:
            # Show "WARNING" sign if far.
            # Show NOTHING if close (Memory Zone).
            # UNLESS Game Over! Then show everything!
            
            is_visible = (dist > self.obs_depth // 2)
            
            # Cheat mode if terminated (show why you died)
            if hasattr(self, "game_over") and self.game_over:
                is_visible = True
            
            y = game_height - (dist * scale_y)
            x = lane * lane_width + lane_width // 2
            
            if is_visible:
                # Draw Sign or Obstacle?
                # If far -> Sign
                # If close -> Obstacle Block
                is_far = (dist > self.obs_depth // 2)
                if is_far:
                    rect = (x - 20, y - 15, 40, 30)
                    self.renderer.draw_box(rect, COLOR_NEON_AMBER, filled=True)
                    self.renderer.draw_text("!", x, y, COLOR_RED, center=True)
                else:
                    # Actual Obstacle
                    rect = (x - 20, y - 20, 40, 40)
                    self.renderer.draw_box(rect, COLOR_RED, filled=True)
                    # Skull?
                    self.renderer.draw_text("X", x, y, COLOR_WHITE, center=True)
            else:
                # In memory zone! 
                rect = (x - 10, y - 10, 20, 20)
                pygame.draw.rect(self.renderer.surface, (30, 0, 0), rect, 1)
        
        # Draw ghost markers for remembered obstacles in the hidden zone
        for lane, dist in self.memory_marks:
            y = game_height - (dist * scale_y)
            x = lane * lane_width + lane_width // 2
            rect = (x - 12, y - 12, 24, 24)
            self.renderer.draw_box(rect, COLOR_NEON_CYAN, filled=False, width=1)
            self.renderer.draw_text("?", x, y, COLOR_NEON_CYAN, center=True, size=14)

        # Draw "Memory Horizon" line and tint masked area so its boundary is obvious
        horizon_y = game_height - ((self.obs_depth // 2) * scale_y)
        memory_overlay_h = game_height - horizon_y
        if memory_overlay_h > 0:
            # Subtle translucent overlay for the hidden/remembered zone
            overlay = pygame.Surface((game_width, memory_overlay_h), pygame.SRCALPHA)  # type: ignore[attr-defined]
            overlay.fill((20, 0, 0, 60))
            self.renderer.surface.blit(overlay, (0, horizon_y))
        pygame.draw.line(self.renderer.surface, COLOR_NEON_CYAN, (0, horizon_y), (game_width, horizon_y), 1)
        self.renderer.draw_text("MEMORY ZONE", game_width//2, horizon_y + 10, COLOR_NEON_CYAN, center=True, size=14)

        # Sidebar
        instructions = [
            "1. Warnings (!) appear",
            "   at the top",
            "2. Obstacles vanish",
            "   in Memory Zone",
            "3. Remember where",
            "   they are!",
            "4. Dodge them"
        ]
        controls = [
            ("LEFT", "Change Lane"),
            ("RIGHT", "Change Lane")
        ]
        
        level_est = (self.n_lanes - 3) * 3 + 1
        self.renderer.draw_sidebar(game_width, sidebar_width, instructions, controls, level=level_est)
        
        # Status HUD
        self.renderer.draw_text(f"SCORE {self.score}", 10, 10, COLOR_NEON_PINK, size=18)
        
        # Game Over overlay
        if hasattr(self, "game_over") and self.game_over:
            msg = "CRASHED!" if self.score < 0 else "TIME UP"
            self.renderer.draw_text(msg, game_width//2, game_height//2, COLOR_RED, center=True, size=60)

        self.renderer.draw_scanlines()
        
        if self.render_mode == "human":
            self.window.blit(self.renderer.surface, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(30)
            
        return self.renderer.get_frame()
