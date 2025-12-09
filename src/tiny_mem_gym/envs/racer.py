"""Memory Racer Environment.

Lane-based obstacle avoidance with a twist.
Obstacles are visible at the horizon but disappear as they get closer (Memory Zone).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

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
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Actions: 0=Left, 1=Stay, 2=Right
        self.action_space = spaces.Discrete(3)
        
        # Observation:
        # Simplified grid: 20 rows x n_lanes
        self.grid_height = 20
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.grid_height, self.n_lanes),
            dtype=np.float32
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
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            if "n_lanes" in options:
                self.n_lanes = options["n_lanes"]
                self.observation_space = spaces.Box(
                    low=0, high=1,
                    shape=(self.grid_height, self.n_lanes),
                    dtype=np.float32
                )
            if "obs_depth" in options:
                self.obs_depth = options["obs_depth"]
            if "speed" in options:
                self.speed = options["speed"]
        
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
        
        if self.timer >= self.max_steps:
            truncated = True
            self.game_over = True
        
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
                    
                    # Entering memory zone? Capture ghost mark for the lane
                    visible_threshold = self.obs_depth // 2
                    if dist > visible_threshold >= new_dist:
                        new_memory_marks.append((lane, new_dist))
            
            self.obstacles = next_obstacles
            self.memory_marks = new_memory_marks
            
            # Spawn new obstacles
            if self.np_random.random() < 0.1:
                lane = self.np_random.integers(0, self.n_lanes)
                dist = self.obs_depth
                # Check collision at spawn
                free = True
                for l, d in self.obstacles:
                    if l == lane and d > self.obs_depth - 10:
                        free = False
                        break
                if free:
                    self.obstacles.append([lane, dist])
            
            if not terminated:
                reward = 0.1
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, {"success": self.score > 500}

    def _get_obs(self):
        obs = np.zeros((self.grid_height, self.n_lanes), dtype=np.float32)
        bin_size = self.obs_depth / self.grid_height
        visible_threshold = self.obs_depth // 2
        
        for lane, dist in self.obstacles:
            if dist > visible_threshold:
                row = int(dist / bin_size)
                if 0 <= row < self.grid_height:
                    obs[row, lane] = 1.0
        return obs

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def render(self):
        if self.render_mode is None: return
        
        game_width = 400
        game_height = 600
        sidebar_width = 250
        total_width = game_width + sidebar_width
        
        if not self.renderer:
            self.renderer = RetroRenderer(total_width, game_height, "Memory Racer")
            
        if self.render_mode == "human" and not self.window:
            pygame.init()
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
            overlay = pygame.Surface((game_width, memory_overlay_h), pygame.SRCALPHA)
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
        if not self.game_over:
            time_left = max(0, self.max_steps - self.timer)
            self.renderer.draw_text(f"FOCUS {time_left}", game_width - 120, 10, COLOR_NEON_AMBER, size=18)
        
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
