"""Dungeon Escape Environment.

Spatial Memory Challenge.
The player is shown a dungeon map for a brief period.
Then the lights go out (Fog of War).
The player must navigate to the exit without hitting walls or traps.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from tiny_mem_gym.drawing import (
    RetroRenderer,
    COLOR_NEON_GREEN,
    COLOR_NEON_PINK,
    COLOR_NEON_CYAN,
    COLOR_NEON_AMBER,
    COLOR_WHITE,
    COLOR_RED,
    COLOR_GRID,
)

class DungeonEscapeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}
    
    # Tile types
    EMPTY = 0
    WALL = 1
    START = 2
    EXIT = 3
    TRAP = 4
    
    def __init__(
        self,
        grid_size: int = 7,
        memorization_time: int = 30, # Steps/Frames
        max_steps: int = 100,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        
        self.grid_size = grid_size
        self.memorization_time = memorization_time
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(5) # Up, Right, Down, Left, Wait
        
        # Observation:
        # We can't give the full map during "blind" phase if we want it to be memory based.
        # But for an RL agent, "blind" usually means masking the map in the observation.
        # Let's provide: 
        # 1. Current local view (3x3) - only immediate walls visible? Or nothing?
        #    If it's pure memory, maybe "touch" sensation? (Bumped wall).
        #    Let's go with: Fully observable map during Phase 1.
        #    During Phase 2: Masked map (all 0s) except agent pos and maybe visited tiles?
        #    Let's try: Channel 0: Walls (visible or not), Channel 1: Agent, Channel 2: Exit (visible or not)
        
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(grid_size, grid_size, 4), # Wall, Trap, Agent, Exit
            dtype=np.float32
        )
        
        self.renderer = None
        self.window = None
        self.clock = None
        
        self.map = None
        self.agent_pos = None
        self.exit_pos = None
        self.timer = 0
        self.phase = "mem" # "mem" or "act"
        self.game_over = False
        self.message = ""
        self.trail: list[tuple[int, int]] = []

    def _generate_map(self):
        """Generate a traversable map with bounded retries to avoid infinite recursion."""
        max_attempts = 200
        for _ in range(max_attempts):
            size = self.grid_size
            self.map = np.zeros((size, size), dtype=int)
            
            # Borders
            self.map[0, :] = self.WALL
            self.map[-1, :] = self.WALL
            self.map[:, 0] = self.WALL
            self.map[:, -1] = self.WALL
            
            # Random internal walls (density 0.2)
            inner = self.map[1:-1, 1:-1]
            walls = self.np_random.random(inner.shape) < 0.2
            inner[walls] = self.WALL
            
            # Traps (density 0.05)
            traps = self.np_random.random(inner.shape) < 0.05
            # Don't overwrite walls
            inner[traps & ~walls] = self.TRAP
            
            # Start and Exit
            empty_coords = np.argwhere(self.map == self.EMPTY)
            if len(empty_coords) < 2:
                continue
                
            indices = self.np_random.choice(len(empty_coords), 2, replace=False)
            start = tuple(empty_coords[indices[0]])
            exit_ = tuple(empty_coords[indices[1]])
            
            self.map[start] = self.START
            self.map[exit_] = self.EXIT
            
            self.agent_pos = list(start)
            self.exit_pos = exit_
            
            # Ensure path exists (BFS)
            if self._path_exists(start, exit_):
                return
        
        raise RuntimeError("Failed to generate a traversable dungeon map after max attempts")

    def _path_exists(self, start, end):
        q = [start]
        visited = {start}
        while q:
            curr = q.pop(0)
            if curr == end:
                return True
            
            r, c = curr
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if (nr, nc) not in visited and self.map[nr, nc] != self.WALL:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            if "grid_size" in options and options["grid_size"] != self.grid_size:
                raise ValueError(
                    "grid_size is fixed after initialization; create a new env for a different size."
                )
            if "memorization_time" in options:
                self.memorization_time = options["memorization_time"]
                
        self._generate_map()
        self.timer = 0
        self.phase = "mem"
        self.game_over = False
        self.message = ""
        self.trail = [tuple(self.agent_pos)]
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        
        self.timer += 1
        
        if self.phase == "mem":
            if action == 4:
                self.phase = "act"
                self.timer = 0
            elif self.timer >= self.memorization_time:
                self.phase = "act"
                self.timer = 0  # Reset timer for step limit in act phase
            # During mem phase, agent cannot move
        else:
            # Act phase
            if self.timer >= self.max_steps:
                truncated = True
                self.game_over = True
                self.message = "TIMEOUT"
            
            # Movement
            dr, dc = 0, 0
            if action == 0: dr = -1
            elif action == 1: dc = 1
            elif action == 2: dr = 1
            elif action == 3: dc = -1
            # action == 4 is intentional wait/no-op
            
            r, c = self.agent_pos
            nr, nc = r + dr, c + dc
            
            # Collision Check
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                tile = self.map[nr, nc]
                if tile == self.WALL:
                    reward = -0.1
                    self.message = "BUMPED WALL"
                elif tile == self.TRAP:
                    reward = -1.0
                    terminated = True  # Trap = Die
                    self.game_over = True
                    self.message = "TRAPPED"
                elif tile == self.EXIT:
                    reward = 10.0
                    terminated = True
                    self.agent_pos = [nr, nc]
                    self.game_over = True
                    self.message = "ESCAPED!"
                else:
                    self.agent_pos = [nr, nc]
                    self.trail.append((nr, nc))
                    reward = -0.01  # Small step penalty to encourage speed
            else:
                 # Out of bounds behaves like a wall bump
                 reward = -0.1
                 self.message = "BUMPED WALL"

        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, {"success": reward > 1.0}

    def _get_obs(self):
        # Shape: (H, W, 4) -> Walls, Traps, Agent, Exit
        obs = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)
        
        visible = (self.phase == "mem") or self.game_over
        agent_r, agent_c = self.agent_pos
        
        # Layer 0: Walls
        if visible:
            obs[:, :, 0] = (self.map == self.WALL).astype(np.float32)
        else:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = agent_r + dr, agent_c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.map[nr, nc] == self.WALL:
                            obs[nr, nc, 0] = 1.0
        
        # Layer 1: Traps
        if visible:
            obs[:, :, 1] = (self.map == self.TRAP).astype(np.float32)
        else:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = agent_r + dr, agent_c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.map[nr, nc] == self.TRAP:
                            obs[nr, nc, 1] = 1.0
            
        # Layer 2: Agent (Always visible to self)
        obs[agent_r, agent_c, 2] = 1.0
        
        # Layer 3: Exit
        if visible:
            exit_r, exit_c = self.exit_pos
            obs[exit_r, exit_c, 3] = 1.0
        else:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = agent_r + dr, agent_c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.map[nr, nc] == self.EXIT:
                            obs[nr, nc, 3] = 1.0
            
        return obs

    def render(self):
        if self.render_mode is None: return
        
        game_size = 600
        sidebar_width = 250
        total_width = game_size + sidebar_width
        
        if not self.renderer:
            self.renderer = RetroRenderer(total_width, game_size, "Dungeon Escape")
            
        if self.render_mode == "human" and not self.window:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((total_width, game_size))
            pygame.display.set_caption("Dungeon Escape")
            self.clock = pygame.time.Clock()
            
        self.renderer.clear()
        
        cell_size = game_size // self.grid_size
        
        # If game over, show full map to explain death
        visible = (self.phase == "mem") or self.game_over
        agent_r, agent_c = self.agent_pos
        
        # Draw Map
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = c * cell_size
                y = r * cell_size
                rect = (x, y, cell_size, cell_size)
                
                tile = self.map[r, c]
                
                # Fog of War Logic
                # If phase is act, we only see agent and maybe visited?
                # For pure memory game: total darkness except agent.
                
                is_agent = (r == agent_r and c == agent_c)
                in_local_view = (abs(r - agent_r) <= 1 and abs(c - agent_c) <= 1)
                
                if visible or is_agent or in_local_view:
                    if tile == self.WALL:
                        self.renderer.draw_box(rect, COLOR_GRID, filled=True)
                        pygame.draw.rect(self.renderer.surface, COLOR_NEON_AMBER, rect, 1)
                    elif tile == self.TRAP:
                         self.renderer.draw_text("X", x + cell_size//2, y + cell_size//2, COLOR_RED, center=True, size=int(cell_size*0.8))
                    elif tile == self.EXIT:
                         self.renderer.draw_box(rect, COLOR_NEON_GREEN, filled=True, width=2)
                         self.renderer.draw_text("EXIT", x + cell_size//2, y + cell_size//2, COLOR_WHITE, center=True, size=int(cell_size*0.3))
                
                if is_agent:
                    # Draw Agent
                    agent_rect = (x + 4, y + 4, cell_size - 8, cell_size - 8)
                    self.renderer.draw_box(agent_rect, COLOR_NEON_CYAN, filled=True)
                    
                # If hidden, maybe draw a faint grid and breadcrumbs for intuition
                if not visible and not is_agent and not in_local_view:
                    pygame.draw.rect(self.renderer.surface, (20, 20, 20), rect, 1)
                    if (r, c) in self.trail:
                        crumb_rect = (x + cell_size//2 - 3, y + cell_size//2 - 3, 6, 6)
                        self.renderer.draw_box(crumb_rect, COLOR_NEON_CYAN, filled=True, width=1)

        # UI Overlay in Game Area
        if self.game_over:
             color = COLOR_NEON_GREEN if "ESCAPED" in self.message else COLOR_RED
             self.renderer.draw_text(self.message, game_size//2, 30, color, center=True, size=50)
        elif self.phase == "mem":
            time_left = max(0, self.memorization_time - self.timer)
            self.renderer.draw_text(f"MEMORIZE! {time_left}", game_size//2, 30, COLOR_NEON_PINK, center=True, size=40)
        else:
             steps_left = max(0, self.max_steps - self.timer)
             self.renderer.draw_text(f"ESCAPE! {steps_left}", game_size//2, 30, COLOR_NEON_GREEN, center=True, size=40)

        # Sidebar
        instructions = [
            "1. Memorize the Map",
            "   (Walls, Traps, Exit)",
            "2. Lights go out!",
            "3. Navigate to Exit",
            "4. Avoid Traps (X)",
            "   and Walls"
        ]
        controls = [
            ("ARROWS", "Move Agent"),
            ("SPACE", "Wait / Skip")
        ]
        
        # Infer level from grid size?
        # Base size 7. Level = size - 6 approx.
        level = self.grid_size - 6
        if level < 1: level = 1
        
        self.renderer.draw_sidebar(game_size, sidebar_width, instructions, controls, level=level)
             
        self.renderer.draw_scanlines()
        
        if self.render_mode == "human":
            self.window.blit(self.renderer.surface, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(30)
            
        return self.renderer.get_frame()

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()
            self.window = None
