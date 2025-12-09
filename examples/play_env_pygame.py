"""Interactive keyboard control for tiny-mem-gym environments using pygame.

Now features the new "Cool" environments.
"""

import argparse
import sys
import time

import pygame

import tiny_mem_gym
from tiny_mem_gym.envs import DungeonEscapeEnv, MemoryRacerEnv, CyberHackingEnv


def make_env(name: str):
    name = name.lower()
    if name in {"dungeon", "escape"}:
        return DungeonEscapeEnv(render_mode="human")
    if name in {"racer", "driving"}:
        return MemoryRacerEnv(render_mode="human")
    if name in {"hacking", "cyber"}:
        return CyberHackingEnv(render_mode="human")
    raise ValueError(f"Unknown env name: {name!r}")


def get_difficulty_options(env, level: int):
    """Return options dict for env.reset() based on level."""
    if isinstance(env, DungeonEscapeEnv):
        # Keep grid size fixed to avoid changing observation shape; shorten mem time as level rises.
        base_mem = 60
        new_mem = max(10, base_mem - (level * 2))
        return {"memorization_time": new_mem}
        
    if isinstance(env, MemoryRacerEnv):
        # Keep lane count fixed; increase speed modestly
        base_speed = 1.0
        new_speed = min(3.0, base_speed + (level - 1) * 0.2)
        return {"speed": new_speed}
        
    if isinstance(env, CyberHackingEnv):
        # Increase sequence length; keep grid size fixed to avoid shape changes
        base_len = 4
        new_len = base_len + (level - 1)
        return {"sequence_length": new_len}
        
    return {}


def select_action(env, keys) -> int | None:
    """Map keyboard state to action."""
    
    # Dungeon: Arrows to move, Wait (Space?)
    if isinstance(env, DungeonEscapeEnv):
        if keys[pygame.K_UP]: return 0
        if keys[pygame.K_RIGHT]: return 1
        if keys[pygame.K_DOWN]: return 2
        if keys[pygame.K_LEFT]: return 3
        if keys[pygame.K_SPACE]: return 4 # Wait
        return None

    # Racer: Left/Right arrows
    if isinstance(env, MemoryRacerEnv):
        if keys[pygame.K_LEFT]: return 0 # Left
        if keys[pygame.K_RIGHT]: return 2 # Right
        # Default is Stay (1) if no key pressed?
        # But for "Select Action" function we usually return non-None only on press.
        # But Racer needs constant input (Stay is an action).
        # We handle "Stay" in main loop if this returns None?
        return None

    # Hacking: Numpad/Grid keys?
    # This is tricky for generic grid size.
    # Let's map:
    # 7 8 9
    # 4 5 6
    # 1 2 3
    # To indices.
    if isinstance(env, CyberHackingEnv):
        # Map numpad to grid indices?
        # Or just 1-9 top-left to bottom-right?
        # 1 2 3
        # 4 5 6
        # 7 8 9
        mapping = {
            pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2,
            pygame.K_4: 3, pygame.K_5: 4, pygame.K_6: 5,
            pygame.K_7: 6, pygame.K_8: 7, pygame.K_9: 8,
            pygame.K_q: 0, pygame.K_w: 1, pygame.K_e: 2,
            pygame.K_a: 3, pygame.K_s: 4, pygame.K_d: 5,
            pygame.K_z: 6, pygame.K_x: 7, pygame.K_c: 8,
        }
        
        for k, v in mapping.items():
            if keys[k] and v < env.action_space.n:
                return v
        return None

    return None


def show_level_screen(env, level):
    """Show a simple level transition screen."""
    if not hasattr(env.unwrapped, "renderer") or env.unwrapped.renderer is None:
        return
        
    renderer = env.unwrapped.renderer
    renderer.clear()
    
    cx, cy = renderer.width // 2, renderer.height // 2
    renderer.draw_text(f"LEVEL {level}", cx, cy - 20, (255, 255, 255), center=True, size=40)
    renderer.draw_text("PRESS SPACE", cx, cy + 30, (200, 200, 200), center=True, size=20)
    renderer.draw_scanlines()
    
    if env.unwrapped.window:
        env.unwrapped.window.blit(renderer.surface, (0, 0))
        pygame.display.flip()
        
    waiting = True
    clock = pygame.time.Clock()
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    waiting = False
        clock.tick(30)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play tiny-mem-gym envs with the keyboard.")
    parser.add_argument(
        "--env",
        type=str,
        default="dungeon",
        choices=["dungeon", "racer", "hacking"],
        help="Which environment to play.",
    )
    args = parser.parse_args(argv)

    tiny_mem_gym.register_gymnasium_envs()
    env = make_env(args.env)

    running = True
    level = 1
    
    last_action_time = 0
    action_cooldown = 0.15 

    while running:
        show_level_screen(env, level)
        
        options = get_difficulty_options(env.unwrapped, level)
        obs, info = env.reset(options=options)
        
        episode_done = False
        success = False
        
        while not episode_done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                     if event.key == pygame.K_ESCAPE:
                         running = False

            if not running:
                break

            env.render()
            
            current_time = time.time()
            keys = pygame.key.get_pressed()
            action = select_action(env, keys)
            
            step_taken = False
            
            # Environment Specific Loop Logic
            if isinstance(env.unwrapped, DungeonEscapeEnv):
                # Turn based
                if action is not None:
                    if current_time - last_action_time > action_cooldown:
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_done = terminated or truncated
                        success = info.get("success", False)
                        last_action_time = current_time
                else:
                    # Auto-tick needed for timer?
                    # The env tracks memorization time in steps.
                    # So we need to call step even if waiting?
                    # DungeonEscapeEnv logic: step increments timer.
                    # If phase is "mem", step(action) ignores action but ticks timer.
                    # So we should auto-step if phase is "mem".
                    if env.unwrapped.phase == "mem":
                        if current_time - last_action_time > 0.05: # Fast forward memorization
                            env.step(4) # Wait
                            last_action_time = current_time
            
            elif isinstance(env.unwrapped, MemoryRacerEnv):
                # Real-time
                # If no key, Action 1 (Stay)
                effective_action = action if action is not None else 1
                
                # Slower tick for game loop
                if current_time - last_action_time > 0.1: # 10 steps per sec
                    obs, reward, terminated, truncated, info = env.step(effective_action)
                    episode_done = terminated or truncated
                    success = info.get("success", False)
                    last_action_time = current_time
                    
            elif isinstance(env.unwrapped, CyberHackingEnv):
                # Watch Phase = Auto tick
                if env.unwrapped.phase == "watch":
                     if current_time - last_action_time > 0.05:
                         env.step(0)
                         last_action_time = current_time
                # Input Phase = Turn based
                elif action is not None:
                    if current_time - last_action_time > 0.2:
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_done = terminated or truncated
                        success = info.get("success", False)
                        last_action_time = current_time
            
            # env.unwrapped.clock.tick(60) # Handled in render

        if running:
            env.render()
            
            # If success, brief pause.
            # If fail, long pause (to see reason).
            if success:
                pygame.time.wait(1000)
                level += 1
            else:
                pygame.time.wait(2500) # Wait 2.5s to see why you failed
                pass 
                
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
