"""Interactive keyboard control for tiny-mem-gym environments using pygame.

This script is intentionally lightweight and meant for quick manual play and
visual inspection of the environments.
"""

import argparse
import sys

import pygame

import tiny_mem_gym
from tiny_mem_gym.envs import TinyMemoryBanditEnv, SequenceRecallEnv, TinyPOGridworldEnv


def make_env(name: str):
    name = name.lower()
    if name == "bandit":
        return TinyMemoryBanditEnv()
    if name in {"sequence", "sequence_recall"}:
        return SequenceRecallEnv()
    if name == "gridworld":
        return TinyPOGridworldEnv()
    raise ValueError(f"Unknown env name: {name!r}")


def select_action(env, keys) -> int:
    """Map keyboard state to a discrete action for the given env."""
    # Gridworld: arrows move, staying still is a valid action.
    if isinstance(env, TinyPOGridworldEnv):
        if keys[pygame.K_UP]:
            return 0
        if keys[pygame.K_RIGHT]:
            return 1
        if keys[pygame.K_DOWN]:
            return 2
        if keys[pygame.K_LEFT]:
            return 3
        return 4  # stay

    # Bandit: keys 1..n_arms select an arm, default to 0.
    if isinstance(env, TinyMemoryBanditEnv):
        for k, arm in (
            (pygame.K_1, 0),
            (pygame.K_2, 1),
            (pygame.K_3, 2),
            (pygame.K_4, 3),
        ):
            if arm < env.n_arms and keys[k]:
                return arm
        return 0

    # Sequence recall: assume Discrete(n); keys 1..n select actions.
    if isinstance(env, SequenceRecallEnv):
        n = env.action_space.n
        for idx, key in enumerate(
            (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6)
        ):
            if idx < n and keys[key]:
                return idx
        return 0

    # Fallback: random action.
    return int(env.action_space.sample())


def _wrap_text(line: str, font: pygame.font.Font, max_width: int) -> list[str]:
    """Wrap a single line of text to fit within max_width pixels.

    This keeps instructions readable even in small windows.
    """
    words = line.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and font.size(candidate)[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play tiny-mem-gym envs with the keyboard.")
    parser.add_argument(
        "--env",
        type=str,
        default="gridworld",
        choices=["bandit", "sequence", "sequence_recall", "gridworld"],
        help="Which environment to play.",
    )
    args = parser.parse_args(argv)

    tiny_mem_gym.register_gymnasium_envs()
    env = make_env(args.env)

    obs, info = env.reset()
    reward: float = 0.0
    episode_done = False

    pygame.init()
    pygame.font.init()
    # Use a modest default window size and scale the frame to fit.
    window_width, window_height = 256, 256
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"tiny-mem-gym: {env.__class__.__name__}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 16)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
            continue

        # If an episode has finished, wait for SPACE before starting the next
        # one so humans have time to read the outcome.
        if episode_done and keys[pygame.K_SPACE]:
            obs, info = env.reset()
            reward = 0.0
            episode_done = False

        if not episode_done:
            action = select_action(env, keys)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_done = terminated or truncated
        else:
            terminated = False
            truncated = False

        # Render current frame and display in the window.
        frame = env.render()  # H x W x 3 uint8
        # Scale frame to the window size for convenience.
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.smoothscale(surf, (window_width, window_height))
        screen.blit(surf, (0, 0))

        # Overlay simple HUD with cost and controls to make the games
        # understandable and fun for humans.
        display_cost = -float(reward)
        hud_lines = []
        hud_lines.append(f"Step cost: {display_cost:.2f}")
        if isinstance(env, TinyMemoryBanditEnv):
            hud_lines.append("Bandit: at the start one green column flashes = winning arm.")
            hud_lines.append("Press 1-4 to choose an arm; middle strip green = correct, red = miss.")
        elif isinstance(env, SequenceRecallEnv):
            hud_lines.append(
                f"Sequence (n-back={getattr(env, 'n_back', 1)}): "
                "blue = watch the color sequence (no keys)."
            )
            hud_lines.append("Yellow = answer: press 1-4 for the color from n steps ago; bottom bar green = correct.")
        elif isinstance(env, TinyPOGridworldEnv):
            hud_lines.append("Gridworld: you are the red square; green square is the goal.")
            hud_lines.append("Move with arrow keys to reach the goal before steps run out.")
        if episode_done:
            total_cost = -float(info.get("cumulative_reward", 0.0))
            success = bool(info.get("success"))
            hud_lines.append("Episode complete!")
            hud_lines.append(f"Total cost this episode: {total_cost:.2f}")
            hud_lines.append(f"Status: {'success' if success else 'not successful yet'}")
            hud_lines.append("Press SPACE for next episode.")

        hud_lines.append("Esc: quit game")

        y = 4
        max_text_width = window_width - 8
        for base_line in hud_lines:
            for line in _wrap_text(base_line, font, max_text_width):
                text_surf = font.render(line, True, (255, 255, 255))
                screen.blit(text_surf, (4, y))
                y += text_surf.get_height() + 2

        pygame.display.flip()

        clock.tick(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])


