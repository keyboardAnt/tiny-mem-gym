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

    pygame.init()
    # Use a modest default window size and scale the frame to fit.
    window_width, window_height = 256, 256
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"tiny-mem-gym: {env.__class__.__name__}")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
            continue

        action = select_action(env, keys)

        obs, reward, terminated, truncated, info = env.step(action)

        # Render current frame and display in the window.
        frame = env.render()  # H x W x 3 uint8
        # Scale frame to the window size for convenience.
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.smoothscale(surf, (window_width, window_height))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            obs, info = env.reset()

        clock.tick(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])


