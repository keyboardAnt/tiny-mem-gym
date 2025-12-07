"""Run a random policy in the tiny memory bandit environment."""

import gymnasium as gym

import tiny_mem_gym
from tiny_mem_gym.envs import TinyMemoryBanditEnv


def main() -> None:
    tiny_mem_gym.register_gymnasium_envs()

    print("Running TinyMemoryBanditEnv via direct construction")
    env = TinyMemoryBanditEnv()
    for episode in range(3):
        obs, info = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {episode}: return={total_reward:.2f}")

    print("\nRunning TinyMemoryBanditEnv via gymnasium.make")
    env = gym.make("TinyMem-Bandit-v0")
    for episode in range(3):
        obs, info = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"[gym.make] Episode {episode}: return={total_reward:.2f}")


if __name__ == "__main__":
    main()


