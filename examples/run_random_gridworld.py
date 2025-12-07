"""Run a random policy in the tiny partially observable gridworld."""

import tiny_mem_gym
from tiny_mem_gym.envs import TinyPOGridworldEnv


def main() -> None:
    tiny_mem_gym.register_gymnasium_envs()

    print("Running TinyPOGridworldEnv via direct construction")
    env = TinyPOGridworldEnv()
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

        # Optionally render a final frame.
        frame = env.render()
        print(f"Final frame shape: {frame.shape}")


if __name__ == "__main__":
    main()


