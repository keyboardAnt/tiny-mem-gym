import numpy as np
import gymnasium as gym

from tiny_mem_gym.envs import (
    TinyMemoryBanditEnv,
    SequenceRecallEnv,
    TinyPOGridworldEnv,
)


def _make_envs():
    return [
        TinyMemoryBanditEnv(),
        SequenceRecallEnv(),
        TinyPOGridworldEnv(),
    ]


def test_reset_and_step_api():
    for env in _make_envs():
        obs, info = env.reset(seed=0)
        assert isinstance(info, dict)
        assert env.observation_space.contains(obs)

        done = False
        steps = 0
        while not done and steps < env.max_episode_steps + 5:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            done = terminated or truncated
            steps += 1

        assert done


