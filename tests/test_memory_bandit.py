from tiny_mem_gym.envs import TinyMemoryBanditEnv


def test_memory_bandit_basic_behavior():
    env = TinyMemoryBanditEnv(n_arms=3, max_episode_steps=5, cue_duration=2)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["timestep"] == 0

    # Take one step with a valid action.
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert reward in (0.0, 1.0)
    assert not terminated
    assert isinstance(truncated, bool)


