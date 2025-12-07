from tiny_mem_gym.envs import SequenceRecallEnv


def test_sequence_recall_basic_behavior():
    env = SequenceRecallEnv(
        n_symbols=4,
        sequence_length=4,
        n_back=1,
        query_length=2,
        query_mode="predict",
    )
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["timestep"] == 0

    # Run through the entire episode with random actions to ensure it terminates.
    done = False
    steps = 0
    while not done and steps < env.max_episode_steps + 5:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        done = terminated or truncated
        steps += 1

    assert done


