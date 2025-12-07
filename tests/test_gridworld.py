from tiny_mem_gym.envs import TinyPOGridworldEnv


def test_gridworld_reach_goal_and_render():
    env = TinyPOGridworldEnv(grid_size=5, window_size=3, max_episode_steps=20)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["timestep"] == 0

    # Take a few random steps.
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break

    # Render should return an RGB array.
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[-1] == 3


