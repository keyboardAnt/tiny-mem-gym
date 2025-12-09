import numpy as np
from tiny_mem_gym.envs import DungeonEscapeEnv, MemoryRacerEnv, CyberHackingEnv

def test_dungeon_escape():
    env = DungeonEscapeEnv(grid_size=5, memorization_time=10, max_steps=20, render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    
    # Check phases
    assert env.phase == "mem"
    
    # Step through mem phase
    for _ in range(11):
        env.step(4) # Wait
        
    assert env.phase == "act"
    
    # Basic movement
    obs, _, _, _, _ = env.step(1) # Right
    assert env.observation_space.contains(obs)
    
    # Test Render
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (600, 850, 3) # 600 height, 600+250 width

def test_memory_racer():
    env = MemoryRacerEnv(n_lanes=3, obs_depth=100, max_steps=50, render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (20, 3, 2)
    
    # Run for a bit
    for _ in range(10):
        obs, _, terminated, _, _ = env.step(1) # Stay
        if terminated:
            break
        
    assert not terminated
    
    # Test Render
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (600, 650, 3) # 600 height, 400+250 width

def test_cyber_hacking():
    env = CyberHackingEnv(grid_size=3, sequence_length=3, display_time=10, render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    
    assert env.phase == "watch"
    
    # Step through watch
    for _ in range(11):
        env.step(0)
        
    assert env.phase == "input"
    
    # We can peek at sequence for test
    target = env.sequence[0]
    _, reward, terminated, _, _ = env.step(target)
    
    assert reward == 1.0
    assert not terminated
    
    # Test Render
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (512, 512+250, 3)

def test_dungeon_trail_and_wait():
    env = DungeonEscapeEnv(grid_size=5, memorization_time=0, max_steps=10, render_mode=None)
    env.reset(seed=1, options={"memorization_time": 0})
    start_pos = tuple(env.agent_pos)
    # Wait action should not move the agent
    env.step(4)
    assert tuple(env.agent_pos) == start_pos
    
    # Find a safe move to grow the breadcrumb trail
    moved = False
    moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    for action, (dr, dc) in moves.items():
        r, c = env.agent_pos
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.grid_size and 0 <= nc < env.grid_size:
            tile = env.map[nr, nc]
            if tile not in (env.WALL, env.TRAP):
                env.step(action)
                moved = True
                break
    if moved:
        assert len(env.trail) >= 2
    else:
        assert len(env.trail) == 1

def test_memory_racer_records_memory_marks():
    env = MemoryRacerEnv(n_lanes=3, obs_depth=20, speed=10, max_steps=5, render_mode=None)
    env.reset(seed=0)
    env.obstacles = [[1, 15]]  # visible warning, will enter memory zone after one step
    env.step(1)  # stay
    assert env.memory_marks
    lane, dist = env.memory_marks[0]
    assert lane == 1
    assert dist < env.obs_depth // 2
