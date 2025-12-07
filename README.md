## Tiny-mem-gym

Tiny-mem-gym is a small collection of Gymnasium-compatible environments that emphasize memory, partial observability, and simple control.
It is intended as a lightweight benchmark suite and teaching tool for reinforcement learning research focused on memory.

### Features

- **Memory-focused tasks**: bandits with hidden cues, sequence recall / N-back, and tiny partially observable gridworlds.
- **Gymnasium API**: all environments follow the Gymnasium 0.27+ API (`reset`, `step`, `observation_space`, `action_space`).
- **Tiny observation spaces**: low-dimensional observations for fast experimentation.
- **Minimal dependencies**: just `gymnasium` and `numpy` for core usage.

### Installation

With `uv`:

```bash
uv sync
```

Or with `pip` in editable mode:

```bash
pip install -e .
```

### Quickstart

Create and step through a tiny memory bandit:

```python
import gymnasium as gym
import tiny_mem_gym

from tiny_mem_gym.envs import TinyMemoryBanditEnv

env = TinyMemoryBanditEnv()
obs, info = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

Once environments are registered with Gymnasium, you can also use:

```python
import gymnasium as gym
import tiny_mem_gym

tiny_mem_gym.register_gymnasium_envs()
env = gym.make("TinyMem-Bandit-v0")
```

### Environments

- **Memory bandits**: an initial cue reveals the rewarding arm, which must be remembered over several timesteps.
- **Sequence recall / N-back**: agents observe symbol streams and must answer N-back style queries.
- **Tiny partially observable gridworld**: an agent moves in a small grid with only a local egocentric view.

See the examples under `examples/` and the docstrings in `tiny_mem_gym.envs` for more details.


