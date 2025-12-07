## Contributing to tiny-mem-gym

Thanks for your interest in contributing tiny memory-focused environments!

### Development environment

- Install dependencies (using `uv` is recommended):
  - `uv sync` to create a virtual environment and install dependencies.
- Alternatively, use `pip`:
  - `pip install -e .[dev]`

### Coding style and API conventions

- Environments must be compatible with **Gymnasium 0.27+**:
  - Subclass `gymnasium.Env`.
  - Implement `reset(self, *, seed=None, options=None)` returning `(obs, info)`.
  - Implement `step(self, action)` returning `(obs, reward, terminated, truncated, info)`.
  - Define `observation_space` and `action_space` as `gymnasium.spaces.Space` instances.
- Observations should be **small** (roughly 4–64 dimensions) and use NumPy dtypes (`np.float32` by default).
- Each env constructor should accept `max_episode_steps` and respect it when computing `truncated`.
- Always return an `info` dict, even if empty, with task-generic keys such as `"timestep"`, `"success"`, and `"cumulative_reward"`.

### Tests

- Tests are written with `pytest` and live under `tests/`.
- To run the test suite:

```bash
uv run pytest
# or
pytest
```

- New environments should include:
  - Smoke tests for API compliance (`reset`, `step`, spaces).
  - Basic behavior tests (e.g. rewards and termination conditions).

### Adding a new environment

1. Add a module under `src/tiny_mem_gym/envs/` implementing your env.
2. Export the env from `src/tiny_mem_gym/envs/__init__.py`.
3. Optionally register a Gymnasium ID in `tiny_mem_gym.register_gymnasium_envs`.
4. Add tests and, if helpful, a small example script under `examples/`.

### Releasing

- Update `CHANGELOG.md` with a new entry.
- Bump the version in `pyproject.toml` and `tiny_mem_gym.__version__`.
- Build and publish with `uv`:

```bash
uv build
uv publish
```


