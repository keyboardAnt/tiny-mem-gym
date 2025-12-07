"""Tiny memory-focused Gymnasium environments.

This package provides a small collection of environments that emphasize memory,
partial observability, and simple continuous or discrete control.
"""

from . import envs

__all__ = ["envs"]

__version__ = "0.1.0"


def register_gymnasium_envs() -> None:
    """Register tiny-mem-gym environments with Gymnasium's registry.

    This is a light convenience wrapper so users can call
    ``gymnasium.make("TinyMem-Bandit-v0")`` and similar IDs.
    """
    try:
        import gymnasium as gym
        from gymnasium.envs.registration import register
    except Exception:  # pragma: no cover - Gymnasium not installed or changed
        return

    # Avoid re-registering if IDs already exist.
    registry = getattr(gym, "envs", None)
    existing_ids = set(getattr(registry, "registry", {}).keys()) if registry else set()

    def _safe_register(env_id: str, entry_point: str, max_episode_steps: int | None = None) -> None:
        if env_id in existing_ids:
            return

        kwargs: dict[str, object] = {"id": env_id, "entry_point": entry_point}
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = max_episode_steps
        register(**kwargs)

    _safe_register("TinyMem-Bandit-v0", "tiny_mem_gym.envs:TinyMemoryBanditEnv", max_episode_steps=50)
    _safe_register("TinyMem-SequenceRecall-v0", "tiny_mem_gym.envs:SequenceRecallEnv", max_episode_steps=50)
    _safe_register("TinyMem-Gridworld-v0", "tiny_mem_gym.envs:TinyPOGridworldEnv", max_episode_steps=100)


def make(id: str, **kwargs):
    """Thin wrapper around :func:`gymnasium.make`.

    This exists purely for convenience and to avoid importing Gymnasium
    directly in simple scripts.
    """
    import gymnasium as gym

    return gym.make(id, **kwargs)


