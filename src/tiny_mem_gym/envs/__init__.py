from gymnasium.envs.registration import register

from tiny_mem_gym.envs.dungeon import DungeonEscapeEnv
from tiny_mem_gym.envs.racer import MemoryRacerEnv
from tiny_mem_gym.envs.hacking import CyberHackingEnv

def register_gymnasium_envs():
    register(
        id="TinyMemory-Dungeon-v0",
        entry_point="tiny_mem_gym.envs.dungeon:DungeonEscapeEnv",
        max_episode_steps=200,
    )
    register(
        id="TinyMemory-Racer-v0",
        entry_point="tiny_mem_gym.envs.racer:MemoryRacerEnv",
        max_episode_steps=1000,
    )
    register(
        id="TinyMemory-Hacking-v0",
        entry_point="tiny_mem_gym.envs.hacking:CyberHackingEnv",
        max_episode_steps=100,
    )
