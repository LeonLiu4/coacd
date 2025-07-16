from gymnasium.envs.registration import register
from src.envs.coacd_env import CoACDEnv

register(
    id="CoACD-v0",
    entry_point="src.envs.coacd_env:CoACDEnv",
)