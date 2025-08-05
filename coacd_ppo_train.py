#!/usr/bin/env python3
import os
import time

import gymnasium as gym
import src.envs                       # registers "CoACD-v0"
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from src.models.pointnet_param_net import PointNetFeatureExtractor

# ---------------------------------------------------------------------
# 2. Config (must come *before* make_env)
# ---------------------------------------------------------------------
MESH_PATH   = "assets/bunny_simplified.obj"
N_STEPS     = 256
TOTAL_STEPS = 4096
LOG_DIR     = "logs/ppo_pointnet"
MODEL_DIR   = "models"

class PrintTimestepCallback(BaseCallback):
    """Print global timestep every `print_freq` env steps."""
    def __init__(self, print_freq: int = N_STEPS, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"[RL] global timestep: {self.num_timesteps}")
        return True

# ---------------------------------------------------------------------
# 3. Build the VecEnv
# ---------------------------------------------------------------------
def make_env():
    # Now MESH_PATH is in scope!
    return gym.make("CoACD-v0", mesh_path=MESH_PATH)

def main():
    # prepare folders
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # single-env vectorized wrapper
    env = DummyVecEnv([make_env])

    # policy / feature-extractor setup
    policy_kwargs = dict(
        features_extractor_class  = PointNetFeatureExtractor,
        features_extractor_kwargs = dict(features_dim=128),
        net_arch                  = dict(pi=[256, 128], vf=[256, 128]),
    )

    # -----------------------------------------------------------------
    # 4. Set up PPO
    # -----------------------------------------------------------------
    model = PPO(
        "MlpPolicy",
        env,
        n_steps         = N_STEPS,
        batch_size      = 32,
        learning_rate   = 3e-4,
        verbose         = 1,
        tensorboard_log = LOG_DIR,
        policy_kwargs   = policy_kwargs,
    )

    # -----------------------------------------------------------------
    # 5. Train with timestep-printing callback
    # -----------------------------------------------------------------
    callback = PrintTimestepCallback(print_freq=N_STEPS)
    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = callback,
    )

    # save & clean up
    model.save(os.path.join(MODEL_DIR, f"ppo_pointnet_{int(time.time())}"))
    env.close()
    print("Training finished ✔️")

if __name__ == "__main__":
    main()