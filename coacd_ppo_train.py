# src/models/coacd_ppo_train.py

import os
import time

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.envs import CoACDEnv            # your env implementation
from src.models.pointnet_param_net import PointNetFeatureExtractor

# ───────────────────────────────────────────────────────────────────
# 1) Module-level config (must come *before* make_env so MESH_PATH is in scope)
# ───────────────────────────────────────────────────────────────────
MESH_PATH   = "assets/bunny_simplified.obj"
N_STEPS     = 256
TOTAL_STEPS = 4096
LOG_DIR     = "logs/ppo_pointnet"
MODEL_DIR   = "models"


class PrintTimestepCallback(BaseCallback):
    """Print global timestep every `print_freq` environment steps."""
    def __init__(self, print_freq: int = N_STEPS, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"[RL] global timestep: {self.num_timesteps}")
        return True


def make_env():
    """Factory for our custom CoACDEnv; MESH_PATH is captured from module scope."""
    return gym.make("CoACD-v0", mesh_path=MESH_PATH)


def main():
    # ─────────────────────────────────────────────────────────────────
    # Register our custom environment
    # ─────────────────────────────────────────────────────────────────
    register(
        id="CoACD-v0",
        entry_point="src.envs:CoACDEnv",
    )

    # ─────────────────────────────────────────────────────────────────
    # Prepare log & model dirs
    # ─────────────────────────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # Build the (monitored) vectorized environment
    # ─────────────────────────────────────────────────────────────────
    # We wrap each sub‐env with Monitor (to record episode returns & lengths),
    # then wrap the DummyVecEnv in VecMonitor to aggregate across processes.
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    # ─────────────────────────────────────────────────────────────────
    # Set up PPO with a custom PointNet feature extractor
    # ─────────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class  = PointNetFeatureExtractor,
        features_extractor_kwargs = dict(features_dim=128),
        net_arch                  = dict(pi=[256, 128], vf=[256, 128]),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=N_STEPS,
        batch_size=32,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs,
    )

    # ─────────────────────────────────────────────────────────────────
    # Train & save
    # ─────────────────────────────────────────────────────────────────
    callback = PrintTimestepCallback(print_freq=N_STEPS)
    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

    model.save(os.path.join(MODEL_DIR, f"ppo_pointnet_{int(time.time())}"))
    env.close()
    print("Training finished ✔️")


if __name__ == "__main__":
    main()