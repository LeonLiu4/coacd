# coacd_ppo_train.py

import os
import time

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

# side‐effect: imports and registers your CoACDEnv under "CoACD-v0"
import src.envs

from src.models.pointnet_param_net import PointNetFeatureExtractor

# ───────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────
MESH_PATH    = "assets/bunny_simplified.obj"
N_STEPS      = 32             # steps per rollout
TOTAL_STEPS  = 4096           # total training timesteps
MAX_EPISODE  = N_STEPS        # force episode end after N_STEPS
LOG_DIR      = "logs/ppo_pointnet"
MODEL_DIR    = "models"


def make_env():
    """
    Factory to create a single CoACD environment, capped in length
    and wrapped so that Monitor records ep stats into `info["episode"]`.
    """
    # 1) Instantiate your custom CoACD gym env
    env = gym.make("CoACD-v0", mesh_path=MESH_PATH)

    # 2) Force a hard cap so episodes actually terminate
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE)

    # 3) Wrap in SB3 Monitor so it writes `info["episode"] = {"r":…, "l":…}`
    env = Monitor(env)

    return env


def main():
    # Ensure the env ID is registered (no harm if src.envs already did it)
    register(id="CoACD-v0", entry_point="src.envs:CoACDEnv")

    # Prepare output folders
    os.makedirs(LOG_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Build a single‐env vectorized wrapper, then add VecMonitor
    vec_env = DummyVecEnv([make_env])
    vec_env = VecMonitor(vec_env)

    # Configure our custom PointNet policy
    policy_kwargs = dict(
        features_extractor_class  = PointNetFeatureExtractor,
        features_extractor_kwargs = dict(features_dim=128),
        net_arch                  = dict(pi=[256,128], vf=[256,128]),
    )

    # Instantiate PPO
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps        = N_STEPS,
        batch_size     = 32,
        learning_rate  = 3e-4,
        verbose        = 1,
        tensorboard_log= LOG_DIR,
        policy_kwargs  = policy_kwargs,
    )

    # Train
    model.learn(total_timesteps=TOTAL_STEPS)

    # Save & cleanup
    outpath = os.path.join(MODEL_DIR, f"ppo_pointnet_{int(time.time())}")
    model.save(outpath)
    vec_env.close()
    print("Training finished ✔️")


if __name__ == "__main__":
    main()