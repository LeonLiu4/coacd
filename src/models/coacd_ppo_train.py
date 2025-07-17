import os, time
import src.envs 
from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.models.pointnet_param_net import PointNetFeatureExtractor

# ---------------------------------------------------------------------
# 2.  Config
# ---------------------------------------------------------------------
MESH_PATH   = "assets/bunny.obj"
N_STEPS     = 2
TOTAL_STEPS = 1
LOG_DIR     = "logs/ppo_pointnet"
MODEL_DIR   = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 3.  Build the VecEnv
# ---------------------------------------------------------------------
def make_env():
    return gym.make("CoACD-v0", mesh_path=MESH_PATH)

env = DummyVecEnv([make_env])

# ---------------------------------------------------------------------
# 4.  Set up PPO
# ---------------------------------------------------------------------
policy_kwargs = dict(
    features_extractor_class  = PointNetFeatureExtractor,
    features_extractor_kwargs = dict(features_dim=128),
    net_arch                  = dict(pi=[256, 128], vf=[256, 128]),
)

model = PPO(
    "MlpPolicy",
    env,
    n_steps      = N_STEPS,
    batch_size   = 2,
    learning_rate= 3e-4,
    verbose      = 1,
    tensorboard_log = LOG_DIR,
)

# ---------------------------------------------------------------------
# 5.  Train
# ---------------------------------------------------------------------
model.learn(total_timesteps=TOTAL_STEPS)
model.save(f"{MODEL_DIR}/ppo_pointnet_{int(time.time())}")
env.close()
print("Training finished ✔️")
