# coacd_ppo_train.py  – train PPO on CoACD + save best model via EvalCallback
import os, time, gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.monitor     import Monitor
from stable_baselines3.common.vec_env     import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks   import EvalCallback, StopTrainingOnRewardThreshold

# side-effect: adds "CoACD-v0" to the Gym registry
import src.envs                              # noqa: F401
from src.models.pointnet_param_net import PointNetFeatureExtractor

# ─────────────────────────────────────────────
# config
# ─────────────────────────────────────────────
MESH_TRAIN = "assets/bunny_simplified.obj"
MESH_EVAL  = "assets/bunny_simplified.obj"
N_STEPS      = 32
TOTAL_STEPS  = 64
MAX_EPISODE  = N_STEPS
LOG_DIR      = "logs/ppo_pointnet"
MODEL_DIR    = "models"
BEST_DIR     = os.path.join(MODEL_DIR, "best")  # where EvalCallback writes

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────
def _make_env(mesh_path: str):
    """Instantiate *one* CoACD env with TimeLimit + Monitor."""
    def _factory():
        env = gym.make("CoACD-v0", mesh_path=mesh_path)
        env = TimeLimit(env, MAX_EPISODE)
        env = Monitor(env)
        return env

    return _factory


def main() -> None:
    # (re-)register – harmless if already present
    register(id="CoACD-v0", entry_point="src.envs:CoACDEnv")

    # ── training env (single-proc vec) ───────────────────────────────
    train_env = VecMonitor(DummyVecEnv([_make_env(MESH_TRAIN)]))

    # ── evaluation env (single instance, no vector wrapper needed) ───
    eval_env  = Monitor(_make_env(MESH_EVAL)())

    # ── callbacks ────────────────────────────────────────────────────
    # stop early if we ever beat some reward (optional)
    stop_cb   = StopTrainingOnRewardThreshold(reward_threshold=0.0, verbose=1)

    eval_cb   = EvalCallback(
        eval_env,
        callback_on_new_best=stop_cb,          # stop when threshold reached
        best_model_save_path=BEST_DIR,
        log_path            =LOG_DIR,
        eval_freq           =N_STEPS,
        deterministic       =True,
        render              =False,
    )

    # ── PPO policy hyper-params ──────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class  = PointNetFeatureExtractor,
        features_extractor_kwargs = dict(features_dim=128),
        net_arch                  = dict(pi=[256, 128], vf=[256, 128]),
    )

    model = PPO(
        policy          ="MlpPolicy",
        env             =train_env,
        n_steps         =N_STEPS,
        batch_size      =32,
        learning_rate   =3e-4,
        tensorboard_log =LOG_DIR,
        verbose         =1,
        policy_kwargs   =policy_kwargs,
    )

    # ── train with evaluation callback ───────────────────────────────
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback       =eval_cb,
        progress_bar   =False,
    )

    # last checkpoint (even if not best)
    ts = int(time.time())
    model.save(os.path.join(MODEL_DIR, f"ppo_pointnet_final_{ts}"))
    train_env.close(), eval_env.close()
    print("✓ training done – best model in", BEST_DIR)


if __name__ == "__main__":
    main()