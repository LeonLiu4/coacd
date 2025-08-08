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
TOTAL_STEPS  = 4096
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
    # Removed early stopping to allow full training duration
    eval_cb   = EvalCallback(
        eval_env,
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
    
    # ── visualize best results ───────────────────────────────────────
    print("\n" + "="*50)
    print("VISUALIZING BEST RESULTS")
    print("="*50)
    
    # Load the best model and visualize
    try:
        from visualize_results import load_best_model, evaluate_model_on_mesh
        from src.utils.visualization import visualize_best_decomposition
        
        best_model = load_best_model(BEST_DIR)
        if best_model:
            print("Evaluating best model for visualization...")
            best_params = evaluate_model_on_mesh(best_model, MESH_TRAIN, n_episodes=3)
            
            if best_params:
                print(f"Best parameters found:")
                for key, value in best_params.items():
                    print(f"  {key}: {value}")
                
                print("\nGenerating visualizations...")
                os.makedirs("visualizations", exist_ok=True)
                visualize_best_decomposition(MESH_TRAIN, best_params, "visualizations")
                print("✓ Visualizations saved to 'visualizations/' directory")
            else:
                print("No valid parameters found during evaluation")
        else:
            print("Could not load best model for visualization")
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("You can run visualization manually with: python visualize_results.py")


if __name__ == "__main__":
    main()