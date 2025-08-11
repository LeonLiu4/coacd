# coacd_ppo_train.py  – train PPO on CoACD + save best model via EvalCallback
import os, time, gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.monitor     import Monitor
from stable_baselines3.common.vec_env     import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks   import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CallbackList

# side-effect: adds "CoACD-v0" to the Gym registry
import src.envs                              # noqa: F401
from src.models.pointnet_param_net import PointNetFeatureExtractor

# ─────────────────────────────────────────────
# config
# ─────────────────────────────────────────────
MESH_TRAIN = "assets/bunny_simplified.obj"
MESH_EVAL  = "assets/bunny_simplified.obj"
N_STEPS      = 32
TOTAL_STEPS  = 1024
MAX_EPISODE  = N_STEPS
LOG_DIR      = "logs/ppo_pointnet"
MODEL_DIR    = "models"
BEST_DIR     = os.path.join(MODEL_DIR, "best")  # where EvalCallback writes

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# custom callback for detailed logging
# ─────────────────────────────────────────────
class DetailedLoggingCallback(BaseCallback):
    """Custom callback to log detailed metrics to TensorBoard"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

        self.total_steps = 0
        
    def _on_step(self) -> bool:
        self.total_steps += 1

        if self.total_steps % 10 == 0:
            # Log episode-level metrics when episode ends
            if len(self.locals.get('dones', [])) > 0 and any(self.locals['dones']):
                # Get info from the environment
                infos = self.locals.get('infos', [])
                for info in infos:
                    if info:  # info might be empty dict
                        # Log custom metrics if available
                        if 'H' in info:
                            self.logger.record_mean('custom/hausdorff_distance', info['H'])
                        if 'T' in info:
                            self.logger.record_mean('custom/runtime', info['T'])
                        if 'V' in info:
                            self.logger.record_mean('custom/total_vertices', info['V'])
                        if 'num_parts' in info:
                            self.logger.record_mean('custom/num_parts', info['num_parts'])
                        if 'success' in info:
                            self.logger.record_mean('custom/success', float(info['success']))
                            
                # Log general training metrics
                rewards = self.locals.get('rewards', [])
                if len(rewards) > 0:
                    self.logger.record_mean('custom/step_reward_mean', float(rewards.mean()))
                    self.logger.record_mean('custom/step_reward_std', float(rewards.std()))

                self.logger.dump(self.total_steps)
                
        return True


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
    # Custom logging callback for detailed metrics
    logging_cb = DetailedLoggingCallback(verbose=1)
    
    # Evaluation callback (removed early stopping to allow full training duration)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path            =LOG_DIR,
        eval_freq           =N_STEPS,
        deterministic       =True,
        render              =False,
        n_eval_episodes     =3,
    )
    
    # Combine callbacks
    callback = CallbackList([logging_cb, eval_cb])

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

    # ── train with combined callbacks ────────────────────────────────
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback       =callback,
        progress_bar   =False,
    )

    # last checkpoint (even if not best)
    ts = int(time.time())
    model.save(os.path.join(MODEL_DIR, f"ppo_pointnet_final_{ts}"))
    train_env.close(), eval_env.close()
    print("✓ training done – best model in", BEST_DIR)
    print(f"\n📊 To view training curves in TensorBoard, run:")
    print(f"   tensorboard --logdir {LOG_DIR}")
    print(f"   Then open http://localhost:6006 in your browser")
    
    # ── visualize best results ───────────────────────────────────────
    print("\n" + "="*50)
    print("VISUALIZING BEST RESULTS")
    print("="*50)
    
    try:
        from visualize_results import load_best_model, evaluate_model_on_mesh
        from src.utils.visualization import visualize_best_decomposition
        from src.envs.coacd_env import CoACDEnv
        
        # Get global best parameters from training
        global_best_params = CoACDEnv.get_global_best_params()
        
        if global_best_params:
            print(f"\n🏆 GLOBAL BEST PERFORMANCE SUMMARY:")
            print(f"   Hausdorff Distance: {global_best_params.get('hausdorff', 'N/A'):.6f}")
            print(f"   Runtime: {global_best_params.get('runtime', 'N/A'):.3f}s")
            print(f"   Total Vertices: {global_best_params.get('vertices', 'N/A')}")
            print(f"   Number of Parts: {global_best_params.get('num_parts', 'N/A')}")
            print(f"   Parameters: threshold={global_best_params.get('threshold', 'N/A'):.3f}, "
                  f"no_merge={global_best_params.get('no_merge', 'N/A')}, "
                  f"max_hull={global_best_params.get('max_hull', 'N/A')}")
            
            print("\nGenerating visualizations...")
            os.makedirs("visualizations", exist_ok=True)
            visualize_best_decomposition(MESH_TRAIN, global_best_params, "visualizations")
            print("✓ Visualizations saved to 'visualizations/' directory")
        else:
            print("No global best parameters found during training")
            
            # Fallback to model evaluation
            best_model = load_best_model(BEST_DIR)
            if best_model:
                print("Evaluating best model for visualization...")
                best_params = evaluate_model_on_mesh(best_model, MESH_TRAIN, n_episodes=3)
                
                if best_params:
                    print(f"\n🏆 BEST MODEL PERFORMANCE SUMMARY:")
                    print(f"   Hausdorff Distance: {best_params.get('hausdorff', 'N/A'):.6f}")
                    print(f"   Runtime: {best_params.get('runtime', 'N/A'):.3f}s")
                    print(f"   Total Vertices: {best_params.get('vertices', 'N/A')}")
                    print(f"   Number of Parts: {best_params.get('num_parts', 'N/A')}")
                    print(f"   Parameters: threshold={best_params.get('threshold', 'N/A'):.3f}, "
                          f"no_merge={best_params.get('no_merge', 'N/A')}, "
                          f"max_hull={best_params.get('max_hull', 'N/A')}")
                    
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