#!/usr/bin/env python3
"""
Utility functions for loading and evaluating trained models.
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from src.envs.coacd_env import CoACDEnv
from src.utils.visualization import visualize_best_decomposition


def load_best_model(model_dir: str):
    """Load the best model from the specified directory."""
    try:
        # Look for the best model file
        best_model_path = None
        for file in os.listdir(model_dir):
            if file.startswith("best_model") and file.endswith(".zip"):
                best_model_path = os.path.join(model_dir, file)
                break
        
        if best_model_path is None:
            print(f"No best model found in {model_dir}")
            return None
            
        print(f"Loading best model from: {best_model_path}")
        model = PPO.load(best_model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def evaluate_model_on_mesh(model, mesh_path: str, n_episodes: int = 3):
    """Evaluate a trained model on a specific mesh and return the best parameters."""
    try:
        # Create environment
        env = CoACDEnv(mesh_path=mesh_path)
        
        best_reward = float('-inf')
        best_params = None
        
        print(f"Evaluating model on {mesh_path} for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            episode_params = None
            
            for step in range(100):  # Max 100 steps per episode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                # Store parameters from this step
                if 'params' in info:
                    episode_params = info['params'].copy()
                    episode_params['hausdorff'] = info.get('H', float('inf'))
                    episode_params['runtime'] = info.get('T', float('inf'))
                    episode_params['vertices'] = info.get('V', 0)
                    episode_params['num_parts'] = info.get('num_parts', 0)
                
                if terminated or truncated:
                    break
            
            print(f"Episode {episode + 1}: Reward = {total_reward:.3f}")
            
            # Update best parameters if this episode had better reward
            if total_reward > best_reward and episode_params is not None:
                best_reward = total_reward
                best_params = episode_params
        
        env.close()
        return best_params
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None


def main():
    """Main function for standalone visualization."""
    mesh_path = "assets/bunny_simplified.obj"
    model_dir = "models/best"
    
    print("=== Model Evaluation and Visualization ===")
    
    # Try to load the best model
    model = load_best_model(model_dir)
    
    if model is not None:
        # Evaluate the model
        best_params = evaluate_model_on_mesh(model, mesh_path, n_episodes=3)
        
        if best_params:
            print(f"\n🏆 BEST MODEL PERFORMANCE:")
            print(f"   Hausdorff Distance: {best_params.get('hausdorff', 'N/A'):.6f}")
            print(f"   Runtime: {best_params.get('runtime', 'N/A'):.3f}s")
            print(f"   Total Vertices: {best_params.get('vertices', 'N/A')}")
            print(f"   Number of Parts: {best_params.get('num_parts', 'N/A')}")
            print(f"   Parameters: threshold={best_params.get('threshold', 'N/A'):.3f}, "
                  f"no_merge={best_params.get('no_merge', 'N/A')}, "
                  f"max_hull={best_params.get('max_hull', 'N/A')}")
            
            # Create visualization
            print("\nGenerating visualization...")
            os.makedirs("visualizations", exist_ok=True)
            visualize_best_decomposition(mesh_path, best_params, "visualizations")
            print("✓ Visualization saved to 'visualizations/' directory")
        else:
            print("No valid parameters found during evaluation")
    else:
        print("Could not load model. Make sure you have trained a model first.")
        
        # Fallback to global best parameters
        global_best_params = CoACDEnv.get_global_best_params()
        if global_best_params:
            print(f"\nUsing global best parameters from training:")
            print(f"   Hausdorff Distance: {global_best_params.get('hausdorff', 'N/A'):.6f}")
            print(f"   Runtime: {global_best_params.get('runtime', 'N/A'):.3f}s")
            print(f"   Total Vertices: {global_best_params.get('vertices', 'N/A')}")
            print(f"   Number of Parts: {global_best_params.get('num_parts', 'N/A')}")
            
            print("\nGenerating visualization...")
            os.makedirs("visualizations", exist_ok=True)
            visualize_best_decomposition(mesh_path, global_best_params, "visualizations")
            print("✓ Visualization saved to 'visualizations/' directory")
        else:
            print("No global best parameters found. Please run training first.")


if __name__ == "__main__":
    main()
