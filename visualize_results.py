#!/usr/bin/env python3
"""
Visualization script for CoACD RL results.
This script loads a trained model and visualizes the best decomposition found.
"""

import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from src.envs.coacd_env import CoACDEnv
from src.utils.visualization import visualize_best_decomposition
import coacd
import trimesh


def load_best_model(model_dir: str = "models/best"):
    """Load the best model from training."""
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found!")
        return None
    
    # Find the best model file
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    if not model_files:
        print(f"No model files found in {model_dir}")
        return None
    
    # Load the first model file (assuming it's the best one)
    model_path = os.path.join(model_dir, model_files[0])
    print(f"Loading model from: {model_path}")
    
    try:
        model = PPO.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def evaluate_model_on_mesh(model, mesh_path: str, n_episodes: int = 5):
    """Evaluate the model on a mesh and return the best parameters found."""
    env = CoACDEnv(mesh_path, npts=4096)  # Use same number of points as training
    
    best_params = None
    best_hausdorff = float('inf')
    
    print(f"Evaluating model on {mesh_path} for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 50:  # Limit steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            
            if info.get('success', False):
                print(f"Episode {episode + 1}: Success! H={info['H']:.4f}")
                break
        
        # Check if this episode had better results
        if env.get_best_params() is not None:
            current_h = env.get_best_params()['hausdorff']
            if current_h < best_hausdorff:
                best_hausdorff = current_h
                best_params = env.get_best_params().copy()
                print(f"New best! H={best_hausdorff:.4f}")
    
    env.close()
    return best_params


def visualize_custom_parameters(mesh_path: str, params: dict, save_dir: str = "visualizations"):
    """Visualize decomposition with custom parameters."""
    print(f"Visualizing with parameters: {params}")
    visualize_best_decomposition(mesh_path, params, save_dir)


def interactive_parameter_testing(mesh_path: str):
    """Interactive mode for testing different parameters."""
    print("\n=== Interactive Parameter Testing ===")
    print("Enter parameters to test (or 'quit' to exit):")
    
    while True:
        try:
            threshold = input("Threshold (0.01-1.0, default 0.05): ").strip()
            if threshold.lower() == 'quit':
                break
            threshold = float(threshold) if threshold else 0.05
            
            no_merge = input("No merge (y/n, default n): ").strip().lower()
            no_merge = no_merge == 'y' if no_merge else False
            
            max_hull = input("Max hull (1-100, default 10): ").strip()
            max_hull = int(max_hull) if max_hull else 10
            
            params = {
                "threshold": threshold,
                "no_merge": no_merge,
                "max_hull": max_hull
            }
            
            print(f"\nTesting parameters: {params}")
            visualize_best_decomposition(mesh_path, params, "visualizations")
            
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or interrupted. Exiting...")
            break


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize CoACD RL results")
    parser.add_argument("--mesh", type=str, default="assets/bunny_simplified.obj",
                       help="Path to input mesh")
    parser.add_argument("--model-dir", type=str, default="models/best",
                       help="Directory containing the best model")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to evaluate")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode for parameter testing")
    parser.add_argument("--custom-params", type=str,
                       help="Custom parameters as 'threshold,no_merge,max_hull'")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Skip cleanup (keep generated files)")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up generated files after visualization (default behavior)")
    
    args = parser.parse_args()
    
    # Check if mesh exists
    if not os.path.exists(args.mesh):
        print(f"Mesh file {args.mesh} not found!")
        return
    
    # Create visualizations directory
    os.makedirs("visualizations", exist_ok=True)
    
    if args.interactive:
        # Interactive mode
        interactive_parameter_testing(args.mesh)
        
    elif args.custom_params:
        # Custom parameters mode
        try:
            threshold, no_merge, max_hull = args.custom_params.split(',')
            params = {
                "threshold": float(threshold),
                "no_merge": no_merge.lower() == 'true',
                "max_hull": int(max_hull)
            }
            visualize_custom_parameters(args.mesh, params)
        except Exception as e:
            print(f"Error parsing custom parameters: {e}")
            print("Format should be: threshold,no_merge,max_hull")
            print("Example: 0.05,false,10")
    
    else:
        # Model evaluation mode
        model = load_best_model(args.model_dir)
        if model is None:
            print("Could not load model. Try interactive mode or custom parameters.")
            return
        
        best_params = evaluate_model_on_mesh(model, args.mesh, args.episodes)
        
        if best_params:
            print(f"\nBest parameters found:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
            
            print("\nGenerating visualizations...")
            visualize_best_decomposition(args.mesh, best_params, "visualizations")
        else:
            print("No valid parameters found during evaluation.")
    
    # Always cleanup unless explicitly disabled
    if not args.no_cleanup:
        from cleanup import cleanup_generated_files
        cleanup_generated_files()


if __name__ == "__main__":
    main()
