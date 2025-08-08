#!/usr/bin/env python3
"""
Demo script showing how to use the CoACD visualization features.
This script demonstrates both the environment rendering and standalone visualization.
"""

import os
import numpy as np
from src.envs.coacd_env import CoACDEnv
from src.utils.visualization import visualize_best_decomposition
import trimesh


def demo_environment_rendering():
    """Demo of rendering within the environment."""
    print("=== Environment Rendering Demo ===")
    
    # Create environment
    mesh_path = "assets/bunny_simplified.obj"
    env = CoACDEnv(mesh_path, npts=1024)
    
    # Run a few random steps to get some results
    print("Running a few random steps...")
    obs, _ = env.reset()
    
    for step in range(10):
        # Generate random action
        action = np.random.uniform(-1, 1, 3)
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step {step + 1}: H={info.get('H', 'N/A'):.4f}, "
              f"Reward={reward:.3f}, Success={info.get('success', False)}")
        
        if done or truncated:
            break
    
    # Render the best result found
    print("\nRendering best result...")
    env.render()
    
    # Get best parameters
    best_params = env.get_best_params()
    if best_params:
        print(f"Best parameters found: {best_params}")
    
    env.close()


def demo_standalone_visualization():
    """Demo of standalone visualization with custom parameters."""
    print("\n=== Interactive Visualization Demo ===")
    
    mesh_path = "assets/bunny_simplified.obj"
    
    # Test different parameter sets
    test_params = [
        {
            "threshold": 0.05,
            "no_merge": False,
            "max_hull": 10,
            "name": "Conservative"
        },
        {
            "threshold": 0.1,
            "no_merge": True,
            "max_hull": 5,
            "name": "Aggressive"
        },
        {
            "threshold": 0.02,
            "no_merge": False,
            "max_hull": 20,
            "name": "Fine-grained"
        }
    ]
    
    for i, params in enumerate(test_params):
        print(f"\nTesting {params['name']} parameters...")
        # Remove the 'name' key for visualization
        viz_params = {k: v for k, v in params.items() if k != 'name'}
        
        try:
            # Suppress CoACD output by redirecting stdout/stderr
            import sys
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                # Also redirect C++ stdout/stderr
                old_stdout_fd = os.dup(1)
                old_stderr_fd = os.dup(2)
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                try:
                    visualize_best_decomposition(
                        mesh_path=mesh_path,
                        best_params=viz_params,
                        save_dir=f"visualizations/demo_{i+1}"
                    )
                finally:
                    os.dup2(old_stdout_fd, 1)
                    os.dup2(old_stderr_fd, 2)
                    os.close(old_stdout_fd)
                    os.close(old_stderr_fd)
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
        except Exception as e:
            print(f"Error with {params['name']}: {e}")
            import traceback
            traceback.print_exc()





def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CoACD Visualization Demo")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Skip cleanup (keep generated files)")
    parser.add_argument("--cleanup", action="store_true", 
                       help="Clean up generated files after demo (default behavior)")
    
    args = parser.parse_args()
    
    print("CoACD Visualization Demo")
    print("=" * 50)
    
    # Check if mesh exists
    mesh_path = "assets/bunny_simplified.obj"
    if not os.path.exists(mesh_path):
        print(f"Mesh file {mesh_path} not found!")
        print("Please ensure the bunny mesh is available.")
        return
    
    # Create visualizations directory
    os.makedirs("visualizations", exist_ok=True)
    
    try:
        # Run demos
        demo_environment_rendering()
        demo_standalone_visualization()
        
        print("\n" + "=" * 50)
        print("Demo completed!")
        
        # Always cleanup unless explicitly disabled
        if not args.no_cleanup:
            from cleanup import cleanup_generated_files
            cleanup_generated_files()
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure all dependencies are installed (pyrender, matplotlib, etc.)")


if __name__ == "__main__":
    main()
