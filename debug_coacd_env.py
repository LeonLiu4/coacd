#!/usr/bin/env python3
"""
Debug version of CoACD environment that disables multiprocessing for debugging
"""
import os, time, gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register
import numpy as np
import torch
import trimesh
import coacd
import json
from gymnasium import spaces
from src.utils.geometry import hausdorff, sample_points, sample_surface_points_from_parts_fast

# Global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tic(name):
    """Context manager for timing"""
    class Timer:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            self.end = time.time()
            print(f"{self.name}: {self.end - self.start:.3f}s")
    return Timer(name)

class DebugCoACDEnv(gym.Env):
    """Debug version of CoACD environment without multiprocessing"""
    
    metadata = {"render_modes": []}

    def __init__(self, mesh_path: str, npts: int = 4096, baseline_file: str = "baseline_metrics.json"):
        super().__init__()
        self.mesh_path = mesh_path
        self.npts = npts

        self.mesh = trimesh.load_mesh(mesh_path, process=False)
        self._coacd_mesh = coacd.Mesh(
            self.mesh.vertices.astype(np.float32),
            self.mesh.faces.astype(np.int64),
        )

        self.baseline_metrics = self._load_baseline_metrics(baseline_file)
        
        self.reward_coefficients = {
            'hausdorff': 15.0,
            'runtime': 0.5,
            'vertices': 0.002,
            'num_parts': 0.2
        }

        self.observation_space = spaces.Box(-1.0, 1.0, (npts, 3), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

        self.best_params = None
        self._sample_fixed_eval_points()

    def _load_baseline_metrics(self, baseline_file: str):
        """Load baseline metrics from JSON file."""
        try:
            if os.path.exists(baseline_file):
                with open(baseline_file, 'r') as f:
                    metrics = json.load(f)
                print(f"Loaded baseline metrics from {baseline_file}")
                return metrics
            else:
                print(f"Warning: Baseline file {baseline_file} not found. Using fallback values.")
                return {
                    'hausdorff_distance': 0.0036803423427045345,
                    'runtime': 5.734677076339722,
                    'total_vertices': 1592,
                    'num_parts': 14
                }
        except Exception as e:
            print(f"Error loading baseline metrics: {e}")
            return {
                'hausdorff_distance': 0.0036803423427045345,
                'runtime': 5.734677076339722,
                'total_vertices': 1592,
                'num_parts': 14
            }

    def _sample_fixed_eval_points(self, seed: int = 42):
        """Pre-sample fixed evaluation points for consistent Hausdorff calculation"""
        np.random.seed(seed)
        pts = sample_points(self.mesh, self.npts)
        self.eval_src_pts = torch.as_tensor(pts, dtype=torch.float32)[None].to(device)

    def _create_observation(self, mesh):
        """Create normalized observation by sampling points from given mesh"""
        pts = sample_points(mesh, self.npts, seed=42).astype(np.float32)
        # Normalize to [-1, 1] range
        c = pts.mean(0, keepdims=True)
        s = np.linalg.norm(pts - c, axis=1).max() + 1e-8
        return ((pts - c) / s).astype(np.float32)

    def _hausdorff_vs_fixed(self, dec_mesh, parts=None):
        """Compute Hausdorff distance against fixed evaluation points"""
        print("DEBUG: _hausdorff_vs_fixed called - this should hit your breakpoint!")
        if parts is not None:
            # Use fast depth-based sampling for training with 25 camera angles
            dec_pts = sample_surface_points_from_parts_fast(parts, self.npts, seed=42, num_angles=25).astype(np.float32)
        else:
            # Use regular sampling for single meshes
            dec_pts = sample_points(dec_mesh, self.npts, seed=42).astype(np.float32)
        dec_pts = torch.from_numpy(dec_pts)[None].to(device)
        return hausdorff(self.eval_src_pts, dec_pts)

    def _calculate_comparative_reward(self, hausdorff_dist, runtime, vertices, num_parts):
        """Calculate reward based on comparison with baseline metrics"""
        baseline = self.baseline_metrics
        relaxation_factor = 1.5
        
        hausdorff_better = hausdorff_dist < baseline['hausdorff_distance'] * relaxation_factor
        runtime_better = runtime < baseline['runtime'] * relaxation_factor
        vertices_better = vertices < baseline['total_vertices'] * relaxation_factor
        parts_better = num_parts <= baseline['num_parts'] * relaxation_factor
        
        hausdorff_improvement = (baseline['hausdorff_distance'] * relaxation_factor - hausdorff_dist) / (baseline['hausdorff_distance'] * relaxation_factor)
        runtime_improvement = (baseline['runtime'] * relaxation_factor - runtime) / (baseline['runtime'] * relaxation_factor)
        vertices_improvement = (baseline['total_vertices'] * relaxation_factor - vertices) / (baseline['total_vertices'] * relaxation_factor)
        parts_improvement = max(0, (baseline['num_parts'] * relaxation_factor - num_parts) / (baseline['num_parts'] * relaxation_factor))
        
        if hausdorff_better and runtime_better and vertices_better and parts_better:
            reward = (
                self.reward_coefficients['hausdorff'] * hausdorff_improvement +
                self.reward_coefficients['runtime'] * runtime_improvement +
                self.reward_coefficients['vertices'] * vertices_improvement +
                self.reward_coefficients['num_parts'] * parts_improvement
            )
            return max(0.1, reward)
        else:
            worse_count = sum([not hausdorff_better, not runtime_better, not vertices_better, not parts_better])
            return -worse_count * 0.5

    def reset(self, *, seed=None, **kwargs):
        super().reset(seed=seed)
        obs = self._create_observation(self.mesh)
        return obs, {}

    def step(self, action: np.ndarray):
        print("DEBUG: step called - processing action...")
        
        with tic("map-action"):
            raw = action[0] * 0.5 + 0.5
            threshold = float(max(0.01, min(0.01 + raw * 0.99, 1.0)))
            no_merge = bool(action[1] > 0)
            max_hull = int(10 + (action[2] * 0.5 + 0.5) * 90)

        print(f"DEBUG: Running CoACD with threshold={threshold}, no_merge={no_merge}, max_hull={max_hull}")
        
        # DEBUG VERSION: Run CoACD directly without multiprocessing
        t0 = time.time()
        try:
            parts = coacd.run_coacd(
                self._coacd_mesh,
                threshold=threshold,
                merge=not no_merge,
                max_convex_hull=max_hull,
                seed=42,
            )
            runtime = time.time() - t0
            print(f"DEBUG: CoACD completed in {runtime:.3f}s with {len(parts)} parts")
            
            # Calculate metrics
            total_vertices = sum(len(v) for v, _ in parts)
            hausdorff_dist = self._hausdorff_vs_fixed(None, parts=parts)
            
            reward = self._calculate_comparative_reward(hausdorff_dist, runtime, total_vertices, len(parts))
            
            info = {
                "T": runtime,
                "H": hausdorff_dist,
                "V": total_vertices,
                "num_parts": len(parts),
                "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
                "success": True,
                "improvement": reward > 0,
                "error_type": None
            }
            
            # Create observation from the result
            obs = self._create_observation(self.mesh)  # For now, use original mesh
            
        except Exception as e:
            print(f"DEBUG: CoACD failed with error: {e}")
            runtime = time.time() - t0
            reward = -10.0
            info = {
                "T": runtime,
                "H": float('inf'),
                "V": 0,
                "num_parts": 0,
                "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
                "success": False,
                "improvement": False,
                "error_type": str(e)
            }
            obs = self._create_observation(self.mesh)

        return obs, reward, False, False, info

def test_debug_env():
    """Test the debug environment"""
    print("Testing debug CoACD environment...")
    
    # Register the debug environment
    register(id="DebugCoACD-v0", entry_point="debug_coacd_env:DebugCoACDEnv")
    
    # Create environment
    env = gym.make("DebugCoACD-v0", mesh_path="assets/bunny_simplified.obj")
    
    # Test one step
    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    
    # Take a random action
    action = np.random.uniform(-1, 1, 3)
    print(f"Taking action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step result - reward: {reward}, info: {info}")
    
    env.close()
    print("Debug environment test completed!")

if __name__ == "__main__":
    test_debug_env()
