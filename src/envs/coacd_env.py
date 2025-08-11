import time
import json
from contextlib import contextmanager
from multiprocessing import Process, Queue
import os
import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import trimesh
import coacd

from src.utils.geometry import sample_points, hausdorff


@contextmanager
def tic(label: str):
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1e3
    print(f"[TIMER] {label:<15} {dt:7.1f} ms")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global tracking for best performance across all environment instances
_global_best_H = float('inf')
_global_best_params = None


def _coacd_worker(queue: Queue, mesh, threshold: float, merge: bool, max_hull: int):
    """Run coacd.run_coacd in a subprocess and return the parts, while squelching all console output."""
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        sys.stdout = devnull
        sys.stderr = devnull

        parts = coacd.run_coacd(
            mesh,
            threshold=threshold,
            merge=merge,
            max_convex_hull=max_hull,
        )

    queue.put(parts)

class CoACDEnv(gym.Env):
    """Gym environment wrapping a single CoACD call as an RL step."""

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
            'hausdorff': 10.0,
            'runtime': 1.0,
            'vertices': 0.001,
            'num_parts': 0.1
        }

        self.observation_space = spaces.Box(-1.0, 1.0, (npts, 3), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

        self.best_params = None

        # Pre-sample fixed evaluation points for consistent Hausdorff calculation
        self._sample_fixed_eval_points()
    def _load_baseline_metrics(self, baseline_file: str):
        """Load baseline metrics from JSON file."""
        try:
            if os.path.exists(baseline_file):
                with open(baseline_file, 'r') as f:
                    metrics = json.load(f)
                print(f"Loaded baseline metrics from {baseline_file}")
                print(f"  Baseline Hausdorff: {metrics['hausdorff_distance']:.6f}")
                print(f"  Baseline Runtime: {metrics['runtime']:.3f}s")
                print(f"  Baseline Vertices: {metrics['total_vertices']}")
                print(f"  Baseline Parts: {metrics['num_parts']}")
                return metrics
            else:
                print(f"Warning: Baseline file {baseline_file} not found. Using fallback values.")
                return {
                    'hausdorff_distance': 0.068,
                    'runtime': 6.43,
                    'total_vertices': 1653,
                    'num_parts': 14
                }
        except Exception as e:
            print(f"Error loading baseline metrics: {e}")
            return {
                'hausdorff_distance': 0.068,
                'runtime': 6.43,
                'total_vertices': 1653,
                'num_parts': 14
            }

    def _sample_fixed_eval_points(self, seed: int = 42):
        """Pre-sample fixed evaluation points for consistent Hausdorff calculation"""
        np.random.seed(seed)
        pts = sample_points(self.mesh, self.npts)
        self.eval_src_pts = torch.as_tensor(pts, dtype=torch.float32)[None].to(device)

    def _sample_obs(self, seed: int | None = None):
        """Sample points from the original mesh for observation"""
        if seed is not None:
            np.random.seed(seed)
        pts = sample_points(self.mesh, self.npts)
        self.src_pts = torch.as_tensor(pts, dtype=torch.float32)[None]

    def _create_observation(self, mesh):
        """Create normalized observation by sampling points from given mesh"""
        pts = sample_points(mesh, self.npts, seed=42).astype(np.float32)
        # Normalize to [-1, 1] range
        c = pts.mean(0, keepdims=True)
        s = np.linalg.norm(pts - c, axis=1).max() + 1e-8
        return ((pts - c) / s).astype(np.float32)

    def _hausdorff_vs_fixed(self, dec_mesh):
        """Compute Hausdorff distance against fixed evaluation points"""
        dec_pts = sample_points(dec_mesh, self.npts, seed=42).astype(np.float32)
        dec_pts = torch.from_numpy(dec_pts)[None].to(device)
        return hausdorff(self.eval_src_pts, dec_pts)

    def _calculate_comparative_reward(self, hausdorff_dist, runtime, vertices, num_parts):
        """Calculate reward based on comparison with baseline metrics."""
        baseline = self.baseline_metrics
        
        hausdorff_better = hausdorff_dist < baseline['hausdorff_distance']
        runtime_better = runtime < baseline['runtime']
        vertices_better = vertices < baseline['total_vertices']
        parts_better = num_parts <= baseline['num_parts']
        
        if not (hausdorff_better and runtime_better and vertices_better and parts_better):
            return -1.0
        
        hausdorff_delta = baseline['hausdorff_distance'] - hausdorff_dist
        runtime_delta = baseline['runtime'] - runtime
        vertices_delta = baseline['total_vertices'] - vertices
        parts_delta = max(0, baseline['num_parts'] - num_parts)
        
        reward = (
            self.reward_coefficients['hausdorff'] * hausdorff_delta +
            self.reward_coefficients['runtime'] * runtime_delta +
            self.reward_coefficients['vertices'] * vertices_delta +
            self.reward_coefficients['num_parts'] * parts_delta
        )
        
        return reward
    def reset(self, *, seed=None, **kwargs):
        super().reset(seed=seed)
        self._sample_obs(seed)
        
        # Return normalized observation
        obs = self._create_observation(self.mesh)
        return obs, {}

    def step(self, action: np.ndarray):
        
        with tic("map-action"):
            raw = action[0] * 0.5 + 0.5
            threshold = float(max(0.01, min(0.01 + raw * 0.99, 1.0)))
            no_merge = bool(action[1] > 0)
            max_hull = int(10 + (action[2] * 0.5 + 0.5) * 90)

        limit_sec = self.baseline_metrics['runtime']
        queue = Queue()
        proc = Process(
            target=_coacd_worker,
            args=(queue, self._coacd_mesh, threshold, not no_merge, max_hull),
        )
        t0 = time.time()
        proc.start()
        proc.join(timeout=limit_sec)

        terminated = True
        truncated = False

        if proc.is_alive():
            proc.terminate()
            proc.join()
            reward = -10.0
            truncated = True
            info = {
                "timeout": True,
                "T": limit_sec,
                "H": float('inf'),
                "V": 0,
                "num_parts": 0,
                "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
                "success": False,
                "improvement": False,
                "error_type": "timeout"
            }
            obs = self._create_observation(self.mesh)
            queue.close()
            queue.join_thread()
            return obs, reward, terminated, truncated, info

        runtime = time.time() - t0
        parts = queue.get_nowait()
        queue.close()
        queue.join_thread()

        if not parts or len(parts) == 0:
            reward = -5.0
            info = {
                "timeout": False,
                "T": runtime,
                "H": float('inf'),
                "V": 0,
                "num_parts": 0,
                "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
                "success": False,
                "improvement": False,
                "error_type": "failed_decomposition"
            }
            obs = self._create_observation(self.mesh)
            return obs, reward, terminated, truncated, info

        try:
            verts_list, faces_list = zip(*parts)
            all_verts = np.vstack(verts_list)
            all_faces = []
            v_off = 0
            for verts, faces in zip(verts_list, faces_list):
                all_faces.append(faces + v_off)
                v_off += verts.shape[0]
            all_faces = np.vstack(all_faces)
            dec_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
        except Exception as e:
            print(f"Error creating decomposed mesh: {e}")
            reward = -5.0
            info = {
                "timeout": False,
                "T": runtime,
                "H": float('inf'),
                "V": 0,
                "num_parts": 0,
                "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
                "success": False,
                "improvement": False,
                "error_type": "mesh_creation_failed"
            }
            obs = self._create_observation(self.mesh)
            return obs, reward, terminated, truncated, info

        with tic("Hausdorff"):
            H = self._hausdorff_vs_fixed(dec_mesh)

        with tic("assemble reward"):
            V_raw = dec_mesh.vertices.shape[0]
            num_parts = len(parts)
            
            reward = self._calculate_comparative_reward(H, runtime, V_raw, num_parts)
            
            # Store parameters for this step (for potential visualization)
            self.best_params = {
                "threshold": threshold,
                "no_merge": no_merge,
                "max_hull": max_hull,
                "hausdorff": float(H),
                "vertices": int(V_raw),
                "runtime": float(runtime),
                "num_parts": num_parts
            }
            
            # Check for global improvement
            global _global_best_H, _global_best_params
            improvement = False
            
            if H < _global_best_H:
                improvement = True
                _global_best_H = H
                _global_best_params = self.best_params.copy()
                print(f"\n🏆 GLOBAL BEST IMPROVEMENT!")
                print(f"   Hausdorff: {H:.6f} (previous best: {_global_best_H:.6f})")
                print(f"   Runtime: {runtime:.3f}s (baseline: {self.baseline_metrics['runtime']:.3f}s)")
                print(f"   Vertices: {V_raw} (baseline: {self.baseline_metrics['total_vertices']})")
                print(f"   Parts: {num_parts} (baseline: {self.baseline_metrics['num_parts']})")
                print(f"   Parameters: threshold={threshold:.3f}, no_merge={no_merge}, max_hull={max_hull}")
            
            success = (H < self.baseline_metrics['hausdorff_distance'] * 0.5 and 
                      runtime < self.baseline_metrics['runtime'] * 0.8)
            if success:
                reward += 10.0
                terminated = True

        # Create normalized observation from decomposed mesh
        obs = self._create_observation(dec_mesh)

        info = {
            "H": float(H),
            "V": int(V_raw),
            "T": runtime,
            "num_parts": num_parts,
            "timeout": False,
            "success": success,
            "improvement": improvement,
            "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
            "error_type": None
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if mode == "human":
            if hasattr(self, 'best_params') and self.best_params is not None:
                from src.utils.visualization import visualize_best_decomposition
                with open(os.devnull, 'w') as devnull:
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = devnull
                    sys.stderr = devnull
                    old_stdout_fd = os.dup(1)
                    old_stderr_fd = os.dup(2)
                    os.dup2(devnull.fileno(), 1)
                    os.dup2(devnull.fileno(), 2)
                    try:
                        visualize_best_decomposition(
                            mesh_path=self.mesh_path,
                            best_params=self.best_params,
                            save_dir="visualizations"
                        )
                    finally:
                        os.dup2(old_stdout_fd, 1)
                        os.dup2(old_stderr_fd, 2)
                        os.close(old_stdout_fd)
                        os.close(old_stderr_fd)
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
            else:
                print("No best parameters available for rendering. Run some steps first.")
        return None

    def get_best_params(self):
        """Get the best parameters found so far."""
        return self.best_params
    
    @staticmethod
    def get_global_best_params():
        """Get the global best parameters across all environment instances."""
        global _global_best_params
        return _global_best_params
    
    @staticmethod
    def get_global_best_H():
        """Get the global best Hausdorff distance across all environment instances."""
        global _global_best_H
        return _global_best_H