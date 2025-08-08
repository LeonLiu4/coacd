# src/envs/coacd_env.py
import time
import json
from contextlib import contextmanager
from multiprocessing import Process, Queue

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import trimesh
import os, sys, contextlib
from multiprocessing import Process, Queue
import coacd

from src.utils.geometry import sample_points, hausdorff


# ── timing helper ────────────────────────────────────────────────────
@contextmanager
def tic(label: str):
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1e3
    print(f"[TIMER] {label:<15} {dt:7.1f} ms")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── background worker that calls CoACD CLI ───────────────────────────
def _coacd_worker(queue: Queue, mesh, threshold: float, merge: bool, max_hull: int):
    """
    Run coacd.run_coacd in a subprocess and return the parts,
    while squelching **all** console output produced by the C++ library.
    """
    # 1) open the null device once
    with open(os.devnull, "w") as devnull:
        # 2) duplicate its fd over stdout (1) and stderr (2)
        os.dup2(devnull.fileno(), 1)   # C / C++ stdout
        os.dup2(devnull.fileno(), 2)   # C / C++ stderr
        # 3) optional: also silence Python-level streams
        sys.stdout = devnull
        sys.stderr = devnull

        # 4) call CoACD
        parts = coacd.run_coacd(
            mesh,
            threshold       = threshold,
            merge           = merge,
            max_convex_hull = max_hull,
        )

    # 5) hand results back to parent
    queue.put(parts)

# ── main Gym environment ─────────────────────────────────────────────
class CoACDEnv(gym.Env):
    """
    Gym environment wrapping a single CoACD call as an RL step.
    """

    metadata = {"render_modes": []}

    def __init__(self, mesh_path: str, npts: int = 4096, baseline_file: str = "baseline_metrics.json"):
        super().__init__()
        self.mesh_path = mesh_path
        self.npts = npts

        # load & wrap mesh for CoACD
        self.mesh = trimesh.load_mesh(mesh_path, process=False)
        self._coacd_mesh = coacd.Mesh(
            self.mesh.vertices.astype(np.float32),
            self.mesh.faces.astype(np.int64),
        )

        # Load baseline metrics for comparison
        self.baseline_metrics = self._load_baseline_metrics(baseline_file)
        
        # Reward coefficients for different metrics
        self.reward_coefficients = {
            'hausdorff': 10.0,      # Higher weight for quality
            'runtime': 1.0,         # Time efficiency
            'vertices': 0.001,      # Complexity (lower is better)
            'num_parts': 0.1        # Number of parts (fewer is often better)
        }

        # gym spaces
        self.observation_space = spaces.Box(-1.0, 1.0, (npts, 3), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (3,),     np.float32)

        # tracking variables
        self.best_H = None
        self.best_params = None  # Track best parameters found
        self.step_count = 0
        self.success_threshold = 0.01  # Hausdorff distance threshold for success

        # initial observation
        self._sample_obs()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
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

    def _sample_obs(self, seed: int | None = None):
        """Sample points from the original mesh"""
        if seed is not None:
            np.random.seed(seed)
        pts = sample_points(self.mesh, self.npts)
        self.src_pts = torch.as_tensor(pts, dtype=torch.float32)[None]

    def _create_observation(self, mesh):
        """Create observation by sampling points from given mesh"""
        pts = sample_points(mesh, self.npts)
        return pts

    def _calculate_comparative_reward(self, hausdorff_dist, runtime, vertices, num_parts):
        """
        Calculate reward based on comparison with baseline metrics.
        If all metrics are better than baseline: reward = coefficient * delta
        If any metric is worse than baseline: reward = -1
        """
        baseline = self.baseline_metrics
        
        # Check if each metric is better than baseline (lower is better for all these metrics)
        hausdorff_better = hausdorff_dist < baseline['hausdorff_distance']
        runtime_better = runtime < baseline['runtime']
        vertices_better = vertices < baseline['total_vertices']
        parts_better = num_parts <= baseline['num_parts']  # Allow equal number of parts
        
        # If any metric is worse, return -1
        if not (hausdorff_better and runtime_better and vertices_better and parts_better):
            return -1.0
        
        # All metrics are better - calculate reward based on deltas
        hausdorff_delta = baseline['hausdorff_distance'] - hausdorff_dist
        runtime_delta = baseline['runtime'] - runtime
        vertices_delta = baseline['total_vertices'] - vertices
        parts_delta = max(0, baseline['num_parts'] - num_parts)  # Only reward if fewer parts
        
        # Calculate weighted reward
        reward = (
            self.reward_coefficients['hausdorff'] * hausdorff_delta +
            self.reward_coefficients['runtime'] * runtime_delta +
            self.reward_coefficients['vertices'] * vertices_delta +
            self.reward_coefficients['num_parts'] * parts_delta
        )
        
        return reward

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, **kwargs):
        super().reset(seed=seed)
        self._sample_obs(seed)
        
        # Reset tracking variables
        self.best_H = None
        self.step_count = 0
        
        return self.src_pts.squeeze(0).cpu().numpy(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        
        # 1) map normalized action → real parameters
        with tic("map-action"):
            raw       = action[0] * 0.5 + 0.5
            threshold = float(max(0.01, min(0.01 + raw * 0.99, 1.0)))
            no_merge  = bool(action[1] > 0)
            max_hull  = int(10 + (action[2] * 0.5 + 0.5) * 90)

        # Debug prints
        if self.step_count <= 5 or self.step_count % 10 == 0:
            print(f"Step {self.step_count}: Action={action}, Threshold={threshold:.3f}, No_merge={no_merge}, Max_hull={max_hull}")

        # 2) run CoACD in a separate process with timeout
        limit_sec = 10  # Increased timeout slightly
        queue     = Queue()
        proc      = Process(
            target=_coacd_worker,
            args=(queue, self._coacd_mesh, threshold, not no_merge, max_hull),
        )
        t0 = time.time()
        proc.start()
        proc.join(timeout=limit_sec)

        terminated = False
        truncated  = False

        if proc.is_alive():
            # timed out → penalty, mark as truncated
            proc.terminate()
            proc.join()
            reward     = -10.0  # More reasonable timeout penalty
            truncated  = True
            info       = {
                "timeout": True,
                "T": limit_sec,
                "H": float('inf'),
                "V": 0,
                "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
                "success": False,
                "improvement": False,
            }
            # Return original mesh points as observation
            obs = self.src_pts.squeeze(0).cpu().numpy()
            queue.close(); queue.join_thread()
            
            if self.step_count <= 5 or self.step_count % 10 == 0:
                print(f"  → TIMEOUT! Reward: {reward:.3f}")
            
            return obs, reward, terminated, truncated, info

        # normal completion
        runtime = time.time() - t0
        parts   = queue.get_nowait()
        queue.close(); queue.join_thread()

        # Check if decomposition failed (empty parts)
        if not parts or len(parts) == 0:
            reward = -5.0
            info = {
                "timeout": False,
                "T": runtime,
                "H": float('inf'),
                "V": 0,
                "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
                "success": False,
                "improvement": False,
                "failed_decomposition": True,
            }
            obs = self.src_pts.squeeze(0).cpu().numpy()
            return obs, reward, terminated, truncated, info

        # 3) stitch all convex‐hull parts into a single mesh
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
                "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
                "success": False,
                "improvement": False,
                "mesh_creation_failed": True,
            }
            obs = self.src_pts.squeeze(0).cpu().numpy()
            return obs, reward, terminated, truncated, info

        # 4) sample interior points
        with tic("sample pts"):
            vol_in  = sample_points(self.mesh,    self.npts)
            vol_out = sample_points(dec_mesh,     self.npts)

        # 5) compute Hausdorff distance
        with tic("Hausdorff"):
            H = hausdorff(
                torch.as_tensor(vol_in )[None].to(device),
                torch.as_tensor(vol_out)[None].to(device),
            )

        # 6) assemble reward using comparative baseline method
        with tic("assemble reward"):
            V_raw = dec_mesh.vertices.shape[0]
            num_parts = len(parts)
            
            # Calculate comparative reward
            reward = self._calculate_comparative_reward(H, runtime, V_raw, num_parts)
            
            # Track improvement
            improvement = False
            if self.best_H is None or H < self.best_H:
                improvement = True
                self.best_H = H
                # Store best parameters
                self.best_params = {
                    "threshold": threshold,
                    "no_merge": no_merge,
                    "max_hull": max_hull,
                    "hausdorff": float(H),
                    "vertices": int(V_raw),
                    "runtime": float(runtime),
                    "num_parts": num_parts
                }
            
            # Success bonus - if significantly better than baseline
            success = (H < self.baseline_metrics['hausdorff_distance'] * 0.5 and 
                      runtime < self.baseline_metrics['runtime'] * 0.8)
            if success:
                reward += 10.0  # big bonus for achieving significant improvement
                terminated = True   # end episode on success

        # 7) Create observation from the decomposed mesh (key change!)
        obs = self._create_observation(dec_mesh)

        info = {
            "H": float(H),
            "V": int(V_raw),
            "T": runtime,
            "num_parts": num_parts,
            "timeout": False,
            "success": success,
            "improvement": improvement,
            "best_H": float(self.best_H) if self.best_H is not None else float('inf'),
            "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
        }

        # Debug prints
        if self.step_count <= 5 or self.step_count % 10 == 0:
            print(f"  → H: {H:.4f}, V: {V_raw}, T: {runtime:.3f}s, Parts: {num_parts}, Reward: {reward:.3f}, Success: {success}, Improvement: {improvement}")

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if mode == "human":
            if hasattr(self, 'best_params') and self.best_params is not None:
                from src.utils.visualization import visualize_best_decomposition
                # Suppress CoACD output during rendering
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