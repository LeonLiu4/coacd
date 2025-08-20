import time
import json
from contextlib import contextmanager
import os
import sys
from queue import Empty
import multiprocessing as mp
from multiprocessing import Queue as MPQueue  # <-- added

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import trimesh
import coacd

from src.utils.geometry import sample_points, hausdorff, sample_surface_points_from_parts, sample_surface_points_from_parts_fast

# ---------- Timing helper ----------
@contextmanager
def tic(label: str):
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1e3
    print(f"[TIMER] {label:<15} {dt:7.1f} ms")

# ---------- Device selection ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Globals ----------
_global_best_H = float('inf')
_global_best_params = None
max_H = 10.0  # Maximum Hausdorff distance instead of infinity

# ---------- Worker configuration ----------
USE_GPU_IN_WORKER = False     # Most env workers should be CPU-only
WORKER_QUEUE_TIMEOUT = 30.0   # Seconds to wait for results from worker
WORKER_STDIO_SILENT = True    # Squelch worker stdout/stderr (noisy libraries)

def _squelch_stdio():
    """Redirect stdout/stderr to /dev/null (POSIX)."""
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    sys.stdout = devnull
    sys.stderr = devnull
    return devnull

def _coacd_worker(result_q: MPQueue, err_q: MPQueue,   # <-- changed
                  mesh, threshold: float, merge: bool, max_hull: int,
                  use_gpu: bool, silence: bool):
    """Run coacd.run_coacd in a subprocess and return parts, capturing exceptions."""
    devnull = None
    try:
        if not use_gpu:
            # Ensure this process doesn't touch a CUDA device unless explicitly requested
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if silence:
            devnull = _squelch_stdio()

        # Run CoACD decomposition
        parts = coacd.run_coacd(
            mesh,
            threshold=threshold,
            merge=merge,
            max_convex_hull=max_hull,
            seed=42,  # Fixed seed for deterministic results
        )

        # Send back the result
        result_q.put(parts)

    except Exception as e:
        import traceback
        err_q.put(traceback.format_exc())
    finally:
        if devnull is not None:
            try:
                devnull.close()
            except Exception:
                pass

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

        # Reward coefficients (no baseline deltas, straight weighted sum of metric values)
        self.reward_coefficients = {
            'hausdorff': 100.0,  # High weight for hausdorff (small values, big impact)
            'runtime': 0.1,      # Low weight for runtime (seconds scale)
            'vertices': 0.0005,  # Very low weight for vertices (large numbers)
            'num_parts': 0.05    # Low weight for parts (small numbers)
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

    def get_baseline_summary(self):
        """Return a summary of the current baseline metrics."""
        baseline = self.baseline_metrics
        return {
            'hausdorff': baseline['hausdorff_distance'],
            'runtime': baseline['runtime'],
            'vertices': baseline['total_vertices'],
            'num_parts': baseline['num_parts']
        }

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

    def _hausdorff_vs_fixed(self, dec_mesh, parts=None):
        """Compute Hausdorff distance against fixed evaluation points"""
        if parts is not None:
            # Use fast depth-based sampling for training with 25 camera angles
            dec_pts = sample_surface_points_from_parts_fast(parts, self.npts, seed=42).astype(np.float32)
        else:
            # Use regular sampling for single meshes
            dec_pts = sample_points(dec_mesh, self.npts, seed=42).astype(np.float32)
        dec_pts = torch.from_numpy(dec_pts)[None].to(device)

        # Reset PyRender context to prevent memory leaks
        from src.utils.geometry import reset_pyrender_context
        reset_pyrender_context()

        return hausdorff(self.eval_src_pts, dec_pts)

    def _calculate_comparative_reward(self, hausdorff_dist, runtime, vertices, num_parts):
        """Calculate reward based on actual metric values (no baseline comparison)."""
        reward = (
            -self.reward_coefficients['hausdorff'] * hausdorff_dist +
            -self.reward_coefficients['runtime'] * runtime +
            -self.reward_coefficients['vertices'] * vertices +
            -self.reward_coefficients['num_parts'] * num_parts
        )
        return reward

    def reset(self, *, seed=None, **kwargs):
        super().reset(seed=seed)
        # Return normalized observation with consistent sampling
        obs = self._create_observation(self.mesh)
        return obs, {}

    def _run_coacd_subprocess(self, threshold: float, no_merge: bool, max_hull: int, time_budget: float):
        """
        Launch coacd in a child process and wait for result with robust timeout + error handling.

        Returns:
            (status, parts, runtime, error_type, timeout_flag, worker_exitcode, worker_trace)
            - status: "ok" | "error"
            - parts: list[(verts, faces)] or None
            - runtime: float seconds
            - error_type: str or None
            - timeout_flag: bool
            - worker_exitcode: int | None
            - worker_trace: str | None
        """
        ctx = mp.get_context("spawn")
        result_q = ctx.Queue()
        err_q = ctx.Queue()

        # Build worker
        proc = ctx.Process(
            target=_coacd_worker,
            args=(result_q, err_q, self._coacd_mesh, threshold, not no_merge, max_hull, USE_GPU_IN_WORKER, WORKER_STDIO_SILENT),
            name="coacd-worker",
        )

        t0 = time.time()
        timed_out = False
        worker_trace = None
        parts = None
        error_type = None
        exitcode = None

        try:
            proc.start()

            # Wait up to time_budget for a result from the worker
            try:
                parts = result_q.get(timeout=time_budget)
            except Empty:
                # No result yet: if still alive, treat as timeout; if dead, fetch error
                if proc.is_alive():
                    timed_out = True
                    error_type = "timeout"
                else:
                    # Worker exited but didn't post a result; try to read its traceback
                    try:
                        worker_trace = err_q.get_nowait()
                        error_type = "worker_exception"
                    except Empty:
                        error_type = "no_result"

            # If we timed out, kill the worker
            if timed_out and proc.is_alive():
                proc.terminate()

            # Small grace window for late queue posts
            if parts is None and worker_trace is None and not timed_out:
                try:
                    parts = result_q.get(timeout=min(WORKER_QUEUE_TIMEOUT, 2.0))
                except Empty:
                    try:
                        worker_trace = err_q.get_nowait()
                        error_type = "worker_exception"
                    except Empty:
                        error_type = error_type or "no_result"

            proc.join()
            exitcode = proc.exitcode

        finally:
            # Clean up queues to avoid hanging background threads
            try:
                result_q.close()
                result_q.join_thread()
            except Exception:
                pass
            try:
                err_q.close()
                err_q.join_thread()
            except Exception:
                pass

        runtime = time.time() - t0

        # If worker died with an exception, prefer that signal
        if worker_trace is not None:
            return "error", None, runtime, error_type, timed_out, exitcode, worker_trace

        # If we have parts, all good
        if parts is not None:
            return "ok", parts, runtime, None, timed_out, exitcode, None

        # Otherwise an error path
        return "error", None, runtime, error_type or ("timeout" if timed_out else "unknown"), timed_out, exitcode, worker_trace

    def step(self, action: np.ndarray):
        with tic("map-action"):
            raw = action[0] * 0.5 + 0.5
            threshold = float(max(0.01, min(0.01 + raw * 0.99, 1.0)))
            no_merge = bool(action[1] > 0)
            max_hull = int(10 + (action[2] * 0.5 + 0.5) * 90)

        # Defaults
        terminated = True
        truncated = False
        success = False
        improvement = False
        error_type = None
        timeout = False
        H = max_H
        V_raw = 0
        num_parts = 0
        obs = self._create_observation(self.mesh)
        reward = -5.0

        # Time budget for worker is the baseline runtime
        limit_sec = float(self.baseline_metrics['runtime'])

        #print("max_hull", max_hull)
        #print("threshold", threshold)
        #print("no_merge", no_merge)

        # --------- Run worker robustly ----------
        status, parts, runtime, error_type, timeout, exitcode, worker_trace = self._run_coacd_subprocess(
            threshold=threshold,
            no_merge=no_merge,
            max_hull=max_hull,
            time_budget=limit_sec
        )

        if status == "error":
            # Timeout: harsher penalty, mark truncated
            if error_type == "timeout":
                reward = -10.0
                truncated = True
            else:
                reward = -7.5
                terminated = True

            if worker_trace:
                print("\n[CoACD Worker Error]\n" + worker_trace)
        else:
            # Validate parts
            if not parts or len(parts) == 0:
                print(f"Warning: CoACD returned 0 parts for threshold={threshold}, no_merge={no_merge}, max_hull={max_hull}")
                error_type = "failed_decomposition"
            else:
                # Validate each part individually
                valid_parts = []
                for i, (verts, faces) in enumerate(parts):
                    if len(verts) == 0 or len(faces) == 0:
                        print(f"Warning: Part {i} is empty (verts={len(verts)}, faces={len(faces)})")
                        continue
                    if np.any(np.isnan(verts)) or np.any(np.isinf(verts)):
                        print(f"Warning: Part {i} contains NaN/Inf vertices")
                        continue
                    valid_parts.append((verts, faces))

                if len(valid_parts) == 0:
                    print(f"Warning: No valid parts after validation for threshold={threshold}, no_merge={no_merge}, max_hull={max_hull}")
                    error_type = "no_valid_parts"
                else:
                    # Create decomposed mesh
                    try:
                        verts_list, faces_list = zip(*valid_parts)
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
                        error_type = "mesh_creation_failed"
                    else:
                        # Calculate Hausdorff and metrics
                        with tic("Hausdorff"):
                            H = self._hausdorff_vs_fixed(dec_mesh, valid_parts)

                        V_raw = dec_mesh.vertices.shape[0]
                        num_parts = len(parts)

                        # Calculate reward and check for improvement
                        reward = self._calculate_comparative_reward(H, runtime, V_raw, num_parts)

                        # Store parameters for this step
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
                        if H < _global_best_H:
                            improvement = True
                            old_best_H = _global_best_H
                            _global_best_H = H
                            _global_best_params = self.best_params.copy()
                            print(f"\n🏆 GLOBAL BEST IMPROVEMENT!")
                            print(f"   Hausdorff: {H:.6f} (previous best: {old_best_H:.6f})")
                            print(f"   Runtime: {runtime:.3f}s (baseline: {self.baseline_metrics['runtime']:.3f}s)")
                            print(f"   Vertices: {V_raw} (baseline: {self.baseline_metrics['total_vertices']})")
                            print(f"   Parts: {num_parts} (baseline: {self.baseline_metrics['num_parts']})")
                            print(f"   Parameters: threshold={threshold:.3f}, no_merge={no_merge}, max_hull={max_hull}")

                        # Check for success
                        success = (H < self.baseline_metrics['hausdorff_distance'] * 0.75 and
                                   runtime < self.baseline_metrics['runtime'] * 1.2)
                        if success:
                            reward += 10.0
                            terminated = True

                        # Create observation from decomposed mesh
                        obs = self._create_observation(dec_mesh)

        # Build final info dictionary
        info = {
            "H": float(H),
            "V": int(V_raw),
            "T": runtime if 'runtime' in locals() else None,
            "num_parts": num_parts,
            "timeout": timeout,
            "success": success,
            "improvement": improvement,
            "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
            "error_type": error_type,
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