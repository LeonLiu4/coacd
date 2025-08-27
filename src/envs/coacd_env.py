#!/usr/bin/env python3
import os
import sys
import time
import json
import ctypes
import multiprocessing as mp
from contextlib import contextmanager
from queue import Empty

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import trimesh
import coacd

# Optional: psutil for clean memory stats (highly recommended)
try:
    import psutil
except Exception:
    psutil = None

from src.utils.geometry import (
    sample_points,
    hausdorff,
    sample_surface_points_from_parts,
    sample_surface_points_from_parts_fast,
)

# ---------- Timing helper ----------
@contextmanager
def tic(label: str):
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1e3
    print(f"[TIMER] {label:<18} {dt:7.1f} ms")

# ---------- Device selection ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Globals ----------
_global_best_reward = float('-inf')
_global_best_params = None
max_H = 10.0  # Maximum Hausdorff distance instead of infinity

# ---------- Reward knobs (ONLY CHANGE YOU ASKED FOR) ----------
TIMEOUT_REWARD = -1.0   # make timeouts clearly the worst outcome
ERROR_REWARD   = -1.0   # generic hard error / invalid result

# ---------- Worker configuration ----------
USE_GPU_IN_WORKER = False     # Most env workers should be CPU-only
WORKER_STDIO_SILENT = True    # Squelch worker stdout/stderr (noisy libraries)
WORKER_QUEUE_TIMEOUT = 30.0   # Seconds to wait for late posts after result

# =========================
# Worker helpers
# =========================

def _squelch_stdio():
    """Redirect stdout/stderr to /dev/null (POSIX)."""
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    sys.stdout = devnull
    sys.stderr = devnull
    return devnull

def _coacd_worker_loop(cmd_q, res_q, verts, faces, use_gpu, silence):
    """
    Persistent worker:
      - Holds CoACD mesh in its own memory.
      - Receives small param dicts and returns 'parts' results.
    """
    devnull = None
    try:
        if not use_gpu:
            # Ensure this process doesn't touch a CUDA device unless explicitly requested
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if silence:
            devnull = _squelch_stdio()

        # ---- enforce contiguous arrays + dtypes ----
        verts_c = np.ascontiguousarray(verts, dtype=np.float32)
        faces_c = np.ascontiguousarray(faces, dtype=np.int32)   # int32 faces

        # Build once and keep in this process
        co_mesh = coacd.Mesh(verts_c, faces_c)

        # Basic hello so parent can confirm worker PID
        res_q.put({"ok": True, "msg": "worker_ready", "pid": os.getpid()})

        while True:
            msg = cmd_q.get()
            if not isinstance(msg, dict) or "cmd" not in msg:
                continue

            if msg["cmd"] == "stop":
                break

            if msg["cmd"] == "run":
                try:
                    parts = coacd.run_coacd(
                        co_mesh,
                        threshold=msg["threshold"],
                        max_convex_hull=-1,
                        preprocess_mode="auto",
                        preprocess_resolution=msg["prep_resolution"],
                        mcts_iterations=msg["mcts_iteration"],
                        mcts_max_depth=msg["mcts_depth"],
                        mcts_nodes=msg["mcts_node"],
                        resolution=msg["resolution"],
                        pca=msg["pca"],
                        merge=True,
                        decimate=False,
                        seed=msg.get("seed", 42),
                    )
                    # Send back parts. NOTE: This copies arrays to parent.
                    res_q.put({"ok": True, "parts": parts})
                except Exception as e:
                    import traceback
                    res_q.put({"ok": False, "trace": traceback.format_exc()})

            elif msg["cmd"] == "mem":
                # Return approximate RSS of worker
                rss_mb = None
                if psutil:
                    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                res_q.put({"ok": True, "rss_mb": rss_mb})

    finally:
        if devnull is not None:
            try:
                devnull.close()
            except Exception:
                pass

# =========================
# Environment
# =========================

class CoACDEnv(gym.Env):
    """Gym environment wrapping a single CoACD call as an RL step (persistent worker)."""
    metadata = {"render_modes": []}

    def __init__(self, mesh_path: str, npts: int = 40960, baseline_file: str = "baseline_metrics.json"):
        super().__init__()
        self.mesh_path = mesh_path
        self.npts = npts

        # ---- load & sanitize mesh; ensure contiguous + correct dtypes ----
        mesh = trimesh.load_mesh(mesh_path, process=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        if mesh.faces.shape[1] != 3:
            mesh = mesh.triangulate()

        verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)   # int32 faces

        # Keep original mesh for sampling/obs; keep sanitized arrays for worker
        self.mesh = mesh
        self._verts = verts
        self._faces = faces
        
        # Input point cloud
        self.obs = self._create_observation(mesh)

        # Baseline
        self.baseline_metrics = self._load_baseline_metrics(baseline_file)
        baseline = self.baseline_metrics
        self.baseline_runtime = baseline['runtime']
        self.runtime_limit = self.baseline_runtime * 2.0

        # Reward coefficients based on baseline values - hausdorff and runtime most important
        self.reward_coefficients = {
            'hausdorff': 1, #60.0,
            'runtime':   0, #0.03,
            'vertices':  0, #0.00015,
            'num_parts': 0, #0.015,
        }

        self.observation_space = spaces.Box(-1.0, 1.0, (npts, 3), np.float32)
        
        # Change action space
        self.action_space = spaces.MultiDiscrete([
            10, # threshold : [0.01, 1.0], multiple of 0.01
            9, # prep_resolution : [20, 100] integer, multiple of 10
            15, # mcts_iteration : [60, 200] integer, multiple of 10
            6, # mcts_depth : [2, 7] integer
            4, # mcts_node : [10, 40] integer, multiple of 10
            10, # resolution : [1000, 10000] integer, multiple of 1000
            2, # pca : [0, 1] boolean
        ])

        self.best_params = None

        # Pre-sample fixed evaluation points for consistent Hausdorff calculation
        self._sample_fixed_eval_points()

        # ---- persistent worker that owns the mesh ----
        self.ctx = mp.get_context("spawn")
        self._cmd_q = self.ctx.Queue()
        self._res_q = self.ctx.Queue()
        self._worker = self.ctx.Process(
            target=_coacd_worker_loop,
            args=(self._cmd_q, self._res_q, self._verts, self._faces, USE_GPU_IN_WORKER, WORKER_STDIO_SILENT),
            name="coacd-worker",
        )
        self._worker.start()

        # Wait for worker_ready
        self._worker_pid = None
        try:
            hello = self._res_q.get(timeout=10.0)
            if hello.get("ok") and hello.get("msg") == "worker_ready":
                self._worker_pid = int(hello.get("pid", 0)) or self._worker.pid
        except Empty:
            print("[WARN] Worker did not signal readiness; continuing.")

        # Worker PID for memory debug
        self._worker_pid = self._worker_pid

    # --------------- Memory debug helpers ---------------

    def _rss_mb(self, pid):
        """Get RSS in MB for a process ID, with fresh psutil handle."""
        if not pid or not psutil:
            return None
        try:
            proc = psutil.Process(pid)
            return proc.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None
        except Exception:
            return None

    def _print_mem_debug(self, label: str):
        """Print memory debug info with fresh process handles."""
        parent_pid = os.getpid()
        child_pid = self._worker_pid if hasattr(self, '_worker_pid') else None
        
        p_rss = self._rss_mb(parent_pid)
        c_rss = self._rss_mb(child_pid)
        
        if p_rss is not None and c_rss is not None:
            print(f"[MEM] {label:<18} parent_rss={p_rss:.1f} MB  worker_rss={c_rss:.1f} MB")
        else:
            print(f"[MEM] {label:<18} parent_rss={p_rss} MB  worker_rss={c_rss} MB")

    def _malloc_trim(self):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass  # non-glibc platforms

    # --------------- Baseline / sampling ---------------
    def _load_baseline_metrics(self, baseline_file: str):
        try:
            if os.path.exists(baseline_file):
                with open(baseline_file, 'r') as f:
                    metrics = json.load(f)
                print(f"Loaded baseline metrics from {baseline_file}")
                print(f"  Baseline Hausdorff: {metrics['hausdorff_distance']:.6f}")
                print(f"  Baseline Runtime:   {metrics['runtime']:.3f}s")
                print(f"  Baseline Vertices:  {metrics['total_vertices']}")
                print(f"  Baseline Parts:     {metrics['num_parts']}")
                return metrics
            else:
                print(f"Warning: Baseline file {baseline_file} not found. Using fallback values.")
        except Exception as e:
            print(f"Error loading baseline metrics: {e}")

        return {
            'hausdorff_distance': 0.002679404802620411,
            'runtime': 2.776,
            'total_vertices': 1475,
            'num_parts': 11
        }

    def _sample_fixed_eval_points(self, seed: int = 42):
        np.random.seed(seed)
        pts = sample_points(self.mesh, self.npts)
        self.eval_src_pts = torch.as_tensor(pts, dtype=torch.float32)[None].to(device)

    def get_baseline_summary(self):
        b = self.baseline_metrics
        return {'hausdorff': b['hausdorff_distance'], 'runtime': b['runtime'],
                'vertices': b['total_vertices'], 'num_parts': b['num_parts']}

    def _sample_obs(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        pts = sample_points(self.mesh, self.npts)
        self.src_pts = torch.as_tensor(pts, dtype=torch.float32)[None]

    def _create_observation(self, mesh):
        pts = sample_points(mesh, self.npts, seed=42).astype(np.float32)
        c = pts.mean(0, keepdims=True)
        s = np.linalg.norm(pts - c, axis=1).max() + 1e-8
        return ((pts - c) / s).astype(np.float32)

    def _hausdorff_vs_fixed(self, dec_mesh, parts=None):
        if parts is not None:
            dec_pts = sample_surface_points_from_parts_fast(parts, self.npts, seed=42).astype(np.float32)
        else:
            dec_pts = sample_points(dec_mesh, self.npts, seed=42).astype(np.float32)
        dec_pts = torch.from_numpy(dec_pts)[None].to(device)

        # Reset PyRender context to prevent memory leaks
        from src.utils.geometry import reset_pyrender_context
        reset_pyrender_context()

        return hausdorff(self.eval_src_pts, dec_pts)

    def _calculate_comparative_reward(self, hausdorff_dist, runtime, vertices, num_parts):
        # Set threshold to be smaller one between 3 * baseline and 10sec
        if runtime > self.runtime_limit:
            reward = -1.0
        else:
            r = runtime / (self.baseline_runtime + 1e-8)
            h = hausdorff_dist / (self.baseline_metrics['hausdorff_distance'] + 1e-8)
            alpha = (r - 1) * 8
            alpha = 1.0 / (1.0 + np.exp(-alpha)) # sigmoid
            runtime_reward = alpha * np.clip(2 - r, 0, 1)
            H_reward = (1 - alpha) * np.clip(1 - h, 0, 1)
            
            reward = runtime_reward + H_reward
            # reward = (
            #     np.exp(-hausdorff_dist * self.reward_coefficients['hausdorff']) + 
            #     np.exp(-runtime * self.reward_coefficients['runtime']) +
            #     np.exp(-vertices * self.reward_coefficients['vertices']) +
            #     np.exp(-num_parts * self.reward_coefficients['num_parts'])
            # )
        
        return reward

    # --------------- Worker management ---------------
    def _ensure_worker_alive(self):
        if self._worker is None or (not self._worker.is_alive()):
            print("[INFO] Respawning CoACD worker...")
            # close old queues
            for q in (self._cmd_q, self._res_q):
                try:
                    q.close(); q.join_thread()
                except Exception:
                    pass
            # rebuild queues and worker
            self._cmd_q = self.ctx.Queue()
            self._res_q = self.ctx.Queue()
            self._worker = self.ctx.Process(
                target=_coacd_worker_loop,
                args=(self._cmd_q, self._res_q, self._verts, self._faces, USE_GPU_IN_WORKER, WORKER_STDIO_SILENT),
                name="coacd-worker",
            )
            self._worker.start()
            # wait hello
            self._worker_pid = self._worker.pid
            try:
                hello = self._res_q.get(timeout=10.0)
                if hello.get("ok") and hello.get("msg") == "worker_ready":
                    self._worker_pid = int(hello.get("pid", 0)) or self._worker.pid
            except Empty:
                pass
                         # worker PID is already updated above

    def _run_coacd_via_worker(
        self, 
        threshold: float,
        prep_resolution: int, 
        mcts_iteration: int, 
        mcts_depth: int, 
        mcts_node: int, 
        resolution: int, 
        pca: bool, 
        time_budget: float
    ):
        """
        Send a 'run' command to the persistent worker and wait up to time_budget.
        Returns: (status, parts, runtime, error_type, timeout_flag, exitcode, trace)
        """
        self._ensure_worker_alive()

        msg = {
            "cmd": "run",
            "threshold": float(threshold),
            "prep_resolution": int(prep_resolution),
            "mcts_iteration": int(mcts_iteration),
            "mcts_depth": int(mcts_depth),
            "mcts_node": int(mcts_node),
            "resolution": int(resolution),
            "pca": bool(pca),
            "seed": 42,
        }

        t0 = time.time()
        timed_out = False
        parts = None
        err_type = None
        trace = None

        # Send command
        self._cmd_q.put(msg)

        try:
            reply = self._res_q.get(timeout=time_budget)
            if reply.get("ok"):
                parts = reply.get("parts", None)
            else:
                err_type = "worker_exception"
                trace = reply.get("trace")
        except Empty:
            timed_out = True
            err_type = "timeout"
            # Kill and respawn the stuck worker to keep system healthy
            if self._worker.is_alive():
                self._worker.terminate()
                self._worker.join(timeout=2)
            self._ensure_worker_alive()

        runtime = time.time() - t0
        exitcode = self._worker.exitcode if (self._worker is not None) else None

        if parts is not None:
            return "ok", parts, runtime, None, timed_out, exitcode, None
        else:
            return "error", None, runtime, err_type, timed_out, exitcode, trace

    # --------------- Gym API ---------------
    def reset(self, *, seed=None, **kwargs):
        super().reset(seed=seed)
        obs = self.obs          # Do not resample observation here
        return obs, {}

    def step(self, action: np.ndarray):
        # Debug: memory before step
        self._print_mem_debug("before step")

        with tic("map-action"):
            threshold = action[0] * 0.01 + 0.01          # [0.01, 1.0]
            prep_resolution = action[1] * 10 + 20        # [20, 100]
            mcts_iteration = action[2] * 10 + 60        # [60, 200]
            mcts_depth = action[3] * 1 + 2               # [2, 7]
            mcts_node = action[4] * 10 + 10              # [10, 40]
            resolution = action[5] * 1000 + 1000          # [1000, 10000]
            pca = bool(action[6])

        # (unchanged defaults)
        terminated = True
        truncated = False
        success = False
        improvement = False
        error_type = None
        timeout = False
        H = max_H
        V_raw = 0
        num_parts = 0
        obs = self.obs          # Do not resample observation here
        reward = -5.0

        limit_sec = self.runtime_limit

        # --------- Run on persistent worker ----------
        status, parts, runtime, error_type, timeout, exitcode, worker_trace = self._run_coacd_via_worker(
            threshold=threshold,
            prep_resolution=prep_resolution,
            mcts_iteration=mcts_iteration,
            mcts_depth=mcts_depth,
            mcts_node=mcts_node,
            resolution=resolution,
            pca=pca,
            time_budget=limit_sec,
        )

        if status == "error":
            # -------- CHANGED: make timeout worst outcome --------
            if error_type == "timeout":
                reward = TIMEOUT_REWARD
            else:
                reward = ERROR_REWARD
            if worker_trace:
                print("\n[CoACD Worker Error]\n" + worker_trace)
        else:
            # Validate parts
            if not parts or len(parts) == 0:
                print(f"Warning: CoACD returned 0 parts")
                error_type = "failed_decomposition"
                # -------- CHANGED: treat as hard error --------
                reward = ERROR_REWARD
            else:
                valid_parts = []
                for i, (verts_i, faces_i) in enumerate(parts):
                    if len(verts_i) == 0 or len(faces_i) == 0:
                        print(f"Warning: Part {i} is empty (verts={len(verts_i)}, faces={len(faces_i)})")
                        continue
                    if np.any(np.isnan(verts_i)) or np.any(np.isinf(verts_i)):
                        print(f"Warning: Part {i} contains NaN/Inf vertices")
                        continue
                    valid_parts.append((verts_i, faces_i))

                if len(valid_parts) == 0:
                    print(f"Warning: No valid parts after validation")
                    error_type = "no_valid_parts"
                    # -------- CHANGED: treat as hard error --------
                    reward = ERROR_REWARD
                else:
                    # Create decomposed mesh
                    try:
                        verts_list, faces_list = zip(*valid_parts)
                        all_verts = np.vstack(verts_list)
                        all_faces = []
                        v_off = 0
                        for v_i, f_i in zip(verts_list, faces_list):
                            all_faces.append(f_i + v_off)
                            v_off += v_i.shape[0]
                        all_faces = np.vstack(all_faces)
                        dec_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
                    except Exception as e:
                        print(f"Error creating decomposed mesh: {e}")
                        error_type = "mesh_creation_failed"
                        # -------- CHANGED: treat as hard error --------
                        reward = ERROR_REWARD
                    else:
                        with tic("Hausdorff"):
                            H = self._hausdorff_vs_fixed(dec_mesh, valid_parts)

                        V_raw = dec_mesh.vertices.shape[0]
                        num_parts = len(valid_parts)

                        reward = self._calculate_comparative_reward(H, runtime, V_raw, num_parts)
                        
                        global _global_best_reward, _global_best_params
                        if reward > _global_best_reward:
                            _global_best_reward = reward
                            _global_best_params = {
                                "threshold": float(threshold),
                                "prep_resolution": int(prep_resolution),
                                "mcts_iteration": int(mcts_iteration),
                                "mcts_depth": int(mcts_depth),
                                "mcts_node": int(mcts_node),
                                "resolution": int(resolution),
                                "pca": bool(pca),
                            }
                            print(f"\n🏆 GLOBAL BEST IMPROVEMENT!")
                            print(f"   Hausdorff: {H:.6f} (baseline: {self.baseline_metrics['hausdorff_distance']:.6f})")
                            print(f"   Runtime:   {runtime:.3f}s (baseline: {self.baseline_metrics['runtime']:.3f}s)")
                            print(f"   Vertices:  {V_raw} (baseline: {self.baseline_metrics['total_vertices']})")
                            print(f"   Parts:     {num_parts} (baseline: {self.baseline_metrics['num_parts']})") 
                            print(f"   Params:    threshold={threshold:.3f}, prep_resolution={prep_resolution}, mcts_iteration={mcts_iteration}, mcts_depth={mcts_depth}, mcts_node={mcts_node}, resolution={resolution}, pca={pca}")
                            
                            # Save best results
                            import yaml
                            with open("best_results.yaml", "w") as f:
                                yaml.dump({
                                    "parameters": {
                                        "threshold": float(threshold),
                                        "prep_resolution": int(prep_resolution),
                                        "mcts_iteration": int(mcts_iteration),
                                        "mcts_depth": int(mcts_depth),
                                        "mcts_node": int(mcts_node),
                                        "resolution": int(resolution),
                                        "pca": bool(pca),
                                    },
                                    "metrics": {
                                        "hausdorff": float(H),
                                        "vertices": int(V_raw),
                                        "runtime": float(runtime),
                                        "num_parts": int(num_parts),
                                    },
                                    "baseline_metrics": {
                                        "hausdorff": float(self.baseline_metrics['hausdorff_distance']),
                                        "runtime": float(self.baseline_metrics['runtime']),
                                        "vertices": int(self.baseline_metrics['total_vertices']),
                                        "num_parts": int(self.baseline_metrics['num_parts']),
                                    }
                                }, f)

                    # Cleanup heavy objects before step ends
                    try:
                        del dec_mesh
                    except Exception:
                        pass

        # Memory trim & debug after step
        del parts
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._malloc_trim()
        self._print_mem_debug("after step")

        info = {
            "H": float(H),
            "V": int(V_raw),
            "T": runtime if 'runtime' in locals() else None,
            "num_parts": num_parts,
            "timeout": timeout,
            "success": success,
            "improvement": improvement,
            "params": {"threshold": threshold, "prep_resolution": prep_resolution, "mcts_iteration": mcts_iteration, "mcts_depth": mcts_depth, "mcts_node": mcts_node, "resolution": resolution, "pca": pca},
            "error_type": error_type,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            if hasattr(self, 'best_params') and self.best_params is not None:
                from src.utils.visualization import visualize_best_decomposition
                with open(os.devnull, 'w') as devnull:
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = devnull
                    sys.stderr = devnull
                    old_stdout_fd = os.dup(1)
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
                        os.close(old_stdout_fd)
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
            else:
                print("No best parameters available for rendering. Run some steps first.")
        return None

    def get_best_params(self):
        return self.best_params

    @staticmethod
    def get_global_best_params():
        global _global_best_params
        return _global_best_params

    @staticmethod
    def get_global_best_H():
        global _global_best_H
        return _global_best_H

    # Make sure to stop worker and free queues
    def close(self):
        try:
            if self._cmd_q:
                self._cmd_q.put({"cmd": "stop"})
        except Exception:
            pass
        try:
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=3)
        finally:
            for q in (getattr(self, "_cmd_q", None), getattr(self, "_res_q", None)):
                if q is not None:
                    try:
                        q.close(); q.join_thread()
                    except Exception:
                        pass
            self._worker = None