# src/envs/coacd_env.py
import time
from contextlib import contextmanager
from multiprocessing import Process, Queue

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import trimesh
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
    Run coacd.run_coacd in a subprocess and push the resulting parts onto the queue.
    """
    parts = coacd.run_coacd(
        mesh,
        threshold=threshold,
        merge=merge,
        max_convex_hull=max_hull,
    )
    queue.put(parts)


# ── main Gym environment ─────────────────────────────────────────────
class CoACDEnv(gym.Env):
    """
    Gym environment wrapping a single CoACD call as an RL step.
    """

    metadata = {"render_modes": []}

    def __init__(self, mesh_path: str, npts: int = 4096):
        super().__init__()
        self.mesh_path = mesh_path
        self.npts = npts

        # load & wrap mesh for CoACD
        self.mesh = trimesh.load_mesh(mesh_path, process=False)
        self._coacd_mesh = coacd.Mesh(
            self.mesh.vertices.astype(np.float32),
            self.mesh.faces.astype(np.int64),
        )

        # gym spaces
        self.observation_space = spaces.Box(-1.0, 1.0, (npts, 3), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (3,),     np.float32)

        # tracking variables
        self.best_H = None
        self.step_count = 0
        self.success_threshold = 0.01  # Hausdorff distance threshold for success

        # initial observation
        self._sample_obs()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
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

        # 6) assemble reward with better scaling and progress tracking
        with tic("assemble reward"):
            V_raw  = dec_mesh.vertices.shape[0]
            V      = V_raw / 1e4                           # complexity penalty (0-10+ typically)
            T_norm = min(runtime / 10.0, 1.0)             # time penalty capped at 1.0
            
            # Base reward (negative, since we want to minimize)
            base_reward = -(1.0 * H + 0.1 * V + 0.05 * T_norm)
            
            # Track improvement
            improvement = False
            if self.best_H is None or H < self.best_H:
                improvement = True
                if self.best_H is not None:
                    improvement_bonus = 2.0 * (self.best_H - H)  # bonus proportional to improvement
                    base_reward += improvement_bonus
                self.best_H = H
            
            # Success bonus
            success = H < self.success_threshold
            if success:
                base_reward += 5.0  # big bonus for achieving success
                terminated = True   # end episode on success
            
            reward = base_reward

        # 7) Create observation from the decomposed mesh (key change!)
        obs = self._create_observation(dec_mesh)

        info = {
            "H": float(H),
            "V": int(V_raw),
            "T": runtime,
            "timeout": False,
            "success": success,
            "improvement": improvement,
            "best_H": float(self.best_H) if self.best_H is not None else float('inf'),
            "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
        }

        # Debug prints
        if self.step_count <= 5 or self.step_count % 10 == 0:
            print(f"  → H: {H:.4f}, V: {V_raw}, T: {runtime:.3f}s, Reward: {reward:.3f}, Success: {success}, Improvement: {improvement}")

        return obs, reward, terminated, truncated, info