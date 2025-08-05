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

@contextmanager
def tic(label: str):
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1e3
    print(f"[TIMER] {label:<15} {dt:7.1f} ms")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # pre-sample fixed interior points
        pts = sample_points(self.mesh, npts)
        self.src_pts = torch.as_tensor(pts, dtype=torch.float32)[None]

    def reset(self, *, seed=None, **kwargs):
        super().reset(seed=seed)
        return self.src_pts.squeeze(0).cpu().numpy(), {}

    def step(self, action: np.ndarray):
        # 1) map normalized action → real parameters
        with tic("map-action"):
            raw       = action[0] * 0.5 + 0.5
            threshold = float(max(0.01, min(0.01 + raw * 0.99, 1.0)))
            no_merge  = bool(action[1] > 0)
            max_hull  = int(10 + (action[2] * 0.5 + 0.5) * 90)

        # 2) run CoACD in a separate process with timeout
        limit_sec = 1.0
        queue     = Queue()
        proc      = Process(
            target=_coacd_worker,
            args=(queue, self._coacd_mesh, threshold, not no_merge, max_hull),
        )
        t0 = time.time()
        proc.start()
        proc.join(timeout=limit_sec)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            # timed out → heavy penalty
            reward = -1e3
            info = {
                "timeout": True,
                "T": limit_sec,
                "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
            }
            obs = self.src_pts.squeeze(0).cpu().numpy()
            return obs, reward, True, False, info

        runtime = time.time() - t0
        parts   = queue.get_nowait()

        # 3) stitch all convex‐hull parts into a single mesh
        verts_list, faces_list = zip(*parts)
        all_verts = np.vstack(verts_list)
        all_faces = []
        v_off = 0
        for verts, faces in zip(verts_list, faces_list):
            all_faces.append(faces + v_off)
            v_off += verts.shape[0]
        all_faces = np.vstack(all_faces)
        dec_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)

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

        # 6) assemble reward
        with tic("assemble reward"):
            V      = dec_mesh.vertices.shape[0] / 1e4    # complexity penalty
            T_norm = runtime / 10.0                      # time penalty (≈0–1)
            reward = -(1.0 * H + 0.1 * V + 0.1 * T_norm)

        info = {
            "H": H,
            "V": int(V * 1e4),
            "T": runtime,
            "timeout": False,
            "params": {"threshold": threshold, "no_merge": no_merge, "max_hull": max_hull},
        }

        obs = self.src_pts.squeeze(0).cpu().numpy()
        return obs, reward, True, False, info