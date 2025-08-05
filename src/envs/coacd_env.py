# src/envs/coacd_env.py
import time
from contextlib import contextmanager
import concurrent.futures

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

class CoACDEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mesh_path: str, npts: int = 4096):
        super().__init__()
        self.mesh_path = mesh_path
        self.npts = npts

        # load source mesh
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
        # map action → parameters
        with tic("map-action"):
            raw = action[0] * 0.5 + 0.5
            threshold = float(max(0.01, min(0.01 + raw * 0.99, 1.0)))
            no_merge  = bool(action[1] > 0)
            max_hull  = int(10 + (action[2] * 0.5 + 0.5) * 90)

        # run CoACD via Python API with timeout
        limit_sec = 0.1
        t0 = time.time()
        timed_out = False

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        future = executor.submit(
            coacd.run_coacd,
            self._coacd_mesh,
            threshold=threshold,
            merge=not no_merge,
            max_convex_hull=max_hull,
        )
        try:
            parts = future.result(timeout=limit_sec)
            runtime = time.time() - t0
            executor.shutdown()
        except concurrent.futures.TimeoutError:
            future.cancel()
            runtime = limit_sec
            timed_out = True
            executor.shutdown(wait=False, cancel_futures=True)

        if timed_out:
            reward = -1e3
            info = {
                "timeout": True,
                "T": runtime,
                "params": {
                    "threshold": threshold,
                    "no_merge": no_merge,
                    "max_hull": max_hull,
                },
            }
            obs = self.src_pts.squeeze(0).cpu().numpy()
            return obs, reward, True, False, info

        # stitch resulting parts into one mesh
        verts_list, faces_list = zip(*parts)
        all_verts = np.vstack(verts_list)
        all_faces = []
        v_off = 0
        for verts, faces in zip(verts_list, faces_list):
            all_faces.append(faces + v_off)
            v_off += verts.shape[0]
        all_faces = np.vstack(all_faces)
        dec_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)

        # sample points
        with tic("sample pts"):
            vol_in  = sample_points(self.mesh,    self.npts)
            vol_out = sample_points(dec_mesh,     self.npts)

        # Hausdorff
        with tic("Hausdorff"):
            H = hausdorff(
                torch.as_tensor(vol_in)[None].to(device),
                torch.as_tensor(vol_out)[None].to(device),
            )

        # assemble reward
        with tic("assemble reward"):
            V = dec_mesh.vertices.shape[0] / 1e4
            T = runtime / 10.0
            reward = -(1.0 * H + 0.1 * V + 0.1 * T)

        info = {
            "H": H,
            "V": int(V * 1e4),
            "T": runtime,
            "timeout": False,
            "params": {
                "threshold": threshold,
                "no_merge": no_merge,
                "max_hull": max_hull,
            },
        }
        obs = self.src_pts.squeeze(0).cpu().numpy()
        return obs, reward, True, False, info