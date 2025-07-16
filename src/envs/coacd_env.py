# src/envs/coacd_env.py
import tempfile, time, os
import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import trimesh

from ..utils.geometry import run_coacd, sample_points, hausdorff
import time
from contextlib import contextmanager

@contextmanager
def tic(label: str):
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1e3  # → ms
    print(f"[TIMER] {label:<15} {dt:7.1f} ms")

# ---------------------------------------------------------------------
#  Device
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoACDEnv(gym.Env):
    """
    One‑shot RL environment for CoACD parameter search.

    Observation
    -----------
    • (N, 3) float32 point cloud sampled *inside* the original mesh
      (values already centred/normalised to ~[‑1,1] by trimesh).

    Action
    ------
    • 3 floats in [‑1,1] mapped to:
        0 → threshold   ∈ [0.01, 1.00]
        1 → no‑merge    {False, True}
        2 → max_hull    ∈ [10, 100]

    Episode terminates after exactly one call to CoACD (`done=True`).
    """

    metadata = {"render_modes": []}

    def __init__(self, mesh_path: str, npts: int = 4096):
        super().__init__()
        self.mesh_path = mesh_path
        self.npts = npts
        self.mesh = trimesh.load_mesh(mesh_path, process=False)

        # --- Gym spaces ------------------------------------------------
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(npts, 3), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Pre‑sample fixed interior points from the source mesh
        self.src_pts = torch.as_tensor(sample_points(self.mesh, npts).copy(), dtype=torch.float32)[None]

    # ------------------------------------------------------------------
    #  Standard Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, **_):
        super().reset(seed=seed)
        return self.src_pts.squeeze(0).cpu().numpy(), {}

    def step(self, action: np.ndarray):
        """
        Execute one CoACD decomposition step and return the usual Gym tuple.
        Added [TIMER] prints show wall‑time for each block in milliseconds.
        """
        # -------- map normalized action -> real CoACD parameters ------------
        with tic("map‑action"):
            threshold = 0.01 + (action[0] * 0.5 + 0.5) * 0.99      # [0.01,1]
            no_merge  = bool(action[1] > 0)                         # True/False
            max_hull  = int(10 + (action[2] * 0.5 + 0.5) * 90)      # [10,100]

        # -------------------------- run CoACD -------------------------------
        with tic("run CoACD"):
            with tempfile.TemporaryDirectory() as tmp:
                t0 = time.time()
                out_mesh_path = run_coacd(
                    self.mesh_path,
                    tmp,
                    threshold=threshold,
                    no_merge=no_merge,
                    max_hull=max_hull,
                )
                runtime = time.time() - t0
                dec_mesh = trimesh.load_mesh(out_mesh_path, process=False)

        # -------------------- sample interior points -----------------------
        with tic("sample pts"):
            vol_in  = sample_points(self.mesh,    self.npts)
            vol_out = sample_points(dec_mesh,     self.npts)

        # ---------------------- Hausdorff distance --------------------------
        with tic("Hausdorff"):
            H = hausdorff(
                torch.as_tensor(vol_in )[None].to(device),
                torch.as_tensor(vol_out)[None].to(device),
            )

        # ----------------------- reward assembly ---------------------------
        with tic("assemble reward"):
            V = dec_mesh.vertices.shape[0] / 1e4   # complexity penalty
            T = runtime / 10.0                     # time penalty (≈0‑1)
            reward = -(1.0 * H + 0.1 * V + 0.1 * T)

        # ---------------------------- info ---------------------------------
        info = dict(
            H=H,
            V=int(V * 1e4),
            T=runtime,
            params=dict(threshold=threshold, no_merge=no_merge, max_hull=max_hull),
        )

        # ---------------------------- return -------------------------------
        obs = self.src_pts.squeeze(0).cpu().numpy()
        return obs, reward, True, False, info