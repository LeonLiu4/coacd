import tempfile, time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch, trimesh

from utils.geometry import run_coacd, sample_surface, hausdorff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoACDEnv(gym.Env):
    """
    One‑step RL env:
      • obs  : 4096×3 point cloud  (Box[-1,1])
      • act  : 3 floats in [-1,1]  → (threshold, no_merge, max_hull)
      • done : True after one call to CoACD
    """
    metadata = {"render_modes": []}

    def __init__(self, mesh_path: str, npts: int = 4096):
        super().__init__()
        self.mesh_path = mesh_path
        self.npts = npts
        self.mesh = trimesh.load_mesh(mesh_path, process=False)

        self.observation_space = spaces.Box(-1, 1, shape=(npts, 3), dtype=np.float32)
        self.action_space      = spaces.Box(-1, 1, shape=(3,),    dtype=np.float32)

        self.src_pts = sample_surface(self.mesh, npts).to(device)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, **_):
        super().reset(seed=seed)
        return self.src_pts.squeeze(0).cpu().numpy(), {}

    def step(self, action):
        # map [-1,1] → real params
        threshold = 0.01 + (action[0] * 0.5 + 0.5) * 0.99
        no_merge  = bool(action[1] > 0)
        max_hull  = int(10 + (action[2] * 0.5 + 0.5) * 90)

        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.time()
            out_mesh = run_coacd(self.mesh_path, tmp,
                                 threshold=threshold,
                                 no_merge=no_merge,
                                 max_hull=max_hull)
            runtime = time.time() - t0
            dec = trimesh.load_mesh(out_mesh, process=False)
            tgt_pts = sample_surface(dec, self.npts).to(device)

        H = hausdorff(self.src_pts, tgt_pts)
        V = dec.vertices.shape[0] / 1e4
        T = runtime / 10.0
        reward = -(1.0 * H + 0.1 * V + 0.1 * T)

        info = dict(H=H, V=V*1e4, T=runtime,
                    params=dict(threshold=threshold,
                                no_merge=no_merge,
                                max_hull=max_hull))
        obs = self.src_pts.squeeze(0).cpu().numpy()
        return obs, reward, True, False, info
