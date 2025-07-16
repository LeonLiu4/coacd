import os, subprocess, torch, trimesh
import numpy as np
from pytorch3d.ops import knn_points


# ------------------------------------------------------------------ #
#  external CLI wrapper                                              #
# ------------------------------------------------------------------ #
def run_coacd(mesh_path: str, out_dir: str, *, threshold: float,
              no_merge: bool, max_hull: int) -> str:
    """Invoke CoACD CLI and return the path to the decomposed OBJ."""
    out_path = os.path.join(out_dir, "decomp.obj")
    cmd = [
        "coacd",
        "--input", mesh_path,
        "--output", out_path,
        "--threshold", f"{threshold:.4f}",
        "--no-merge" if no_merge else "--merge",
        "--max_hull", str(max_hull),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
    return out_path


# ------------------------------------------------------------------ #
#  distance / sampling helpers                                       #
# ------------------------------------------------------------------ #
@torch.no_grad()
def hausdorff(a_pts: torch.Tensor, b_pts: torch.Tensor) -> float:
    """Symmetric Hausdorff distance between (1,N,3) and (1,M,3) tensors."""
    d_ab = knn_points(a_pts, b_pts, K=1).dists.squeeze(-1).max()
    d_ba = knn_points(b_pts, a_pts, K=1).dists.squeeze(-1).max()
    return max(d_ab, d_ba).item()


def sample_points(mesh: trimesh.Trimesh, n_pts: int) -> np.ndarray:
    """Return exactly `n_pts` points every time."""
    pts = mesh.sample(n_pts)              # surface sample is fast & always fills
    # mesh.sample already guarantees size, but be safe:
    if pts.shape[0] < n_pts:
        extra = mesh.sample(n_pts - pts.shape[0])
        pts   = np.vstack([pts, extra])
    return np.ascontiguousarray(pts, dtype=np.float32)