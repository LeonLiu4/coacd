import numpy as np
import trimesh
import torch
from pytorch3d.ops import knn_points


@torch.no_grad()
def hausdorff(a_pts: torch.Tensor, b_pts: torch.Tensor) -> float:
    """Symmetric Hausdorff distance between (1,N,3) and (1,M,3) tensors."""
    d_ab = knn_points(a_pts, b_pts, K=1).dists.squeeze(-1).max()
    d_ba = knn_points(b_pts, a_pts, K=1).dists.squeeze(-1).max()
    return max(d_ab, d_ba).item()


def sample_points(mesh: trimesh.Trimesh, n_pts: int, seed: int = None) -> np.ndarray:
    """Return exactly `n_pts` points every time (surface sampling)."""
    if seed is not None:
        np.random.seed(seed)
    pts = mesh.sample(n_pts)
    if pts.shape[0] < n_pts:
        extra = mesh.sample(n_pts - pts.shape[0])
        pts = np.vstack([pts, extra])
    return np.ascontiguousarray(pts, dtype=np.float32)


def sample_surface_points_from_parts(parts, n_pts: int, seed: int = 42, num_angles: int = 100) -> np.ndarray:
    """Sample surface points from convex hull parts using raycasting."""
    return ray_sample_surface(parts, n_pts=n_pts, num_dirs=num_angles, rays_per_dir=100, 
                             tol=1e-3, seed=seed)





def enhanced_viewing_dirs(n: int = 100) -> np.ndarray:
    """Azimuth×altitude grid; for n<=100 defaults to 10×10."""
    grid_size = int(np.sqrt(n))
    azimuth_count = grid_size
    altitude_count = grid_size
    if azimuth_count * altitude_count != n:
        if n <= 100:
            azimuth_count = 10
            altitude_count = 10
        else:
            azimuth_count = int(np.sqrt(n))
            altitude_count = int(np.sqrt(n))
    azimuth_angles = np.linspace(0, 2 * np.pi, azimuth_count, endpoint=False)
    altitude_angles = np.linspace(-np.pi/2, np.pi/2, altitude_count)
    directions = []
    for az in azimuth_angles:
        for alt in altitude_angles:
            x = np.cos(alt) * np.cos(az)
            y = np.cos(alt) * np.sin(az)
            z = np.sin(alt)
            directions.append([x, y, z])
    directions = np.array(directions, dtype=np.float64)
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def build_combined(parts) -> trimesh.Trimesh:
    meshes = [trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in parts]
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def first_hits_locations(mesh: trimesh.Trimesh,
                         origins: np.ndarray,
                         directions: np.ndarray) -> tuple:
    """Return (locations, ray_indices, distances) for first hits; safe fallback if pyembree not available."""
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        rmi = RayMeshIntersector(mesh)
        loc, idx_ray, _ = rmi.intersects_location(origins, directions, multiple_hits=False)
        if len(loc) == 0:
            return loc, np.array([], dtype=int), np.array([])
        # Compute distance along each ray: t = dot(loc - origin[idx], dir[idx])
        t = np.einsum('ij,ij->i', (loc - origins[idx_ray]), directions[idx_ray])
        return loc, idx_ray, t
    except Exception:
        loc, idx_ray, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions
        )
        if len(loc) == 0:
            return loc, np.array([], dtype=int), np.array([])
        # Compute distance along each ray
        t = np.einsum('ij,ij->i', (loc - origins[idx_ray]), directions[idx_ray])
        return loc, idx_ray, t



def _pose_from_lookat(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """WORLD-from-CAMERA pose; camera looks along -Z."""
    f = target - eye; f = f / (np.linalg.norm(f) + 1e-12)
    u = up_hint / (np.linalg.norm(up_hint) + 1e-12)
    r = np.cross(f, u)
    if np.linalg.norm(r) < 1e-8:
        u = np.array([1.0, 0.0, 0.0]) if abs(f[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
        u = u / (np.linalg.norm(u) + 1e-12)
        r = np.cross(f, u)
    r = r / (np.linalg.norm(r) + 1e-12)
    u = np.cross(r, f)
    T = np.eye(4); T[:3, 0] = r; T[:3, 1] = u; T[:3, 2] = -f; T[:3, 3] = eye
    return T


def _fit_camera_params(mesh: trimesh.Trimesh, yfov_rad: float, aspect: float = 1.0,
                       margin: float = 1.15):
    bbmin, bbmax = mesh.bounds
    center = 0.5 * (bbmin + bbmax)
    radius = 0.5 * np.linalg.norm(bbmax - bbmin) * 1.05
    base_distance = radius / np.sin(max(1e-3, yfov_rad * 0.5))
    cam_distance = margin * base_distance
    znear = max(1e-4, cam_distance - 2.5 * radius)
    zfar = cam_distance + 2.5 * radius
    return center, radius, cam_distance, znear, zfar


def _pixel_rays_world(eye: np.ndarray, pose: np.ndarray, yfov: float, aspect: float,
                      width: int, height: int):
    """Return per-pixel ray directions in WORLD coords for a full image grid."""
    tx = np.tan(yfov * 0.5) * aspect
    ty = np.tan(yfov * 0.5)

    u = (np.arange(width) + 0.5) / width  * 2.0 - 1.0
    v = 1.0 - (np.arange(height) + 0.5) / height * 2.0
    U, V = np.meshgrid(u, v, indexing='xy')

    dir_cam = np.stack([U * tx, V * ty, -np.ones_like(U)], axis=-1)
    dir_cam /= (np.linalg.norm(dir_cam, axis=-1, keepdims=True) + 1e-12)

    R = pose[:3, :3]
    dir_world = dir_cam @ R.T
    dir_world /= (np.linalg.norm(dir_world, axis=-1, keepdims=True) + 1e-12)

    eye_world = eye.reshape(1, 1, 3).repeat(height, axis=0).repeat(width, axis=1)
    return eye_world, dir_world


def backproject_depth_to_points(depth: np.ndarray,
                                eye: np.ndarray,
                                pose: np.ndarray,
                                yfov: float,
                                aspect: float,
                                width: int,
                                height: int,
                                stride: int = 2) -> np.ndarray:
    """Convert a depth map into a point cloud by shooting per-pixel rays."""
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")

    eye_grid, dir_world = _pixel_rays_world(eye, pose, yfov, aspect, width, height)

    valid = depth > 0.0
    if stride > 1:
        valid[::stride, :] = valid[::stride, :]
        valid[:, ::stride] = valid[:, ::stride]

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    d = depth[..., None]
    pts = eye_grid + dir_world * d
    pts = pts[valid]
    return np.ascontiguousarray(pts.astype(np.float32))


import numpy as np

def ray_sample_surface(parts, n_pts: int = 4096, num_dirs: int = 100, rays_per_dir: int = 4096,
                       tol: float = 1e-3, seed: int = 42) -> np.ndarray:
    """Sample surface points from convex hulls using raycasting."""
    # Set global numpy seed for deterministic behavior
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    # Combine all convex hulls into a single mesh
    mesh = build_combined(parts)
    
    bbmin, bbmax = mesh.bounds
    center = 0.5 * (bbmin + bbmax)
    extent = (bbmax - bbmin)
    R = float(np.linalg.norm(extent))
    side = 1.25 * float(np.max(extent))

    # Bounding-sphere based outward push (core fix support)
    R_sphere = 0.5 * float(np.linalg.norm(extent)) * np.sqrt(3.0)  # circumscribes AABB
    margin = 0.5 * float(np.linalg.norm(extent))

    dirs = enhanced_viewing_dirs(num_dirs)
    all_points = []

    # Create deterministic grid
    g = int(np.ceil(np.sqrt(rays_per_dir)))
    uv = np.linspace(-1.1, 1.1, g)
    U, V = np.meshgrid(uv, uv, indexing='xy')
    base_grid = np.stack([U.ravel(), V.ravel()], axis=1)
    if len(base_grid) > rays_per_dir:
        # Use deterministic selection instead of random
        step = len(base_grid) / rays_per_dir
        indices = np.arange(0, len(base_grid), step, dtype=int)[:rays_per_dir]
        base_grid = base_grid[indices]

    # Create deterministic rotation matrices
    rotation_matrices = []
    for i in range(len(dirs)):
        # Use deterministic angle based on index and seed
        theta = (i * 137.5 + seed * 42) % (2.0 * np.pi)  # Golden angle approximation
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrices.append(np.array([[c, -s], [s, c]], dtype=np.float64))

    total_points = 0
    for i, d in enumerate(dirs):
        if total_points > 2 * n_pts:
            break
        d = d / (np.linalg.norm(d) + 1e-12)
        g2 = base_grid @ rotation_matrices[i].T
        # Use deterministic noise based on index
        noise_seed = (seed * 1000 + i * 100) % 2**32
        np.random.seed(noise_seed)
        g2 = g2 + 0.01 * np.random.standard_normal(size=g2.shape)

        a = np.array([1.0, 0.0, 0.0]) if abs(d[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        p2 = np.cross(d, a); p2 /= (np.linalg.norm(p2) + 1e-12)
        p1 = np.cross(p2, d); p1 /= (np.linalg.norm(p1) + 1e-12)

        # --- core origin fix: place on plane, then push outward along ±d ---
        plane_pts = center + side * (g2[:, 0:1] * p1 + g2[:, 1:2] * p2)

        # Cast from +d side shooting inward (-d)
        origins_1 = plane_pts + d * (R_sphere + margin)
        dirs_1 = np.repeat((-d)[None, :], len(origins_1), axis=0)

        # Cast from -d side shooting inward (+d)  <-- handles backface culling / flipped normals
        origins_2 = plane_pts - d * (R_sphere + margin)
        dirs_2 = np.repeat((d)[None, :], len(origins_2), axis=0)

        # Raycast against the combined convex hull mesh
        try:
            loc1, idx1, t1 = first_hits_locations(mesh, origins_1, dirs_1)
            if len(loc1) > 0:
                t1 = np.asarray(t1)
                keep1 = t1 > 1e-8
                if np.any(keep1):
                    all_points.extend(np.asarray(loc1)[keep1])
                    total_points += int(np.count_nonzero(keep1))
        except Exception:
            pass

        try:
            loc2, idx2, t2 = first_hits_locations(mesh, origins_2, dirs_2)
            if len(loc2) > 0:
                t2 = np.asarray(t2)
                keep2 = t2 > 1e-8
                if np.any(keep2):
                    all_points.extend(np.asarray(loc2)[keep2])
                    total_points += int(np.count_nonzero(keep2))
        except Exception:
            pass

    if not all_points:
        return np.ascontiguousarray(mesh.sample(n_pts).astype(np.float32))

    pts = np.vstack(all_points)
    
    # Deduplicate
    q = np.round(pts / tol).astype(np.int64)
    _, unique_idx = np.unique(q, axis=0, return_index=True)
    pts = pts[np.sort(unique_idx)]

    if len(pts) >= n_pts:
        step = len(pts) / n_pts
        indices = np.arange(0, len(pts), step, dtype=int)[:n_pts]
        pts = pts[indices]
    else:
        # Use deterministic sampling for fallback
        np.random.seed(seed)  # Ensure deterministic fallback
        extra = mesh.sample(n_pts - len(pts))
        pts = np.vstack([pts, extra])

    return np.ascontiguousarray(pts.astype(np.float32))