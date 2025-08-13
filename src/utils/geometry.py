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
                         directions: np.ndarray) -> np.ndarray:
    """Return first-hit locations for each ray; safe fallback if pyembree not available."""
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        rmi = RayMeshIntersector(mesh)
        loc, _, _ = rmi.intersects_location(origins, directions, multiple_hits=False)
        return loc
    except Exception:
        loc, idx_ray, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions
        )
        if len(loc) == 0:
            return loc
        # Keep the closest hit per ray
        t = np.einsum('ij,ij->i', (loc - origins[idx_ray]), directions[idx_ray])
        order = np.lexsort((t, idx_ray))
        idx_ray_sorted = idx_ray[order]
        loc_sorted = loc[order]
        keep = np.r_[True, idx_ray_sorted[1:] != idx_ray_sorted[:-1]]
        return loc_sorted[keep]


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


def ray_sample_surface(parts, n_pts: int = 4096, num_dirs: int = 100, rays_per_dir: int = 4096,
                       tol: float = 1e-3, seed: int = 42, outer_mesh: trimesh.Trimesh = None) -> np.ndarray:
    """Sample surface points from convex hulls with optional outer mesh filtering."""
    rng = np.random.default_rng(seed)
    
    # Convert parts to individual hull meshes
    hull_meshes = [trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in parts]
    
    # Bounds
    if outer_mesh is not None:
        mesh = outer_mesh
    else:
        mesh = build_combined(parts)
    
    bbmin, bbmax = mesh.bounds
    center = 0.5 * (bbmin + bbmax)
    extent = (bbmax - bbmin)
    R = float(np.linalg.norm(extent))
    side = 1.25 * float(np.max(extent))

    dirs = enhanced_viewing_dirs(num_dirs)
    all_points = []

    g = int(np.ceil(np.sqrt(rays_per_dir)))
    uv = np.linspace(-1.1, 1.1, g)
    U, V = np.meshgrid(uv, uv, indexing='xy')
    base_grid = np.stack([U.ravel(), V.ravel()], axis=1)
    if len(base_grid) > rays_per_dir:
        pick = rng.choice(len(base_grid), size=rays_per_dir, replace=False)
        base_grid = base_grid[pick]

    rotation_matrices = []
    for _ in range(len(dirs)):
        theta = rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrices.append(np.array([[c, -s], [s, c]], dtype=np.float64))

    total_points = 0
    for i, d in enumerate(dirs):
        if total_points > 2 * n_pts:
            break
        d = d / (np.linalg.norm(d) + 1e-12)
        g2 = base_grid @ rotation_matrices[i].T
        g2 = g2 + 0.01 * rng.standard_normal(size=g2.shape)

        a = np.array([1.0, 0.0, 0.0]) if abs(d[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        p2 = np.cross(d, a); p2 /= (np.linalg.norm(p2) + 1e-12)
        p1 = np.cross(p2, d); p1 /= (np.linalg.norm(p1) + 1e-12)

        origins = center + d * (2.0 * R) + side * (g2[:, 0:1] * p1 + g2[:, 1:2] * p2)
        directions = np.repeat((-d)[None, :], len(origins), axis=0)

        # Find closest hits across all hull meshes
        closest_hits = []
        for hull_mesh in hull_meshes:
            try:
                loc = first_hits_locations(hull_mesh, origins, directions)
                if len(loc) > 0:
                    # Calculate distances from origins
                    distances = np.linalg.norm(loc - origins[:len(loc)], axis=1)
                    for j, (point, dist) in enumerate(zip(loc, distances)):
                        closest_hits.append((point, dist, i * rays_per_dir + j))
            except Exception:
                continue

        # Keep only the closest hit for each ray
        if closest_hits:
            closest_hits.sort(key=lambda x: x[1])  # Sort by distance
            seen_rays = set()
            for point, dist, ray_idx in closest_hits:
                if ray_idx not in seen_rays:
                    all_points.append(point)
                    seen_rays.add(ray_idx)
                    total_points += 1

    if not all_points:
        if outer_mesh is not None:
            return np.ascontiguousarray(outer_mesh.sample(n_pts).astype(np.float32))
        else:
            return np.ascontiguousarray(mesh.sample(n_pts).astype(np.float32))

    pts = np.vstack(all_points)
    
    # Apply outer-surface filter if outer_mesh is provided
    if outer_mesh is not None:
        try:
            from trimesh.proximity import signed_distance
            sd = signed_distance(outer_mesh, pts)
            keep = np.abs(sd) <= tol
            pts = pts[keep]
        except Exception:
            try:
                from trimesh.proximity import closest_point
                closest, distances, _ = closest_point(outer_mesh, pts)
                keep = distances <= tol
                pts = pts[keep]
            except Exception:
                pass  # Keep all points if filtering fails

    # Deduplicate
    q = np.round(pts / tol).astype(np.int64)
    _, unique_idx = np.unique(q, axis=0, return_index=True)
    pts = pts[np.sort(unique_idx)]

    if len(pts) >= n_pts:
        step = len(pts) / n_pts
        indices = np.arange(0, len(pts), step, dtype=int)[:n_pts]
        pts = pts[indices]
    else:
        if outer_mesh is not None:
            extra = outer_mesh.sample(n_pts - len(pts))
        else:
            extra = mesh.sample(n_pts - len(pts))
        pts = np.vstack([pts, extra])

    return np.ascontiguousarray(pts.astype(np.float32))