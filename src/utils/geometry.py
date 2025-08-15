import numpy as np
import trimesh
import torch
from pytorch3d.ops import knn_points
import pyrender
import os
import pymeshlab


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


def simplify_point_cloud_meshlab(points: np.ndarray, target_points: int = 100000) -> np.ndarray:
    """
    Use uniform grid sampling to create uniform point clouds.
    
    Args:
        points: Input point cloud as numpy array (N, 3)
        target_points: Target number of points after simplification
        
    Returns:
        Simplified point cloud as numpy array (target_points, 3)
    """
    try:
        print(f"Simplifying {len(points)} points to {target_points} points using uniform grid sampling...")
        
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Estimate grid resolution based on target points
        volume = np.prod(max_coords - min_coords)
        grid_resolution = (volume / target_points) ** (1/3)
        
        # Create grid cells
        grid_coords = np.floor((points - min_coords) / grid_resolution).astype(int)
        
        # Use a hash function to assign points to grid cells
        grid_hash = grid_coords[:, 0] + grid_coords[:, 1] * 1000 + grid_coords[:, 2] * 1000000
        
        # Find unique grid cells and sample one point from each
        unique_hashes, unique_indices = np.unique(grid_hash, return_index=True)
        simplified_points = points[unique_indices]
        
        print(f"Grid sampling reduced to {len(simplified_points)} points")
        
        # If we still have too many points, use random sampling
        if len(simplified_points) > target_points:
            indices = np.random.choice(len(simplified_points), target_points, replace=False)
            simplified_points = simplified_points[indices]
            print(f"Further reduced to {len(simplified_points)} points via random sampling")
        elif len(simplified_points) < target_points:
            # If we have too few points, add some random points
            needed = target_points - len(simplified_points)
            extra_indices = np.random.choice(len(points), needed, replace=False)
            extra_points = points[extra_indices]
            simplified_points = np.vstack([simplified_points, extra_points])
            print(f"Added {needed} random points to reach {len(simplified_points)} points")
        
        return simplified_points.astype(np.float32)
        
    except Exception as e:
        print(f"Point cloud simplification failed: {e}")
        print("Falling back to random sampling...")
        # Fallback to random sampling
        if len(points) > target_points:
            indices = np.random.choice(len(points), target_points, replace=False)
            return points[indices].astype(np.float32)
        else:
            return points.astype(np.float32)


def sample_surface_points_from_parts(parts, n_pts: int, seed: int = 42, num_angles: int = 100, 
                                    resolution: int = 1024) -> np.ndarray:
    """Sample surface points from convex hull parts using depth-based rendering."""
    return sample_surface_points_via_depth(parts, n_pts=n_pts, num_dirs=num_angles,
                                          resolution=resolution, yfov_deg=60.0,
                                          tol=1e-3, seed=seed, save_debug=False)


def sample_surface_points_from_parts_fast(parts, n_pts: int, seed: int = 42, num_angles: int = 25) -> np.ndarray:
    """Fast sampling for training - optimized for speed with lower resolution and efficient parameters."""
    return sample_surface_points_via_depth(parts, n_pts=n_pts, num_dirs=num_angles,
                                          resolution=512, yfov_deg=60.0,  # Lower resolution for speed
                                          tol=1e-2, seed=seed, save_debug=False,  # Higher tolerance for speed
                                          use_meshlab_simplification=False)  # Skip MeshLab for speed


def sample_surface_points_via_depth(parts,
                                    n_pts: int,
                                    num_dirs: int = 100,
                                    resolution: int = 1024,
                                    yfov_deg: float = 60.0,
                                    tol: float = 1e-3,
                                    seed: int = 42,
                                    save_debug: bool = False,
                                    use_meshlab_simplification: bool = True) -> np.ndarray:
    """Depth-based sampling on the combined convex hulls using depth map backprojection."""
    np.random.seed(seed)  # Ensure deterministic behavior
    
    mesh = build_combined(parts)
    directions = enhanced_viewing_dirs(num_dirs)
    
    print(f"Rendering depth maps from {num_dirs} directions at {resolution}x{resolution} resolution...")
    
    # Create debug output directory if requested
    if save_debug:
        debug_dir = "debug_analysis"
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Saving debug data to {debug_dir}/")
    
    # Calculate stride for efficient sampling
    # For training: use larger stride to sample fewer points per view
    if num_dirs <= 25:  # Training mode
        per_view_target = max(100, int(np.ceil(n_pts / max(1, num_dirs))))
        stride_guess = max(4, int(np.sqrt((resolution * resolution) / (per_view_target * 2.0))))
    else:  # High-quality mode
        per_view_target = max(256, int(np.ceil(n_pts / max(1, num_dirs))))
        stride_guess = max(1, int(np.sqrt((resolution * resolution) / (per_view_target * 1.25))))
    
    all_pts = []
    valid_views = 0
    
    for i, (depth, eye, pose, yfov, aspect, W, H) in enumerate(render_depth_packets(mesh, directions, resolution, yfov_deg)):
        if not np.any(depth > 0):
            continue
            
        valid_views += 1
        pts = backproject_depth_to_points(depth, eye, pose, yfov, aspect, W, H, stride=stride_guess)
        
        # Save debug data if requested
        if save_debug and pts.size > 0:
            # Save depth map as PNG
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 8))
            plt.imshow(depth, cmap='viridis')
            plt.colorbar(label='Depth')
            plt.title(f'Depth Map - View {i:03d}')
            plt.savefig(f"{debug_dir}/depth_map_{i:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save corresponding points as PLY
            with open(f"{debug_dir}/points_view_{i:03d}.ply", "w") as f:
                f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
                f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
                for p in pts:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
            # Save camera info
            np.savez(f"{debug_dir}/camera_info_{i:03d}.npz", 
                     eye=eye, pose=pose, yfov=yfov, aspect=aspect, 
                     width=W, height=H, direction=directions[i])
        
        if pts.size:
            all_pts.append(pts)
            
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_dirs} views, collected {sum(len(pts) for pts in all_pts)} points so far")
    
    print(f"Completed depth rendering: {valid_views}/{num_dirs} valid views")
    
    if not all_pts:
        print("Warning: No valid depth maps generated, falling back to mesh sampling")
        return mesh.sample(n_pts).astype(np.float32)
    
    pts = np.vstack(all_pts)
    print(f"Total points from depth backprojection: {len(pts)}")
    
    # Deduplicate points
    q = np.round(pts / tol).astype(np.int64)
    _, keep = np.unique(q, axis=0, return_index=True)
    pts = pts[np.sort(keep)]
    print(f"After deduplication: {len(pts)} unique points")
    
    # Use MeshLab for uniform point cloud simplification if requested
    if use_meshlab_simplification and len(pts) > n_pts:
        print(f"Using MeshLab to simplify {len(pts)} points to {n_pts} uniform points...")
        pts = simplify_point_cloud_meshlab(pts, n_pts)
    else:
        # Fallback to original downsampling strategy
        if len(pts) > n_pts:
            step = len(pts) / n_pts
            idx = np.arange(0, len(pts), step, dtype=int)[:n_pts]
            pts = pts[idx]
            print(f"Downsampled to {len(pts)} points")
        elif len(pts) < n_pts:
            print(f"Need {n_pts - len(pts)} more points, sampling from mesh...")
            np.random.seed(seed)  # Ensure deterministic fallback
            extra = mesh.sample(n_pts - len(pts))
            pts = np.vstack([pts, extra])
            print(f"Final point count: {len(pts)}")
    
    if save_debug:
        print(f"\nDebug data saved to {debug_dir}/:")
        print(f"  - depth_map_XXX.png: Depth maps as images")
        print(f"  - points_view_XXX.ply: Point clouds from each view")
        print(f"  - camera_info_XXX.npz: Camera parameters for each view")
    
    return np.ascontiguousarray(pts.astype(np.float32))


def render_depth_packets(mesh: trimesh.Trimesh,
                         directions: np.ndarray,
                         resolution: int = 512,
                         yfov_deg: float = 60.0):
    """Yield (depth, eye, pose, yfov, aspect, width, height) per view."""
    yfov = np.deg2rad(yfov_deg)
    aspect = 1.0

    center, radius, cam_distance, znear, zfar = _fit_camera_params(mesh, yfov, aspect)

    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    for prim in mesh_node.primitives:
        prim.material.doubleSided = True

    renderer = pyrender.OffscreenRenderer(resolution, resolution)

    for d in directions:
        d = d / (np.linalg.norm(d) + 1e-12)
        scene = pyrender.Scene(bg_color=[1, 1, 1, 0])

        scene.add(mesh_node)

        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(up, d)) > 0.9:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        eye = center + cam_distance * d
        pose = _pose_from_lookat(eye, center, up)

        cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect, znear=znear, zfar=zfar)
        scene.add(cam, pose=pose)

        try:
            _, depth = renderer.render(scene)
        except Exception:
            depth = np.zeros((resolution, resolution), dtype=np.float32)

        yield depth, eye.astype(np.float64), pose.astype(np.float64), yfov, aspect, resolution, resolution

    renderer.delete()


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
    """Return per-pixel ray directions (world) and camera-space v_z for a full image grid."""
    tx = np.tan(yfov * 0.5) * aspect
    ty = np.tan(yfov * 0.5)

    u = (np.arange(width) + 0.5) / width  * 2.0 - 1.0
    v = 1.0 - (np.arange(height) + 0.5) / height * 2.0
    U, V = np.meshgrid(u, v, indexing='xy')

    # Camera-space rays (before normalization, -Z is forward)
    dir_cam = np.stack([U * tx, V * ty, -np.ones_like(U)], axis=-1)
    dir_cam /= (np.linalg.norm(dir_cam, axis=-1, keepdims=True) + 1e-12)
    v_z = dir_cam[..., 2]  # <= negative

    # Rotate to world
    R = pose[:3, :3]
    dir_world = dir_cam @ R.T
    dir_world /= (np.linalg.norm(dir_world, axis=-1, keepdims=True) + 1e-12)

    eye_world = eye.reshape(1, 1, 3).repeat(height, axis=0).repeat(width, axis=1)
    return eye_world, dir_world, v_z


def backproject_depth_to_points(depth: np.ndarray,
                                eye: np.ndarray,
                                pose: np.ndarray,
                                yfov: float,
                                aspect: float,
                                width: int,
                                height: int,
                                stride: int = 2) -> np.ndarray:
    """
    Convert a depth map into a point cloud.
    Assumes `depth` is an OpenGL depth buffer in [0,1]. If your renderer already
    outputs linear camera-space depth (in world units), replace the call to
    `_linearize_depth_gl` with `z_lin = depth`.
    """
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")

    eye_grid, dir_world, v_z = _pixel_rays_world(eye, pose, yfov, aspect, width, height)

    # Valid mask
    valid = (depth > 0.0) # & (depth < 1.0)
    if stride > 1:
        valid[::stride, :] = valid[::stride, :]
        valid[:, ::stride] = valid[:, ::stride]
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    # 1) Linearize depth to camera-space distance along -Z (positive)
    z_lin = depth # _linearize_depth_gl(depth, znear, zfar)

    # 2) Ray parameter t so that p_cam = t * v_cam, with p_cam_z = v_z * t = -z_lin
    # => t = z_lin / (-v_z).  Clamp to avoid division by ~0 at grazing angles.
    denom = np.maximum(1e-8, -v_z)
    t = z_lin / denom
    t = t[..., None]  # broadcast

    # 3) Back-project
    pts = eye_grid + dir_world * t
    pts = pts[valid]
    return np.ascontiguousarray(pts.astype(np.float32))