#!/usr/bin/env python3
"""
Depth-first surface sampling from convex hulls.
"""

import os as _os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    _os.environ.setdefault(_k, "1")

import argparse
import os
import numpy as np
import trimesh
import coacd
import pyrender

from src.utils.geometry import (
    enhanced_viewing_dirs,
    build_combined,
    _pose_from_lookat,
    _fit_camera_params,
    backproject_depth_to_points,
    ray_sample_surface,
)

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


def sample_surface_points_via_depth(parts,
                                    n_pts: int,
                                    num_dirs: int = 100,
                                    resolution: int = 512,
                                    yfov_deg: float = 60.0,
                                    tol: float = 1e-3,
                                    seed: int = 42) -> np.ndarray:
    """Depth-first sampling on the combined convex hulls."""
    rng = np.random.default_rng(seed)

    mesh = build_combined(parts)
    directions = enhanced_viewing_dirs(num_dirs)

    per_view_target = max(256, int(np.ceil(n_pts / max(1, num_dirs))))
    stride_guess = max(1, int(np.sqrt((resolution * resolution) / (per_view_target * 1.25))))

    all_pts = []

    for depth, eye, pose, yfov, aspect, W, H in render_depth_packets(mesh, directions, resolution, yfov_deg):
        if not np.any(depth > 0):
            continue

        pts = backproject_depth_to_points(depth, eye, pose, yfov, aspect, W, H, stride=stride_guess)
        if pts.size:
            all_pts.append(pts)

    if not all_pts:
        return mesh.sample(n_pts).astype(np.float32)

    pts = np.vstack(all_pts)

    # Deduplicate and fit to exactly n_pts
    q = np.round(pts / tol).astype(np.int64)
    _, keep = np.unique(q, axis=0, return_index=True)
    pts = pts[np.sort(keep)]

    if len(pts) > n_pts:
        step = len(pts) / n_pts
        idx = np.arange(0, len(pts), step, dtype=int)[:n_pts]
        pts = pts[idx]
    elif len(pts) < n_pts:
        extra = mesh.sample(n_pts - len(pts))
        pts = np.vstack([pts, extra])

    return np.ascontiguousarray(pts.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Depth-first surface sampling from convex hulls")
    parser.add_argument("--mesh", type=str, default="assets/bunny_simplified.obj")
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--dirs", type=int, default=100)
    parser.add_argument("--rays-per-dir", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--yfov", type=float, default=60.0)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rays-fallback", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    base_mesh = trimesh.load(args.mesh, process=False)
    if base_mesh.is_empty or len(base_mesh.faces) == 0:
        raise ValueError("Loaded mesh is empty or has no faces.")
    print(f"Input mesh: {len(base_mesh.vertices)} vtx, {len(base_mesh.faces)} faces")

    print("Running CoACD decomposition...")
    coacd_mesh = coacd.Mesh(base_mesh.vertices.astype(np.float32),
                            base_mesh.faces.astype(np.int64))
    parts = coacd.run_coacd(coacd_mesh, threshold=args.threshold, merge=True, max_convex_hull=-1)
    print(f"CoACD produced {len(parts)} convex parts")

    if args.rays_fallback:
        print("Using legacy ray sampler...")
        pts = ray_sample_surface(parts, n_pts=args.points, num_dirs=args.dirs,
                                 rays_per_dir=args.rays_per_dir, tol=args.tol, seed=args.seed,
                                 outer_mesh=base_mesh)
    else:
        print("Sampling outer surface via first-hit rays (filtered by original mesh)...")
        pts = ray_sample_surface(parts, n_pts=args.points, num_dirs=args.dirs,
                                 rays_per_dir=args.rays_per_dir, tol=args.tol, seed=args.seed,
                                 outer_mesh=base_mesh)

    print(f"Collected {len(pts)} points on outer surface")

    os.makedirs("point_clouds", exist_ok=True)
    np.save("point_clouds/surface_raycast_points.npy", pts)
    with open("point_clouds/surface_raycast_points.ply", "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    print("Saved: point_clouds/surface_raycast_points.(npy|ply)")


if __name__ == "__main__":
    main()