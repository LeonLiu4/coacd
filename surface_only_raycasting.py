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

from src.utils.geometry import (
    sample_surface_points_via_depth,
)


def main():
    parser = argparse.ArgumentParser(description="Depth-based surface sampling from convex hulls")
    parser.add_argument("--mesh", type=str, default="assets/bunny_simplified.obj")
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--dirs", type=int, default=100, help="Number of viewing directions for depth rendering")
    parser.add_argument("--resolution", type=int, default=1024, help="Depth map resolution (width=height)")
    parser.add_argument("--yfov", type=float, default=60.0, help="Camera field of view in degrees")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for point deduplication")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rays-fallback", action="store_true", help="Use ray sampling instead of depth-based sampling")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--save-debug", action="store_true", help="Save depth maps and per-view point clouds for debugging")
    parser.add_argument("--use-meshlab", action="store_true", default=True, help="Use MeshLab for uniform point cloud simplification")
    # Legacy ray sampling parameters (only used with --rays-fallback)
    parser.add_argument("--rays-per-dir", type=int, default=10, help="Rays per direction (only for ray sampling)")
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
        pass
    else:
        print("Sampling outer surface via depth-based rendering...")
        pts = sample_surface_points_via_depth(parts, n_pts=args.points, num_dirs=args.dirs,
                                              resolution=args.resolution, yfov_deg=args.yfov,
                                              tol=args.tol, seed=args.seed, save_debug=args.save_debug,
                                              use_meshlab_simplification=args.use_meshlab)

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