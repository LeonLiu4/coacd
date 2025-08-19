#!/usr/bin/env python3
"""
Script to create point clouds from baseline CoACD convex hulls using simple sampling.
This uses the original sample_points function (not depth rendering).
"""

import sys
import os
import numpy as np
import trimesh
import json
from pathlib import Path

# Add CoACD to path
sys.path.append('extern/CoACD')

# Import our utilities
from src.utils.geometry import sample_points

def run_baseline_coacd_and_simple_sample():
    """Run CoACD with baseline parameters and sample points using simple sampling."""
    
    print("Loading mesh...")
    mesh_path = "assets/bunny_simplified.obj"
    mesh = trimesh.load(mesh_path)
    print(f"Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Baseline CoACD parameters
    baseline_params = {
        'threshold': 0.1,
        'max_convex_hull': -1,
        'preprocess_mode': 'auto',
        'preprocess_resolution': 30,
        'resolution': 1000,
        'mcts_nodes': 20,
        'mcts_iterations': 100,
        'mcts_max_depth': 3,
        'pca': False,
        'merge': True,
        'decimate': False
    }
    
    print("Running CoACD with baseline parameters:")
    for key, value in baseline_params.items():
        print(f"  {key}: {value}")
    
    # Import CoACD
    try:
        import coacd
    except ImportError:
        print("Error: CoACD not found. Make sure you're in the coacd_clean environment.")
        return None
    
    # Run CoACD
    print("\nRunning CoACD decomposition...")
    # Create CoACD mesh
    coacd_mesh = coacd.Mesh(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int64),
    )
    
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=baseline_params['threshold'],
        max_convex_hull=baseline_params['max_convex_hull'],
        preprocess_mode=baseline_params['preprocess_mode'],
        preprocess_resolution=baseline_params['preprocess_resolution'],
        resolution=baseline_params['resolution'],
        mcts_nodes=baseline_params['mcts_nodes'],
        mcts_iterations=baseline_params['mcts_iterations'],
        mcts_max_depth=baseline_params['mcts_max_depth'],
        pca=baseline_params['pca'],
        merge=baseline_params['merge'],
        decimate=baseline_params['decimate'],
        seed=42  # Fixed seed for deterministic results
    )
    
    print(f"CoACD completed: {len(parts)} parts")
    
    # Sample points from each part using simple sampling
    print("\nSampling points from each convex hull part using simple sampling...")
    n_pts_per_part = 4096 // len(parts)  # Distribute points evenly
    remaining_pts = 4096 % len(parts)    # Distribute remainder
    
    all_points = []
    part_info = []
    
    for i, (verts, faces) in enumerate(parts):
        # Create trimesh for this part
        part_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        
        # Calculate points for this part
        if i < remaining_pts:
            pts_for_this_part = n_pts_per_part + 1
        else:
            pts_for_this_part = n_pts_per_part
            
        # Sample points using simple sampling
        points = sample_points(part_mesh, pts_for_this_part, seed=42 + i)
        
        all_points.append(points)
        part_info.append({
            'part_id': i,
            'vertices': len(verts),
            'faces': len(faces),
            'points_sampled': len(points)
        })
        
        print(f"  Part {i+1}: {len(verts)} vertices, {len(faces)} faces → {len(points)} points")
    
    # Combine all points
    combined_points = np.vstack(all_points)
    print(f"\nTotal points sampled: {len(combined_points)}")
    
    # Save the point cloud
    output_npy = "simple_baseline_pointcloud.npy"
    output_ply = "simple_baseline_pointcloud.ply"
    
    np.save(output_npy, combined_points)
    print(f"Saved point cloud to: {output_npy}")
    
    # Also save as PLY for visualization
    def save_point_cloud_ply(points, filename):
        """Save point cloud as PLY file."""
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    save_point_cloud_ply(combined_points, output_ply)
    print(f"Saved point cloud to: {output_ply}")
    
    # Save part information
    part_info_file = "simple_baseline_part_info.json"
    with open(part_info_file, 'w') as f:
        json.dump(part_info, f, indent=2)
    print(f"Saved part information to: {part_info_file}")
    
    # Print some statistics
    print(f"\nPoint cloud statistics:")
    print(f"  Number of points: {len(combined_points)}")
    print(f"  Number of parts: {len(parts)}")
    print(f"  Bounding box: {np.min(combined_points, axis=0)} to {np.max(combined_points, axis=0)}")
    print(f"  Mean position: {np.mean(combined_points, axis=0)}")
    
    # Print part breakdown
    print(f"\nPart breakdown:")
    for info in part_info:
        print(f"  Part {info['part_id']+1}: {info['vertices']} vertices, {info['faces']} faces, {info['points_sampled']} points")
    
    return combined_points, part_info

if __name__ == "__main__":
    print("Creating simple baseline point cloud from CoACD convex hulls...")
    result = run_baseline_coacd_and_simple_sample()
    
    if result is not None:
        points, part_info = result
        print("\n✅ Successfully created simple baseline point cloud!")
        print("Files created:")
        print("  - simple_baseline_pointcloud.npy (NumPy array)")
        print("  - simple_baseline_pointcloud.ply (PLY format for visualization)")
        print("  - simple_baseline_part_info.json (Part information)")
        print(f"\nTotal points: {len(points)}")
        print(f"Total parts: {len(part_info)}")
    else:
        print("\n❌ Failed to create simple baseline point cloud.")
        sys.exit(1)
