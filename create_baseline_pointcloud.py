#!/usr/bin/env python3
"""
Script to create a point cloud using baseline CoACD parameters with fast sampling.
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
from src.utils.geometry import sample_surface_points_from_parts_fast

def run_baseline_coacd_and_sample():
    """Run CoACD with baseline parameters and sample points from the result."""
    
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
    
    # Sample points from the decomposed parts using fast sampling
    print("\nSampling points from decomposed parts using fast sampling...")
    n_pts = 4096  # Same as training
    num_angles = 25  # Same as training
    
    try:
        points = sample_surface_points_from_parts_fast(
            parts, 
            n_pts=n_pts, 
            seed=42, 
            num_angles=num_angles
        )
        
        print(f"Successfully sampled {len(points)} points")
        
        # Save the point cloud
        output_npy = "baseline_pointcloud.npy"
        output_ply = "baseline_pointcloud.ply"
        
        np.save(output_npy, points)
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
        
        save_point_cloud_ply(points, output_ply)
        print(f"Saved point cloud to: {output_ply}")
        
        # Print some statistics
        print(f"\nPoint cloud statistics:")
        print(f"  Number of points: {len(points)}")
        print(f"  Bounding box: {np.min(points, axis=0)} to {np.max(points, axis=0)}")
        print(f"  Mean position: {np.mean(points, axis=0)}")
        
        return points
        
    except Exception as e:
        print(f"Error sampling points: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Creating baseline point cloud with CoACD parameters...")
    points = run_baseline_coacd_and_sample()
    
    if points is not None:
        print("\n✅ Successfully created baseline point cloud!")
        print("Files created:")
        print("  - baseline_pointcloud.npy (NumPy array)")
        print("  - baseline_pointcloud.ply (PLY format for visualization)")
    else:
        print("\n❌ Failed to create baseline point cloud.")
        sys.exit(1)
