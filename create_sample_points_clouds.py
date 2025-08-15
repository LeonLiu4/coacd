#!/usr/bin/env python3
"""
Create point clouds using sample_points function for comparison with raycast sampling.
"""
import numpy as np
import trimesh
import coacd
import os
from src.utils.geometry import sample_points, build_combined

def create_sample_points_clouds():
    """Create point clouds using sample_points function for both decompositions."""
    
    # Load mesh
    mesh = trimesh.load_mesh('assets/bunny_simplified.obj')
    coacd_mesh = coacd.Mesh(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int64),
    )
    
    print("Creating baseline decomposition (14 parts)...")
    baseline_parts = coacd.run_coacd(
        coacd_mesh, 
        threshold=0.1, 
        merge=True, 
        max_convex_hull=-1, 
        seed=42
    )
    
    print("Creating 3-part decomposition...")
    three_parts = coacd.run_coacd(
        coacd_mesh, 
        threshold=0.351, 
        merge=False, 
        max_convex_hull=100, 
        seed=42
    )
    
    # Sample points using sample_points function
    print("Sampling from baseline decomposition using sample_points...")
    baseline_combined = build_combined(baseline_parts)
    baseline_points = sample_points(baseline_combined, 100000, seed=42)
    
    print("Sampling from 3-part decomposition using sample_points...")
    three_combined = build_combined(three_parts)
    three_points = sample_points(three_combined, 100000, seed=42)
    
    # Create output directory
    os.makedirs('point_clouds', exist_ok=True)
    
    # Save as PLY files
    baseline_ply_path = 'point_clouds/baseline_14parts_sample.ply'
    three_part_ply_path = 'point_clouds/three_part_sample.ply'
    
    # Create PLY files with point cloud data
    def save_ply(points, filename):
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            for point in points:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"Saving baseline point cloud to {baseline_ply_path}")
    save_ply(baseline_points, baseline_ply_path)
    
    print(f"Saving 3-part point cloud to {three_part_ply_path}")
    save_ply(three_points, three_part_ply_path)
    
    # Also save as numpy arrays for convenience
    baseline_npy_path = 'point_clouds/baseline_14parts_sample.npy'
    three_part_npy_path = 'point_clouds/three_part_sample.npy'
    
    np.save(baseline_npy_path, baseline_points)
    np.save(three_part_npy_path, three_points)
    
    print(f"Saved numpy arrays to {baseline_npy_path} and {three_part_npy_path}")
    
    # Print statistics
    print("\n=== POINT CLOUD STATISTICS ===")
    print(f"Baseline ({len(baseline_parts)} parts): {len(baseline_points)} points")
    print(f"3-part ({len(three_parts)} parts): {len(three_points)} points")
    print(f"Original mesh: {len(mesh.vertices)} vertices")
    
    # Verify that points are from convex hulls (not original mesh)
    print("\n=== VERIFICATION ===")
    try:
        from trimesh.proximity import signed_distance
        
        # Check against combined convex hull meshes
        baseline_sd = signed_distance(baseline_combined, baseline_points)
        three_sd = signed_distance(three_combined, three_points)
        
        baseline_on_surface = np.sum(np.abs(baseline_sd) <= 1e-3)
        three_on_surface = np.sum(np.abs(three_sd) <= 1e-3)
        
        print(f"Baseline points on combined hull surface: {baseline_on_surface}/{len(baseline_points)} ({baseline_on_surface/len(baseline_points)*100:.1f}%)")
        print(f"3-part points on combined hull surface: {three_on_surface}/{len(three_points)} ({three_on_surface/len(three_points)*100:.1f}%)")
        
        # Check against original mesh
        baseline_original_sd = signed_distance(mesh, baseline_points)
        three_original_sd = signed_distance(mesh, three_points)
        
        baseline_outside = np.sum(baseline_original_sd > 1e-3)
        three_outside = np.sum(three_original_sd > 1e-3)
        
        print(f"Baseline points outside original mesh: {baseline_outside}/{len(baseline_points)} ({baseline_outside/len(baseline_points)*100:.1f}%)")
        print(f"3-part points outside original mesh: {three_outside}/{len(three_points)} ({three_outside/len(three_points)*100:.1f}%)")
        
        if baseline_on_surface > 0.9 * len(baseline_points) and three_on_surface > 0.9 * len(three_points):
            print("✅ SUCCESS: Both point clouds are on the combined hull surfaces!")
        else:
            print("❌ WARNING: Some points may not be on the surface!")
            
        if baseline_outside > 0 and three_outside > 0:
            print("✅ SUCCESS: Both point clouds represent convex hull approximations!")
        else:
            print("❌ WARNING: Point clouds may not represent convex hull approximations!")
            
    except ImportError:
        print("Could not verify convex hull sampling (trimesh.proximity not available)")
    
    print("\nPoint cloud files created successfully!")
    print("Files saved:")
    print(f"  - {baseline_ply_path}")
    print(f"  - {three_part_ply_path}")
    print(f"  - {baseline_npy_path}")
    print(f"  - {three_part_npy_path}")
    
    print("\n=== COMPARISON WITH RAYCAST SAMPLING ===")
    print("These point clouds use sample_points() function (area-uniform sampling)")
    print("Compare with raycast sampling files:")
    print("  - point_clouds/baseline_14parts_raycast.ply")
    print("  - point_clouds/three_part_raycast.ply")

if __name__ == "__main__":
    create_sample_points_clouds()
