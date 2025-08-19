#!/usr/bin/env python3
"""
Test CoACD with specific parameters from debug output.
"""

import sys
import os
import numpy as np
import trimesh
import coacd
import time
import torch

# Import our utilities
from src.utils.geometry import hausdorff, sample_points, sample_surface_points_from_parts_fast, build_combined

def test_coacd_params():
    """Test CoACD with specific parameters."""
    
    # Parameters from debug output
    threshold = 0.214
    no_merge = False
    max_hull = 90
    
    print(f"Testing CoACD with parameters:")
    print(f"  threshold = {threshold}")
    print(f"  no_merge = {no_merge}")
    print(f"  max_hull = {max_hull}")
    print()
    
    # Load mesh
    mesh_path = "assets/bunny_simplified.obj"
    mesh = trimesh.load(mesh_path)
    print(f"Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Create CoACD mesh
    coacd_mesh = coacd.Mesh(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int64),
    )
    
    # Run CoACD with timeout
    limit_sec = 10.0  # 10 second timeout
    t0 = time.time()
    
    try:
        parts = coacd.run_coacd(
            coacd_mesh,
            threshold=threshold,
            merge=not no_merge,
            max_convex_hull=max_hull,
            seed=42
        )
        
        runtime = time.time() - t0
        
        print(f"\nCoACD completed in {runtime:.3f}s")
        print(f"Result: {len(parts)} parts")
        
        if parts:
            total_verts = sum(len(verts) for verts, _ in parts)
            total_faces = sum(len(faces) for _, faces in parts)
            print(f"Total: {total_verts} vertices, {total_faces} faces")
            
            # Show each part
            for i, (verts, faces) in enumerate(parts):
                print(f"Part {i}: {len(verts)} vertices, {len(faces)} faces")
                if len(verts) == 0:
                    print(f"  WARNING: Part {i} has no vertices!")
        else:
            print("WARNING: No parts generated!")
            return None
            
    except Exception as e:
        runtime = time.time() - t0
        print(f"\nCoACD failed after {runtime:.3f}s")
        print(f"Error: {e}")
        return None
    
    # Calculate Hausdorff distance
    if parts:
        print(f"\n=== Calculating Hausdorff Distance ===")
        
        # Load baseline metrics for comparison
        import json
        try:
            with open('baseline_metrics.json', 'r') as f:
                baseline_metrics = json.load(f)
            print(f"Baseline Hausdorff: {baseline_metrics['hausdorff_distance']:.6f}")
        except:
            print("Could not load baseline metrics")
        
        # Sample points from the decomposed mesh
        print("Sampling points from decomposed mesh...")
        dec_pts = sample_surface_points_from_parts_fast(parts, 4096, seed=42, num_angles=25)
        print(f"Generated {len(dec_pts)} points")
        
        # Sample points from original mesh (baseline)
        print("Sampling points from original mesh...")
        orig_pts = sample_points(mesh, 4096, seed=42)
        print(f"Generated {len(orig_pts)} points")
        
        # Calculate Hausdorff distance
        print("Calculating Hausdorff distance...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dec_pts_tensor = torch.from_numpy(dec_pts.astype(np.float32))[None].to(device)
        orig_pts_tensor = torch.from_numpy(orig_pts.astype(np.float32))[None].to(device)
        
        hausdorff_dist = hausdorff(orig_pts_tensor, dec_pts_tensor)
        print(f"Hausdorff distance: {hausdorff_dist:.6f}")
        
        # Compare with baseline
        try:
            improvement = baseline_metrics['hausdorff_distance'] - hausdorff_dist
            if improvement > 0:
                print(f"✅ IMPROVEMENT: {improvement:.6f} better than baseline")
            else:
                print(f"❌ WORSE: {abs(improvement):.6f} worse than baseline")
        except:
            pass
    
    return parts

if __name__ == "__main__":
    test_coacd_params()
