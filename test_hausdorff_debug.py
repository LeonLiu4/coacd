#!/usr/bin/env python3
"""
Test script to debug infinite Hausdorff distance issues.
This will test each component separately to identify where the problem occurs.
"""

import sys
import os
import numpy as np
import trimesh
import torch
import coacd

# Add CoACD to path
sys.path.append('extern/CoACD')

# Import our utilities
from src.utils.geometry import hausdorff, sample_points, sample_surface_points_from_parts_fast, build_combined
from src.envs.coacd_env import CoACDEnv

def test_baseline_coacd():
    """Test CoACD with baseline parameters."""
    print("=== Testing Baseline CoACD ===")
    
    # Load mesh
    mesh_path = "assets/bunny_simplified.obj"
    mesh = trimesh.load(mesh_path)
    print(f"Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Baseline parameters
    threshold = 0.1
    no_merge = False
    max_hull = 10
    
    # Create CoACD mesh
    coacd_mesh = coacd.Mesh(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int64),
    )
    
    # Run CoACD
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        merge=not no_merge,
        max_convex_hull=max_hull,
        seed=42
    )
    
    print(f"CoACD result: {len(parts)} parts")
    if parts:
        total_verts = sum(len(verts) for verts, _ in parts)
        total_faces = sum(len(faces) for _, faces in parts)
        print(f"Total: {total_verts} vertices, {total_faces} faces")
        
        # Test each part individually
        for i, (verts, faces) in enumerate(parts):
            print(f"Part {i}: {len(verts)} vertices, {len(faces)} faces")
            if len(verts) == 0:
                print(f"ERROR: Part {i} has no vertices!")
    
    return parts

def test_point_sampling(parts):
    """Test point sampling from parts."""
    print("\n=== Testing Point Sampling ===")
    
    if not parts:
        print("ERROR: No parts to sample from!")
        return None
    
    try:
        # Test simple sampling first
        print("Testing simple sampling...")
        combined_mesh = build_combined(parts)
        simple_pts = sample_points(combined_mesh, 4096, seed=42)
        print(f"Simple sampling: {len(simple_pts)} points")
        print(f"Bounds: min={np.min(simple_pts, axis=0)}, max={np.max(simple_pts, axis=0)}")
        
        # Test depth-based sampling
        print("\nTesting depth-based sampling...")
        depth_pts = sample_surface_points_from_parts_fast(parts, 4096, seed=42, num_angles=25)
        print(f"Depth sampling: {len(depth_pts)} points")
        print(f"Bounds: min={np.min(depth_pts, axis=0)}, max={np.max(depth_pts, axis=0)}")
        
        return depth_pts
        
    except Exception as e:
        print(f"ERROR: Point sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_hausdorff_calculation(pts):
    """Test Hausdorff distance calculation."""
    print("\n=== Testing Hausdorff Calculation ===")
    
    if pts is None or len(pts) == 0:
        print("ERROR: No points for Hausdorff calculation!")
        return float('inf')
    
    try:
        # Create test point clouds
        print(f"Input points: {len(pts)} points")
        print(f"Input bounds: min={np.min(pts, axis=0)}, max={np.max(pts, axis=0)}")
        
        # Check for invalid values
        if np.any(np.isnan(pts)):
            print("ERROR: NaN values in input points!")
            return float('inf')
        
        if np.any(np.isinf(pts)):
            print("ERROR: Inf values in input points!")
            return float('inf')
        
        # Convert to torch tensors
        pts_tensor = torch.from_numpy(pts.astype(np.float32))[None]
        print(f"Torch tensor shape: {pts_tensor.shape}")
        
        # Test Hausdorff with itself (should be 0)
        print("Testing Hausdorff with same point cloud...")
        hausdorff_self = hausdorff(pts_tensor, pts_tensor)
        print(f"Hausdorff with self: {hausdorff_self}")
        
        # Test with slightly perturbed points
        print("Testing Hausdorff with perturbed points...")
        perturbed_pts = pts + np.random.normal(0, 0.01, pts.shape)
        perturbed_tensor = torch.from_numpy(perturbed_pts.astype(np.float32))[None]
        hausdorff_perturbed = hausdorff(pts_tensor, perturbed_tensor)
        print(f"Hausdorff with perturbed: {hausdorff_perturbed}")
        
        return hausdorff_perturbed
        
    except Exception as e:
        print(f"ERROR: Hausdorff calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

def test_environment_step():
    """Test a single environment step."""
    print("\n=== Testing Environment Step ===")
    
    try:
        # Create environment
        mesh_path = "assets/bunny_simplified.obj"
        env = CoACDEnv(mesh_path)
        print("Environment created successfully")
        
        # Test with baseline action
        action = np.array([0.0, 0.0, 0.0])  # Should give baseline parameters
        print(f"Testing with action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Environment step completed:")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Info: {info}")
        
        return info.get('H', float('inf'))
        
    except Exception as e:
        print(f"ERROR: Environment step failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

def main():
    """Run all tests."""
    print("=== Hausdorff Distance Debug Test ===")
    
    # Test 1: CoACD
    parts = test_baseline_coacd()
    
    # Test 2: Point sampling
    pts = test_point_sampling(parts)
    
    # Test 3: Hausdorff calculation
    hausdorff_result = test_hausdorff_calculation(pts)
    
    # Test 4: Environment step
    env_hausdorff = test_environment_step()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"CoACD parts: {len(parts) if parts else 0}")
    print(f"Point sampling: {'SUCCESS' if pts is not None and len(pts) > 0 else 'FAILED'}")
    print(f"Hausdorff calculation: {hausdorff_result}")
    print(f"Environment Hausdorff: {env_hausdorff}")
    
    if np.isinf(hausdorff_result):
        print("❌ Hausdorff calculation returned infinity!")
    else:
        print("✅ Hausdorff calculation successful")
    
    if np.isinf(env_hausdorff):
        print("❌ Environment Hausdorff returned infinity!")
    else:
        print("✅ Environment Hausdorff successful")

if __name__ == "__main__":
    main()
