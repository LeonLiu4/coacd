#!/usr/bin/env python3
"""
Diagnostic script to identify Hausdorff and NaN issues
"""
import numpy as np
import torch
import trimesh
import coacd
from src.utils.geometry import hausdorff, sample_points, sample_surface_points_from_parts_fast

def test_hausdorff_edge_cases():
    """Test various edge cases that could cause infinity Hausdorff"""
    print("=== Testing Hausdorff Edge Cases ===")
    
    # Test 1: Empty point clouds
    print("\n1. Testing empty point clouds...")
    empty_pts = torch.empty(1, 0, 3)
    normal_pts = torch.randn(1, 100, 3)
    result = hausdorff(empty_pts, normal_pts)
    print(f"Empty vs Normal: {result}")
    
    # Test 2: NaN values
    print("\n2. Testing NaN values...")
    nan_pts = torch.randn(1, 100, 3)
    nan_pts[0, 0, 0] = float('nan')
    result = hausdorff(nan_pts, normal_pts)
    print(f"NaN vs Normal: {result}")
    
    # Test 3: Inf values
    print("\n3. Testing Inf values...")
    inf_pts = torch.randn(1, 100, 3)
    inf_pts[0, 0, 0] = float('inf')
    result = hausdorff(inf_pts, normal_pts)
    print(f"Inf vs Normal: {result}")
    
    # Test 4: Very small point clouds
    print("\n4. Testing very small point clouds...")
    small_pts = torch.randn(1, 1, 3)
    result = hausdorff(small_pts, normal_pts)
    print(f"Small vs Normal: {result}")

def test_coacd_parameter_extremes():
    """Test CoACD with extreme parameters that might cause issues"""
    print("\n=== Testing CoACD Parameter Extremes ===")
    
    # Load the bunny mesh
    mesh = trimesh.load_mesh("assets/bunny_simplified.obj", process=False)
    coacd_mesh = coacd.Mesh(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int64),
    )
    
    # Test extreme parameters
    test_params = [
        {"threshold": 0.01, "merge": True, "max_hull": 10},   # Very low threshold
        {"threshold": 0.99, "merge": True, "max_hull": 10},   # Very high threshold
        {"threshold": 0.5, "merge": False, "max_hull": 1},    # Single hull, no merge
        {"threshold": 0.5, "merge": True, "max_hull": 100},   # Many hulls
    ]
    
    for i, params in enumerate(test_params):
        print(f"\nTest {i+1}: {params}")
        try:
            parts = coacd.run_coacd(
                coacd_mesh,
                threshold=params["threshold"],
                merge=params["merge"],
                max_convex_hull=params["max_hull"],
                seed=42,
            )
            
            print(f"  Parts generated: {len(parts)}")
            
            # Check each part
            for j, (vertices, faces) in enumerate(parts):
                print(f"  Part {j}: {len(vertices)} vertices, {len(faces)} faces")
                
                # Check for empty or invalid parts
                if len(vertices) == 0:
                    print(f"    WARNING: Part {j} has no vertices!")
                if len(faces) == 0:
                    print(f"    WARNING: Part {j} has no faces!")
                
                # Check for NaN/Inf in vertices
                if np.any(np.isnan(vertices)):
                    print(f"    WARNING: Part {j} has NaN vertices!")
                if np.any(np.isinf(vertices)):
                    print(f"    WARNING: Part {j} has Inf vertices!")
                
                # Test point sampling from this part
                if len(vertices) > 0 and len(faces) > 0:
                    try:
                        sampled_pts = sample_surface_points_from_parts_fast([(vertices, faces)], 1000, seed=42)
                        print(f"    Sampled {len(sampled_pts)} points successfully")
                        
                        # Check sampled points for issues
                        if np.any(np.isnan(sampled_pts)):
                            print(f"    WARNING: Sampled points contain NaN!")
                        if np.any(np.isinf(sampled_pts)):
                            print(f"    WARNING: Sampled points contain Inf!")
                            
                    except Exception as e:
                        print(f"    ERROR: Failed to sample points: {e}")
            
            # Test Hausdorff calculation
            if len(parts) > 0:
                try:
                    # Create a simple reference mesh for testing
                    ref_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                    ref_pts = sample_points(ref_mesh, 1000, seed=42)
                    ref_tensor = torch.from_numpy(ref_pts.astype(np.float32))[None]
                    
                    # Sample from parts
                    parts_pts = sample_surface_points_from_parts_fast(parts, 1000, seed=42)
                    parts_tensor = torch.from_numpy(parts_pts.astype(np.float32))[None]
                    
                    hausdorff_dist = hausdorff(ref_tensor, parts_tensor)
                    print(f"  Hausdorff distance: {hausdorff_dist}")
                    
                except Exception as e:
                    print(f"  ERROR: Failed to calculate Hausdorff: {e}")
            
        except Exception as e:
            print(f"  ERROR: CoACD failed: {e}")

def test_neural_network_stability():
    """Test for potential neural network instability issues"""
    print("\n=== Testing Neural Network Stability ===")
    
    # Test 1: Check for gradient explosion
    print("\n1. Testing for gradient explosion...")
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simulate extreme gradients
    for i in range(100):
        optimizer.zero_grad()
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        # Add some extreme values to simulate instability
        if i % 10 == 0:
            x[0, 0] = float('inf')  # Simulate inf input
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"  WARNING: NaN loss detected at step {i}")
            break
            
        loss.backward()
        
        # Check for gradient explosion
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        if total_norm > 1000:
            print(f"  WARNING: Large gradients detected at step {i}: {total_norm}")
        
        optimizer.step()
    
    print("  Neural network stability test completed")

def main():
    """Run all diagnostic tests"""
    print("Starting diagnostic tests for Hausdorff and NaN issues...")
    
    test_hausdorff_edge_cases()
    test_coacd_parameter_extremes()
    test_neural_network_stability()
    
    print("\n=== Diagnostic Summary ===")
    print("Common causes of infinity Hausdorff:")
    print("1. Empty point clouds from CoACD failures")
    print("2. NaN/Inf values in mesh vertices")
    print("3. Extreme CoACD parameters (threshold too high/low)")
    print("4. Degenerate meshes (all vertices collapse)")
    
    print("\nCommon causes of NaN explained variance:")
    print("1. Gradient explosion in neural network")
    print("2. Invalid loss values")
    print("3. Numerical instability in optimization")
    print("4. Inf/NaN inputs to the network")
    
    print("\nRecommendations:")
    print("1. Add parameter bounds to prevent extreme values")
    print("2. Add gradient clipping to prevent explosion")
    print("3. Add validation checks for CoACD outputs")
    print("4. Use more conservative learning rates")

if __name__ == "__main__":
    main()
