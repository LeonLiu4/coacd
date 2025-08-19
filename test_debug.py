#!/usr/bin/env python3
"""
Simple test script to verify that breakpoints work in geometry.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.geometry import hausdorff, sample_surface_points_from_parts_fast
import torch
import numpy as np

def test_geometry_functions():
    print("Testing geometry functions...")
    
    # Test hausdorff function
    print("Testing hausdorff function...")
    a_pts = torch.randn(1, 100, 3)
    b_pts = torch.randn(1, 100, 3)
    result = hausdorff(a_pts, b_pts)
    print(f"Hausdorff result: {result}")
    
    # Test sample_surface_points_from_parts_fast function
    print("Testing sample_surface_points_from_parts_fast function...")
    # Create dummy parts (vertices and faces)
    parts = [
        (np.random.randn(100, 3).astype(np.float32), 
         np.random.randint(0, 100, (50, 3)).astype(np.int64))
    ]
    result = sample_surface_points_from_parts_fast(parts, n_pts=1000, seed=42)
    print(f"Sampling result shape: {result.shape}")
    
    print("Test completed!")

if __name__ == "__main__":
    test_geometry_functions()
