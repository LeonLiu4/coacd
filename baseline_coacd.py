#!/usr/bin/env python3
"""
Fixed baseline CoACD script that handles PyRender display issues.
"""
import os
import time
import json
import sys

# Disable CUDA and set display environment
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["DISPLAY"] = ":99"  # Use Xvfb display
os.environ["PYOPENGL_PLATFORM"] = "egl"  # Use EGL instead of X11

def run_baseline_coacd(mesh_path: str, output_file: str = "baseline_metrics.json"):
    """Run CoACD with default parameters and save baseline metrics."""
    print(f"Running baseline CoACD on: {mesh_path}")
    
    # Load mesh
    import trimesh
    mesh = trimesh.load_mesh(mesh_path, process=False)
    print(f"Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Create CoACD mesh
    import coacd
    import numpy as np
    coacd_mesh = coacd.Mesh(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int64),
    )
    
    # Default parameters
    default_params = {
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
    
    print("Running with default parameters:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")
    
    # Run CoACD with timing
    start_time = time.time()
    try:
        parts = coacd.run_coacd(
            coacd_mesh,
            threshold=default_params['threshold'],
            max_convex_hull=default_params['max_convex_hull'],
            preprocess_mode=default_params['preprocess_mode'],
            preprocess_resolution=default_params['preprocess_resolution'],
            resolution=default_params['resolution'],
            mcts_nodes=default_params['mcts_nodes'],
            mcts_iterations=default_params['mcts_iterations'],
            mcts_max_depth=default_params['mcts_max_depth'],
            pca=default_params['pca'],
            merge=default_params['merge'],
            decimate=default_params['decimate'],
            seed=42  # Fixed seed for deterministic results
        )
        runtime = time.time() - start_time
        success = True
    except Exception as e:
        print(f"CoACD failed: {e}")
        runtime = time.time() - start_time
        parts = []
        success = False
    
    if success and parts:
        # Calculate metrics
        total_vertices = sum(len(verts) for verts, _ in parts)
        total_faces = sum(len(faces) for _, faces in parts)
        num_parts = len(parts)
        
        # Calculate Hausdorff distance using the same method as training environment
        # Combine all parts into one mesh for comparison
        all_vertices = []
        all_faces = []
        face_offset = 0
        
        for verts, faces in parts:
            all_vertices.append(verts)
            all_faces.append(faces + face_offset)
            face_offset += len(verts)
        
        if all_vertices:
            combined_vertices = np.vstack(all_vertices)
            combined_faces = np.vstack(all_faces)
            
            # Use the same sampling method as the training environment
            np.random.seed(42)
            from src.utils.geometry import sample_points, sample_surface_points_from_parts_fast, hausdorff
            
            # Sample fixed evaluation points from original mesh (same as training)
            fixed_eval_pts = sample_points(mesh, 4096)
            
            # Sample points from the decomposed mesh using the fast surface-only approach
            # This matches exactly what the training environment does
            reconstructed_pts = sample_surface_points_from_parts_fast(parts, 4096, seed=42, num_angles=25)
            
            # Force CPU tensors
            import torch
            torch.set_default_device('cpu')
            fixed_eval_tensor = torch.from_numpy(fixed_eval_pts).unsqueeze(0).cpu()
            reconstructed_tensor = torch.from_numpy(reconstructed_pts).unsqueeze(0).cpu()
            
            hausdorff_dist = hausdorff(fixed_eval_tensor, reconstructed_tensor)
        else:
            hausdorff_dist = float('inf')
        
        baseline_metrics = {
            'parameters': default_params,
            'success': success,
            'runtime': runtime,
            'num_parts': num_parts,
            'total_vertices': total_vertices,
            'total_faces': total_faces,
            'hausdorff_distance': hausdorff_dist,
            'input_vertices': len(mesh.vertices),
            'input_faces': len(mesh.faces)
        }
        
        print(f"\nBaseline Results:")
        print(f"  Runtime: {runtime:.3f}s")
        print(f"  Parts: {num_parts}")
        print(f"  Total vertices: {total_vertices}")
        print(f"  Total faces: {total_faces}")
        print(f"  Hausdorff distance: {hausdorff_dist:.6f}")
        
    else:
        baseline_metrics = {
            'parameters': default_params,
            'success': False,
            'runtime': runtime,
            'num_parts': 0,
            'total_vertices': 0,
            'total_faces': 0,
            'hausdorff_distance': float('inf'),
            'input_vertices': len(mesh.vertices),
            'input_faces': len(mesh.faces)
        }
        print(f"\nBaseline Failed:")
        print(f"  Runtime: {runtime:.3f}s")
    
    # Save baseline metrics
    with open(output_file, 'w') as f:
        json.dump(baseline_metrics, f, indent=2)
    
    print(f"\nBaseline metrics saved to: {output_file}")
    return baseline_metrics


if __name__ == "__main__":
    baseline = run_baseline_coacd("assets/bunny_simplified.obj")
