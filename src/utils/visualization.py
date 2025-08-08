#!/usr/bin/env python3
"""
Interactive Pyrender viewer for CoACD mesh visualization.
Provides real-time interaction with input and decomposed meshes.
"""

import pyrender
import numpy as np
import trimesh
import coacd
import os
import sys
from typing import List, Tuple, Optional, Dict


def _look_at(camera_position: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """
    Create a 4x4 camera pose matrix that looks from camera_position to target with the given up vector.
    """
    camera_position = np.asarray(camera_position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    z_axis = target - camera_position
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        z_axis = np.array([0, 0, 1], dtype=np.float32)
    else:
        z_axis /= z_norm

    x_axis = np.cross(z_axis, up)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        # up was parallel to z; pick a fallback up
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x_axis = np.cross(z_axis, up)
        x_axis /= (np.linalg.norm(x_axis) + 1e-8)
    else:
        x_axis /= x_norm

    y_axis = np.cross(x_axis, z_axis)

    pose = np.eye(4, dtype=np.float32)
    pose[0, :3] = x_axis
    pose[1, :3] = y_axis
    pose[2, :3] = z_axis
    pose[:3, 3] = camera_position
    return pose


class PyrenderInteractiveViewer:
    """Interactive Pyrender viewer for mesh visualization."""

    def __init__(self):
        # Create scene
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        self.geometries = []  # list of (pyrender.Mesh, name)
        # Track combined bounds for camera positioning
        self._bbox_min = None
        self._bbox_max = None

    def _update_bounds_from_trimesh(self, mesh: trimesh.Trimesh) -> None:
        bmin, bmax = mesh.bounds  # (2,3)
        if self._bbox_min is None:
            self._bbox_min = bmin.copy()
            self._bbox_max = bmax.copy()
        else:
            self._bbox_min = np.minimum(self._bbox_min, bmin)
            self._bbox_max = np.maximum(self._bbox_max, bmax)

    def add_mesh(self, mesh: trimesh.Trimesh, color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
                 name: str = "Mesh") -> None:
        """Add a mesh to the viewer."""
        # Convert trimesh to pyrender mesh
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)

        # Create material with proper transparency
        if "Convex_Part" in name or "Decomposed" in name or "Part_" in name:
            # Transparent material for convex hull parts
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[color[0], color[1], color[2], 0.25],  # 25% opacity
                metallicFactor=0.0,
                roughnessFactor=0.7,
                alphaMode='BLEND'  # Enable transparency
            )
        else:
            # Opaque material for input mesh
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[color[0], color[1], color[2], 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.7
            )

        # Create mesh
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)

        # Add to scene
        self.scene.add(mesh_pyrender)
        self.geometries.append((mesh_pyrender, name))

        # Update combined bounds (for camera framing)
        self._update_bounds_from_trimesh(mesh)

        print(f"Added {name} with {len(vertices)} vertices and {len(faces)} faces")

    def add_mesh_parts(self, parts: List[trimesh.Trimesh], colors: Optional[List[Tuple[float, float, float]]] = None) -> None:
        """Add multiple mesh parts with different colors."""
        if colors is None:
            colors = [
                (0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.8, 0.8, 0.2),
                (0.8, 0.2, 0.8), (0.2, 0.8, 0.8), (0.8, 0.4, 0.2),
                (0.4, 0.8, 0.2), (0.2, 0.4, 0.8), (0.8, 0.2, 0.4)
            ]

        for i, part in enumerate(parts):
            color = colors[i % len(colors)]
            self.add_mesh(part, color, f"Convex_Part_{i+1}")

    def _add_centered_camera(self) -> None:
        """Add a camera that centers the view on the scene."""
        # Calculate appropriate camera distance based on scene bounds
        if self._bbox_min is not None and self._bbox_max is not None:
            extents = self._bbox_max - self._bbox_min
            max_extent = np.max(extents)
            # Use a distance that ensures the object is visible
            camera_distance = max(2.0, max_extent * 3.0)
            print(f"Scene bounds: {self._bbox_min} to {self._bbox_max}")
            print(f"Scene extents: {extents}, max_extent: {max_extent}")
            print(f"Camera distance: {camera_distance}")
        else:
            camera_distance = 2.0
            print("No bounds available, using default camera distance")
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.5)
        
        # Adjust camera pose to move view down and left
        # Move camera much more right and up to make the object appear way more down/left
        camera_x = 1.2  # Move camera much more right to shift view way more left
        camera_y = 0.8  # Move camera much more up to shift view way more down
        
        camera_pose = np.array([
            [1.0, 0.0, 0.0, camera_x],     # Right vector + x offset
            [0.0, 1.0, 0.0, camera_y],     # Up vector + y offset  
            [0.0, 0.0, 1.0, camera_distance],  # Forward vector (camera at z=distance)
            [0.0, 0.0, 0.0, 1.0]           # Homogeneous
        ])
        
        self.scene.add(camera, pose=camera_pose)
        
        # Add lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # Position light with camera
        self.scene.add(light, pose=camera_pose)

    def run_interactive(self) -> None:
        """Run the interactive viewer."""
        print("\n=== Pyrender Interactive Viewer ===")
        print("Controls:")
        print("  - Left mouse: Rotate view")
        print("  - Right mouse: Zoom")
        print("  - Middle mouse or Shift+Left: Pan")
        print("  - Mouse wheel: Zoom")
        print("  - Q: Quit viewer")
        print("  - Z: Reset camera")
        print("  - W: Toggle wireframe")
        print("  - S: Save screenshot")
        print("  - R: Record GIF")
        print("  - H: Toggle shadows")
        print("  - L: Toggle lighting")
        print("  - I: Toggle axes")
        print("  - O: Toggle orthographic")
        print()

        # Add a camera positioned to center the view
        self._add_centered_camera()
        
        # Create viewer with the scene
        viewer = pyrender.Viewer(
            self.scene,
            viewport_size=(1200, 800),
            use_raymond_lighting=False,  # We'll add our own light
            render_flags={'shadows': True, 'cull_faces': True},
            viewer_flags={'rotate': True, 'zoom': True, 'pan': True}
        )

        # The viewer will run until closed
        viewer.close()


def visualize_best_decomposition(mesh_path: str,
                                 best_params: Dict,
                                 save_dir: str = "visualizations") -> None:
    """
    Create an interactive Pyrender visualization of the mesh decomposition.
    """
    # Load mesh
    print(f"Loading mesh from: {mesh_path}")
    mesh = trimesh.load_mesh(mesh_path, process=False)

    # Ensure mesh has clean colors (remove any existing vertex/face colors)
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        mesh.visual.vertex_colors = None
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
        mesh.visual.face_colors = None

    # Store the mesh center for later translation, but run CoACD on original coordinates
    mesh_center = mesh.centroid
    
    # Create CoACD mesh from original (non-translated) mesh
    coacd_mesh = coacd.Mesh(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int64),
    )

    # Run CoACD with best parameters (suppressed output)
    threshold = best_params.get('threshold', 0.05)
    no_merge = best_params.get('no_merge', False)
    max_hull = best_params.get('max_hull', 10)

    print("Running CoACD with parameters:")
    print(f"  Threshold: {threshold}")
    print(f"  No merge: {no_merge}")
    print(f"  Max hull: {max_hull}")

    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        old_stdout_fd, old_stderr_fd = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            parts = coacd.run_coacd(
                coacd_mesh,
                threshold=threshold,
                merge=not no_merge,
                max_convex_hull=max_hull,
            )
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            sys.stdout, sys.stderr = old_stdout, old_stderr

    if not parts:
        print("Decomposition failed - no parts generated")
        return

    # Create viewer
    viewer = PyrenderInteractiveViewer()

    # Add input mesh (opaque solid blue) - translate to center
    mesh.apply_translation(-mesh_center)
    viewer.add_mesh(mesh, color=(0.1, 0.4, 0.9), name="Input_Mesh")

    # Add individual convex hull parts with distinct colors and transparency
    np.random.seed(42)
    colors = []
    for i in range(len(parts)):
        hue = i / max(1, len(parts))
        saturation = 0.7 + 0.3 * np.random.random()
        value = 0.8 + 0.2 * np.random.random()

        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        r, g, b = r + m, g + m, b + m
        colors.append((r, g, b))

    # Add convex hull parts - they should already be in the same coordinate space as the input mesh
    for i, (verts, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        # Apply the same translation to center the parts (same as input mesh)
        part_mesh.apply_translation(-mesh_center)
        viewer.add_mesh(part_mesh, color=colors[i], name=f"Convex_Part_{i+1}")

    print(f"Created {len(parts)} convex hull parts")
    print("Starting interactive viewer...")
    viewer.run_interactive()


def demo_interactive_viewer():
    """Demo the interactive Pyrender viewer."""
    mesh_path = "assets/bunny_simplified.obj"

    test_params = {
        "threshold": 0.05,
        "no_merge": False,
        "max_hull": 10
    }

    print("=== Pyrender Interactive Viewer Demo ===")
    visualize_best_decomposition(
        mesh_path=mesh_path,
        best_params=test_params,
        save_dir="visualizations"
    )


if __name__ == "__main__":
    demo_interactive_viewer()