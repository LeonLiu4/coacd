#!/usr/bin/env python3
"""
Cleanup script to remove all runtime-generated files and directories.
This ensures a clean state for fresh runs.
"""

import os
import shutil
import glob
from pathlib import Path


def cleanup_directory(directory_path: str, description: str):
    """Remove a directory if it exists."""
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"✓ Removed {description}: {directory_path}")
        except Exception as e:
            print(f"✗ Failed to remove {description}: {e}")
    else:
        print(f"- {description} not found: {directory_path}")


def cleanup_files(pattern: str, description: str):
    """Remove files matching a pattern."""
    files = glob.glob(pattern)
    if files:
        for file in files:
            try:
                os.remove(file)
                print(f"✓ Removed {description}: {file}")
            except Exception as e:
                print(f"✗ Failed to remove {description}: {e}")
    else:
        print(f"- No {description} found")


def cleanup_pycache():
    """Remove all __pycache__ directories."""
    pycache_dirs = []
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_dirs.append(os.path.join(root, dir_name))
    
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"✓ Removed __pycache__: {pycache_dir}")
        except Exception as e:
            print(f"✗ Failed to remove __pycache__: {e}")


def cleanup_py_files():
    """Remove .pyc files."""
    pyc_files = glob.glob("**/*.pyc", recursive=True)
    for pyc_file in pyc_files:
        try:
            os.remove(pyc_file)
            print(f"✓ Removed .pyc file: {pyc_file}")
        except Exception as e:
            print(f"✗ Failed to remove .pyc file: {e}")


def main():
    """Main cleanup function."""
    print("🧹 CLEANING UP RUNTIME FILES")
    print("=" * 50)
    
    # Directories created during training
    cleanup_directory("logs", "training logs directory")
    cleanup_directory("models", "models directory")
    cleanup_directory("visualizations", "visualizations directory")
    cleanup_directory("point_clouds", "point clouds directory")
    
    # TensorBoard logs
    cleanup_directory("logs/ppo_pointnet", "TensorBoard logs")
    
    # Model files
    cleanup_files("models/*.zip", "model checkpoint files")
    cleanup_files("models/best/*.zip", "best model files")
    
    # Visualization files
    cleanup_files("visualizations/*.png", "visualization images")
    cleanup_files("visualizations/*.jpg", "visualization images")
    cleanup_files("visualizations/*.gif", "visualization animations")
    
    # Point cloud files
    cleanup_files("point_clouds/*.ply", "point cloud files")
    cleanup_files("point_clouds/*.npy", "numpy point cloud files")
    
    # Python cache files
    cleanup_pycache()
    cleanup_py_files()
    
    # Temporary files
    cleanup_files("*.tmp", "temporary files")
    cleanup_files("*.temp", "temporary files")
    cleanup_files("*.log", "log files")
    
    # Jupyter notebook checkpoints
    cleanup_directory(".ipynb_checkpoints", "Jupyter notebook checkpoints")
    
    # IDE files
    cleanup_files(".vscode/settings.json", "VSCode settings")
    cleanup_files(".idea/*", "PyCharm files")
    
    # OS files
    cleanup_files(".DS_Store", "macOS system files")
    cleanup_files("Thumbs.db", "Windows system files")
    
    # CoACD output files
    cleanup_files("*.obj", "CoACD output meshes")
    cleanup_files("decomp.obj", "decomposition output")
    
    # Training artifacts
    cleanup_files("*.pth", "PyTorch model files")
    cleanup_files("*.pt", "PyTorch model files")
    cleanup_files("*.ckpt", "checkpoint files")
    
    print("\n" + "=" * 50)
    print("✅ CLEANUP COMPLETE")
    print("=" * 50)
    print("Removed all runtime-generated files and directories.")
    print("Your project is now clean and ready for fresh runs!")
    
    # Show what remains
    print("\n📁 REMAINING PROJECT FILES:")
    remaining_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')]
    remaining_files = [f for f in os.listdir('.') if os.path.isfile(f) and not f.startswith('.')]
    
    if remaining_dirs:
        print("Directories:")
        for d in sorted(remaining_dirs):
            print(f"  📁 {d}/")
    
    if remaining_files:
        print("Files:")
        for f in sorted(remaining_files):
            print(f"  📄 {f}")


if __name__ == "__main__":
    main()
