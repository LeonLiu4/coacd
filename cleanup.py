#!/usr/bin/env python3
"""
Cleanup script to remove generated files and directories.
"""

import os
import shutil
import glob


def cleanup_generated_files():
    """Remove all generated files and directories."""
    
    # Files and directories to remove
    items_to_remove = [
        "visualizations/",
        "logs/",
        "models/",
        "__pycache__/",
        "src/__pycache__/",
        "src/envs/__pycache__/",
        "src/models/__pycache__/",
        "src/utils/__pycache__/",
        "*.pyc",
        "*.pyo",
        ".pytest_cache/",
        ".coverage",
        "htmlcov/",
        ".mypy_cache/",
        ".ruff_cache/"
    ]
    
    print("Cleaning up generated files...")
    
    for item in items_to_remove:
        if item.endswith('/'):
            # Directory
            if os.path.exists(item):
                shutil.rmtree(item)
                print(f"✓ Removed directory: {item}")
        else:
            # File pattern
            files = glob.glob(item)
            for file in files:
                try:
                    os.remove(file)
                    print(f"✓ Removed file: {file}")
                except Exception as e:
                    print(f"✗ Could not remove {file}: {e}")
    
    print("Cleanup completed!")


if __name__ == "__main__":
    cleanup_generated_files()
