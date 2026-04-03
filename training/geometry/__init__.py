"""
LevelSmith Geometry Module

Modular geometry generation system split from generate_level.py.
Provides building blocks for procedural 3D level generation.

Modules:
- core: Main interfaces for building and scene generation
- primitives: Basic geometric primitives (walls, doors, windows, roofs)
- materials: Material and color systems
- layout: Building footprint and placement
- export: GLB/FBX export utilities
- utils: Helper functions and math utilities
"""

__version__ = "1.0.0"
__all__ = ["core", "primitives", "materials", "layout", "export", "utils"]