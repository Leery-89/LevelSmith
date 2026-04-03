"""
Utility functions for geometry operations.

Math, conversion, validation, and debugging utilities.
"""

import math
import numpy as np
import trimesh


def deg_to_rad(degrees):
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def rad_to_deg(radians):
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi


def rotate_points_2d(points, angle_deg, center=(0, 0)):
    """
    Rotate 2D points around a center.
    
    Args:
        points: Nx2 array of points
        angle_deg: Rotation angle in degrees
        center: Rotation center (x, y)
    
    Returns:
        Rotated points
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    angle_rad = deg_to_rad(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Translate to origin
    translated = points - np.array(center)
    
    # Rotate
    rotated = np.zeros_like(translated)
    rotated[:, 0] = translated[:, 0] * cos_a - translated[:, 1] * sin_a
    rotated[:, 1] = translated[:, 0] * sin_a + translated[:, 1] * cos_a
    
    # Translate back
    return rotated + np.array(center)


def create_transform_matrix(translation=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
    """
    Create a 4x4 transformation matrix.
    
    Args:
        translation: (x, y, z) translation
        rotation: (rx, ry, rz) rotation in degrees
        scale: (sx, sy, sz) scale factors
    
    Returns:
        4x4 transformation matrix
    """
    # Start with identity
    matrix = np.eye(4)
    
    # Apply translation
    matrix[0, 3] = translation[0]
    matrix[1, 3] = translation[1]
    matrix[2, 3] = translation[2]
    
    # Apply rotation (simplified - for full rotation need quaternion or Euler)
    # This is a simplified version - for production use a proper rotation matrix
    rx, ry, rz = [deg_to_rad(r) for r in rotation]
    
    # Apply scale
    matrix[0, 0] = scale[0]
    matrix[1, 1] = scale[1]
    matrix[2, 2] = scale[2]
    
    return matrix


def merge_meshes(mesh_list):
    """
    Merge multiple trimesh meshes into one.
    
    Args:
        mesh_list: List of trimesh.Trimesh objects
    
    Returns:
        Single merged trimesh.Trimesh
    """
    if not mesh_list:
        return None
    
    if len(mesh_list) == 1:
        return mesh_list[0].copy()
    
    # Use trimesh's scene merging
    scene = trimesh.Scene()
    for mesh in mesh_list:
        if mesh is not None:
            scene.add_geometry(mesh)
    
    # Convert scene to single mesh
    merged = scene.dump(concatenate=True)
    return merged


def validate_mesh(mesh):
    """
    Validate a trimesh mesh for common issues.
    
    Args:
        mesh: trimesh.Trimesh object
    
    Returns:
        Dictionary with validation results
    """
    if mesh is None:
        return {"valid": False, "errors": ["Mesh is None"]}
    
    errors = []
    warnings = []
    
    # Check for NaN vertices
    if np.any(np.isnan(mesh.vertices)):
        errors.append("Mesh contains NaN vertices")
    
    # Check for infinite vertices
    if np.any(np.isinf(mesh.vertices)):
        errors.append("Mesh contains infinite vertices")
    
    # Check for duplicate vertices
    if len(mesh.vertices) != len(np.unique(mesh.vertices, axis=0)):
        warnings.append("Mesh contains duplicate vertices")
    
    # Check for degenerate triangles
    if hasattr(mesh, 'faces'):
        # Simple check: triangle area should be > 0
        areas = mesh.area_faces
        if np.any(areas <= 0):
            warnings.append(f"Mesh contains {np.sum(areas <= 0)} degenerate triangles")
    
    # Check for manifoldness (simplified)
    if not mesh.is_watertight:
        warnings.append("Mesh is not watertight")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces) if hasattr(mesh, 'faces') else 0,
        "bounds": mesh.bounds.tolist() if hasattr(mesh, 'bounds') else None
    }


def create_debug_visualization(scene_elements, colors=None):
    """
    Create a debug visualization of scene elements.
    
    Args:
        scene_elements: List of meshes or geometry objects
        colors: Optional list of colors for each element
    
    Returns:
        trimesh.Trimesh with debug visualization
    """
    import trimesh
    
    debug_meshes = []
    
    for i, element in enumerate(scene_elements):
        if element is None:
            continue
        
        if isinstance(element, trimesh.Trimesh):
            mesh = element.copy()
            
            # Apply color if provided
            if colors is not None and i < len(colors):
                if isinstance(colors[i], (list, tuple)) and len(colors[i]) == 4:
                    # Create vertex colors
                    vertex_colors = np.tile(colors[i], (len(mesh.vertices), 1))
                    mesh.visual.vertex_colors = vertex_colors
            
            debug_meshes.append(mesh)
    
    if not debug_meshes:
        return None
    
    return merge_meshes(debug_meshes)


def calculate_bounding_box(mesh_list):
    """
    Calculate combined bounding box for multiple meshes.
    
    Args:
        mesh_list: List of trimesh.Trimesh objects
    
    Returns:
        Dictionary with min/max bounds
    """
    if not mesh_list:
        return {"min": [0, 0, 0], "max": [0, 0, 0], "size": [0, 0, 0]}
    
    all_vertices = []
    for mesh in mesh_list:
        if mesh is not None and hasattr(mesh, 'vertices'):
            all_vertices.append(mesh.vertices)
    
    if not all_vertices:
        return {"min": [0, 0, 0], "max": [0, 0, 0], "size": [0, 0, 0]}
    
    all_vertices = np.vstack(all_vertices)
    min_coords = np.min(all_vertices, axis=0)
    max_coords = np.max(all_vertices, axis=0)
    size = max_coords - min_coords
    
    return {
        "min": min_coords.tolist(),
        "max": max_coords.tolist(),
        "size": size.tolist(),
        "center": ((min_coords + max_coords) / 2).tolist()
    }