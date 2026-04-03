"""
Basic geometric primitives for LevelSmith.

This module contains functions for creating basic 3D geometric elements:
- Walls, doors, windows
- Roofs and architectural features
- Structural elements and decorations

All functions are migrated from generate_level.py for better modularity.
"""

import numpy as np
import trimesh
from shapely.geometry import Polygon

# Import from other geometry modules
from . import materials, utils


# ============================================================================
# Wall System
# ============================================================================

def create_wall_segment(start, end, height, thickness, material_color):
    """
    Create a wall segment between two points.
    
    Args:
        start: Start point (x, z)
        end: End point (x, z)
        height: Wall height
        thickness: Wall thickness
        material_color: RGBA color for the wall
    
    Returns:
        trimesh.Trimesh object
    """
    # This is a placeholder - actual implementation will be migrated
    # from build_x_wall and build_z_wall functions
    raise NotImplementedError("To be migrated from generate_level.py")


def create_wall_with_opening(segment, opening_type, opening_params, material_color):
    """
    Create a wall segment with an opening (door or window).
    
    Args:
        segment: Wall segment definition
        opening_type: "door" or "window"
        opening_params: Dictionary with width, height, position
        material_color: RGBA color
    
    Returns:
        trimesh.Trimesh object
    """
    # This is a placeholder - actual implementation will be migrated
    # from add_door_panel, add_door_frame, add_glass_x, add_glass_z
    raise NotImplementedError("To be migrated from generate_level.py")


def _boolean_wall(solid, openings_list, color):
    """
    Boolean subtract openings from a solid wall.
    Creates a wall mesh (with openings removed).
    
    Args:
        solid: Base wall trimesh.Trimesh object
        openings_list: List of trimesh.Trimesh opening boxes
        color: RGBA color array
    
    Returns:
        trimesh.Trimesh with openings cut out
    
    Note: Migrated from generate_level.py _boolean_wall function
    """
    result = solid
    for hole in openings_list:
        try:
            result = trimesh.boolean.difference([result, hole], engine='manifold')
        except Exception:
            pass  # Skip failed boolean operations, keep original
    trimesh.repair.fix_normals(result)
    c = np.array(color, dtype=np.uint8)
    result.visual.face_colors = np.tile(c, (len(result.faces), 1))
    return result


def build_x_wall(total_w, height, thickness, openings, color, wx, wz):
    """
    Build a wall along the X direction (front/back wall).
    Uses boolean operations to subtract door/window openings from complete wall → no seams, continuous normals.
    
    Args:
        total_w: Total wall width
        height: Wall height
        thickness: Wall thickness
        openings: List of opening dictionaries with x, w, h, y keys
        color: RGBA color array
        wx, wz: Wall position coordinates
    
    Returns:
        List of trimesh.Trimesh objects
    
    Note: Migrated from generate_level.py build_x_wall function
    """
    # Complete solid wall
    solid = trimesh.creation.box(extents=[total_w, height, thickness])
    solid.apply_translation([wx + total_w / 2, height / 2, wz])

    # Opening cutting bodies (2x wall thickness to ensure penetration)
    holes = []
    for op in openings:
        y0 = max(0.0, op.get("y", 0.0))
        h  = op.get("h", height)
        w  = op.get("w", 1.0)
        x  = op.get("x", 0.0)
        hole = trimesh.creation.box(extents=[w, h, thickness * 2])
        hole.apply_translation([wx + x + w / 2, y0 + h / 2, wz])
        holes.append(hole)

    if not holes:
        c = np.array(color, dtype=np.uint8)
        solid.visual.face_colors = np.tile(c, (len(solid.faces), 1))
        return [solid]

    return [_boolean_wall(solid, holes, color)]


def build_z_wall(total_d, height, thickness, openings_z, color, wx, wz):
    """
    Build a wall along the Z direction (left/right wall).
    Boolean operations: complete solid wall - window openings.
    
    Args:
        total_d: Total wall depth
        height: Wall height
        thickness: Wall thickness
        openings_z: List of opening dictionaries with z, w, h, y keys
        color: RGBA color array
        wx, wz: Wall position coordinates
    
    Returns:
        List of trimesh.Trimesh objects
    
    Note: Migrated from generate_level.py build_z_wall function
    """
    solid = trimesh.creation.box(extents=[thickness, height, total_d])
    solid.apply_translation([wx, height / 2, wz + total_d / 2])

    holes = []
    for op in openings_z:
        y0 = max(0.0, op.get("y", 0.0))
        h  = op.get("h", height)
        w  = op.get("w", 1.0)
        z  = op.get("z", 0.0)
        hole = trimesh.creation.box(extents=[thickness * 2, h, w])
        hole.apply_translation([wx, y0 + h / 2, wz + z + w / 2])
        holes.append(hole)

    if not holes:
        c = np.array(color, dtype=np.uint8)
        solid.visual.face_colors = np.tile(c, (len(solid.faces), 1))
        return [solid]

    return [_boolean_wall(solid, holes, color)]


def build_polygon_walls(polygon, height, thickness, material_color, style="baseline"):
    """
    Build walls around a polygon footprint.
    
    Args:
        polygon: Shapely Polygon defining the footprint
        height: Wall height
        thickness: Wall thickness
        material_color: RGBA color or style key
        style: Style key for material lookup
    
    Returns:
        List of trimesh.Trimesh objects
    """
    # This is a placeholder - actual implementation will be migrated
    # from build_polygon_walls function (131 lines)
    raise NotImplementedError("To be migrated from generate_level.py")


def build_edge_wall(polygon, height, thickness, material_color, style="baseline"):
    """
    Build edge walls for a polygon.
    
    Args:
        polygon: Shapely Polygon
        height: Wall height
        thickness: Wall thickness
        material_color: RGBA color or style key
        style: Style key for material lookup
    
    Returns:
        trimesh.Trimesh object
    """
    # This is a placeholder - actual implementation will be migrated
    # from build_edge_wall function (75 lines)
    raise NotImplementedError("To be migrated from generate_level.py")


# ============================================================================
# Door and Window System
# ============================================================================

def create_door(wall_segment, width, height, material_color, style="baseline"):
    """
    Create a door in a wall segment.
    
    Args:
        wall_segment: Wall segment definition
        width: Door width
        height: Door height
        material_color: RGBA color or style key
        style: Style key for material lookup
    
    Returns:
        trimesh.Trimesh object for the door
    """
    # This is a placeholder - actual implementation will be migrated
    # from add_door_panel and add_door_frame functions
    raise NotImplementedError("To be migrated from generate_level.py")


def create_window(wall_segment, width, height, material_color, style="baseline"):
    """
    Create a window in a wall segment.
    
    Args:
        wall_segment: Wall segment definition
        width: Window width
        height: Window height
        material_color: RGBA color or style key
        style: Style key for material lookup
    
    Returns:
        trimesh.Trimesh object for the window
    """
    # This is a placeholder - actual implementation will be migrated
    # from add_glass_x and add_glass_z functions
    raise NotImplementedError("To be migrated from generate_level.py")


def place_windows_x(wall_len, height, win_w, win_h, density, subdiv, sill=0.9):
    """
    Place windows on X-direction wall.
    
    Args:
        wall_len: Wall length
        height: Wall height
        win_w: Window width
        win_h: Window height
        density: Window density factor
        subdiv: Subdivision factor
        sill: Window sill height
    
    Returns:
        List of window opening dictionaries
    
    Note: Migrated from generate_level.py place_windows_x function
    """
    if density <= 0.0:
        return []
    n = max(1, round(density * subdiv))
    n = min(n, max(1, int(wall_len / (win_w + 0.5))))
    gap = max(0.35, (wall_len - n * win_w) / (n + 1))
    result, x = [], gap
    for _ in range(n):
        if x + win_w > wall_len - 0.15:
            break
        h_actual = min(win_h, height - sill - 0.1)
        if h_actual > 0.1:
            result.append({"x": x, "w": win_w, "y": sill, "h": h_actual})
        x += win_w + gap
    return result


def place_windows_z(wall_dep, height, win_w, win_h, density, subdiv, sill=0.9):
    """
    Place windows on Z-direction wall (adjusted density).
    
    Args:
        wall_dep: Wall depth
        height: Wall height
        win_w: Window width
        win_h: Window height
        density: Window density factor
        subdiv: Subdivision factor
        sill: Window sill height
    
    Returns:
        List of window opening dictionaries
    
    Note: Migrated from generate_level.py place_windows_z function
    """
    if density <= 0.0:
        return []
    n = max(1, round(density * subdiv * 0.7))
    n = min(n, max(1, int(wall_dep / (win_w + 0.5))))
    gap = max(0.35, (wall_dep - n * win_w) / (n + 1))
    result, z = [], gap
    for _ in range(n):
        if z + win_w > wall_dep - 0.15:
            break
        h_actual = min(win_h, height - sill - 0.1)
        if h_actual > 0.1:
            result.append({"z": z, "w": win_w, "y": sill, "h": h_actual})
        z += win_w + gap
    return result


def place_windows_on_wall(wall_segment, window_count, window_width, window_height, 
                          material_color, style="baseline"):
    """
    Place multiple windows on a wall segment.
    
    Args:
        wall_segment: Wall segment definition
        window_count: Number of windows
        window_width: Width of each window
        window_height: Height of each window
        material_color: RGBA color or style key
        style: Style key for material lookup
    
    Returns:
        List of window meshes
    """
    # This is a placeholder - actual implementation will be migrated
    # from place_windows_edge function
    raise NotImplementedError("To be migrated from generate_level.py")


# ============================================================================
# Roof System
# ============================================================================

def create_roof(footprint, roof_type, pitch, material_color, style="baseline"):
    """
    Create a roof for a building footprint.
    
    Args:
        footprint: Shapely Polygon defining the building footprint
        roof_type: "gabled", "hip", "flat", "pagoda", "turret"
        pitch: Roof pitch angle in degrees
        material_color: RGBA color or style key
        style: Style key for material lookup
    
    Returns:
        trimesh.Trimesh object
    """
    # This is a placeholder - actual implementation will be migrated
    # from various roof building functions
    raise NotImplementedError("To be migrated from generate_level.py")


def build_gabled_roof(vertices, faces, colors, width, depth, height, style="baseline"):
    """
    Build a gabled roof.
    
    Args:
        vertices: Existing vertices array
        faces: Existing faces array
        colors: Existing colors array
        width: Building width
        depth: Building depth
        height: Roof height
        style: Style key for material lookup
    
    Returns:
        Updated (vertices, faces, colors)
    """
    # This is a placeholder - actual implementation will be migrated
    # from build_gabled_roof function (49 lines)
    raise NotImplementedError("To be migrated from generate_level.py")


def build_hip_roof(vertices, faces, colors, width, depth, height, style="baseline"):
    """
    Build a hip roof.
    
    Args:
        vertices: Existing vertices array
        faces: Existing faces array
        colors: Existing colors array
        width: Building width
        depth: Building depth
        height: Roof height
        style: Style key for material lookup
    
    Returns:
        Updated (vertices, faces, colors)
    """
    # This is a placeholder - actual implementation will be migrated
    # from build_hip_roof function (75 lines)
    raise NotImplementedError("To be migrated from generate_level.py")


# ============================================================================
# Structural and Decorative Elements
# ============================================================================

def build_columns(positions, height, diameter, material_color, style="baseline"):
    """
    Build columns at specified positions.
    
    Args:
        positions: List of (x, z) positions for columns
        height: Column height
        diameter: Column diameter
        material_color: RGBA color or style key
        style: Style key for material lookup
    
    Returns:
        List of column meshes
    """
    # This is a placeholder - actual implementation will be migrated
    # from build_columns function (54 lines)
    raise NotImplementedError("To be migrated from generate_level.py")


def build_battlements(wall_top_positions, height, width, material_color, style="baseline"):
    """
    Build battlements (crenellations) along wall tops.
    
    Args:
        wall_top_positions: List of wall top segment positions
        height: Battlement height
        width: Battlement width
        material_color: RGBA color or style key
        style: Style key for material lookup
    
    Returns:
        List of battlement meshes
    """
    # This is a placeholder - actual implementation will be migrated
    # from build_battlements function (44 lines)
    raise NotImplementedError("To be migrated from generate_level.py")


# ============================================================================
# Basic Geometric Primitives
# ============================================================================

def create_box(size, center, color):
    """
    Create a box (cuboid) primitive.
    
    Args:
        size: [width, height, depth] dimensions
        center: [x, y, z] center position
        color: RGBA color array
    
    Returns:
        trimesh.Trimesh object
    
    Note:
        Migrated from generate_level.py make_box function.
        Only dimensions >= 0.05 are rounded to 3 decimal places.
        This avoids degeneration of thin panels (doors/windows).
    """
    extents = np.array(size, dtype=np.float64)
    center_pos = np.array(center, dtype=np.float64)
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center_pos)
    # Only round dimensions >= 0.05 to eliminate wall joint gaps without damaging thin panels
    for ax in range(3):
        if extents[ax] >= 0.05:
            b.vertices[:, ax] = np.round(b.vertices[:, ax], 3)
    c = np.array(color, dtype=np.uint8)
    b.visual.face_colors = np.tile(c, (len(b.faces), 1))
    return b


def create_extruded_polygon(polygon, height, color):
    """
    Create a mesh by extruding a 2D polygon.
    
    Args:
        polygon: Shapely Polygon
        height: Extrusion height
        color: RGBA color array
    
    Returns:
        trimesh.Trimesh object
    
    Note:
        Migrated from generate_level.py make_extruded_polygon function.
        Output coordinate system: X=right, Y=up, Z=front (depth).
        extrude_polygon draws outline in XY plane, extrudes along Z.
        Rotated +90° around X axis: shapely XY → world XZ, extrude Z → world -Y.
        After rotation mesh top is at Y=0, bottom at Y=-height.
    """
    mesh = trimesh.creation.extrude_polygon(polygon, height)
    # +90° around X: new_Y = -old_Z, new_Z = old_Y
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    mesh.apply_transform(rot)
    c = np.array(color, dtype=np.uint8)
    mesh.visual.face_colors = np.tile(c, (len(mesh.faces), 1))
    return mesh


# ============================================================================
# Utility Functions
# ============================================================================

def merge_wall_segments(segments):
    """
    Merge multiple wall segments into a single mesh.
    
    Args:
        segments: List of wall segment meshes
    
    Returns:
        Single merged mesh
    """
    return utils.merge_meshes(segments)


def validate_primitive(mesh, primitive_type):
    """
    Validate a primitive mesh.
    
    Args:
        mesh: trimesh.Trimesh object
        primitive_type: Type of primitive for validation rules
    
    Returns:
        Validation result dictionary
    """
    return utils.validate_mesh(mesh)