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


def _make_colored_box(extents, translation, color):
    """Create a colored box mesh at a given position."""
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(translation)
    c = np.array(color, dtype=np.uint8)
    box.visual.face_colors = np.tile(c, (len(box.faces), 1))
    return box


def _segment_wall_around_openings(wall_len, height, thickness, openings,
                                  color, make_trans):
    """
    Build a wall by splitting it into solid segments around openings.
    No boolean operations — each segment is a clean box.

    For each opening: left-segment, right-segment, and lintel above.
    Segments between openings are merged into single boxes.

    Args:
        wall_len: Total wall length (along primary axis)
        height: Wall height
        thickness: Wall thickness
        openings: Sorted list of dicts with keys:
                  pos (position along wall), w, h, y (bottom of opening)
        color: RGBA color
        make_trans: Callable(pos, y, along_pos) -> [x, y, z] translation

    Returns:
        List of trimesh.Trimesh segments
    """
    if not openings:
        return [_make_colored_box(
            make_trans("extents", wall_len, height),
            make_trans("center", wall_len / 2, height / 2),
            color)]

    parts = []
    # Sort openings by position along wall
    sorted_ops = sorted(openings, key=lambda o: o["pos"])

    # Build solid segments between openings
    cursor = 0.0
    for op in sorted_ops:
        pos = op["pos"]
        w = op["w"]
        h = op["h"]
        y0 = max(0.0, op.get("y", 0.0))
        left_edge = pos
        right_edge = pos + w

        # Solid segment before this opening
        seg_w = left_edge - cursor
        if seg_w > 0.01:
            parts.append(_make_colored_box(
                make_trans("extents", seg_w, height),
                make_trans("center", cursor + seg_w / 2, height / 2),
                color))

        # Lintel above opening
        lintel_h = height - (y0 + h)
        if lintel_h > 0.01:
            parts.append(_make_colored_box(
                make_trans("extents", w, lintel_h),
                make_trans("center", pos + w / 2, y0 + h + lintel_h / 2),
                color))

        # Sill below opening (for windows with y > 0)
        if y0 > 0.05:
            parts.append(_make_colored_box(
                make_trans("extents", w, y0),
                make_trans("center", pos + w / 2, y0 / 2),
                color))

        cursor = right_edge

    # Solid segment after last opening
    remaining = wall_len - cursor
    if remaining > 0.01:
        parts.append(_make_colored_box(
            make_trans("extents", remaining, height),
            make_trans("center", cursor + remaining / 2, height / 2),
            color))

    return parts


def build_x_wall(total_w, height, thickness, openings, color, wx, wz):
    """
    Build a wall along the X direction (front/back wall).
    Uses manual segmentation — no boolean operations.

    Args:
        total_w: Total wall width
        height: Wall height
        thickness: Wall thickness
        openings: List of opening dicts with x, w, h, y keys
        color: RGBA color array
        wx, wz: Wall position coordinates

    Returns:
        List of trimesh.Trimesh objects
    """
    # Convert openings to segmentation format
    seg_openings = []
    for op in openings:
        seg_openings.append({
            "pos": op.get("x", 0.0),
            "w": op.get("w", 1.0),
            "h": op.get("h", height),
            "y": max(0.0, op.get("y", 0.0)),
        })

    def make_trans(mode, along, y_val):
        if mode == "extents":
            return [along, y_val, thickness]
        else:  # "center"
            return [wx + along, y_val, wz]

    return _segment_wall_around_openings(
        total_w, height, thickness, seg_openings, color, make_trans)


def build_z_wall(total_d, height, thickness, openings_z, color, wx, wz):
    """
    Build a wall along the Z direction (left/right wall).
    Uses manual segmentation — no boolean operations.

    Args:
        total_d: Total wall depth
        height: Wall height
        thickness: Wall thickness
        openings_z: List of opening dicts with z, w, h, y keys
        color: RGBA color array
        wx, wz: Wall position coordinates

    Returns:
        List of trimesh.Trimesh objects
    """
    seg_openings = []
    for op in openings_z:
        seg_openings.append({
            "pos": op.get("z", 0.0),
            "w": op.get("w", 1.0),
            "h": op.get("h", height),
            "y": max(0.0, op.get("y", 0.0)),
        })

    def make_trans(mode, along, y_val):
        if mode == "extents":
            return [thickness, y_val, along]
        else:  # "center"
            return [wx, y_val, wz + along]

    return _segment_wall_around_openings(
        total_d, height, thickness, seg_openings, color, make_trans)


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