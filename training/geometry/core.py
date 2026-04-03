"""
Core interfaces for LevelSmith building and scene generation.

This module provides high-level interfaces for generating buildings and scenes.
Functions are migrated from generate_level.py for better modularity.
"""

import json
from pathlib import Path
import trimesh
import numpy as np

# Import from other geometry modules
from . import materials, layout, primitives, utils


def generate_building(style_params, footprint=None, options=None):
    """
    Generate a complete building from style parameters and footprint.
    
    Args:
        style_params: Dictionary of style parameters (height, wall_thickness, etc.)
        footprint: Shapely Polygon defining the building footprint (optional)
        options: Additional options dictionary
    
    Returns:
        trimesh.Trimesh object representing the building
    """
    # This is a high-level wrapper that will coordinate calls to primitives module
    # Actual implementation will be migrated from build_room function
    
    if footprint is None:
        footprint = layout.make_rect_footprint()
    
    if options is None:
        options = {}
    
    # Extract parameters
    height_range = style_params.get("height_range", [2.5, 3.5])
    wall_thickness = style_params.get("wall_thickness", 0.30)
    floor_thickness = style_params.get("floor_thickness", 0.15)
    
    # Get style for materials
    style = options.get("style", "baseline")
    
    # This is a placeholder - actual implementation will be migrated
    # from build_room function (290 lines)
    raise NotImplementedError("To be migrated from generate_level.py build_room function")


def generate_scene(buildings, layout_type="grid", options=None):
    """
    Generate a scene with multiple buildings arranged in a layout.
    
    Args:
        buildings: List of building definitions (style_params, footprint, position)
        layout_type: Layout type ("grid", "cluster", "linear", "random")
        options: Additional options dictionary
    
    Returns:
        trimesh.Trimesh object representing the complete scene
    """
    if options is None:
        options = {}
    
    scene_meshes = []
    
    for i, building_def in enumerate(buildings):
        style_params = building_def.get("style_params", {})
        footprint = building_def.get("footprint")
        position = building_def.get("position", (0, 0))
        
        # Generate building
        building_options = options.copy()
        building_options.update(building_def.get("options", {}))
        
        building = generate_building(style_params, footprint, building_options)
        
        if building is not None:
            # Apply position transformation
            building.apply_translation([position[0], 0, position[1]])
            scene_meshes.append(building)
    
    # Merge all building meshes
    if scene_meshes:
        return utils.merge_meshes(scene_meshes)
    else:
        return None


def build_room(style, params, footprint=None, options=None):
    """
    Build a room with the given style and parameters.
    
    This is the main room generation function migrated from generate_level.py.
    
    Args:
        style: Style key ("medieval", "modern", "industrial", "baseline")
        params: Style parameters dictionary
        footprint: Shapely Polygon defining the room footprint
        options: Additional options
    
    Returns:
        trimesh.Trimesh object
    """
    # This function will be fully migrated from the original build_room function
    # It's kept here for backward compatibility during migration
    
    if footprint is None:
        footprint = layout.make_rect_footprint()
    
    if options is None:
        options = {}
    
    # This is a placeholder - actual 290-line implementation will be migrated
    raise NotImplementedError("To be migrated from generate_level.py build_room function")


def build_scene(style_map, use_style_palette=True, footprints=None):
    """
    Build a scene with multiple buildings in different styles.
    
    This is migrated from the original build_scene function in generate_level.py.
    
    Args:
        style_map: Dictionary mapping style names to parameter dictionaries
        use_style_palette: Whether to use style-specific color palettes
        footprints: Optional dictionary mapping style names to footprints
    
    Returns:
        trimesh.Scene object
    """
    # This function will be fully migrated from the original build_scene function
    
    if footprints is None:
        footprints = {}
    
    scene = trimesh.Scene()
    
    x_offset = 0
    for style_name, style_params in style_map.items():
        # Get footprint for this style
        footprint = footprints.get(style_name)
        if footprint is None:
            footprint = layout.make_rect_footprint()
        
        # Apply X offset for positioning
        positioned_footprint = layout.translate_footprint(footprint, x_offset, 0)
        
        # Generate building
        options = {
            "style": style_name if use_style_palette else "baseline",
            "position_offset": (x_offset, 0)
        }
        
        building = build_room(style_name, style_params, positioned_footprint, options)
        
        if building is not None:
            scene.add_geometry(building, node_name=f"building_{style_name}")
        
        # Update offset for next building
        bounds = layout.footprint_bounds(positioned_footprint)
        x_offset = bounds[2] + materials.GAP  # max_x + gap
    
    return scene


def load_style_params(json_path):
    """
    Load style parameters from a JSON file.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        Dictionary mapping style names to parameter dictionaries
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Style parameters file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract style parameters
    style_map = {}
    if "styles" in data:
        for style_name, style_data in data["styles"].items():
            if "params" in style_data:
                style_map[style_name] = style_data["params"]
    
    return style_map


def export_scene(scene, output_path, format="glb", options=None):
    """
    Export a scene to a file.
    
    Args:
        scene: trimesh.Scene or trimesh.Trimesh object
        output_path: Output file path
        format: Export format ("glb", "obj", "fbx")
        options: Export options
    
    Returns:
        True if successful
    """
    output_path = Path(output_path)
    
    if options is None:
        options = {}
    
    try:
        if isinstance(scene, trimesh.Scene):
            scene.export(str(output_path))
        elif isinstance(scene, trimesh.Trimesh):
            scene.export(str(output_path))
        else:
            raise ValueError(f"Unsupported scene type: {type(scene)}")
        
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def create_baseline_scene():
    """
    Create a baseline scene with default parameters.
    
    Returns:
        trimesh.Scene object
    """
    baseline_params = materials.get_baseline_params()
    style_map = {style: baseline_params for style in materials.get_supported_styles()}
    
    return build_scene(style_map, use_style_palette=True)


def validate_generation_output(scene, checks=None):
    """
    Validate the output of a generation process.
    
    Args:
        scene: Generated scene
        checks: List of checks to perform
    
    Returns:
        Dictionary with validation results
    """
    if checks is None:
        checks = ["mesh_integrity", "bounds", "material_coverage"]
    
    results = {}
    
    if isinstance(scene, trimesh.Scene):
        # Validate each geometry in the scene
        for name, geometry in scene.geometry.items():
            mesh_results = utils.validate_mesh(geometry)
            results[name] = mesh_results
    
    elif isinstance(scene, trimesh.Trimesh):
        results["main_mesh"] = utils.validate_mesh(scene)
    
    # Calculate overall bounds
    if isinstance(scene, (trimesh.Scene, trimesh.Trimesh)):
        bounds = utils.calculate_bounding_box(
            [scene] if isinstance(scene, trimesh.Trimesh) else list(scene.geometry.values())
        )
        results["bounds"] = bounds
    
    return results