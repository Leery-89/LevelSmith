"""
Material and color system for LevelSmith geometry.

Provides style-based material definitions and color palettes.
Migrated from generate_level.py for better modularity.
"""

# ─── 颜色方案 (RGBA uint8) ────────────────────────────────────
STYLE_PALETTES = {
    "medieval": {
        "floor":    [115,  95,  75, 255],
        "ceiling":  [135, 115,  90, 255],
        "wall":     [162, 146, 122, 255],
        "door":     [ 88,  58,  28, 255],
        "window":   [155, 195, 215, 180],
        "internal": [148, 133, 110, 255],
        "ground":   [ 90,  80,  65, 255],
    },
    "modern": {
        "floor":    [215, 215, 215, 255],
        "ceiling":  [242, 242, 242, 255],
        "wall":     [198, 198, 198, 255],
        "door":     [ 68,  68,  68, 255],
        "window":   [135, 195, 228, 180],
        "internal": [182, 182, 182, 255],
        "ground":   [160, 160, 155, 255],
    },
    "industrial": {
        "floor":    [ 72,  72,  68, 255],
        "ceiling":  [ 52,  52,  48, 255],
        "wall":     [ 92,  88,  82, 255],
        "door":     [158,  82,  38, 255],
        "window":   [112, 138, 112, 180],
        "internal": [ 82,  78,  72, 255],
        "ground":   [ 55,  55,  52, 255],
    },
    "baseline": {
        "floor":    [172, 158, 142, 255],
        "ceiling":  [192, 182, 172, 255],
        "wall":     [182, 172, 162, 255],
        "door":     [ 98,  78,  58, 255],
        "window":   [148, 175, 200, 180],
        "internal": [168, 160, 152, 255],
        "ground":   [135, 128, 118, 255],
    },
}

# ─── 默认参数 (baseline) ──────────────────────────────────────
BASELINE_PARAMS = {
    "height_range":    [2.5, 3.5],
    "wall_thickness":  0.30,
    "floor_thickness": 0.15,
    "door_spec":       {"width": 0.90, "height": 2.10},
    "win_spec":        {"width": 1.20, "height": 1.20, "density": 0.40},
    "subdivision":     2,
}

# ─── 布局常量 ──────────────────────────────────────────────────
ROOM_W = 12.0   # 房间宽度 (X 轴)
ROOM_D =  8.0   # 房间深度 (Z 轴)
GAP    =  4.0   # 房间间距

STYLES = ["medieval", "modern", "industrial"]


def get_material_color(style_key, element_type):
    """
    Get RGBA color for a specific style and element type.
    
    Args:
        style_key: Style identifier ("medieval", "modern", "industrial", "baseline")
        element_type: Element type ("floor", "ceiling", "wall", "door", "window", "internal", "ground")
    
    Returns:
        List of 4 integers [R, G, B, A]
    """
    if style_key not in STYLE_PALETTES:
        style_key = "baseline"
    
    if element_type not in STYLE_PALETTES[style_key]:
        # Fallback to wall color
        return STYLE_PALETTES[style_key]["wall"]
    
    return STYLE_PALETTES[style_key][element_type]


def get_baseline_params():
    """Get baseline geometric parameters."""
    return BASELINE_PARAMS.copy()


def get_room_dimensions():
    """Get default room dimensions."""
    return {"width": ROOM_W, "depth": ROOM_D, "gap": GAP}


def get_supported_styles():
    """Get list of supported style keys."""
    return STYLES.copy()