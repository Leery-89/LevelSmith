"""
Style Base Profiles — Mesh grammar rules for grey-model differentiation.

Each profile describes the architectural silhouette, roof language, opening rhythm,
and edge/base treatment for a style family. These rules drive parameter overrides
in apply_style_profile() to produce visually distinct grey models.

Only geometry-affecting rules. No materials, no aging, no lighting.
"""

STYLE_BASE_PROFILES = {

    # ─── Japanese ───────────────────────────────────────────────
    "japanese": {
        "_subdivision_override": 1,     # single storey
        "_height_range_override": [2.5, 4.0],  # low but livable
        "_win_size_override": [1.2, 0.8],      # wide, short (shoji)
        "group_rules": {
            "lot_fill_ratio": 0.45,
            "building_gap_min": 8.0,
            "building_gap_max": 15.0,
            "setback_min": 3.0,
            "setback_max": 6.0,
            "alignment": "staggered",
            "lot_width_min": 12.0,
            "lot_width_max": 22.0,
            "ambient_frequency": 3,
        },
        "silhouette_rules": {
            "width_to_height": 1.6,     # strongly horizontal
            "symmetry": 0.85,           # highly symmetrical
            "mass_distribution": "low_wide",
        },
        "roof_rules": {
            "type": "hipped_curved",
            "pitch": 0.50,
            "eave_overhang": 0.90,      # deep eaves, defining feature
            "ridge_decoration": True,
        },
        "opening_rules": {
            "shape": "rectangular",
            "size": "medium_wide",      # wide but short windows
            "density": 0.12,            # restrained — fewer openings
            "rhythm": "regular_modular",
        },
        "edge_and_base_rules": {
            "base_height": 0.45,        # raised platform / engawa
            "corner_quoins": False,
            "buttress": False,
            "battlements": False,
            "pillar_emphasis": True,     # exposed wooden columns
        },
        "material_rules": {
            "color_palette": {
                "wall": [0.85, 0.82, 0.78],   # warm white plaster
            },
            "surface_roughness": 0.35,         # smooth plaster
            "material_variation": 0.08,        # subtle differences
        },
    },

    "japanese_temple": {
        "_subdivision_override": 1,
        "_height_range_override": [3.5, 6.0],  # taller for temple hall
        "silhouette_rules": {
            "width_to_height": 1.4,
            "symmetry": 0.95,
            "mass_distribution": "low_wide",
        },
        "roof_rules": {
            "type": "hipped_curved",
            "pitch": 0.65,
            "eave_overhang": 1.00,      # maximum eave depth
            "ridge_decoration": True,
        },
        "opening_rules": {
            "shape": "rectangular",
            "size": "medium",
            "density": 0.30,            # fewer openings, more solid walls
            "rhythm": "regular_modular",
        },
        "edge_and_base_rules": {
            "base_height": 0.60,        # high stone platform
            "corner_quoins": False,
            "buttress": False,
            "battlements": False,
            "pillar_emphasis": True,
        },
        "material_rules": {
            "color_palette": {"wall": [0.82, 0.78, 0.72]},
            "surface_roughness": 0.40,
            "material_variation": 0.06,
        },
    },

    "japanese_machiya": {
        "_subdivision_override": 2,     # two storey townhouse
        "_height_range_override": [2.8, 5.0],
        "silhouette_rules": {
            "width_to_height": 1.8,     # very horizontal, narrow-front deep
            "symmetry": 0.60,
            "mass_distribution": "low_wide",
        },
        "roof_rules": {
            "type": "gabled_shallow",
            "pitch": 0.45,
            "eave_overhang": 0.70,
            "ridge_decoration": False,
        },
        "opening_rules": {
            "shape": "rectangular",
            "size": "medium_wide",
            "density": 0.40,
            "rhythm": "regular_modular",
        },
        "edge_and_base_rules": {
            "base_height": 0.25,
            "corner_quoins": False,
            "buttress": False,
            "battlements": False,
            "pillar_emphasis": True,
        },
        "material_rules": {
            "color_palette": {"wall": [0.28, 0.24, 0.20]},  # dark wood
            "surface_roughness": 0.55,
            "material_variation": 0.10,
        },
    },

    # ─── Medieval ───────────────────────────────────────────────
    "medieval": {
        "_subdivision_override": 2,     # two storey
        "_height_range_override": [6.0, 10.0],
        "_win_size_override": [0.4, 0.9],       # narrow, tall (arrow slit feel)
        "group_rules": {
            "lot_fill_ratio": 0.85,
            "building_gap_min": 1.5,
            "building_gap_max": 4.0,
            "setback_min": 0.5,
            "setback_max": 2.0,
            "alignment": "irregular",
            "lot_width_min": 6.0,
            "lot_width_max": 14.0,
            "ambient_frequency": 5,
        },
        "silhouette_rules": {
            "width_to_height": 0.9,     # roughly cubic, slightly tall
            "symmetry": 0.35,           # bolder: asymmetric → L/U shapes
            "mass_distribution": "blocky",
        },
        "roof_rules": {
            "type": "steep_gabled",
            "pitch": 0.75,              # steeper (bolder)
            "eave_overhang": 0.15,      # minimal eave
            "ridge_decoration": False,
        },
        "opening_rules": {
            "shape": "arched_narrow",
            "size": "small",
            "density": 0.15,            # very few, small windows (bolder)
            "rhythm": "irregular_clustered",
        },
        "edge_and_base_rules": {
            "base_height": 0.15,
            "corner_quoins": True,      # stone corner blocks
            "buttress": True,           # wall buttresses
            "battlements": True,
            "pillar_emphasis": False,
        },
        "material_rules": {
            "color_palette": {
                "wall": [0.50, 0.48, 0.44],   # cold grey stone
            },
            "surface_roughness": 0.75,         # rough stone
            "material_variation": 0.12,        # stone color varies
        },
    },

    "medieval_keep": {
        "_subdivision_override": 4,     # tall multi-storey tower
        "_height_range_override": [10.0, 18.0],
        "silhouette_rules": {
            "width_to_height": 0.7,     # tall tower
            "symmetry": 0.70,
            "mass_distribution": "blocky_tall",
        },
        "roof_rules": {
            "type": "flat_or_barrel_vault",
            "pitch": 0.10,
            "eave_overhang": 0.00,
            "ridge_decoration": False,
        },
        "opening_rules": {
            "shape": "arched_small",
            "size": "minimal",
            "density": 0.08,            # arrow slits only
            "rhythm": "sparse_defensive",
        },
        "edge_and_base_rules": {
            "base_height": 0.10,
            "corner_quoins": True,
            "buttress": True,
            "battlements": True,
            "pillar_emphasis": False,
        },
        "material_rules": {
            "color_palette": {"wall": [0.45, 0.43, 0.40]},
            "surface_roughness": 0.80,
            "material_variation": 0.10,
        },
    },

    "medieval_chapel": {
        "_subdivision_override": 2,
        "_height_range_override": [5.0, 8.0],
        "silhouette_rules": {
            "width_to_height": 0.85,
            "symmetry": 0.90,
            "mass_distribution": "blocky",
        },
        "roof_rules": {
            "type": "steep_gabled",
            "pitch": 0.75,
            "eave_overhang": 0.15,
            "ridge_decoration": False,
        },
        "opening_rules": {
            "shape": "pointed_arch",
            "size": "medium_tall",      # tall lancet windows
            "density": 0.25,
            "rhythm": "regular_modular",
        },
        "edge_and_base_rules": {
            "base_height": 0.10,
            "corner_quoins": True,
            "buttress": True,
            "battlements": False,
            "pillar_emphasis": False,
        },
        "material_rules": {
            "color_palette": {"wall": [0.55, 0.52, 0.48]},
            "surface_roughness": 0.70,
            "material_variation": 0.08,
        },
    },

    # ─── Industrial ─────────────────────────────────────────────
    "industrial": {
        "_subdivision_override": 1,     # single storey shed
        "_height_range_override": [7.0, 12.0],
        "_win_size_override": [0.8, 0.8],       # square, uniform grid
        "group_rules": {
            "lot_fill_ratio": 0.65,
            "building_gap_min": 5.0,
            "building_gap_max": 8.0,
            "setback_min": 2.0,
            "setback_max": 3.0,
            "alignment": "aligned",
            "lot_width_min": 10.0,
            "lot_width_max": 14.0,
            "ambient_frequency": 0,
        },
        "silhouette_rules": {
            "width_to_height": 1.4,     # wide sheds (bolder)
            "symmetry": 0.50,
            "mass_distribution": "box",
        },
        "roof_rules": {
            "type": "flat_or_sawtooth",
            "pitch": 0.08,              # nearly flat (bolder)
            "eave_overhang": 0.02,      # almost none
            "ridge_decoration": False,
        },
        "opening_rules": {
            "shape": "rectangular",
            "size": "medium",
            "density": 0.45,            # dense grid of windows (bolder)
            "rhythm": "strict_grid",
        },
        "edge_and_base_rules": {
            "base_height": 0.05,
            "corner_quoins": False,
            "buttress": False,
            "battlements": False,
            "pillar_emphasis": False,
        },
        "material_rules": {
            "color_palette": {
                "wall": [0.55, 0.35, 0.26],   # red-brown brick
            },
            "surface_roughness": 0.65,         # brick texture
            "material_variation": 0.15,        # noticeable variation
        },
    },

    "industrial_workshop": {
        "_subdivision_override": 1,
        "_height_range_override": [5.0, 8.0],
        "silhouette_rules": {
            "width_to_height": 1.2,
            "symmetry": 0.55,
            "mass_distribution": "box",
        },
        "roof_rules": {
            "type": "gabled_shallow",
            "pitch": 0.30,
            "eave_overhang": 0.15,
            "ridge_decoration": False,
        },
        "opening_rules": {
            "shape": "rectangular",
            "size": "medium",
            "density": 0.30,
            "rhythm": "strict_grid",
        },
        "edge_and_base_rules": {
            "base_height": 0.05,
            "corner_quoins": False,
            "buttress": False,
            "battlements": False,
            "pillar_emphasis": False,
        },
        "material_rules": {
            "color_palette": {"wall": [0.58, 0.42, 0.32]},
            "surface_roughness": 0.60,
            "material_variation": 0.12,
        },
    },

    "industrial_powerplant": {
        "_subdivision_override": 1,
        "_height_range_override": [10.0, 18.0],
        "silhouette_rules": {
            "width_to_height": 1.0,     # tall industrial halls
            "symmetry": 0.60,
            "mass_distribution": "box_tall",
        },
        "roof_rules": {
            "type": "flat",
            "pitch": 0.05,
            "eave_overhang": 0.00,
            "ridge_decoration": False,
        },
        "opening_rules": {
            "shape": "rectangular",
            "size": "small",
            "density": 0.15,            # few high windows
            "rhythm": "strict_grid",
        },
        "edge_and_base_rules": {
            "base_height": 0.02,
            "corner_quoins": False,
            "buttress": False,
            "battlements": False,
            "pillar_emphasis": False,
        },
        "material_rules": {
            "color_palette": {"wall": [0.42, 0.40, 0.38]},
            "surface_roughness": 0.70,
            "material_variation": 0.08,
        },
    },

    # ─── Other families (stubs, can be expanded later) ──────────
    # "fantasy": { ... },
    # "horror": { ... },
    # "modern": { ... },
    # "desert": { ... },
}


# ─── Mapping tables ────────────────────────────────────────────

_ROOF_TYPE_MAP = {
    "flat": 0, "flat_or_sawtooth": 0, "flat_or_barrel_vault": 0,
    "gabled": 1, "gabled_shallow": 1, "steep_gabled": 1,
    "steep_broken": 1, "steep_varied": 1,
    "hipped": 2, "hipped_curved": 2, "hipped_heavy": 2,
    "pointed_spire": 3,
    "dome": 4,
}

_WINDOW_SHAPE_MAP = {
    "rectangular": 0, "barred_rectangular": 0,
    "arched": 1, "arched_narrow": 1, "arched_small": 1, "arched_low": 1,
    "pointed_arch": 2,
    "round": 3, "horseshoe_arch": 3,
    "irregular_gothic": 2,
}


def apply_style_profile(params: dict, profile: dict) -> dict:
    """
    Override geometry-related params using a style base profile.
    Only touches mesh-grammar parameters. No materials, no aging.

    params:  existing style params dict (from trained_style_params.json)
    profile: one entry from STYLE_BASE_PROFILES

    Returns: new params dict with profile overrides applied.
    """
    result = dict(params)

    # Deep-copy nested dicts
    if "height_range" in result:
        result["height_range"] = list(result["height_range"])
    if "win_spec" in result:
        result["win_spec"] = dict(result["win_spec"])
    if "door_spec" in result:
        result["door_spec"] = dict(result["door_spec"])

    sr = profile.get("silhouette_rules", {})
    rr = profile.get("roof_rules", {})
    op = profile.get("opening_rules", {})
    eb = profile.get("edge_and_base_rules", {})

    # ─── Body proportions (absolute height override) ───
    height_override = profile.get("_height_range_override")
    if height_override:
        result["height_range"][0] = height_override[0]
        result["height_range"][1] = height_override[1]
    else:
        wh_ratio = sr.get("width_to_height", 1.0)
        if wh_ratio > 1.5:
            result["height_range"][0] *= 0.6
            result["height_range"][1] *= 0.65
        elif wh_ratio < 0.8:
            result["height_range"][0] *= 1.3
            result["height_range"][1] *= 1.4

    # ─── Roof language ───
    roof_type_str = rr.get("type", "gabled")
    result["roof_type"] = _ROOF_TYPE_MAP.get(roof_type_str, 1)
    result["roof_pitch"] = round(rr.get("pitch", 0.5), 3)
    result["eave_overhang"] = round(rr.get("eave_overhang", 0.15), 3)

    # ─── Opening rhythm (direct density override) ───
    density = op.get("density", 0.3)
    if "win_spec" in result:
        result["win_spec"]["density"] = round(density, 3)
    else:
        result["win_density"] = round(density, 3)

    shape_str = op.get("shape", "rectangular")
    result["window_shape"] = _WINDOW_SHAPE_MAP.get(shape_str, 0)

    # Window size — direct override if profile specifies absolute values
    win_override = profile.get("_win_size_override")
    if win_override and "win_spec" in result:
        result["win_spec"]["width"] = win_override[0]
        result["win_spec"]["height"] = win_override[1]
    else:
        size_str = op.get("size", "medium")
        size_lower = size_str.lower()
        if "win_spec" in result:
            ws = result["win_spec"]
            if "minimal" in size_lower:
                ws["width"] = round(ws.get("width", 0.8) * 0.45, 2)
                ws["height"] = round(ws.get("height", 1.0) * 0.55, 2)
            elif "small" in size_lower:
                ws["width"] = round(ws.get("width", 0.8) * 0.5, 2)
                ws["height"] = round(ws.get("height", 1.0) * 0.8, 2)
            elif "wide" in size_lower:
                ws["width"] = round(ws.get("width", 0.8) * 1.4, 2)
                ws["height"] = round(ws.get("height", 1.0) * 0.7, 2)
            elif "tall" in size_lower:
                ws["height"] = round(ws.get("height", 1.0) * 1.5, 2)
            elif "large" in size_lower or "floor" in size_lower:
                ws["width"] = round(ws.get("width", 0.8) * 1.5, 2)
                ws["height"] = round(ws.get("height", 1.0) * 1.4, 2)

    # ─── Wall thickness (direct override for bold difference) ───
    base_h = eb.get("base_height", 0.1)
    if base_h > 0.4:
        # High platform (japanese): thin walls on heavy base
        result["wall_thickness"] = 0.25
    elif eb.get("buttress", False):
        # Buttressed (medieval): thick stone walls
        result["wall_thickness"] = 0.80
    else:
        # Default
        result["wall_thickness"] = round(0.25 + base_h * 0.5, 2)

    # ─── Decoration (direct set, not max()) ───
    result["has_battlements"] = 1 if eb.get("battlements", False) else 0
    result["has_arch"] = 1 if eb.get("corner_quoins", False) else 0

    if eb.get("pillar_emphasis", False):
        result["column_count"] = 6   # Bold: many columns for japanese
    elif eb.get("buttress", False):
        result["column_count"] = 2   # Buttresses for medieval
    else:
        result["column_count"] = 0   # Zero for industrial

    # ─── Subdivision (direct override) ───
    subdiv = profile.get("_subdivision_override")
    if subdiv is not None:
        result["subdivision"] = subdiv

    # ─── Symmetry bias (consumed by _pick_footprint_type) ───
    result["_symmetry_bias"] = sr.get("symmetry", 0.5)

    # ─── Material color ───
    mr = profile.get("material_rules", {})
    palette = mr.get("color_palette", {})
    if "wall" in palette:
        r, g, b = palette["wall"]
        result["wall_color"] = [r, g, b]

        # High roughness → slightly darken (stone/brick feel)
        roughness = mr.get("surface_roughness", 0.5)
        if roughness > 0.6:
            dim = 0.92
            result["wall_color"] = [r * dim, g * dim, b * dim]

    result["_material_variation"] = mr.get("material_variation", 0.1)

    # ─── Group rules (consumed by _subdivide_street_lots / _assign_buildings_to_lots) ───
    gr = profile.get("group_rules", {})
    for key, val in gr.items():
        result[f"_{key}"] = val

    return result


def get_profile_for_style(style_key: str):
    """
    Look up profile: try exact match first, then base family.
    Returns profile dict or None.
    """
    if style_key in STYLE_BASE_PROFILES:
        return STYLE_BASE_PROFILES[style_key]
    # Try base family (e.g. "medieval_keep" → "medieval")
    base = style_key.split("_")[0]
    return STYLE_BASE_PROFILES.get(base)
