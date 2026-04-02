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
        "group_rules": {"lot_fill_ratio": 0.45, "building_gap_min": 8.0, "building_gap_max": 15.0, "setback_min": 3.0, "setback_max": 6.0, "alignment": "staggered", "lot_width_min": 12.0, "lot_width_max": 22.0, "ambient_frequency": 3},
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
        "group_rules": {"lot_fill_ratio": 0.55, "building_gap_min": 5.0, "building_gap_max": 10.0, "setback_min": 2.0, "setback_max": 4.0, "alignment": "irregular", "lot_width_min": 8.0, "lot_width_max": 16.0, "ambient_frequency": 4},
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
        "group_rules": {"lot_fill_ratio": 0.85, "building_gap_min": 1.5, "building_gap_max": 4.0, "setback_min": 0.5, "setback_max": 2.0, "alignment": "irregular", "lot_width_min": 6.0, "lot_width_max": 14.0, "ambient_frequency": 5},
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
        "group_rules": {"lot_fill_ratio": 0.75, "building_gap_min": 2.0, "building_gap_max": 5.0, "setback_min": 1.0, "setback_max": 3.0, "alignment": "irregular", "lot_width_min": 8.0, "lot_width_max": 16.0, "ambient_frequency": 5},
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
        "group_rules": {"lot_fill_ratio": 0.65, "building_gap_min": 5.0, "building_gap_max": 8.0, "setback_min": 2.0, "setback_max": 3.0, "alignment": "aligned", "lot_width_min": 10.0, "lot_width_max": 14.0, "ambient_frequency": 0},
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
        "group_rules": {"lot_fill_ratio": 0.6, "building_gap_min": 6.0, "building_gap_max": 10.0, "setback_min": 3.0, "setback_max": 5.0, "alignment": "aligned", "lot_width_min": 12.0, "lot_width_max": 18.0, "ambient_frequency": 0},
    },

    # ─── Modern ─────────────────────────────────────────────────
    "modern": {
        "_subdivision_override": 2,
        "_height_range_override": [5.0, 10.0],
        "_win_size_override": [1.5, 1.8],
        "_min_footprint": [8.0, 8.0],
        "silhouette_rules": {"width_to_height": 1.0, "symmetry": 0.6, "mass_distribution": "clean_geometric"},
        "roof_rules": {"type": "flat", "pitch": 0.0, "eave_overhang": 0.15, "ridge_decoration": False},
        "opening_rules": {"shape": "rectangular", "size": "large", "density": 0.55, "rhythm": "regular_grid"},
        "edge_and_base_rules": {"base_height": 0.1, "corner_quoins": False, "buttress": False, "battlements": False, "pillar_emphasis": False},
        "material_rules": {"color_palette": {"wall": [0.90, 0.88, 0.85]}, "surface_roughness": 0.15, "material_variation": 0.05},
        "group_rules": {"lot_fill_ratio": 0.7, "building_gap_min": 4.0, "building_gap_max": 8.0, "setback_min": 1.0, "setback_max": 2.0, "alignment": "aligned", "lot_width_min": 14.0, "lot_width_max": 24.0, "ambient_frequency": 5},
    },

    "modern_loft": {
        "_subdivision_override": 2,
        "_height_range_override": [6.0, 12.0],
        "_win_size_override": [1.8, 2.2],
        "_min_footprint": [8.0, 8.0],
        "silhouette_rules": {"width_to_height": 0.9, "symmetry": 0.5, "mass_distribution": "converted_industrial"},
        "roof_rules": {"type": "flat", "pitch": 0.0, "eave_overhang": 0.05, "ridge_decoration": False},
        "opening_rules": {"shape": "rectangular", "size": "large", "density": 0.5, "rhythm": "industrial_grid"},
        "edge_and_base_rules": {"base_height": 0.15, "corner_quoins": False, "buttress": False, "battlements": False, "pillar_emphasis": False},
        "material_rules": {"color_palette": {"wall": [0.62, 0.40, 0.30]}, "surface_roughness": 0.5, "material_variation": 0.3},
        "group_rules": {"lot_fill_ratio": 0.7, "building_gap_min": 3.0, "building_gap_max": 6.0, "setback_min": 1.0, "setback_max": 2.0, "alignment": "aligned", "lot_width_min": 14.0, "lot_width_max": 20.0, "ambient_frequency": 5},
    },

    "modern_villa": {
        "_subdivision_override": 1,
        "_height_range_override": [3.5, 6.0],
        "_win_size_override": [2.0, 2.5],
        "_min_footprint": [10.0, 8.0],
        "silhouette_rules": {"width_to_height": 1.6, "symmetry": 0.3, "mass_distribution": "cantilevered_planes"},
        "roof_rules": {"type": "flat", "pitch": 0.0, "eave_overhang": 0.4, "ridge_decoration": False},
        "opening_rules": {"shape": "rectangular", "size": "large", "density": 0.6, "rhythm": "asymmetric_composed"},
        "edge_and_base_rules": {"base_height": 0.05, "corner_quoins": False, "buttress": False, "battlements": False, "pillar_emphasis": False},
        "material_rules": {"color_palette": {"wall": [0.93, 0.91, 0.88]}, "surface_roughness": 0.1, "material_variation": 0.05},
        "group_rules": {"lot_fill_ratio": 0.5, "building_gap_min": 6.0, "building_gap_max": 12.0, "setback_min": 2.0, "setback_max": 4.0, "alignment": "staggered", "lot_width_min": 16.0, "lot_width_max": 28.0, "ambient_frequency": 4},
    },

    # ─── Fantasy ────────────────────────────────────────────────
    "fantasy": {
        "_subdivision_override": 2,
        "_height_range_override": [6.0, 12.0],
        "_win_size_override": [0.8, 1.5],
        "_min_footprint": [6.0, 6.0],
        "silhouette_rules": {"width_to_height": 0.9, "symmetry": 0.5, "mass_distribution": "ornate_varied"},
        "roof_rules": {"type": "steep_varied", "pitch": 0.75, "eave_overhang": 0.25, "ridge_decoration": True},
        "opening_rules": {"shape": "arched", "size": "medium", "density": 0.3, "rhythm": "organic"},
        "edge_and_base_rules": {"base_height": 0.35, "corner_quoins": True, "buttress": True, "battlements": False, "pillar_emphasis": False},
        "material_rules": {"color_palette": {"wall": [0.70, 0.65, 0.58]}, "surface_roughness": 0.4, "material_variation": 0.3},
        "group_rules": {"lot_fill_ratio": 0.7, "building_gap_min": 3.0, "building_gap_max": 8.0, "setback_min": 1.0, "setback_max": 3.0, "alignment": "staggered", "lot_width_min": 12.0, "lot_width_max": 20.0, "ambient_frequency": 4},
    },

    "fantasy_dungeon": {
        "_subdivision_override": 1,
        "_height_range_override": [3.0, 5.0],
        "_win_size_override": [0.3, 0.5],
        "_min_footprint": [5.0, 5.0],
        "silhouette_rules": {"width_to_height": 1.3, "symmetry": 0.3, "mass_distribution": "squat_menacing"},
        "roof_rules": {"type": "flat_or_barrel_vault", "pitch": 0.15, "eave_overhang": 0.0, "ridge_decoration": False},
        "opening_rules": {"shape": "arched_low", "size": "minimal", "density": 0.08, "rhythm": "sparse_irregular"},
        "edge_and_base_rules": {"base_height": 0.5, "corner_quoins": True, "buttress": True, "battlements": False, "pillar_emphasis": False},
        "material_rules": {"color_palette": {"wall": [0.32, 0.30, 0.28]}, "surface_roughness": 0.85, "material_variation": 0.4},
        "group_rules": {"lot_fill_ratio": 0.75, "building_gap_min": 2.0, "building_gap_max": 5.0, "setback_min": 1.0, "setback_max": 3.0, "alignment": "irregular", "lot_width_min": 8.0, "lot_width_max": 14.0, "ambient_frequency": 4},
    },

    "fantasy_palace": {
        "_subdivision_override": 3,
        "_height_range_override": [10.0, 18.0],
        "_win_size_override": [0.9, 2.0],
        "_min_footprint": [8.0, 8.0],
        "silhouette_rules": {"width_to_height": 0.7, "symmetry": 0.75, "mass_distribution": "grand_spired"},
        "roof_rules": {"type": "pointed_spire", "pitch": 0.9, "eave_overhang": 0.2, "ridge_decoration": True},
        "opening_rules": {"shape": "pointed_arch", "size": "medium_tall", "density": 0.3, "rhythm": "regular_grand"},
        "edge_and_base_rules": {"base_height": 0.4, "corner_quoins": True, "buttress": True, "battlements": True, "pillar_emphasis": False},
        "material_rules": {"color_palette": {"wall": [0.85, 0.82, 0.78]}, "surface_roughness": 0.2, "material_variation": 0.15},
        "group_rules": {"lot_fill_ratio": 0.65, "building_gap_min": 4.0, "building_gap_max": 10.0, "setback_min": 1.5, "setback_max": 3.0, "alignment": "staggered", "lot_width_min": 14.0, "lot_width_max": 24.0, "ambient_frequency": 3},
    },

    # ─── Horror ─────────────────────────────────────────────────
    "horror": {
        "_subdivision_override": 2,
        "_height_range_override": [5.0, 10.0],
        "_win_size_override": [0.5, 0.9],
        "silhouette_rules": {"width_to_height": 0.75, "symmetry": 0.35, "mass_distribution": "gaunt_angular"},
        "roof_rules": {"type": "steep_broken", "pitch": 0.8, "eave_overhang": 0.2, "ridge_decoration": False},
        "opening_rules": {"shape": "irregular_gothic", "size": "small", "density": 0.2, "rhythm": "deliberately_off"},
        "edge_and_base_rules": {"base_height": 0.3, "corner_quoins": True, "buttress": False, "battlements": False, "pillar_emphasis": False},
        "material_rules": {"color_palette": {"wall": [0.42, 0.38, 0.35]}, "surface_roughness": 0.8, "material_variation": 0.5},
        "group_rules": {"lot_fill_ratio": 0.6, "building_gap_min": 4.0, "building_gap_max": 10.0, "setback_min": 1.5, "setback_max": 5.0, "alignment": "irregular", "lot_width_min": 8.0, "lot_width_max": 16.0, "ambient_frequency": 3},
    },

    "horror_asylum": {
        "_subdivision_override": 2,
        "_height_range_override": [5.0, 9.0],
        "_win_size_override": [0.5, 0.7],
        "silhouette_rules": {"width_to_height": 1.2, "symmetry": 0.7, "mass_distribution": "institutional_oppressive"},
        "roof_rules": {"type": "hipped_heavy", "pitch": 0.4, "eave_overhang": 0.1, "ridge_decoration": False},
        "opening_rules": {"shape": "barred_rectangular", "size": "small", "density": 0.25, "rhythm": "oppressive_regular"},
        "edge_and_base_rules": {"base_height": 0.35, "corner_quoins": True, "buttress": True, "battlements": False, "pillar_emphasis": False},
        "material_rules": {"color_palette": {"wall": [0.38, 0.30, 0.25]}, "surface_roughness": 0.7, "material_variation": 0.35},
        "group_rules": {"lot_fill_ratio": 0.7, "building_gap_min": 3.0, "building_gap_max": 6.0, "setback_min": 1.5, "setback_max": 3.0, "alignment": "aligned", "lot_width_min": 10.0, "lot_width_max": 16.0, "ambient_frequency": 5},
    },

    "horror_crypt": {
        "_subdivision_override": 1,
        "_height_range_override": [2.5, 4.0],
        "_win_size_override": [0.3, 0.4],
        "silhouette_rules": {"width_to_height": 1.5, "symmetry": 0.4, "mass_distribution": "sunken_heavy"},
        "roof_rules": {"type": "flat_or_barrel_vault", "pitch": 0.15, "eave_overhang": 0.0, "ridge_decoration": False},
        "opening_rules": {"shape": "arched_low", "size": "minimal", "density": 0.05, "rhythm": "single_entrance"},
        "edge_and_base_rules": {"base_height": 0.6, "corner_quoins": True, "buttress": False, "battlements": False, "pillar_emphasis": False},
        "material_rules": {"color_palette": {"wall": [0.35, 0.33, 0.30]}, "surface_roughness": 0.9, "material_variation": 0.45},
        "group_rules": {"lot_fill_ratio": 0.7, "building_gap_min": 3.0, "building_gap_max": 8.0, "setback_min": 1.0, "setback_max": 3.0, "alignment": "irregular", "lot_width_min": 8.0, "lot_width_max": 14.0, "ambient_frequency": 4},
    },

    # ─── Desert ─────────────────────────────────────────────────
    "desert": {
        "_subdivision_override": 1,
        "_height_range_override": [3.0, 5.0],
        "_win_size_override": [0.4, 0.5],
        "silhouette_rules": {"width_to_height": 1.3, "symmetry": 0.4, "mass_distribution": "cubic_stepped"},
        "roof_rules": {"type": "flat", "pitch": 0.0, "eave_overhang": 0.0, "ridge_decoration": False},
        "opening_rules": {"shape": "arched_small", "size": "small", "density": 0.12, "rhythm": "sparse"},
        "edge_and_base_rules": {"base_height": 0.1, "corner_quoins": False, "buttress": False, "battlements": False, "pillar_emphasis": False},
        "material_rules": {"color_palette": {"wall": [0.78, 0.65, 0.48]}, "surface_roughness": 0.6, "material_variation": 0.25},
        "group_rules": {"lot_fill_ratio": 0.7, "building_gap_min": 2.0, "building_gap_max": 5.0, "setback_min": 0.5, "setback_max": 2.0, "alignment": "irregular", "lot_width_min": 6.0, "lot_width_max": 12.0, "ambient_frequency": 4},
    },

    "desert_palace": {
        "_subdivision_override": 2,
        "_height_range_override": [6.0, 12.0],
        "_win_size_override": [0.6, 1.5],
        "_min_footprint": [8.0, 8.0],
        "silhouette_rules": {"width_to_height": 0.9, "symmetry": 0.8, "mass_distribution": "domed_ornate"},
        "roof_rules": {"type": "dome", "pitch": 0.6, "eave_overhang": 0.1, "ridge_decoration": True},
        "opening_rules": {"shape": "horseshoe_arch", "size": "medium_tall", "density": 0.2, "rhythm": "colonnade"},
        "edge_and_base_rules": {"base_height": 0.3, "corner_quoins": False, "buttress": False, "battlements": True, "pillar_emphasis": True},
        "material_rules": {"color_palette": {"wall": [0.82, 0.72, 0.58]}, "surface_roughness": 0.3, "material_variation": 0.2},
        "group_rules": {"lot_fill_ratio": 0.55, "building_gap_min": 5.0, "building_gap_max": 10.0, "setback_min": 2.0, "setback_max": 5.0, "alignment": "staggered", "lot_width_min": 10.0, "lot_width_max": 20.0, "ambient_frequency": 3},
    },
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


def apply_style_profile(params: dict, profile: dict, role: str = "primary") -> dict:
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

    # ─── Minimum footprint (prevent overly thin buildings) ───
    min_fp = profile.get("_min_footprint")
    if min_fp:
        result["_min_footprint_w"] = min_fp[0]
        result["_min_footprint_d"] = min_fp[1]

    # ─── Role-based inheritance decay ───
    if role == "secondary":
        result["column_count"] = max(0, result.get("column_count", 0) - 1)
        result["has_battlements"] = 0
        result["height_range"][0] *= 0.8
        result["height_range"][1] *= 0.85
        if "win_spec" in result:
            wd = result["win_spec"].get("density", 0.2)
            result["win_spec"]["density"] = round(min(0.4, wd * 1.3), 3)

    elif role == "tertiary":
        result["column_count"] = 0
        result["has_battlements"] = 0
        result["has_arch"] = 0
        result["height_range"][0] *= 0.6
        result["height_range"][1] *= 0.65
        result["eave_overhang"] = round(result.get("eave_overhang", 0) * 0.7, 3)
        result["wall_thickness"] = round(result.get("wall_thickness", 0.3) * 0.8, 2)
        if "win_spec" in result:
            wd = result["win_spec"].get("density", 0.2)
            result["win_spec"]["density"] = round(min(0.35, wd * 1.5), 3)

    elif role == "ambient":
        result["column_count"] = 0
        result["has_battlements"] = 0
        result["has_arch"] = 0
        result["height_range"][0] *= 0.4
        result["height_range"][1] *= 0.45
        result["eave_overhang"] = round(result.get("eave_overhang", 0) * 0.5, 3)
        if "win_spec" in result:
            result["win_spec"]["density"] = 0.05
        result["subdivision"] = 1

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
