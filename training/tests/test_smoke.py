"""
LevelSmith smoke tests — lightweight checks that core modules load
and produce correct shapes/types without running full generation.

Run:  cd training && python -m pytest tests/ -v
"""

import sys
import os

# Ensure the training/ directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


# ── Style registry ──────────────────────────────────────────────

def test_style_registry_loads():
    from style_registry import STYLE_REGISTRY, FEATURE_DIM, OUTPUT_DIM
    assert len(STYLE_REGISTRY) == 20, f"Expected 20 styles, got {len(STYLE_REGISTRY)}"
    assert FEATURE_DIM == 16
    assert OUTPUT_DIM == 23


def test_all_styles_have_correct_feature_dim():
    from style_registry import STYLE_REGISTRY, FEATURE_DIM
    for name, style in STYLE_REGISTRY.items():
        assert len(style.feature_vector) == FEATURE_DIM, (
            f"Style '{name}' has {len(style.feature_vector)}-dim features, expected {FEATURE_DIM}"
        )


def test_all_styles_have_base_params():
    from style_registry import STYLE_REGISTRY, OUTPUT_KEYS
    required_base = {"height_range_min", "height_range_max", "wall_thickness",
                     "door_width", "door_height", "win_width", "win_height"}
    for name, style in STYLE_REGISTRY.items():
        for key in required_base:
            assert key in style.base_params, (
                f"Style '{name}' missing base param '{key}'"
            )


def test_get_param_vector_shape():
    from style_registry import STYLE_REGISTRY, get_param_vector, OUTPUT_DIM
    for name in STYLE_REGISTRY:
        vec = get_param_vector(name)
        assert vec.shape == (OUTPUT_DIM,), (
            f"Style '{name}' param vector shape {vec.shape}, expected ({OUTPUT_DIM},)"
        )
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0), (
            f"Style '{name}' param vector has values outside [0,1]"
        )


def test_get_feature_vector_shape():
    from style_registry import STYLE_REGISTRY, get_feature_vector, FEATURE_DIM
    for name in STYLE_REGISTRY:
        vec = get_feature_vector(name)
        assert vec.shape == (FEATURE_DIM,), (
            f"Style '{name}' feature vector shape {vec.shape}, expected ({FEATURE_DIM},)"
        )


# ── Model ───────────────────────────────────────────────────────

def test_model_forward_pass():
    import torch
    from model import StyleParamMLP

    model = StyleParamMLP(input_dim=16, output_dim=23)
    model.eval()
    x = torch.rand(4, 16)  # batch of 4
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 23), f"Model output shape {y.shape}, expected (4, 23)"
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0), "Sigmoid output outside [0,1]"


def test_model_parameter_count():
    from model import StyleParamMLP, count_parameters

    model = StyleParamMLP(input_dim=16, output_dim=23)
    params = count_parameters(model)
    assert params == 13719, f"Expected 13719 params, got {params}"


def test_build_model_default():
    from model import build_model
    model = build_model(device="cpu")
    assert model is not None


# ── Denormalization ─────────────────────────────────────────────

def test_denormalize_params_keys():
    from style_registry import denormalize_params, OUTPUT_KEYS
    vec = np.full(23, 0.5, dtype=np.float32)
    result = denormalize_params(vec)
    assert set(result.keys()) == set(OUTPUT_KEYS)


def test_denormalize_boundary_values():
    from style_registry import denormalize_params, OUTPUT_PARAMS, OUTPUT_KEYS
    # All zeros → should return min values
    vec_min = np.zeros(23, dtype=np.float32)
    result_min = denormalize_params(vec_min)
    for key in OUTPUT_KEYS:
        lo = OUTPUT_PARAMS[key]["range"][0]
        assert abs(result_min[key] - lo) < 0.01, (
            f"Param '{key}' at 0.0 should be ~{lo}, got {result_min[key]}"
        )

    # All ones → should return max values
    vec_max = np.ones(23, dtype=np.float32)
    result_max = denormalize_params(vec_max)
    for key in OUTPUT_KEYS:
        hi = OUTPUT_PARAMS[key]["range"][1]
        assert abs(result_max[key] - hi) < 0.01, (
            f"Param '{key}' at 1.0 should be ~{hi}, got {result_max[key]}"
        )


# ── Layout constants ────────────────────────────────────────────

def test_style_default_layout_mapping():
    """Every style should have a default layout for auto-selection."""
    from api import _STYLE_DEFAULT_LAYOUT
    from style_registry import STYLE_REGISTRY
    valid_layouts = {"street", "grid", "plaza", "organic", "random"}
    for style_name in STYLE_REGISTRY:
        layout = _STYLE_DEFAULT_LAYOUT.get(style_name)
        assert layout is not None, f"Style '{style_name}' has no default layout"
        assert layout in valid_layouts, (
            f"Style '{style_name}' default layout '{layout}' not in {valid_layouts}"
        )
