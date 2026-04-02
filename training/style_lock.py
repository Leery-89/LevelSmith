"""
Style Lock — Graybox baseline metrics and validation.

Freezes the 5-dimension evaluation results as acceptable ranges.
Run validate_style_lock() after any change to ensure style differentiation
hasn't been diluted.
"""

STYLE_BASELINES = {
    "japanese": {
        "silhouette": {
            "avg_height": {"target": 4.1, "range": [2.5, 5.5]},
            "width_height_ratio": {"target": 2.43, "range": [1.8, 3.0]},
        },
        "roof": {
            "roof_height": {"target": 2.5, "range": [1.5, 3.5]},
            "eave_overhang": {"target": 2.7, "range": [2.0, 4.0]},
            "type": "hipped",
        },
        "opening": {
            "windows_per_bldg": {"target": 2.6, "range": [1, 5]},
            "window_wh_ratio": {"target": 1.50, "range": [1.2, 2.0]},
        },
        "base_eave": {
            "wall_thickness": {"target": 0.25, "range": [0.15, 0.35]},
            "battlements": False,
            "columns_min": 4,
        },
        "massing": {
            "coverage_ratio": {"target": 0.106, "range": [0.05, 0.18]},
        },
    },
    "medieval": {
        "silhouette": {
            "avg_height": {"target": 8.5, "range": [6.0, 11.0]},
            "width_height_ratio": {"target": 1.29, "range": [0.8, 1.6]},
        },
        "roof": {
            "roof_height": {"target": 4.1, "range": [3.0, 6.0]},
            "eave_overhang": {"target": 0.5, "range": [0.2, 1.0]},
            "type": "gabled",
        },
        "opening": {
            "windows_per_bldg": {"target": 6.5, "range": [3, 10]},
            "window_wh_ratio": {"target": 0.44, "range": [0.3, 0.6]},
        },
        "base_eave": {
            "wall_thickness": {"target": 0.80, "range": [0.6, 1.0]},
            "battlements": True,
            "columns_min": 1,
        },
        "massing": {
            "coverage_ratio": {"target": 0.116, "range": [0.08, 0.20]},
        },
    },
    "industrial": {
        "silhouette": {
            "avg_height": {"target": 9.8, "range": [7.0, 13.0]},
            "width_height_ratio": {"target": 1.64, "range": [1.2, 2.0]},
        },
        "roof": {
            "roof_height": {"target": 0.6, "range": [0.0, 1.2]},
            "eave_overhang": {"target": 0.1, "range": [0.0, 0.3]},
            "type": "flat",
        },
        "opening": {
            "windows_per_bldg": {"target": 21.6, "range": [15, 30]},
            "window_wh_ratio": {"target": 1.00, "range": [0.8, 1.2]},
        },
        "base_eave": {
            "wall_thickness": {"target": 0.28, "range": [0.2, 0.4]},
            "battlements": False,
            "columns_min": 0,
        },
        "massing": {
            "coverage_ratio": {"target": 0.303, "range": [0.20, 0.45]},
        },
    },
}


def validate_style_lock(style: str, metrics: dict) -> dict:
    """
    Validate generation metrics against frozen baselines.

    metrics keys: avg_height, width_height_ratio, roof_height, eave_overhang,
                  windows_per_bldg, window_wh_ratio, wall_thickness, coverage_ratio

    Returns: {"passed": bool, "violations": [str]}
    """
    baseline = STYLE_BASELINES.get(style)
    if not baseline:
        return {"passed": True, "violations": []}

    violations = []
    for dimension, checks in baseline.items():
        if not isinstance(checks, dict):
            continue
        for metric, spec in checks.items():
            if isinstance(spec, dict) and "range" in spec:
                val = metrics.get(metric)
                if val is not None:
                    lo, hi = spec["range"]
                    if val < lo or val > hi:
                        violations.append(
                            f"{dimension}.{metric}: {val:.2f} outside [{lo}, {hi}]")

    return {"passed": len(violations) == 0, "violations": violations}
