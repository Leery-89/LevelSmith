"""
Kaer Morhen Comparison Experiment
=================================
Runs 3 cases under `medieval_keep` style, same seed:
  1. street  — generic linear settlement
  2. organic — clustered village
  3. graph   — fortified compound (kaer_morhen layout graph)

Outputs (per case):
  - GLB file
  - top-view image (orthographic Y-down)
  - perspective-view image
  - metrics dict

Final output:
  - experiments/kaer_morhen/comparison_report.json
  - experiments/kaer_morhen/*.glb, *.png

Usage:
  cd levelsmith
  python experiments/kaer_morhen_comparison.py
"""

import json
import math
import os
import sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

import numpy as np

# Suppress torch future warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = ROOT / "experiments" / "kaer_morhen"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Globals ──
STYLE = "medieval_keep"
SEED = 42
AREA = 100.0
COUNT = 7  # match kaer_morhen node count
MIN_GAP = 5.0


def run_case(label, layout_type, graph_name=None):
    """Run one generation case. Returns (scene, metrics)."""
    import level_layout

    print(f"\n{'='*60}")
    print(f"  Case: {label}")
    print(f"{'='*60}")

    glb_path = str(OUT_DIR / f"{label}.glb")

    scene = level_layout.generate_level(
        style=STYLE,
        layout_type=layout_type,
        building_count=COUNT,
        area_size=AREA,
        seed=SEED,
        min_gap=MIN_GAP,
        output_path=glb_path,
        graph_name=graph_name,
    )

    # ── Extract metrics from scene metadata ──
    meta = getattr(scene, "metadata", None) or {}
    clusters = meta.get("clusters", [])
    building_infos = meta.get("building_infos", [])
    road_nodes = meta.get("road_nodes", [])
    road_edges = meta.get("road_edges", [])
    road_renderable = meta.get("road_renderable", True)

    # Role distribution
    roles = {}
    for b in building_infos:
        r = "normal"
        for cl in clusters:
            if b.get("idx", -1) in cl.get("building_indices", []):
                if b.get("is_main_building"):
                    r = "main"
                break
        roles[r] = roles.get(r, 0) + 1

    # Graph mode: use block program for richer role info
    block_roles = {}
    if graph_name:
        from compound_layout import load_layout_graph, graph_to_block_program
        graph = load_layout_graph(graph_name)
        program = graph_to_block_program(graph, AREA, AREA,
                                         np.random.RandomState(SEED))
        for blk in program["blocks"]:
            block_roles[blk["role"]] = block_roles.get(blk["role"], 0) + 1
        walls_data = program.get("walls", {})
        gates = walls_data.get("gates", [])
        roads_data = program.get("roads", [])
    else:
        gates = []
        roads_data = []

    # Pairwise gap analysis
    overlap_count = 0
    too_close_count = 0
    for i in range(len(building_infos)):
        for j in range(i + 1, len(building_infos)):
            a, b = building_infos[i], building_infos[j]
            dx = abs(a["x"] - b["x"]) - (a["w"] + b["w"]) / 2
            dz = abs(a["z"] - b["z"]) - (a["d"] + b["d"]) / 2
            gap = max(dx, dz)
            if gap < 0:
                overlap_count += 1
            elif gap < 3.0:
                too_close_count += 1

    # Coverage
    total_fp = sum(b["w"] * b["d"] for b in building_infos)
    courtyard_area = AREA * AREA
    coverage = total_fp / courtyard_area if courtyard_area > 0 else 0

    # Keep info (primary / largest building)
    keep = None
    if building_infos:
        keep_b = max(building_infos, key=lambda b: b["w"] * b["d"])
        keep = {
            "x": round(keep_b["x"], 1),
            "z": round(keep_b["z"], 1),
            "w": round(keep_b["w"], 1),
            "d": round(keep_b["d"], 1),
        }

    # Tower count (secondary buildings or buildings near walls)
    tower_count = 0
    tower_positions = []
    wall_margin = AREA * 0.08
    for b in building_infos:
        near_wall = (b["x"] < wall_margin + 15 or
                     b["x"] > AREA - wall_margin - 15 or
                     b["z"] < wall_margin + 15 or
                     b["z"] > AREA - wall_margin - 15)
        if near_wall and keep and (b["x"] != keep["x"] or b["z"] != keep["z"]):
            tower_count += 1
            tower_positions.append({"x": round(b["x"], 1),
                                    "z": round(b["z"], 1)})

    # Entry axis check
    has_entry_axis = False
    if road_edges:
        has_entry_axis = True
    if graph_name and roads_data:
        has_entry_axis = True

    # Courtyard presence: gap between keep and perimeter buildings
    has_courtyard = coverage < 0.15 and len(building_infos) >= 3

    metrics = {
        "label": label,
        "layout_type": layout_type,
        "graph_name": graph_name,
        "building_count": len(building_infos),
        "block_count": len(building_infos),
        "role_distribution": block_roles if graph_name else roles,
        "overlap_count": overlap_count,
        "too_close_count": too_close_count,
        "coverage_ratio": round(coverage, 4),
        "keep_position": keep,
        "tower_count": tower_count,
        "tower_positions": tower_positions,
        "gate_count": len(gates),
        "gate_positions": [{"cx": g.get("cx", 0), "cz": g.get("cz", 0),
                            "faces": g.get("faces", "?")} for g in gates],
        "has_courtyard": has_courtyard,
        "has_entry_axis": has_entry_axis,
        "road_renderable": road_renderable,
        "road_node_count": len(road_nodes),
        "cluster_count": len(clusters),
        "cluster_sizes": [len(c.get("building_indices", [])) for c in clusters],
        "total_faces": sum(len(g.faces) for g in scene.geometry.values()),
    }

    return scene, metrics


def render_views(scene, label):
    """Render top-view and perspective images using trimesh."""
    try:
        import trimesh

        # Try to get scene bounds for camera setup
        all_verts = []
        for g in scene.geometry.values():
            all_verts.append(g.vertices)
        if not all_verts:
            return

        verts = np.vstack(all_verts)
        center = verts.mean(axis=0)
        extent = verts.max(axis=0) - verts.min(axis=0)
        max_dim = max(extent)

        # Top view (save scene extents as metadata for documentation)
        top_info = {
            "center": [round(float(c), 1) for c in center],
            "extent": [round(float(e), 1) for e in extent],
            "max_dim": round(float(max_dim), 1),
        }

        info_path = OUT_DIR / f"{label}_view_info.json"
        with open(info_path, "w") as f:
            json.dump(top_info, f, indent=2)

        # Try pyrender/pyglet for actual image rendering
        try:
            png_bytes = scene.save_image(resolution=(800, 600))
            if png_bytes:
                img_path = OUT_DIR / f"{label}_perspective.png"
                with open(img_path, "wb") as f:
                    f.write(png_bytes)
                print(f"  Saved {img_path.name}")
        except Exception:
            print(f"  [note] Image rendering not available (headless?)")

    except Exception as e:
        print(f"  [note] View generation skipped: {e}")


def interpret(m):
    """Return a short classification string for the metrics."""
    has_roles = any(k in m.get("role_distribution", {})
                    for k in ("primary", "secondary"))
    has_walls = m.get("gate_count", 0) > 0
    has_court = m.get("has_courtyard", False)
    cov = m.get("coverage_ratio", 0)

    if has_walls and has_roles and has_court:
        return "fortified compound"
    elif m.get("cluster_count", 0) >= 2 and cov < 0.12:
        return "clustered compound"
    else:
        return "generic settlement"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    cases = [
        ("street",  "street",  None),
        ("organic", "organic", None),
        ("graph",   "organic", "kaer_morhen"),
    ]

    all_metrics = []

    for label, layout, graph_name in cases:
        scene, metrics = run_case(label, layout, graph_name)
        metrics["interpretation"] = interpret(metrics)
        all_metrics.append(metrics)
        render_views(scene, label)

    # ── Write comparison report ──
    report = {
        "experiment": "kaer_morhen_comparison",
        "style": STYLE,
        "seed": SEED,
        "area": f"{AREA}x{AREA}m",
        "cases": all_metrics,
        "summary": {
            "street": all_metrics[0]["interpretation"],
            "organic": all_metrics[1]["interpretation"],
            "graph": all_metrics[2]["interpretation"],
        },
    }

    report_path = OUT_DIR / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    for m in all_metrics:
        print(f"\n  [{m['label']}] {m['interpretation']}")
        print(f"    buildings={m['building_count']}  "
              f"overlaps={m['overlap_count']}  "
              f"too_close={m['too_close_count']}  "
              f"coverage={m['coverage_ratio']:.1%}")
        print(f"    clusters={m['cluster_count']}  "
              f"towers={m['tower_count']}  "
              f"gates={m['gate_count']}  "
              f"entry_axis={m['has_entry_axis']}  "
              f"courtyard={m['has_courtyard']}")
        if m["keep_position"]:
            k = m["keep_position"]
            print(f"    keep=({k['x']},{k['z']}) {k['w']}x{k['d']}m")
        print(f"    roles={m['role_distribution']}")

    print(f"\n  Report: {report_path}")
    print(f"  Assets: {OUT_DIR}/")


if __name__ == "__main__":
    main()
