"""
compound_layout.py
Layout graph → block placement program for fortified compound scenes.

Reads a layout graph JSON (nodes + edges + enclosure + layout_priors)
and produces a block program that the existing generate_level() can consume.

Usage:
    from compound_layout import load_layout_graph, graph_to_block_program
    graph = load_layout_graph("kaer_morhen")
    program = graph_to_block_program(graph, 100, 100)
"""

import json
import os

import numpy as np


def load_layout_graph(name: str = "kaer_morhen") -> dict:
    """Load a layout graph JSON from training/data/."""
    path = os.path.join(os.path.dirname(__file__), "data", f"{name}_layout_graph.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def graph_to_block_program(graph: dict, area_w: float = 100.0,
                           area_d: float = 100.0,
                           rng: np.random.RandomState = None) -> dict:
    """
    Convert layout graph → block placement program.

    Rule-based (no LLM) — hardcoded spatial logic:
      primary   → center, offset toward rear (highest elevation)
      secondary → radially adjacent to primary
      tertiary  → courtyard fill
      ambient   → open spaces near primary
      walls     → perimeter rectangle with gates

    Returns dict with keys: blocks, walls, roads, metadata.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    enclosure = graph.get("enclosure", {})
    priors = graph.get("layout_priors", {})

    blocks = []
    cx = area_w / 2
    cz = area_d / 2

    # Keep reference position (primary block center) for secondary placement
    primary_cx, primary_cz = cx, cz

    # ─── Primary: center of compound, offset toward rear ───
    for node in nodes:
        if node["role"] == "primary":
            primary_cz = cz + area_d * 0.15
            blocks.append({
                "id": node["id"],
                "role": "primary",
                "style_key": "medieval_keep",
                "building_type": node.get("inferred_function", "keep"),
                "cx": cx,
                "cz": primary_cz,
                "w": area_w * 0.18,
                "d": area_d * 0.14,
                "elevation": 0,
                "yaw": 0,
            })

    # ─── Secondary: adjacent to primary (radial placement) ───
    secondary_nodes = [n for n in nodes if n["role"] == "secondary"]
    if secondary_nodes:
        n_sec = len(secondary_nodes)
        # Spread from east through south to west (front half arc)
        angles = np.linspace(-np.pi * 0.4, np.pi * 0.4, n_sec)
        # Offset so defensive towers go toward entrance (south)
        radius = area_w * 0.22

        for i, node in enumerate(secondary_nodes):
            angle = angles[i]
            bx = primary_cx + radius * np.sin(angle)
            bz = primary_cz - radius * np.cos(angle)

            # Clamp to scene bounds
            bx = max(area_w * 0.15, min(area_w * 0.85, bx))
            bz = max(area_d * 0.15, min(area_d * 0.85, bz))

            func = node.get("inferred_function", "tower")
            if "tower" in func:
                style_key = "medieval_keep"
                w, d = area_w * 0.08, area_d * 0.08
            else:
                style_key = "medieval"
                w, d = area_w * 0.12, area_d * 0.10

            blocks.append({
                "id": node["id"],
                "role": "secondary",
                "style_key": style_key,
                "building_type": func,
                "cx": bx, "cz": bz,
                "w": w, "d": d,
                "elevation": 0,
                "yaw": float(rng.uniform(-15, 15)),
            })

    # ─── Tertiary + Ambient: courtyard fill ───
    courtyard_nodes = [n for n in nodes if n["role"] in ("tertiary", "ambient")]
    # Place in courtyard between primary and entrance (southern half)
    courtyard_cx = cx
    courtyard_cz = cz - area_d * 0.05  # slightly south of center

    for i, node in enumerate(courtyard_nodes):
        angle = float(rng.uniform(0, 2 * np.pi))
        r = area_w * 0.12 * float(rng.uniform(0.3, 0.9))
        bx = courtyard_cx + r * np.cos(angle)
        bz = courtyard_cz + r * np.sin(angle)

        bx = max(area_w * 0.2, min(area_w * 0.8, bx))
        bz = max(area_d * 0.2, min(area_d * 0.8, bz))

        blocks.append({
            "id": node["id"],
            "role": node["role"],
            "style_key": "medieval",
            "building_type": node.get("inferred_function", "misc"),
            "cx": bx, "cz": bz,
            "w": area_w * 0.06, "d": area_d * 0.06,
            "elevation": 0,
            "yaw": float(rng.uniform(0, 360)),
        })

    # ─── Walls: perimeter rectangle ───
    wall_margin = area_w * 0.08
    wall_perimeter = [
        (wall_margin, wall_margin),
        (area_w - wall_margin, wall_margin),
        (area_w - wall_margin, area_d - wall_margin),
        (wall_margin, area_d - wall_margin),
    ]

    # ─── Gates from enclosure data ───
    gate_types = enclosure.get("gate_types", ["main_gate"])
    gates = [{
        "type": "main_gate",
        "cx": cx,
        "cz": wall_margin,
        "faces": "south",
        "width": 8,
    }]
    if len(gate_types) > 1:
        gates.append({
            "type": gate_types[1],
            "cx": cx,
            "cz": area_d - wall_margin,
            "faces": "north",
            "width": 5,
        })

    walls = {
        "type": "curtain_wall",
        "perimeter": wall_perimeter,
        "gates": gates,
        "tower_interval": 25,
        "battlement_variants": enclosure.get("battlement_variants", 4),
    }

    # ─── Approach road: barbican → keep ───
    roads = [{
        "type": "approach_axis",
        "points": [(cx, wall_margin), (cx, primary_cz - area_d * 0.07)],
    }]

    return {
        "blocks": blocks,
        "walls": walls,
        "roads": roads,
        "metadata": {
            "source": graph.get("scene", "unknown"),
            "topology": graph.get("topology", "unknown"),
            "node_count": len(blocks),
            "edge_count": len(edges),
        },
    }


if __name__ == "__main__":
    graph = load_layout_graph("kaer_morhen")
    program = graph_to_block_program(graph, 100, 100, np.random.RandomState(42))

    print(f"Blocks: {len(program['blocks'])}")
    for b in program["blocks"]:
        print(f"  {b['id']:25s} role={b['role']:10s} style={b['style_key']:15s} "
              f"pos=({b['cx']:.1f},{b['cz']:.1f}) size=({b['w']:.1f}x{b['d']:.1f})")

    print(f"\nWalls: {len(program['walls']['perimeter'])} corners, "
          f"{len(program['walls']['gates'])} gates")
    for g in program["walls"]["gates"]:
        print(f"  gate: {g['type']} at ({g['cx']:.1f},{g['cz']:.1f}) "
              f"faces={g['faces']} w={g['width']}m")

    print(f"\nRoads: {len(program['roads'])}")
    print(f"Metadata: {program['metadata']}")
