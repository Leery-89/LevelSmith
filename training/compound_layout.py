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


# ─── Collision detection ───────────────────────────────────────────

def _blocks_overlap(b1: dict, b2: dict, min_gap: float = 3.0) -> bool:
    """AABB overlap test between two blocks (center-based) with min_gap."""
    hw1 = b1["w"] / 2 + min_gap
    hd1 = b1["d"] / 2 + min_gap
    hw2 = b2["w"] / 2
    hd2 = b2["d"] / 2
    return (abs(b1["cx"] - b2["cx"]) < hw1 + hw2 and
            abs(b1["cz"] - b2["cz"]) < hd1 + hd2)


def _out_of_bounds(b: dict, wall_margin: float,
                   area_w: float, area_d: float, inset: float = 3.0) -> bool:
    """Check if block extends outside the walled area (with inset from wall)."""
    hw, hd = b["w"] / 2, b["d"] / 2
    lo = wall_margin + inset
    return (b["cx"] - hw < lo or b["cx"] + hw > area_w - lo or
            b["cz"] - hd < lo or b["cz"] + hd > area_d - lo)


def _place_with_collision(candidate: dict, placed: list,
                          min_gap: float = 3.0) -> bool:
    """Return True if candidate does NOT overlap any placed block."""
    return not any(_blocks_overlap(candidate, p, min_gap) for p in placed)


# ─── Main converter ───────────────────────────────────────────────

def graph_to_block_program(graph: dict, area_w: float = 100.0,
                           area_d: float = 100.0,
                           rng: np.random.RandomState = None) -> dict:
    """
    Convert layout graph → block placement program.

    Rule-based (no LLM) — hardcoded spatial logic:
      primary   → center, offset toward rear (highest elevation)
      secondary → along inner wall perimeter (towers at corners)
      tertiary  → courtyard fill with collision avoidance
      ambient   → open spaces with collision avoidance
      walls     → perimeter rectangle with gates

    All placements use AABB collision detection with min_gap.

    Returns dict with keys: blocks, walls, roads, metadata.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    enclosure = graph.get("enclosure", {})

    wall_margin = area_w * 0.08
    wall_inset = 5.0  # building offset from inner wall face
    cx = area_w / 2
    cz = area_d / 2

    placed = []  # collision tracking list

    # ─── Primary: center of compound, offset toward rear ───
    primary_cx, primary_cz = cx, cz + area_d * 0.2
    for node in nodes:
        if node["role"] == "primary":
            block = {
                "id": node["id"],
                "role": "primary",
                "style_key": "medieval_keep",
                "building_type": node.get("inferred_function", "keep"),
                "cx": primary_cx,
                "cz": primary_cz,
                "w": area_w * 0.15,
                "d": area_d * 0.12,
                "elevation": 0,
                "yaw": 0,
            }
            placed.append(block)

    # ─── Secondary: towers along inner wall perimeter ───
    secondary_nodes = [n for n in nodes if n["role"] == "secondary"]
    inner_lo = wall_margin + wall_inset
    inner_hi_w = area_w - wall_margin - wall_inset
    inner_hi_d = area_d - wall_margin - wall_inset

    # Predefined tower positions: corners + mid-wall
    tower_slots = [
        (inner_lo,   inner_lo),                       # SW corner
        (inner_hi_w, inner_lo),                       # SE corner
        (inner_lo,   inner_hi_d),                     # NW corner
        (inner_hi_w, inner_hi_d),                     # NE corner
        (cx,         inner_lo),                       # S mid (near gate)
        (inner_lo,   cz),                             # W mid
        (inner_hi_w, cz),                             # E mid
    ]

    # Place gatehouses first at south-center (gate position)
    gatehouse_nodes = [n for n in secondary_nodes
                       if "gatehouse" in n.get("inferred_function", "")]
    tower_only_nodes = [n for n in secondary_nodes if n not in gatehouse_nodes]

    for node in gatehouse_nodes:
        w, d = area_w * 0.10, area_d * 0.08
        candidate = {
            "id": node["id"], "role": "secondary",
            "style_key": "medieval_keep",
            "building_type": node.get("inferred_function", "gatehouse"),
            "cx": cx, "cz": inner_lo, "w": w, "d": d,
            "elevation": 0, "yaw": 0,
        }
        if _place_with_collision(candidate, placed, min_gap=5.0):
            placed.append(candidate)
        else:
            placed.append(candidate)  # gatehouse gets priority

    for i, node in enumerate(tower_only_nodes):
        func = node.get("inferred_function", "tower")
        if "tower" in func:
            style_key = "medieval_keep"
            w, d = area_w * 0.08, area_d * 0.08
        else:
            style_key = "medieval"
            w, d = area_w * 0.10, area_d * 0.08

        block = None
        for attempt in range(20):
            if attempt < len(tower_slots):
                slot_idx = (i + attempt) % len(tower_slots)
                tx, tz = tower_slots[slot_idx]
            else:
                # Fallback: random position along inner wall perimeter
                side = rng.randint(0, 4)
                if side == 0:    # south
                    tx = float(rng.uniform(inner_lo, inner_hi_w))
                    tz = inner_lo
                elif side == 1:  # north
                    tx = float(rng.uniform(inner_lo, inner_hi_w))
                    tz = inner_hi_d
                elif side == 2:  # west
                    tx = inner_lo
                    tz = float(rng.uniform(inner_lo, inner_hi_d))
                else:            # east
                    tx = inner_hi_w
                    tz = float(rng.uniform(inner_lo, inner_hi_d))

            candidate = {
                "id": node["id"], "role": "secondary",
                "style_key": style_key,
                "building_type": func,
                "cx": tx, "cz": tz, "w": w, "d": d,
                "elevation": 0,
                "yaw": float(rng.uniform(-10, 10)),
            }
            if _place_with_collision(candidate, placed, min_gap=5.0):
                block = candidate
                break

        if block is None:
            # Last resort: place at proposed position anyway
            block = candidate
        placed.append(block)

    # ─── Tertiary + Ambient: courtyard fill ───
    courtyard_nodes = [n for n in nodes if n["role"] in ("tertiary", "ambient")]
    # Courtyard = area between keep and south wall
    court_x_lo = wall_margin + wall_inset + 8
    court_x_hi = area_w - wall_margin - wall_inset - 8
    court_z_lo = wall_margin + wall_inset + 8
    court_z_hi = primary_cz - 10  # stop well before the keep

    for node in courtyard_nodes:
        w = area_w * 0.05
        d = area_d * 0.05

        block = None
        for attempt in range(30):
            bx = float(rng.uniform(court_x_lo, court_x_hi))
            bz = float(rng.uniform(court_z_lo, court_z_hi))
            candidate = {
                "id": node["id"], "role": node["role"],
                "style_key": "medieval",
                "building_type": node.get("inferred_function", "misc"),
                "cx": bx, "cz": bz, "w": w, "d": d,
                "elevation": 0,
                "yaw": float(rng.uniform(0, 360)),
            }
            if (_place_with_collision(candidate, placed, min_gap=5.0) and
                    not _out_of_bounds(candidate, wall_margin, area_w, area_d)):
                block = candidate
                break

        if block is None:
            block = candidate
        placed.append(block)

    # ─── Coverage check — shrink small buildings if too dense ───
    courtyard_area = ((area_w - 2 * wall_margin) * (area_d - 2 * wall_margin))
    total_fp = sum(b["w"] * b["d"] for b in placed)
    coverage = total_fp / courtyard_area if courtyard_area > 0 else 1.0

    if coverage > 0.35:
        scale = (0.30 * courtyard_area / total_fp) ** 0.5
        for b in placed:
            if b["role"] in ("tertiary", "ambient"):
                b["w"] *= scale
                b["d"] *= scale

    # Recompute final coverage
    total_fp = sum(b["w"] * b["d"] for b in placed)
    coverage = total_fp / courtyard_area if courtyard_area > 0 else 0

    # ─── Walls: perimeter rectangle ───
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

    # ─── Approach road: gate → keep ───
    roads = [{
        "type": "approach_axis",
        "points": [(cx, wall_margin), (cx, primary_cz - 10)],
    }]

    return {
        "blocks": placed,
        "walls": walls,
        "roads": roads,
        "metadata": {
            "source": graph.get("scene", "unknown"),
            "topology": graph.get("topology", "unknown"),
            "node_count": len(placed),
            "edge_count": len(edges),
            "courtyard_coverage": round(coverage, 3),
        },
    }


if __name__ == "__main__":
    graph = load_layout_graph("kaer_morhen")
    program = graph_to_block_program(graph, 100, 100, np.random.RandomState(42))

    print(f"Blocks: {len(program['blocks'])}")
    for b in program["blocks"]:
        print(f"  {b['id']:25s} role={b['role']:10s} style={b['style_key']:15s} "
              f"pos=({b['cx']:.1f},{b['cz']:.1f}) size=({b['w']:.1f}x{b['d']:.1f})")

    # Pairwise distances
    blocks = program["blocks"]
    print(f"\nPairwise distances:")
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            dx = blocks[i]["cx"] - blocks[j]["cx"]
            dz = blocks[i]["cz"] - blocks[j]["cz"]
            dist = (dx**2 + dz**2) ** 0.5
            gap_w = abs(dx) - blocks[i]["w"]/2 - blocks[j]["w"]/2
            gap_d = abs(dz) - blocks[i]["d"]/2 - blocks[j]["d"]/2
            gap = max(gap_w, gap_d)
            overlap = "OVERLAP" if gap < 0 else "ok"
            print(f"  {blocks[i]['id']:20s} ↔ {blocks[j]['id']:20s}"
                  f"  dist={dist:6.1f}m  gap={gap:6.1f}m  {overlap}")

    print(f"\nWalls: {len(program['walls']['perimeter'])} corners, "
          f"{len(program['walls']['gates'])} gates")
    print(f"Courtyard coverage: {program['metadata']['courtyard_coverage']:.1%}")
