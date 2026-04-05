"""
Validate graph family files and the JSONL training dataset.

Usage:
    python validate_graph_families.py                 # validate all
    python validate_graph_families.py --stats         # print dataset statistics
    python validate_graph_families.py --family NAME   # inspect one family file
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "data"
FAMILIES_DIR = DATA_DIR / "graph_families"
DATASET_PATH = DATA_DIR / "graph_family_dataset.jsonl"
SCHEMA_PATH = DATA_DIR / "graph_family_schema.json"

# ── Vocabulary (from schema) ──────────────────────────────────────

VALID_INTENTS = {"defense", "residence", "worship", "commerce", "industry", "governance", "mixed"}
VALID_ENTRY_MODES = {"single_gate", "double_gate", "open_approach", "bridge_crossing", "tunnel", "multi_gate", "ramp"}
VALID_PERIMETER = {"curtain_wall", "palisade", "moat_and_wall", "cliff_edge", "open", "partial_wall", "terraced_wall"}
VALID_TOWER_DIST = {"corners_only", "corners_and_midwall", "gate_flanking", "regular_interval", "clustered_at_entry", "none"}
VALID_COURTYARD = {"single_open", "divided", "tiered", "cloistered", "none", "central_plaza", "ring"}
VALID_DAMAGE = {"pristine", "weathered", "battle_scarred", "ruined", "overgrown"}
VALID_ROLES = {"primary", "secondary", "tertiary", "ambient"}
VALID_ELEVATIONS = {"highest", "high", "mid", "low"}
VALID_SIZE_HINTS = {"large", "medium", "small", "tiny"}
VALID_EDGE_TYPES = {"approach_axis", "adjacent", "flanking", "within_courtyard",
                    "across_bridge", "above", "below", "faces", "backs_onto", "along_wall"}
VALID_SYMMETRY = {"axial", "bilateral", "radial", "none"}

NODE_VOCABULARY = {
    "central_keep", "great_hall", "throne_room",
    "gatehouse", "defensive_tower", "corner_tower", "watchtower",
    "residential_tower", "barracks", "guard_post",
    "chapel", "temple", "shrine",
    "stable", "smithy", "granary", "storehouse", "armory",
    "market", "tavern", "inn",
    "well", "fountain", "cistern",
    "training_ground", "arena",
    "garden", "courtyard_tree", "statue",
    "workshop", "furnace", "chimney_stack", "warehouse",
    "dock", "crane", "silo",
    "prison", "dungeon",
    "library", "scriptorium",
    "kitchen", "bakery",
    "cemetery", "crypt",
}


def validate_family_file(path: Path) -> list[str]:
    """Validate a single graph family JSON file. Returns list of error strings."""
    errors = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"JSON parse error: {e}"]

    # Top-level required fields
    for field in ["family_name", "intent", "style_bias", "entry_mode", "perimeter_type",
                  "tower_distribution", "courtyard_mode", "support_program", "damage_state", "prototype"]:
        if field not in data:
            errors.append(f"missing required field: {field}")

    # Enum validation
    if data.get("intent") not in VALID_INTENTS:
        errors.append(f"invalid intent: {data.get('intent')}")
    if data.get("entry_mode") not in VALID_ENTRY_MODES:
        errors.append(f"invalid entry_mode: {data.get('entry_mode')}")
    if data.get("perimeter_type") not in VALID_PERIMETER:
        errors.append(f"invalid perimeter_type: {data.get('perimeter_type')}")
    if data.get("tower_distribution") not in VALID_TOWER_DIST:
        errors.append(f"invalid tower_distribution: {data.get('tower_distribution')}")
    if data.get("courtyard_mode") not in VALID_COURTYARD:
        errors.append(f"invalid courtyard_mode: {data.get('courtyard_mode')}")
    if data.get("damage_state") not in VALID_DAMAGE:
        errors.append(f"invalid damage_state: {data.get('damage_state')}")

    # Support program items should be in node vocabulary
    for item in data.get("support_program", []):
        if item not in NODE_VOCABULARY:
            errors.append(f"support_program item not in node vocabulary: {item}")

    # Prototype validation
    proto = data.get("prototype", {})
    node_ids = set()
    has_primary = False

    for node in proto.get("nodes", []):
        nid = node.get("id", "")
        if not nid:
            errors.append("node missing id")
        if nid in node_ids:
            errors.append(f"duplicate node id: {nid}")
        node_ids.add(nid)

        role = node.get("role")
        if role not in VALID_ROLES:
            errors.append(f"node {nid}: invalid role {role}")
        if role == "primary":
            has_primary = True

        func = node.get("function")
        if func not in NODE_VOCABULARY:
            errors.append(f"node {nid}: function '{func}' not in vocabulary")

        elev = node.get("elevation", "low")
        if elev not in VALID_ELEVATIONS:
            errors.append(f"node {nid}: invalid elevation {elev}")

        size = node.get("size_hint", "medium")
        if size not in VALID_SIZE_HINTS:
            errors.append(f"node {nid}: invalid size_hint {size}")

    if not has_primary:
        errors.append("no primary node defined")

    for edge in proto.get("edges", []):
        if edge.get("from") not in node_ids:
            errors.append(f"edge references unknown node: {edge.get('from')}")
        if edge.get("to") not in node_ids:
            errors.append(f"edge references unknown node: {edge.get('to')}")
        if edge.get("type") not in VALID_EDGE_TYPES:
            errors.append(f"edge {edge.get('from')}->{edge.get('to')}: invalid type {edge.get('type')}")

    # Check approach_axis exists
    has_approach = any(e.get("type") == "approach_axis" for e in proto.get("edges", []))
    if not has_approach and data.get("entry_mode") != "open_approach":
        errors.append("no approach_axis edge but entry_mode is not open_approach")

    # Placement hints
    hints = proto.get("placement_hints", {})
    sym = hints.get("symmetry")
    if sym and sym not in VALID_SYMMETRY:
        errors.append(f"invalid symmetry: {sym}")

    return errors


def validate_dataset(path: Path) -> tuple[int, int, list[str]]:
    """Validate JSONL dataset. Returns (total, valid, errors)."""
    errors = []
    total = 0
    valid = 0

    for i, line in enumerate(path.read_text(encoding="utf-8").strip().split("\n"), 1):
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {i}: JSON parse error: {e}")
            continue

        line_errors = []
        for field in ["prompt", "style", "intent", "graph_family"]:
            if field not in row:
                line_errors.append(f"missing {field}")

        if row.get("intent") not in VALID_INTENTS:
            line_errors.append(f"invalid intent: {row.get('intent')}")
        if row.get("entry_mode") and row["entry_mode"] not in VALID_ENTRY_MODES:
            line_errors.append(f"invalid entry_mode: {row['entry_mode']}")
        if row.get("perimeter_type") and row["perimeter_type"] not in VALID_PERIMETER:
            line_errors.append(f"invalid perimeter_type: {row['perimeter_type']}")
        if row.get("tower_distribution") and row["tower_distribution"] not in VALID_TOWER_DIST:
            line_errors.append(f"invalid tower_distribution: {row['tower_distribution']}")
        if row.get("courtyard_mode") and row["courtyard_mode"] not in VALID_COURTYARD:
            line_errors.append(f"invalid courtyard_mode: {row['courtyard_mode']}")
        if row.get("damage_state") and row["damage_state"] not in VALID_DAMAGE:
            line_errors.append(f"invalid damage_state: {row['damage_state']}")

        for item in row.get("support_program", []):
            if item not in NODE_VOCABULARY:
                line_errors.append(f"support_program '{item}' not in vocabulary")

        if line_errors:
            errors.extend([f"line {i}: {e}" for e in line_errors])
        else:
            valid += 1

    return total, valid, errors


def print_stats(path: Path):
    """Print dataset distribution statistics."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().split("\n")]

    print(f"\n{'='*60}")
    print(f"  Graph Family Dataset Statistics")
    print(f"  {len(rows)} examples")
    print(f"{'='*60}")

    for field in ["graph_family", "intent", "style", "entry_mode", "perimeter_type",
                  "tower_distribution", "courtyard_mode", "damage_state"]:
        counts = Counter(row.get(field, "N/A") for row in rows)
        print(f"\n  {field}:")
        for val, count in counts.most_common():
            bar = "#" * count
            print(f"    {val:25s} {count:3d}  {bar}")

    # Support program frequency
    sp_counts = Counter()
    for row in rows:
        for item in row.get("support_program", []):
            sp_counts[item] += 1
    print(f"\n  support_program (top 15):")
    for item, count in sp_counts.most_common(15):
        bar = "#" * count
        print(f"    {item:25s} {count:3d}  {bar}")


def inspect_family(name: str):
    """Pretty-print a family file."""
    path = FAMILIES_DIR / f"{name}.json"
    if not path.exists():
        print(f"Family file not found: {path}")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    proto = data.get("prototype", {})
    nodes = proto.get("nodes", [])
    edges = proto.get("edges", [])

    print(f"\n{'='*60}")
    print(f"  Family: {data['family_name']}")
    print(f"{'='*60}")
    print(f"  Intent:       {data['intent']}")
    print(f"  Style bias:   {data['style_bias']}")
    print(f"  Entry:        {data['entry_mode']}")
    print(f"  Perimeter:    {data['perimeter_type']}")
    print(f"  Towers:       {data['tower_distribution']}")
    print(f"  Courtyard:    {data['courtyard_mode']}")
    print(f"  Damage:       {data['damage_state']}")
    print(f"  Support:      {', '.join(data['support_program'])}")
    print(f"\n  Nodes ({len(nodes)}):")
    for n in nodes:
        print(f"    [{n['role']:10s}] {n['id']:20s}  {n['function']:20s}  elev={n.get('elevation','low'):7s}  size={n.get('size_hint','medium')}")
    print(f"\n  Edges ({len(edges)}):")
    for e in edges:
        print(f"    {e['from']:20s} --{e['type']:20s}--> {e['to']}")

    enc = proto.get("enclosure", {})
    if enc:
        print(f"\n  Enclosure:")
        print(f"    walls={enc.get('wall_segments',0)}  gates={enc.get('gate_count',0)}  bridges={enc.get('bridge_count',0)}  stairs={enc.get('stair_count',0)}")

    hints = proto.get("placement_hints", {})
    if hints:
        print(f"\n  Placement hints:")
        for k, v in hints.items():
            print(f"    {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Validate graph family schema and dataset")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--family", type=str, help="Inspect a specific family")
    args = parser.parse_args()

    if args.family:
        inspect_family(args.family)
        return

    if args.stats:
        if DATASET_PATH.exists():
            print_stats(DATASET_PATH)
        else:
            print(f"Dataset not found: {DATASET_PATH}")
        return

    # ── Full validation ──
    print("Validating graph family files...")
    all_ok = True

    family_files = sorted(FAMILIES_DIR.glob("*.json")) if FAMILIES_DIR.exists() else []
    for fp in family_files:
        errs = validate_family_file(fp)
        status = "PASS" if not errs else "FAIL"
        print(f"  {status}: {fp.name}")
        for e in errs:
            print(f"    - {e}")
            all_ok = False

    if not family_files:
        print("  (no family files found)")

    print(f"\nValidating dataset: {DATASET_PATH.name}...")
    if DATASET_PATH.exists():
        total, valid, errs = validate_dataset(DATASET_PATH)
        print(f"  {valid}/{total} examples valid")
        for e in errs:
            print(f"    - {e}")
            all_ok = False
    else:
        print("  (dataset file not found)")
        all_ok = False

    print(f"\nValidating schema: {SCHEMA_PATH.name}...")
    if SCHEMA_PATH.exists():
        try:
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            print("  PASS: valid JSON")
        except json.JSONDecodeError as e:
            print(f"  FAIL: {e}")
            all_ok = False
    else:
        print("  (schema file not found)")

    print(f"\n{'='*40}")
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED")
    print(f"{'='*40}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
