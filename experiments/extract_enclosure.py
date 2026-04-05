"""
extract_enclosure.py
Batch-extract enclosure parameters from W3 Novigrad scene data.

For each scene (grouped by parent folder):
  - boundary_coverage: (360 - max_gap) / 360
  - opening_count: gaps > 45 deg
  - max_opening_width: largest angular gap
  - center_focus_strength: fraction of buildings outside center 30m radius

Output: training_data/enclosure_dataset.json
"""

import json
import math
import sys
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
W3_NOVIGRAD = Path("D:/W3Depot/levels/novigrad")
OUT_PATH = SCRIPT_DIR / "training_data" / "enclosure_dataset.json"

# Only parse files with these keywords (building-related)
BUILDING_FILE_KEYWORDS = [
    "exterior", "building", "static", "structure",
    "streets", "square", "houses",
]

# Skip files with these keywords (non-building content)
SKIP_KEYWORDS = [
    "rain", "fx", "plants", "puddle", "loot", "light", "sound",
    "trigger", "action", "dialog", "scene", "quest", "guard",
    "animal", "bird", "crowd", "encounter", "waypoint", "path",
    "decal", "stripe", "env_probe", "collision", "proxy",
    "navmesh", "door", "mappins",
]

CENTER_RADIUS = 30.0  # meters for focus strength calc
MIN_BUILDINGS = 5     # skip scenes with fewer buildings


def _is_building_file(filepath):
    name = filepath.lower()
    if any(sk in name for sk in SKIP_KEYWORDS):
        return False
    return any(bk in name for bk in BUILDING_FILE_KEYWORDS)


def _compute_enclosure(xs, ys):
    """Compute 4 enclosure parameters from building XY positions."""
    n = len(xs)
    if n < 3:
        return None

    cx, cy = np.mean(xs), np.mean(ys)

    # Angular distribution
    angles = sorted(math.degrees(math.atan2(y - cy, x - cx)) for x, y in zip(xs, ys))

    # Compute gaps between consecutive angles
    gaps = []
    for i in range(len(angles)):
        j = (i + 1) % len(angles)
        gap = angles[j] - angles[i]
        if gap < 0:
            gap += 360
        # Handle wrap-around for last->first
        if i == len(angles) - 1:
            gap = (angles[0] + 360) - angles[-1]
        gaps.append(gap)

    max_gap = max(gaps) if gaps else 360
    boundary_coverage = (360 - max_gap) / 360
    opening_count = sum(1 for g in gaps if g > 45)

    # Center focus strength: fraction of buildings OUTSIDE center radius
    dists = [math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)]
    inner = sum(1 for d in dists if d < CENTER_RADIUS)
    focus_strength = 1.0 - (inner / n)

    return {
        "boundary_coverage": round(boundary_coverage, 3),
        "opening_count": opening_count,
        "max_opening_width": round(max_gap, 1),
        "center_focus_strength": round(focus_strength, 3),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  extract_enclosure.py — W3 Novigrad enclosure analysis")
    print("=" * 60)

    sys.path.insert(0, str(SCRIPT_DIR))
    import parse_w2l

    # Find all .w2l files under novigrad levels
    all_files = list(W3_NOVIGRAD.rglob("*.w2l"))
    print(f"Total .w2l files: {len(all_files)}")

    # Filter to building-related files
    building_files = [f for f in all_files if _is_building_file(f.name)]
    print(f"Building-related: {len(building_files)}")

    # Group by scene (grandparent folder — the area name)
    # Path structure: levels/novigrad/levels/novigrad/<area>/<sub>/<file>.w2l
    scenes = defaultdict(list)
    for f in building_files:
        parts = f.relative_to(W3_NOVIGRAD).parts
        # Use the first meaningful directory as scene name
        # Skip 'levels/novigrad' prefix
        scene_parts = [p for p in parts[:-1] if p not in ("levels", "novigrad")]
        if scene_parts:
            scene_name = scene_parts[0]
        else:
            scene_name = "_root"
        scenes[scene_name].append(f)

    print(f"Scenes (folders): {len(scenes)}")

    # Process each scene
    results = []
    for scene_name in sorted(scenes):
        files = scenes[scene_name]

        # Parse all buildings in this scene
        all_buildings = []
        for f in files:
            try:
                entities = parse_w2l.parse_cr2w_w2l(f)
                all_buildings.extend(entities)
            except Exception:
                pass

        # Filter: remove dummy coords (0,0) (1,1)
        valid = [b for b in all_buildings
                 if abs(b["x"]) > 5 or abs(b["y"]) > 5]

        if len(valid) < MIN_BUILDINGS:
            continue

        xs = [b["x"] for b in valid]
        ys = [b["y"] for b in valid]

        # Filter outliers: remove points > 3 std from mean
        cx, cy = np.mean(xs), np.mean(ys)
        dists = [math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)]
        d_mean, d_std = np.mean(dists), np.std(dists)
        threshold = d_mean + 3 * d_std
        filtered = [(x, y) for x, y, d in zip(xs, ys, dists) if d < threshold]

        if len(filtered) < MIN_BUILDINGS:
            continue

        xs_f = [p[0] for p in filtered]
        ys_f = [p[1] for p in filtered]

        enc = _compute_enclosure(xs_f, ys_f)
        if enc is None:
            continue

        is_enclosed = enc["boundary_coverage"] > 0.6 and enc["opening_count"] <= 2

        record = {
            "scene": scene_name,
            "building_count": len(filtered),
            "center": [round(np.mean(xs_f), 1), round(np.mean(ys_f), 1)],
            "span_x": round(max(xs_f) - min(xs_f), 1),
            "span_y": round(max(ys_f) - min(ys_f), 1),
            "is_enclosed": is_enclosed,
            **enc,
        }
        results.append(record)

    # Print results
    print(f"\n{'='*60}")
    print(f"  Results: {len(results)} scenes analyzed")
    print(f"{'='*60}")

    n_enclosed = sum(1 for r in results if r["is_enclosed"])
    print(f"  is_enclosed=True: {n_enclosed}/{len(results)} "
          f"({n_enclosed/max(1,len(results))*100:.0f}%)")

    coverages = [r["boundary_coverage"] for r in results]
    openings = [r["opening_count"] for r in results]
    max_gaps = [r["max_opening_width"] for r in results]
    focuses = [r["center_focus_strength"] for r in results]

    print(f"\n  boundary_coverage:  min={min(coverages):.2f}  avg={np.mean(coverages):.2f}  max={max(coverages):.2f}")
    print(f"  opening_count:      min={min(openings)}  avg={np.mean(openings):.1f}  max={max(openings)}")
    print(f"  max_opening_width:  min={min(max_gaps):.0f}  avg={np.mean(max_gaps):.0f}  max={max(max_gaps):.0f}")
    print(f"  center_focus:       min={min(focuses):.2f}  avg={np.mean(focuses):.2f}  max={max(focuses):.2f}")

    # Print top 10 most enclosed scenes
    print(f"\n  Top 10 enclosed scenes:")
    for r in sorted(results, key=lambda x: -x["boundary_coverage"])[:10]:
        print(f"    {r['scene']:30s}  cov={r['boundary_coverage']:.2f}  "
              f"gaps={r['opening_count']}  bld={r['building_count']:>4}  "
              f"{'ENCLOSED' if r['is_enclosed'] else ''}")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Output: {OUT_PATH.name} ({OUT_PATH.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
