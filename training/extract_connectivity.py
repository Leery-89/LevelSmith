"""
extract_connectivity.py
Infer building connectivity from W3 layout data.

Connected (positive):  distance < 3m AND same_zone=1
Negative:              distance 3-15m (random sampled at 2x positive count)

Features per pair:
  - distance, distance_norm (/ 15m)
  - yaw_diff_norm (/ 180)
  - relative_angle (/ 360)
  - connected, same_zone, facing

Output: training_data/connectivity_dataset.json
"""

import json
import math
import random
import sys
import numpy as np
from pathlib import Path
from itertools import combinations

SCRIPT_DIR = Path(__file__).parent
RAW_DIR    = SCRIPT_DIR / "training_data" / "w3_layouts" / "raw"
OUT_PATH   = SCRIPT_DIR / "training_data" / "connectivity_dataset.json"

CONNECTED_DIST = 3.0     # positive: distance < 3m AND same_zone
NEG_DIST_MAX   = 15.0    # negative: 3-15m
FACING_YAW_THRESH = 30.0
NEG_RATIO = 2            # negative samples = 2x positive
MAX_PAIRS_PER_SCENE = 8000


def _yaw_diff(a_deg, b_deg):
    """Shortest angular difference in degrees, [0, 180]."""
    d = abs(a_deg - b_deg) % 360
    return d if d <= 180 else 360 - d


def _relative_angle(a, b):
    """Angle from A to B in degrees [0, 360)."""
    dx = b["nx"] - a["nx"]
    dy = b["ny"] - a["ny"]
    ang = math.degrees(math.atan2(dy, dx)) % 360
    return ang


def _is_facing(a, b):
    dx = b["nx"] - a["nx"]
    dy = b["ny"] - a["ny"]
    dist = math.hypot(dx, dy)
    if dist < 0.1:
        return False
    angle_a_to_b = math.degrees(math.atan2(dy, dx))
    diff_a = _yaw_diff(a["yaw_deg"], angle_a_to_b)
    diff_b = _yaw_diff(b["yaw_deg"], angle_a_to_b + 180)
    return diff_a < FACING_YAW_THRESH and diff_b < FACING_YAW_THRESH


def _make_record(a, b, dist, connected, same_zone):
    facing = 1 if connected and _is_facing(a, b) else 0
    return {
        "building_a": {"nx": a["nx"], "ny": a["ny"], "yaw_deg": a["yaw_deg"]},
        "building_b": {"nx": b["nx"], "ny": b["ny"], "yaw_deg": b["yaw_deg"]},
        "distance":       round(dist, 3),
        "distance_norm":  round(dist / NEG_DIST_MAX, 4),
        "yaw_diff_norm":  round(_yaw_diff(a["yaw_deg"], b["yaw_deg"]) / 180.0, 4),
        "relative_angle": round(_relative_angle(a, b) / 360.0, 4),
        "connected":      connected,
        "same_zone":      same_zone,
        "facing":         facing,
    }


def process_scene(scene_path: Path) -> tuple:
    """Return (positives, negatives) lists."""
    data = json.loads(scene_path.read_text("utf-8"))
    buildings = data.get("buildings", [])
    if len(buildings) < 2:
        return [], []

    n = len(buildings)

    # Build pair indices (windowed for large scenes)
    if n * (n - 1) // 2 <= MAX_PAIRS_PER_SCENE:
        pair_indices = list(combinations(range(n), 2))
    else:
        sorted_idx = sorted(range(n), key=lambda i: buildings[i]["nx"])
        pair_indices = set()
        window = 40
        for pos, i in enumerate(sorted_idx):
            for j_pos in range(max(0, pos - window), min(n, pos + window)):
                j = sorted_idx[j_pos]
                if i < j:
                    pair_indices.add((i, j))
        pair_indices = list(pair_indices)

    positives = []
    negatives = []

    for i, j in pair_indices:
        a, b = buildings[i], buildings[j]
        dx = a["nx"] - b["nx"]
        dy = a["ny"] - b["ny"]
        dist = math.hypot(dx, dy)
        same_zone = 1 if a.get("source") == b.get("source") else 0

        if dist < CONNECTED_DIST and same_zone:
            positives.append(_make_record(a, b, dist, 1, same_zone))
        elif CONNECTED_DIST <= dist <= NEG_DIST_MAX:
            negatives.append(_make_record(a, b, dist, 0, same_zone))

    return positives, negatives


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  extract_connectivity.py (v2)")
    print("=" * 60)

    if not RAW_DIR.exists():
        sys.exit(f"Raw data dir not found: {RAW_DIR}")

    scene_files = sorted(RAW_DIR.glob("*.json"))
    print(f"Scene files: {len(scene_files)}")

    all_pos = []
    all_neg = []

    for sf in scene_files:
        pos, neg = process_scene(sf)
        print(f"  {sf.name}: pos={len(pos):,}  neg={len(neg):,}")
        all_pos.extend(pos)
        all_neg.extend(neg)

    print(f"\n  Raw: positives={len(all_pos):,}  negatives={len(all_neg):,}")

    # Downsample negatives to NEG_RATIO * positives
    target_neg = len(all_pos) * NEG_RATIO
    if len(all_neg) > target_neg:
        random.seed(42)
        all_neg = random.sample(all_neg, target_neg)
        print(f"  Downsampled negatives: {len(all_neg):,} (ratio 1:{NEG_RATIO})")

    dataset = all_pos + all_neg
    random.seed(42)
    random.shuffle(dataset)

    # Stats
    total = len(dataset)
    n_conn = sum(1 for p in dataset if p["connected"])
    n_disc = total - n_conn
    n_sz   = sum(1 for p in dataset if p["same_zone"])
    n_face = sum(1 for p in dataset if p["facing"])

    dists = np.array([p["distance"] for p in dataset])

    print(f"\n{'='*60}")
    print(f"  Final Dataset")
    print(f"{'='*60}")
    print(f"  Total samples:  {total:,}")
    print(f"  connected=1:    {n_conn:,}  ({n_conn/total*100:.1f}%)")
    print(f"  connected=0:    {n_disc:,}  ({n_disc/total*100:.1f}%)")
    print(f"  same_zone=1:    {n_sz:,}  ({n_sz/total*100:.1f}%)")
    print(f"  facing=1:       {n_face:,}  ({n_face/total*100:.1f}%)")
    print(f"\n  Distance distribution:")
    print(f"    <3m:   {(dists<3).sum():,}")
    print(f"    3-8m:  {((dists>=3)&(dists<8)).sum():,}")
    print(f"    8-15m: {((dists>=8)&(dists<=15)).sum():,}")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(dataset), encoding="utf-8")
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n  Output: {OUT_PATH.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
