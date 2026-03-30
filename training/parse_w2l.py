"""
parse_w2l.py
Parse Witcher 3 .w2l level layer files (binary CR2W v163) and extract
building layout data for training.

Usage:
  python parse_w2l.py <file_or_dir> [--out NAME] [--no-normalize]

  <file_or_dir>  Single .w2l file OR directory (scans recursively for .w2l)
  --out NAME     Output map name (default: inferred from filename)
  --no-normalize Keep raw world coordinates (default: normalise to centroid)

Output:
  training_data/w3_layouts/{map_name}.json
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUT_BASE   = SCRIPT_DIR / "training_data" / "w3_layouts"

CR2W_MAGIC   = b"CR2W"
CR2W_VERSION = 163          # Witcher 3

# ─── path-keyword helpers ────────────────────────────────────────────────────

BUILDING_KEYWORDS = [
    "house", "building", "tavern", "chapel", "castle", "mill", "barn",
    "inn", "stable", "blacksmith", "church", "tower", "wall", "gate",
    "warehouse", "storage", "bakery", "forge", "windmill",
    "hut", "cottage", "manor", "hall", "keep",
    "dom", "karczma", "kaplica", "zamek", "mlyn", "stodola", "kuznia",
]
EXCLUDE_KEYWORDS = [
    "npc", "monster", "animal", "vegetation", "tree", "bush", "grass",
    "rock", "stone_", "particle", "vfx", "fx_", "sound", "trigger",
    "player", "camera", "light", "torch", "candle", "barrel",
    "crate", "loot", "item", "weapon", "armor",
    "waypoint", "spawnpoint", "fence", "rope", "beam", "sticker",
]
TYPE_MAP = {
    "castle": "castle", "zamek": "castle", "tower": "castle", "keep": "castle",
    "chapel": "church", "kaplica": "church", "church": "church",
    "tavern": "commercial", "karczma": "commercial", "inn": "commercial",
    "bakery": "commercial", "blacksmith": "commercial", "forge": "commercial",
    "kuznia": "commercial",
    "mill": "industrial", "mlyn": "industrial", "windmill": "industrial",
    "barn": "industrial", "stodola": "industrial", "stable": "industrial",
    "warehouse": "industrial", "storage": "industrial",
    "house": "house", "dom": "house", "hut": "house",
    "cottage": "house", "building": "house", "manor": "house",
    "wall": "fortification", "gate": "fortification",
}


def _classify(path: str) -> str:
    p = path.lower()
    for kw, t in TYPE_MAP.items():
        if kw in p:
            return t
    return "building"


def _is_building(path: str) -> bool:
    p = path.lower()
    if any(ex in p for ex in EXCLUDE_KEYWORDS):
        return False
    return any(kw in p for kw in BUILDING_KEYWORDS)


# ─── Binary CR2W v163 parser ─────────────────────────────────────────────────

def _read_cr2w_names(data: bytes) -> list[str]:
    """Return the names table from a CR2W binary blob."""
    if data[:4] != CR2W_MAGIC:
        return []
    version, = struct.unpack_from("<I", data, 4)
    if version != CR2W_VERSION:
        return []

    # Header layout (40 bytes total):
    #   0x00  magic  4B
    #   0x04  ver    4B
    #   0x08  flags  4B
    #   0x0C  ts     8B
    #   0x14  bver   4B
    #   0x18  fsize  4B
    #   0x1C  bsize  4B
    #   0x20  crc    4B
    #   0x24  nchk   4B
    # Then 10 table descriptors @ 0x28, each 12 B: {offset, count, crc}

    str_off, str_cnt, _ = struct.unpack_from("<III", data, 0x28)
    nam_off, nam_cnt, _ = struct.unpack_from("<III", data, 0x28 + 12)

    strings_blob = data[str_off:nam_off]
    strtable: dict[int, str] = {}
    i = 0
    while i < len(strings_blob):
        end = strings_blob.find(b"\x00", i)
        if end < 0:
            end = len(strings_blob)
        strtable[i] = strings_blob[i:end].decode("utf-8", errors="replace")
        i = end + 1

    names: list[str] = []
    for n in range(nam_cnt):
        s_off, _ = struct.unpack_from("<II", data, nam_off + n * 8)
        names.append(strtable.get(s_off, ""))
    return names


def _extract_transforms(data: bytes, names: list[str]) -> list[dict]:
    """
    Scan for EngineTransform property blobs and extract positions + yaw.

    Property header (8 bytes):  nameIdx(u16) typeIdx(u16) size(u32)
    Then 1 flags byte followed by floats:
      flags=1 (001) → 3 floats: X, Y, Z                  (position only)
      flags=3 (011) → 7 floats: X Y Z pitchX rollY yawZ 0  (pos + rotation)
      flags=7 (111) → 10 floats: X Y Z pitchX rollY yawZ scaleX scaleY scaleZ 0
    In W3 coordinate system: X=east, Y=north, Z=height(up).
    Euler order: floats[3]=pitchX, floats[4]=rollY, floats[5]=yawZ (degrees).
    """
    t_idx  = next((i for i, n in enumerate(names) if n == "transform"),       None)
    et_idx = next((i for i, n in enumerate(names) if n == "EngineTransform"), None)
    if t_idx is None or et_idx is None:
        return []

    pattern = struct.pack("<HH", t_idx, et_idx)
    transforms: list[dict] = []

    for m in re.finditer(re.escape(pattern), data):
        off = m.start()
        try:
            size, = struct.unpack_from("<I", data, off + 4)
            flags = data[off + 8]
            # Minimum: 1 byte flags + 3 floats = 13 bytes inside the property
            if size < 13:
                continue
            x, y, z = struct.unpack_from("<fff", data, off + 9)
            # Sanity check: world coords in W3 are < ±10 km
            if not (-10_000 < x < 10_000 and -10_000 < y < 10_000 and -50 < z < 500):
                continue
            yaw = 0.0
            # flags bit 1 = hasRotation; rotation stored as pitchX, rollY, yawZ degrees
            # yawZ is float index 5 (offset +29 from property start)
            if (flags & 0x02) and size >= 29:
                yaw = struct.unpack_from("<f", data, off + 29)[0]
            transforms.append({"x": x, "y": y, "z": z, "yaw_deg": yaw})
        except (struct.error, IndexError):
            pass

    return transforms


def _nearby_resource_path(data: bytes, offset: int, window: int = 512) -> str:
    """
    Look for a depot-path string (contains backslash + known extension)
    within ±window bytes of offset.
    """
    lo = max(0, offset - window)
    hi = min(len(data), offset + window)
    chunk = data[lo:hi]
    # Look for .w2ent or .w2mesh paths
    for ext in (b".w2ent", b".w2mesh"):
        for m in re.finditer(re.escape(ext), chunk):
            end = m.end()
            start = m.start()
            # scan back for printable ASCII chars (path chars)
            s = start
            while s > 0 and (0x20 <= chunk[s - 1] < 0x7F):
                s -= 1
            path = chunk[s:end].decode("latin-1", errors="replace")
            if len(path) > 4:
                return path
    return ""


def parse_cr2w_w2l(path: Path) -> list[dict]:
    """Parse a single .w2l binary file and return building entities."""
    data = path.read_bytes()
    # Auto-decompress zlib-wrapped CR2W files (magic 78 9C / 78 DA)
    if len(data) >= 2 and data[0] == 0x78 and data[1] in (0x9C, 0xDA, 0x01, 0x5E):
        import zlib
        try:
            data = zlib.decompress(data)
        except Exception:
            pass
    if data[:4] != CR2W_MAGIC:
        print(f"  [skip] {path.name}: not a CR2W file")
        return []
    version, = struct.unpack_from("<I", data, 4)
    if version != CR2W_VERSION:
        print(f"  [skip] {path.name}: CR2W v{version} (need v{CR2W_VERSION})")
        return []

    names = _read_cr2w_names(data)
    transforms = _extract_transforms(data, names)
    print(f"  {path.name}: {len(transforms)} transforms")

    results: list[dict] = []
    for t in transforms:
        off_approx = 0  # no per-entity offset yet; use global scan
        path_str = ""
        # If there are per-entity offsets, we'd use them here.
        # For now classify based on the .w2l filename itself.
        fname = path.stem.lower()
        btype = _classify(fname) if any(kw in fname for kw in BUILDING_KEYWORDS) else "building"

        results.append({
            "x":       round(t["x"], 3),
            "y":       round(t["y"], 3),
            "z":       round(t["z"], 3),
            "yaw_deg": round(t["yaw_deg"], 2),
            "type":    btype,
            "source":  path.name,
        })
    return results


def scan_directory(root: Path) -> list[dict]:
    """Recursively scan a directory for .w2l files and parse all."""
    all_entities: list[dict] = []
    for p in sorted(root.rglob("*.w2l")):
        entities = parse_cr2w_w2l(p)
        all_entities.extend(entities)
    return all_entities


# ─── normalise + save ─────────────────────────────────────────────────────────

def normalize_layout(buildings: list[dict]) -> dict:
    """
    Normalise coordinates to centroid origin in the XY horizontal plane.
    In W3: X=east, Y=north, Z=up.  We use X/Y for 2D layout.
    """
    if not buildings:
        return {"buildings": [], "meta": {}}

    xs = [b["x"] for b in buildings]
    ys = [b["y"] for b in buildings]

    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2

    out = []
    for b in buildings:
        nb = dict(b)
        nb["nx"] = round(b["x"] - cx, 2)
        nb["ny"] = round(b["y"] - cy, 2)
        out.append(nb)

    type_dist: dict[str, int] = {}
    for b in out:
        type_dist[b["type"]] = type_dist.get(b["type"], 0) + 1

    meta = {
        "center_x":    round(cx, 2),
        "center_y":    round(cy, 2),
        "span_x_m":    round(max(xs) - min(xs), 1),
        "span_y_m":    round(max(ys) - min(ys), 1),
        "n_buildings": len(out),
        "type_dist":   type_dist,
        "coord_system": {
            "origin": "centroid of building XY positions",
            "nx": "east(+) / west(-), metres",
            "ny": "north(+) / south(-), metres",
            "z":  "W3 world height (up), not normalised",
            "yaw_deg": "degrees, 0=east(+X), CCW",
        },
    }
    return {"buildings": out, "meta": meta}


# ─── entry point ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Parse W3 .w2l binary → layout JSON")
    ap.add_argument("input", help=".w2l file or directory")
    ap.add_argument("--out", default=None, help="Output map name")
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"Not found: {inp}")

    map_name = args.out
    if not map_name:
        stem = inp.name
        for sfx in (".w2l",):
            if stem.lower().endswith(sfx):
                stem = stem[: -len(sfx)]
        map_name = re.sub(r"[^\w\-]", "_", stem).strip("_") or "unknown"

    print("=" * 60)
    print(f"  parse_w2l.py  —  {map_name}")
    print("=" * 60)

    if inp.is_dir():
        buildings = scan_directory(inp)
    else:
        buildings = parse_cr2w_w2l(inp)

    if not buildings:
        print("No entities found.")
        return

    result = (normalize_layout(buildings) if not args.no_normalize
              else {"buildings": buildings, "meta": {"n_buildings": len(buildings)}})
    result["map_name"]    = map_name
    result["source_file"] = str(inp)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    out_path = OUT_BASE / f"{map_name}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = result["meta"]
    print(f"\n{'='*60}")
    print(f"  Result: {map_name}")
    print(f"{'='*60}")
    print(f"  Entities      : {meta['n_buildings']}")
    if not args.no_normalize:
        print(f"  Scene span    : {meta['span_x_m']:.0f}m × {meta['span_y_m']:.0f}m")
        print(f"  Types         : " +
              "  ".join(f"{k}:{v}" for k, v in
                        sorted(meta["type_dist"].items(), key=lambda kv: -kv[1])))
    print(f"  Output        : {out_path}")

    print(f"\n  First 12 entities:")
    print(f"  {'#':>3}  {'nx':>9}  {'ny':>9}  {'z':>6}  {'yaw':>6}  type")
    print("  " + "-" * 50)
    for i, b in enumerate(result["buildings"][:12], 1):
        print(f"  {i:>3}  {b.get('nx', b['x']):>9.1f}  "
              f"{b.get('ny', b['y']):>9.1f}  "
              f"{b['z']:>6.1f}  {b['yaw_deg']:>6.1f}  {b['type']}")


if __name__ == "__main__":
    main()
