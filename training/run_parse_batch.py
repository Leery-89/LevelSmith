"""
run_parse_batch.py
Step 1: List D:/W3Depot/levels structure
Step 2: Test parse 5 .w2l files from prolog_village
Step 3: Batch parse village + living_world + ard_skellig + novigrad living_world
        Filter: only keep buildings with yaw_deg != 0
Save results to training_data/w3_layouts/
"""
import os, sys, json, struct, re, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PARSE_SCRIPT = SCRIPT_DIR / "parse_w2l.py"
W3_LEVELS = Path("D:/W3Depot/levels")
OUT_BASE = SCRIPT_DIR / "training_data" / "w3_layouts"

# ── Step 1: list structure ────────────────────────────────────────────
def step1_list():
    print("\n" + "="*60)
    print("STEP 1 — D:/W3Depot/levels/ 目录结构")
    print("="*60)
    if not W3_LEVELS.exists():
        print("  ERROR: 目录不存在！")
        return
    for lvl_dir in sorted(W3_LEVELS.iterdir()):
        if not lvl_dir.is_dir():
            continue
        total = sum(1 for _ in lvl_dir.rglob("*.w2l"))
        print(f"\n  {lvl_dir.name}/  →  {total} .w2l files")
        # sub-areas
        area_root = lvl_dir / "levels" / lvl_dir.name
        if area_root.is_dir():
            subs = sorted([d for d in area_root.iterdir() if d.is_dir()])
            for s in subs[:20]:
                c = sum(1 for _ in s.rglob("*.w2l"))
                print(f"      {s.name}/  {c}")
            if len(subs) > 20:
                print(f"      ... and {len(subs)-20} more subdirs")

# ── CR2W binary parser (from parse_w2l.py) ──────────────────────────
CR2W_MAGIC   = b"CR2W"
CR2W_VERSION = 163

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

def _classify(path):
    p = path.lower()
    for kw, t in TYPE_MAP.items():
        if kw in p:
            return t
    return "building"

def _is_building(path):
    p = path.lower()
    if any(ex in p for ex in EXCLUDE_KEYWORDS):
        return False
    return any(kw in p for kw in BUILDING_KEYWORDS)

def _read_cr2w_names(data):
    if data[:4] != CR2W_MAGIC:
        return []
    version, = struct.unpack_from("<I", data, 4)
    if version != CR2W_VERSION:
        return []
    str_off, str_cnt, _ = struct.unpack_from("<III", data, 0x28)
    nam_off, nam_cnt, _ = struct.unpack_from("<III", data, 0x28 + 12)
    strings_blob = data[str_off:nam_off]
    strtable = {}
    i = 0
    while i < len(strings_blob):
        end = strings_blob.find(b"\x00", i)
        if end < 0:
            end = len(strings_blob)
        strtable[i] = strings_blob[i:end].decode("utf-8", errors="replace")
        i = end + 1
    names = []
    for n in range(nam_cnt):
        s_off, _ = struct.unpack_from("<II", data, nam_off + n * 8)
        names.append(strtable.get(s_off, ""))
    return names

def _extract_transforms(data, names):
    t_idx  = next((i for i, n in enumerate(names) if n == "transform"),       None)
    et_idx = next((i for i, n in enumerate(names) if n == "EngineTransform"), None)
    if t_idx is None or et_idx is None:
        return []
    pattern = struct.pack("<HH", t_idx, et_idx)
    transforms = []
    for m in re.finditer(re.escape(pattern), data):
        off = m.start()
        try:
            size, = struct.unpack_from("<I", data, off + 4)
            flags = data[off + 8]
            if size < 13:
                continue
            x, y, z = struct.unpack_from("<fff", data, off + 9)
            if not (-10_000 < x < 10_000 and -10_000 < y < 10_000 and -50 < z < 500):
                continue
            yaw = 0.0
            if flags >= 7 and size >= 29:
                _, yaw, _ = struct.unpack_from("<fff", data, off + 21)
            transforms.append({"x": x, "y": y, "z": z, "yaw_deg": yaw})
        except (struct.error, IndexError):
            pass
    return transforms

def parse_w2l_file(path):
    data = path.read_bytes()
    if data[:4] != CR2W_MAGIC:
        return []
    version, = struct.unpack_from("<I", data, 4)
    if version != CR2W_VERSION:
        return []
    names = _read_cr2w_names(data)
    transforms = _extract_transforms(data, names)
    results = []
    fname = path.stem.lower()
    btype = _classify(fname) if any(kw in fname for kw in BUILDING_KEYWORDS) else "building"
    for t in transforms:
        results.append({
            "x":       round(t["x"], 3),
            "y":       round(t["y"], 3),
            "z":       round(t["z"], 3),
            "yaw_deg": round(t["yaw_deg"], 2),
            "type":    btype,
            "source":  path.name,
        })
    return results

# ── Step 2: test parse 5 files ────────────────────────────────────────
def step2_test():
    print("\n" + "="*60)
    print("STEP 2 — 测试解析 prolog_village 前5个 .w2l 文件")
    print("="*60)
    pv = W3_LEVELS / "prolog_village"
    if not pv.exists():
        print("  ERROR: prolog_village 目录不存在")
        return
    files = sorted(pv.rglob("*.w2l"))[:5]
    for f in files:
        entities = parse_w2l_file(f)
        rel = f.relative_to(pv)
        print(f"\n  [{rel}]  {len(entities)} transforms")
        for e in entities[:5]:
            print(f"    x={e['x']:9.2f}  y={e['y']:9.2f}  z={e['z']:6.2f}  "
                  f"yaw={e['yaw_deg']:7.2f}  type={e['type']}")
        if len(entities) > 5:
            nonzero = sum(1 for e in entities if abs(e['yaw_deg']) > 0.01)
            print(f"    ... ({len(entities)} total, {nonzero} with yaw!=0)")

# ── Step 3: batch parse & filter yaw!=0 ───────────────────────────────
TARGETS = [
    # (label, path_relative_to_map_levels, map)
    ("prolog_village__village",     "prolog_village/village",     "prolog_village"),
    ("prolog_village__living_world","prolog_village/living_world", "prolog_village"),
    ("skellige__ard_skellig",       "skellige/ard_skellig",        "skellige"),
    ("novigrad__living_world",      "novigrad/living_world",       "novigrad"),
]

def normalize(buildings):
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
    type_dist = {}
    for b in out:
        type_dist[b["type"]] = type_dist.get(b["type"], 0) + 1
    return {
        "buildings": out,
        "meta": {
            "center_x": round(cx, 2),
            "center_y": round(cy, 2),
            "span_x_m": round(max(xs)-min(xs), 1),
            "span_y_m": round(max(ys)-min(ys), 1),
            "n_buildings": len(out),
            "type_dist": type_dist,
        }
    }

def step3_batch():
    print("\n" + "="*60)
    print("STEP 3 — 批量解析 + 过滤 yaw_deg != 0")
    print("="*60)
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    total_all = 0
    summary = []

    for label, rel_path, map_name in TARGETS:
        scan_dir = W3_LEVELS / map_name / "levels" / rel_path
        if not scan_dir.exists():
            # try without levels/ prefix
            scan_dir = W3_LEVELS / map_name / rel_path
        if not scan_dir.exists():
            print(f"\n  [SKIP] {label}: 目录不存在 ({scan_dir})")
            continue

        print(f"\n  解析: {label}")
        print(f"  路径: {scan_dir}")

        all_entities = []
        w2l_count = 0
        for w2l_path in sorted(scan_dir.rglob("*.w2l")):
            entities = parse_w2l_file(w2l_path)
            all_entities.extend(entities)
            w2l_count += 1

        # Filter: yaw_deg != 0
        filtered = [e for e in all_entities if abs(e["yaw_deg"]) > 0.01]

        print(f"  .w2l files: {w2l_count}")
        print(f"  Total transforms: {len(all_entities)}")
        print(f"  After yaw!=0 filter: {len(filtered)}")

        if filtered:
            result = normalize(filtered)
            result["map_name"] = label
            result["source_dir"] = str(scan_dir)
            out_path = OUT_BASE / (label + ".json")
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Saved: {out_path}")
            total_all += len(filtered)
            summary.append((label, len(filtered), str(out_path)))
        else:
            print(f"  WARNING: 无有效建筑数据")

    print("\n" + "="*60)
    print(f"汇总 — 总计 yaw!=0 建筑条目: {total_all}")
    print("="*60)
    for lbl, cnt, path in summary:
        print(f"  {lbl}: {cnt} 条  -> {path}")
    if total_all >= 3000:
        print(f"\n  目标达成！{total_all} >= 3000")
    else:
        print(f"\n  WARNING: {total_all} < 3000，需要扩大扫描范围")
    return total_all, summary


if __name__ == "__main__":
    step1_list()
    step2_test()
    total, summary = step3_batch()
