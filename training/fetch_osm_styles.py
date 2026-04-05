"""
fetch_osm_styles.py
从 OpenStreetMap 获取 4 种建筑风格的真实数据，应用风格专属过滤条件。

Japanese  : Kyoto / Nara           — 只保留寺庙/神社/传统建筑
Desert    : Marrakech / Cairo      — 只保留石砌/泥砖/土坯建筑材料
Modern    : Singapore / Shinjuku   — 只保留 building:levels >= 5 高层建筑
Industrial: Manchester / Birmingham — 只保留 industrial/warehouse/factory 标签
"""

import json
import re
import time
import math
from pathlib import Path

import numpy as np

try:
    import osmnx as ox
except ImportError:
    raise SystemExit("请先安装依赖: pip install osmnx geopandas")

OUTPUT_DIR = Path(__file__).parent / "validation_data"

# ─── 屋顶形状映射 (同 medieval) ────────────────────────────────
ROOF_SHAPE_MAP = {
    "flat": 0, "yes": 0, "no": 0,
    "gabled": 1, "saltbox": 1, "gambrel": 1, "mansard": 2, "skillion": 1,
    "hipped": 2, "half-hipped": 2, "pyramidal": 2,
    "dome": 3, "cone": 3, "onion": 3, "round": 3,
}

# ─── 材质 → wall_color [r,g,b] ────────────────────────────────
MATERIAL_COLOR = {
    # 石材
    "stone":          [0.65, 0.60, 0.52],
    "limestone":      [0.88, 0.84, 0.70],
    "sandstone":      [0.80, 0.68, 0.48],
    "granite":        [0.55, 0.52, 0.50],
    # 泥砖 / 土坯
    "mud_brick":      [0.72, 0.54, 0.36],
    "mudbrick":       [0.72, 0.54, 0.36],
    "adobe":          [0.78, 0.60, 0.40],
    "clay":           [0.74, 0.56, 0.38],
    "earth":          [0.68, 0.52, 0.36],
    "rammed_earth":   [0.70, 0.54, 0.38],
    "pise":           [0.70, 0.54, 0.38],
    # 砖
    "brick":          [0.70, 0.38, 0.28],
    "red_brick":      [0.72, 0.30, 0.22],
    # 木材
    "wood":           [0.65, 0.48, 0.30],
    "timber":         [0.60, 0.44, 0.28],
    "timber_framing": [0.65, 0.48, 0.30],
    # 混凝土 / 玻璃幕墙
    "concrete":       [0.70, 0.70, 0.68],
    "glass":          [0.68, 0.78, 0.85],
    "steel":          [0.55, 0.58, 0.60],
    "metal":          [0.50, 0.52, 0.55],
    # 灰泥 / 涂抹
    "plaster":        [0.90, 0.88, 0.82],
    "render":         [0.85, 0.82, 0.75],
    # 建筑类型推断
    "yes":            [0.70, 0.68, 0.65],
}
DEFAULT_COLOR = [0.70, 0.68, 0.65]

# ─── 日式风格材质颜色 ───────────────────────────────────────────
JAPANESE_MATERIAL_COLOR = {
    "wood":    [0.65, 0.48, 0.30],
    "timber":  [0.62, 0.45, 0.28],
    "plaster": [0.92, 0.88, 0.80],
    "clay":    [0.78, 0.62, 0.42],
    "tile":    [0.35, 0.35, 0.38],  # 黑瓦
    "stone":   [0.62, 0.60, 0.56],
    "yes":     [0.72, 0.58, 0.40],
}

# ─── 沙漠风格材质颜色 ──────────────────────────────────────────
DESERT_MATERIAL_COLOR = {
    "stone":        [0.82, 0.72, 0.55],
    "limestone":    [0.88, 0.80, 0.62],
    "sandstone":    [0.85, 0.72, 0.50],
    "mud_brick":    [0.75, 0.58, 0.38],
    "mudbrick":     [0.75, 0.58, 0.38],
    "adobe":        [0.78, 0.62, 0.42],
    "clay":         [0.72, 0.58, 0.40],
    "earth":        [0.70, 0.55, 0.36],
    "rammed_earth": [0.72, 0.56, 0.38],
    "yes":          [0.78, 0.65, 0.48],
}


# ─── 通用辅助函数 ──────────────────────────────────────────────

def _str(val) -> str:
    if val is None:
        return ""
    try:
        if math.isnan(float(val)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def parse_height(val) -> float | None:
    if val is None:
        return None
    s = _str(val).lower().replace(",", ".")
    m = re.match(r"([\d.]+)\s*(m|ft|'|meters?|feet)?", s)
    if not m:
        return None
    v = float(m.group(1))
    unit = (m.group(2) or "m").strip("'")
    if unit in ("ft", "feet"):
        v *= 0.3048
    return v if v > 0 else None


def parse_levels(val) -> int | None:
    if val is None:
        return None
    try:
        return max(1, int(float(_str(val).split(";")[0].strip())))
    except (ValueError, AttributeError):
        return None


def _get_tags(row) -> dict:
    """将 GeoDataFrame 行转为 dict，NaN → None"""
    raw = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    tags = {}
    for k, v in raw.items():
        try:
            tags[k] = None if (isinstance(v, float) and math.isnan(v)) else v
        except TypeError:
            tags[k] = v
    return tags


def _get_wall_color(tags: dict, color_map: dict) -> list:
    for key in ("building:material", "building:facade:material", "building"):
        val = _str(tags.get(key)).lower()
        if val in color_map:
            return color_map[val]
    return color_map.get("yes", DEFAULT_COLOR)


def _get_roof(tags: dict, default_roof: int) -> int:
    rs = _str(tags.get("roof:shape")).lower()
    return ROOF_SHAPE_MAP.get(rs, default_roof)


def _height_from_tags(tags: dict, level_scale: float = 3.5):
    h_direct = parse_height(_str(tags.get("height")) or None)
    levels   = parse_levels(_str(tags.get("building:levels")) or None)
    if h_direct is not None:
        return min(h_direct, 80.0), 0.0
    if levels is not None:
        return float(levels) * level_scale, 0.0
    return None, None


# ════════════════════════════════════════════════════════════════
#  风格专属过滤 & 参数提取
# ════════════════════════════════════════════════════════════════

# ── 1. Japanese ───────────────────────────────────────────────

JAPANESE_BUILDING_TAGS = {
    "temple", "shrine", "pagoda", "traditional", "monastery",
    "cathedral", "chapel", "yes",  # 'yes' only if also historic
}
JAPANESE_HISTORIC_TAGS = {"temple", "shrine", "building", "castle", "monument"}
JAPANESE_AMENITY_TAGS  = {"place_of_worship"}


def japanese_filter(tags: dict) -> bool:
    building = _str(tags.get("building")).lower()
    historic = _str(tags.get("historic")).lower()
    amenity  = _str(tags.get("amenity")).lower()
    religion = _str(tags.get("religion")).lower()
    tourism  = _str(tags.get("tourism")).lower()

    is_temple_tag  = building in ("temple", "shrine", "pagoda", "monastery")
    is_historic    = historic in JAPANESE_HISTORIC_TAGS
    is_worship     = amenity in JAPANESE_AMENITY_TAGS and religion in ("buddhist", "shinto", "")
    is_traditional = building == "traditional"

    return is_temple_tag or is_historic or is_worship or is_traditional


def extract_japanese(tags: dict, city: str) -> dict | None:
    building = _str(tags.get("building")).lower()
    historic = _str(tags.get("historic")).lower()
    amenity  = _str(tags.get("amenity")).lower()

    h_max, h_min = _height_from_tags(tags, level_scale=3.5)
    if h_max is None:
        # 日本寺庙常常只有 1-2 层，无高度标签时默认 4m
        h_max, h_min = 4.0, 0.0
    if h_max < 2.0:
        return None

    levels = parse_levels(_str(tags.get("building:levels")) or None) or 1

    # 屋顶类型：日式默认翘角坡屋顶 (type=3)，若 roof:shape 指定则用映射
    roof_type = _get_roof(tags, default_roof=3)

    wall_color = _get_wall_color(tags, JAPANESE_MATERIAL_COLOR)

    # 寺庙/神社 → 大挑檐 + 多柱
    eave = 0.70 if building in ("temple", "shrine", "pagoda") else 0.45
    cols = 4 if building in ("temple", "pagoda") else (2 if building == "shrine" else 0)
    has_arch = 1 if amenity == "place_of_worship" else 0

    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name":   (_str(tags.get("name")) or "")[:80],
        "building_type": building or "temple",
        "style": "japanese",
        "height_range_min":  round(h_min, 2),
        "height_range_max":  round(min(h_max, 20.0), 2),
        "wall_thickness":    0.15,
        "floor_thickness":   0.25,
        "door_width":        0.90,
        "door_height":       2.00,
        "win_width":         0.80,
        "win_height":        1.20,
        "win_density":       0.45,
        "subdivision":       min(8, levels),
        "roof_type":         roof_type,
        "roof_pitch":        0.55,
        "wall_color":        [round(c, 3) for c in wall_color],
        "has_battlements":   0,
        "has_arch":          has_arch,
        "eave_overhang":     eave,
        "column_count":      cols,
        "window_shape":      0,
    }


# ── 2. Desert ─────────────────────────────────────────────────

DESERT_MATERIALS = {
    "stone", "limestone", "sandstone", "granite", "flint",
    "mud_brick", "mudbrick", "adobe", "clay", "earth",
    "rammed_earth", "pise", "pisé", "cob",
}


def desert_filter(tags: dict) -> bool:
    material = _str(tags.get("building:material")).lower()
    facade   = _str(tags.get("building:facade:material")).lower()
    building = _str(tags.get("building")).lower()
    historic = _str(tags.get("historic")).lower()

    has_material = material in DESERT_MATERIALS or facade in DESERT_MATERIALS
    is_traditional = building in ("traditional", "house", "residential") or historic != ""
    # 允许无材质标签的传统建筑（中东/北非普遍缺 OSM 材质标签）
    return has_material or (building not in ("", "yes") and is_traditional)


def extract_desert(tags: dict, city: str) -> dict | None:
    h_max, h_min = _height_from_tags(tags, level_scale=3.0)  # 低层建筑，层高3m
    if h_max is None:
        h_max, h_min = 3.5, 0.0
    if h_max < 2.0:
        return None

    levels   = parse_levels(_str(tags.get("building:levels")) or None) or 1
    material = _str(tags.get("building:material")).lower()
    historic = _str(tags.get("historic")).lower()
    amenity  = _str(tags.get("amenity")).lower()

    wall_color = _get_wall_color(tags, DESERT_MATERIAL_COLOR)
    roof_type  = _get_roof(tags, default_roof=0)  # 沙漠默认平顶

    # 有历史标签 → 拱形窗
    has_arch = 1 if (historic or amenity == "place_of_worship") else 0

    # 泥砖/土坯建筑通常比较厚重
    wall_thick = 0.80 if material in {"mud_brick", "mudbrick", "adobe", "clay", "earth"} else 0.55

    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name":   (_str(tags.get("name")) or "")[:80],
        "building_type": _str(tags.get("building")).lower() or "yes",
        "style": "desert",
        "height_range_min":  round(h_min, 2),
        "height_range_max":  round(min(h_max, 20.0), 2),
        "wall_thickness":    wall_thick,
        "floor_thickness":   0.18,
        "door_width":        0.85,
        "door_height":       2.10,
        "win_width":         0.35,
        "win_height":        0.50,
        "win_density":       0.12,
        "subdivision":       min(8, levels),
        "roof_type":         roof_type,
        "roof_pitch":        0.05,
        "wall_color":        [round(c, 3) for c in wall_color],
        "has_battlements":   0,
        "has_arch":          has_arch,
        "eave_overhang":     0.05,
        "column_count":      0,
        "window_shape":      4 if has_arch else 0,
    }


# ── 3. Modern ─────────────────────────────────────────────────

MODERN_MATERIAL_COLOR = {
    "concrete": [0.72, 0.72, 0.68],
    "glass":    [0.65, 0.78, 0.88],
    "steel":    [0.60, 0.62, 0.65],
    "aluminum": [0.72, 0.74, 0.76],
    "metal":    [0.60, 0.62, 0.65],
    "curtain_wall": [0.65, 0.78, 0.88],
    "brick":    [0.72, 0.55, 0.42],
    "yes":      [0.72, 0.72, 0.68],
}


def modern_filter(tags: dict) -> bool:
    levels = parse_levels(_str(tags.get("building:levels")) or None)
    return levels is not None and levels >= 5


def extract_modern(tags: dict, city: str) -> dict | None:
    levels = parse_levels(_str(tags.get("building:levels")) or None)
    if levels is None or levels < 5:
        return None

    h_max, h_min = _height_from_tags(tags, level_scale=3.5)
    if h_max is None:
        h_max = float(levels) * 3.5
        h_min = 0.0
    if h_max < 15.0:
        h_max = float(levels) * 3.5

    material = _str(tags.get("building:material")).lower()
    wall_color = _get_wall_color(tags, MODERN_MATERIAL_COLOR)

    # 高层建筑：大窗
    win_density = 0.70 if levels >= 10 else 0.55
    win_w = 2.0 if "glass" in material or "curtain" in material else 1.60
    wall_thick = 0.20 if "glass" in material else 0.25

    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name":   (_str(tags.get("name")) or "")[:80],
        "building_type": _str(tags.get("building")).lower() or "yes",
        "style": "modern",
        "height_range_min":  round(h_min, 2),
        "height_range_max":  round(min(h_max, 80.0), 2),
        "wall_thickness":    wall_thick,
        "floor_thickness":   0.20,
        "door_width":        1.20,
        "door_height":       2.40,
        "win_width":         win_w,
        "win_height":        2.00,
        "win_density":       win_density,
        "subdivision":       min(8, levels),
        "roof_type":         0,     # 平屋顶
        "roof_pitch":        0.0,
        "wall_color":        [round(c, 3) for c in wall_color],
        "has_battlements":   0,
        "has_arch":          0,
        "eave_overhang":     0.0,
        "column_count":      0,
        "window_shape":      0,
    }


# ── 4. Industrial ─────────────────────────────────────────────

INDUSTRIAL_TAGS = {
    "industrial", "warehouse", "factory", "manufacture",
    "storage_tank", "hangar", "shed", "garages",
    "service", "retail",   # 偶尔工业区出现
}

INDUSTRIAL_MATERIAL_COLOR = {
    "brick":    [0.58, 0.42, 0.32],
    "red_brick":[0.62, 0.35, 0.25],
    "concrete": [0.60, 0.60, 0.58],
    "steel":    [0.52, 0.55, 0.58],
    "metal":    [0.48, 0.50, 0.52],
    "wood":     [0.55, 0.48, 0.38],
    "yes":      [0.55, 0.50, 0.45],
}


def industrial_filter(tags: dict) -> bool:
    building = _str(tags.get("building")).lower()
    landuse  = _str(tags.get("landuse")).lower()
    return (building in INDUSTRIAL_TAGS or
            landuse in ("industrial", "warehouse"))


def extract_industrial(tags: dict, city: str) -> dict | None:
    h_max, h_min = _height_from_tags(tags, level_scale=6.0)  # 工业层高高
    if h_max is None:
        h_max, h_min = 8.0, 0.0
    if h_max < 3.0:
        return None

    levels = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h_max / 6))
    building = _str(tags.get("building")).lower()
    wall_color = _get_wall_color(tags, INDUSTRIAL_MATERIAL_COLOR)
    roof_type  = _get_roof(tags, default_roof=1)  # 工业默认坡屋顶/锯齿

    # 大型工业建筑 → 宽门
    is_warehouse = building in ("warehouse", "storage_tank", "hangar")
    door_w = 3.0 if is_warehouse else 2.0
    door_h = 4.5 if is_warehouse else 4.0
    win_d  = 0.25 if building == "factory" else 0.35

    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name":   (_str(tags.get("name")) or "")[:80],
        "building_type": building or "industrial",
        "style": "industrial",
        "height_range_min":  round(h_min, 2),
        "height_range_max":  round(min(h_max, 50.0), 2),
        "wall_thickness":    0.40,
        "floor_thickness":   0.25,
        "door_width":        door_w,
        "door_height":       door_h,
        "win_width":         1.20,
        "win_height":        0.80,
        "win_density":       win_d,
        "subdivision":       min(8, levels),
        "roof_type":         roof_type,
        "roof_pitch":        0.30,
        "wall_color":        [round(c, 3) for c in wall_color],
        "has_battlements":   0,
        "has_arch":          0,
        "eave_overhang":     0.10,
        "column_count":      0,
        "window_shape":      0,
    }


# ════════════════════════════════════════════════════════════════
#  城市抓取
# ════════════════════════════════════════════════════════════════

STYLE_CITY_CONFIGS = {
    "japanese": {
        "cities": [
            {"name": "Kyoto", "query": "Kyoto, Japan"},
            {"name": "Nara",  "query": "Nara, Japan"},
        ],
        "filter_fn":  japanese_filter,
        "extract_fn": extract_japanese,
    },
    "desert": {
        "cities": [
            {"name": "Marrakech", "query": "Marrakech, Morocco"},
            {"name": "Cairo",     "query": "Cairo, Egypt"},
        ],
        "filter_fn":  desert_filter,
        "extract_fn": extract_desert,
    },
    "modern": {
        "cities": [
            {"name": "Singapore", "query": "Singapore"},
            {"name": "Shinjuku",  "query": "Shinjuku, Tokyo, Japan"},
        ],
        "filter_fn":  modern_filter,
        "extract_fn": extract_modern,
    },
    "industrial": {
        "cities": [
            {"name": "Manchester",  "query": "Manchester, England"},
            {"name": "Birmingham",  "query": "Birmingham, England"},
        ],
        "filter_fn":  industrial_filter,
        "extract_fn": extract_industrial,
    },
}


def fetch_city(city_cfg: dict, filter_fn, extract_fn) -> list[dict]:
    name  = city_cfg["name"]
    query = city_cfg["query"]
    print(f"    [{name}] 正在查询 OSM...")
    try:
        gdf = ox.features_from_place(query, tags={"building": True})
    except Exception as e:
        print(f"      [失败] {e}")
        return []

    print(f"      原始建筑: {len(gdf)} 条")
    results, skipped = [], 0
    for idx, row in gdf.iterrows():
        row_copy = row.copy()
        row_copy["_city"] = name
        tags = _get_tags(row_copy)
        tags["osmid"] = idx[1] if isinstance(idx, tuple) else idx
        if not filter_fn(tags):
            skipped += 1
            continue
        rec = extract_fn(tags, name)
        if rec is None:
            skipped += 1
            continue
        results.append(rec)

    print(f"      有效: {len(results)}  跳过: {skipped}")
    return results


# ─── 主程序 ────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  fetch_osm_styles.py  —  4 风格 OSM 建筑数据抓取")
    print("=" * 65)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    for style, cfg in STYLE_CITY_CONFIGS.items():
        print(f"\n{'─'*65}")
        print(f"  风格: {style.upper()}")
        all_records = []

        for city_cfg in cfg["cities"]:
            records = fetch_city(city_cfg, cfg["filter_fn"], cfg["extract_fn"])
            all_records.extend(records)
            time.sleep(1.5)   # 礼貌延迟

        # 统计
        by_city = {}
        for r in all_records:
            by_city.setdefault(r["city"], 0)
            by_city[r["city"]] += 1

        print(f"\n  [{style}] 总有效建筑: {len(all_records)} 条")
        for city, cnt in by_city.items():
            print(f"    {city:<20}: {cnt:4d} 条")

        if all_records:
            heights = [r["height_range_max"] for r in all_records]
            print(f"  高度: min={min(heights):.1f}m  "
                  f"median={sorted(heights)[len(heights)//2]:.1f}m  "
                  f"max={max(heights):.1f}m")

        # 保存
        out_path = OUTPUT_DIR / f"{style}_osm.json"
        output = {
            "total": len(all_records),
            "cities": list(by_city.keys()),
            "buildings": all_records,
        }
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  已保存: {out_path.name}")
        summary[style] = len(all_records)

    # 汇总
    print(f"\n{'═'*65}")
    print("  抓取完成汇总")
    print(f"{'═'*65}")
    for style, cnt in summary.items():
        status = "✓" if cnt >= 100 else ("△" if cnt >= 20 else "✗")
        print(f"  {status}  {style:<15}: {cnt:5d} 条")
    print()


if __name__ == "__main__":
    main()
