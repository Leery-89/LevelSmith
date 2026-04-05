"""
fetch_osm_styles2.py
从 OpenStreetMap 获取 4 种新建筑风格的真实数据，应用风格专属过滤条件。

Fantasy        : Prague, Czech Republic / Edinburgh, Scotland
                 — building=castle/cathedral/church/historic
Horror         : Sighisoara, Romania / New Orleans, USA
                 — building=church/historic，condition=disused/abandoned
Medieval_Chapel: Chartres, France / Canterbury, England
                 — building=chapel/church
Medieval_Keep  : Windsor, England / Carcassonne, France
                 — building=castle/tower/fort
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

# ─── 屋顶形状映射 ────────────────────────────────────────────────
ROOF_SHAPE_MAP = {
    "flat": 0, "yes": 0, "no": 0,
    "gabled": 1, "saltbox": 1, "gambrel": 1, "mansard": 2, "skillion": 1,
    "hipped": 2, "half-hipped": 2, "pyramidal": 2,
    "dome": 3, "cone": 3, "onion": 3, "round": 3,
}

# ─── 风格专属材质颜色 ─────────────────────────────────────────────
FANTASY_MATERIAL_COLOR = {
    "stone":      [0.65, 0.62, 0.58],
    "limestone":  [0.88, 0.84, 0.72],
    "sandstone":  [0.78, 0.68, 0.50],
    "granite":    [0.55, 0.52, 0.50],
    "brick":      [0.68, 0.52, 0.42],
    "wood":       [0.60, 0.48, 0.32],
    "plaster":    [0.90, 0.88, 0.82],
    "yes":        [0.68, 0.64, 0.58],
}

HORROR_MATERIAL_COLOR = {
    "stone":      [0.38, 0.35, 0.32],   # 暗沉、风化
    "brick":      [0.42, 0.30, 0.24],
    "wood":       [0.32, 0.28, 0.22],
    "plaster":    [0.45, 0.42, 0.38],
    "concrete":   [0.35, 0.33, 0.30],
    "yes":        [0.32, 0.29, 0.26],
}

CHAPEL_MATERIAL_COLOR = {
    "stone":      [0.75, 0.72, 0.65],
    "limestone":  [0.88, 0.85, 0.75],
    "sandstone":  [0.80, 0.72, 0.55],
    "granite":    [0.60, 0.58, 0.54],
    "flint":      [0.55, 0.55, 0.50],
    "brick":      [0.70, 0.52, 0.40],
    "yes":        [0.74, 0.70, 0.62],
}

KEEP_MATERIAL_COLOR = {
    "stone":      [0.52, 0.50, 0.46],   # 深灰石块
    "limestone":  [0.65, 0.62, 0.55],
    "granite":    [0.48, 0.46, 0.44],
    "flint":      [0.40, 0.40, 0.38],
    "brick":      [0.55, 0.40, 0.32],
    "yes":        [0.52, 0.48, 0.44],
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
    return color_map.get("yes", [0.65, 0.62, 0.58])


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

# ── 1. Fantasy ────────────────────────────────────────────────

FANTASY_BUILDING_TAGS = {
    "castle", "cathedral", "church", "chapel", "palace",
    "monastery", "manor", "tower",
}
FANTASY_HISTORIC_TAGS = {
    "castle", "church", "palace", "manor_house", "monastery",
    "building", "monument", "archaeological_site",
}


def fantasy_filter(tags: dict) -> bool:
    building = _str(tags.get("building")).lower()
    historic = _str(tags.get("historic")).lower()
    amenity  = _str(tags.get("amenity")).lower()
    tourism  = _str(tags.get("tourism")).lower()
    return (building in FANTASY_BUILDING_TAGS or
            historic in FANTASY_HISTORIC_TAGS or
            amenity == "place_of_worship" or
            tourism in ("attraction",) and building not in ("", "yes"))


def extract_fantasy(tags: dict, city: str) -> dict | None:
    h_max, h_min = _height_from_tags(tags, level_scale=4.5)
    if h_max is None:
        h_max, h_min = 12.0, 0.0
    if h_max < 3.0:
        return None

    building = _str(tags.get("building")).lower()
    historic = _str(tags.get("historic")).lower()
    levels   = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h_max / 4.5))

    wall_color = _get_wall_color(tags, FANTASY_MATERIAL_COLOR)
    roof_type  = _get_roof(tags, default_roof=2)   # 默认四坡顶

    is_castle  = building in ("castle", "tower") or historic in ("castle",)
    is_church  = building in ("cathedral", "church", "chapel", "monastery")

    has_battlements = 1 if is_castle else 0
    has_arch        = 1  # 所有奇幻建筑都有拱形元素
    window_shape    = 1  # 尖拱窗
    roof_pitch      = 0.80 if is_church else 0.65

    # 城堡有箭孔小窗；教堂有高细窗
    if is_castle:
        win_w, win_h, win_d = 0.55, 1.80, 0.28
        door_w, door_h = 1.60, 3.50
        wall_thick = 1.20
        eave = 0.15
        cols = 0
    else:
        win_w, win_h, win_d = 0.70, 2.50, 0.38
        door_w, door_h = 2.00, 4.00
        wall_thick = 0.80
        eave = 0.30
        cols = 4 if is_church else 2

    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name":   (_str(tags.get("name")) or "")[:80],
        "building_type": building or "castle",
        "style": "fantasy",
        "height_range_min":  round(h_min, 2),
        "height_range_max":  round(min(h_max, 40.0), 2),
        "wall_thickness":    wall_thick,
        "floor_thickness":   0.25,
        "door_width":        door_w,
        "door_height":       door_h,
        "win_width":         win_w,
        "win_height":        win_h,
        "win_density":       win_d,
        "subdivision":       min(8, levels),
        "roof_type":         roof_type,
        "roof_pitch":        roof_pitch,
        "wall_color":        [round(c, 3) for c in wall_color],
        "has_battlements":   has_battlements,
        "has_arch":          has_arch,
        "eave_overhang":     eave,
        "column_count":      cols,
        "window_shape":      window_shape,
    }


# ── 2. Horror ────────────────────────────────────────────────

HORROR_BUILDING_TAGS = {
    "church", "chapel", "cathedral", "monastery", "ruins",
    "historic", "yes",
}
HORROR_CONDITION_TAGS = {
    "disused", "abandoned", "ruins", "ruin", "derelict",
    "deteriorating", "bad", "very_bad",
}


def horror_filter(tags: dict) -> bool:
    building  = _str(tags.get("building")).lower()
    historic  = _str(tags.get("historic")).lower()
    condition = _str(tags.get("building:condition")).lower()
    ruins     = _str(tags.get("ruins")).lower()
    disused   = _str(tags.get("disused:building")).lower()
    abandoned = _str(tags.get("abandoned:building")).lower()

    is_spooky_building = (building in HORROR_BUILDING_TAGS or historic != "")
    is_derelict = (condition in HORROR_CONDITION_TAGS or
                   ruins in ("yes", "true") or
                   disused != "" or abandoned != "")

    return is_spooky_building or is_derelict


def extract_horror(tags: dict, city: str) -> dict | None:
    h_max, h_min = _height_from_tags(tags, level_scale=4.0)
    if h_max is None:
        h_max, h_min = 8.0, 0.0
    if h_max < 2.5:
        return None

    building  = _str(tags.get("building")).lower()
    historic  = _str(tags.get("historic")).lower()
    condition = _str(tags.get("building:condition")).lower()
    ruins     = _str(tags.get("ruins")).lower()
    levels    = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h_max / 4.0))

    wall_color = _get_wall_color(tags, HORROR_MATERIAL_COLOR)
    roof_type  = _get_roof(tags, default_roof=1)   # 默认坡顶

    is_ruin    = (condition in HORROR_CONDITION_TAGS or ruins in ("yes", "true"))
    is_church  = building in ("church", "chapel", "cathedral", "monastery")

    # 废墟建筑更暗更低矮；保留建筑更高
    if is_ruin:
        h_max = min(h_max, 10.0)
        win_d = 0.10
        wall_color = [max(0.15, c - 0.12) for c in wall_color]
    else:
        win_d = 0.18 if is_church else 0.22

    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name":   (_str(tags.get("name")) or "")[:80],
        "building_type": building or "church",
        "style": "horror",
        "height_range_min":  round(h_min, 2),
        "height_range_max":  round(min(h_max, 25.0), 2),
        "wall_thickness":    0.75,
        "floor_thickness":   0.22,
        "door_width":        1.20,
        "door_height":       2.80,
        "win_width":         0.50,
        "win_height":        1.80,
        "win_density":       win_d,
        "subdivision":       min(6, levels),
        "roof_type":         roof_type,
        "roof_pitch":        0.70,
        "wall_color":        [round(c, 3) for c in wall_color],
        "has_battlements":   0,
        "has_arch":          1,
        "eave_overhang":     0.20,
        "column_count":      0,
        "window_shape":      1,   # 尖拱窗
    }


# ── 3. Medieval Chapel ───────────────────────────────────────

CHAPEL_BUILDING_TAGS = {
    "chapel", "church", "cathedral", "monastery", "shrine",
}


def chapel_filter(tags: dict) -> bool:
    building = _str(tags.get("building")).lower()
    amenity  = _str(tags.get("amenity")).lower()
    historic = _str(tags.get("historic")).lower()
    return (building in CHAPEL_BUILDING_TAGS or
            amenity == "place_of_worship" or
            historic in ("church", "chapel", "monastery"))


def extract_chapel(tags: dict, city: str) -> dict | None:
    h_max, h_min = _height_from_tags(tags, level_scale=4.0)
    if h_max is None:
        h_max, h_min = 7.0, 0.0
    if h_max < 3.0:
        return None

    building = _str(tags.get("building")).lower()
    levels   = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h_max / 4.0))

    wall_color = _get_wall_color(tags, CHAPEL_MATERIAL_COLOR)
    roof_type  = _get_roof(tags, default_roof=1)   # 礼拜堂默认坡顶

    is_cathedral = building in ("cathedral", "monastery")
    # 大教堂比小礼拜堂更高更宏伟
    win_h  = 3.00 if is_cathedral else 2.20
    win_w  = 0.65 if is_cathedral else 0.55
    win_d  = 0.45
    door_h = 4.00 if is_cathedral else 3.00
    door_w = 1.80 if is_cathedral else 1.20
    cols   = 4    if is_cathedral else 2

    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name":   (_str(tags.get("name")) or "")[:80],
        "building_type": building or "chapel",
        "style": "medieval_chapel",
        "height_range_min":  round(h_min, 2),
        "height_range_max":  round(min(h_max, 30.0), 2),
        "wall_thickness":    0.65,
        "floor_thickness":   0.22,
        "door_width":        door_w,
        "door_height":       door_h,
        "win_width":         win_w,
        "win_height":        win_h,
        "win_density":       win_d,
        "subdivision":       min(8, levels),
        "roof_type":         roof_type,
        "roof_pitch":        0.75,
        "wall_color":        [round(c, 3) for c in wall_color],
        "has_battlements":   0,
        "has_arch":          1,
        "eave_overhang":     0.18,
        "column_count":      cols,
        "window_shape":      1,   # 尖拱窗
    }


# ── 4. Medieval Keep ────────────────────────────────────────

KEEP_BUILDING_TAGS = {
    "castle", "tower", "fort", "fortification", "defensive",
    "bunker", "gatehouse",
}
KEEP_HISTORIC_TAGS = {
    "castle", "tower", "fort", "fortification", "city_gate",
}


def keep_filter(tags: dict) -> bool:
    building = _str(tags.get("building")).lower()
    historic = _str(tags.get("historic")).lower()
    military = _str(tags.get("military")).lower()
    return (building in KEEP_BUILDING_TAGS or
            historic in KEEP_HISTORIC_TAGS or
            military in ("bunker", "fortification", "castle"))


def extract_keep(tags: dict, city: str) -> dict | None:
    h_max, h_min = _height_from_tags(tags, level_scale=5.0)
    if h_max is None:
        h_max, h_min = 15.0, 0.0
    if h_max < 3.0:
        return None

    building = _str(tags.get("building")).lower()
    levels   = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h_max / 5.0))

    wall_color = _get_wall_color(tags, KEEP_MATERIAL_COLOR)
    roof_type  = _get_roof(tags, default_roof=0)   # 要塞默认平顶/雉堞

    is_tower = building in ("tower", "gatehouse") or (h_max / max(1, levels) > 6.0)

    # 箭孔窗（很窄很高）
    win_w  = 0.30 if is_tower else 0.40
    win_h  = 0.80 if is_tower else 0.60
    win_d  = 0.08   # 极少窗

    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name":   (_str(tags.get("name")) or "")[:80],
        "building_type": building or "castle",
        "style": "medieval_keep",
        "height_range_min":  round(h_min, 2),
        "height_range_max":  round(min(h_max, 40.0), 2),
        "wall_thickness":    1.80,
        "floor_thickness":   0.30,
        "door_width":        1.20,
        "door_height":       2.20,
        "win_width":         win_w,
        "win_height":        win_h,
        "win_density":       win_d,
        "subdivision":       min(8, levels),
        "roof_type":         roof_type,
        "roof_pitch":        0.15,
        "wall_color":        [round(c, 3) for c in wall_color],
        "has_battlements":   1,
        "has_arch":          1,
        "eave_overhang":     0.05,
        "column_count":      0,
        "window_shape":      3,   # 箭孔
    }


# ════════════════════════════════════════════════════════════════
#  城市配置
# ════════════════════════════════════════════════════════════════

STYLE_CITY_CONFIGS = {
    "fantasy": {
        "cities": [
            {"name": "Prague",    "query": "Prague, Czech Republic"},
            {"name": "Edinburgh", "query": "Edinburgh, Scotland, UK"},
        ],
        "filter_fn":  fantasy_filter,
        "extract_fn": extract_fantasy,
    },
    "horror": {
        "cities": [
            {"name": "Sighisoara",  "query": "Sighisoara, Romania"},
            {"name": "New_Orleans", "query": "New Orleans, Louisiana, USA"},
            {"name": "Brasov",      "query": "Brasov, Romania"},   # 备用：特兰西瓦尼亚另一城
        ],
        "filter_fn":  horror_filter,
        "extract_fn": extract_horror,
    },
    "medieval_chapel": {
        "cities": [
            {"name": "Chartres",   "query": "Chartres, France"},
            {"name": "Canterbury", "query": "Canterbury, England, UK"},
            {"name": "Bruges",     "query": "Bruges, Belgium"},    # 备用：中世纪小城
        ],
        "filter_fn":  chapel_filter,
        "extract_fn": extract_chapel,
    },
    "medieval_keep": {
        "cities": [
            {"name": "Windsor",      "query": "Windsor, Berkshire, England"},
            {"name": "Carcassonne",  "query": "Carcassonne, France"},
            {"name": "Conwy",        "query": "Conwy, Wales, UK"},   # 备用：威尔士要塞城镇
        ],
        "filter_fn":  keep_filter,
        "extract_fn": extract_keep,
    },
}


# ─── 城市抓取 ─────────────────────────────────────────────────

def fetch_city(city_cfg: dict, filter_fn, extract_fn) -> list[dict]:
    name  = city_cfg["name"]
    query = city_cfg["query"]
    print(f"    [{name}] 正在查询 OSM...", flush=True)
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
    import sys
    # Windows Unicode fix
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65, flush=True)
    print("  fetch_osm_styles2.py  —  4 风格 OSM 建筑数据抓取", flush=True)
    print("=" * 65, flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    for style, cfg in STYLE_CITY_CONFIGS.items():
        print(f"\n{'─'*65}", flush=True)
        print(f"  风格: {style.upper()}", flush=True)
        all_records = []

        for city_cfg in cfg["cities"]:
            records = fetch_city(city_cfg, cfg["filter_fn"], cfg["extract_fn"])
            all_records.extend(records)
            time.sleep(1.5)

        by_city = {}
        for r in all_records:
            by_city.setdefault(r["city"], 0)
            by_city[r["city"]] += 1

        print(f"\n  [{style}] 总有效建筑: {len(all_records)} 条", flush=True)
        for city, cnt in by_city.items():
            print(f"    {city:<20}: {cnt:4d} 条")

        if all_records:
            heights = [r["height_range_max"] for r in all_records]
            print(f"  高度: min={min(heights):.1f}m  "
                  f"median={sorted(heights)[len(heights)//2]:.1f}m  "
                  f"max={max(heights):.1f}m", flush=True)

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
        print(f"  已保存: {out_path.name}", flush=True)
        summary[style] = len(all_records)

    print(f"\n{'='*65}", flush=True)
    print("  抓取完成汇总", flush=True)
    print(f"{'='*65}", flush=True)
    for style, cnt in summary.items():
        status = "ok" if cnt >= 100 else ("△" if cnt >= 20 else "X")
        print(f"  {status}  {style:<18}: {cnt:5d} 条")
    print()


if __name__ == "__main__":
    main()
