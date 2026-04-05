"""
fetch_osm_medieval.py
从 OpenStreetMap 获取中世纪历史城区的建筑数据，
提取建筑参数后保存到 validation_data/medieval_osm.json。

目标城区:
  - Carcassonne, France  （中世纪城堡城市）
  - York, England        （保存完好的中世纪城区）
  - Bruges, Belgium      （中世纪商业城市）

参数映射:
  building:levels → height_range（× 3.5 m/层）
  height          → height_range_max（直接使用，优先）
  roof:shape      → roof_type
  building:material / building → wall_color（RGB 估算）
  start_date      → 过滤年份 > 1900 的现代建筑
"""

import json
import re
import time
from pathlib import Path

import numpy as np

try:
    import osmnx as ox
    import geopandas as gpd
except ImportError:
    raise SystemExit("请先安装依赖: pip install osmnx geopandas")

# ─── 配置 ──────────────────────────────────────────────────────
OUTPUT_PATH = Path(__file__).parent / "validation_data" / "medieval_osm.json"

# 目标城区（place query → osmnx 可解析的地名）
CITIES = [
    {"name": "Carcassonne",  "query": "Carcassonne, France",  "country": "FR"},
    {"name": "York",         "query": "York, England",        "country": "GB"},
    {"name": "Bruges",       "query": "Bruges, Belgium",      "country": "BE"},
]

# roof:shape → roof_type 映射
ROOF_SHAPE_MAP = {
    "flat":       0,
    "yes":        0,
    "gabled":     1,
    "hipped":     2,
    "half-hipped": 2,
    "pyramidal":  2,
    "dome":       3,
    "cone":       3,
    "onion":      3,
    "round":      3,
    "saltbox":    1,
    "gambrel":    1,
    "mansard":    2,
    "skillion":   1,
}

# building:material / building → 归一化 wall_color [r, g, b] (0-1)
MATERIAL_COLOR = {
    # 石材
    "stone":        [0.65, 0.60, 0.52],
    "limestone":    [0.88, 0.84, 0.70],
    "sandstone":    [0.80, 0.68, 0.48],
    "granite":      [0.55, 0.52, 0.50],
    "flint":        [0.40, 0.38, 0.36],
    "chalk":        [0.92, 0.90, 0.85],
    # 砖材
    "brick":        [0.70, 0.38, 0.28],
    "red_brick":    [0.72, 0.30, 0.22],
    "dark_brick":   [0.52, 0.28, 0.20],
    # 木材/灰泥
    "timber_framing": [0.72, 0.63, 0.48],
    "half_timbered":  [0.72, 0.63, 0.48],
    "plaster":      [0.85, 0.82, 0.75],
    "render":       [0.83, 0.80, 0.72],
    "concrete":     [0.68, 0.66, 0.62],
    # 建筑类型推断
    "castle":       [0.60, 0.56, 0.50],
    "cathedral":    [0.72, 0.68, 0.58],
    "church":       [0.70, 0.66, 0.56],
    "monastery":    [0.65, 0.62, 0.54],
    "yes":          [0.64, 0.59, 0.52],   # 默认石灰石色
}
DEFAULT_COLOR = [0.64, 0.59, 0.52]


# ─── 辅助函数 ──────────────────────────────────────────────────

def parse_height(val) -> float | None:
    """解析 OSM height 字段（可能含单位）→ 米。"""
    if val is None:
        return None
    s = str(val).strip().lower().replace(",", ".")
    # 去掉单位
    m = re.match(r"([\d.]+)\s*(m|ft|'|meters?|feet)?", s)
    if not m:
        return None
    v = float(m.group(1))
    unit = (m.group(2) or "m").strip("'")
    if unit in ("ft", "feet"):
        v *= 0.3048
    return v if v > 0 else None


def parse_levels(val) -> int | None:
    """解析 building:levels → 整数楼层数。"""
    if val is None:
        return None
    try:
        return max(1, int(float(str(val).split(";")[0].strip())))
    except (ValueError, AttributeError):
        return None


def parse_start_date(val) -> int | None:
    """从 start_date 字段提取年份，允许 '1150', '12th century', '~1450' 等格式。"""
    if val is None:
        return None
    s = str(val)
    # 提取4位年份
    m = re.search(r"\b(1\d{3}|2\d{3})\b", s)
    if m:
        return int(m.group(1))
    # 世纪表达：'12th century' → ~1150
    m = re.search(r"(\d{1,2})(st|nd|rd|th)\s*century", s, re.I)
    if m:
        return (int(m.group(1)) - 1) * 100 + 50
    return None


def get_wall_color(tags: dict) -> list:
    """从 building:material 或 building 标签推断 wall_color。"""
    for key in ("building:material", "building:facade:material", "building"):
        val = _str(tags.get(key)).lower()
        if val in MATERIAL_COLOR:
            return MATERIAL_COLOR[val]
    return DEFAULT_COLOR


def _str(val) -> str:
    """安全转换 OSM 字段为字符串；NaN / None 返回空字符串。"""
    if val is None:
        return ""
    try:
        import math
        if math.isnan(float(val)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def extract_building(row) -> dict | None:
    """
    从 GeoDataFrame 行提取建筑参数，返回 dict 或 None（不满足条件时）。
    """
    # 将 pandas Series 转成普通 dict，NaN → None
    raw = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    tags = {}
    for k, v in raw.items():
        try:
            import math
            tags[k] = None if (isinstance(v, float) and math.isnan(v)) else v
        except TypeError:
            tags[k] = v

    # ── 年份过滤：跳过 > 1900 的现代建筑 ──────────────────────
    year = parse_start_date(_str(tags.get("start_date")) or None)
    if year is not None and year > 1900:
        return None

    # ── 高度 ──────────────────────────────────────────────────
    h_direct = parse_height(_str(tags.get("height")) or None)
    levels   = parse_levels(_str(tags.get("building:levels")) or None)

    if h_direct is not None:
        h_max = min(h_direct, 50.0)
        h_min = 0.0
    elif levels is not None:
        h_max = float(levels) * 3.5
        h_min = 0.0
    else:
        return None   # 无高度信息，跳过

    if h_max < 2.0:   # 太矮，不是建筑
        return None

    # ── 屋顶类型 ──────────────────────────────────────────────
    roof_shape = _str(tags.get("roof:shape")).lower()
    roof_type  = ROOF_SHAPE_MAP.get(roof_shape, 1)   # 中世纪默认山墙

    # ── 墙色 ──────────────────────────────────────────────────
    wall_color = get_wall_color(tags)

    # ── 额外字段 ──────────────────────────────────────────────
    name   = _str(tags.get("name")) or _str(tags.get("addr:street")) or ""
    osm_id = tags.get("osmid", "")
    b_type = _str(tags.get("building")).lower() or "yes"

    # 有城垛标签？
    historic = _str(tags.get("historic")).lower()
    has_battlements = int(
        historic in ("castle", "fort", "city_gate", "tower") or
        b_type in ("castle", "fortification", "tower")
    )

    # 拱门推断：教堂/大教堂通常有拱门
    amenity  = _str(tags.get("amenity")).lower()
    has_arch = int(
        b_type in ("cathedral", "church", "monastery", "chapel") or
        amenity in ("place_of_worship",)
    )

    # 细分（层数近似）
    subdivision = min(8, levels or 1)

    return {
        "source":            "osm",
        "city":              tags.get("_city", ""),
        "osm_id":            str(osm_id),
        "name":              str(name)[:80],
        "building_type":     b_type,
        "style":             "medieval",
        "start_date_parsed": year,
        # ── 提取的参数 ──
        "height_range_min":  round(h_min, 2),
        "height_range_max":  round(h_max, 2),
        "wall_thickness":    0.80,    # 中世纪石砌墙，固定估算
        "floor_thickness":   0.35,
        "door_width":        1.4,
        "door_height":       2.8,
        "win_width":         0.7,
        "win_height":        1.2,
        "win_density":       0.12,
        "subdivision":       subdivision,
        "roof_type":         roof_type,
        "roof_pitch":        0.45,
        "wall_color":        [round(c, 3) for c in wall_color],
        "has_battlements":   has_battlements,
        "has_arch":          has_arch,
        "eave_overhang":     0.15,
        "column_count":      0,
        "window_shape":      1,   # 中世纪默认尖拱窗
    }


# ─── 城市抓取 ──────────────────────────────────────────────────

def fetch_city(city_cfg: dict) -> list[dict]:
    name  = city_cfg["name"]
    query = city_cfg["query"]
    print(f"\n  [{name}] 正在查询 OSM...")

    try:
        tags = ox.settings
        # 获取建筑物 GeoDataFrame
        gdf = ox.features_from_place(
            query,
            tags={"building": True},
        )
    except Exception as e:
        print(f"    [失败] {e}")
        return []

    print(f"    原始建筑: {len(gdf)} 条")

    results = []
    skipped_modern = 0
    skipped_no_height = 0

    for idx, row in gdf.iterrows():
        row_dict = row.copy()
        row_dict["_city"] = name
        rec = extract_building(row_dict)
        if rec is None:
            # 判断跳过原因（近似）
            year = parse_start_date(_str(row.get("start_date")) or None)
            if year and year > 1900:
                skipped_modern += 1
            else:
                skipped_no_height += 1
            continue
        results.append(rec)

    print(f"    有效建筑: {len(results)}  "
          f"(跳过现代: {skipped_modern}, 无高度: {skipped_no_height})")
    return results


# ─── 主程序 ────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  fetch_osm_medieval.py  —  OpenStreetMap 中世纪建筑数据")
    print("=" * 65)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_records = []

    for city_cfg in CITIES:
        records = fetch_city(city_cfg)
        all_records.extend(records)
        time.sleep(1.0)   # 礼貌延迟

    # ── 统计 ──────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  总有效建筑: {len(all_records)} 条")
    by_city = {}
    for r in all_records:
        by_city.setdefault(r["city"], 0)
        by_city[r["city"]] += 1
    for city, cnt in by_city.items():
        print(f"    {city:<20}: {cnt:4d} 条")

    if all_records:
        heights = [r["height_range_max"] for r in all_records]
        print(f"\n  高度分布: min={min(heights):.1f}m  "
              f"median={sorted(heights)[len(heights)//2]:.1f}m  "
              f"max={max(heights):.1f}m")

        roof_counts = {}
        for r in all_records:
            roof_counts[r["roof_type"]] = roof_counts.get(r["roof_type"], 0) + 1
        print(f"  屋顶类型分布: "
              + "  ".join(f"type{k}={v}" for k, v in sorted(roof_counts.items())))

    # ── 保存 ──────────────────────────────────────────────────
    output = {
        "total": len(all_records),
        "cities": list(by_city.keys()),
        "buildings": all_records,
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  已保存: {OUTPUT_PATH.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
