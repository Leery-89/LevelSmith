"""
fetch_osm_layout.py
从 OSM 提取中世纪历史城镇建筑布局数据，作为关卡布局模型训练集。

提取内容（每栋建筑）：
  - 中心坐标：相对城镇中心，单位米，归一化到 ±50m（即 100m×100m 范围内）
  - 朝向角度：-180~180°，0° = 正北
  - 尺寸：length（长边）、width（短边），单位米
  - 建筑类型标签：church/castle/house/residential/commercial/industrial/civic/other
  - 原始 OSM tags（building 字段）

输出：
  training_data/osm_layouts.json          汇总（全部城镇）
  training_data/{town}_layout.json        每个城镇单独文件

用法：
  python fetch_osm_layout.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import osmnx as ox
except ImportError:
    raise SystemExit("pip install osmnx")

try:
    from shapely.geometry import MultiPolygon
except ImportError:
    raise SystemExit("pip install shapely")

# ─── 目标城镇 ────────────────────────────────────────────────────
TOWNS = [
    {
        "name":    "carcassonne",
        "display": "Carcassonne, France",
        "query":   "Carcassonne, Aude, France",
        # 只取老城（Cité）核心区，bbox: (west, south, east, north)
        "bbox":    (2.358, 43.202, 2.374, 43.212),
    },
    {
        "name":    "bruges",
        "display": "Bruges, Belgium",
        "query":   "Bruges, West Flanders, Belgium",
        # 老城中心环绕运河区域
        "bbox":    (3.208, 51.196, 3.240, 51.218),
    },
    {
        "name":    "york",
        "display": "York, England",
        "query":   "York, England, United Kingdom",
        # 城墙以内历史核心区
        "bbox":    (-1.094, 53.953, -1.073, 53.966),
    },
    {
        "name":    "rothenburg",
        "display": "Rothenburg ob der Tauber, Germany",
        "query":   "Rothenburg ob der Tauber, Bavaria, Germany",
        # 完整老城范围（城墙内）
        "bbox":    (10.172, 49.373, 10.192, 49.383),
    },
]

# ─── 面积过滤阈值 ────────────────────────────────────────────────
AREA_MIN = 10.0     # m²
AREA_MAX = 2000.0   # m²

# ─── 归一化目标尺寸（输出坐标范围） ─────────────────────────────
TARGET_HALF = 50.0  # 输出坐标 ±50m → 100m×100m 网格

# ─── 建筑类型映射 ────────────────────────────────────────────────
_TAG_MAP = {
    # 教堂 / 宗教
    "church":       "church",
    "cathedral":    "church",
    "chapel":       "church",
    "monastery":    "church",
    "religious":    "church",
    "temple":       "church",
    # 城堡 / 防御
    "castle":       "castle",
    "fortress":     "castle",
    "tower":        "castle",
    "city_wall":    "castle",
    # 住宅
    "house":        "house",
    "residential":  "house",
    "detached":     "house",
    "semidetached_house": "house",
    "terrace":      "house",
    "apartments":   "house",
    "dormitory":    "house",
    # 商业
    "commercial":   "commercial",
    "retail":       "commercial",
    "shop":         "commercial",
    "supermarket":  "commercial",
    "hotel":        "commercial",
    "inn":          "commercial",
    # 工业
    "industrial":   "industrial",
    "warehouse":    "industrial",
    "barn":         "industrial",
    "farm":         "industrial",
    # 公共 / 市政
    "civic":        "civic",
    "public":       "civic",
    "school":       "civic",
    "university":   "civic",
    "hospital":     "civic",
    "government":   "civic",
    "townhall":     "civic",
    "museum":       "civic",
    "library":      "civic",
}


def _classify(tags: dict) -> str:
    """将 OSM building 标签映射为简化类型。"""
    raw = str(tags.get("building", "")).lower()
    return _TAG_MAP.get(raw, "other")


def _latlon_to_meters(lat: float, lon: float,
                      ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """
    将 (lat, lon) 转换为以 (ref_lat, ref_lon) 为原点的平面坐标（米）。
    使用等距柱状投影近似：
      x = (lon - ref_lon) * cos(ref_lat) * 111320
      y = (lat - ref_lat) * 110540
    """
    x = (lon - ref_lon) * math.cos(math.radians(ref_lat)) * 111_320.0
    y = (lat - ref_lat) * 110_540.0
    return x, y   # x=东, y=北


def _poly_to_record(geom, tags: dict,
                    ref_lat: float, ref_lon: float) -> dict | None:
    """
    将单个建筑多边形转换为数据记录。
    返回 None 表示跳过（面积不符、几何无效等）。
    """
    # 取外轮廓（MultiPolygon 取最大子多边形）
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon" or geom.is_empty:
        return None

    # ── 投影面积过滤 ──────────────────────────────────────────
    # 用 Shapely 的面积（地理坐标°²），粗略换算 m²
    # 更准确：用投影坐标计算
    coords = list(geom.exterior.coords)
    if len(coords) < 4:
        return None

    # 将轮廓点投影为平面米坐标（相对参考点）
    pts_m = [_latlon_to_meters(lat, lon, ref_lat, ref_lon)
             for lon, lat in coords]  # shapely coords = (lon, lat)

    # 用 Shoelace 公式计算实际面积（m²）
    n = len(pts_m)
    area_m2 = abs(sum(
        pts_m[i][0] * pts_m[(i + 1) % n][1] -
        pts_m[(i + 1) % n][0] * pts_m[i][1]
        for i in range(n)
    )) / 2.0

    if area_m2 < AREA_MIN or area_m2 > AREA_MAX:
        return None

    # ── 建筑中心 ──────────────────────────────────────────────
    cx_m = sum(p[0] for p in pts_m[:-1]) / (n - 1)
    cy_m = sum(p[1] for p in pts_m[:-1]) / (n - 1)

    # ── 最小外接矩形（OBB）求朝向和尺寸 ─────────────────────
    from shapely.geometry import Polygon as ShapelyPolygon
    poly_m = ShapelyPolygon(pts_m)
    try:
        obb = poly_m.minimum_rotated_rectangle
    except Exception:
        obb = poly_m.envelope

    obb_coords = list(obb.exterior.coords)[:4]
    # 取最长边方向作为 length
    edges = []
    for i in range(len(obb_coords)):
        p0 = obb_coords[i]
        p1 = obb_coords[(i + 1) % len(obb_coords)]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        edges.append((math.hypot(dx, dy), math.degrees(math.atan2(dx, dy))))
    edges.sort(key=lambda e: e[0], reverse=True)
    length = edges[0][0]
    width  = edges[1][0] if len(edges) > 1 else edges[0][0]

    # 朝向：最长边与正北（+Y）的夹角，-180~180°
    orientation_deg = edges[0][1]
    # 标准化到 -180~180
    orientation_deg = ((orientation_deg + 180) % 360) - 180

    return {
        "cx_m":            round(cx_m, 2),
        "cy_m":            round(cy_m, 2),
        "area_m2":         round(area_m2, 1),
        "length_m":        round(length, 2),
        "width_m":         round(width,  2),
        "orientation_deg": round(orientation_deg, 1),
        "type":            _classify(tags),
        "osm_building":    str(tags.get("building", "")),
        "osm_name":        str(tags.get("name", "")),
    }


def fetch_town(town: dict, out_dir: Path) -> list[dict]:
    """
    从 OSM 提取单个城镇的建筑数据。
    返回归一化后的建筑记录列表。
    """
    name    = town["name"]
    display = town["display"]
    bbox    = town["bbox"]   # (west, south, east, north)

    print(f"\n{'─'*60}")
    print(f"  {display}")
    print(f"  bbox: W={bbox[0]}, S={bbox[1]}, E={bbox[2]}, N={bbox[3]}")

    # 参考点 = bbox 中心
    ref_lon = (bbox[0] + bbox[2]) / 2
    ref_lat = (bbox[1] + bbox[3]) / 2
    print(f"  参考原点: ({ref_lat:.5f}°N, {ref_lon:.5f}°E)")

    # ── 从 OSM 拉取建筑 ──────────────────────────────────────
    print("  [OSM] 拉取建筑数据...", end=" ", flush=True)
    try:
        gdf = ox.features_from_bbox(
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            tags={"building": True},
        )
    except Exception as e:
        print(f"失败: {e}")
        return []

    raw_count = len(gdf)
    print(f"{raw_count} 条原始记录")

    # ── 逐个处理 ─────────────────────────────────────────────
    records = []
    skipped_geom  = 0
    skipped_area  = 0

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            skipped_geom += 1
            continue

        tags = row.to_dict()
        rec  = _poly_to_record(geom, tags, ref_lat, ref_lon)
        if rec is None:
            skipped_area += 1
            continue
        records.append(rec)

    print(f"  过滤: 几何无效 {skipped_geom}  面积不符 {skipped_area}")
    print(f"  有效建筑: {len(records)} 栋")

    if not records:
        return []

    # ── 归一化到 ±50m（100m×100m）─────────────────────────────
    all_cx = [r["cx_m"] for r in records]
    all_cy = [r["cy_m"] for r in records]
    span_x = max(all_cx) - min(all_cx) if len(all_cx) > 1 else 1.0
    span_y = max(all_cy) - min(all_cy) if len(all_cy) > 1 else 1.0
    scale  = TARGET_HALF * 2 / max(span_x, span_y, 1.0)  # 保持长宽比

    # 中心质心
    centroid_x = (max(all_cx) + min(all_cx)) / 2
    centroid_y = (max(all_cy) + min(all_cy)) / 2

    for r in records:
        r["nx"] = round((r["cx_m"] - centroid_x) * scale, 2)
        r["ny"] = round((r["cy_m"] - centroid_y) * scale, 2)

    # ── 统计 ─────────────────────────────────────────────────
    types = {}
    for r in records:
        types[r["type"]] = types.get(r["type"], 0) + 1
    print("  类型分布:", "  ".join(f"{k}:{v}" for k, v in sorted(types.items())))

    # ── 保存单城镇文件 ────────────────────────────────────────
    town_data = {
        "town":       name,
        "display":    display,
        "ref_lat":    ref_lat,
        "ref_lon":    ref_lon,
        "bbox":       {"west": bbox[0], "south": bbox[1],
                       "east": bbox[2], "north": bbox[3]},
        "scale_m_per_unit": round(1.0 / scale, 4),
        "raw_count":   raw_count,
        "building_count": len(records),
        "buildings":  records,
    }
    out_path = out_dir / f"{name}_layout.json"
    out_path.write_text(json.dumps(town_data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"  保存: {out_path.name}")

    return records


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 60)
    print("  fetch_osm_layout.py  —  中世纪城镇建筑布局提取")
    print("=" * 60)

    # 输出目录
    out_dir = Path(__file__).parent / "training_data"
    out_dir.mkdir(exist_ok=True)

    # 配置 osmnx 缓存
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    ox.settings.cache_folder = str(cache_dir)
    ox.settings.use_cache    = True
    ox.settings.log_console  = False

    # ── 逐城镇提取 ────────────────────────────────────────────
    all_records  = {}
    total_buildings = 0

    for town in TOWNS:
        records = fetch_town(town, out_dir)
        all_records[town["name"]] = records
        total_buildings += len(records)

    # ── 保存汇总文件 ──────────────────────────────────────────
    summary = {
        "description": "Medieval town building layouts extracted from OSM",
        "towns": [t["name"] for t in TOWNS],
        "area_filter": {"min_m2": AREA_MIN, "max_m2": AREA_MAX},
        "coordinate_system": {
            "origin": "town center (centroid of all buildings)",
            "range":  f"±{TARGET_HALF}m (normalized to 100m×100m)",
            "x":      "east (+) / west (-)",
            "y":      "north (+) / south (-)",
            "orientation_deg": "-180~180, 0=north, 90=east",
        },
        "per_town": {
            t["name"]: len(all_records[t["name"]]) for t in TOWNS
        },
        "total_buildings": total_buildings,
    }

    summary_path = out_dir / "osm_layouts.json"
    # 将各城镇数据嵌入汇总
    summary["data"] = {k: v for k, v in all_records.items()}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    # ── 最终汇总 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  提取结果汇总")
    print(f"{'='*60}")
    for t in TOWNS:
        n = len(all_records[t["name"]])
        print(f"  {t['display']:<42} {n:>4} 栋")
    print(f"  {'─'*50}")
    print(f"  {'合计':<42} {total_buildings:>4} 栋")
    print(f"\n  汇总文件: {summary_path}")
    print(f"  单城镇文件: training_data/{{town}}_layout.json")


if __name__ == "__main__":
    main()
