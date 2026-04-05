"""
fetch_osm_styles3.py
从 OpenStreetMap 获取剩余 11 种建筑风格的真实数据。

modern_loft         : Brooklyn NY / Berlin           — industrial, levels<=5
modern_villa        : Beverly Hills / Singapore      — house/villa, levels<=3
industrial_workshop : Sheffield / Essen              — industrial/workshop
industrial_powerplant: Dortmund / Duisburg           — industrial, height>=20m
fantasy_dungeon     : Edinburgh                      — historic/castle
fantasy_palace      : Versailles / Vienna            — palace/castle
horror_asylum       : London                         — hospital/historic + disused
horror_crypt        : Paris                          — tomb/chapel/crypt
japanese_temple     : Kyoto                          — temple/shrine
japanese_machiya    : Kyoto Gion district            — traditional/historic
desert_palace       : Fez Morocco / Isfahan Iran     — palace/historic
"""

import json
import re
import time
import math
from pathlib import Path

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
LOFT_COLOR = {
    "brick":      [0.62, 0.46, 0.34],
    "red_brick":  [0.65, 0.38, 0.28],
    "concrete":   [0.65, 0.63, 0.60],
    "steel":      [0.55, 0.56, 0.58],
    "wood":       [0.58, 0.50, 0.40],
    "yes":        [0.62, 0.55, 0.46],
}

VILLA_COLOR = {
    "plaster":    [0.95, 0.93, 0.88],
    "render":     [0.92, 0.90, 0.85],
    "concrete":   [0.88, 0.88, 0.86],
    "stone":      [0.82, 0.78, 0.70],
    "glass":      [0.72, 0.82, 0.90],
    "wood":       [0.75, 0.68, 0.55],
    "yes":        [0.92, 0.90, 0.85],
}

WORKSHOP_COLOR = {
    "brick":      [0.58, 0.42, 0.32],
    "red_brick":  [0.62, 0.35, 0.25],
    "concrete":   [0.58, 0.58, 0.56],
    "steel":      [0.50, 0.52, 0.54],
    "metal":      [0.48, 0.50, 0.52],
    "wood":       [0.52, 0.46, 0.36],
    "yes":        [0.55, 0.50, 0.44],
}

POWERPLANT_COLOR = {
    "concrete":   [0.48, 0.48, 0.46],
    "steel":      [0.42, 0.44, 0.46],
    "metal":      [0.40, 0.42, 0.44],
    "brick":      [0.50, 0.38, 0.30],
    "yes":        [0.45, 0.44, 0.42],
}

DUNGEON_COLOR = {
    "stone":      [0.28, 0.26, 0.24],
    "granite":    [0.25, 0.24, 0.22],
    "brick":      [0.32, 0.24, 0.20],
    "yes":        [0.30, 0.28, 0.26],
}

PALACE_COLOR = {
    "limestone":  [0.92, 0.88, 0.78],
    "stone":      [0.85, 0.80, 0.70],
    "sandstone":  [0.88, 0.78, 0.60],
    "plaster":    [0.95, 0.92, 0.85],
    "marble":     [0.95, 0.94, 0.90],
    "yes":        [0.88, 0.84, 0.75],
}

ASYLUM_COLOR = {
    "brick":      [0.58, 0.52, 0.46],
    "plaster":    [0.68, 0.65, 0.60],
    "concrete":   [0.55, 0.53, 0.50],
    "stone":      [0.52, 0.50, 0.46],
    "yes":        [0.55, 0.52, 0.48],
}

CRYPT_COLOR = {
    "stone":      [0.22, 0.20, 0.20],
    "limestone":  [0.35, 0.33, 0.30],
    "granite":    [0.20, 0.20, 0.20],
    "yes":        [0.25, 0.23, 0.22],
}

TEMPLE_COLOR = {
    "wood":       [0.60, 0.44, 0.28],
    "timber":     [0.58, 0.42, 0.26],
    "tile":       [0.30, 0.30, 0.32],
    "plaster":    [0.90, 0.86, 0.78],
    "clay":       [0.72, 0.58, 0.42],
    "stone":      [0.60, 0.58, 0.54],
    "yes":        [0.68, 0.52, 0.36],
}

MACHIYA_COLOR = {
    "wood":       [0.42, 0.32, 0.22],
    "timber":     [0.40, 0.30, 0.20],
    "clay":       [0.68, 0.55, 0.40],
    "plaster":    [0.82, 0.78, 0.70],
    "yes":        [0.45, 0.35, 0.24],
}

DESERT_PALACE_COLOR = {
    "stone":      [0.85, 0.72, 0.52],
    "limestone":  [0.88, 0.78, 0.58],
    "sandstone":  [0.88, 0.75, 0.55],
    "marble":     [0.92, 0.88, 0.80],
    "plaster":    [0.90, 0.80, 0.65],
    "yes":        [0.85, 0.75, 0.55],
}


# ─── 通用辅助 ────────────────────────────────────────────────────

def _str(val) -> str:
    if val is None:
        return ""
    try:
        if math.isnan(float(val)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()

def parse_height(val):
    if val is None:
        return None
    s = _str(val).lower().replace(",", ".")
    m = re.match(r"([\d.]+)\s*(m|ft|'|meters?|feet)?", s)
    if not m:
        return None
    v = float(m.group(1))
    if (m.group(2) or "m").strip("'") in ("ft", "feet"):
        v *= 0.3048
    return v if v > 0 else None

def parse_levels(val):
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

def _wall_color(tags, cmap):
    for key in ("building:material", "building:facade:material", "building"):
        val = _str(tags.get(key)).lower()
        if val in cmap:
            return cmap[val]
    return cmap.get("yes", [0.65, 0.62, 0.58])

def _roof(tags, default):
    return ROOF_SHAPE_MAP.get(_str(tags.get("roof:shape")).lower(), default)

def _height(tags, level_scale=3.5):
    h = parse_height(_str(tags.get("height")) or None)
    lv = parse_levels(_str(tags.get("building:levels")) or None)
    if h:
        return min(h, 80.0), 0.0
    if lv:
        return float(lv) * level_scale, 0.0
    return None, None


# ════════════════════════════════════════════════════════════════
#  风格专属过滤 & 参数提取 (11 styles)
# ════════════════════════════════════════════════════════════════

# ── 1. modern_loft ───────────────────────────────────────────────

def loft_filter(tags):
    building = _str(tags.get("building")).lower()
    levels   = parse_levels(_str(tags.get("building:levels")) or None)
    if building not in ("industrial", "warehouse", "factory", "yes"):
        return False
    if levels is not None and levels > 5:
        return False
    return True

def extract_loft(tags, city):
    h, _ = _height(tags, 4.0)
    if h is None:
        h = 4.5
    if h < 2.5 or h > 18.0:
        return None
    lv = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h / 4.0))
    wc = _wall_color(tags, LOFT_COLOR)
    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name": (_str(tags.get("name")) or "")[:80],
        "building_type": _str(tags.get("building")).lower() or "industrial",
        "style": "modern_loft",
        "height_range_min": 0.0, "height_range_max": round(min(h, 18.0), 2),
        "wall_thickness": 0.25, "floor_thickness": 0.18,
        "door_width": 1.40, "door_height": 2.80,
        "win_width": 1.80, "win_height": 1.60,
        "win_density": 0.65, "subdivision": min(5, lv),
        "roof_type": _roof(tags, 0), "roof_pitch": 0.0,
        "wall_color": [round(c, 3) for c in wc],
        "has_battlements": 0, "has_arch": 0, "eave_overhang": 0.0,
        "column_count": 0, "window_shape": 0,
    }


# ── 2. modern_villa ──────────────────────────────────────────────

def villa_filter(tags):
    building = _str(tags.get("building")).lower()
    levels   = parse_levels(_str(tags.get("building:levels")) or None)
    if building not in ("house", "villa", "detached", "residential", "bungalow",
                        "semidetached_house", "terrace"):
        return False
    if levels is not None and levels > 3:
        return False
    return True

def extract_villa(tags, city):
    h, _ = _height(tags, 3.5)
    if h is None:
        h = 5.0
    if h < 2.5 or h > 12.0:
        return None
    lv = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h / 3.5))
    wc = _wall_color(tags, VILLA_COLOR)
    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name": (_str(tags.get("name")) or "")[:80],
        "building_type": _str(tags.get("building")).lower() or "house",
        "style": "modern_villa",
        "height_range_min": 0.0, "height_range_max": round(min(h, 12.0), 2),
        "wall_thickness": 0.18, "floor_thickness": 0.15,
        "door_width": 1.10, "door_height": 2.20,
        "win_width": 1.60, "win_height": 1.40,
        "win_density": 0.85, "subdivision": min(3, lv),
        "roof_type": _roof(tags, 0), "roof_pitch": 0.0,
        "wall_color": [round(c, 3) for c in wc],
        "has_battlements": 0, "has_arch": 0, "eave_overhang": 0.30,
        "column_count": 0, "window_shape": 0,
    }


# ── 3. industrial_workshop ───────────────────────────────────────

WORKSHOP_TAGS = {"industrial", "warehouse", "factory", "manufacture",
                 "workshop", "shed", "hangar", "service"}

def workshop_filter(tags):
    building = _str(tags.get("building")).lower()
    return building in WORKSHOP_TAGS

def extract_workshop(tags, city):
    h, _ = _height(tags, 5.5)
    if h is None:
        h = 6.0
    if h < 3.0:
        return None
    lv = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h / 5.5))
    building = _str(tags.get("building")).lower()
    wc = _wall_color(tags, WORKSHOP_COLOR)
    is_large = building in ("warehouse", "hangar")
    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name": (_str(tags.get("name")) or "")[:80],
        "building_type": building or "workshop",
        "style": "industrial_workshop",
        "height_range_min": 0.0, "height_range_max": round(min(h, 25.0), 2),
        "wall_thickness": 0.45, "floor_thickness": 0.22,
        "door_width": 3.0 if is_large else 2.0,
        "door_height": 4.5 if is_large else 3.5,
        "win_width": 1.20, "win_height": 0.80,
        "win_density": 0.25 if is_large else 0.35,
        "subdivision": min(6, lv),
        "roof_type": _roof(tags, 1), "roof_pitch": 0.25,
        "wall_color": [round(c, 3) for c in wc],
        "has_battlements": 0, "has_arch": 0, "eave_overhang": 0.15,
        "column_count": 0, "window_shape": 0,
    }


# ── 4. industrial_powerplant ─────────────────────────────────────

POWERPLANT_TAGS = {"industrial", "warehouse", "factory", "manufacture",
                   "storage_tank", "chimney", "power"}

def powerplant_filter(tags):
    building = _str(tags.get("building")).lower()
    power    = _str(tags.get("power")).lower()
    if building not in POWERPLANT_TAGS and power not in ("plant", "generator", "substation"):
        return False
    # 需要高大建筑：检查 height 或 levels
    h = parse_height(_str(tags.get("height")) or None)
    lv = parse_levels(_str(tags.get("building:levels")) or None)
    if h is not None and h >= 20.0:
        return True
    if lv is not None and lv >= 4:
        return True
    if h is None and lv is None:
        return building in ("chimney", "storage_tank") or power == "plant"
    return False

def extract_powerplant(tags, city):
    h, _ = _height(tags, 6.0)
    if h is None:
        h = 20.0
    if h < 8.0:
        return None
    lv = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h / 6.0))
    wc = _wall_color(tags, POWERPLANT_COLOR)
    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name": (_str(tags.get("name")) or "")[:80],
        "building_type": _str(tags.get("building")).lower() or "industrial",
        "style": "industrial_powerplant",
        "height_range_min": 0.0, "height_range_max": round(min(h, 50.0), 2),
        "wall_thickness": 0.60, "floor_thickness": 0.28,
        "door_width": 2.50, "door_height": 4.00,
        "win_width": 0.80, "win_height": 0.60,
        "win_density": 0.15, "subdivision": min(8, lv),
        "roof_type": _roof(tags, 0), "roof_pitch": 0.0,
        "wall_color": [round(c, 3) for c in wc],
        "has_battlements": 0, "has_arch": 0, "eave_overhang": 0.0,
        "column_count": 0, "window_shape": 0,
    }


# ── 5. fantasy_dungeon ───────────────────────────────────────────

DUNGEON_BUILDING_TAGS = {"castle", "tower", "fort", "fortification", "historic",
                         "ruins", "bunker", "gatehouse"}
DUNGEON_HISTORIC_TAGS = {"castle", "tower", "fort", "fortification", "building",
                         "archaeological_site"}

def dungeon_filter(tags):
    building = _str(tags.get("building")).lower()
    historic = _str(tags.get("historic")).lower()
    return (building in DUNGEON_BUILDING_TAGS or historic in DUNGEON_HISTORIC_TAGS)

def extract_dungeon(tags, city):
    h, _ = _height(tags, 4.0)
    if h is None:
        h = 4.0
    h = min(h, 8.0)   # 地牢强制低矮
    if h < 2.0:
        return None
    lv = parse_levels(_str(tags.get("building:levels")) or None) or 1
    wc = _wall_color(tags, DUNGEON_COLOR)
    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name": (_str(tags.get("name")) or "")[:80],
        "building_type": _str(tags.get("building")).lower() or "castle",
        "style": "fantasy_dungeon",
        "height_range_min": 0.0, "height_range_max": round(h, 2),
        "wall_thickness": 0.90, "floor_thickness": 0.28,
        "door_width": 0.90, "door_height": 2.00,
        "win_width": 0.25, "win_height": 0.50,
        "win_density": 0.05, "subdivision": min(4, lv),
        "roof_type": _roof(tags, 0), "roof_pitch": 0.0,
        "wall_color": [round(c, 3) for c in wc],
        "has_battlements": 0, "has_arch": 1, "eave_overhang": 0.0,
        "column_count": 0, "window_shape": 1,
    }


# ── 6. fantasy_palace ────────────────────────────────────────────

PALACE_BUILDING_TAGS = {"palace", "castle", "cathedral", "manor", "mansion",
                        "monastery", "government"}
PALACE_HISTORIC_TAGS = {"palace", "castle", "manor_house", "building", "monument"}

def palace_filter(tags):
    building = _str(tags.get("building")).lower()
    historic = _str(tags.get("historic")).lower()
    tourism  = _str(tags.get("tourism")).lower()
    return (building in PALACE_BUILDING_TAGS or
            historic in PALACE_HISTORIC_TAGS or
            tourism in ("attraction",) and building not in ("", "yes", "residential"))

def extract_palace(tags, city):
    h, _ = _height(tags, 5.0)
    if h is None:
        h = 15.0
    if h < 5.0:
        return None
    building = _str(tags.get("building")).lower()
    lv = parse_levels(_str(tags.get("building:levels")) or None) or max(2, int(h / 5.0))
    wc = _wall_color(tags, PALACE_COLOR)
    roof_t = _roof(tags, 4)   # 默认四角炮楼/华丽顶
    is_palace = building in ("palace", "manor", "mansion")
    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name": (_str(tags.get("name")) or "")[:80],
        "building_type": building or "palace",
        "style": "fantasy_palace",
        "height_range_min": 0.0, "height_range_max": round(min(h, 40.0), 2),
        "wall_thickness": 0.70, "floor_thickness": 0.25,
        "door_width": 2.50 if is_palace else 2.00,
        "door_height": 4.50 if is_palace else 4.00,
        "win_width": 1.20, "win_height": 2.50,
        "win_density": 0.60, "subdivision": min(8, lv),
        "roof_type": roof_t, "roof_pitch": 0.85,
        "wall_color": [round(c, 3) for c in wc],
        "has_battlements": 1, "has_arch": 1, "eave_overhang": 0.30,
        "column_count": 6, "window_shape": 1,
    }


# ── 7. horror_asylum ─────────────────────────────────────────────

ASYLUM_BUILDING_TAGS = {"hospital", "clinic", "school", "government",
                        "office", "yes", "historic"}
ASYLUM_CONDITION_TAGS = {"disused", "abandoned", "derelict", "ruins",
                         "deteriorating", "bad", "very_bad"}

def asylum_filter(tags):
    building  = _str(tags.get("building")).lower()
    historic  = _str(tags.get("historic")).lower()
    condition = _str(tags.get("building:condition")).lower()
    disused   = _str(tags.get("disused:building")).lower()
    abandoned = _str(tags.get("abandoned:building")).lower()
    is_inst   = (building in ASYLUM_BUILDING_TAGS or historic != "")
    is_derel  = (condition in ASYLUM_CONDITION_TAGS or
                 disused != "" or abandoned != "" or
                 _str(tags.get("ruins")).lower() in ("yes",))
    return is_inst or is_derel

def extract_asylum(tags, city):
    h, _ = _height(tags, 3.8)
    if h is None:
        h = 5.0
    if h < 3.0:
        return None
    condition = _str(tags.get("building:condition")).lower()
    is_bad    = condition in ASYLUM_CONDITION_TAGS
    lv = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h / 3.8))
    wc = _wall_color(tags, ASYLUM_COLOR)
    if is_bad:
        wc = [max(0.25, c - 0.10) for c in wc]
    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name": (_str(tags.get("name")) or "")[:80],
        "building_type": _str(tags.get("building")).lower() or "hospital",
        "style": "horror_asylum",
        "height_range_min": 0.0, "height_range_max": round(min(h, 20.0), 2),
        "wall_thickness": 0.40, "floor_thickness": 0.20,
        "door_width": 1.20, "door_height": 2.50,
        "win_width": 0.70, "win_height": 1.00,
        "win_density": 0.25 if is_bad else 0.32,
        "subdivision": min(6, lv),
        "roof_type": _roof(tags, 1), "roof_pitch": 0.38,
        "wall_color": [round(c, 3) for c in wc],
        "has_battlements": 0, "has_arch": 0, "eave_overhang": 0.12,
        "column_count": 0, "window_shape": 0,
    }


# ── 8. horror_crypt ──────────────────────────────────────────────

CRYPT_BUILDING_TAGS = {"tomb", "chapel", "church", "mausoleum",
                       "crypt", "ruins", "historic"}
CRYPT_HISTORIC_TAGS = {"tomb", "memorial", "archaeological_site", "building",
                       "church", "monastery", "chapel"}

def crypt_filter(tags):
    building  = _str(tags.get("building")).lower()
    historic  = _str(tags.get("historic")).lower()
    amenity   = _str(tags.get("amenity")).lower()
    return (building in CRYPT_BUILDING_TAGS or
            historic in CRYPT_HISTORIC_TAGS or
            amenity in ("grave_yard", "place_of_worship"))

def extract_crypt(tags, city):
    h, _ = _height(tags, 4.0)
    if h is None:
        h = 3.5
    h = min(h, 10.0)
    if h < 2.0:
        return None
    lv = parse_levels(_str(tags.get("building:levels")) or None) or 1
    wc = _wall_color(tags, CRYPT_COLOR)
    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name": (_str(tags.get("name")) or "")[:80],
        "building_type": _str(tags.get("building")).lower() or "tomb",
        "style": "horror_crypt",
        "height_range_min": 0.0, "height_range_max": round(h, 2),
        "wall_thickness": 0.85, "floor_thickness": 0.25,
        "door_width": 0.80, "door_height": 1.80,
        "win_width": 0.20, "win_height": 0.40,
        "win_density": 0.02, "subdivision": min(3, lv),
        "roof_type": _roof(tags, 2), "roof_pitch": 0.60,
        "wall_color": [round(c, 3) for c in wc],
        "has_battlements": 0, "has_arch": 1, "eave_overhang": 0.05,
        "column_count": 0, "window_shape": 1,
    }


# ── 9. japanese_temple ───────────────────────────────────────────

TEMPLE_BUILDING_TAGS = {"temple", "shrine", "pagoda", "monastery",
                        "cathedral", "chapel"}
TEMPLE_HISTORIC_TAGS = {"temple", "shrine", "building", "castle"}
TEMPLE_AMENITY_TAGS  = {"place_of_worship"}

def temple_filter(tags):
    building = _str(tags.get("building")).lower()
    historic = _str(tags.get("historic")).lower()
    amenity  = _str(tags.get("amenity")).lower()
    religion = _str(tags.get("religion")).lower()
    return (building in TEMPLE_BUILDING_TAGS or
            historic in TEMPLE_HISTORIC_TAGS or
            (amenity in TEMPLE_AMENITY_TAGS and religion in ("buddhist", "shinto", "")))

def extract_temple(tags, city):
    h, _ = _height(tags, 3.5)
    if h is None:
        h = 6.0
    if h < 2.5:
        return None
    building = _str(tags.get("building")).lower()
    lv = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h / 3.5))
    wc = _wall_color(tags, TEMPLE_COLOR)
    is_pagoda = building == "pagoda"
    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name": (_str(tags.get("name")) or "")[:80],
        "building_type": building or "temple",
        "style": "japanese_temple",
        "height_range_min": 0.0, "height_range_max": round(min(h, 20.0), 2),
        "wall_thickness": 0.20, "floor_thickness": 0.22,
        "door_width": 1.20, "door_height": 2.20,
        "win_width": 0.90, "win_height": 1.50,
        "win_density": 0.30, "subdivision": min(5, lv),
        "roof_type": 3,   # 强制翘角屋顶
        "roof_pitch": 0.60,
        "wall_color": [round(c, 3) for c in wc],
        "has_battlements": 0, "has_arch": 0, "eave_overhang": 0.65,
        "column_count": 4 if is_pagoda else 2, "window_shape": 0,
    }


# ── 10. japanese_machiya ─────────────────────────────────────────

MACHIYA_BUILDING_TAGS = {"house", "residential", "traditional", "yes",
                         "terrace", "semidetached_house"}
MACHIYA_HISTORIC_TAGS = {"building", "monument"}

def machiya_filter(tags):
    building  = _str(tags.get("building")).lower()
    historic  = _str(tags.get("historic")).lower()
    arch_per  = _str(tags.get("architectural_period")).lower()
    return (building in MACHIYA_BUILDING_TAGS or
            historic in MACHIYA_HISTORIC_TAGS or
            arch_per in ("edo", "meiji", "taisho", "traditional"))

def extract_machiya(tags, city):
    h, _ = _height(tags, 3.0)
    if h is None:
        h = 4.0
    if h < 2.0 or h > 8.0:
        return None
    lv = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h / 3.0))
    wc = _wall_color(tags, MACHIYA_COLOR)
    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name": (_str(tags.get("name")) or "")[:80],
        "building_type": _str(tags.get("building")).lower() or "house",
        "style": "japanese_machiya",
        "height_range_min": 0.0, "height_range_max": round(h, 2),
        "wall_thickness": 0.12, "floor_thickness": 0.15,
        "door_width": 0.90, "door_height": 1.90,
        "win_width": 0.60, "win_height": 1.00,
        "win_density": 0.40, "subdivision": min(3, lv),
        "roof_type": _roof(tags, 1), "roof_pitch": 0.45,
        "wall_color": [round(c, 3) for c in wc],
        "has_battlements": 0, "has_arch": 0, "eave_overhang": 0.40,
        "column_count": 0, "window_shape": 0,
    }


# ── 11. desert_palace ────────────────────────────────────────────

DESERT_PALACE_BUILDING_TAGS = {"palace", "castle", "fort", "fortification",
                                "monastery", "cathedral", "mansion"}
DESERT_PALACE_HISTORIC_TAGS = {"palace", "castle", "fort", "building",
                                "monument", "archaeological_site"}

def desert_palace_filter(tags):
    building = _str(tags.get("building")).lower()
    historic = _str(tags.get("historic")).lower()
    tourism  = _str(tags.get("tourism")).lower()
    return (building in DESERT_PALACE_BUILDING_TAGS or
            historic in DESERT_PALACE_HISTORIC_TAGS or
            tourism in ("attraction",) and building not in ("", "yes", "residential"))

def extract_desert_palace(tags, city):
    h, _ = _height(tags, 4.5)
    if h is None:
        h = 10.0
    if h < 4.0:
        return None
    building = _str(tags.get("building")).lower()
    lv = parse_levels(_str(tags.get("building:levels")) or None) or max(1, int(h / 4.5))
    wc = _wall_color(tags, DESERT_PALACE_COLOR)
    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name": (_str(tags.get("name")) or "")[:80],
        "building_type": building or "palace",
        "style": "desert_palace",
        "height_range_min": 0.0, "height_range_max": round(min(h, 30.0), 2),
        "wall_thickness": 0.80, "floor_thickness": 0.22,
        "door_width": 2.00, "door_height": 3.50,
        "win_width": 0.60, "win_height": 1.20,
        "win_density": 0.30, "subdivision": min(6, lv),
        "roof_type": _roof(tags, 4), "roof_pitch": 0.50,
        "wall_color": [round(c, 3) for c in wc],
        "has_battlements": 1, "has_arch": 1, "eave_overhang": 0.15,
        "column_count": 4, "window_shape": 4,
    }


# ════════════════════════════════════════════════════════════════
#  城市配置
# ════════════════════════════════════════════════════════════════

STYLE_CITY_CONFIGS = {
    "modern_loft": {
        "cities": [
            {"name": "Brooklyn",  "query": "Brooklyn, New York, USA"},
            {"name": "Berlin",    "query": "Berlin, Germany"},
        ],
        "filter_fn": loft_filter, "extract_fn": extract_loft,
    },
    "modern_villa": {
        "cities": [
            {"name": "Beverly_Hills", "query": "Beverly Hills, California, USA"},
            {"name": "Singapore",     "query": "Singapore"},
        ],
        "filter_fn": villa_filter, "extract_fn": extract_villa,
    },
    "industrial_workshop": {
        "cities": [
            {"name": "Sheffield",  "query": "Sheffield, England"},
            {"name": "Essen",      "query": "Essen, Germany"},
        ],
        "filter_fn": workshop_filter, "extract_fn": extract_workshop,
    },
    "industrial_powerplant": {
        "cities": [
            {"name": "Dortmund",  "query": "Dortmund, Germany"},
            {"name": "Duisburg",  "query": "Duisburg, Germany"},
        ],
        "filter_fn": powerplant_filter, "extract_fn": extract_powerplant,
    },
    "fantasy_dungeon": {
        "cities": [
            {"name": "Edinburgh",  "query": "Edinburgh, Scotland, UK"},
            {"name": "Stirling",   "query": "Stirling, Scotland, UK"},
        ],
        "filter_fn": dungeon_filter, "extract_fn": extract_dungeon,
    },
    "fantasy_palace": {
        "cities": [
            {"name": "Versailles",  "query": "Versailles, France"},
            {"name": "Vienna",      "query": "Vienna, Austria"},
        ],
        "filter_fn": palace_filter, "extract_fn": extract_palace,
    },
    "horror_asylum": {
        "cities": [
            {"name": "London",       "query": "London, England"},
            {"name": "Manchester",   "query": "Manchester, England"},
        ],
        "filter_fn": asylum_filter, "extract_fn": extract_asylum,
    },
    "horror_crypt": {
        "cities": [
            {"name": "Paris",    "query": "Paris, France"},
            {"name": "Bruges",   "query": "Bruges, Belgium"},
        ],
        "filter_fn": crypt_filter, "extract_fn": extract_crypt,
    },
    "japanese_temple": {
        "cities": [
            {"name": "Kyoto",   "query": "Kyoto, Japan"},
            {"name": "Nara",    "query": "Nara, Japan"},
        ],
        "filter_fn": temple_filter, "extract_fn": extract_temple,
    },
    "japanese_machiya": {
        "cities": [
            {"name": "Gion",     "query": "Gion, Kyoto, Japan"},
            {"name": "Fushimi",  "query": "Fushimi, Kyoto, Japan"},
        ],
        "filter_fn": machiya_filter, "extract_fn": extract_machiya,
    },
    "desert_palace": {
        "cities": [
            {"name": "Fez",       "query": "Fez, Morocco"},
            {"name": "Marrakech", "query": "Marrakech, Morocco"},
        ],
        "filter_fn": desert_palace_filter, "extract_fn": extract_desert_palace,
    },
}


# ─── 城市抓取 ─────────────────────────────────────────────────

def fetch_city(city_cfg, filter_fn, extract_fn):
    name, query = city_cfg["name"], city_cfg["query"]
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
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65, flush=True)
    print("  fetch_osm_styles3.py  —  11 风格 OSM 建筑数据抓取", flush=True)
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
        for c, n in by_city.items():
            print(f"    {c:<20}: {n:4d} 条")
        if all_records:
            hh = [r["height_range_max"] for r in all_records]
            print(f"  高度: min={min(hh):.1f}m  median={sorted(hh)[len(hh)//2]:.1f}m  max={max(hh):.1f}m")

        out = OUTPUT_DIR / f"{style}_osm.json"
        out.write_text(json.dumps({"total": len(all_records),
                                   "cities": list(by_city.keys()),
                                   "buildings": all_records},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  已保存: {out.name}", flush=True)
        summary[style] = len(all_records)

    print(f"\n{'='*65}", flush=True)
    print("  抓取完成汇总", flush=True)
    for style, cnt in summary.items():
        status = "ok" if cnt >= 100 else ("△" if cnt >= 20 else "X ")
        print(f"  {status}  {style:<22}: {cnt:6d} 条")


if __name__ == "__main__":
    main()
