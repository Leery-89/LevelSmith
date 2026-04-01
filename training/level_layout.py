"""
level_layout.py
关卡群生成器：根据风格和布局类型自动排列多个建筑，输出完整场景 GLB。

支持布局类型:
  grid    — 网格排列（适合现代/工业风格）
  street  — 沿街道两侧排列（适合中世纪/日式）
  plaza   — 围绕中心广场排列（适合奇幻/沙漠）
  random  — 随机散布（适合恐怖风格）
  organic — 有机城镇生长：中心锚点 + 泊松圆盘扩散 + 外圈围合

用法:
    python level_layout.py --style medieval --layout street --count 10 --out level_test.glb
    python level_layout.py --style modern --layout grid --count 12 --area 120
    python level_layout.py --style horror --layout random --count 8 --variation 0.8
    python level_layout.py --style medieval --layout organic --count 15 --out level_organic_test.glb
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import trimesh
except ImportError:
    raise SystemExit("pip install trimesh")

try:
    from shapely.geometry import Polygon
    from shapely.affinity import translate as shapely_translate, rotate as shapely_rotate
except ImportError:
    raise SystemExit("pip install shapely")

import generate_level as gl

SCRIPT_DIR   = Path(__file__).parent
PARAMS_JSON  = SCRIPT_DIR / "trained_style_params.json"

# ─── 建筑尺寸基准（每风格的典型宽/深范围，单位 m）────────────────
_STYLE_SIZE = {
    # style: (w_min, w_max, d_min, d_max)
    "medieval":             ( 9.0, 14.0,  7.0, 11.0),
    "modern":               (12.0, 20.0, 10.0, 16.0),
    "industrial":           (14.0, 24.0, 12.0, 20.0),
    "fantasy":              (10.0, 16.0,  8.0, 13.0),
    "horror":               ( 8.0, 15.0,  7.0, 12.0),
    "japanese":             ( 8.0, 13.0,  7.0, 11.0),
    "desert":               (10.0, 16.0,  9.0, 14.0),
    "medieval_chapel":      (10.0, 18.0,  8.0, 14.0),
    "medieval_keep":        ( 8.0, 13.0,  8.0, 13.0),
    "modern_loft":          (12.0, 20.0, 10.0, 16.0),
    "modern_villa":         (10.0, 18.0,  9.0, 15.0),
    "industrial_workshop":  (12.0, 22.0, 10.0, 18.0),
    "industrial_powerplant":(16.0, 28.0, 14.0, 22.0),
    "fantasy_dungeon":      ( 8.0, 14.0,  8.0, 14.0),
    "fantasy_palace":       (14.0, 22.0, 12.0, 18.0),
    "horror_asylum":        (12.0, 20.0, 10.0, 16.0),
    "horror_crypt":         ( 7.0, 12.0,  7.0, 12.0),
    "japanese_temple":      (10.0, 16.0,  9.0, 14.0),
    "japanese_machiya":     ( 6.0, 10.0,  8.0, 14.0),
    "desert_palace":        (14.0, 22.0, 12.0, 18.0),
}
_DEFAULT_SIZE = (10.0, 16.0, 8.0, 13.0)

# ─── 风格调色板（floor/ceiling/door/window，wall 由 params.wall_color 覆盖）──
_STYLE_PALETTES = {
    "medieval": {
        "floor":   [115,  95,  75, 255], "ceiling": [135, 115,  90, 255],
        "wall":    [162, 146, 122, 255], "door":    [ 88,  58,  28, 255],
        "window":  [155, 195, 215, 180], "internal":[148, 133, 110, 255],
        "ground":  [ 90,  80,  65, 255],
    },
    "modern": {
        "floor":   [215, 215, 215, 255], "ceiling": [242, 242, 242, 255],
        "wall":    [198, 198, 198, 255], "door":    [ 68,  68,  68, 255],
        "window":  [135, 195, 228, 180], "internal":[182, 182, 182, 255],
        "ground":  [160, 160, 155, 255],
    },
    "industrial": {
        "floor":   [ 72,  72,  68, 255], "ceiling": [ 52,  52,  48, 255],
        "wall":    [ 92,  88,  82, 255], "door":    [158,  82,  38, 255],
        "window":  [112, 138, 112, 180], "internal":[ 82,  78,  72, 255],
        "ground":  [ 55,  55,  52, 255],
    },
    "fantasy": {
        "floor":   [100,  80,  60, 255], "ceiling": [120,  95,  70, 255],
        "wall":    [138, 110,  85, 255], "door":    [ 70,  40,  20, 255],
        "window":  [180, 145, 210, 180], "internal":[128, 102,  78, 255],
        "ground":  [ 80,  68,  52, 255],
    },
    "horror": {
        "floor":   [ 55,  50,  45, 255], "ceiling": [ 40,  36,  33, 255],
        "wall":    [ 68,  62,  56, 255], "door":    [ 35,  28,  22, 255],
        "window":  [ 60,  80,  60, 120], "internal":[ 60,  55,  50, 255],
        "ground":  [ 42,  38,  34, 255],
    },
    "japanese": {
        "floor":   [165, 130,  90, 255], "ceiling": [180, 148, 108, 255],
        "wall":    [190, 165, 128, 255], "door":    [110,  70,  30, 255],
        "window":  [220, 210, 175, 180], "internal":[175, 150, 115, 255],
        "ground":  [120,  98,  72, 255],
    },
    "desert": {
        "floor":   [200, 175, 130, 255], "ceiling": [218, 192, 148, 255],
        "wall":    [215, 190, 148, 255], "door":    [140, 105,  60, 255],
        "window":  [180, 210, 240, 180], "internal":[200, 178, 138, 255],
        "ground":  [180, 158, 118, 255],
    },
}
# 其余风格由 wall_color 动态派生
# ─── 每种风格的中心锚点风格（标志性最大建筑）────────────────────
_ANCHOR_STYLE = {
    "medieval":             "medieval_keep",
    "modern":               "modern_loft",
    "industrial":           "industrial_powerplant",
    "fantasy":              "fantasy_palace",
    "horror":               "horror_asylum",
    "japanese":             "japanese_temple",
    "desert":               "desert_palace",
    # 子风格使用自身
    "medieval_chapel":      "medieval_chapel",
    "medieval_keep":        "medieval_keep",
    "modern_loft":          "modern_loft",
    "modern_villa":         "modern_villa",
    "industrial_workshop":  "industrial_workshop",
    "industrial_powerplant":"industrial_powerplant",
    "fantasy_dungeon":      "fantasy_dungeon",
    "fantasy_palace":       "fantasy_palace",
    "horror_asylum":        "horror_asylum",
    "horror_crypt":         "horror_crypt",
    "japanese_temple":      "japanese_temple",
    "japanese_machiya":     "japanese_machiya",
    "desert_palace":        "desert_palace",
}

_BASELINE_PALETTE = {
    "floor":   [172, 158, 142, 255], "ceiling": [192, 182, 172, 255],
    "wall":    [182, 172, 162, 255], "door":    [ 98,  78,  58, 255],
    "window":  [148, 175, 200, 180], "internal":[168, 160, 152, 255],
    "ground":  [135, 128, 118, 255],
}


def _derive_palette(style: str, params: dict) -> dict:
    """从风格名或 wall_color 推导调色板。"""
    base = _STYLE_PALETTES.get(style)
    if base is None:
        # 从 wall_color 派生
        wc = params.get("wall_color")
        if wc:
            wr, wg, wb = [int(c * 255) for c in wc]
            base = {
                "floor":   [max(0,wr-25), max(0,wg-25), max(0,wb-25), 255],
                "ceiling": [min(255,wr+25), min(255,wg+25), min(255,wb+25), 255],
                "wall":    [wr, wg, wb, 255],
                "door":    [max(0,wr-50), max(0,wg-50), max(0,wb-50), 255],
                "window":  [min(255,wr+55), min(255,wg+70), min(255,wb+90), 175],
                "internal":[max(0,wr-15), max(0,wg-15), max(0,wb-15), 255],
                "ground":  [max(0,wr-35), max(0,wg-35), max(0,wb-30), 255],
            }
        else:
            base = dict(_BASELINE_PALETTE)
    return dict(base)


# ─── 平面轮廓工厂 ───────────────────────────────────────────────

def _make_footprint(fp_type: str, w: float, d: float,
                    rng: np.random.Generator) -> Polygon:
    """生成矩形/L形/U形平面轮廓，原点在左前角。"""
    from shapely.geometry import box as sbox
    if fp_type == "L":
        cx = rng.uniform(0.35, 0.55)
        cz = rng.uniform(0.35, 0.55)
        rect = sbox(0, 0, w, d)
        cut  = sbox(w * (1 - cx), d * (1 - cz), w, d)
        fp   = rect.difference(cut)
        return fp if not fp.is_empty else sbox(0, 0, w, d)
    elif fp_type == "U":
        nx = rng.uniform(0.35, 0.50)
        nz = rng.uniform(0.40, 0.60)
        rect = sbox(0, 0, w, d)
        nx0  = w * (0.5 - nx / 2)
        nx1  = w * (0.5 + nx / 2)
        cut  = sbox(nx0, 0, nx1, d * nz)
        fp   = rect.difference(cut)
        return fp if not fp.is_empty else sbox(0, 0, w, d)
    else:
        from shapely.geometry import box as sbox
        return sbox(0, 0, w, d)


def _pick_footprint_type(variation: float, rng: np.random.Generator) -> str:
    if variation < 0.25:
        return "rect"
    elif variation < 0.55:
        return rng.choice(["rect", "rect", "L"])
    elif variation < 0.80:
        return rng.choice(["rect", "L", "U"])
    else:
        return rng.choice(["rect", "L", "L", "U"])


def _building_size(style: str, variation: float,
                   rng: np.random.Generator,
                   role: str = "normal") -> tuple[float, float]:
    """
    Generate building size based on style and role.
    Roles:
      "anchor"     — 1.15x scale (central / landmark building)
      "sub_anchor" — 1.0x scale, biased toward upper range
      "filler"     — 0.7x scale, biased toward lower range
      "normal"     — standard random range (default)
    """
    wmin, wmax, dmin, dmax = _STYLE_SIZE.get(style, _DEFAULT_SIZE)
    noise = variation * 0.5            # 最大 ±50% of range
    if role == "anchor":
        w = rng.uniform(wmax * 0.95, wmax * 1.15)
        d = rng.uniform(dmax * 0.95, dmax * 1.15)
    elif role == "sub_anchor":
        w = rng.uniform(wmin + (wmax - wmin) * 0.5, wmax)
        d = rng.uniform(dmin + (dmax - dmin) * 0.5, dmax)
    elif role == "filler":
        w = rng.uniform(wmin * 0.85, wmin + (wmax - wmin) * 0.4)
        d = rng.uniform(dmin * 0.85, dmin + (dmax - dmin) * 0.4)
    else:
        w = rng.uniform(wmin, wmin + (wmax - wmin) * (0.5 + noise))
        d = rng.uniform(dmin, dmin + (dmax - dmin) * (0.5 + noise))
    return round(w, 1), round(d, 1)


# ─── OBB 碰撞检测（有向包围盒，考虑旋转角）────────────────────
#
# 检测规则：
#   • 用旋转矩形（OBB）代替 AABB，正确处理任意偏航角
#   • 每栋建筑外围加 OBB_BUFFER（1.5 m）安全边距
#   • 强制最小边到边间距 = MIN_GAP（3 m）
#   • 快速预筛：包围圆不相交则跳过 Shapely 计算

SAFETY_GAP = 1.5      # 建筑边到边最小间距（m），可通过 set_gap() 或 --min-gap 覆盖
OBB_BUFFER = 0.5      # 每个 OBB 外围安全边距（m）

# ─── 密度控制参数 ──────────────────────────────────────────────
MIN_BUILDING_GAP   = 5.0    # 任意两栋建筑外墙最小间距(m)
MAX_DENSITY_PER_CELL = 2    # 每个密度网格单元最多 2 栋建筑（允许 cluster）
DENSITY_CELL_SIZE  = 15.0   # 密度检测网格单元大小(m)

# ─── 围墙间距 ──────────────────────────────────────────────────
WALL_THICKNESS = 1.0
WALL_MARGIN    = 3.0
WALL_TOTAL     = WALL_THICKNESS + WALL_MARGIN  # 4m from area edge
WALL_DOOR_THRESHOLD = 5.0  # building edge-to-wall < 5m → that direction blocked


# 提示词 → 密度映射关键词
_DENSITY_COMPACT  = {"紧凑", "密集", "拥挤", "compact", "dense", "crowded"}
_DENSITY_SPACIOUS = {"宽松", "分散", "稀疏", "spacious", "sparse", "spread"}


def parse_density_prompt(prompt: str) -> float:
    """
    从自然语言提示词自动识别布局密度，返回对应 min_gap（m）。
    - 紧凑/密集/拥挤  → 0.5 m
    - 宽松/分散/稀疏  → 4.0 m
    - 其他           → 1.5 m（默认）
    """
    tokens = set(prompt.lower().replace(",", " ").split())
    if tokens & _DENSITY_COMPACT:
        return 0.5
    if tokens & _DENSITY_SPACIOUS:
        return 4.0
    return 1.5


def set_gap(min_gap: float) -> None:
    """全局更新 SAFETY_GAP（影响当前进程内所有后续布局调用）。"""
    global SAFETY_GAP
    SAFETY_GAP = float(min_gap)


def _obb_polygon(cx: float, cz: float, w: float, d: float,
                 yaw_deg: float = 0.0) -> "Polygon":
    """以中心 (cx, cz)、宽 w、深 d、偏航角 yaw_deg 创建 OBB Shapely 多边形。"""
    from shapely.geometry import box as _sbox
    rect = _sbox(-w / 2, -d / 2, w / 2, d / 2)
    if abs(yaw_deg) > 0.01:
        rect = shapely_rotate(rect, yaw_deg, origin=(0, 0), use_radians=False)
    return shapely_translate(rect, cx, cz)


def _obb_collides(cx: float, cz: float, w: float, d: float,
                  yaw_deg: float, placed: list,
                  safety: float = SAFETY_GAP) -> bool:
    """
    候选 OBB 是否与已放置列表中的任意 OBB 碰撞（边到边距离 < safety）。
    placed : list[Polygon]（_obb_polygon 返回的 Shapely 多边形）。
    每个 OBB 再向外扩展 OBB_BUFFER，确保视觉间距。
    返回 True = 碰撞，False = 安全。
    """
    half_diag = math.hypot(w, d) / 2
    # 候选多边形加 OBB_BUFFER 边距
    candidate = _obb_polygon(cx, cz, w, d, yaw_deg).buffer(OBB_BUFFER)

    for p_poly in placed:
        # 快速预筛：包围圆距离不足时才做精确计算
        b = p_poly.bounds                       # (minx, miny, maxx, maxy)
        p_cx = (b[0] + b[2]) / 2
        p_cz = (b[1] + b[3]) / 2
        p_half = math.hypot(b[2] - b[0], b[3] - b[1]) / 2
        if math.hypot(cx - p_cx, cz - p_cz) > half_diag + p_half + safety + OBB_BUFFER * 2:
            continue
        # 精确 Shapely 相交检测（已放置的也加 OBB_BUFFER）
        if candidate.intersects(p_poly.buffer(OBB_BUFFER)):
            return True
    return False


# ─── 密度检查 ─────────────────────────────────────────────────────

def _edge_to_edge_distance(ax, az, aw, ad, bx, bz, bw, bd):
    """AABB 边到边距离（中心坐标 + 半宽/半深）。"""
    dx = max(0.0, abs(ax - bx) - (aw + bw) / 2)
    dz = max(0.0, abs(az - bz) - (ad + bd) / 2)
    return math.hypot(dx, dz)


def _zone_density_factor(px: float, pz: float,
                         area_w: float, area_d: float) -> float:
    """
    Return 0.6~1.0 density factor: higher near center, lower at edges.
    Used to probabilistically thin out buildings at the periphery.
    """
    # Normalized distance from center (0 = center, 1 = corner)
    dx = (px - area_w / 2) / (area_w / 2)
    dz = (pz - area_d / 2) / (area_d / 2)
    dist = min(1.0, math.hypot(dx, dz))
    # Linear falloff: center=1.0, edge=0.6
    return 1.0 - 0.4 * dist


def check_density(cx, cz, w, d, existing_slots, area_w, area_d):
    """
    密度检查：
    1. 最小间距：新建筑与任何已放置建筑的边到边距离 >= MIN_BUILDING_GAP
    2. 区域密度：同一网格单元内最多 MAX_DENSITY_PER_CELL 栋建筑
    返回 True = 可以放置。
    """
    # 1. 最小间距
    for s in existing_slots:
        sx = s.x_off + s.w / 2
        sz = s.z_off + s.d / 2
        gap = _edge_to_edge_distance(cx, cz, w, d, sx, sz, s.w, s.d)
        if gap < MIN_BUILDING_GAP:
            return False
    # 2. 区域密度
    cell_x = int(cx / DENSITY_CELL_SIZE)
    cell_z = int(cz / DENSITY_CELL_SIZE)
    count = 0
    for s in existing_slots:
        sx = s.x_off + s.w / 2
        sz = s.z_off + s.d / 2
        if int(sx / DENSITY_CELL_SIZE) == cell_x and \
           int(sz / DENSITY_CELL_SIZE) == cell_z:
            count += 1
    if count >= MAX_DENSITY_PER_CELL:
        return False
    return True


def enforce_wall_clearance(slots, area_w, area_d):
    """
    Clamp all building slots so they don't overlap with perimeter walls.
    Applied after layout functions, before mesh generation.
    """
    for s in slots:
        hw, hd = s.w / 2, s.d / 2
        cx = s.x_off + hw
        cz = s.z_off + hd
        cx = max(WALL_TOTAL + hw, min(area_w - WALL_TOTAL - hw, cx))
        cz = max(WALL_TOTAL + hd, min(area_d - WALL_TOTAL - hd, cz))
        s.x_off = cx - hw
        s.z_off = cz - hd


def get_allowed_yaws(bx, bz, bw, bd, area_w, area_d):
    """
    Return (allowed, blocked) yaw sets.
    Door starts at local z_min; after yaw rotation around center:
      yaw=0   → door at z_min side → faces -Z → blocked if near bottom wall
      yaw=90  → door at x_min side → faces -X → blocked if near left wall
      yaw=180 → door at z_max side → faces +Z → blocked if near top wall
      yaw=270 → door at x_max side → faces +X → blocked if near right wall
    """
    hw, hd = bw / 2, bd / 2
    dist_left   = bx - hw              # door-to-left-wall  (yaw=90 puts door here)
    dist_right  = area_w - (bx + hw)   # door-to-right-wall (yaw=270 puts door here)
    dist_bottom = bz - hd              # door-to-bottom-wall (yaw=0 puts door here)
    dist_top    = area_d - (bz + hd)   # door-to-top-wall   (yaw=180 puts door here)

    blocked = set()
    if dist_bottom < WALL_DOOR_THRESHOLD:
        blocked.add(0)      # yaw=0 → door at -Z → near bottom wall
    if dist_left < WALL_DOOR_THRESHOLD:
        blocked.add(90)     # yaw=90 → door at -X → near left wall
    if dist_top < WALL_DOOR_THRESHOLD:
        blocked.add(180)    # yaw=180 → door at +Z → near top wall
    if dist_right < WALL_DOOR_THRESHOLD:
        blocked.add(270)    # yaw=270 → door at +X → near right wall

    allowed = [y for y in (0, 90, 180, 270) if y not in blocked]
    if not allowed:
        dists = {0: dist_bottom, 90: dist_left,
                 180: dist_top, 270: dist_right}
        allowed = [max(dists, key=dists.get)]
    return allowed, blocked


def _angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def fix_door_facing_wall(effective_yaw, building_infos, area_w, area_d,
                         layout_type="street"):
    """
    For each building: check if its yaw direction is blocked by a wall.
    Snap layouts (street/grid/random): pick closest allowed 90° direction.
    Free layouts (plaza/organic): flip 180°, or rotate 90° if still blocked.
    Modifies effective_yaw in place.
    """
    snap = layout_type in ("street", "grid")

    for i, b in enumerate(building_infos):
        ideal = effective_yaw.get(i, 0.0)
        allowed, blocked = get_allowed_yaws(
            b["x"], b["z"], b["w"], b["d"], area_w, area_d)

        old_yaw = ideal
        if snap:
            snapped = round(ideal / 90.0) * 90.0 % 360
            if snapped not in blocked:
                effective_yaw[i] = snapped
            else:
                effective_yaw[i] = min(allowed,
                                       key=lambda a: _angle_diff(a, ideal))
        else:
            nearest_90 = round(ideal / 90.0) * 90.0 % 360
            if nearest_90 not in blocked:
                effective_yaw[i] = ideal
            else:
                flipped = (ideal + 180) % 360
                flip_90 = round(flipped / 90.0) * 90.0 % 360
                if flip_90 not in blocked:
                    effective_yaw[i] = flipped
                else:
                    effective_yaw[i] = (ideal + 90) % 360

        final = effective_yaw[i]
        if final != old_yaw:
            print(f"    b{i}: wall fix {old_yaw:.0f} -> {final:.0f} "
                  f"(blocked={blocked})")


# ─── 布局算法 ───────────────────────────────────────────────────

@dataclass
class BuildingSlot:
    x_off:     float
    z_off:     float
    w:         float
    d:         float
    fp_type:   str   = "rect"
    yaw_deg:   float = 0.0    # 绕 Y 轴旋转
    is_anchor: bool  = False  # True = 中心锚点建筑，使用锚点风格参数
    style_key: str   = ""     # 覆盖风格（空=使用场景主风格）
    role:      str   = ""     # archetype role: primary/secondary/tertiary/ambient


def layout_grid(n: int, area_w: float, area_d: float,
                style: str, variation: float,
                rng: np.random.Generator) -> list[BuildingSlot]:
    """
    网格布局：建筑强制落在网格单元中心，道路在网格线上。
    yaw=0，由 Phase 4 access edge 决定门朝向。
    """
    cols = max(1, math.ceil(math.sqrt(n * area_w / area_d)))
    rows = math.ceil(n / cols)
    cell_w = area_w / cols
    cell_d = area_d / rows

    # Store grid info for road generation
    layout_grid._cell_w = cell_w
    layout_grid._cell_d = cell_d
    layout_grid._cols = cols
    layout_grid._rows = rows

    slots: list[BuildingSlot] = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            w, d = _building_size(style, variation, rng)
            w = min(w, cell_w * 0.65)
            d = min(d, cell_d * 0.65)
            # Cell center + small jitter to break symmetry for yaw calc
            jx = float(rng.uniform(-cell_w * 0.1, cell_w * 0.1))
            jz = float(rng.uniform(-cell_d * 0.1, cell_d * 0.1))
            cx = (c + 0.5) * cell_w + jx
            cz = (r + 0.5) * cell_d + jz
            x_off = cx - w / 2
            z_off = cz - d / 2
            if not check_density(cx, cz, w, d, slots, area_w, area_d):
                continue
            fp = _pick_footprint_type(variation, rng)
            slots.append(BuildingSlot(x_off, z_off, w, d, fp, yaw_deg=0.0))
            idx += 1

    return slots


STREET_WIDTH = 10.0   # road width (fixed)
STREET_SETBACK = 2.0  # building outer wall to road edge


def layout_street(n: int, area_w: float, area_d: float,
                  style: str, variation: float,
                  rng: np.random.Generator) -> list[BuildingSlot]:
    """
    Street layout using bounding-circle abstraction.
    Road centered at area_d/2, buildings placed by radius so nothing crosses the road.
    """
    road_center_z = area_d / 2
    road_left  = road_center_z - STREET_WIDTH / 2
    road_right = road_center_z + STREET_WIDTH / 2

    n_left  = (n + 1) // 2
    n_right = n // 2
    gap_min = max(SAFETY_GAP, 3.0)

    slots = []

    for side, count, yaw in [("left", n_left, 180.0), ("right", n_right, 0.0)]:
        slot_w = (area_w - gap_min) / max(count, 1)

        for i in range(count):
            w, d = _building_size(style, variation, rng)
            w = min(w, slot_w * 0.85)
            radius = max(w, d) / 2

            # X: evenly distributed along street + jitter
            cx = i * slot_w + slot_w / 2
            cx += rng.uniform(-variation * 1.0, variation * 1.0)
            cx = max(radius, min(area_w - radius, cx))

            # Z: place by radius so building circle doesn't cross road edge
            if side == "left":
                cz = road_left - STREET_SETBACK - radius
                cz = max(radius + 1.0, cz)
            else:
                cz = road_right + STREET_SETBACK + radius
                cz = min(area_d - radius - 1.0, cz)

            # Slot x_off/z_off = top-left corner (center - half)
            x_off = cx - w / 2
            z_off = cz - d / 2

            fp = _pick_footprint_type(variation, rng)
            slots.append(BuildingSlot(x_off, z_off, w, d, fp, yaw_deg=yaw))

    # Store road geometry info for _make_street_road
    layout_street._road_center_z = road_center_z
    layout_street._road_width = STREET_WIDTH

    return slots


def _compute_coverage(slots, cx, cz):
    """Compute boundary_coverage and gap info from slots."""
    if len(slots) < 3:
        return 0.0, 360.0, 0, []
    angles = sorted(
        math.degrees(math.atan2(s.z_off + s.d / 2 - cz,
                                s.x_off + s.w / 2 - cx))
        for s in slots
    )
    gaps = []
    for i in range(len(angles)):
        j = (i + 1) % len(angles)
        gap = angles[j] - angles[i]
        if gap <= 0:
            gap += 360
        if i == len(angles) - 1:
            gap = (angles[0] + 360) - angles[-1]
        gaps.append((gap, (angles[i] + gap / 2) if i < len(angles) - 1
                     else (angles[-1] + gap / 2)))
    max_gap = max(g for g, _ in gaps) if gaps else 360
    coverage = (360 - max_gap) / 360
    opening_count = sum(1 for g, _ in gaps if g > 45)
    return coverage, max_gap, opening_count, gaps


def _closed_bezier_loop(cx, cz, base_r, n_ctrl=6, jitter=15.0, rng=None):
    """
    Generate a closed loop of Bezier curve points.
    n_ctrl control points evenly spaced on a circle, each jittered ±jitter meters.
    Returns list of (x, z) points sampled along the loop.
    """
    # Control points on a noisy circle
    ctrl = []
    for i in range(n_ctrl):
        angle = 2 * math.pi * i / n_ctrl
        r = base_r + rng.uniform(-jitter, jitter)
        ctrl.append((cx + r * math.cos(angle), cz + r * math.sin(angle)))
    ctrl.append(ctrl[0])  # close loop

    # Sample cubic Bezier between consecutive ctrl triplets
    pts = []
    n_seg = 20  # samples per segment
    for i in range(len(ctrl) - 1):
        p0 = ctrl[i]
        p3 = ctrl[(i + 1) % len(ctrl)]
        # Control handles: pull toward circle center for smoothness
        mid_x = (p0[0] + p3[0]) / 2
        mid_z = (p0[1] + p3[1]) / 2
        pull = 0.3
        p1 = (p0[0] + (mid_x - p0[0]) * pull + rng.uniform(-3, 3),
              p0[1] + (mid_z - p0[1]) * pull + rng.uniform(-3, 3))
        p2 = (p3[0] + (mid_x - p3[0]) * pull + rng.uniform(-3, 3),
              p3[1] + (mid_z - p3[1]) * pull + rng.uniform(-3, 3))
        for j in range(n_seg):
            t = j / n_seg
            u = 1 - t
            x = u**3*p0[0] + 3*u**2*t*p1[0] + 3*u*t**2*p2[0] + t**3*p3[0]
            z = u**3*p0[1] + 3*u**2*t*p1[1] + 3*u*t**2*p2[1] + t**3*p3[1]
            pts.append((x, z))
    return pts


def layout_plaza(n: int, area_w: float, area_d: float,
                 style: str, variation: float,
                 rng: np.random.Generator) -> list[BuildingSlot]:
    """
    环形广场布局：建筑均匀分布在环形带上，面朝广场中心。
    多栋时可分多圈，密度由 check_density 控制。
    """
    cx = area_w / 2
    cz = area_d / 2
    plaza_radius = min(area_w, area_d) * 0.15

    # Ring belt: inner → outer
    inner_r = plaza_radius + MIN_BUILDING_GAP + 5.0   # leave road space
    outer_r = min(area_w, area_d) / 2 - 8.0
    if outer_r <= inner_r:
        outer_r = inner_r + 10.0

    # How many per ring?
    buildings_per_ring = max(6, n)
    n_rings = math.ceil(n / buildings_per_ring)

    slots: list[BuildingSlot] = []
    placed: list = []   # Polygon list for OBB
    idx = 0

    for ring in range(n_rings):
        if n_rings == 1:
            r = (inner_r + outer_r) / 2
        else:
            r = inner_r + (outer_r - inner_r) * ring / (n_rings - 1)
        n_in_ring = min(buildings_per_ring, n - idx)

        for i in range(n_in_ring):
            if idx >= n:
                break
            angle = 2 * math.pi * i / n_in_ring
            # Small jitter
            angle += rng.uniform(-0.15, 0.15) * variation
            r_jit = r + rng.uniform(-2.0, 2.0) * variation

            w, d = _building_size(style, variation, rng)
            bx_c = cx + r_jit * math.cos(angle)
            bz_c = cz + r_jit * math.sin(angle)

            # Clamp to area bounds
            bx_c = max(w / 2 + 2, min(area_w - w / 2 - 2, bx_c))
            bz_c = max(d / 2 + 2, min(area_d - d / 2 - 2, bz_c))

            # yaw = 0, Phase 4 will set door toward center via access edge
            yaw = 0.0

            # OBB collision + density
            if _obb_collides(bx_c, bz_c, w, d, yaw, placed,
                             MIN_BUILDING_GAP):
                # Try 10 small offsets
                ok = False
                for _ in range(10):
                    bx_c2 = bx_c + rng.uniform(-3, 3)
                    bz_c2 = bz_c + rng.uniform(-3, 3)
                    if not _obb_collides(bx_c2, bz_c2, w, d, yaw, placed,
                                         MIN_BUILDING_GAP):
                        bx_c, bz_c = bx_c2, bz_c2
                        ok = True
                        break
                if not ok:
                    continue

            x_off = bx_c - w / 2
            z_off = bz_c - d / 2
            if not check_density(bx_c, bz_c, w, d, slots, area_w, area_d):
                continue

            placed.append(_obb_polygon(bx_c, bz_c, w, d, yaw))
            fp = _pick_footprint_type(variation, rng)
            slots.append(BuildingSlot(x_off, z_off, w, d, fp, yaw_deg=yaw))
            idx += 1

    # Generate ring road for road mesh generation
    loop_pts = []
    ring_r = (inner_r + plaza_radius) / 2 + 2.0  # road between plaza and buildings
    for i in range(60):
        a = 2 * math.pi * i / 60
        loop_pts.append((cx + ring_r * math.cos(a),
                         cz + ring_r * math.sin(a)))
    layout_plaza._last_loop = loop_pts
    layout_plaza._plaza_radius = plaza_radius

    print(f"  [plaza] buildings={len(slots)} inner_r={inner_r:.0f} "
          f"outer_r={outer_r:.0f} rings={n_rings}")

    return slots


def _gen_random_roads(area_w, area_d, n_roads, rng):
    """
    Generate random street skeleton: each road goes from an edge point
    toward the interior with a slight bend via a midpoint.
    Returns list of [(x,z), ...] polylines (3 points each, sampled).
    """
    margin = 6.0
    roads = []
    for _ in range(n_roads):
        # Random start on one of 4 edges
        edge = int(rng.integers(0, 4))
        if edge == 0:    # top
            sx, sz = float(rng.uniform(margin, area_w - margin)), margin
        elif edge == 1:  # bottom
            sx, sz = float(rng.uniform(margin, area_w - margin)), area_d - margin
        elif edge == 2:  # left
            sx, sz = margin, float(rng.uniform(margin, area_d - margin))
        else:            # right
            sx, sz = area_w - margin, float(rng.uniform(margin, area_d - margin))
        # End toward interior or opposite side
        ex = float(rng.uniform(area_w * 0.2, area_w * 0.8))
        ez = float(rng.uniform(area_d * 0.2, area_d * 0.8))
        # Midpoint with slight bend
        mx = (sx + ex) / 2 + float(rng.uniform(-8, 8))
        mz = (sz + ez) / 2 + float(rng.uniform(-8, 8))
        # Sample the polyline at ~2m intervals
        pts = []
        for seg_start, seg_end in [((sx, sz), (mx, mz)), ((mx, mz), (ex, ez))]:
            dx = seg_end[0] - seg_start[0]
            dz = seg_end[1] - seg_start[1]
            seg_len = math.hypot(dx, dz)
            n_pts = max(2, int(seg_len / 2.0))
            for k in range(n_pts + 1):
                t = k / n_pts
                pts.append((seg_start[0] + dx * t, seg_start[1] + dz * t))
        roads.append(pts)
    return roads


def layout_random(n: int, area_w: float, area_d: float,
                  style: str, variation: float,
                  rng: np.random.Generator,
                  gap: float = 3.0,
                  max_tries: int = 200) -> list[BuildingSlot]:
    """
    Novigrad-style: generate random street skeleton, place buildings along roads.
    """
    n_roads = int(rng.integers(3, 6))
    roads = _gen_random_roads(area_w, area_d, n_roads, rng)
    layout_random._last_roads = roads

    # Sample placement points along roads
    road_samples = _sample_along_roads(roads, interval=10.0, rng=rng)
    rng.shuffle(road_samples)

    slots: list[BuildingSlot] = []
    placed: list = []  # OBB polygons
    margin = 4.0
    idx = 0

    _last_side = 1
    for i, (px, pz, ri, tx, tz) in enumerate(road_samples):
        if idx >= n:
            break
        # Probabilistic side: 70% alternate, 30% same side
        if rng.random() < 0.7:
            _last_side = -_last_side
        side = _last_side
        # Angular jitter: rotate perpendicular by ±15 degrees
        perp_x, perp_z = -tz, tx
        jitter_rad = float(rng.uniform(-0.26, 0.26))  # ±15 deg
        cos_j, sin_j = math.cos(jitter_rad), math.sin(jitter_rad)
        perp_x, perp_z = (perp_x * cos_j - perp_z * sin_j,
                          perp_x * sin_j + perp_z * cos_j)
        offset = float(rng.uniform(6.0, 12.0))
        bx_c = px + perp_x * offset * side
        bz_c = pz + perp_z * offset * side

        if bx_c < margin or bz_c < margin or \
           bx_c > area_w - margin or bz_c > area_d - margin:
            continue

        # Zone density: probabilistically skip peripheral placements
        if rng.random() > _zone_density_factor(bx_c, bz_c, area_w, area_d):
            continue

        # Role: first 2 are sub_anchor, rest are filler
        role = "sub_anchor" if idx < 2 else "filler"
        w, d = _building_size(style, variation, rng, role=role)
        yaw = 0.0  # Phase 4 sets final yaw

        if _obb_collides(bx_c, bz_c, w, d, yaw, placed, MIN_BUILDING_GAP):
            continue
        if not check_density(bx_c, bz_c, w, d, slots, area_w, area_d):
            continue

        placed.append(_obb_polygon(bx_c, bz_c, w, d, yaw))
        fp = _pick_footprint_type(variation, rng)
        slots.append(BuildingSlot(bx_c - w / 2, bz_c - d / 2, w, d, fp,
                                  yaw_deg=yaw))
        idx += 1

    # Fill remaining: retry with wider road offsets (no grid fallback)
    if idx < n:
        road_samples2 = _sample_along_roads(roads, interval=6.0, rng=rng)
        rng.shuffle(road_samples2)
        for px, pz, ri, tx, tz in road_samples2:
            if idx >= n:
                break
            perp_x, perp_z = -tz, tx
            jitter_rad = float(rng.uniform(-0.4, 0.4))
            cos_j, sin_j = math.cos(jitter_rad), math.sin(jitter_rad)
            perp_x, perp_z = (perp_x * cos_j - perp_z * sin_j,
                              perp_x * sin_j + perp_z * cos_j)
            side = 1 if rng.random() < 0.5 else -1
            offset = float(rng.uniform(12.0, 20.0))
            bx_c = px + perp_x * offset * side
            bz_c = pz + perp_z * offset * side
            if bx_c < margin or bz_c < margin or \
               bx_c > area_w - margin or bz_c > area_d - margin:
                continue
            w, d = _building_size(style, variation, rng)
            if _obb_collides(bx_c, bz_c, w, d, 0, placed, MIN_BUILDING_GAP):
                continue
            if not check_density(bx_c, bz_c, w, d, slots, area_w, area_d):
                continue
            placed.append(_obb_polygon(bx_c, bz_c, w, d, 0))
            fp = _pick_footprint_type(variation, rng)
            slots.append(BuildingSlot(bx_c - w / 2, bz_c - d / 2, w, d, fp,
                                      yaw_deg=0.0))
            idx += 1

    print(f"  [random] roads={n_roads} buildings={len(slots)} "
          f"(samples={len(road_samples)})")
    return slots


# ─── Organic 布局辅助函数 ────────────────────────────────────────

def _center_yaw(bx: float, bz: float, cx: float, cz: float,
                variation: float, rng: np.random.Generator) -> float:
    """建筑基准朝向：面向场景中心，叠加 ±45°×variation 随机偏转。"""
    base = math.degrees(math.atan2(cz - bz, cx - bx))
    return base + rng.uniform(-45.0, 45.0) * variation


def _try_place_ring(
    slots: list, placed: list, cx: float, cz: float,
    r_min: float, r_max: float, gap: float,
    area_w: float, area_d: float, margin: float,
    style: str, variation: float, rng: np.random.Generator,
    max_tries: int = 250,
) -> bool:
    """在环形区域内随机放置一栋建筑，返回是否成功。"""
    for _ in range(max_tries):
        angle = rng.uniform(0.0, 2 * math.pi)
        r     = rng.uniform(r_min, r_max)
        w, d  = _building_size(style, variation, rng)
        bx = cx + r * math.cos(angle) - w / 2
        bz = cz + r * math.sin(angle) - d / 2
        bx = max(margin, min(area_w - w - margin, bx))
        bz = max(margin, min(area_d - d - margin, bz))
        yaw = _center_yaw(bx + w/2, bz + d/2, cx, cz, variation, rng)
        safety = math.hypot(w, d) / 2 + SAFETY_GAP
        if not _obb_collides(bx + w/2, bz + d/2, w, d, yaw, placed, safety):
            placed.append(_obb_polygon(bx + w/2, bz + d/2, w, d, yaw))
            fp = _pick_footprint_type(variation, rng)
            slots.append(BuildingSlot(bx, bz, w, d, fp, yaw))
            return True
    return False


def _try_place_arc(
    slots: list, placed: list, cx: float, cz: float,
    angle: float, r_min: float, r_max: float, gap: float,
    area_w: float, area_d: float, margin: float,
    style: str, variation: float, rng: np.random.Generator,
    max_tries: int = 200,
) -> bool:
    """在指定角度方向的弧形区域内放置一栋建筑，返回是否成功。"""
    for _ in range(max_tries):
        r    = rng.uniform(r_min, r_max)
        w, d = _building_size(style, variation, rng)
        bx = cx + r * math.cos(angle) - w / 2
        bz = cz + r * math.sin(angle) - d / 2
        bx = max(margin, min(area_w - w - margin, bx))
        bz = max(margin, min(area_d - d - margin, bz))
        yaw = _center_yaw(bx + w/2, bz + d/2, cx, cz, variation, rng)
        safety = math.hypot(w, d) / 2 + SAFETY_GAP
        if not _obb_collides(bx + w/2, bz + d/2, w, d, yaw, placed, safety):
            placed.append(_obb_polygon(bx + w/2, bz + d/2, w, d, yaw))
            fp = _pick_footprint_type(variation, rng)
            slots.append(BuildingSlot(bx, bz, w, d, fp, yaw))
            return True
    return False


def _bezier_curve(p0, p1, p2, p3, n_pts=40):
    """3 次贝塞尔曲线采样，返回 [(x,z), ...]"""
    pts = []
    for i in range(n_pts + 1):
        t = i / n_pts
        u = 1 - t
        x = u**3*p0[0] + 3*u**2*t*p1[0] + 3*u*t**2*p2[0] + t**3*p3[0]
        z = u**3*p0[1] + 3*u**2*t*p1[1] + 3*u*t**2*p2[1] + t**3*p3[1]
        pts.append((x, z))
    return pts


def _gen_village_roads(cx, cz, area_w, area_d, n_roads, rng):
    """
    生成 n_roads 条从中心向外辐射的自然弯曲道路（贝塞尔曲线）。
    返回 list[ list[(x,z)] ]
    """
    margin = 8.0
    roads = []
    base_angles = np.linspace(0, 2 * math.pi, n_roads, endpoint=False)
    base_angles += rng.uniform(0, math.pi / n_roads)

    for angle in base_angles:
        # 起点：中心附近
        r0 = 5.0
        p0 = (cx + r0 * math.cos(angle), cz + r0 * math.sin(angle))

        # 终点：场景边缘
        r3 = min(area_w, area_d) / 2 - margin
        end_angle = angle + rng.uniform(-0.3, 0.3)
        p3 = (cx + r3 * math.cos(end_angle), cz + r3 * math.sin(end_angle))

        # 控制点：制造自然弯曲
        r1 = r3 * 0.35
        r2 = r3 * 0.70
        bend1 = angle + rng.uniform(-0.4, 0.4)
        bend2 = end_angle + rng.uniform(-0.25, 0.25)
        p1 = (cx + r1 * math.cos(bend1), cz + r1 * math.sin(bend1))
        p2 = (cx + r2 * math.cos(bend2), cz + r2 * math.sin(bend2))

        roads.append(_bezier_curve(p0, p1, p2, p3, n_pts=40))

    return roads


def _nearest_road_dir(bx, bz, roads):
    """找到距建筑最近的路段点，返回 (yaw_deg, distance)"""
    best_dist = float('inf')
    best_yaw  = 0.0
    for road in roads:
        for i in range(len(road) - 1):
            rx, rz = road[i]
            d = math.hypot(bx - rx, bz - rz)
            if d < best_dist:
                best_dist = d
                # 道路方向向量
                dx = road[i+1][0] - road[i][0]
                dz = road[i+1][1] - road[i][1]
                # 建筑面朝道路 = 垂直于道路方向，朝向道路
                # 法线方向取决于建筑在道路哪一侧
                nx, nz = -dz, dx   # 道路法线
                # 如果建筑在法线反侧，翻转
                to_b = (bx - rx, bz - rz)
                if nx * to_b[0] + nz * to_b[1] < 0:
                    nx, nz = -nx, -nz
                best_yaw = math.degrees(math.atan2(nx, nz))
    return best_yaw, best_dist


def _ml_generate_positions(n: int, area_w: float, area_d: float,
                            seed: int = 42) -> list[dict]:
    """调用 layout_model_w3.pt 生成建筑位置/朝向���"""
    import torch
    from layout_model import load_model, generate_layout, MODEL_W3_PATH, MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODEL_W3_PATH if MODEL_W3_PATH.exists() else MODEL_PATH
    if not model_path.exists():
        return []  # 无模型则降级

    model = load_model(model_path, device)
    rng_ml = np.random.default_rng(seed)
    return generate_layout(model, device, n, temperature=0.75, rng=rng_ml)


def _infer_roads_from_positions(positions, cx, cz, area_w, area_d):
    """
    从 ML 生成的建筑位置推断道路走向：
    1. 对建筑位置做极坐标排序
    2. 按角度扇区分组
    3. 每组的中心连线 → 贝塞尔曲线
    """
    if len(positions) < 3:
        return []

    # 建筑极坐标
    polar = []
    for b in positions:
        bx = b["nx"] + area_w / 2
        bz = b["ny"] + area_d / 2
        angle = math.atan2(bz - cz, bx - cx)
        dist  = math.hypot(bx - cx, bz - cz)
        polar.append((angle, dist, bx, bz))
    polar.sort(key=lambda p: p[0])

    # 分成 2~3 个扇区，每个扇区生成一条路
    n_roads = min(3, max(1, len(polar) // 4))
    roads = []
    sector_size = len(polar) // n_roads

    for si in range(n_roads):
        sector = polar[si * sector_size: (si + 1) * sector_size]
        if len(sector) < 2:
            continue

        # 路径：中心 → 按距离排序的建筑中间穿过
        sector.sort(key=lambda p: p[1])
        # 起点
        p0 = (cx, cz)
        # 终点：最远建筑再延伸一段
        far = sector[-1]
        ext_angle = far[0]
        ext_r = far[1] + 10
        p3 = (cx + ext_r * math.cos(ext_angle), cz + ext_r * math.sin(ext_angle))
        # 控制点：沿扇区中位建筑方向弯曲
        mid = sector[len(sector) // 2]
        p1 = (cx + mid[1] * 0.4 * math.cos(mid[0]),
              cz + mid[1] * 0.4 * math.sin(mid[0]))
        p2 = (cx + far[1] * 0.7 * math.cos((mid[0] + far[0]) / 2),
              cz + far[1] * 0.7 * math.sin((mid[0] + far[0]) / 2))

        roads.append(_bezier_curve(p0, p1, p2, p3, n_pts=40))

    return roads


def _sample_along_roads(roads, interval=15.0, rng=None):
    """
    Sample points along road centerlines at given interval with jitter.
    Each sample's threshold is interval * uniform(0.7, 1.3) for natural spacing.
    Returns [(x, z, road_idx, tangent_x, tangent_z), ...].
    """
    samples = []
    for ri, road in enumerate(roads):
        dist_acc = 0.0
        jittered = interval * (rng.uniform(0.7, 1.3) if rng is not None
                               else 1.0)
        for i in range(len(road) - 1):
            dx = road[i+1][0] - road[i][0]
            dz = road[i+1][1] - road[i][1]
            seg_len = math.hypot(dx, dz)
            dist_acc += seg_len
            if dist_acc >= jittered and seg_len > 0.1:
                dist_acc = 0.0
                # New independent jitter for next interval
                jittered = interval * (rng.uniform(0.7, 1.3) if rng is not None
                                       else 1.0)
                tx, tz = dx / seg_len, dz / seg_len
                samples.append((road[i][0], road[i][1], ri, tx, tz))
    return samples


def layout_organic(
    n: int, area_w: float, area_d: float,
    style: str, variation: float,
    rng: np.random.Generator,
) -> list[BuildingSlot]:
    """
    有机布局（路径优先）：
    1. 先生成贝塞尔道路骨架
    2. 中心放锚点建筑
    3. 沿道路两侧交替放置建筑（距道路 8~15m）
    4. 密度检查确保不扎堆
    """
    cx, cz = area_w / 2, area_d / 2
    margin = 4.0

    # ── 1. 生成道路骨架 ──
    n_roads = min(4, max(3, (n + 2) // 3))
    roads = _gen_village_roads(cx, cz, area_w, area_d, n_roads, rng)
    layout_organic._last_roads = roads

    placed: list = []
    slots: list[BuildingSlot] = []

    # ── 2. 中心锚点 ──
    wmin, wmax, dmin, dmax = _STYLE_SIZE.get(style, _DEFAULT_SIZE)
    aw = max(wmax * 1.15, 14.0)
    ad = max(dmax * 1.15, 11.0)
    ax, az = cx - aw / 2, cz - ad / 2
    placed.append(_obb_polygon(cx, cz, aw, ad, 0.0))
    slots.append(BuildingSlot(ax, az, aw, ad, "rect", 0.0, is_anchor=True))

    # ── 3. 沿道路两侧放置建筑 ──
    budget = n - 1
    road_samples = _sample_along_roads(roads, interval=8.0, rng=rng)
    rng.shuffle(road_samples)

    _last_side_org = 1
    for i, (px, pz, ri, tx, tz) in enumerate(road_samples):
        if len(slots) - 1 >= budget:
            break
        # Probabilistic side: 70% alternate, 30% same side
        if rng.random() < 0.7:
            _last_side_org = -_last_side_org
        side = _last_side_org
        # Angular jitter: rotate perpendicular by ±15 degrees
        perp_x, perp_z = -tz, tx
        jitter_rad = float(rng.uniform(-0.26, 0.26))  # ±15 deg
        cos_j, sin_j = math.cos(jitter_rad), math.sin(jitter_rad)
        perp_x, perp_z = (perp_x * cos_j - perp_z * sin_j,
                          perp_x * sin_j + perp_z * cos_j)
        offset = rng.uniform(6.0, 12.0)

        bx_c = px + perp_x * offset * side
        bz_c = pz + perp_z * offset * side

        # Bounds check
        if bx_c < margin or bz_c < margin or \
           bx_c > area_w - margin or bz_c > area_d - margin:
            continue

        # Zone density: probabilistically skip peripheral placements
        if rng.random() > _zone_density_factor(bx_c, bz_c, area_w, area_d):
            continue

        # Role: first 2 after anchor are sub_anchor, rest are filler
        placed_count = len(slots) - 1  # exclude anchor
        role = "sub_anchor" if placed_count < 2 else "filler"
        w, d = _building_size(style, variation, rng, role=role)
        yaw = 0.0  # Phase 4 will orient door toward road

        # OBB collision + density
        if _obb_collides(bx_c, bz_c, w, d, yaw, placed, MIN_BUILDING_GAP):
            continue
        if not check_density(bx_c, bz_c, w, d, slots, area_w, area_d):
            continue

        x_off = bx_c - w / 2
        z_off = bz_c - d / 2
        placed.append(_obb_polygon(bx_c, bz_c, w, d, yaw))
        fp = _pick_footprint_type(variation, rng)
        slots.append(BuildingSlot(x_off, z_off, w, d, fp, yaw_deg=yaw))

    print(f"  [organic] roads={n_roads} buildings={len(slots)} "
          f"(budget={n}, samples={len(road_samples)})")

    return slots


# ─── RoadGraph 路网数据结构 ──────────────────────────────────────

@dataclass
class RoadNode:
    """Road network node."""
    id: int
    x: float
    z: float
    ntype: str = "road"        # "road" | "intersection" | "access"
    building_idx: int = -1     # for ntype="access": which building this serves


@dataclass
class RoadEdge:
    """Road network edge."""
    from_id: int
    to_id: int
    road_type: str = "main"    # "main" | "secondary" | "access"
    width: float = 5.0


class RoadGraph:
    """
    Road network data structure.
    Nodes: intersections, road sample points, building access points.
    Edges: road segments, access paths.
    """

    def __init__(self):
        self.nodes: list[RoadNode] = []
        self.edges: list[RoadEdge] = []
        self._next_id: int = 0
        self.renderable: bool = True   # False → skip mesh generation

    def add_node(self, x: float, z: float, ntype: str = "road",
                 building_idx: int = -1) -> int:
        nid = self._next_id
        self._next_id += 1
        self.nodes.append(RoadNode(nid, round(x, 2), round(z, 2),
                                   ntype, building_idx))
        return nid

    def add_edge(self, from_id: int, to_id: int,
                 road_type: str = "main", width: float = 5.0):
        self.edges.append(RoadEdge(from_id, to_id, road_type, width))

    def road_points(self) -> list[tuple[float, float]]:
        """(x, z) for road/intersection/endpoint nodes (excluding access)."""
        return [(n.x, n.z) for n in self.nodes
                if n.ntype in ("road", "intersection", "endpoint")]

    def all_points(self) -> list[tuple[float, float]]:
        return [(n.x, n.z) for n in self.nodes]

    def bfs_connected(self) -> bool:
        """Check if all road+intersection nodes are BFS-reachable."""
        road_ids = {n.id for n in self.nodes
                    if n.ntype in ("road", "intersection", "endpoint")}
        if len(road_ids) <= 1:
            return True
        adj: dict[int, set[int]] = {nid: set() for nid in road_ids}
        for e in self.edges:
            if e.from_id in adj and e.to_id in adj:
                adj[e.from_id].add(e.to_id)
                adj[e.to_id].add(e.from_id)
        start = next(iter(road_ids))
        visited: set[int] = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj[node] - visited)
        return len(visited) == len(road_ids)

    def get_access_edge(self, building_idx: int) -> Optional[RoadEdge]:
        """Find the access edge for a specific building."""
        for e in self.edges:
            if e.road_type == "access":
                n = self.nodes[e.from_id]
                if n.building_idx == building_idx:
                    return e
        return None

    def get_access_direction(self, building_idx: int) -> Optional[float]:
        """
        Direction (degrees) from building toward its road access node.
        raw_angle = atan2(road_z - bld_z, road_x - bld_x).
        """
        edge = self.get_access_edge(building_idx)
        if edge is None:
            return None
        a = self.nodes[edge.from_id]
        b = self.nodes[edge.to_id]
        return math.degrees(math.atan2(b.z - a.z, b.x - a.x))

    def validate(self, clusters: list,
                 max_access_dist: float = 20.0) -> list[str]:
        """Validate constraints C1-C5. Returns issue strings."""
        issues = []
        # C2: every cluster has an access edge
        for cl in clusters:
            if self.get_access_edge(cl["id"]) is None:
                issues.append(f"C2: cluster {cl['id']} has no access edge")
        # C3: BFS connected
        if not self.bfs_connected():
            issues.append("C3: road network not BFS connected")
        # C4: access distance ≤ max
        for e in self.edges:
            if e.road_type == "access":
                n1 = self.nodes[e.from_id]
                n2 = self.nodes[e.to_id]
                d = math.hypot(n1.x - n2.x, n1.z - n2.z)
                if d > max_access_dist:
                    issues.append(
                        f"C4: bld {n1.building_idx} access={d:.1f}m "
                        f"> {max_access_dist}m")
        return issues

    def generate_meshes(self, style: str) -> list:
        """Generate road surface meshes from edges. Returns [] if not renderable."""
        if not self.renderable:
            return []
        gc = _ground_color_for_style(style)
        road_c = [max(0, int(gc[0]*0.75)), max(0, int(gc[1]*0.75)),
                  max(0, int(gc[2]*0.75)), 255]
        access_c = [max(0, int(gc[0]*0.60)), max(0, int(gc[1]*0.60)),
                    max(0, int(gc[2]*0.60)), 255]
        meshes = []

        # Plaza: smooth ring annulus instead of box segments
        if getattr(self, '_plaza_ring', False):
            cx = self._plaza_cx
            cz = self._plaza_cz
            r = self._plaza_ring_r
            w = 4.0
            n_seg = 64
            verts = []
            faces = []
            for k in range(n_seg):
                a = 2 * math.pi * k / n_seg
                ix = cx + (r - w / 2) * math.cos(a)
                iz = cz + (r - w / 2) * math.sin(a)
                ox = cx + (r + w / 2) * math.cos(a)
                oz = cz + (r + w / 2) * math.sin(a)
                verts.append([ix, 0.08, iz])
                verts.append([ox, 0.08, oz])
            for k in range(n_seg):
                j = (k + 1) % n_seg
                v0, v1 = k * 2, k * 2 + 1
                v2, v3 = j * 2 + 1, j * 2
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])
            ring_mesh = trimesh.Trimesh(
                vertices=np.array(verts, dtype=np.float64),
                faces=np.array(faces, dtype=np.int64))
            c = np.array(road_c, dtype=np.uint8)
            ring_mesh.visual.face_colors = np.tile(c, (len(ring_mesh.faces), 1))
            meshes.append(ring_mesh)

        # All edges → road segments (skip plaza main edges, ring handles them)
        skip_plaza_main = getattr(self, '_plaza_ring', False)
        for e in self.edges:
            if skip_plaza_main and e.road_type == "main":
                continue
            n1 = self.nodes[e.from_id]
            n2 = self.nodes[e.to_id]
            color = access_c if e.road_type == "access" else road_c
            width = e.width if e.road_type != "access" else min(e.width, 2.0)
            seg = _make_road_segment((n1.x, n1.z), (n2.x, n2.z),
                                     width, 0.06, color)
            if seg:
                meshes.append(seg)
        return meshes


# ─── 地面 / 道路 / 广场 ──────────────────────────────────────────

def _ground_color_for_style(style: str) -> list:
    """根据风格类别返回 RGBA 地面颜色。"""
    s = style.lower()
    if any(k in s for k in ("medieval", "horror")):
        return [int(0.35*255), int(0.28*255), int(0.20*255), 255]
    if "japanese" in s:
        return [int(0.45*255), int(0.45*255), int(0.40*255), 255]
    if "desert" in s:
        return [int(0.76*255), int(0.65*255), int(0.40*255), 255]
    if any(k in s for k in ("modern", "industrial")):
        return [int(0.50*255), int(0.50*255), int(0.50*255), 255]
    if "fantasy" in s:
        return [int(0.30*255), int(0.28*255), int(0.32*255), 255]
    return [int(0.40*255), int(0.35*255), int(0.28*255), 255]


def _colored_box(extents, center, rgba):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    c = np.array(rgba, dtype=np.uint8)
    m.visual.face_colors = np.tile(c, (len(m.faces), 1))
    return m


def _make_organic_roads(roads: list, style: str, road_w: float = 3.5) -> list:
    """
    从贝塞尔曲线路网生成路面 mesh（一段段 box 铺设）。
    每段 box 沿曲线方���旋转，略高于地面。
    """
    gc = _ground_color_for_style(style)
    road_c = [max(0, int(gc[0]*0.75)), max(0, int(gc[1]*0.75)),
              max(0, int(gc[2]*0.75)), 255]
    meshes = []
    for road in roads:
        for i in range(len(road) - 1):
            x0, z0 = road[i]
            x1, z1 = road[i + 1]
            dx, dz = x1 - x0, z1 - z0
            seg_len = math.hypot(dx, dz)
            if seg_len < 0.1:
                continue
            angle = math.atan2(dz, dx)
            mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
            m = trimesh.creation.box(extents=[seg_len + 0.1, 0.05, road_w])
            rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
            m.apply_transform(rot)
            m.apply_translation([mx, 0.025, mz])
            c = np.array(road_c, dtype=np.uint8)
            m.visual.face_colors = np.tile(c, (len(m.faces), 1))
            meshes.append(m)
    return meshes


def _make_ground_plane(area_w: float, area_d: float,
                       style: str, params: dict) -> trimesh.Trimesh:
    """地面平板：场景 + 10m 边距，0.3m 厚。"""
    margin = 10.0
    gc = _ground_color_for_style(style)
    return _colored_box(
        [area_w + margin * 2, 0.3, area_d + margin * 2],
        [area_w / 2, -0.15, area_d / 2],
        gc)


def _make_street_road(area_w: float, area_d: float,
                      style: str, building_infos=None) -> list:
    """
    street 布局：两排建筑之间的路面。
    如果 building_infos 提供，道路会绕过建筑（留缺口）。
    """
    street_w = STREET_WIDTH
    road_w   = min(6.0, street_w - 2.0)   # road surface = street minus sidewalks
    side_d   = (area_d - street_w) / 2
    road_z   = area_d / 2                  # road centered in scene
    gc = _ground_color_for_style(style)
    road_c = [max(0, int(gc[0]*0.8)), max(0, int(gc[1]*0.8)),
              max(0, int(gc[2]*0.8)), 255]

    # Road X range clamped to wall interior
    wall_inset = 2.0
    x_lo = wall_inset
    x_hi = area_w - wall_inset

    if not building_infos:
        road_len = x_hi - x_lo
        return [_colored_box(
            [road_len, 0.05, road_w],
            [(x_lo + x_hi) / 2, 0.025, road_z],
            road_c)]

    # Collect X-ranges blocked by buildings on the road
    blocked = []
    for b in building_infos:
        bx = b["x"]
        bz = b["z"]
        hw = b.get("w", 8) / 2 + 1.0
        hd = b.get("d", 8) / 2 + 1.0
        # Check if building overlaps the road Z band
        if abs(bz - road_z) < hd + street_w / 2:
            blocked.append((bx - hw, bx + hw))

    # Merge overlapping blocked ranges
    blocked.sort()
    merged = []
    for lo, hi in blocked:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    # Generate road segments in the gaps (clamped to wall interior)
    meshes = []
    x_start = x_lo
    for blo, bhi in merged:
        seg_len = blo - x_start
        if seg_len > 1.0:
            cx = (x_start + blo) / 2
            meshes.append(_colored_box(
                [seg_len, 0.05, road_w],
                [cx, 0.025, road_z],
                road_c))
        x_start = bhi

    # Final segment after last blocked range
    seg_len = x_hi - x_start
    if seg_len > 1.0:
        cx = (x_start + x_hi) / 2
        meshes.append(_colored_box(
            [seg_len, 0.05, street_w],
            [cx, 0.025, road_z],
            road_c))

    return meshes


def _make_plaza_floor(area_w: float, area_d: float,
                      style: str, slots=None) -> list:
    """
    plaza 布局：中心广场石板地面。
    如果 slots 提供，动态计算凸包内圈地面。
    否则使用固定大小。
    """
    stone_c = [int(0.60*255), int(0.58*255), int(0.55*255), 255]

    if slots and len(slots) >= 3:
        # Compute building centers
        centers = [(s.x_off + s.w / 2, s.z_off + s.d / 2) for s in slots]
        cx = sum(x for x, _ in centers) / len(centers)
        cz = sum(z for _, z in centers) / len(centers)
        # Average distance to center
        dists = [math.hypot(x - cx, z - cz) for x, z in centers]
        avg_dist = sum(dists) / len(dists)
        # Floor radius = 70% of avg building distance (covers inner courtyard)
        floor_r = avg_dist * 0.7
        floor_r = max(5.0, min(floor_r, min(area_w, area_d) * 0.4))

        # Generate octagonal floor (approximates circle with 8-sided polygon)
        from shapely.geometry import Point
        circle = Point(cx, cz).buffer(floor_r, resolution=8)
        # Extrude to thin slab
        floor_mesh = trimesh.creation.extrude_polygon(circle, 0.05)
        rot = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        floor_mesh.apply_transform(rot)
        floor_mesh.apply_translation([0, 0.05, 0])
        c = np.array(stone_c, dtype=np.uint8)
        floor_mesh.visual.face_colors = np.tile(c, (len(floor_mesh.faces), 1))
        return [floor_mesh]
    else:
        # Fallback: fixed square
        plaza_r = min(area_w, area_d) * 0.28
        side = plaza_r * 2
        return [_colored_box(
            [side, 0.05, side],
            [area_w / 2, 0.025, area_d / 2],
            stone_c)]


# ─── 围墙 ─────────────────────────────────────────────────────

_WALL_STYLES = {"medieval", "medieval_keep", "medieval_chapel",
                "fantasy", "fantasy_dungeon", "fantasy_palace",
                "horror", "horror_asylum", "horror_crypt"}


def _find_gate_points(area_w: float, area_d: float,
                      road_graph: "RoadGraph", inset: float) -> list:
    """
    Compute gate positions where roads cross the perimeter wall.

    Returns list of dicts:
      {"x": float, "z": float, "wall": "front"|"back"|"left"|"right",
       "road_width": float, "pos_along_wall": float}
    """
    from shapely.geometry import LineString, MultiPoint, Point, box as shapely_box

    wall_rect = shapely_box(inset, inset, area_w - inset, area_d - inset)
    wall_boundary = wall_rect.boundary  # LinearRing

    # Classify which wall a point sits on
    def _classify_wall(x, z):
        eps = 0.5
        if abs(z - inset) < eps:
            return "front", x
        if abs(z - (area_d - inset)) < eps:
            return "back", x
        if abs(x - inset) < eps:
            return "left", z
        if abs(x - (area_w - inset)) < eps:
            return "right", z
        # Fallback: nearest edge
        dists = [("front", abs(z - inset), x),
                 ("back",  abs(z - (area_d - inset)), x),
                 ("left",  abs(x - inset), z),
                 ("right", abs(x - (area_w - inset)), z)]
        dists.sort(key=lambda d: d[1])
        return dists[0][0], dists[0][2]

    gate_points = []
    seen_positions = set()  # deduplicate nearby hits

    for edge in road_graph.edges:
        if edge.road_type == "access":
            continue
        n1 = road_graph.nodes[edge.from_id]
        n2 = road_graph.nodes[edge.to_id]
        road_line = LineString([(n1.x, n1.z), (n2.x, n2.z)])
        intersection = road_line.intersection(wall_boundary)

        pts = []
        if intersection.is_empty:
            continue
        if isinstance(intersection, Point):
            pts = [intersection]
        elif isinstance(intersection, MultiPoint):
            pts = list(intersection.geoms)
        elif hasattr(intersection, 'geoms'):
            for g in intersection.geoms:
                if isinstance(g, Point):
                    pts.append(g)

        for pt in pts:
            px, pz = pt.x, pt.y
            wall_name, pos_along = _classify_wall(px, pz)
            # Deduplicate: skip if another gate within 3m on same wall
            key = (wall_name, round(pos_along / 3.0))
            if key in seen_positions:
                continue
            seen_positions.add(key)
            gate_points.append({
                "x": px, "z": pz,
                "wall": wall_name,
                "road_width": edge.width,
                "pos_along_wall": pos_along,
            })

    return gate_points


def _merge_gates_on_wall(gates: list, min_merge_dist: float = 10.0) -> list:
    """
    Merge gate points on the same wall that are closer than min_merge_dist.
    Returns list of {"center": float, "width": float} per merged gate.
    """
    if not gates:
        return []
    gates = sorted(gates, key=lambda g: g["pos_along_wall"])
    merged = []
    cur = {"center": gates[0]["pos_along_wall"],
           "width": max(gates[0]["road_width"] + 2.0, 6.0),
           "lo": gates[0]["pos_along_wall"],
           "hi": gates[0]["pos_along_wall"]}
    for g in gates[1:]:
        pos = g["pos_along_wall"]
        gw = max(g["road_width"] + 2.0, 6.0)
        if pos - cur["hi"] < min_merge_dist:
            # merge
            cur["hi"] = pos
            cur["center"] = (cur["lo"] + cur["hi"]) / 2
            cur["width"] = max(cur["width"], cur["hi"] - cur["lo"] + gw)
        else:
            merged.append({"center": cur["center"], "width": cur["width"]})
            cur = {"center": pos, "width": gw, "lo": pos, "hi": pos}
    merged.append({"center": cur["center"], "width": cur["width"]})
    return merged


def _build_wall_with_gates(wall_start: float, wall_end: float,
                           gates: list, wall_h: float, wall_t: float,
                           color: list, wall_name: str,
                           inset: float, area_w: float, area_d: float) -> list:
    """
    Build wall segments and gate pillars along one wall edge.

    wall_start/wall_end: position range along the wall's axis
    gates: list of {"center": float, "width": float} on this wall
    Returns: list of trimesh meshes
    """
    meshes = []
    pillar_w = 0.8
    pillar_h = wall_h + 1.0

    # Sort gates by center position
    gates = sorted(gates, key=lambda g: g["center"])

    # Clamp gates to wall range
    for g in gates:
        half = g["width"] / 2
        g["lo"] = max(wall_start, g["center"] - half)
        g["hi"] = min(wall_end, g["center"] + half)

    # Build wall segments between gates
    seg_starts = [wall_start]
    for g in gates:
        seg_starts.append(g["hi"])
    seg_ends = []
    for g in gates:
        seg_ends.append(g["lo"])
    seg_ends.append(wall_end)

    for s, e in zip(seg_starts, seg_ends):
        seg_len = e - s
        if seg_len < 0.5:
            continue
        mid = (s + e) / 2
        if wall_name in ("front", "back"):
            z_pos = inset if wall_name == "front" else area_d - inset
            meshes.append(_colored_box(
                [seg_len, wall_h, wall_t],
                [mid, wall_h / 2, z_pos], color))
        else:  # left or right
            x_pos = inset if wall_name == "left" else area_w - inset
            meshes.append(_colored_box(
                [wall_t, wall_h, seg_len],
                [x_pos, wall_h / 2, mid], color))

    # Gate pillars at each gate opening edge
    for g in gates:
        for edge_pos in [g["lo"], g["hi"]]:
            if wall_name in ("front", "back"):
                z_pos = inset if wall_name == "front" else area_d - inset
                meshes.append(_colored_box(
                    [pillar_w, pillar_h, pillar_w],
                    [edge_pos, pillar_h / 2, z_pos], color))
            else:
                x_pos = inset if wall_name == "left" else area_w - inset
                meshes.append(_colored_box(
                    [pillar_w, pillar_h, pillar_w],
                    [x_pos, pillar_h / 2, edge_pos], color))

    return meshes


def _make_perimeter_wall(area_w: float, area_d: float,
                         style: str, params: dict,
                         road_graph: "RoadGraph" = None,
                         wall_h: float = 2.5,
                         wall_t: float = 0.5) -> list:
    """
    Generate perimeter walls with gates aligned to road crossings.

    If road_graph is provided, gate positions are computed from
    road-wall intersections. Otherwise falls back to a single
    centered gate on the front wall (backward compatible).
    """
    if style not in _WALL_STYLES:
        return []

    h, t = wall_h, wall_t
    inset = 2.0
    palette = _derive_palette(style, params)
    wc = palette["wall"]
    c = [max(0, int(wc[0]*0.9)), max(0, int(wc[1]*0.9)),
         max(0, int(wc[2]*0.9)), wc[3]]

    # Wall axis ranges (along the wall)
    wall_defs = {
        "front": (inset, area_w - inset),    # x range
        "back":  (inset, area_w - inset),    # x range
        "left":  (inset, area_d - inset),    # z range
        "right": (inset, area_d - inset),    # z range
    }

    # Compute gates from road network
    if road_graph is not None:
        gate_pts = _find_gate_points(area_w, area_d, road_graph, inset)
    else:
        gate_pts = []

    # Group by wall
    gates_by_wall = {"front": [], "back": [], "left": [], "right": []}
    for gp in gate_pts:
        gates_by_wall[gp["wall"]].append(gp)

    # Merge nearby gates on each wall
    merged_gates = {}
    for wall_name, gpts in gates_by_wall.items():
        merged_gates[wall_name] = _merge_gates_on_wall(gpts)

    # Fallback: if no gates at all, place one centered on front wall
    if not any(merged_gates.values()):
        gate_w = 6.0
        merged_gates["front"] = [{"center": area_w / 2, "width": gate_w}]

    # Build all 4 walls with their respective gates
    meshes = []
    for wall_name, (ws, we) in wall_defs.items():
        meshes.extend(_build_wall_with_gates(
            ws, we, merged_gates.get(wall_name, []),
            h, t, c, wall_name, inset, area_w, area_d))

    return meshes


# ─── 路灯 / 火把柱 ──────────────────────────────────────────────

def _make_street_lamps(area_w: float, area_d: float,
                       style: str, params: dict,
                       layout_type: str) -> list:
    """沿街道/场景边缘每隔 8m 放一根灯柱。"""
    if layout_type not in ("street", "plaza", "grid"):
        return []

    palette = _derive_palette(style, params)
    wc = palette["wall"]
    # 灯柱颜色：比墙色深 30%
    c = [max(0, int(wc[0]*0.7)), max(0, int(wc[1]*0.7)),
         max(0, int(wc[2]*0.7)), wc[3]]

    is_medieval = any(k in style for k in ("medieval", "fantasy", "horror"))
    pole_h = 3.0
    pole_w = 0.15 if not is_medieval else 0.20
    top_w  = 0.10 if not is_medieval else 0.30
    spacing = 8.0

    meshes = []

    if layout_type == "street":
        road_cz = area_d / 2
        # 路灯沿街道两侧
        for z_line in [road_cz - STREET_WIDTH / 2 + 0.5,
                        road_cz + STREET_WIDTH / 2 - 0.5]:
            x = spacing / 2
            while x < area_w - 1:
                # 柱身
                meshes.append(_colored_box(
                    [pole_w, pole_h, pole_w],
                    [x, pole_h / 2, z_line], c))
                # 顶部（medieval 方块 / modern 小球近似）
                meshes.append(_colored_box(
                    [top_w, 0.25, top_w],
                    [x, pole_h + 0.125, z_line], c))
                x += spacing
    else:
        # grid/plaza：沿场景边缘放灯
        for edge_pts in [
            [(x, 3.0) for x in np.arange(spacing/2, area_w, spacing)],
            [(x, area_d - 3.0) for x in np.arange(spacing/2, area_w, spacing)],
            [(3.0, z) for z in np.arange(spacing/2, area_d, spacing)],
            [(area_w - 3.0, z) for z in np.arange(spacing/2, area_d, spacing)],
        ]:
            for px, pz in edge_pts:
                meshes.append(_colored_box(
                    [pole_w, pole_h, pole_w],
                    [px, pole_h / 2, pz], c))
                meshes.append(_colored_box(
                    [top_w, 0.25, top_w],
                    [px, pole_h + 0.125, pz], c))

    return meshes


# ─── 连通性保证系统 ────────────────────────────────────────────

def build_adjacency_graph(buildings):
    """
    建筑列表 → 邻近图。
    buildings: [{"x": center_x, "z": center_z, "w": width, "d": depth, "height": h}, ...]
    坐标系: X=东, Y=上, Z=南 → 水平距离用 XZ 平面。
    """
    n = len(buildings)
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = buildings[i], buildings[j]
            dist = math.hypot(a["x"] - b["x"], a["z"] - b["z"])
            gap = dist - (a.get("w", 8) / 2 + b.get("w", 8) / 2)
            if gap < 8.0:
                graph[i].append(j)
                graph[j].append(i)
    return graph


def find_components(graph):
    """BFS 找所有连通分量。"""
    visited = set()
    components = []
    for start in graph:
        if start in visited:
            continue
        comp = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            queue.extend(graph[node])
        components.append(comp)
    return components


def connect_components_mst(buildings, components):
    """用 MST 思路把孤立分量连回主图，返回 [(i, j, dist), ...]。"""
    if len(components) <= 1:
        return []
    components = sorted(components, key=len, reverse=True)
    merged = set(components[0])
    edges = []
    for comp in components[1:]:
        best_dist = float("inf")
        best_pair = None
        for i in merged:
            for j in comp:
                a, b = buildings[i], buildings[j]
                d = math.hypot(a["x"] - b["x"], a["z"] - b["z"])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)
        if best_pair:
            ctype = "corridor" if best_dist < 15 else "path"
            print(f"  MST edge: {best_pair[0]}->{best_pair[1]}, "
                  f"dist={best_dist:.1f}m, type={ctype}")
            edges.append((best_pair[0], best_pair[1], best_dist))
            merged |= comp
    return edges


def _make_single_segment(ax, az, bx, bz, corridor_h, corridor_w, is_path, palette_wall):
    """Build one corridor/path segment between two XZ points."""
    dx, dz = bx - ax, bz - az
    length = math.hypot(dx, dz)
    if length < 0.2:
        return None
    angle = math.atan2(dz, dx)
    mx, mz = (ax + bx) / 2, (az + bz) / 2

    if is_path:
        m = trimesh.creation.box(extents=[length, 0.3, 2.5])
        m.apply_translation([0, 0.15, 0])
        c = np.array([80, 60, 40, 255], dtype=np.uint8)
    else:
        m = trimesh.creation.box(extents=[length, corridor_h, corridor_w])
        m.apply_translation([0, corridor_h / 2, 0])
        c = np.array(palette_wall, dtype=np.uint8)

    rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
    m.apply_transform(rot)
    m.apply_translation([mx, 0, mz])
    m.visual.face_colors = np.tile(c, (len(m.faces), 1))
    return m


def make_corridor(b_a, b_b, palette_wall, style="medieval", door_w=1.2,
                  avoid_center=None, avoid_radius=0.0):
    """
    Generate corridor/path between two buildings.
    avoid_center: (cx, cz) — if the straight line crosses this zone, route around it.
    avoid_radius: radius of the zone to avoid.
    """
    ax, az = b_a["x"], b_a["z"]
    bx, bz = b_b["x"], b_b["z"]
    dx, dz = bx - ax, bz - az
    center_dist = math.hypot(dx, dz)
    if center_dist < 0.1:
        return None

    ux, uz = dx / center_dist, dz / center_dist
    offset_a = max(b_a.get("w", 8), b_a.get("d", 8)) / 2
    offset_b = max(b_b.get("w", 8), b_b.get("d", 8)) / 2
    corridor_len = center_dist - offset_a - offset_b
    if corridor_len <= 0.1:
        return None

    # Edge points (start/end at building walls)
    ex_a = (ax + ux * offset_a, az + uz * offset_a)
    ex_b = (bx - ux * offset_b, bz - uz * offset_b)

    is_path = corridor_len >= 15.0
    corridor_w = max(door_w, 1.2)
    corridor_h = min(b_a.get("height", 5), b_b.get("height", 5)) * 0.6
    corridor_h = max(2.5, corridor_h)

    # Check if straight line crosses the avoidance zone (plaza center)
    needs_detour = False
    if avoid_center and avoid_radius > 2:
        acx, acz = avoid_center
        # Midpoint of corridor
        mid_x = (ex_a[0] + ex_b[0]) / 2
        mid_z = (ex_a[1] + ex_b[1]) / 2
        dist_to_center = math.hypot(mid_x - acx, mid_z - acz)
        if dist_to_center < avoid_radius:
            needs_detour = True

    if needs_detour:
        # Route around: go via a waypoint on the edge of the avoidance circle
        acx, acz = avoid_center
        # Angle from center to midpoint → perpendicular detour
        mid_angle = math.atan2((ex_a[1]+ex_b[1])/2 - acz,
                               (ex_a[0]+ex_b[0])/2 - acx)
        # Pick the side that's further from center
        perp1 = (acx + (avoid_radius + 3) * math.cos(mid_angle + 1.2),
                 acz + (avoid_radius + 3) * math.sin(mid_angle + 1.2))
        perp2 = (acx + (avoid_radius + 3) * math.cos(mid_angle - 1.2),
                 acz + (avoid_radius + 3) * math.sin(mid_angle - 1.2))
        # Pick the waypoint closest to both endpoints
        d1 = math.hypot(perp1[0]-ex_a[0], perp1[1]-ex_a[1]) + \
             math.hypot(perp1[0]-ex_b[0], perp1[1]-ex_b[1])
        d2 = math.hypot(perp2[0]-ex_a[0], perp2[1]-ex_a[1]) + \
             math.hypot(perp2[0]-ex_b[0], perp2[1]-ex_b[1])
        wp = perp1 if d1 < d2 else perp2

        # Two segments: A→waypoint, waypoint→B
        meshes = []
        seg1 = _make_single_segment(ex_a[0], ex_a[1], wp[0], wp[1],
                                     corridor_h, corridor_w, is_path, palette_wall)
        seg2 = _make_single_segment(wp[0], wp[1], ex_b[0], ex_b[1],
                                     corridor_h, corridor_w, is_path, palette_wall)
        if seg1 and seg2:
            combined = trimesh.util.concatenate([seg1, seg2])
            return combined
        return seg1 or seg2

    # Straight corridor
    return _make_single_segment(ex_a[0], ex_a[1], ex_b[0], ex_b[1],
                                corridor_h, corridor_w, is_path, palette_wall)


# ─── 多区域合并 ────────────────────────────────────────────────

def merge_zones(scenes: list) -> trimesh.Scene:
    """
    将多个 trimesh.Scene 水平拼合，按区域自动排列。
    每个 zone 沿 X 轴偏移，间距根据各自 bounding box 计算。
    """
    merged = trimesh.Scene()
    x_offset = 0.0
    ZONE_GAP = 20.0  # 区域之间的间距（米）

    for scene in scenes:
        # 计算当前 scene 的 bounding box
        all_verts = []
        for geom in scene.geometry.values():
            all_verts.append(geom.vertices)
        if not all_verts:
            continue

        verts = np.vstack(all_verts)
        min_x = verts[:, 0].min()
        max_x = verts[:, 0].max()
        width = max_x - min_x

        # 平移所有 mesh 到正确位置
        shift = x_offset - min_x
        for name, geom in scene.geometry.items():
            new_geom = geom.copy()
            new_geom.vertices[:, 0] += shift
            # 避免 mesh 名称冲突
            unique_name = f"{name}_{id(scene)}"
            merged.add_geometry(new_geom, geom_name=unique_name)

        x_offset += width + ZONE_GAP

    return merged




# ─── 路网工具函数 ─────────────────────────────────────────────────

def _segment_avoids_buildings(p1, p2, building_infos, margin=1.5):
    """
    Check if line segment p1→p2 intersects any building bbox.
    If so, return a detour path [p1, waypoint, p2].
    Otherwise return [p1, p2].
    """
    for b in building_infos:
        bx, bz = b["x"], b["z"]
        hw = b.get("w", 8) / 2 + margin
        hd = b.get("d", 8) / 2 + margin
        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]
        seg_len = math.hypot(dx, dz)
        if seg_len < 0.1:
            continue
        for t in [0.25, 0.5, 0.75]:
            px = p1[0] + dx * t
            pz = p1[1] + dz * t
            if abs(px - bx) < hw and abs(pz - bz) < hd:
                corners = [
                    (bx - hw, bz - hd), (bx + hw, bz - hd),
                    (bx + hw, bz + hd), (bx - hw, bz + hd),
                ]
                mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                best = min(corners,
                           key=lambda c: math.hypot(c[0]-mid[0], c[1]-mid[1]))
                return [p1, best, p2]
    return [p1, p2]


def _make_road_segment(p1, p2, width, height, color):
    """Generate a single road segment mesh between two XZ points."""
    dx = p2[0] - p1[0]
    dz = p2[1] - p1[1]
    length = math.hypot(dx, dz)
    if length < 0.3:
        return None
    angle = math.atan2(dz, dx)
    mx, mz = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
    m = trimesh.creation.box(extents=[length, height, width])
    m.apply_translation([0, height/2, 0])
    rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
    m.apply_transform(rot)
    m.apply_translation([mx, 0, mz])
    c = np.array(color, dtype=np.uint8)
    m.visual.face_colors = np.tile(c, (len(m.faces), 1))
    return m


def _node_inside_building(x, z, building_infos, margin=1.5):
    """Check if point (x,z) is inside any building bbox + margin."""
    for b in building_infos:
        if (abs(x - b["x"]) < b.get("w", 8) / 2 + margin and
            abs(z - b["z"]) < b.get("d", 8) / 2 + margin):
            return True
    return False


def _add_road_chain(graph: "RoadGraph", points: list, building_infos: list,
                    road_type: str = "main", width: float = 5.0,
                    margin: float = 1.5, keep_connected: bool = False):
    """
    Add a chain of road nodes + edges from a list of (x,z) points.
    Skips nodes that fall inside buildings (C1).
    keep_connected=True: skip nodes but don't break chain (for loops/organic).
    keep_connected=False: break chain at building intersections (for streets/grids).
    """
    prev_id = None
    for x, z in points:
        if _node_inside_building(x, z, building_infos, margin):
            if not keep_connected:
                prev_id = None  # break the chain
            continue
        nid = graph.add_node(x, z, "road")
        if prev_id is not None:
            graph.add_edge(prev_id, nid, road_type, width)
        prev_id = nid


def _add_cluster_access_edges(graph: "RoadGraph", clusters: list,
                              building_infos: list):
    """
    For each cluster, add ONE access edge from main_building door point
    to nearest road node. Access node stores cluster_id (not building_idx).
    """
    road_pts = graph.road_points()
    if not road_pts:
        return
    road_arr = np.array(road_pts)
    road_node_ids = [n.id for n in graph.nodes
                     if n.ntype in ("road", "intersection")]

    for cl in clusters:
        main_idx = cl["main_building_idx"]
        mb = building_infos[main_idx]
        bx, bz = mb["x"], mb["z"]

        dists = np.sqrt((road_arr[:, 0] - bx) ** 2 +
                        (road_arr[:, 1] - bz) ** 2)
        sorted_idx = np.argsort(dists)

        half_size = max(mb.get("w", 8), mb.get("d", 8)) / 2
        best_road_nid = None

        for rank in range(min(20, len(sorted_idx))):
            ri = int(sorted_idx[rank])
            d = float(dists[ri])
            if d < half_size * 0.3:
                continue
            best_road_nid = road_node_ids[ri]
            break

        if best_road_nid is None and len(sorted_idx) > 0:
            best_road_nid = road_node_ids[int(sorted_idx[0])]

        if best_road_nid is not None:
            # Access node at main building center, storing cluster id
            access_nid = graph.add_node(bx, bz, "access",
                                        building_idx=cl["id"])
            graph.add_edge(access_nid, best_road_nid, "access", 2.0)
            # Update cluster door point
            rn = graph.nodes[best_road_nid]
            cl["main_building_door"] = [rn.x, rn.z]


def _delaunay_mst_edges(points):
    """
    Compute MST edges from a set of 2D points via Delaunay triangulation.
    Returns list of (i, j) index pairs.
    """
    from scipy.spatial import Delaunay
    if len(points) < 2:
        return []
    pts = np.array(points)
    if len(pts) < 3:
        return [(0, 1)]
    try:
        tri = Delaunay(pts)
    except Exception:
        return [(0, 1)]
    # Collect all Delaunay edges with distances
    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in [(0,1), (1,2), (0,2)]:
            i, j = int(simplex[a]), int(simplex[b])
            if i > j:
                i, j = j, i
            edge_set.add((i, j))
    edges = [(i, j, float(np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])))
             for i, j in edge_set]
    edges.sort(key=lambda e: e[2])
    # Kruskal MST
    parent = list(range(len(pts)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    mst = []
    for i, j, d in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            mst.append((i, j))
            if len(mst) == len(pts) - 1:
                break
    return mst


def generate_road_network(clusters: list, building_infos: list,
                          area_w: float, area_d: float,
                          layout_type: str, style: str) -> "RoadGraph":
    """
    Build a RoadGraph connecting clusters (not individual buildings).
    Phase 1: road skeleton (layout-specific)
    Phase 2: ONE access edge per cluster (main_building → road)
    """
    graph = RoadGraph()
    road_w = 5.0

    # random/organic: keep graph data for pathfinding but don't render roads
    if layout_type in ("random", "organic"):
        graph.renderable = False

    # Wall inset for clamping road endpoints
    _inset = 2.0

    if layout_type == "street":
        # Street: two endpoints clamped to wall interior + one main edge.
        road_z = area_d / 2
        n0 = graph.add_node(_inset, road_z, "endpoint")
        n1 = graph.add_node(area_w - _inset, road_z, "endpoint")
        graph.add_edge(n0, n1, "main", STREET_WIDTH)

    elif layout_type == "grid":
        cell_w = getattr(layout_grid, '_cell_w', area_w / 4)
        cell_d = getattr(layout_grid, '_cell_d', area_d / 3)
        cols = getattr(layout_grid, '_cols', 4)
        rows = getattr(layout_grid, '_rows', 3)
        road_xs = [c * cell_w for c in range(cols + 1)]
        road_zs = [r * cell_d for r in range(rows + 1)]
        # Clamp grid road nodes to wall interior
        road_xs = [max(_inset, min(area_w - _inset, x)) for x in road_xs]
        road_zs = [max(_inset, min(area_d - _inset, z)) for z in road_zs]
        grid_nids: dict[tuple[int, int], Optional[int]] = {}
        for xi, x in enumerate(road_xs):
            for zi, z in enumerate(road_zs):
                nid = graph.add_node(x, z, "intersection")
                grid_nids[(xi, zi)] = nid
        for xi in range(len(road_xs)):
            for zi in range(len(road_zs)):
                nid = grid_nids[(xi, zi)]
                if xi + 1 < len(road_xs):
                    graph.add_edge(nid, grid_nids[(xi+1, zi)],
                                   "main", road_w * 0.7)
                if zi + 1 < len(road_zs):
                    graph.add_edge(nid, grid_nids[(xi, zi+1)],
                                   "main", road_w * 0.7)

    elif layout_type == "plaza":
        # Smooth ring road: evenly spaced nodes on a circle, all connected
        if hasattr(layout_plaza, '_last_loop') and layout_plaza._last_loop:
            plaza_r = getattr(layout_plaza, '_plaza_radius',
                              min(area_w, area_d) * 0.15)
            ring_r = plaza_r + MIN_BUILDING_GAP + 2.0
            cx, cz = area_w / 2, area_d / 2
            n_seg = 32
            ring_nids = []
            for k in range(n_seg):
                a = 2 * math.pi * k / n_seg
                rx = cx + ring_r * math.cos(a)
                rz = cz + ring_r * math.sin(a)
                nid = graph.add_node(rx, rz, "road")
                ring_nids.append(nid)
            # Connect ring
            for k in range(n_seg):
                graph.add_edge(ring_nids[k],
                               ring_nids[(k + 1) % n_seg], "main", 4.0)
            # Store ring for custom mesh generation
            graph._plaza_ring = True
            graph._plaza_cx = cx
            graph._plaza_cz = cz
            graph._plaza_ring_r = ring_r

    elif layout_type == "organic":
        if hasattr(layout_organic, '_last_roads'):
            first_nids = []
            for road in layout_organic._last_roads:
                n_before = len(graph.nodes)
                _add_road_chain(graph, road, building_infos, "main", 3.5,
                                margin=0.5, keep_connected=True)
                for nd in graph.nodes[n_before:]:
                    if nd.ntype == "road":
                        first_nids.append(nd.id)
                        break
            for i in range(1, len(first_nids)):
                graph.add_edge(first_nids[0], first_nids[i],
                               "secondary", 3.0)

    elif layout_type == "random":
        # Use stored road skeleton from layout_random
        if hasattr(layout_random, '_last_roads'):
            first_nids = []
            for road in layout_random._last_roads:
                n_before = len(graph.nodes)
                _add_road_chain(graph, road, building_infos, "main", 4.0,
                                margin=0.5, keep_connected=True)
                for nd in graph.nodes[n_before:]:
                    if nd.ntype == "road":
                        first_nids.append(nd.id)
                        break
            # Connect road starts for BFS connectivity (all meet near center)
            for i in range(1, len(first_nids)):
                graph.add_edge(first_nids[0], first_nids[i],
                               "secondary", 3.0)

    # Fallback spine
    if not graph.road_points():
        pts = [(area_w / 2, float(z)) for z in np.arange(0, area_d, 5.0)]
        _add_road_chain(graph, pts, building_infos, "main", road_w)

    # Access edges: street and grid skip (hardcoded yaw / grid-line yaw)
    if layout_type not in ("street", "grid"):
        _add_cluster_access_edges(graph, clusters, building_infos)

    return graph


# ─── Cluster 系统 ────────────────────────────────────────────────

CLUSTER_DISTANCE_THRESHOLD = 12.0  # edge-to-edge < 12m → same cluster


def identify_clusters(building_infos, layout_type="street"):
    """
    BFS 聚类：edge_to_edge 距离 < CLUSTER_DISTANCE_THRESHOLD → 同一 cluster。
    Returns list of cluster dicts.
    """
    n = len(building_infos)
    if n == 0:
        return []

    # Build adjacency by edge-to-edge distance
    adj = {i: [] for i in range(n)}
    for i in range(n):
        a = building_infos[i]
        for j in range(i + 1, n):
            b = building_infos[j]
            gap = _edge_to_edge_distance(
                a["x"], a["z"], a["w"], a["d"],
                b["x"], b["z"], b["w"], b["d"])
            if gap < CLUSTER_DISTANCE_THRESHOLD:
                adj[i].append((j, gap))
                adj[j].append((i, gap))

    # BFS to find clusters
    visited = set()
    clusters = []
    for start in range(n):
        if start in visited:
            continue
        members = []
        edges = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            members.append(node)
            for nb, gap in adj[node]:
                if nb not in visited:
                    queue.append(nb)
                    edges.append({"from": node, "to": nb,
                                  "distance": round(gap, 1),
                                  "coupling": "simple"})

        # Main building = largest area (or closest to road for organic)
        if layout_type == "organic" and hasattr(layout_organic, '_last_roads'):
            roads = layout_organic._last_roads
            def _road_dist(idx):
                b = building_infos[idx]
                _, d = _nearest_road_dir(b["x"], b["z"], roads)
                return d
            main_idx = min(members, key=_road_dist)
        else:
            main_idx = max(members,
                           key=lambda i: building_infos[i]["w"] *
                                         building_infos[i]["d"])

        mb = building_infos[main_idx]
        # Door front point: offset from main building center toward road area
        half = max(mb["w"], mb["d"]) / 2
        clusters.append({
            "id": len(clusters),
            "building_indices": sorted(members),
            "main_building_idx": main_idx,
            "main_building_pos": [mb["x"], mb["z"]],
            "main_building_door": [mb["x"], mb["z"]],  # updated by road
            "center": [
                sum(building_infos[i]["x"] for i in members) / len(members),
                sum(building_infos[i]["z"] for i in members) / len(members),
            ],
            "tip_building_idx": None,
            "internal_relations": edges,
        })

    return clusters


def orient_doors_street(building_infos, area_d):
    """Street: left row yaw=180 (door faces +Z toward road), right row yaw=0 (door faces -Z)."""
    road_z = area_d / 2
    yaw = {}
    for i, b in enumerate(building_infos):
        yaw[i] = 180.0 if b["z"] < road_z else 0.0
    return yaw


def orient_doors_grid(building_infos):
    """Grid: yaw toward nearest cell edge (0/90/180/270 only).
    Uses distance to the 4 edges of the cell the building sits in.
    Alternating tiebreaker by (col+row)%2 to avoid all-same yaw.
    """
    cell_w = getattr(layout_grid, '_cell_w', 25.0)
    cell_d = getattr(layout_grid, '_cell_d', 25.0)
    cols = getattr(layout_grid, '_cols', 4)
    rows = getattr(layout_grid, '_rows', 3)

    yaw = {}
    for i, b in enumerate(building_infos):
        bx, bz = b["x"], b["z"]
        col = min(int(bx / cell_w), cols - 1)
        row = min(int(bz / cell_d), rows - 1)

        d_left   = bx - col * cell_w
        d_right  = (col + 1) * cell_w - bx
        d_bottom = bz - row * cell_d
        d_top    = (row + 1) * cell_d - bz

        # Corrected yaw mapping (door starts at local z_min):
        #   yaw=0   → door at -Z side     yaw=180 → door at +Z side
        #   yaw=90  → door at -X side     yaw=270 → door at +X side
        # Alternating preference by (col+row)%2 to break uniformity.
        if (col + row) % 2 == 0:
            # Prefer facing a horizontal (Z) road line
            if d_bottom <= d_top:
                yaw[i] = 0.0     # door at -Z → faces bottom line
            else:
                yaw[i] = 180.0   # door at +Z → faces top line
        else:
            # Prefer facing a vertical (X) road line
            if d_left <= d_right:
                yaw[i] = 90.0    # door at -X → faces left line
            else:
                yaw[i] = 270.0   # door at +X → faces right line
    return yaw


def _nearest_point_on_roads(bx, bz, roads):
    """Find closest point on any road polyline to (bx,bz). Returns (rx,rz)."""
    best_d = float('inf')
    best_pt = (bx, bz)
    for road in roads:
        for rx, rz in road:
            d = math.hypot(bx - rx, bz - rz)
            if d < best_d:
                best_d = d
                best_pt = (rx, rz)
    return best_pt


def _yaw_facing(dx, dz):
    """Compute yaw so the door (at local z_min) faces direction (dx, dz).
    Formula: yaw = atan2(-dx, -dz). Verified against trimesh rotation."""
    return math.degrees(math.atan2(-dx, -dz)) % 360


def orient_doors_random(building_infos, clusters, rng=None, **_kw):
    """Random: main→outward from cluster center, others→main. ±8° jitter."""
    yaw = {}
    for cl in clusters:
        main_idx = cl["main_building_idx"]
        mb = building_infos[main_idx]
        ccx, ccz = cl["center"]

        # Main faces outward (away from cluster center)
        dx = mb["x"] - ccx
        dz = mb["z"] - ccz
        if abs(dx) < 0.01 and abs(dz) < 0.01:
            raw_yaw = 0.0
        else:
            raw_yaw = _yaw_facing(dx, dz)
        jitter = float(rng.uniform(-8.0, 8.0)) if rng is not None else 0.0
        yaw[main_idx] = (raw_yaw + jitter) % 360

        # Others face toward main building
        for bi in cl["building_indices"]:
            if bi == main_idx:
                continue
            b = building_infos[bi]
            dx2 = mb["x"] - b["x"]
            dz2 = mb["z"] - b["z"]
            if abs(dx2) < 0.01 and abs(dz2) < 0.01:
                yaw[bi] = yaw[main_idx]
            else:
                jitter = float(rng.uniform(-8.0, 8.0)) if rng is not None else 0.0
                yaw[bi] = (_yaw_facing(dx2, dz2) + jitter) % 360
    return yaw


def orient_doors_plaza(building_infos, area_w, area_d, rng):
    """Plaza: face toward center + ±15° jitter for natural look."""
    cx, cz = area_w / 2, area_d / 2
    yaw = {}
    for i, b in enumerate(building_infos):
        dx = cx - b["x"]
        dz = cz - b["z"]
        if abs(dx) < 0.01 and abs(dz) < 0.01:
            yaw[i] = 0.0
        else:
            jitter = float(rng.uniform(-15.0, 15.0))
            yaw[i] = (_yaw_facing(dx, dz) + jitter) % 360
    return yaw


def orient_doors_organic(building_infos, clusters, rng=None, **_kw):
    """Organic: main→outward from cluster center, others→main. ±8° jitter."""
    yaw = {}
    for cl in clusters:
        main_idx = cl["main_building_idx"]
        mb = building_infos[main_idx]
        ccx, ccz = cl["center"]

        # Main faces outward
        dx = mb["x"] - ccx
        dz = mb["z"] - ccz
        if abs(dx) < 0.01 and abs(dz) < 0.01:
            yaw[main_idx] = 0.0
        else:
            jitter = float(rng.uniform(-8.0, 8.0)) if rng is not None else 0.0
            yaw[main_idx] = (_yaw_facing(dx, dz) + jitter) % 360

        # Others face toward main
        for bi in cl["building_indices"]:
            if bi == main_idx:
                continue
            b = building_infos[bi]
            dx2 = mb["x"] - b["x"]
            dz2 = mb["z"] - b["z"]
            if abs(dx2) < 0.01 and abs(dz2) < 0.01:
                yaw[bi] = yaw[main_idx]
            else:
                jitter = float(rng.uniform(-8.0, 8.0)) if rng is not None else 0.0
                yaw[bi] = (_yaw_facing(dx2, dz2) + jitter) % 360
    return yaw


# ─── 道路骨架 + 地块系统 ────────────────────────────────────────────

def generate_road_skeleton(area_w: float, area_d: float,
                           layout_type: str, rng) -> list:
    """
    Generate road center-lines + widths BEFORE building placement.
    Returns list of {"points": [[x,z],...], "width": float, "type": str}.
    Only street and grid produce skeletons here; organic/random/plaza
    generate roads post-placement (unchanged).
    """
    from shapely.geometry import box as shapely_box
    segments = []
    inset = 2.0  # wall inset

    if layout_type == "street":
        road_z = area_d / 2
        # Extend past scene bounds so the road polygon fully bisects the rectangle
        segments.append({
            "points": [[-1, road_z], [area_w + 1, road_z]],
            "width": STREET_WIDTH,
            "type": "main",
        })

    elif layout_type == "grid":
        n_x = max(2, round(area_w / 35))
        n_z = max(2, round(area_d / 35))
        road_w = 6.0
        for i in range(1, n_x):
            rx = area_w * i / n_x
            segments.append({
                "points": [[rx, -1], [rx, area_d + 1]],
                "width": road_w, "type": "main",
            })
        for j in range(1, n_z):
            rz = area_d * j / n_z
            segments.append({
                "points": [[-1, rz], [area_w + 1, rz]],
                "width": road_w, "type": "main",
            })

    return segments


def _find_road_edges(poly, road_union) -> list:
    """Return list of (side, length) where the lot touches a road."""
    if road_union is None:
        return []
    shared = poly.boundary.intersection(road_union)
    if shared.is_empty:
        return []
    edges = []
    minx, minz, maxx, maxz = poly.bounds
    cx, cz = poly.centroid.x, poly.centroid.y
    # Classify by which side of the lot centroid the shared boundary lies
    if hasattr(shared, 'geoms'):
        pts = []
        for g in shared.geoms:
            if hasattr(g, 'coords'):
                pts.extend(g.coords)
        if not pts:
            return []
    elif hasattr(shared, 'coords'):
        pts = list(shared.coords)
    else:
        return []
    avg_x = sum(p[0] for p in pts) / len(pts)
    avg_z = sum(p[1] for p in pts) / len(pts)
    if avg_z < cz - 1:
        edges.append("south")
    elif avg_z > cz + 1:
        edges.append("north")
    if avg_x < cx - 1:
        edges.append("west")
    elif avg_x > cx + 1:
        edges.append("east")
    if not edges:
        edges.append("south")
    return edges


def generate_lots(area_w: float, area_d: float, layout_type: str,
                  road_segments: list, road_width: float,
                  rng) -> list:
    """
    Cut the scene rectangle by road polygons to produce buildable lots.
    Returns list of lot dicts with polygon, centroid, area, road_edges.
    """
    from shapely.geometry import box as shapely_box, LineString, MultiPolygon
    from shapely.ops import unary_union

    scene = shapely_box(0, 0, area_w, area_d)

    road_polys = []
    for seg in road_segments:
        line = LineString(seg["points"])
        w = seg.get("width", road_width)
        road_polys.append(line.buffer(w / 2, cap_style=2))  # flat cap

    road_union = unary_union(road_polys) if road_polys else None
    buildable = scene.difference(road_union) if road_union else scene

    if isinstance(buildable, MultiPolygon):
        raw_lots = list(buildable.geoms)
    else:
        raw_lots = [buildable]

    # Filter tiny slivers
    raw_lots = [p for p in raw_lots if p.area > 20]
    raw_lots.sort(key=lambda p: p.area, reverse=True)

    lots = []
    for poly in raw_lots:
        cx, cz = poly.centroid.x, poly.centroid.y
        lots.append({
            "polygon": poly,
            "centroid": (cx, cz),
            "area": poly.area,
            "road_edges": _find_road_edges(poly, road_union),
            "setback": 2.0,
            "fill_ratio": 0.7,
            "role_hint": "standard",
        })
    return lots


def _subdivide_street_lots(lots: list, area_w: float, area_d: float,
                           rng) -> list:
    """
    Subdivide the two large street-side lots into rhythmic building plots.
    Lot widths vary (8-18m), setbacks vary (1-4m), ambient slots inserted.
    """
    from shapely.geometry import box as shapely_box
    subdivided = []

    for lot in lots:
        poly = lot["polygon"]
        minx, minz, maxx, maxz = poly.bounds
        span_x = maxx - minx
        span_z = maxz - minz

        # Street lots are wide along X, narrow along Z
        if span_x < 15:
            subdivided.append(lot)
            continue

        cursor = 0.0
        rhythm_counter = 0
        ambient_interval = int(rng.integers(3, 6))

        while cursor < span_x - 5:
            rhythm_counter += 1

            if rhythm_counter % ambient_interval == 0 and cursor + 4 < span_x:
                w = float(rng.uniform(3.0, 5.0))
                role_hint = "ambient"
                setback = float(rng.uniform(0.5, 2.0))
                fill_ratio = float(rng.uniform(0.3, 0.5))
            else:
                w = float(rng.uniform(8.0, 18.0))
                w = min(w, span_x - cursor)
                role_hint = "standard"
                setback = float(rng.uniform(1.0, 4.0))
                fill_ratio = float(rng.uniform(0.5, 0.85))

            if w < 3:
                break

            sub_poly = shapely_box(minx + cursor, minz, minx + cursor + w, maxz)
            sub_poly = sub_poly.intersection(poly)

            if sub_poly.area > 5:
                cx, cz = sub_poly.centroid.x, sub_poly.centroid.y
                subdivided.append({
                    "polygon": sub_poly,
                    "centroid": (cx, cz),
                    "area": sub_poly.area,
                    "road_edges": lot["road_edges"],
                    "setback": setback,
                    "fill_ratio": fill_ratio,
                    "role_hint": role_hint,
                })

            cursor += w + float(rng.uniform(0.5, 2.0))

    return subdivided


def _subdivide_grid_blocks(lots: list, rng) -> list:
    """Subdivide large grid blocks into 2-4 building plots."""
    from shapely.geometry import box as shapely_box
    subdivided = []

    for lot in lots:
        if lot["area"] > 400:
            poly = lot["polygon"]
            minx, minz, maxx, maxz = poly.bounds
            mid_x = (minx + maxx) / 2 + float(rng.uniform(-3, 3))
            mid_z = (minz + maxz) / 2 + float(rng.uniform(-3, 3))
            quarters = [
                shapely_box(minx, minz, mid_x, mid_z),
                shapely_box(mid_x, minz, maxx, mid_z),
                shapely_box(minx, mid_z, mid_x, maxz),
                shapely_box(mid_x, mid_z, maxx, maxz),
            ]
            for q in quarters:
                qc = q.intersection(poly)
                if qc.area > 20:
                    cx, cz = qc.centroid.x, qc.centroid.y
                    subdivided.append({
                        "polygon": qc, "centroid": (cx, cz), "area": qc.area,
                        "road_edges": lot["road_edges"],
                        "setback": float(rng.uniform(1.5, 3.5)),
                        "fill_ratio": float(rng.uniform(0.5, 0.8)),
                        "role_hint": "standard",
                    })
        else:
            subdivided.append(lot)
    return subdivided


def _yaw_toward_road_edge(lot: dict, cx: float, cz: float) -> float:
    """Compute yaw so the door faces the road edge of this lot."""
    edges = lot.get("road_edges", [])
    if "south" in edges:
        return 0.0      # door at -Z
    if "north" in edges:
        return 180.0     # door at +Z
    if "west" in edges:
        return 90.0      # door at -X
    if "east" in edges:
        return 270.0     # door at +X
    return 0.0


def _assign_buildings_to_lots(building_roles: list,
                              spatial_rules: Optional[list],
                              lots: list,
                              style: str, variation: float,
                              rng) -> list:
    """
    Assign archetype building roles to lots.
    Primary → largest lot, secondary → near primary, tertiary → fill,
    ambient → ambient-hinted lots.
    """
    expanded = []
    for b in building_roles:
        for _ in range(b.get("count", 1)):
            entry = dict(b)
            entry["count"] = 1
            expanded.append(entry)

    _role_ord = {"primary": 0, "secondary": 1, "tertiary": 2, "ambient": 3}
    sorted_roles = sorted(expanded, key=lambda b: _role_ord.get(b.get("role"), 9))

    ambient_lots = [i for i, l in enumerate(lots) if l["role_hint"] == "ambient"]
    normal_lots = [i for i, l in enumerate(lots) if l["role_hint"] != "ambient"]

    slots: list[BuildingSlot] = []
    used: set[int] = set()
    primary_lot_idx = None

    for binfo in sorted_roles:
        role = binfo.get("role", "tertiary")
        bstyle = binfo.get("style_key", style) or style
        size_role = {"primary": "anchor", "secondary": "sub_anchor",
                     "ambient": "filler"}.get(role, "normal")
        w, d = _building_size(bstyle, variation, rng, role=size_role)

        # Pick lot
        lot_idx = None
        if role == "primary":
            for li in normal_lots:
                if li not in used:
                    lot_idx = li
                    break
        elif role == "secondary":
            # Find nearest unused lot to primary
            if primary_lot_idx is not None:
                ref = lots[primary_lot_idx]["centroid"]
                candidates = [(li, math.hypot(
                    lots[li]["centroid"][0] - ref[0],
                    lots[li]["centroid"][1] - ref[1]))
                    for li in normal_lots if li not in used]
                candidates.sort(key=lambda c: c[1])
                if candidates:
                    lot_idx = candidates[0][0]
            else:
                for li in normal_lots:
                    if li not in used:
                        lot_idx = li
                        break
        elif role == "ambient":
            for li in ambient_lots:
                if li not in used:
                    lot_idx = li
                    break
            if lot_idx is None:
                # Fallback: smallest unused normal lot
                candidates = [(li, lots[li]["area"])
                              for li in normal_lots if li not in used]
                candidates.sort(key=lambda c: c[1])
                if candidates:
                    lot_idx = candidates[0][0]
        else:  # tertiary
            for li in normal_lots:
                if li not in used:
                    lot_idx = li
                    break

        if lot_idx is None:
            continue

        used.add(lot_idx)
        if role == "primary":
            primary_lot_idx = lot_idx

        lot = lots[lot_idx]
        cx, cz = lot["centroid"]
        fill_ratio = lot.get("fill_ratio", 0.7)
        bounds = lot["polygon"].bounds  # (minx, minz, maxx, maxz)
        max_w = (bounds[2] - bounds[0]) * fill_ratio
        max_d = (bounds[3] - bounds[1]) * fill_ratio
        w = min(w, max(5.0, max_w))
        d = min(d, max(5.0, max_d))

        yaw = _yaw_toward_road_edge(lot, cx, cz)
        cx += float(rng.uniform(-1.5, 1.5))
        cz += float(rng.uniform(-1.5, 1.5))
        # Clamp to lot interior
        cx = max(bounds[0] + w / 2 + 1, min(bounds[2] - w / 2 - 1, cx))
        cz = max(bounds[1] + d / 2 + 1, min(bounds[3] - d / 2 - 1, cz))

        yaw += float(rng.uniform(-8.0, 8.0))
        fp = _pick_footprint_type(variation, rng)
        slots.append(BuildingSlot(
            cx - w / 2, cz - d / 2, w, d, fp,
            yaw_deg=yaw % 360,
            is_anchor=(role == "primary"),
            style_key=bstyle,
            role=role,
        ))

    return slots


# ─── Archetype 放置辅助函数 ────────────────────────────────────────

_ROLE_ORDER = {"primary": 0, "secondary": 1, "tertiary": 2, "ambient": 3}

_SCALE_TO_ROLE = {
    "largest": "anchor", "large": "sub_anchor",
    "medium": "normal", "small": "filler", "tiny": "filler",
}


def _archetype_role_to_size_role(building_info: dict) -> str:
    """Map archetype building role/scale to _building_size role param."""
    role = building_info.get("role", "tertiary")
    if role == "primary":
        return "anchor"
    if role == "secondary":
        return "sub_anchor"
    if role == "ambient":
        return "filler"
    return "normal"


def _is_on_road(cz: float, half_d: float, road_corridor) -> bool:
    """Check if a building's Z span overlaps the road corridor."""
    if road_corridor is None:
        return False
    return (cz - half_d) < road_corridor[1] and (cz + half_d) > road_corridor[0]


def _place_primary(building_info: dict, area_w: float, area_d: float,
                   style: str, variation: float,
                   rng: np.random.Generator,
                   road_corridor=None) -> BuildingSlot:
    """Place primary building at scene center, avoiding road corridor."""
    bstyle = building_info.get("style_key", style)
    role = _archetype_role_to_size_role(building_info)
    w, d = _building_size(bstyle, variation, rng, role=role)
    cx = area_w / 2 + float(rng.uniform(-2.0, 2.0))

    if road_corridor:
        # Place north of road corridor
        cz = road_corridor[1] + d / 2 + 3.0 + float(rng.uniform(0, 2.0))
    else:
        cz = area_d / 2 + float(rng.uniform(-2.0, 2.0))

    fp = _pick_footprint_type(variation, rng)
    return BuildingSlot(
        cx - w / 2, cz - d / 2, w, d, fp, yaw_deg=0.0,
        is_anchor=True, style_key=bstyle, role="primary")


def _find_relationship(from_label: str, to_label: str,
                       spatial_rules: list) -> Optional[dict]:
    """Find spatial relationship between two buildings in the rules list.
    Rules can be dicts {"from": ..., "to": ..., "relation": ...} or plain strings (ignored)."""
    if not spatial_rules:
        return None
    for r in spatial_rules:
        if isinstance(r, str):
            continue  # skip free-text rules from LLM
        rf = r.get("from", "")
        rt = r.get("to", "")
        if (rf == from_label and rt == to_label) or \
           (rf == to_label and rt == from_label):
            return r
    return None


_INVERSE_REL = {
    "facing": "facing", "flanking": "flanking", "behind": "facing",
    "adjacent": "adjacent", "surrounding": "adjacent", "distant": "distant",
}


def _place_relative(building_info: dict, ref_slot: BuildingSlot,
                    relationship: Optional[dict],
                    placed: list, area_w: float, area_d: float,
                    style: str, variation: float,
                    rng: np.random.Generator,
                    road_corridor=None) -> BuildingSlot:
    """Place a building relative to ref_slot, avoiding road corridor."""
    bstyle = building_info.get("style_key", style)
    role = _archetype_role_to_size_role(building_info)
    w, d = _building_size(bstyle, variation, rng, role=role)

    ref_cx = ref_slot.x_off + ref_slot.w / 2
    ref_cz = ref_slot.z_off + ref_slot.d / 2
    offset_dist = max(ref_slot.w, ref_slot.d) / 2 + max(w, d) / 2 + 6.0

    rel_type = "adjacent"
    if relationship:
        rel_type = relationship.get("relation", "adjacent")

    margin = WALL_TOTAL + 2.0
    tx, tz = ref_cx, ref_cz  # init for fallback

    for attempt in range(30):
        if rel_type == "facing":
            tx = ref_cx + float(rng.uniform(-3.0, 3.0))
            tz = ref_cz - offset_dist + float(rng.uniform(-2.0, 2.0))
        elif rel_type == "flanking":
            side = 1 if rng.random() < 0.5 else -1
            tx = ref_cx + side * offset_dist + float(rng.uniform(-2.0, 2.0))
            tz = ref_cz + float(rng.uniform(-4.0, 4.0))
        elif rel_type == "behind":
            tx = ref_cx + float(rng.uniform(-3.0, 3.0))
            tz = ref_cz + offset_dist + float(rng.uniform(-2.0, 2.0))
        elif rel_type == "surrounding":
            angle = float(rng.uniform(0, 2 * math.pi))
            tx = ref_cx + math.cos(angle) * offset_dist
            tz = ref_cz + math.sin(angle) * offset_dist
        elif rel_type == "distant":
            angle = float(rng.uniform(0, 2 * math.pi))
            tx = ref_cx + math.cos(angle) * (offset_dist * 2.0)
            tz = ref_cz + math.sin(angle) * (offset_dist * 2.0)
        else:  # adjacent
            angle = float(rng.uniform(0, 2 * math.pi))
            tx = ref_cx + math.cos(angle) * offset_dist * 0.7
            tz = ref_cz + math.sin(angle) * offset_dist * 0.7

        tx = max(margin + w / 2, min(area_w - margin - w / 2, tx))
        tz = max(margin + d / 2, min(area_d - margin - d / 2, tz))

        if _is_on_road(tz, d / 2, road_corridor):
            continue
        if not _obb_collides(tx, tz, w, d, 0, placed, MIN_BUILDING_GAP):
            fp = _pick_footprint_type(variation, rng)
            return BuildingSlot(
                tx - w / 2, tz - d / 2, w, d, fp, yaw_deg=0.0,
                style_key=bstyle, role=building_info.get("role", "secondary"))

        # Retry with jitter
        offset_dist += float(rng.uniform(1.0, 3.0))

    # Fallback: place anyway at last attempted position
    fp = _pick_footprint_type(variation, rng)
    return BuildingSlot(
        tx - w / 2, tz - d / 2, w, d, fp, yaw_deg=0.0,
        style_key=bstyle, role=building_info.get("role", "secondary"))


def _place_fill(building_info: dict, placed: list,
                area_w: float, area_d: float,
                style: str, variation: float,
                rng: np.random.Generator,
                road_corridor=None) -> Optional[BuildingSlot]:
    """Place a building in gaps between existing buildings, avoiding road corridor."""
    bstyle = building_info.get("style_key", style)
    role = _archetype_role_to_size_role(building_info)
    w, d = _building_size(bstyle, variation, rng, role=role)

    margin = WALL_TOTAL + 2.0
    best_slot = None
    best_dist = float("inf")

    for _ in range(60):
        tx = float(rng.uniform(margin + w / 2, area_w - margin - w / 2))
        tz = float(rng.uniform(margin + d / 2, area_d - margin - d / 2))

        if _is_on_road(tz, d / 2, road_corridor):
            continue
        if _obb_collides(tx, tz, w, d, 0, placed, MIN_BUILDING_GAP):
            continue

        # Prefer positions that are close-but-not-touching existing buildings
        if placed:
            # Shapely .bounds = (minx, miny, maxx, maxy)
            min_d = min(
                _edge_to_edge_distance(
                    tx, tz, w, d,
                    (b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                    b[2] - b[0], b[3] - b[1])
                for b in (p.bounds for p in placed))
        else:
            min_d = 50.0

        # Sweet spot: 5-20m from nearest (cluster-friendly but not on top)
        if 5.0 < min_d < 20.0 and min_d < best_dist:
            best_dist = min_d
            fp = _pick_footprint_type(variation, rng)
            best_slot = BuildingSlot(
                tx - w / 2, tz - d / 2, w, d, fp, yaw_deg=0.0,
                style_key=bstyle,
                role=building_info.get("role", "tertiary"))

    return best_slot


def _expand_building_roles(building_roles: list) -> list:
    """Expand building role entries with count > 1 into individual entries."""
    expanded = []
    for b in building_roles:
        count = b.get("count", 1)
        for _ in range(count):
            entry = dict(b)
            entry["count"] = 1
            expanded.append(entry)
    return expanded


def _archetype_placement(building_roles: list, spatial_rules: Optional[list],
                         area_w: float, area_d: float,
                         style: str, variation: float,
                         rng: np.random.Generator,
                         layout_type: str = "organic") -> list:
    """
    Phased placement driven by archetype plan.

    Street / grid layouts use the lot system:
      road skeleton → lot cutting → rhythmic subdivision → assign buildings.
    Organic / random / plaza use the original phased placement:
      Phase A: primary (center) → B: secondary → C: tertiary → D: ambient.
    """
    # ── Lot-based path (street / grid) ──
    if layout_type in ("street", "grid"):
        road_skeleton = generate_road_skeleton(area_w, area_d, layout_type, rng)
        lots = generate_lots(area_w, area_d, layout_type, road_skeleton, 8.0, rng)
        if layout_type == "street":
            lots = _subdivide_street_lots(lots, area_w, area_d, rng)
        else:
            lots = _subdivide_grid_blocks(lots, rng)
        return _assign_buildings_to_lots(
            building_roles, spatial_rules, lots, style, variation, rng)

    # ── Original phased path (organic / random / plaza) ──
    expanded = _expand_building_roles(building_roles)
    sorted_roles = sorted(expanded, key=lambda b: _ROLE_ORDER.get(b.get("role"), 9))

    # Filter out ambient from building_count, keep for phase D
    non_ambient = [b for b in sorted_roles if b.get("role") != "ambient"]
    ambients = [b for b in sorted_roles if b.get("role") == "ambient"]

    # Road corridor exclusion zone for street layout
    road_corridor = None
    if layout_type == "street":
        road_z = area_d / 2
        road_half_w = 7.0  # road half-width + safety margin
        road_corridor = (road_z - road_half_w, road_z + road_half_w)

    slots: list[BuildingSlot] = []
    placed_obbs: list = []  # shapely polygons for collision

    # ── Phase A: primary ──
    primaries = [b for b in non_ambient if b.get("role") == "primary"]
    primary_slot = None
    for p in primaries[:1]:  # exactly 1
        slot = _place_primary(p, area_w, area_d, style, variation, rng,
                              road_corridor=road_corridor)
        slots.append(slot)
        placed_obbs.append(_obb_polygon(
            slot.x_off + slot.w / 2, slot.z_off + slot.d / 2,
            slot.w, slot.d, 0.0))
        primary_slot = slot

    # ── Phase B: secondary (relative to primary) ──
    secondaries = [b for b in non_ambient if b.get("role") == "secondary"]
    for sec in secondaries:
        ref = primary_slot or slots[0] if slots else None
        if ref is None:
            continue
        rel = _find_relationship(
            sec.get("label", ""), primaries[0].get("label", "") if primaries else "",
            spatial_rules)
        slot = _place_relative(
            sec, ref, rel, placed_obbs, area_w, area_d,
            style, variation, rng, road_corridor=road_corridor)
        slots.append(slot)
        placed_obbs.append(_obb_polygon(
            slot.x_off + slot.w / 2, slot.z_off + slot.d / 2,
            slot.w, slot.d, 0.0))

    # ── Phase C: tertiary (fill) ──
    tertiaries = [b for b in non_ambient if b.get("role") == "tertiary"]
    for ter in tertiaries:
        slot = _place_fill(
            ter, placed_obbs, area_w, area_d, style, variation, rng,
            road_corridor=road_corridor)
        if slot:
            slots.append(slot)
            placed_obbs.append(_obb_polygon(
                slot.x_off + slot.w / 2, slot.z_off + slot.d / 2,
                slot.w, slot.d, 0.0))

    # ── Phase D: ambient (best effort) ──
    for amb in ambients:
        slot = _place_fill(
            amb, placed_obbs, area_w, area_d, style, variation, rng,
            road_corridor=road_corridor)
        if slot:
            slots.append(slot)
            placed_obbs.append(_obb_polygon(
                slot.x_off + slot.w / 2, slot.z_off + slot.d / 2,
                slot.w, slot.d, 0.0))

    return slots


# ─── 主生成器（Phase 0-7 cluster pipeline）───────────────────────

def generate_level(
    style: str,
    layout_type: str = "street",
    building_count: int = 10,
    area_size: float = 100.0,
    variation: float = 0.4,
    seed: int = 42,
    params_path: Optional[str] = None,
    output_path: Optional[str] = None,
    area_w: Optional[float] = None,
    area_d: Optional[float] = None,
    min_gap: Optional[float] = None,
    # ── archetype plan fields (optional) ──
    building_roles: Optional[list] = None,
    spatial_rules: Optional[list] = None,
    enclosure_config: Optional[dict] = None,
) -> trimesh.Scene:
    """
    Phase 0: Area setup
    Phase 1: Layout → building slots
    Phase 2: Identify clusters
    Phase 3: Generate road network (cluster-based)
    Phase 4: Door yaw (main→road, others→main)
    Phase 5: Building mesh generation
    Phase 6: Road mesh + infrastructure
    Phase 7: Stats + export
    """
    if params_path is None:
        params_path = str(PARAMS_JSON)
    data        = json.loads(Path(params_path).read_text("utf-8"))
    styles_data = data.get("styles", {})

    if style not in styles_data:
        raise ValueError(f"风格 '{style}' 未找到。可用风格: {list(styles_data)}")

    params  = styles_data[style]["params"]
    palette = _derive_palette(style, params)
    rng     = np.random.default_rng(seed)

    # ══ Phase 0: Area ════════════════════════════════════════════
    aw = area_w or area_size
    ad = area_d or area_size
    building_count = max(1, min(40, building_count))

    if layout_type == "organic" and area_w is None:
        compact_size = max(60.0, building_count * 8.0)
        aw = min(aw, compact_size)
        ad = min(ad, compact_size)

    anchor_style  = _ANCHOR_STYLE.get(style, style)
    anchor_params = styles_data.get(anchor_style, styles_data[style])["params"]

    if min_gap is not None:
        set_gap(min_gap)

    layout_fn = {
        "grid": layout_grid, "street": layout_street,
        "plaza": layout_plaza, "random": layout_random,
        "organic": layout_organic,
    }.get(layout_type)
    if layout_fn is None:
        raise ValueError(
            f"未知布局类型: {layout_type}，可选: grid/street/plaza/random/organic")

    # ══ Phase 1: Layout → slots ══════════════════════════════════
    if building_roles:
        # Archetype-driven phased placement
        slots = _archetype_placement(
            building_roles, spatial_rules, aw, ad, style, variation, rng,
            layout_type=layout_type)
        print(f"  [archetype] placed {len(slots)} buildings "
              f"(roles: {[s.role for s in slots]})")
    else:
        # Standard layout function
        slots = layout_fn(building_count, aw, ad, style, variation, rng)

    # Wall clearance: clamp all buildings away from perimeter walls
    enforce_wall_clearance(slots, aw, ad)

    import trimesh.transformations as TF
    build_plans = []
    building_infos = []

    for bi, slot in enumerate(slots):
        # Per-building style_key: use slot.style_key if set, else default
        build_style = slot.style_key if slot.style_key else style
        if slot.is_anchor:
            build_style_for_params = slot.style_key or anchor_style
            bparams  = styles_data.get(build_style_for_params,
                                        styles_data[style])["params"]
            bpalette = _derive_palette(build_style_for_params, bparams)
        elif slot.style_key and slot.style_key in styles_data:
            bparams  = styles_data[slot.style_key]["params"]
            bpalette = _derive_palette(slot.style_key, bparams)
            bparams  = _vary_params(bparams, variation, rng)
        else:
            bparams  = _vary_params(params, variation, rng)
            bpalette = _derive_palette(style, bparams)

        fp_local = _make_footprint(slot.fp_type, slot.w, slot.d, rng)
        y_offset = float(rng.uniform(-0.3, 0.3)) * variation

        building_infos.append({
            "x": slot.x_off + slot.w / 2,
            "z": slot.z_off + slot.d / 2,
            "w": slot.w, "d": slot.d,
            "height": float(bparams["height_range"][1]),
            "yaw_deg": slot.yaw_deg,
            "role": slot.role,
            "style_key": build_style,
        })
        build_plans.append({
            "slot": slot, "bparams": bparams, "bpalette": bpalette,
            "fp_local": fp_local, "y_offset": y_offset, "bi": bi,
        })

    stats = {"buildings": 0, "total_footprint_m2": 0.0, "style": style,
             "layout": layout_type, "area": f"{aw:.0f}m x {ad:.0f}m",
             "variation": variation}

    # ══ Phase 2: Identify clusters ═══════════════════════════════
    clusters = identify_clusters(building_infos, layout_type)
    cl_sizes = [len(c["building_indices"]) for c in clusters]
    main_idxs = [c["main_building_idx"] for c in clusters]
    print(f"  Clusters: {len(clusters)} (sizes: {cl_sizes})")
    print(f"  Main buildings: {main_idxs}")

    # ══ Phase 3: Road network (cluster-based) ════════════════════
    road_graph = generate_road_network(
        clusters, building_infos, aw, ad, layout_type, style)
    n_road = len([nd for nd in road_graph.nodes
                  if nd.ntype in ("road", "intersection", "endpoint")])
    n_acc = len([e for e in road_graph.edges if e.road_type == "access"])
    print(f"  Road nodes: {n_road}")
    print(f"  Road edges: {len(road_graph.edges)}")
    print(f"  Access paths: {n_acc}")
    print(f"  BFS connected: {road_graph.bfs_connected()}")
    print(f"  Tip buildings: 0 (interface reserved)")

    # ══ Phase 4: Door yaw ════════════════════════════════════════
    if layout_type == "street":
        effective_yaw = orient_doors_street(building_infos, ad)
    elif layout_type == "grid":
        effective_yaw = orient_doors_grid(building_infos)
    elif layout_type == "random":
        effective_yaw = orient_doors_random(building_infos, clusters, rng=rng)
    elif layout_type == "plaza":
        effective_yaw = orient_doors_plaza(building_infos, aw, ad, rng)
    else:  # organic
        effective_yaw = orient_doors_organic(building_infos, clusters, rng=rng)

    # ══ Phase 4b: door-not-facing-wall check ═════════════════════
    fix_door_facing_wall(effective_yaw, building_infos, aw, ad,
                         layout_type=layout_type)

    # ══ Phase 5: Building meshes ═════════════════════════════════
    scene = trimesh.Scene()

    for plan in build_plans:
        slot = plan["slot"]
        bparams = plan["bparams"]
        bpalette = plan["bpalette"]
        fp_local = plan["fp_local"]
        y_offset = plan["y_offset"]
        bi = plan["bi"]

        final_yaw = effective_yaw.get(bi, slot.yaw_deg)

        try:
            room_meshes = gl.build_room(
                bparams, bpalette, x_off=0.0, z_off=0.0, footprint=fp_local)

            if abs(final_yaw) > 0.01:
                cx, cz = slot.w / 2, slot.d / 2
                rot = TF.rotation_matrix(
                    math.radians(final_yaw), [0, 1, 0], point=[cx, 0, cz])
                for mesh in room_meshes:
                    mesh.apply_transform(rot)

            for mesh in room_meshes:
                mesh.apply_translation([slot.x_off, y_offset, slot.z_off])

            for mi, mesh in enumerate(room_meshes):
                scene.add_geometry(mesh,
                                   node_name=f"b{bi:02d}_{slot.fp_type}_{mi:03d}")

            stats["buildings"] += 1
            stats["total_footprint_m2"] += fp_local.area

        except Exception as e:
            print(f"  [!] building {bi} failed: {e}")
            continue

    # ══ Phase 6: Road mesh + infrastructure ══════════════════════
    ground = _make_ground_plane(aw, ad, style, params)
    scene.add_geometry(ground, node_name="ground_plane")

    road_meshes = road_graph.generate_meshes(style)
    for ri, rm in enumerate(road_meshes):
        scene.add_geometry(rm, node_name=f"road_{ri:03d}")

    if layout_type == "plaza":
        for pfi, pf in enumerate(_make_plaza_floor(aw, ad, style, slots)):
            scene.add_geometry(pf, node_name=f"plaza_floor_{pfi}")

    # ── Enclosure: archetype config overrides _WALL_STYLES default ──
    wall_meshes = []
    if enclosure_config:
        enc_type = enclosure_config.get("type", "open")
        if enc_type == "walled":
            wall_meshes = _make_perimeter_wall(
                aw, ad, style, params, road_graph=road_graph)
        elif enc_type == "partial":
            wall_meshes = _make_perimeter_wall(
                aw, ad, style, params, road_graph=road_graph,
                wall_h=1.5, wall_t=0.3)
        # enc_type == "open" or unknown → no walls
    else:
        wall_meshes = _make_perimeter_wall(
            aw, ad, style, params, road_graph=road_graph)
    for wi, wm in enumerate(wall_meshes):
        scene.add_geometry(wm, node_name=f"wall_{wi:02d}")
    for li, lm in enumerate(_make_street_lamps(aw, ad, style, params, layout_type)):
        scene.add_geometry(lm, node_name=f"lamp_{li:03d}")

    # ══ Phase 7: Metadata + stats + export ═════════════════════
    # Build cluster_id / is_main lookup for each building
    bld_cluster = {}
    bld_is_main = {}
    for cl in clusters:
        for bi in cl["building_indices"]:
            bld_cluster[bi] = cl["id"]
            bld_is_main[bi] = (bi == cl["main_building_idx"])

    # Enrich building_infos for metadata
    for i, b in enumerate(building_infos):
        b["idx"] = i
        b["yaw_deg"] = effective_yaw.get(i, b.get("yaw_deg", 0.0))
        b["cluster_id"] = bld_cluster.get(i, -1)
        b["is_main_building"] = bld_is_main.get(i, False)

    # Store metadata on scene
    scene.metadata = scene.metadata or {}
    scene.metadata["clusters"] = clusters
    scene.metadata["road_nodes"] = [
        {"id": n.id, "x": n.x, "z": n.z, "type": n.ntype}
        for n in road_graph.nodes
    ]
    scene.metadata["road_edges"] = [
        {"from": e.from_id, "to": e.to_id, "type": e.road_type}
        for e in road_graph.edges
    ]
    scene.metadata["road_renderable"] = road_graph.renderable
    scene.metadata["building_infos"] = building_infos

    n_faces = sum(len(g.faces) for g in scene.geometry.values())
    print(f"\n  Layout    : {layout_type}")
    print(f"  Style     : {style}")
    print(f"  Area      : {stats['area']}")
    print(f"  Buildings : {stats['buildings']}")
    print(f"  Footprint : {stats['total_footprint_m2']:.1f} m2")
    print(f"  Density   : {stats['total_footprint_m2']/(aw*ad)*100:.1f}%")
    print(f"  Faces     : {n_faces:,}")

    if layout_type == "organic":
        print(f"  Anchor    : {anchor_style}")

    if output_path:
        scene.export(output_path)
        print(f"  Output    : {output_path}")

    return scene


def _vary_params(params: dict, variation: float,
                 rng: np.random.Generator) -> dict:
    """在基础参数上叠加随机变体，产生每栋建筑的独立参数。"""
    if variation < 0.01:
        return params

    v = variation
    p = dict(params)

    # 高度变化
    h_min = params["height_range"][0]
    h_max = params["height_range"][1]
    h_range = max(h_max - h_min, 1.0)
    new_max = float(np.clip(
        h_max + rng.uniform(-v * h_range * 0.5, v * h_range * 0.8),
        h_min + 1.5, h_max * 1.6
    ))
    p["height_range"] = [h_min, round(new_max, 2)]

    # 窗户密度变化
    wd = params["win_spec"]["density"]
    p["win_spec"] = {**params["win_spec"],
                     "density": float(np.clip(wd + rng.uniform(-v*0.15, v*0.15), 0.1, 0.8))}

    # 墙色轻微变化（让相邻建筑不完全一样）
    wc = params.get("wall_color")
    if wc and v > 0.1:
        jitter = rng.uniform(-v * 0.08, v * 0.08, size=3)
        p["wall_color"] = [float(np.clip(c + j, 0.0, 1.0))
                           for c, j in zip(wc, jitter)]

    # subdivision 偶尔 ±1
    if v > 0.5 and rng.random() < v * 0.3:
        sub = params["subdivision"]
        p["subdivision"] = int(np.clip(sub + rng.integers(-1, 2), 1, 6))

    return p


# ─── CLI ───────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="LevelSmith 关卡群生成器")
    parser.add_argument("--style",     type=str, default="medieval",
                        help="风格名称 (e.g. medieval, modern, horror)")
    parser.add_argument("--layout",    type=str, default="street",
                        choices=["grid", "street", "plaza", "random", "organic"],
                        help="布局类型")
    parser.add_argument("--count",     type=int, default=10,
                        help="建筑数量 (5~20)")
    parser.add_argument("--area",      type=float, default=100.0,
                        help="场景尺寸 m (正方形)")
    parser.add_argument("--area-w",    type=float, default=None,
                        help="场景宽度 m (覆盖 --area)")
    parser.add_argument("--area-d",    type=float, default=None,
                        help="场景深度 m (覆盖 --area)")
    parser.add_argument("--variation", type=float, default=0.4,
                        help="变体程度 0.0~1.0")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--params",    type=str,   default=None,
                        help="trained_style_params.json 路径")
    parser.add_argument("--out",       type=str,   default=None,
                        help="输出 GLB 文件名 (默认: {style}_{layout}_level.glb)")
    parser.add_argument("--min-gap",   type=float, default=None,
                        help="建筑最小间距 m (紧凑≈0.5, 默认=1.5, 宽松≈4.0)；"
                             "若同时给出 --prompt 则关键词优先")
    parser.add_argument("--prompt",    type=str,   default=None,
                        help="自然语言提示词，自动识别布局密度关键词：\n"
                             "  紧凑/密集/拥挤 → 0.5m\n"
                             "  宽松/分散/稀疏 → 4.0m\n"
                             "  其他           → 1.5m")
    args = parser.parse_args()

    # 密度优先级：--prompt 关键词 > --min-gap 数值 > 默认 1.5
    if args.prompt:
        min_gap = parse_density_prompt(args.prompt)
        print(f"  [提示词] \"{args.prompt}\" → min_gap={min_gap}m")
    elif args.min_gap is not None:
        min_gap = args.min_gap
    else:
        min_gap = SAFETY_GAP   # 使用模块默认值

    out = args.out or f"{args.style}_{args.layout}_level.glb"
    out_path = SCRIPT_DIR / out

    print("=" * 65)
    print("  LevelSmith 关卡群生成器")
    print("=" * 65)
    print(f"  风格: {args.style}  布局: {args.layout}  建筑数: {args.count}")
    print(f"  场景: {args.area_w or args.area:.0f}m × {args.area_d or args.area:.0f}m"
          f"  变体: {args.variation:.2f}  种子: {args.seed}")
    print(f"  最小间距: {min_gap}m")

    generate_level(
        style          = args.style,
        layout_type    = args.layout,
        building_count = args.count,
        area_size      = args.area,
        variation      = args.variation,
        seed           = args.seed,
        params_path    = args.params,
        output_path    = str(out_path),
        area_w         = args.area_w,
        area_d         = args.area_d,
        min_gap        = min_gap,
    )


if __name__ == "__main__":
    main()
