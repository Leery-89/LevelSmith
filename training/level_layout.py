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
                   rng: np.random.Generator) -> tuple[float, float]:
    wmin, wmax, dmin, dmax = _STYLE_SIZE.get(style, _DEFAULT_SIZE)
    noise = variation * 0.5            # 最大 ±50% of range
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


def layout_grid(n: int, area_w: float, area_d: float,
                style: str, variation: float,
                rng: np.random.Generator) -> list[BuildingSlot]:
    """网格布局：按行列均匀分布。"""
    cols = max(1, round(math.sqrt(n * area_w / area_d)))
    rows = math.ceil(n / cols)

    cell_w = area_w / cols
    cell_d = area_d / rows
    slots  = []

    for i in range(n):
        row, col = divmod(i, cols)
        w, d = _building_size(style, variation, rng)
        w = min(w, cell_w * 0.75)
        d = min(d, cell_d * 0.75)
        # 在格子内随机偏移（variation 控制偏移幅度）
        jitter_x = (cell_w - w) * rng.uniform(0, variation * 0.4)
        jitter_z = (cell_d - d) * rng.uniform(0, variation * 0.4)
        x = col * cell_w + (cell_w - w) / 2 + jitter_x
        z = row * cell_d + (cell_d - d) / 2 + jitter_z
        fp = _pick_footprint_type(variation, rng)
        slots.append(BuildingSlot(x, z, w, d, fp))

    return slots


def layout_street(n: int, area_w: float, area_d: float,
                  style: str, variation: float,
                  rng: np.random.Generator) -> list[BuildingSlot]:
    """
    街道布局：沿 X 轴排成两排，中间留街道。
    街道宽度 ≈ area_d × 0.22，两侧各排 n//2 (ceil) 栋建筑。
    """
    street_w = max(6.0, area_d * 0.22)    # 街道宽度（Z 方向）
    side_d   = (area_d - street_w) / 2    # 每侧可用深度
    gap_min  = 3.0

    n_left  = (n + 1) // 2
    n_right = n // 2

    def place_side(count, z_start, face_street):
        side_slots = []
        placed_bboxes: list = []
        slot_w = (area_w - gap_min) / max(count, 1)
        for i in range(count):
            w, d = _building_size(style, variation, rng)
            w = min(w, slot_w * 0.85)
            d = min(d, side_d * 0.82)
            x = i * slot_w + (slot_w - w) / 2
            x += rng.uniform(-variation * 0.8, variation * 0.8)
            x = max(0.0, min(area_w - w, x))
            # 沿街一侧的 z
            if face_street:
                z = z_start + side_d - d          # 靠街道边
            else:
                z = z_start                        # 靠街道边（另一侧）
            z += rng.uniform(-variation * 0.5, variation * 0.5)
            z = max(z_start, min(z_start + side_d - d, z))
            fp = _pick_footprint_type(variation, rng)
            side_slots.append(BuildingSlot(x, z, w, d, fp,
                                           yaw_deg=180.0 if not face_street else 0.0))
        return side_slots

    z_left_start  = 0.0
    z_street      = side_d
    z_right_start = z_street + street_w

    slots  = place_side(n_left, z_left_start, face_street=True)
    slots += place_side(n_right, z_right_start, face_street=False)
    return slots


def layout_plaza(n: int, area_w: float, area_d: float,
                 style: str, variation: float,
                 rng: np.random.Generator) -> list[BuildingSlot]:
    """
    广场布局：建筑围绕中心广场一圈排列，面朝内。
    """
    cx = area_w / 2
    cz = area_d / 2
    plaza_r = min(area_w, area_d) * 0.28    # 广场半径

    slots = []
    for i in range(n):
        angle = 2 * math.pi * i / n + rng.uniform(-0.1, 0.1) * variation
        w, d = _building_size(style, variation, rng)
        # 从中心距离 = plaza_r + d/2 + 随机扰动
        r = plaza_r + d / 2 + rng.uniform(0, variation * 4.0)
        bx = cx + r * math.cos(angle) - w / 2
        bz = cz + r * math.sin(angle) - d / 2
        # 旋转使建筑正面朝中心
        yaw = math.degrees(math.pi - angle)
        fp = _pick_footprint_type(variation, rng)
        slots.append(BuildingSlot(bx, bz, w, d, fp, yaw_deg=yaw))

    return slots


def layout_random(n: int, area_w: float, area_d: float,
                  style: str, variation: float,
                  rng: np.random.Generator,
                  gap: float = 3.0,
                  max_tries: int = 200) -> list[BuildingSlot]:
    """
    随机布局：在场景内随机放置建筑，保证最小间距。
    """
    placed: list = []          # list[Polygon]
    slots:  list[BuildingSlot] = []
    margin = 1.0

    for _ in range(n):
        for attempt in range(max_tries):
            w, d = _building_size(style, variation, rng)
            x = rng.uniform(margin, area_w - w - margin)
            z = rng.uniform(margin, area_d - d - margin)
            yaw = rng.uniform(-15, 15) * variation
            safety = math.hypot(w, d) / 2 + SAFETY_GAP
            cx, cz = x + w / 2, z + d / 2
            if not _obb_collides(cx, cz, w, d, yaw, placed, safety):
                placed.append(_obb_polygon(cx, cz, w, d, yaw))
                fp = _pick_footprint_type(variation, rng)
                slots.append(BuildingSlot(x, z, w, d, fp, yaw_deg=yaw))
                break
        # 超过 max_tries 时跳过该建筑（场地太密）

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


def layout_organic(
    n: int, area_w: float, area_d: float,
    style: str, variation: float,
    rng: np.random.Generator,
) -> list[BuildingSlot]:
    """
    ML 模型 + 路网有机布局：
      1. 用 layout_model_w3.pt 生成建筑位置/朝向
      2. 从 ML 位置推断贝塞尔曲线主路
      3. 建筑朝向对齐最近道路 ±10°
      4. 中心放锚点建筑
    如果 ML 模型不可用，降级为规则贝塞尔路网算法。
    """
    cx, cz = area_w / 2, area_d / 2
    margin = 4.0

    # ── 1. 尝试 ML 模型生成位置 ──────────────────────────────
    seed = int(rng.integers(0, 100000))
    ml_positions = _ml_generate_positions(n - 1, area_w, area_d, seed)
    use_ml = len(ml_positions) >= n // 2

    if use_ml:
        print(f"  [organic] ML 模型生成 {len(ml_positions)} 栋位置")
    else:
        print(f"  [organic] ML 不可用，��级为规则路网算法")

    placed: list = []
    slots: list[BuildingSlot] = []

    # ── 2. 中心锚点 ───────────────────────────────────────────
    wmin, wmax, dmin, dmax = _STYLE_SIZE.get(style, _DEFAULT_SIZE)
    aw = max(wmax * 1.15, 14.0)
    ad = max(dmax * 1.15, 11.0)
    ax, az = cx - aw / 2, cz - ad / 2
    placed.append(_obb_polygon(cx, cz, aw, ad, 0.0))
    slots.append(BuildingSlot(ax, az, aw, ad, "rect", 0.0, is_anchor=True))

    if use_ml:
        # ML 坐标 nx/ny 是 ±20m 范围，缩放到场景的 60% 区域
        ml_xs = [b["nx"] for b in ml_positions]
        ml_ys = [b["ny"] for b in ml_positions]
        ml_span = max(max(ml_xs) - min(ml_xs), max(ml_ys) - min(ml_ys), 1.0)
        target_span = min(area_w, area_d) * 0.6
        scale = target_span / ml_span

        # ── 3a. ML 路径：从缩放后的位置推断路网 ──────────────────
        scaled_positions = [{"nx": b["nx"] * scale, "ny": b["ny"] * scale} for b in ml_positions]
        roads = _infer_roads_from_positions(scaled_positions, cx, cz, area_w, area_d)

        for b in ml_positions:
            bx = b["nx"] * scale + area_w / 2
            bz = b["ny"] * scale + area_d / 2
            w  = max(6.0, min(b.get("length_m", 10.0), 14.0))
            d  = max(5.0, min(b.get("width_m", 8.0), 11.0))

            # 朝向：对齐最近道路 ±10°
            if roads:
                yaw, _ = _nearest_road_dir(bx, bz, roads)
                yaw += rng.uniform(-10, 10) * variation
            else:
                yaw = b.get("orientation_deg", 0.0)

            # 碰撞检测
            if _obb_collides(bx, bz, w, d, yaw, placed, safety=1.5):
                # 尝试缩小
                w *= 0.8
                d *= 0.8
                if _obb_collides(bx, bz, w, d, yaw, placed, safety=1.0):
                    continue

            x_off = bx - w / 2
            z_off = bz - d / 2
            if x_off < margin or z_off < margin or x_off + w > area_w - margin or z_off + d > area_d - margin:
                continue

            fp = _pick_footprint_type(variation, rng)
            placed.append(_obb_polygon(bx, bz, w, d, yaw))
            slots.append(BuildingSlot(x_off, z_off, w, d, fp, yaw_deg=yaw))

    else:
        # ── 3b. 降级：规则贝塞尔路网 + 沿路排列 ────────────────
        n_roads = min(3, max(1, n // 5))
        roads = _gen_village_roads(cx, cz, area_w, area_d, n_roads, rng)
        budget = n - 1
        road_gap = 1.5
        building_gap = 1.5
        avg_w = (wmin + wmax) / 2

        place_points = []
        for ri, road in enumerate(roads):
            step = 0.0
            for i in range(len(road) - 1):
                seg_dx = road[i+1][0] - road[i][0]
                seg_dz = road[i+1][1] - road[i][1]
                seg_len = math.hypot(seg_dx, seg_dz)
                step += seg_len
                if step >= avg_w + building_gap:
                    step = 0
                    if seg_len < 0.1:
                        continue
                    nx, nz = -seg_dz / seg_len, seg_dx / seg_len
                    px, pz = road[i][0], road[i][1]
                    offset = avg_w * 0.6 + road_gap
                    for side in (-1, 1):
                        bx = px + side * nx * offset
                        bz = pz + side * nz * offset
                        if margin < bx < area_w - margin and margin < bz < area_d - margin:
                            place_points.append((bx, bz, ri))

        rng.shuffle(place_points)
        for bx, bz, ri in place_points:
            if len(slots) - 1 >= budget:
                break
            w, d = _building_size(style, variation, rng)
            yaw, _ = _nearest_road_dir(bx, bz, roads)
            yaw += rng.uniform(-10, 10) * variation
            if _obb_collides(bx, bz, w, d, yaw, placed, safety=building_gap):
                continue
            x_off = bx - w / 2
            z_off = bz - d / 2
            if x_off < 0 or z_off < 0 or x_off + w > area_w or z_off + d > area_d:
                continue
            fp = _pick_footprint_type(variation, rng)
            placed.append(_obb_polygon(bx, bz, w, d, yaw))
            slots.append(BuildingSlot(x_off, z_off, w, d, fp, yaw_deg=yaw))

    # ── 4. 保存路网数据供道路 mesh 生成使用 ────────────────────
    layout_organic._last_roads = roads

    return slots


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
    return m


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
                      style: str) -> list:
    """street 布局：两排建筑之间的路面（略高于地面，颜色深 20%）。"""
    street_w = max(6.0, area_d * 0.22)
    side_d   = (area_d - street_w) / 2
    gc = _ground_color_for_style(style)
    road_c = [max(0, int(gc[0]*0.8)), max(0, int(gc[1]*0.8)),
              max(0, int(gc[2]*0.8)), 255]
    return [_colored_box(
        [area_w + 4.0, 0.05, street_w],
        [area_w / 2, 0.025, side_d + street_w / 2],
        road_c)]


def _make_plaza_floor(area_w: float, area_d: float,
                      style: str) -> list:
    """plaza 布局：中心广场石板地面。"""
    plaza_r = min(area_w, area_d) * 0.28
    side = plaza_r * 2
    stone_c = [int(0.60*255), int(0.58*255), int(0.55*255), 255]
    return [_colored_box(
        [side, 0.05, side],
        [area_w / 2, 0.025, area_d / 2],
        stone_c)]


# ─── 围墙 ─────────────────────────────────────────────────────

_WALL_STYLES = {"medieval", "medieval_keep", "medieval_chapel",
                "fantasy", "fantasy_dungeon", "fantasy_palace",
                "horror", "horror_asylum", "horror_crypt"}

def _make_perimeter_wall(area_w: float, area_d: float,
                         style: str, params: dict) -> list:
    """
    在场景边界内侧生成一圈围墙。
    高度 2.5m，厚度 0.5m，前后各留一个城门（6m 宽）。
    """
    if style not in _WALL_STYLES:
        return []

    h, t = 2.5, 0.5
    palette = _derive_palette(style, params)
    wc = palette["wall"]
    # 围墙颜色比建筑墙暗 10%
    c = [max(0, int(wc[0]*0.9)), max(0, int(wc[1]*0.9)),
         max(0, int(wc[2]*0.9)), wc[3]]

    inset = 2.0           # 围墙离场景边缘距离
    gate_w = 6.0          # 城门宽度
    meshes = []

    # 前墙（Z=inset）—— 中间留城门
    front_z = inset
    left_w = (area_w - gate_w) / 2 - inset
    if left_w > 1:
        meshes.append(_colored_box(
            [left_w, h, t],
            [inset + left_w / 2, h / 2, front_z], c))
    right_start = (area_w + gate_w) / 2
    right_w = area_w - right_start - inset
    if right_w > 1:
        meshes.append(_colored_box(
            [right_w, h, t],
            [right_start + right_w / 2, h / 2, front_z], c))

    # 后墙（Z=area_d-inset）—— 完整
    meshes.append(_colored_box(
        [area_w - inset * 2, h, t],
        [area_w / 2, h / 2, area_d - inset], c))

    # 左墙（X=inset）
    wall_d = area_d - inset * 2 - t
    meshes.append(_colored_box(
        [t, h, wall_d],
        [inset, h / 2, area_d / 2], c))

    # 右墙（X=area_w-inset）
    meshes.append(_colored_box(
        [t, h, wall_d],
        [area_w - inset, h / 2, area_d / 2], c))

    # 城门门柱（前门两侧各一个加高方柱）
    pillar_w = 0.8
    pillar_h = h + 1.0
    for px in [(area_w - gate_w) / 2, (area_w + gate_w) / 2]:
        meshes.append(_colored_box(
            [pillar_w, pillar_h, pillar_w],
            [px, pillar_h / 2, front_z], c))

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
        street_w = max(6.0, area_d * 0.22)
        side_d = (area_d - street_w) / 2
        # 路灯沿街道两侧
        for z_line in [side_d + 0.5, side_d + street_w - 0.5]:
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


# ─── 主生成器 ───────────────────────────────────────────────────

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
) -> trimesh.Scene:
    """
    生成完整关卡场景。

    Parameters
    ----------
    style         : 风格名称（如 "medieval"）
    layout_type   : "grid" | "street" | "plaza" | "random"
    building_count: 建筑数量 5~20
    area_size     : 场景尺寸（m），正方形；或通过 area_w/area_d 指定矩形
    variation     : 变体程度 0.0~1.0
    seed          : 随机种子
    params_path   : trained_style_params.json 路径（None 时自动查找）
    output_path   : 输出 GLB 路径（None 时不保存）
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

    aw = area_w or area_size
    ad = area_d or area_size
    building_count = max(1, min(40, building_count))

    # 锚点风格参数（organic 布局使用）
    anchor_style  = _ANCHOR_STYLE.get(style, style)
    anchor_params = styles_data.get(anchor_style, styles_data[style])["params"]

    # ── 选择布局 ──────────────────────────────────────────────
    if min_gap is not None:
        set_gap(min_gap)

    layout_fn = {
        "grid":    layout_grid,
        "street":  layout_street,
        "plaza":   layout_plaza,
        "random":  layout_random,
        "organic": layout_organic,
    }.get(layout_type)
    if layout_fn is None:
        raise ValueError(
            f"未知布局类型: {layout_type}，可选: grid/street/plaza/random/organic")

    slots = layout_fn(building_count, aw, ad, style, variation, rng)

    # ── 构建场景 ──────────────────────────────────────────────
    scene = trimesh.Scene()

    # 地面
    ground = _make_ground_plane(aw, ad, style, params)
    scene.add_geometry(ground, node_name="ground_plane")

    # 道路 / 广场
    if layout_type == "street":
        for ri, rm in enumerate(_make_street_road(aw, ad, style)):
            scene.add_geometry(rm, node_name=f"road_{ri:02d}")
    elif layout_type == "plaza":
        for pi, pm in enumerate(_make_plaza_floor(aw, ad, style)):
            scene.add_geometry(pm, node_name=f"plaza_{pi:02d}")
    elif layout_type == "organic" and hasattr(layout_organic, '_last_roads'):
        for ri, rm in enumerate(_make_organic_roads(layout_organic._last_roads, style)):
            scene.add_geometry(rm, node_name=f"road_{ri:03d}")

    # 围墙
    for wi, wm in enumerate(_make_perimeter_wall(aw, ad, style, params)):
        scene.add_geometry(wm, node_name=f"wall_{wi:02d}")

    # 路灯
    for li, lm in enumerate(_make_street_lamps(aw, ad, style, params, layout_type)):
        scene.add_geometry(lm, node_name=f"lamp_{li:03d}")

    stats = {"buildings": 0, "total_footprint_m2": 0.0, "style": style,
             "layout": layout_type, "area": f"{aw:.0f}m × {ad:.0f}m",
             "variation": variation}

    for bi, slot in enumerate(slots):
        # 锚点建筑使用专属风格参数（不加变体）；其余正常变体
        if slot.is_anchor:
            bparams  = anchor_params
            bpalette = _derive_palette(anchor_style, anchor_params)
        else:
            bparams  = _vary_params(params, variation, rng)
            bpalette = _derive_palette(style, bparams)

        # 生成平面轮廓（始终保持局部坐标不旋转，旋转通过 mesh 变换完成）
        fp_local = _make_footprint(slot.fp_type, slot.w, slot.d, rng)

        # 生成建筑 mesh（在局部坐标 x_off=0, z_off=0 处，轮廓不旋转）
        try:
            room_meshes = gl.build_room(
                bparams, bpalette,
                x_off=0.0, z_off=0.0,
                footprint=fp_local,
            )

            # 如果需要旋转，对所有 mesh 应用世界变换（旋转+平移）
            # 旋转中心 = 建筑局部中心 (w/2, 0, d/2)
            if abs(slot.yaw_deg) > 0.01:
                import trimesh.transformations as TF
                cx, cz = slot.w / 2, slot.d / 2
                # 绕局部中心旋转（Shapely 逆时针 → Y轴正方向旋转）
                rot = TF.rotation_matrix(
                    math.radians(slot.yaw_deg), [0, 1, 0], point=[cx, 0, cz])
                for mesh in room_meshes:
                    mesh.apply_transform(rot)

            # 地基高度随机变化 ±0.3m → 建筑不全在同一水平面
            y_offset = float(rng.uniform(-0.3, 0.3)) * variation

            # 平移到世界位置（含地基高度）
            for mesh in room_meshes:
                mesh.apply_translation([slot.x_off, y_offset, slot.z_off])

            for mi, mesh in enumerate(room_meshes):
                scene.add_geometry(mesh, node_name=f"b{bi:02d}_{slot.fp_type}_{mi:03d}")

            fp_area = fp_local.area
            stats["buildings"] += 1
            stats["total_footprint_m2"] += fp_area

        except Exception as e:
            print(f"  [警告] 建筑 {bi} 生成失败: {e}")
            continue

    # ── 打印统计 ──────────────────────────────────────────────
    n_faces = sum(len(g.faces) for g in scene.geometry.values())
    print(f"\n  布局     : {layout_type}")
    print(f"  风格     : {style}")
    print(f"  场景面积 : {stats['area']}")
    print(f"  建筑数量 : {stats['buildings']}")
    print(f"  建筑总面积: {stats['total_footprint_m2']:.1f} m²")
    print(f"  场地占用率: {stats['total_footprint_m2']/(aw*ad)*100:.1f}%")
    print(f"  总面数   : {n_faces:,}")

    if layout_type == "organic":
        print(f"  锚点风格 : {anchor_style}")
    if output_path:
        scene.export(output_path)
        print(f"  输出     : {output_path}")

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
