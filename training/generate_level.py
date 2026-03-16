"""
LevelSmith GLB 关卡生成器

从 trained_style_params.json 生成 trained.glb，
对比默认参数生成 baseline.glb。

支持多边形平面轮廓（矩形、L形、U形），使用 shapely + trimesh.creation.extrude_polygon。
"""

import json
import sys
import math
import numpy as np
from pathlib import Path

try:
    import trimesh
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "trimesh"])
    import trimesh

try:
    from shapely.geometry import Polygon, box as shapely_box
    from shapely.affinity import translate as shapely_translate
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shapely"])
    from shapely.geometry import Polygon, box as shapely_box
    from shapely.affinity import translate as shapely_translate

# ─── 布局常量 ──────────────────────────────────────────────────
ROOM_W = 12.0   # 房间宽度 (X 轴)
ROOM_D =  8.0   # 房间深度 (Z 轴)
GAP    =  4.0   # 房间间距

STYLES = ["medieval", "modern", "industrial"]

# ─── 颜色方案 (RGBA uint8) ────────────────────────────────────
PALETTES = {
    "medieval": {
        "floor":    [115,  95,  75, 255],
        "ceiling":  [135, 115,  90, 255],
        "wall":     [162, 146, 122, 255],
        "door":     [ 88,  58,  28, 255],
        "window":   [155, 195, 215, 180],
        "internal": [148, 133, 110, 255],
        "ground":   [ 90,  80,  65, 255],
    },
    "modern": {
        "floor":    [215, 215, 215, 255],
        "ceiling":  [242, 242, 242, 255],
        "wall":     [198, 198, 198, 255],
        "door":     [ 68,  68,  68, 255],
        "window":   [135, 195, 228, 180],
        "internal": [182, 182, 182, 255],
        "ground":   [160, 160, 155, 255],
    },
    "industrial": {
        "floor":    [ 72,  72,  68, 255],
        "ceiling":  [ 52,  52,  48, 255],
        "wall":     [ 92,  88,  82, 255],
        "door":     [158,  82,  38, 255],
        "window":   [112, 138, 112, 180],
        "internal": [ 82,  78,  72, 255],
        "ground":   [ 55,  55,  52, 255],
    },
    "baseline": {
        "floor":    [172, 158, 142, 255],
        "ceiling":  [192, 182, 172, 255],
        "wall":     [182, 172, 162, 255],
        "door":     [ 98,  78,  58, 255],
        "window":   [148, 175, 200, 180],
        "internal": [168, 160, 152, 255],
        "ground":   [135, 128, 118, 255],
    },
}

# ─── 默认参数 (baseline) ──────────────────────────────────────
BASELINE_PARAMS = {
    "height_range":    [2.5, 3.5],
    "wall_thickness":  0.30,
    "floor_thickness": 0.15,
    "door_spec":       {"width": 0.90, "height": 2.10},
    "win_spec":        {"width": 1.20, "height": 1.20, "density": 0.40},
    "subdivision":     2,
}


# ─── 平面轮廓工厂 ──────────────────────────────────────────────

def make_rect_footprint(w=ROOM_W, d=ROOM_D):
    """矩形平面轮廓（原始矩形，用于向后兼容）"""
    return shapely_box(0, 0, w, d)


def make_l_footprint(w=ROOM_W, d=ROOM_D, cut_frac_x=0.45, cut_frac_z=0.45):
    """
    L形平面轮廓：在矩形右上角切去一个矩形。
    cut_frac_x: 切去部分占宽度的比例
    cut_frac_z: 切去部分占深度的比例
    """
    rect = shapely_box(0, 0, w, d)
    cut  = shapely_box(w * (1 - cut_frac_x), d * (1 - cut_frac_z), w, d)
    return rect.difference(cut)


def make_u_footprint(w=ROOM_W, d=ROOM_D, notch_frac_x=0.4, notch_frac_z=0.55):
    """
    U形平面轮廓：在矩形中央下方切去一个矩形（开口朝前）。
    notch_frac_x: 缺口占宽度的比例（居中）
    notch_frac_z: 缺口占深度的比例（从前沿切入）
    """
    rect = shapely_box(0, 0, w, d)
    nx0  = w * (0.5 - notch_frac_x / 2)
    nx1  = w * (0.5 + notch_frac_x / 2)
    cut  = shapely_box(nx0, 0, nx1, d * notch_frac_z)
    return rect.difference(cut)


# ─── 几何工具 ─────────────────────────────────────────────────

def make_box(size, center, color):
    """创建带颜色的 box mesh"""
    b = trimesh.creation.box(extents=np.array(size, dtype=float))
    b.apply_translation(np.array(center, dtype=float))
    c = np.array(color, dtype=np.uint8)
    b.visual.face_colors = np.tile(c, (len(b.faces), 1))
    return b


def make_extruded_polygon(polygon, height, color):
    """
    将 shapely Polygon 拉伸为 3D mesh，颜色填充。
    输出坐标系：X=右, Y=上, Z=前（深度）。
    extrude_polygon 在 XY 平面画轮廓，沿 Z 拉伸。
    旋转 +90° 绕 X 轴：shapely XY → 世界 XZ，拉伸 Z → 世界 -Y。
    旋转后 mesh 顶面在 Y=0，底面在 Y=-height。
    """
    mesh = trimesh.creation.extrude_polygon(polygon, height)
    # +90° 绕 X：new_Y = -old_Z, new_Z = old_Y
    rot = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    mesh.apply_transform(rot)
    c = np.array(color, dtype=np.uint8)
    mesh.visual.face_colors = np.tile(c, (len(mesh.faces), 1))
    return mesh


def _wall_segments(total_len, openings):
    """
    将墙长切割成若干区段，返回 [(x0, x1, opening_or_None), ...]
    opening 含 {"x", "w", "y", "h"}
    """
    xs = sorted(set(
        [0.0] +
        [max(0.0, o["x"]) for o in openings] +
        [min(total_len, o["x"] + o["w"]) for o in openings] +
        [total_len]
    ))
    segments = []
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x1 - x0 < 1e-4:
            continue
        op = next(
            (o for o in openings
             if o["x"] <= x0 + 1e-4 and o["x"] + o["w"] >= x1 - 1e-4),
            None
        )
        segments.append((x0, x1, op))
    return segments


def build_x_wall(total_w, height, thickness, openings, color, wx, wz):
    """
    沿 X 方向的墙体（前/后墙），支持开口。
    wx: 世界 X 起点偏移, wz: 世界 Z 中心
    """
    panels = []
    for x0, x1, op in _wall_segments(total_w, openings):
        seg_w = x1 - x0
        cx = wx + (x0 + x1) / 2
        if op is None:
            panels.append(make_box([seg_w, height, thickness], [cx, height / 2, wz], color))
        else:
            y0 = max(0.0, op["y"])
            y1 = min(height, op["y"] + op["h"])
            if y0 > 1e-4:
                panels.append(make_box([seg_w, y0, thickness], [cx, y0 / 2, wz], color))
            above = height - y1
            if above > 1e-4:
                panels.append(make_box([seg_w, above, thickness], [cx, y1 + above / 2, wz], color))
    return panels


def build_z_wall(total_d, height, thickness, openings_z, color, wx, wz):
    """
    沿 Z 方向的墙体（左/右墙），支持开口。
    openings_z: [{"z", "w", "y", "h"}, ...]
    wx: 世界 X 中心, wz: 世界 Z 起点偏移
    """
    ops_as_x = [{"x": o["z"], "w": o["w"], "y": o["y"], "h": o["h"]} for o in openings_z]
    panels = []
    for x0, x1, op in _wall_segments(total_d, ops_as_x):
        seg_len = x1 - x0
        cz = wz + (x0 + x1) / 2
        if op is None:
            panels.append(make_box([thickness, height, seg_len], [wx, height / 2, cz], color))
        else:
            y0 = max(0.0, op["y"])
            y1 = min(height, op["y"] + op["h"])
            if y0 > 1e-4:
                panels.append(make_box([thickness, y0, seg_len], [wx, y0 / 2, cz], color))
            above = height - y1
            if above > 1e-4:
                panels.append(make_box([thickness, above, seg_len], [wx, y1 + above / 2, cz], color))
    return panels


def build_edge_wall(p0, p1, height, wall_t, openings, color, x_off, z_off):
    """
    沿轮廓任意一条边建墙。
    p0, p1: (x, z) 局部坐标（shapely 坐标系，即 X=右, Z=前，Y=上）
    openings: [{"x":沿边距离, "w":宽, "y":下沿高, "h":高度}, ...]
    返回 mesh 列表
    """
    dx = p1[0] - p0[0]
    dz = p1[1] - p0[1]
    edge_len = math.hypot(dx, dz)
    if edge_len < 1e-4:
        return []

    # 单位方向向量（沿边）和法线（向内）
    ux, uz = dx / edge_len, dz / edge_len

    panels = []
    for seg_x0, seg_x1, op in _wall_segments(edge_len, openings):
        seg_len = seg_x1 - seg_x0
        # 段中心（沿边方向，局部坐标）
        mid_along = (seg_x0 + seg_x1) / 2
        # 段在世界坐标中的中心（XZ）
        cx_local = p0[0] + ux * mid_along
        cz_local = p0[1] + uz * mid_along
        cx_world = x_off + cx_local
        cz_world = z_off + cz_local

        # 旋转角度（让墙板沿边方向）
        angle = math.atan2(dz, dx)

        if op is None:
            mesh = trimesh.creation.box(extents=[seg_len, height, wall_t])
            mesh.apply_translation([0, height / 2, 0])
        else:
            # 带开口：分三块（下、上）
            sub_meshes = []
            y0 = max(0.0, op["y"])
            y1 = min(height, op["y"] + op["h"])
            if y0 > 1e-4:
                m = trimesh.creation.box(extents=[seg_len, y0, wall_t])
                m.apply_translation([0, y0 / 2, 0])
                sub_meshes.append(m)
            above = height - y1
            if above > 1e-4:
                m = trimesh.creation.box(extents=[seg_len, above, wall_t])
                m.apply_translation([0, y1 + above / 2, 0])
                sub_meshes.append(m)
            if not sub_meshes:
                continue
            mesh = trimesh.util.concatenate(sub_meshes)

        # 旋转墙板使其沿边方向
        rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        mesh.apply_transform(rot)
        mesh.apply_translation([cx_world, 0, cz_world])

        c = np.array(color, dtype=np.uint8)
        mesh.visual.face_colors = np.tile(c, (len(mesh.faces), 1))
        panels.append(mesh)

        # 添加玻璃（窗户开口）
        if op is not None and op.get("is_window"):
            g_len = seg_len - 0.06
            g_h   = (op["h"] - 0.06)
            if g_len > 0 and g_h > 0:
                glass = trimesh.creation.box(extents=[g_len, g_h, 0.02])
                g_cy  = op["y"] + op["h"] / 2
                glass.apply_translation([0, g_cy, 0])
                glass.apply_transform(rot)
                glass.apply_translation([cx_world, 0, cz_world])
                gc = np.array(color, dtype=np.uint8)  # will be overridden
                # will be assigned window color by caller
                panels.append(("glass", glass))

    return panels


# ─── 窗户布置 ──────────────────────────────────────────────────

def place_windows_x(wall_len, height, win_w, win_h, density, subdiv, sill=0.9):
    """沿 X 方向墙体计算窗户开口列表"""
    if density < 0.12:
        return []
    n = max(1, round(density * subdiv))
    n = min(n, max(1, int(wall_len / (win_w + 0.5))))
    gap = max(0.35, (wall_len - n * win_w) / (n + 1))
    result, x = [], gap
    for _ in range(n):
        if x + win_w > wall_len - 0.15:
            break
        h_actual = min(win_h, height - sill - 0.1)
        if h_actual > 0.1:
            result.append({"x": x, "w": win_w, "y": sill, "h": h_actual})
        x += win_w + gap
    return result


def place_windows_z(wall_dep, height, win_w, win_h, density, subdiv, sill=0.9):
    """沿 Z 方向墙体计算窗户开口列表（密度打七折）"""
    if density < 0.12:
        return []
    n = max(1, round(density * subdiv * 0.7))
    n = min(n, max(1, int(wall_dep / (win_w + 0.5))))
    gap = max(0.35, (wall_dep - n * win_w) / (n + 1))
    result, z = [], gap
    for _ in range(n):
        if z + win_w > wall_dep - 0.15:
            break
        h_actual = min(win_h, height - sill - 0.1)
        if h_actual > 0.1:
            result.append({"z": z, "w": win_w, "y": sill, "h": h_actual})
        z += win_w + gap
    return result


def place_windows_edge(edge_len, height, win_w, win_h, density, subdiv, sill=0.9):
    """沿任意轮廓边计算窗户开口（按边长缩放密度）"""
    if density < 0.12 or edge_len < win_w + 0.5:
        return []
    n = max(1, round(density * subdiv * (edge_len / ROOM_W)))
    n = min(n, max(1, int(edge_len / (win_w + 0.5))))
    gap = max(0.35, (edge_len - n * win_w) / (n + 1))
    result, x = [], gap
    for _ in range(n):
        if x + win_w > edge_len - 0.15:
            break
        h_actual = min(win_h, height - sill - 0.1)
        if h_actual > 0.1:
            result.append({"x": x, "w": win_w, "y": sill, "h": h_actual, "is_window": True})
        x += win_w + gap
    return result


# ─── 玻璃填充 ─────────────────────────────────────────────────

def add_glass_x(openings, wz, wx_base, palette, meshes):
    """在 X 方向墙体的窗口位置放玻璃板"""
    for op in openings:
        h = op["h"]
        meshes.append(make_box(
            [op["w"] - 0.06, h - 0.06, 0.02],
            [wx_base + op["x"] + op["w"] / 2, op["y"] + h / 2, wz],
            palette["window"]
        ))


def add_glass_z(openings, wx, wz_base, palette, meshes):
    """在 Z 方向墙体的窗口位置放玻璃板"""
    for op in openings:
        h = op["h"]
        meshes.append(make_box(
            [0.02, h - 0.06, op["w"] - 0.06],
            [wx, op["y"] + h / 2, wz_base + op["z"] + op["w"] / 2],
            palette["window"]
        ))


# ─── 门框 ─────────────────────────────────────────────────────

def add_door_frame(door_x, door_w, door_h, wall_t, wx, wz, palette, meshes):
    """添加门框（三条边框），用于 X 方向墙体"""
    f = 0.07
    cx = wx + door_x + door_w / 2
    meshes.append(make_box([f, door_h + f, wall_t + 0.04], [wx + door_x,           door_h / 2,  wz], palette["door"]))
    meshes.append(make_box([f, door_h + f, wall_t + 0.04], [wx + door_x + door_w,  door_h / 2,  wz], palette["door"]))
    meshes.append(make_box([door_w + f * 2, f, wall_t + 0.04], [cx, door_h + f / 2, wz], palette["door"]))


# ─── 多边形轮廓墙体生成 ────────────────────────────────────────

def _classify_edge(p0, p1):
    """
    对轮廓边分类：
    返回 ('front'|'back'|'left'|'right'|'other', edge_len)
    front = z 最小边（z≈0），back = z 最大边，left = x 最小边，right = x 最大边
    """
    dx = p1[0] - p0[0]
    dz = p1[1] - p0[1]
    edge_len = math.hypot(dx, dz)
    tol = 0.5  # 判断是否为轴对齐边的阈值

    min_z = min(p0[1], p1[1])
    max_z = max(p0[1], p1[1])
    min_x = min(p0[0], p1[0])
    max_x = max(p0[0], p1[0])

    if abs(dz) < tol and min_z < tol:
        return "front", edge_len
    if abs(dz) < tol and min_z > ROOM_D * 0.4:
        return "back", edge_len
    if abs(dx) < tol and min_x < tol:
        return "left", edge_len
    if abs(dx) < tol and min_x > ROOM_W * 0.4:
        return "right", edge_len
    return "other", edge_len


def build_polygon_walls(footprint, height, wall_t, params, palette, x_off, z_off):
    """
    遍历 shapely Polygon 轮廓的所有边，为每条边生成墙体。
    - 前墙（z≈0）放门，其余边根据密度放窗
    """
    door_w  = float(params["door_spec"]["width"])
    door_h  = min(float(params["door_spec"]["height"]), height - 0.15)
    win_w   = float(params["win_spec"]["width"])
    win_h   = float(params["win_spec"]["height"])
    win_d   = float(params["win_spec"]["density"])
    subdiv  = int(params["subdivision"])

    coords = list(footprint.exterior.coords)
    meshes = []
    door_placed = False

    for i in range(len(coords) - 1):
        p0 = coords[i]
        p1 = coords[i + 1]
        dx = p1[0] - p0[0]
        dz = p1[1] - p0[1]
        edge_len = math.hypot(dx, dz)
        if edge_len < 1e-4:
            continue

        edge_type, _ = _classify_edge(p0, p1)

        # 确定该边朝向是否为水平轴对齐（简化处理）
        is_x_aligned = abs(dz) < 0.1  # 沿 X 方向（前/后墙）
        is_z_aligned = abs(dx) < 0.1  # 沿 Z 方向（左/右墙）

        openings = []
        is_front = (edge_type == "front") and not door_placed

        if is_front and edge_len >= door_w + 0.5:
            # 放门
            door_x = (edge_len - door_w) / 2
            openings = [{"x": door_x, "w": door_w, "y": 0.0, "h": door_h}]
            door_placed = True
        else:
            # 放窗
            wins = place_windows_edge(edge_len, height, win_w, win_h, win_d, subdiv)
            openings = wins

        if is_x_aligned:
            # X 轴对齐的墙：使用简化的 build_x_wall
            # p0.x 较小时方向为正 X，否则为负 X
            wx_start = x_off + min(p0[0], p1[0])
            wz_center = z_off + p0[1]  # z 恒定
            wall_meshes = build_x_wall(edge_len, height, wall_t, openings, palette["wall"], wx_start, wz_center)
            meshes.extend(wall_meshes)

            if is_front and door_placed:
                door_x_abs = (edge_len - door_w) / 2
                add_door_frame(door_x_abs, door_w, door_h, wall_t, wx_start, wz_center, palette, meshes)
            else:
                # 玻璃
                glass_wins = [o for o in openings if o.get("is_window")]
                add_glass_x(glass_wins, wz_center, wx_start, palette, meshes)

        elif is_z_aligned:
            # Z 轴对齐的墙：使用简化的 build_z_wall
            wx_center = x_off + p0[0]
            wz_start  = z_off + min(p0[1], p1[1])
            # 转换开口格式 x→z
            z_openings = [{"z": o["x"], "w": o["w"], "y": o["y"], "h": o["h"]} for o in openings]
            wall_meshes = build_z_wall(edge_len, height, wall_t, z_openings, palette["wall"], wx_center, wz_start)
            meshes.extend(wall_meshes)

            glass_wins_z = [{"z": o["x"], "w": o["w"], "y": o["y"], "h": o["h"]}
                            for o in openings if o.get("is_window")]
            add_glass_z(glass_wins_z, wx_center, wz_start, palette, meshes)

        else:
            # 斜边：生成旋转的墙板
            angle = math.atan2(dz, dx)
            for seg_x0, seg_x1, op in _wall_segments(edge_len, openings):
                seg_len = seg_x1 - seg_x0
                ux, uz = dx / edge_len, dz / edge_len
                mid_along = (seg_x0 + seg_x1) / 2
                cx_world = x_off + p0[0] + ux * mid_along
                cz_world = z_off + p0[1] + uz * mid_along

                if op is None:
                    m = trimesh.creation.box(extents=[seg_len, height, wall_t])
                    m.apply_translation([0, height / 2, 0])
                    rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
                    m.apply_transform(rot)
                    m.apply_translation([cx_world, 0, cz_world])
                    c = np.array(palette["wall"], dtype=np.uint8)
                    m.visual.face_colors = np.tile(c, (len(m.faces), 1))
                    meshes.append(m)
                else:
                    y0 = max(0.0, op["y"])
                    y1 = min(height, op["y"] + op["h"])
                    for (bot, top) in [(0, y0), (y1, height)]:
                        h_seg = top - bot
                        if h_seg < 1e-4:
                            continue
                        m = trimesh.creation.box(extents=[seg_len, h_seg, wall_t])
                        m.apply_translation([0, bot + h_seg / 2, 0])
                        rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
                        m.apply_transform(rot)
                        m.apply_translation([cx_world, 0, cz_world])
                        c = np.array(palette["wall"], dtype=np.uint8)
                        m.visual.face_colors = np.tile(c, (len(m.faces), 1))
                        meshes.append(m)

    return meshes


# ─── 地面垫板（覆盖轮廓包围盒） ────────────────────────────────

def build_ground_pad(footprint, floor_t, palette, x_off, z_off):
    """地面垫板，用轮廓包围盒稍微扩展"""
    bounds = footprint.bounds  # (minx, miny, maxx, maxy)
    w = bounds[2] - bounds[0] + 0.2
    d = bounds[3] - bounds[1] + 0.2
    cx = x_off + (bounds[0] + bounds[2]) / 2
    cz = z_off + (bounds[1] + bounds[3]) / 2
    return make_box([w, 0.05, d], [cx, -floor_t - 0.025, cz], palette["ground"])


# ─── 单间房间（多边形轮廓版） ────────────────────────────────────

def build_room(params, palette, x_off=0.0, z_off=0.0, footprint=None):
    """
    生成单间房间的所有 mesh。
    footprint: shapely Polygon（局部坐标，原点在房间左前角），
               None 时默认使用 ROOM_W × ROOM_D 矩形（向后兼容）。
    """
    if footprint is None:
        footprint = make_rect_footprint()

    height   = float(params["height_range"][1])
    wall_t   = float(params["wall_thickness"])
    floor_t  = float(params["floor_thickness"])
    door_w   = float(params["door_spec"]["width"])
    door_h   = min(float(params["door_spec"]["height"]), height - 0.15)
    win_w    = float(params["win_spec"]["width"])
    win_h    = float(params["win_spec"]["height"])
    win_d    = float(params["win_spec"]["density"])
    subdiv   = int(params["subdivision"])

    meshes = []

    # ── 地面垫板 ──────────────────────────────────────────────
    meshes.append(build_ground_pad(footprint, floor_t, palette, x_off, z_off))

    # ── 地板（拉伸多边形） ────────────────────────────────────
    # make_extruded_polygon 旋转后：顶面 Y=0，底面 Y=-floor_t
    floor_mesh = make_extruded_polygon(footprint, floor_t, palette["floor"])
    floor_mesh.apply_translation([x_off, 0, z_off])
    meshes.append(floor_mesh)

    # ── 天花板（拉伸多边形） ──────────────────────────────────
    # 旋转后：顶面 Y=0，底面 Y=-floor_t；平移到高度位置
    ceil_mesh = make_extruded_polygon(footprint, floor_t, palette["ceiling"])
    ceil_mesh.apply_translation([x_off, height + floor_t, z_off])
    meshes.append(ceil_mesh)

    # ── 外墙（遍历轮廓边） ────────────────────────────────────
    wall_meshes = build_polygon_walls(footprint, height, wall_t, params, palette, x_off, z_off)
    meshes.extend(wall_meshes)

    # ── 内部分隔墙（沿 Z 均分，各有一扇门） ─────────────────
    if subdiv > 1:
        bounds = footprint.bounds
        fp_w = bounds[2] - bounds[0]
        fp_d = bounds[3] - bounds[1]

        int_wall_t = wall_t * 0.65
        int_door_w = min(door_w * 0.85, fp_w * 0.35)
        int_door_h = door_h * 0.88
        z_step = fp_d / subdiv

        for k in range(1, subdiv):
            wz_local = bounds[1] + k * z_step
            wz_world = z_off + wz_local

            # 求分隔墙与轮廓在该 z 位置的交叉范围
            cut_line = footprint.intersection(
                shapely_box(bounds[0] - 1, wz_local - 0.01, bounds[2] + 1, wz_local + 0.01)
            )
            if cut_line.is_empty:
                int_w   = fp_w
                x_start = x_off + bounds[0]
            else:
                cb      = cut_line.bounds
                int_w   = cb[2] - cb[0]
                x_start = x_off + cb[0]
            if int_w < int_door_w + 0.3:
                continue

            int_door_x = (int_w - int_door_w) / 2
            int_ops = [{"x": int_door_x, "w": int_door_w, "y": 0.0, "h": int_door_h}]
            meshes += build_x_wall(int_w, height, int_wall_t, int_ops, palette["internal"], x_start, wz_world)
            add_door_frame(int_door_x, int_door_w, int_door_h, int_wall_t, x_start, wz_world, palette, meshes)

    return meshes


# ─── 场景构建 ─────────────────────────────────────────────────

def build_scene(style_params_map, use_style_palette=True, footprints=None):
    """
    生成包含多个并排房间的 trimesh.Scene。
    style_params_map: {"medieval": params, "modern": params, ...}
    footprints: dict style→shapely Polygon，None 时使用默认矩形
    """
    scene = trimesh.Scene()
    x_off = 0.0
    styles = list(style_params_map.keys())

    for style in styles:
        params   = style_params_map[style]
        palette  = PALETTES.get(style, PALETTES["baseline"]) if use_style_palette else PALETTES["baseline"]
        footprint = None
        if footprints:
            footprint = footprints.get(style)

        # 计算房间包围盒宽度用于间距
        if footprint is not None:
            room_w = footprint.bounds[2] - footprint.bounds[0]
        else:
            room_w = ROOM_W

        room = build_room(params, palette, x_off, 0.0, footprint=footprint)
        for i, mesh in enumerate(room):
            scene.add_geometry(mesh, node_name=f"{style}_{i:03d}")
        x_off += room_w + GAP

    return scene


# ─── 对比报告 ─────────────────────────────────────────────────

def print_comparison(trained_map):
    cols = ["medieval", "modern", "industrial", "baseline"]
    rows = [
        ("height_range_max", "高度 (m)"),
        ("wall_thickness",   "墙厚 (m)"),
        ("floor_thickness",  "楼板厚 (m)"),
        ("door_w",           "门宽 (m)"),
        ("door_h",           "门高 (m)"),
        ("win_w",            "窗宽 (m)"),
        ("win_h",            "窗高 (m)"),
        ("win_density",      "窗密度"),
        ("subdivision",      "内部分隔"),
    ]

    def get(params, key):
        if key == "height_range_max": return params["height_range"][1]
        if key == "wall_thickness":   return params["wall_thickness"]
        if key == "floor_thickness":  return params["floor_thickness"]
        if key == "door_w":           return params["door_spec"]["width"]
        if key == "door_h":           return params["door_spec"]["height"]
        if key == "win_w":            return params["win_spec"]["width"]
        if key == "win_h":            return params["win_spec"]["height"]
        if key == "win_density":      return params["win_spec"]["density"]
        if key == "subdivision":      return params["subdivision"]

    all_params = {**trained_map, "baseline": BASELINE_PARAMS}
    print(f"\n{'参数':<18} {'medieval':>10} {'modern':>10} {'industrial':>12} {'baseline':>10}")
    print("─" * 64)
    for key, label in rows:
        row = f"{label:<18}"
        for col in cols:
            if col not in all_params:
                continue
            v = get(all_params[col], key)
            row += f" {v:>10.3f}" if isinstance(v, float) else f" {v:>10}"
        print(row)


def export_report(trained_map, out_path):
    """输出 JSON 对比报告"""
    report = {
        "trained": {k: v for k, v in trained_map.items()},
        "baseline": BASELINE_PARAMS,
        "differences": {},
    }
    for style in STYLES:
        if style not in trained_map:
            continue
        tp = trained_map[style]
        bp = BASELINE_PARAMS
        report["differences"][style] = {
            "height_delta":    round(tp["height_range"][1] - bp["height_range"][1], 3),
            "wall_t_delta":    round(tp["wall_thickness"]  - bp["wall_thickness"],  3),
            "win_density_delta": round(tp["win_spec"]["density"] - bp["win_spec"]["density"], 3),
            "subdiv_delta":    tp["subdivision"] - bp["subdivision"],
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[报告] {out_path}")


# ─── 入口 ─────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).parent

    # 读取 trained_style_params.json
    json_path = script_dir / "trained_style_params.json"
    if not json_path.exists():
        print(f"[错误] 找不到 {json_path}，请先运行 train.py")
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    trained_map = {
        style: data["styles"][style]["params"]
        for style in STYLES
        if style in data.get("styles", {})
    }

    baseline_map = {style: BASELINE_PARAMS for style in STYLES}

    # ── 生成 trained.glb（矩形轮廓，向后兼容） ───────────────
    print("生成 trained.glb ...")
    trained_scene = build_scene(trained_map, use_style_palette=True)
    trained_path  = script_dir / "trained.glb"
    trained_scene.export(str(trained_path))
    print(f"  => {trained_path}  ({sum(len(g.faces) for g in trained_scene.geometry.values()):,} 面)")

    # ── 生成 baseline.glb ─────────────────────────────────────
    print("生成 baseline.glb ...")
    baseline_scene = build_scene(baseline_map, use_style_palette=True)
    baseline_path  = script_dir / "baseline.glb"
    baseline_scene.export(str(baseline_path))
    print(f"  => {baseline_path}  ({sum(len(g.faces) for g in baseline_scene.geometry.values()):,} 面)")

    # ── 生成 L形 测试文件 ─────────────────────────────────────
    print("\n生成 l_shape_test.glb（medieval + modern，L形轮廓）...")
    l_footprint = make_l_footprint()
    l_styles = {s: trained_map[s] for s in ["medieval", "modern"] if s in trained_map}
    l_footprints = {s: l_footprint for s in l_styles}
    l_scene = build_scene(l_styles, use_style_palette=True, footprints=l_footprints)
    l_path = script_dir / "l_shape_test.glb"
    l_scene.export(str(l_path))
    print(f"  => {l_path}  ({sum(len(g.faces) for g in l_scene.geometry.values()):,} 面)")

    # ── 生成 U形 测试文件 ─────────────────────────────────────
    print("生成 u_shape_test.glb（medieval + modern，U形轮廓）...")
    u_footprint = make_u_footprint()
    u_styles = {s: trained_map[s] for s in ["medieval", "modern"] if s in trained_map}
    u_footprints = {s: u_footprint for s in u_styles}
    u_scene = build_scene(u_styles, use_style_palette=True, footprints=u_footprints)
    u_path = script_dir / "u_shape_test.glb"
    u_scene.export(str(u_path))
    print(f"  => {u_path}  ({sum(len(g.faces) for g in u_scene.geometry.values()):,} 面)")

    # ── 输出对比报告 ──────────────────────────────────────────
    print_comparison(trained_map)
    export_report(trained_map, script_dir / "comparison_report.json")

    print("\n完成。")
    print(f"  trained.glb      — 矩形轮廓，模型推理参数")
    print(f"  baseline.glb     — 矩形轮廓，默认参数")
    print(f"  l_shape_test.glb — L形轮廓，medieval + modern")
    print(f"  u_shape_test.glb — U形轮廓，medieval + modern")


if __name__ == "__main__":
    main()
