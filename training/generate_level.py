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
    from shapely.geometry import Polygon, box as shapely_box, LineString
    from shapely.affinity import translate as shapely_translate
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shapely"])
    from shapely.geometry import Polygon, box as shapely_box, LineString
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
    """创建带颜色的 box mesh。
    只有 extents >= 0.05 的维度才做 round(3) 对齐。
    避免对薄板（门/窗）的微小维度做舍入导致退化。
    """
    extents = np.array(size, dtype=np.float64)
    center  = np.array(center, dtype=np.float64)
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center)
    # 只对 >=0.05 的维度做 round(3) → 消除墙段接缝，不伤薄板
    for ax in range(3):
        if extents[ax] >= 0.05:
            b.vertices[:, ax] = np.round(b.vertices[:, ax], 3)
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


def _r3(v):
    """Round to 3 decimal places — eliminates float seam artifacts."""
    return round(float(v), 3)


def _wall_segments(total_len, openings):
    """
    将墙长切割成若干区段，返回 [(x0, x1, opening_or_None), ...]
    opening 含 {"x", "w", "y", "h"}
    所有坐标 round(3) 保证相邻段顶点完全对齐。
    """
    total_len = _r3(total_len)
    xs = sorted(set(
        [0.0] +
        [_r3(max(0.0, o["x"])) for o in openings] +
        [_r3(min(total_len, o["x"] + o["w"])) for o in openings] +
        [total_len]
    ))
    segments = []
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x1 - x0 < 1e-4:
            continue
        op = next(
            (o for o in openings
             if _r3(o["x"]) <= x0 + 1e-4 and _r3(o["x"] + o["w"]) >= x1 - 1e-4),
            None
        )
        segments.append((x0, x1, op))
    return segments


def _boolean_wall(solid, openings_list, color):
    """
    从完整墙体 solid 中用布尔运算减去所有开口。
    返回单个 mesh（无接缝、法线连续）。
    openings_list: [trimesh.Trimesh, ...] 每个是贯穿墙体的 box
    """
    result = solid
    for hole in openings_list:
        try:
            result = trimesh.boolean.difference([result, hole], engine='manifold')
        except Exception:
            pass   # 布尔失败则跳过该开口，保留实墙
    trimesh.repair.fix_normals(result)
    c = np.array(color, dtype=np.uint8)
    result.visual.face_colors = np.tile(c, (len(result.faces), 1))
    return result


def build_x_wall(total_w, height, thickness, openings, color, wx, wz):
    """
    沿 X 方向的墙体（前/后墙）。
    用布尔运算从完整墙体中减去门窗开口 → 无接缝，法线连续。
    """
    # 完整实墙
    solid = trimesh.creation.box(extents=[total_w, height, thickness])
    solid.apply_translation([wx + total_w / 2, height / 2, wz])

    # 开口切割体（比墙厚 × 2 确保贯穿）
    holes = []
    for op in openings:
        y0 = max(0.0, op.get("y", 0.0))
        h  = op.get("h", height)
        w  = op.get("w", 1.0)
        x  = op.get("x", 0.0)
        hole = trimesh.creation.box(extents=[w, h, thickness * 2])
        hole.apply_translation([wx + x + w / 2, y0 + h / 2, wz])
        holes.append(hole)

    if not holes:
        c = np.array(color, dtype=np.uint8)
        solid.visual.face_colors = np.tile(c, (len(solid.faces), 1))
        return [solid]

    return [_boolean_wall(solid, holes, color)]


def build_z_wall(total_d, height, thickness, openings_z, color, wx, wz):
    """
    沿 Z 方向的墙体（左/右墙）。
    布尔运算：完整实墙 - 窗口开口。
    """
    solid = trimesh.creation.box(extents=[thickness, height, total_d])
    solid.apply_translation([wx, height / 2, wz + total_d / 2])

    holes = []
    for op in openings_z:
        y0 = max(0.0, op.get("y", 0.0))
        h  = op.get("h", height)
        w  = op.get("w", 1.0)
        z  = op.get("z", 0.0)
        hole = trimesh.creation.box(extents=[thickness * 2, h, w])
        hole.apply_translation([wx, y0 + h / 2, wz + z + w / 2])
        holes.append(hole)

    if not holes:
        c = np.array(color, dtype=np.uint8)
        solid.visual.face_colors = np.tile(c, (len(solid.faces), 1))
        return [solid]

    return [_boolean_wall(solid, holes, color)]


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
    if density <= 0.0:
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
    if density <= 0.0:
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
    """
    沿任意轮廓边计算窗户开口（均匀分布）。
    保证：每面外墙至少放 1 扇窗（即使 density=0）。
    """
    if edge_len < win_w + 0.5:
        return []
    # 即使 density 极低，每面墙至少 1 扇窗
    eff_density = max(density, 0.08)
    n = max(1, round(edge_len * eff_density / (win_w * 1.5)))
    n = min(n, max(1, int(edge_len / (win_w + 0.4))))
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

def add_glass_x(openings, wz, wx_base, palette, meshes, wall_t=0.3):
    """X 方向墙体窗口玻璃。
    - 尺寸：比窗洞每边小 2cm
    - 厚度：wall_t * 0.08（最少 2cm，最多 5cm）
    - Z = wz（墙体中心线），严格在 [wz-wall_t/2, wz+wall_t/2] 内
    """
    glass_t = max(0.02, min(0.05, wall_t * 0.08))
    for op in openings:
        h = op["h"]
        gw = op["w"] - 0.04    # 比窗洞每边小 2cm
        gh = h - 0.04
        if gw > 0.05 and gh > 0.05:
            meshes.append(make_box(
                [gw, gh, glass_t],
                [wx_base + op["x"] + op["w"] / 2, op["y"] + h / 2, wz],
                palette["window"]
            ))


def add_glass_z(openings, wx, wz_base, palette, meshes, wall_t=0.3):
    """Z 方向墙体窗口玻璃。
    - 尺寸：比窗洞每边小 2cm
    - 厚度：wall_t * 0.08
    - X = wx（墙体中心线）
    """
    glass_t = max(0.02, min(0.05, wall_t * 0.08))
    for op in openings:
        h = op["h"]
        gw = op["w"] - 0.04
        gh = h - 0.04
        if gw > 0.05 and gh > 0.05:
            meshes.append(make_box(
                [glass_t, gh, gw],
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


def add_door_panel(door_x, door_w, door_h, wall_t, wx, wz, palette, meshes):
    """
    最简门板：一个标准 trimesh box。
    - 尺寸：(door_w - 0.02) x (door_h - 0.02) x 0.04
    - 位置：墙面内侧 2cm 处（wz + wall_t/2 - 0.02）
    - 底边贴地（Y center = panel_h / 2）
    """
    pw = door_w - 0.02
    ph = door_h - 0.02
    pt = 0.04
    if pw < 0.1 or ph < 0.1:
        return
    cx = wx + door_x + door_w / 2
    # 墙内侧 = wz + wall_t/2，门板向内缩 2cm
    pz = wz + wall_t / 2 - 0.02
    panel = trimesh.creation.box(extents=[pw, ph, pt])
    panel.apply_translation([cx, ph / 2, pz])
    c = np.array(palette["door"], dtype=np.uint8)
    panel.visual.face_colors = np.tile(c, (len(panel.faces), 1))
    meshes.append(panel)


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
    door_h  = max(min(float(params["door_spec"]["height"]), height - 0.15),
                  min(height * 0.45, 2.8))
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
            # 放门（居中）+ 门两侧各加一扇窗
            door_x = _r3((edge_len - door_w) / 2)
            openings = [{"x": door_x, "w": door_w, "y": 0.0, "h": door_h}]
            door_placed = True
            win_sill    = 0.9
            win_h_use   = min(win_h, height - win_sill - 0.1)
            if win_h_use > 0.1:
                # 左侧窗：如果空间足够放窗（最小 win_w + 0.3m 边距）
                left_space = door_x
                fit_w = min(win_w, left_space - 0.3) if left_space > 0.6 else 0
                if fit_w > 0.3:
                    lw_x = _r3((left_space - fit_w) / 2)
                    openings.append({"x": lw_x, "w": _r3(fit_w),
                                     "y": win_sill, "h": win_h_use, "is_window": True})
                # 右侧窗
                right_start = door_x + door_w
                right_space = edge_len - right_start
                fit_w = min(win_w, right_space - 0.3) if right_space > 0.6 else 0
                if fit_w > 0.3:
                    rw_x = _r3(right_start + (right_space - fit_w) / 2)
                    openings.append({"x": rw_x, "w": _r3(fit_w),
                                     "y": win_sill, "h": win_h_use, "is_window": True})
        else:
            # 非前墙：按密度放窗（保证至少1扇）
            wins = place_windows_edge(edge_len, height, win_w, win_h, win_d, subdiv)
            openings = wins

        if is_x_aligned:
            # X 轴对齐的墙（前/后墙）
            # 向两端各延伸 wall_t/2 填补墙角缝隙
            wx_start = x_off + min(p0[0], p1[0]) - wall_t / 2
            ext_len  = edge_len + wall_t   # 延伸后总长
            wz_center = z_off + p0[1]
            # 开口坐标也需偏移 wall_t/2（因为墙起点变了）
            shifted_ops = [dict(o, x=o["x"] + wall_t / 2) for o in openings]
            wall_meshes = build_x_wall(ext_len, height, wall_t, shifted_ops, palette["wall"], wx_start, wz_center)
            meshes.extend(wall_meshes)

            if is_front and door_placed:
                door_x_abs = (edge_len - door_w) / 2 + wall_t / 2
                add_door_frame(door_x_abs, door_w, door_h, wall_t, wx_start, wz_center, palette, meshes)
                add_door_panel(door_x_abs, door_w, door_h, wall_t, wx_start, wz_center, palette, meshes)
            else:
                glass_wins = [dict(o, x=o["x"] + wall_t / 2) for o in openings if o.get("is_window")]
                add_glass_x(glass_wins, wz_center, wx_start, palette, meshes, wall_t)

        elif is_z_aligned:
            # Z 轴对齐的墙（左/右墙）— 不延伸，由 X 墙覆盖墙角
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


# ─── 屋顶压顶（沿多边形轮廓顶边） ────────────────────────────────

def build_roof_coping(footprint, height, floor_t, wall_t, palette, x_off, z_off,
                      eave_overhang: float = 0.0):
    """
    沿多边形轮廓各边顶部添加收边压顶（扁平横向box）。
    eave_overhang: 0~1，屋檐向外挑出的比例（相对于 wall_t，max 1.5m）。
    """
    cap_h    = 0.15
    overhang = eave_overhang * 1.5    # 最大挑出 1.5 m
    cap_w    = wall_t + 0.05 + overhang
    y_top    = height + floor_t

    coords = list(footprint.exterior.coords)
    meshes = []

    for i in range(len(coords) - 1):
        p0 = coords[i]
        p1 = coords[i + 1]
        dx = p1[0] - p0[0]
        dz = p1[1] - p0[1]
        edge_len = math.hypot(dx, dz)
        if edge_len < 1e-4:
            continue

        mid_x = (p0[0] + p1[0]) / 2
        mid_z = (p0[1] + p1[1]) / 2
        angle  = math.atan2(dz, dx)

        m = trimesh.creation.box(extents=[edge_len + 0.02, cap_h, cap_w])
        m.apply_translation([0, cap_h / 2, 0])
        rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        m.apply_transform(rot)
        m.apply_translation([_r3(x_off + mid_x), _r3(y_top), _r3(z_off + mid_z)])
        m.vertices = np.round(m.vertices, 3)

        c = np.array(palette["wall"], dtype=np.uint8)
        m.visual.face_colors = np.tile(c, (len(m.faces), 1))
        meshes.append(m)

    return meshes


# ─── 平顶 (flat roof, roof_type=0) ───────────────────────────

def build_flat_roof(footprint, height, floor_t, palette, x_off, z_off):
    """
    用轮廓多边形生成平顶屋顶平板（适用于矩形/L形/U形）。
    底面紧贴墙体顶部（Y = height + floor_t），向上延伸 roof_t。
    颜色使用外墙色，确保从外部可见。
    """
    roof_t   = max(floor_t * 1.3, 0.25)     # 足够厚以可见
    # 屋顶颜色 = 墙色 × 0.7（稍暗于墙体）
    wc = palette["wall"]
    roof_color = [int(wc[0] * 0.7), int(wc[1] * 0.7), int(wc[2] * 0.7), wc[3]]
    roof_mesh = make_extruded_polygon(footprint, roof_t, roof_color)
    # make_extruded_polygon 旋转后：顶面 Y=0，底面 Y=-roof_t
    # 平移使底面 = height + floor_t（墙顶面），顶面 = height + floor_t + roof_t
    roof_mesh.apply_translation([x_off, height + floor_t + roof_t, z_off])
    return [roof_mesh]


# ─── 坡屋顶 (gabled roof, roof_type=1) ───────────────────────

def build_gabled_roof(footprint, height, floor_t, roof_pitch, palette, x_off, z_off):
    """
    生成山墙式坡屋顶：两个斜面板沿 Z 中轴脊线交汇。
    roof_pitch: 0~1，控制屋脊高度比例
    """
    bounds   = footprint.bounds             # (minx, minz, maxx, maxz)
    w        = bounds[2] - bounds[0]
    d        = bounds[3] - bounds[1]
    y_base   = height + floor_t + 0.15     # 0.10 = coping cap_h, roof starts above coping top
    ridge_h  = max(1.0, roof_pitch * d * 0.8)
    panel_t  = 0.20
    half_d   = d / 2
    panel_len = math.hypot(half_d, ridge_h) + 0.15
    tilt     = math.atan2(ridge_h, half_d)

    mid_x  = x_off + (bounds[0] + bounds[2]) / 2
    z_mid  = (bounds[1] + bounds[3]) / 2
    # 屋顶颜色 = 墙色 × 0.7
    wc = palette["wall"]
    color  = np.array([int(wc[0]*0.7), int(wc[1]*0.7), int(wc[2]*0.7), wc[3]], dtype=np.uint8)
    meshes = []

    for sign, cz_local in [(-1, bounds[1] + half_d / 2),
                             ( 1, bounds[3] - half_d / 2)]:
        m = trimesh.creation.box(extents=[w, panel_t, panel_len])
        rot = trimesh.transformations.rotation_matrix(sign * tilt, [1, 0, 0])
        m.apply_transform(rot)
        m.apply_translation([mid_x, y_base + ridge_h / 2, z_off + cz_local])
        m.visual.face_colors = np.tile(color, (len(m.faces), 1))
        meshes.append(m)

    # ── 山墙封口（左右两端三角形） ──────────────────────────────
    z_front = z_off + bounds[1]
    z_back  = z_off + bounds[3]
    z_peak  = z_off + z_mid
    for x_local, face_order in [(bounds[0], [0, 1, 2]), (bounds[2], [0, 2, 1])]:
        x_w = x_off + x_local
        verts = np.array([
            [x_w, y_base,           z_front],   # 0: 前下
            [x_w, y_base,           z_back ],   # 1: 后下
            [x_w, y_base + ridge_h, z_peak ],   # 2: 峰顶
        ], dtype=float)
        # 双面：正面+背面，保证从两侧都可见
        faces = np.array([face_order, face_order[::-1]])
        tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        tri.visual.face_colors = np.tile(color, (len(tri.faces), 1))
        meshes.append(tri)

    return meshes


# ─── 城垛 (battlements) ───────────────────────────────────────

def build_battlements(footprint, height, floor_t, wall_t, palette, x_off, z_off):
    """
    沿多边形轮廓顶部等距摆放雉堞（merlon）。
    """
    merlon_h = 0.55
    merlon_w = 0.55
    gap_w    = 0.50
    period   = merlon_w + gap_w
    y_base   = height + floor_t
    color    = np.array(palette["wall"], dtype=np.uint8)

    coords = list(footprint.exterior.coords)
    meshes = []

    for i in range(len(coords) - 1):
        p0, p1 = coords[i], coords[i + 1]
        dx = p1[0] - p0[0]
        dz = p1[1] - p0[1]
        edge_len = math.hypot(dx, dz)
        if edge_len < period:
            continue

        ux, uz = dx / edge_len, dz / edge_len
        angle  = math.atan2(dz, dx)
        n_merlons  = max(1, int(edge_len / period))
        total_span = n_merlons * period - gap_w
        start_pos  = (edge_len - total_span) / 2 + merlon_w / 2

        for k in range(n_merlons):
            pos = start_pos + k * period
            if pos + merlon_w / 2 > edge_len:
                break
            cx_world = x_off + p0[0] + ux * pos
            cz_world = z_off + p0[1] + uz * pos

            m = trimesh.creation.box(extents=[merlon_w, merlon_h, wall_t + 0.12])
            m.apply_translation([0, merlon_h / 2, 0])
            rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
            m.apply_transform(rot)
            m.apply_translation([cx_world, y_base, cz_world])
            m.visual.face_colors = np.tile(color, (len(m.faces), 1))
            meshes.append(m)

    return meshes


# ─── 拱门 / 尖拱窗 装饰线脚 ──────────────────────────────────

def build_arch_trims(footprint, height, wall_t, params, palette, x_off, z_off,
                     has_arch: bool, window_shape: int):
    """
    在门洞（has_arch）和窗洞（window_shape==1 尖拱）上方添加 box 近似线脚。
    """
    if not has_arch and window_shape != 1:
        return []

    door_w  = float(params["door_spec"]["width"])
    door_h  = min(float(params["door_spec"]["height"]), height - 0.15)
    win_w   = float(params["win_spec"]["width"])
    win_h   = float(params["win_spec"]["height"])
    win_d   = float(params["win_spec"]["density"])
    subdiv  = int(params["subdivision"])

    coords = list(footprint.exterior.coords)
    meshes = []
    door_placed = False
    trim_color  = np.array(palette["wall"], dtype=np.uint8)

    for i in range(len(coords) - 1):
        p0, p1 = coords[i], coords[i + 1]
        dx = p1[0] - p0[0]
        dz = p1[1] - p0[1]
        edge_len = math.hypot(dx, dz)
        if edge_len < 1e-4:
            continue

        ux, uz  = dx / edge_len, dz / edge_len
        angle   = math.atan2(dz, dx)
        edge_type, _ = _classify_edge(p0, p1)
        is_front = (edge_type == "front") and not door_placed

        if is_front and edge_len >= door_w + 0.5:
            door_placed = True
            if has_arch:
                arch_h    = door_w * 0.45
                door_x    = (edge_len - door_w) / 2
                mid_along = door_x + door_w / 2
                cx = x_off + p0[0] + ux * mid_along
                cz = z_off + p0[1] + uz * mid_along
                m  = trimesh.creation.box(extents=[door_w * 0.88, arch_h, wall_t + 0.03])
                m.apply_translation([0, door_h + arch_h / 2, 0])
                rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
                m.apply_transform(rot)
                m.apply_translation([cx, 0, cz])
                m.visual.face_colors = np.tile(trim_color, (len(m.faces), 1))
                meshes.append(m)
            continue  # 前墙已处理（门），跳过窗户线脚避免叠加到门上

        if window_shape == 1:
            wins = place_windows_edge(edge_len, height, win_w, win_h, win_d, subdiv)
            for op in wins:
                arch_h    = op["w"] * 0.42
                mid_along = op["x"] + op["w"] / 2
                cx = x_off + p0[0] + ux * mid_along
                cz = z_off + p0[1] + uz * mid_along
                m  = trimesh.creation.box(extents=[op["w"] * 0.82, arch_h, wall_t + 0.03])
                m.apply_translation([0, op["y"] + op["h"] + arch_h / 2, 0])
                rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
                m.apply_transform(rot)
                m.apply_translation([cx, 0, cz])
                m.visual.face_colors = np.tile(trim_color, (len(m.faces), 1))
                meshes.append(m)

    return meshes


# ─── 四坡屋顶 (hip roof, roof_type=2) ──────────────────────────

def build_hip_roof(footprint, height, floor_t, roof_pitch, palette, x_off, z_off):
    """
    四坡歇山屋顶：4 个真实三角/梯形面板精确收拢到中脊线。
    使用顶点+面索引构造，完全避免 box 穿插问题。

    顶点命名（世界坐标，y_base 高度）:
      A=前左  B=前右  C=后右  D=后左
      E=脊左  F=脊右  (若 w==d 则 E==F 为单点)

    w >= d 时脊沿 X 轴; w < d 时脊沿 Z 轴。
    法线通过 CCW 绕序保证朝外。
    """
    bounds  = footprint.bounds
    x0, z0, x1, z1 = bounds
    w      = x1 - x0
    d      = z1 - z0
    y_base = height + floor_t + 0.15
    ridge_h = max(0.4, roof_pitch * min(w, d) * 0.4)

    # 世界坐标角点（屋顶底边，y = y_base）
    A = np.array([x_off + x0, y_base, z_off + z0])
    B = np.array([x_off + x1, y_base, z_off + z0])
    C = np.array([x_off + x1, y_base, z_off + z1])
    D = np.array([x_off + x0, y_base, z_off + z1])
    ry = y_base + ridge_h
    # 屋顶颜色 = 墙色 × 0.7
    wc = palette["wall"]
    color = np.array([int(wc[0]*0.7), int(wc[1]*0.7), int(wc[2]*0.7), wc[3]], dtype=np.uint8)

    if w >= d:
        # 脊沿 X 轴
        mid_z = z_off + (z0 + z1) / 2
        half  = d / 2
        E = np.array([x_off + x0 + half, ry, mid_z])
        F = np.array([x_off + x1 - half, ry, mid_z])
        # 若 w==d，E/F 重合成单点（金字塔）

        # 面板定义: [(顶点列表, 三角形索引列表)]
        # 法线方向已通过 CCW 绕序验证（朝外 + 朝上）
        panels = [
            # 前坡 (法线 -Z, +Y): A F B + A E F
            ([A, F, B, E], [(0,1,2), (0,3,1)]),
            # 后坡 (法线 +Z, +Y): D C F + D F E
            ([D, C, F, E], [(0,1,2), (0,2,3)]),
            # 左三角坡 (法线 -X, +Y): A D E
            ([A, D, E],    [(0,1,2)]),
            # 右三角坡 (法线 +X, +Y): B F C
            ([B, F, C],    [(0,1,2)]),
        ]
    else:
        # 脊沿 Z 轴
        mid_x = x_off + (x0 + x1) / 2
        half  = w / 2
        E = np.array([mid_x, ry, z_off + z0 + half])
        F = np.array([mid_x, ry, z_off + z1 - half])

        panels = [
            # 左坡 (法线 -X, +Y): A D F + A F E
            ([A, D, F, E], [(0,1,2), (0,2,3)]),
            # 右坡 (法线 +X, +Y): B F C + B E F
            ([B, F, C, E], [(0,1,2), (0,3,1)]),
            # 前三角坡 (法线 -Z, +Y): A E B
            ([A, E, B],    [(0,1,2)]),
            # 后三角坡 (法线 +Z, +Y): D C F
            ([D, C, F],    [(0,1,2)]),
        ]

    meshes = []
    for verts, faces in panels:
        v = np.array(verts, dtype=float)
        f = np.array(faces,  dtype=int)
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        m.visual.face_colors = np.tile(color, (len(m.faces), 1))
        meshes.append(m)
    return meshes


# ─── 东亚翘角屋顶 (pagoda/curved eave, roof_type=3) ─────────────

def build_pagoda_roof(footprint, height, floor_t, roof_pitch, palette, x_off, z_off):
    """
    东亚翘角坡屋顶：在山墙坡顶基础上，四角添加向上翘起的角飞檐。
    """
    meshes = build_gabled_roof(footprint, height, floor_t, roof_pitch, palette, x_off, z_off)

    bounds  = footprint.bounds
    w       = bounds[2] - bounds[0]
    d       = bounds[3] - bounds[1]
    y_base  = height + floor_t + 0.15
    ridge_h = max(0.4, roof_pitch * d * 0.45)
    # 屋顶颜色 = 墙色 × 0.7
    wc = palette["wall"]
    color   = np.array([int(wc[0]*0.7), int(wc[1]*0.7), int(wc[2]*0.7), wc[3]], dtype=np.uint8)

    # 四角翘檐：各角一个细长翘板，内端埋入屋面，外端上翘飞出
    eave_l     = min(w, d) * 0.15          # 翘檐长度 = 短边15%
    tilt       = math.radians(20)          # 上翘角度 ~20°（写实翘檐）
    eave_thick = max(floor_t, 0.12)        # 厚度与屋顶楼板一致
    eave_rise  = eave_l * 0.5 * math.sin(tilt)  # 内端→中心的高度差
    corners = [
        (bounds[0], bounds[1], -1, -1),
        (bounds[2], bounds[1],  1, -1),
        (bounds[2], bounds[3],  1,  1),
        (bounds[0], bounds[3], -1,  1),
    ]
    for cx_local, cz_local, sx, sz in corners:
        m = trimesh.creation.box(extents=[eave_l, eave_thick, eave_l * 0.55])
        # 先绕Z轴倾斜（+X端上翘），再绕Y轴旋转至正确对角方向
        rot_tilt = trimesh.transformations.rotation_matrix(tilt, [0, 0, 1])
        rot_yaw  = trimesh.transformations.rotation_matrix(
            math.atan2(-sz, sx), [0, 1, 0])
        m.apply_transform(rot_tilt)
        m.apply_transform(rot_yaw)
        # 中心置于墙角，内端自然嵌入屋面，外端飞出；Y偏移使内端与屋面齐平
        m.apply_translation([x_off + cx_local, y_base + eave_rise,
                              z_off + cz_local])
        m.visual.face_colors = np.tile(color, (len(m.faces), 1))
        meshes.append(m)

    return meshes


# ─── 四角炮楼顶 (corner turrets, roof_type=4) ──────────────────

def build_turret_roof(footprint, height, floor_t, wall_t, palette, x_off, z_off):
    """
    四角炮楼：在建筑四角各放一个小型方塔，带尖顶。
    """
    bounds   = footprint.bounds
    y_base   = height + floor_t + 0.15
    tower_w  = max(wall_t * 2.5, min(1.6, (bounds[2] - bounds[0]) * 0.18))
    tower_h  = max(1.2, (bounds[1 + 2] - bounds[1]) * 0.22)  # 约 22% 建筑高
    spire_h  = tower_w * 0.8
    color_w  = np.array(palette["wall"],    dtype=np.uint8)
    # 屋顶颜色 = 墙色 × 0.7
    wc = palette["wall"]
    color_r  = np.array([int(wc[0]*0.7), int(wc[1]*0.7), int(wc[2]*0.7), wc[3]], dtype=np.uint8)
    meshes   = []

    corners = [
        (bounds[0], bounds[1]),
        (bounds[2], bounds[1]),
        (bounds[2], bounds[3]),
        (bounds[0], bounds[3]),
    ]
    for cx_local, cz_local in corners:
        # 塔身
        cx = x_off + cx_local
        cz = z_off + cz_local
        body = trimesh.creation.box(extents=[tower_w, tower_h, tower_w])
        body.apply_translation([cx, y_base + tower_h / 2, cz])
        body.visual.face_colors = np.tile(color_w, (len(body.faces), 1))
        meshes.append(body)

        # 尖顶（用细高 box 近似四棱锥）
        spire = trimesh.creation.box(extents=[tower_w * 0.55, spire_h, tower_w * 0.55])
        spire.apply_translation([cx, y_base + tower_h + spire_h / 2, cz])
        spire.visual.face_colors = np.tile(color_r, (len(spire.faces), 1))
        meshes.append(spire)

    return meshes


# ─── 柱子 (columns) ─────────────────────────────────────────────

def build_columns(footprint, height, wall_t, column_count: int, palette,
                  x_off, z_off):
    """
    沿建筑前立面（最长边）等距放置 column_count 根方柱。
    柱截面 = wall_t × wall_t，高度 = 建筑高度，贴着外墙外表面。
    """
    if column_count <= 0:
        return []

    coords     = list(footprint.exterior.coords)
    color      = np.array(palette["wall"], dtype=np.uint8)
    # 加深 15% 使柱子有区分感
    col_rgb    = np.clip(color[:3].astype(int) - 30, 0, 255)
    col_color  = np.array([*col_rgb, 255], dtype=np.uint8)
    meshes     = []

    # 找最长边作为立柱面
    best_edge  = None
    best_len   = 0.0
    for i in range(len(coords) - 1):
        p0, p1 = coords[i], coords[i + 1]
        elen = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if elen > best_len:
            best_len   = elen
            best_edge  = (p0, p1)

    if best_edge is None:
        return []

    p0, p1   = best_edge
    dx       = p1[0] - p0[0]
    dz       = p1[1] - p0[1]
    ux, uz   = dx / best_len, dz / best_len
    angle    = math.atan2(dz, dx)
    col_d    = wall_t          # 柱深（法线方向）
    col_w    = wall_t * 0.9    # 柱宽（沿边方向）

    # n 根柱子均匀排列
    n = max(2, column_count)
    for k in range(n):
        t = (k + 0.5) / n   # 0.5/n … (n-0.5)/n
        pos = t * best_len
        cx  = x_off + p0[0] + ux * pos
        cz  = z_off + p0[1] + uz * pos

        col = trimesh.creation.box(extents=[col_w, height, col_d])
        col.apply_translation([0, height / 2, 0])
        rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        col.apply_transform(rot)
        col.apply_translation([cx, 0, cz])
        col.visual.face_colors = np.tile(col_color, (len(col.faces), 1))
        meshes.append(col)

    return meshes


# ─── 窗户形状装饰 (window_shape = 2 / 3 / 4) ──────────────────

def build_window_shape_trims(footprint, height, wall_t, params, palette,
                              x_off, z_off, window_shape: int):
    """
    window_shape:
      2 = 十字形: 在窗洞处叠加横向压条（玫瑰窗十字格）
      3 = 箭缝/射孔: 在窗洞两侧加竖向侧石柱，窗变细
      4 = 圆拱顶: 在窗洞上方加半圆弧顶盒（宽于尖拱，更饱满）
    """
    if window_shape not in (2, 3, 4):
        return []

    win_w  = float(params["win_spec"]["width"])
    win_h  = float(params["win_spec"]["height"])
    win_d  = float(params["win_spec"]["density"])
    door_w = float(params["door_spec"]["width"])
    subdiv = int(params["subdivision"])

    coords      = list(footprint.exterior.coords)
    trim_c      = np.array(palette["wall"], dtype=np.uint8)
    meshes      = []
    door_placed = False

    for i in range(len(coords) - 1):
        p0, p1  = coords[i], coords[i + 1]
        dx, dz  = p1[0] - p0[0], p1[1] - p0[1]
        edge_len = math.hypot(dx, dz)
        if edge_len < 1e-4:
            continue
        ux, uz     = dx / edge_len, dz / edge_len
        angle      = math.atan2(dz, dx)
        edge_type, _ = _classify_edge(p0, p1)
        is_front   = (edge_type == "front") and not door_placed

        wins = place_windows_edge(edge_len, height, win_w, win_h, win_d, subdiv)

        if is_front and edge_len >= door_w + 0.5:
            door_placed = True
            # 门居中放置：过滤掉与门洞重叠的窗户，只对真正的窗户加装饰
            door_x   = (edge_len - door_w) / 2
            door_mid = door_x + door_w / 2
            wins = [op for op in wins
                    if abs(op["x"] + op["w"] / 2 - door_mid) > (door_w / 2 + op["w"] / 2)]

        for op in wins:
            mid_along = op["x"] + op["w"] / 2
            cx = x_off + p0[0] + ux * mid_along
            cz = z_off + p0[1] + uz * mid_along
            wy = op["y"]
            wh = op["h"]
            ww = op["w"]

            # 线脚统一参数：凸出最多 3cm，宽度 5cm
            trim_depth = wall_t + 0.03   # 比墙厚多 3cm（每侧凸出 1.5cm）
            trim_w     = 0.05            # 线脚宽度 5cm

            if window_shape == 2:
                # 十字：一根水平压条
                bar_h = max(trim_w, wh * 0.06)
                bar   = trimesh.creation.box(extents=[ww * 0.85, bar_h, trim_depth])
                bar.apply_translation([0, wy + wh * 0.55, 0])
                rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
                bar.apply_transform(rot)
                bar.apply_translation([cx, 0, cz])
                bar.visual.face_colors = np.tile(trim_c, (len(bar.faces), 1))
                meshes.append(bar)

            elif window_shape == 3:
                # 箭缝：两侧各一根竖向侧石柱
                for sx in (-1, 1):
                    off_along = mid_along + sx * (ww * 0.5 + trim_w * 0.5)
                    jcx = x_off + p0[0] + ux * off_along
                    jcz = z_off + p0[1] + uz * off_along
                    jamb = trimesh.creation.box(extents=[trim_w, wh * 1.05, trim_depth])
                    jamb.apply_translation([0, wy + wh / 2, 0])
                    rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
                    jamb.apply_transform(rot)
                    jamb.apply_translation([jcx, 0, jcz])
                    jamb.visual.face_colors = np.tile(trim_c, (len(jamb.faces), 1))
                    meshes.append(jamb)

            elif window_shape == 4:
                # 圆拱顶
                arch_h = ww * 0.55
                arch   = trimesh.creation.box(extents=[ww * 0.92, arch_h, trim_depth])
                arch.apply_translation([0, wy + wh + arch_h * 0.45, 0])
                rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
                arch.apply_transform(rot)
                arch.apply_translation([cx, 0, cz])
                arch.visual.face_colors = np.tile(trim_c, (len(arch.faces), 1))
                meshes.append(arch)

    return meshes


# ─── 单间房间（多边形轮廓版） ────────────────────────────────────

def build_room(params, palette, x_off=0.0, z_off=0.0, footprint=None,
               generate_interior: bool = False):
    """
    生成单间房间的所有 mesh。
    footprint: shapely Polygon（局部坐标，原点在房间左前角），
               None 时默认使用 ROOM_W × ROOM_D 矩形（向后兼容）。
    generate_interior: True 时生成内部分隔墙，False（默认）时只生成外壳。
    """
    if footprint is None:
        footprint = make_rect_footprint()

    height   = float(params["height_range"][1])
    wall_t   = float(params["wall_thickness"])
    floor_t  = float(params["floor_thickness"])
    subdiv   = int(params["subdivision"])

    # ── 读取新增视觉参数（兼容旧 params 不含这些字段） ──────────
    roof_type       = int(params.get("roof_type", 0))
    roof_pitch      = float(params.get("roof_pitch", 0.3))
    has_battlements = bool(int(params.get("has_battlements", 0)))
    has_arch        = bool(int(params.get("has_arch", 0)))
    window_shape    = int(params.get("window_shape", 0))
    eave_overhang   = float(params.get("eave_overhang", 0.0))
    column_count    = int(params.get("column_count", 0))
    wall_color      = params.get("wall_color")   # [r,g,b] in 0-1 or None

    # ── 几何复杂度参数（来自 w3_mesh_features） ──────────────
    mesh_complexity = float(params.get("mesh_complexity", 0.3))
    detail_density  = float(params.get("detail_density", 0.5))
    simple_ratio    = float(params.get("simple_ratio", 0.4))

    # mesh_complexity > 0.7 → 增加装饰细节（更多拱门线脚、柱子）
    if mesh_complexity > 0.7:
        if column_count < 2:
            column_count = 2
        has_arch = True

    # detail_density > 0.6 → 增密窗户/门
    if detail_density > 0.6 and "win_spec" in params:
        old_d = params["win_spec"].get("density", 0.3)
        boost = (detail_density - 0.6) * 0.5  # max +0.2 boost
        params = {**params, "win_spec": {**params["win_spec"],
                  "density": min(0.95, old_d + boost)}}

    # simple_ratio > 0.6 → 简化几何（去除城垛、减少柱子、强制平顶）
    if simple_ratio > 0.6:
        has_battlements = False
        column_count = min(column_count, 2)
        if simple_ratio > 0.75:
            roof_type = 0  # 强制平顶

    # ── 墙色覆盖（如果 params 提供 wall_color） ──────────────
    if wall_color is not None:
        r = int(min(255, max(0, wall_color[0] * 255)))
        g = int(min(255, max(0, wall_color[1] * 255)))
        b = int(min(255, max(0, wall_color[2] * 255)))
        palette = {**palette, "wall": [r, g, b, 255]}

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

    # ── 屋顶压顶（含屋檐挑出） ────────────────────────────────
    coping_meshes = build_roof_coping(footprint, height, floor_t, wall_t, palette,
                                      x_off, z_off, eave_overhang=eave_overhang)
    meshes.extend(coping_meshes)

    # ── 屋顶形态 ────────────────────────────────────────────
    # L形/U形轮廓强制使用平顶（坡顶/尖顶算法仅适用于矩形平面）
    _b = footprint.bounds
    _bbox_area = (_b[2] - _b[0]) * (_b[3] - _b[1])
    _is_rect   = abs(footprint.area - _bbox_area) < 1e-2 * _bbox_area
    effective_roof_type = roof_type if _is_rect else 0

    if effective_roof_type == 0:
        meshes.extend(build_flat_roof(
            footprint, height, floor_t, palette, x_off, z_off))
    elif effective_roof_type == 1:
        meshes.extend(build_gabled_roof(
            footprint, height, floor_t, roof_pitch, palette, x_off, z_off))
    elif effective_roof_type == 2:
        meshes.extend(build_hip_roof(
            footprint, height, floor_t, roof_pitch, palette, x_off, z_off))
    elif effective_roof_type == 3:
        meshes.extend(build_pagoda_roof(
            footprint, height, floor_t, roof_pitch, palette, x_off, z_off))
    elif effective_roof_type == 4:
        meshes.extend(build_turret_roof(
            footprint, height, floor_t, wall_t, palette, x_off, z_off))

    # ── 城垛 ─────────────────────────────────────────────────
    if has_battlements:
        meshes.extend(build_battlements(
            footprint, height, floor_t, wall_t, palette, x_off, z_off))

    # ── 拱门 / 尖拱窗 线脚 ──────────────────────────────────
    meshes.extend(build_arch_trims(
        footprint, height, wall_t, params, palette, x_off, z_off,
        has_arch, window_shape))

    # ── 额外窗户形状装饰（shape 2/3/4） ─────────────────────
    meshes.extend(build_window_shape_trims(
        footprint, height, wall_t, params, palette, x_off, z_off, window_shape))

    # ── 柱子 ──────────────────────────────────────────────────
    if column_count > 0:
        meshes.extend(build_columns(
            footprint, height, wall_t, column_count, palette, x_off, z_off))

    # ── 内部分隔墙（沿 Z 均分，各有一扇门） ─────────────────
    if generate_interior and subdiv > 1:
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

            # 用水平线与轮廓求交，精确得到该 z 处的各连续墙段
            h_line = LineString([
                (bounds[0] - 1, wz_local),
                (bounds[2] + 1, wz_local),
            ])
            intersection = footprint.intersection(h_line)

            if intersection.is_empty:
                continue

            # 拆解为独立线段（MultiLineString → 列表；单段直接用）
            if hasattr(intersection, "geoms"):
                segs = [g for g in intersection.geoms
                        if g.geom_type in ("LineString", "MultiLineString")]
            elif intersection.geom_type == "LineString":
                segs = [intersection]
            else:
                continue  # Point / GeometryCollection 等退化情形

            for seg in segs:
                xs = [c[0] for c in seg.coords]
                x0, x1 = min(xs), max(xs)
                seg_w = x1 - x0
                if seg_w < int_door_w + 0.3:
                    continue

                x_start    = x_off + x0
                int_door_x = (seg_w - int_door_w) / 2
                int_ops    = [{"x": int_door_x, "w": int_door_w,
                               "y": 0.0, "h": int_door_h}]
                meshes += build_x_wall(
                    seg_w, height, int_wall_t, int_ops,
                    palette["internal"], x_start, wz_world)
                add_door_frame(
                    int_door_x, int_door_w, int_door_h,
                    int_wall_t, x_start, wz_world, palette, meshes)
                add_door_panel(
                    int_door_x, int_door_w, int_door_h,
                    int_wall_t, x_start, wz_world, palette, meshes)

    return meshes


# ─── 场景构建 ─────────────────────────────────────────────────

def build_scene(style_params_map, use_style_palette=True, footprints=None,
                generate_interior: bool = False):
    """
    生成包含多个并排房间的 trimesh.Scene。
    style_params_map: {"medieval": params, "modern": params, ...}
    footprints: dict style→shapely Polygon，None 时使用默认矩形
    generate_interior: 传递给 build_room，控制是否生成内部分隔墙
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

        room = build_room(params, palette, x_off, 0.0, footprint=footprint,
                          generate_interior=generate_interior)
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

    # ── 生成 medieval_test.glb（含新视觉特征） ───────────────
    if "medieval" in trained_map:
        print("\n生成 medieval_test.glb（medieval，矩形轮廓，完整视觉特征）...")
        med_scene = build_scene({"medieval": trained_map["medieval"]}, use_style_palette=True)
        med_path  = script_dir / "medieval_test.glb"
        med_scene.export(str(med_path))
        print(f"  => {med_path}  ({sum(len(g.faces) for g in med_scene.geometry.values()):,} 面)")

    # ── 输出对比报告 ──────────────────────────────────────────
    print_comparison(trained_map)
    export_report(trained_map, script_dir / "comparison_report.json")

    print("\n完成。")
    print(f"  trained.glb      — 矩形轮廓，模型推理参数")
    print(f"  baseline.glb     — 矩形轮廓，默认参数")
    print(f"  l_shape_test.glb — L形轮廓，medieval + modern")
    print(f"  u_shape_test.glb — U形轮廓，medieval + modern")
    if "medieval" in trained_map:
        print(f"  medieval_test.glb — medieval，完整视觉特征")


if __name__ == "__main__":
    main()
