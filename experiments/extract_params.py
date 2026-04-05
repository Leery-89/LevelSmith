"""
extract_params.py
从 validation_data/ 下的所有 GLB 模型自动提取建筑参数，
保存到 validation_data/extracted_params.json。

提取参数:
  height_range_min / height_range_max — 模型 Y 轴最低/最高点
  wall_thickness   — 最薄实体竖向几何厚度
  floor_thickness  — 水平板状几何厚度
  door_width       — 最大开口宽度（>= 1.8 m 高）
  door_height      — 最大开口高度（>= 1.8 m 高）
  win_width        — 窗口开口宽度（中位数）
  win_height       — 窗口开口高度（中位数）
  win_density      — 开口面积 / 总墙面面积
  subdivision      — 内部竖向分隔数估算
"""

import json
import math
import statistics
import traceback
from pathlib import Path

import numpy as np
import trimesh

# ─── 路径 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).parent
VALIDATION_DIR = SCRIPT_DIR / "validation_data"
OUTPUT_JSON    = VALIDATION_DIR / "extracted_params.json"

# 跳过 rejected 目录
SKIP_DIRS = {"rejected"}

# ─── 尺度检测 ──────────────────────────────────────────────────────────────
def detect_scale(bounds_min, bounds_max) -> float:
    """
    自动检测单位比例，返回「乘以该值后得到 米」的系数。
    启发式：典型建筑高度 4-50 m；
      若 Y 跨度 < 1 → 可能是 km → × 1000
      若 Y 跨度 > 500 → 可能是 cm → × 0.01
      否则假定已是 m → × 1.0
    """
    h = bounds_max[1] - bounds_min[1]
    if h < 0.5:
        return 1000.0   # km → m
    if h > 500:
        return 0.01     # cm → m
    if h > 50:
        return 0.1      # dm? 保守缩放
    return 1.0


# ─── 几何辅助 ──────────────────────────────────────────────────────────────

def merge_scene(scene_or_mesh) -> trimesh.Trimesh | None:
    """将场景/网格合并为单一 Trimesh（忽略空场景）。"""
    if isinstance(scene_or_mesh, trimesh.Scene):
        geoms = [g for g in scene_or_mesh.geometry.values()
                 if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if not geoms:
            return None
        return trimesh.util.concatenate(geoms)
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    return None


def ray_solid_segments(mesh: trimesh.Trimesh,
                       origin: np.ndarray,
                       direction: np.ndarray,
                       max_dist: float = 200.0) -> list[float]:
    """
    沿 direction 方向从 origin 发射射线，返回每段「实体」区间的长度列表。
    实体区间 = 射线从外部进入网格后再出来的距离。
    """
    ray_origins    = np.array([origin])
    ray_directions = np.array([direction / (np.linalg.norm(direction) + 1e-12)])

    try:
        locs, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=True,
        )
    except Exception:
        return []

    if len(locs) < 2:
        return []

    # 按距离排序
    dists = np.dot(locs - origin, ray_directions[0])
    order = np.argsort(dists)
    dists = dists[order]
    dists = dists[dists >= 0]
    dists = dists[dists <= max_dist]

    segments = []
    i = 0
    while i + 1 < len(dists):
        seg = dists[i + 1] - dists[i]
        if seg > 0.005:           # 忽略 5 mm 以下的薄皮
            segments.append(float(seg))
        i += 2
    return segments


# ─── 各参数提取函数 ────────────────────────────────────────────────────────

def extract_height(mesh: trimesh.Trimesh, scale: float) -> tuple[float, float]:
    """返回 (height_min, height_max) 单位 m。"""
    h_min = float(mesh.bounds[0][1]) * scale
    h_max = float(mesh.bounds[1][1]) * scale
    return h_min, h_max


def extract_wall_thickness(mesh: trimesh.Trimesh, scale: float,
                           n_samples: int = 30) -> float:
    """
    沿 X、Z 方向发射多条水平射线，取所有实体段最小值作为壁厚估计。
    """
    bmin, bmax = mesh.bounds
    cx = (bmin[0] + bmax[0]) / 2
    cz = (bmin[2] + bmax[2]) / 2
    dx = (bmax[0] - bmin[0])
    dz = (bmax[2] - bmin[2])

    # 采样高度：避开最底层和最顶层
    y_samples = np.linspace(bmin[1] + (bmax[1] - bmin[1]) * 0.15,
                             bmin[1] + (bmax[1] - bmin[1]) * 0.85,
                             n_samples)

    thicknesses = []
    for y in y_samples:
        for axis, start, length in [
            ([1, 0, 0], [bmin[0] - 0.5, y, cz], dx + 1.0),
            ([0, 0, 1], [cx, y, bmin[2] - 0.5], dz + 1.0),
        ]:
            segs = ray_solid_segments(mesh,
                                      np.array(start, dtype=float),
                                      np.array(axis, dtype=float),
                                      max_dist=length + 1.0)
            thicknesses.extend(segs)

    if not thicknesses:
        return 0.3   # 默认值

    # 取 10 百分位（避免门窗洞的宽度混入）
    thicknesses_m = [t * scale for t in thicknesses]
    # 过滤极端值（> 5 m 的不可能是墙）
    thin = [t for t in thicknesses_m if 0.05 <= t <= 5.0]
    if not thin:
        return 0.3
    thin.sort()
    idx = max(0, int(len(thin) * 0.1))
    return round(thin[idx], 3)


def extract_floor_thickness(mesh: trimesh.Trimesh, scale: float,
                            n_samples: int = 20) -> float:
    """
    向下发射竖向射线，寻找水平板的最薄实体段。
    """
    bmin, bmax = mesh.bounds
    xs = np.linspace(bmin[0] + (bmax[0] - bmin[0]) * 0.1,
                     bmax[0] - (bmax[0] - bmin[0]) * 0.1, n_samples)
    zs = np.linspace(bmin[2] + (bmax[2] - bmin[2]) * 0.1,
                     bmax[2] - (bmax[2] - bmin[2]) * 0.1, n_samples)

    thickness_candidates = []
    for x in xs[::3]:
        for z in zs[::3]:
            origin = np.array([x, bmax[1] + 0.5, z])
            segs = ray_solid_segments(mesh, origin, np.array([0, -1, 0]),
                                      max_dist=(bmax[1] - bmin[1]) + 1.5)
            thickness_candidates.extend(segs)

    if not thickness_candidates:
        return 0.2

    tc_m = [t * scale for t in thickness_candidates]
    thin = [t for t in tc_m if 0.05 <= t <= 2.0]
    if not thin:
        return 0.2
    thin.sort()
    idx = max(0, int(len(thin) * 0.1))
    return round(thin[idx], 3)


def _project_wall_openings(mesh: trimesh.Trimesh,
                            scale: float,
                            grid_res: float = 0.25) -> list[dict]:
    """
    将网格面投影到最大的竖向平面（正面/背面），
    在 2D 网格上找出「空白区域」作为开口，
    返回开口列表 [{w, h, y_bot}]（已换算为 m）。
    """
    bmin, bmax = mesh.bounds
    w_world = bmax[0] - bmin[0]
    h_world = bmax[1] - bmin[1]
    d_world = bmax[2] - bmin[2]

    # 选较长的水平轴作为墙面 U 轴
    if w_world >= d_world:
        u_axis, v_axis, normal_axis = 0, 1, 2   # XY 面 (Z 方向法线)
        u_range = (bmin[0], bmax[0])
        v_range = (bmin[1], bmax[1])
        normal_val = bmin[2]   # 使用 z_min 侧（正面）
    else:
        u_axis, v_axis, normal_axis = 2, 1, 0   # ZY 面 (X 方向法线)
        u_range = (bmin[2], bmax[2])
        v_range = (bmin[1], bmax[1])
        normal_val = bmin[0]

    # 建立网格
    grid_res_raw = grid_res / scale   # 转换为模型单位
    u_cells = max(4, int((u_range[1] - u_range[0]) / grid_res_raw))
    v_cells = max(4, int((v_range[1] - v_range[0]) / grid_res_raw))

    # 面中心投影到 UV
    face_centers = mesh.triangles_center
    normals      = mesh.face_normals

    # 只取朝向正面的面
    tol = 0.3
    if normal_axis == 2:
        mask = normals[:, 2] < -tol    # 朝 -Z
    else:
        mask = normals[:, 0] < -tol    # 朝 -X

    if mask.sum() < 10:
        # 如果正面面数太少，取所有竖向面
        vert_mask = np.abs(normals[:, 1]) < 0.5
        mask = vert_mask

    fc = face_centers[mask]
    if len(fc) == 0:
        return []

    u_coords = fc[:, u_axis]
    v_coords = fc[:, v_axis]

    # 归一化到网格索引
    u_norm = (u_coords - u_range[0]) / (u_range[1] - u_range[0] + 1e-9)
    v_norm = (v_coords - v_range[0]) / (v_range[1] - v_range[0] + 1e-9)

    ui = np.clip((u_norm * u_cells).astype(int), 0, u_cells - 1)
    vi = np.clip((v_norm * v_cells).astype(int), 0, v_cells - 1)

    occupied = np.zeros((u_cells, v_cells), dtype=bool)
    occupied[ui, vi] = True

    # 空白区域 = 开口候选
    empty = ~occupied

    # 用简单的连通块检测找开口（避免引入 scipy 依赖）
    visited = np.zeros_like(empty)
    openings = []

    for start_u in range(u_cells):
        for start_v in range(v_cells):
            if not empty[start_u, start_v] or visited[start_u, start_v]:
                continue
            # BFS
            stack = [(start_u, start_v)]
            cells = []
            while stack:
                cu, cv = stack.pop()
                if cu < 0 or cu >= u_cells or cv < 0 or cv >= v_cells:
                    continue
                if visited[cu, cv] or not empty[cu, cv]:
                    continue
                visited[cu, cv] = True
                cells.append((cu, cv))
                for du, dv in [(-1,0),(1,0),(0,-1),(0,1)]:
                    stack.append((cu+du, cv+dv))
            if len(cells) < 2:
                continue
            us_idx = [c[0] for c in cells]
            vs_idx = [c[1] for c in cells]
            u_span = (max(us_idx) - min(us_idx) + 1) * grid_res_raw * scale
            v_span = (max(vs_idx) - min(vs_idx) + 1) * grid_res_raw * scale
            v_bot  = v_range[0] * scale + min(vs_idx) * grid_res_raw * scale
            openings.append({"w": u_span, "h": v_span, "y_bot": v_bot})

    return openings


def extract_openings(mesh: trimesh.Trimesh, scale: float
                     ) -> tuple[float, float, float, float, float]:
    """
    返回 (door_width, door_height, win_width, win_height, win_density)
    """
    openings = _project_wall_openings(mesh, scale)

    # 分类: 高度 >= 1.8 m 且 宽度 >= 0.5 m → 门；否则 → 窗
    doors = [o for o in openings if o["h"] >= 1.8 and o["w"] >= 0.4]
    wins  = [o for o in openings if o not in doors and o["h"] >= 0.3 and o["w"] >= 0.2]

    # 门: 取最大的
    if doors:
        best_door = max(doors, key=lambda o: o["w"] * o["h"])
        door_w = round(min(best_door["w"], 3.0), 2)
        door_h = round(min(best_door["h"], 5.0), 2)
    else:
        door_w, door_h = 1.0, 2.1

    # 窗: 取中位数（更稳健）
    if wins:
        win_ws = sorted(o["w"] for o in wins)
        win_hs = sorted(o["h"] for o in wins)
        win_w = round(min(statistics.median(win_ws), 3.0), 2)
        win_h = round(min(statistics.median(win_hs), 3.0), 2)
    else:
        win_w, win_h = 0.8, 1.0

    # win_density: 开口总面积 / 总墙面面积估计
    bmin, bmax = mesh.bounds
    wall_area = max(bmax[0] - bmin[0], bmax[2] - bmin[2]) * (bmax[1] - bmin[1]) * scale * scale
    open_area = sum(o["w"] * o["h"] for o in wins)
    win_density = round(min(1.0, open_area / (wall_area + 1e-6)), 3) if wall_area > 0 else 0.1

    return door_w, door_h, win_w, win_h, win_density


def extract_subdivision(mesh: trimesh.Trimesh, scale: float) -> int:
    """
    估算内部水平楼层数：
    向上发射多条竖向射线，统计每条射线实体段数的中位数，减 1 得楼层数。
    """
    bmin, bmax = mesh.bounds
    xs = np.linspace(bmin[0] + (bmax[0] - bmin[0]) * 0.25,
                     bmax[0] - (bmax[0] - bmin[0]) * 0.25, 5)
    zs = np.linspace(bmin[2] + (bmax[2] - bmin[2]) * 0.25,
                     bmax[2] - (bmax[2] - bmin[2]) * 0.25, 5)

    seg_counts = []
    for x in xs:
        for z in zs:
            origin = np.array([x, bmin[1] - 0.5, z])
            segs = ray_solid_segments(mesh, origin, np.array([0, 1, 0]),
                                      max_dist=(bmax[1] - bmin[1]) + 1.5)
            seg_counts.append(len(segs))

    if not seg_counts:
        return 1
    median_segs = statistics.median(seg_counts)
    floors = max(1, int(round(median_segs)) - 1)
    return min(floors, 8)


# ─── 主提取逻辑 ────────────────────────────────────────────────────────────

def extract_model_params(glb_path: Path, style: str) -> dict:
    raw = trimesh.load(str(glb_path), force="scene")
    mesh = merge_scene(raw)
    if mesh is None or len(mesh.faces) == 0:
        raise ValueError("空网格或无法加载")

    scale = detect_scale(mesh.bounds[0], mesh.bounds[1])

    h_min, h_max = extract_height(mesh, scale)
    wall_t       = extract_wall_thickness(mesh, scale)
    floor_t      = extract_floor_thickness(mesh, scale)
    door_w, door_h, win_w, win_h, win_dens = extract_openings(mesh, scale)
    subdiv       = extract_subdivision(mesh, scale)

    # 钳位到 OUTPUT_PARAMS 合理范围
    h_min  = round(max(0.0, h_min), 2)
    h_max  = round(min(50.0, max(h_min + 2.0, h_max)), 2)
    wall_t = round(max(0.1, min(1.5, wall_t)), 3)
    floor_t = round(max(0.1, min(0.6, floor_t)), 3)
    door_w = round(max(0.6, min(3.0, door_w)), 2)
    door_h = round(max(1.8, min(5.0, door_h)), 2)
    win_w  = round(max(0.3, min(3.0, win_w)), 2)
    win_h  = round(max(0.4, min(3.0, win_h)), 2)

    return {
        "file":              glb_path.name,
        "style":             style,
        "scale_factor":      scale,
        "height_range_min":  h_min,
        "height_range_max":  h_max,
        "wall_thickness":    wall_t,
        "floor_thickness":   floor_t,
        "door_width":        door_w,
        "door_height":       door_h,
        "win_width":         win_w,
        "win_height":        win_h,
        "win_density":       win_dens,
        "subdivision":       subdiv,
    }


# ─── 入口 ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  extract_params.py  —  GLB 建筑参数提取")
    print("=" * 65)

    results      = []
    ok_count     = 0
    fail_count   = 0
    fail_records = []

    style_dirs = sorted(
        d for d in VALIDATION_DIR.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    )

    for style_dir in style_dirs:
        style = style_dir.name
        glb_files = sorted(style_dir.glob("*.glb"))
        if not glb_files:
            continue
        print(f"\n[{style}]  {len(glb_files)} 个模型")
        for glb in glb_files:
            print(f"  处理: {glb.name} ...", end=" ", flush=True)
            try:
                params = extract_model_params(glb, style)
                results.append(params)
                ok_count += 1
                print(f"OK  h={params['height_range_max']:.1f}m  "
                      f"wall={params['wall_thickness']:.2f}m  "
                      f"subdiv={params['subdivision']}")
            except Exception as e:
                reason = str(e)
                fail_count += 1
                fail_records.append({"file": glb.name, "style": style, "reason": reason})
                print(f"FAIL  ({reason[:60]})")

    output = {
        "total_ok":   ok_count,
        "total_fail": fail_count,
        "failures":   fail_records,
        "models":     results,
    }

    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print(f"  成功: {ok_count}  失败: {fail_count}")
    print(f"  结果已保存: {OUTPUT_JSON.resolve()}")
    if fail_records:
        print("  失败模型:")
        for r in fail_records:
            print(f"    [{r['style']}] {r['file']}: {r['reason'][:70]}")
    print("=" * 65)


if __name__ == "__main__":
    main()
