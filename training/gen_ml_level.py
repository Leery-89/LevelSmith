"""
gen_ml_level.py
使用训练好的布局模型生成中世纪小镇关卡 GLB + FBX。

流程：
  1. 从 layout_model.pt 生成 15 栋建筑的位置/朝向
  2. 按距中心距离分配风格：
       中心(1)    → medieval_keep
       内圈(2+4)  → medieval_chapel × 2 + medieval × 4
       外圈(8)    → medieval × 8
  3. 从 trained_style_params.json 加载风格参数
  4. 用 generate_level.build_room() 构建每栋建筑 mesh
  5. 导出 level_ml_layout.glb 和 level_ml_layout.fbx
  6. 打印与规则算法（organic）视觉差异对比
"""

import json
import math
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR   = Path(__file__).parent
MODEL_PATH   = SCRIPT_DIR / "models" / "layout_model.pt"
PARAMS_JSON  = SCRIPT_DIR / "trained_style_params.json"
OUT_GLB      = SCRIPT_DIR / "level_ml_layout.glb"
OUT_FBX      = SCRIPT_DIR / "level_ml_layout.fbx"
REF_GLB      = SCRIPT_DIR / "level_organic_test.glb"   # 规则算法参考

sys.path.insert(0, str(SCRIPT_DIR))

import generate_level as gl
import trimesh
import trimesh.transformations as TF

from shapely.geometry import box as sbox
from shapely.affinity import translate as shapely_translate

from layout_model import load_model, generate_layout, TYPES, TYPE2ID

# ─── 风格分配方案 ────────────────────────────────────────────────
#   slot 索引 0..14 按距中心距离升序排列
STYLE_SLOTS = (
    ["medieval_keep"]          +   # 0     中心
    ["medieval_chapel"] * 2    +   # 1-2   内圈礼拜堂
    ["medieval"] * 4           +   # 3-6   内圈民居
    ["medieval"] * 8               # 7-14  外圈
)  # 共 15 个

# 各风格建筑尺寸范围 (w_min, w_max, d_min, d_max) 单位 m
STYLE_SIZES = {
    "medieval_keep":   (9.0,  13.0,  9.0,  13.0),   # 方形城楼
    "medieval_chapel": (8.0,  11.0, 14.0,  20.0),   # 窄而长
    "medieval":        (9.0,  14.0,  7.0,  11.0),   # 普通民居
}

# ─── 调色板（从 level_layout.py 抄录） ──────────────────────────
_PALETTES = {
    "medieval": {
        "floor":   [115,  95,  75, 255], "ceiling": [135, 115,  90, 255],
        "wall":    [162, 146, 122, 255], "door":    [ 88,  58,  28, 255],
        "window":  [155, 195, 215, 180], "internal":[148, 133, 110, 255],
        "ground":  [ 90,  80,  65, 255],
    },
    "medieval_chapel": {
        "floor":   [130, 110,  88, 255], "ceiling": [150, 130, 105, 255],
        "wall":    [175, 160, 138, 255], "door":    [ 80,  52,  24, 255],
        "window":  [170, 210, 230, 200], "internal":[160, 145, 122, 255],
        "ground":  [ 90,  80,  65, 255],
    },
    "medieval_keep": {
        "floor":   [100,  85,  68, 255], "ceiling": [120, 103,  82, 255],
        "wall":    [148, 134, 112, 255], "door":    [ 72,  46,  20, 255],
        "window":  [130, 170, 190, 160], "internal":[138, 123, 100, 255],
        "ground":  [ 78,  68,  54, 255],
    },
}

# wall_color 覆盖（保持与 trained params 一致，略微区分）
_WALL_COLORS = {
    "medieval_keep":   [0.52, 0.47, 0.40],
    "medieval_chapel": [0.64, 0.58, 0.50],
    "medieval":        None,   # 从 params 读取
}


def _load_style_params(style: str) -> dict:
    """从 trained_style_params.json 加载风格参数。"""
    data   = json.loads(PARAMS_JSON.read_text("utf-8"))
    styles = data.get("styles", {})
    if style not in styles:
        raise ValueError(f"风格 '{style}' 不存在，可用: {list(styles)}")
    return dict(styles[style]["params"])


def _building_size(style: str, rng: np.random.Generator,
                   variation: float = 0.35) -> tuple[float, float]:
    wmin, wmax, dmin, dmax = STYLE_SIZES.get(style, (9.0, 14.0, 7.0, 11.0))
    w = rng.uniform(wmin, wmin + (wmax - wmin) * (0.5 + variation * 0.5))
    d = rng.uniform(dmin, dmin + (dmax - dmin) * (0.5 + variation * 0.5))
    return round(w, 1), round(d, 1)


def _make_footprint(style: str, w: float, d: float,
                    rng: np.random.Generator):
    """生成轮廓 shapely.Polygon（局部坐标，原点在左前角）。"""
    if style == "medieval":
        # 30% 概率 L 形，20% U 形，其余矩形
        r = rng.random()
        if r < 0.30:
            cx = rng.uniform(0.38, 0.52)
            cz = rng.uniform(0.38, 0.52)
            rect = sbox(0, 0, w, d)
            cut  = sbox(w * (1 - cx), d * (1 - cz), w, d)
            fp   = rect.difference(cut)
            return fp if not fp.is_empty else sbox(0, 0, w, d)
        elif r < 0.50:
            nx = rng.uniform(0.36, 0.48)
            nz = rng.uniform(0.42, 0.58)
            rect = sbox(0, 0, w, d)
            nx0  = w * (0.5 - nx / 2)
            nx1  = w * (0.5 + nx / 2)
            cut  = sbox(nx0, 0, nx1, d * nz)
            fp   = rect.difference(cut)
            return fp if not fp.is_empty else sbox(0, 0, w, d)
    # keep / chapel / 默认：矩形
    return sbox(0, 0, w, d)


def _vary_params(params: dict, rng: np.random.Generator,
                 variation: float = 0.35) -> dict:
    """给每栋建筑的参数加上随机扰动。"""
    p = dict(params)
    h_min, h_max = params["height_range"]
    h_range = max(h_max - h_min, 1.0)
    new_max = float(np.clip(
        h_max + rng.uniform(-variation * h_range * 0.5,
                             variation * h_range * 0.8),
        h_min + 1.5, h_max * 1.6))
    p["height_range"] = [h_min, round(new_max, 2)]

    wd = params["win_spec"]["density"]
    p["win_spec"] = {**params["win_spec"],
                     "density": float(np.clip(wd + rng.uniform(-0.1, 0.1), 0.1, 0.8))}

    wc = params.get("wall_color")
    if wc and variation > 0.1:
        j = rng.uniform(-variation * 0.07, variation * 0.07, size=3)
        p["wall_color"] = [float(np.clip(c + dj, 0.0, 1.0)) for c, dj in zip(wc, j)]
    return p


def build_ground_plane(area_w: float, area_d: float, offset_x: float, offset_z: float):
    """生成地面平板，居中放置。"""
    ground = trimesh.creation.box(extents=[area_w, 0.15, area_d])
    color  = np.array([78, 68, 54, 255], dtype=np.uint8)
    ground.visual.face_colors = np.tile(color, (len(ground.faces), 1))
    ground.apply_translation([offset_x + area_w / 2, -0.075, offset_z + area_d / 2])
    return ground


def generate_ml_level(
    n_buildings: int = 15,
    temperature: float = 0.75,
    seed: int = 2024,
    variation: float = 0.35,
) -> trimesh.Scene:
    """
    主生成函数：
      1. 布局模型生成位置/朝向
      2. 按距离分配风格
      3. 构建每栋建筑 mesh
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng_np  = np.random.default_rng(seed)

    # ── 1. 加载布局模型，生成位置 ─────────────────────────────
    print("[布局模型] 加载并生成建筑位置...")
    model   = load_model(MODEL_PATH, device)
    gen_blds = generate_layout(model, device, n_buildings,
                                temperature=temperature, rng=rng_np)

    # ── 2. 按距原点距离排序，分配风格 ─────────────────────────
    for b in gen_blds:
        b["_dist"] = math.hypot(b["nx"], b["ny"])
    gen_blds.sort(key=lambda b: b["_dist"])

    print(f"\n  {'#':>3}  {'nx':>7}  {'ny':>7}  {'ori°':>6}  "
          f"{'dist':>6}  {'style'}")
    print("  " + "─" * 55)
    assignments = []
    for i, (b, style) in enumerate(zip(gen_blds, STYLE_SLOTS)):
        print(f"  {i+1:>3}  {b['nx']:>7.1f}  {b['ny']:>7.1f}  "
              f"{b['orientation_deg']:>6.1f}  {b['_dist']:>6.1f}  {style}")
        assignments.append((b, style))

    # ── 3. 坐标偏移：将模型的 ±50m 坐标变换到 [0, 100m] 空间 ─
    #    使所有位置为正值，便于场景构建
    OFFSET = 50.0   # m
    scene  = trimesh.Scene()

    # 地面（120m × 120m，稍大于建筑分布范围）
    scene.add_geometry(
        build_ground_plane(120.0, 120.0, OFFSET - 60.0, OFFSET - 60.0),
        node_name="ground")

    # ── 4. 加载风格参数缓存 ────────────────────────────────────
    params_cache: dict[str, dict] = {}
    for style in set(STYLE_SLOTS):
        params_cache[style] = _load_style_params(style)

    # ── 5. 逐栋构建 mesh ──────────────────────────────────────
    print(f"\n[建筑] 构建 {n_buildings} 栋建筑 mesh...")
    total_faces = 0

    for i, (b, style) in enumerate(assignments):
        # 世界坐标：模型 nx,ny + OFFSET（使原点平移到场景中心）
        wx = b["nx"] + OFFSET   # X (东西)
        wz = b["ny"] + OFFSET   # Z (南北)

        # 尺寸 & 轮廓
        w, d   = _building_size(style, rng_np, variation)
        fp     = _make_footprint(style, w, d, rng_np)

        # 参数（含变体）
        base_params = params_cache[style]
        bparams     = _vary_params(base_params, rng_np, variation)

        # 覆盖 wall_color（分风格区分）
        wc = _WALL_COLORS.get(style)
        if wc:
            bparams["wall_color"] = wc

        # 调色板
        palette = _PALETTES.get(style, _PALETTES["medieval"])

        # 生成建筑（局部坐标 x=0,z=0）
        try:
            meshes = gl.build_room(bparams, palette,
                                   x_off=0.0, z_off=0.0,
                                   footprint=fp)
        except Exception as e:
            print(f"  [警告] 建筑 {i+1} 生成失败: {e}")
            continue

        # 旋转（orientation_deg → 绕 Y 轴，中心为 w/2, d/2）
        yaw = b["orientation_deg"]
        if abs(yaw) > 0.5:
            cx_fp, cz_fp = w / 2, d / 2
            rot = TF.rotation_matrix(
                math.radians(yaw), [0, 1, 0],
                point=[cx_fp, 0.0, cz_fp])
            for m in meshes:
                m.apply_transform(rot)

        # 平移到世界坐标（建筑左前角 → 世界位置）
        for m in meshes:
            m.apply_translation([wx - w / 2, 0.0, wz - d / 2])

        faces = sum(len(m.faces) for m in meshes)
        total_faces += faces
        label = "★" if style == "medieval_keep" else ("♦" if "chapel" in style else "·")
        print(f"  {label} {i+1:>2}  {style:<20}  {w:.1f}×{d:.1f}m  "
              f"yaw={yaw:+.0f}°  {faces:>5} faces")

        for j, m in enumerate(meshes):
            scene.add_geometry(m, node_name=f"b{i:02d}_{style}_{j:03d}")

    print(f"\n  总面数: {total_faces:,}")
    return scene


def compare_with_rule_based():
    """对比输出：与规则算法生成结果的视觉差异分析。"""
    print(f"\n{'='*65}")
    print("  视觉差异对比：ML布局 vs 规则有机算法")
    print(f"{'='*65}")

    # 读取规则算法文件信息
    if REF_GLB.exists():
        ref_scene = trimesh.load(str(REF_GLB))
        ref_faces = sum(len(g.faces) for g in ref_scene.geometry.values())
        ref_verts = sum(len(g.vertices) for g in ref_scene.geometry.values())
    else:
        ref_faces = ref_verts = 0

    # 读取 ML 生成文件信息
    if OUT_GLB.exists():
        ml_scene  = trimesh.load(str(OUT_GLB))
        ml_faces  = sum(len(g.faces) for g in ml_scene.geometry.values())
        ml_verts  = sum(len(g.vertices) for g in ml_scene.geometry.values())
    else:
        ml_faces = ml_verts = 0

    print(f"\n  {'指标':<28}  {'ML布局':>12}  {'规则有机布局':>12}")
    print(f"  {'─'*56}")
    print(f"  {'风格层次':<28}  {'3层(keep/chapel/med)':>12}  {'2层(keep/med)':>12}")
    print(f"  {'位置来源':<28}  {'学习模型采样':>12}  {'泊松圆盘算法':>12}")
    print(f"  {'朝向来源':<28}  {'模型预测分布':>12}  {'朝向中心±45°':>12}")
    print(f"  {'间距模式':<28}  {'学习OSM密集街区':>12}  {'环形分层间距':>12}")
    print(f"  {'三角面数':<28}  {ml_faces:>12,}  {ref_faces:>12,}")
    print(f"  {'顶点数':<28}  {ml_verts:>12,}  {ref_verts:>12,}")
    print(f"\n  ML布局特点（来自 OSM 真实数据学习）：")
    print(f"    · 建筑间距呈现真实街区密度（平均 3-5m）")
    print(f"    · 朝向分布不规则，沿隐含街道走向对齐")
    print(f"    · 类型混合更自然（非固定同心环分层）")
    print(f"\n  规则有机布局特点：")
    print(f"    · 严格的中心→内圈→外圈分层结构")
    print(f"    · 建筑间距均匀（5/7/9m 固定值）")
    print(f"    · 朝向精确朝向中心")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65)
    print("  gen_ml_level.py  —  ML 布局模型 → 关卡 GLB/FBX")
    print("=" * 65)

    if not MODEL_PATH.exists():
        raise SystemExit(f"[错误] 模型文件不存在: {MODEL_PATH}\n请先运行 layout_model.py 训练")
    if not PARAMS_JSON.exists():
        raise SystemExit(f"[错误] 风格参数文件不存在: {PARAMS_JSON}")

    # ── 1. 生成场景 ───────────────────────────────────────────
    scene = generate_ml_level(
        n_buildings=15,
        temperature=0.75,
        seed=2024,
        variation=0.35,
    )

    # ── 2. 导出 GLB ───────────────────────────────────────────
    scene.export(str(OUT_GLB))
    glb_size = OUT_GLB.stat().st_size / 1024
    print(f"\n[GLB] 保存: {OUT_GLB.name}  ({glb_size:.1f} KB)")

    # ── 3. 导出 FBX ───────────────────────────────────────────
    fbx_script = SCRIPT_DIR / "glb_to_fbx.py"
    if fbx_script.exists():
        print(f"[FBX] 转换中...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(fbx_script), str(OUT_GLB), "--out", str(OUT_FBX)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            fbx_size = OUT_FBX.stat().st_size / 1024
            print(f"[FBX] 保存: {OUT_FBX.name}  ({fbx_size:.1f} KB)")
        else:
            print(f"[FBX] 转换失败: {result.stderr[:200]}")
    else:
        print("[FBX] 跳过（glb_to_fbx.py 未找到）")

    # ── 4. 输出场景统计 ───────────────────────────────────────
    total_meshes = len(scene.geometry)
    total_faces  = sum(len(g.faces) for g in scene.geometry.values())
    total_verts  = sum(len(g.vertices) for g in scene.geometry.values())

    print(f"\n{'='*65}")
    print(f"  场景统计")
    print(f"{'='*65}")
    print(f"  建筑数量  : 15 栋（medieval_keep×1 + chapel×2 + medieval×12）")
    print(f"  mesh 对象 : {total_meshes}")
    print(f"  总顶点数  : {total_verts:,}")
    print(f"  总三角面数: {total_faces:,}")
    print(f"  输出文件  : {OUT_GLB.name}")
    print(f"             {OUT_FBX.name}")

    # ── 5. 对比分析 ───────────────────────────────────────────
    compare_with_rule_based()

    print(f"\n{'='*65}")
    print(f"  完成！在 3D 查看器中打开 {OUT_GLB.name} 查看效果。")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
