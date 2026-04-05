"""
gen_two_levels.py
Generate two contrast scenes using the new OSM+W3 layout model:
  1. level_medieval_v2  – 15 buildings, medieval mixed style
  2. level_skellige_v1  – 12 buildings, horror/industrial mixed

Each scene exports GLB + FBX and prints layout metrics.
"""
import json, math, sys, subprocess
from pathlib import Path

import numpy as np
import torch
import trimesh
import trimesh.transformations as TF

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "models" / "layout_model_w3.pt"
PARAMS_JSON = SCRIPT_DIR / "trained_style_params.json"
FBX_SCRIPT  = SCRIPT_DIR / "glb_to_fbx.py"

sys.path.insert(0, str(SCRIPT_DIR))
import generate_level as gl
from shapely.geometry import box as sbox
from layout_model import load_model, generate_layout, evaluate_layout
from level_layout import _obb_polygon, _obb_collides, SAFETY_GAP, OBB_BUFFER
from gen_w3_level import (
    _building_size   as building_size_fn,
    _make_footprint  as make_footprint_fn,
    _vary_params     as vary_params_fn,
    build_ground_plane,
    _load_style_params as load_style_params,
    STYLE_SIZES,
)

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

# ─── Style palettes ────────────────────────────────────────────────────────────
PALETTES = {
    "medieval": {
        "floor": [115, 95, 75, 255], "ceiling": [135, 115, 90, 255],
        "wall": [162, 146, 122, 255], "door": [88, 58, 28, 255],
        "window": [155, 195, 215, 180], "internal": [148, 133, 110, 255],
        "ground": [90, 80, 65, 255],
    },
    "medieval_chapel": {
        "floor": [130, 110, 88, 255], "ceiling": [150, 130, 105, 255],
        "wall": [175, 160, 138, 255], "door": [80, 52, 24, 255],
        "window": [170, 210, 230, 200], "internal": [160, 145, 122, 255],
        "ground": [90, 80, 65, 255],
    },
    "medieval_keep": {
        "floor": [100, 85, 68, 255], "ceiling": [120, 103, 82, 255],
        "wall": [148, 134, 112, 255], "door": [72, 46, 20, 255],
        "window": [130, 170, 190, 160], "internal": [138, 123, 100, 255],
        "ground": [78, 68, 54, 255],
    },
    "horror": {
        "floor": [55, 45, 38, 255], "ceiling": [40, 32, 28, 255],
        "wall": [80, 65, 55, 255], "door": [42, 28, 18, 255],
        "window": [100, 80, 60, 120], "internal": [70, 55, 45, 255],
        "ground": [45, 38, 30, 255],
    },
    "horror_crypt": {
        "floor": [48, 40, 34, 255], "ceiling": [35, 28, 24, 255],
        "wall": [72, 58, 48, 255], "door": [35, 22, 14, 255],
        "window": [85, 65, 50, 100], "internal": [62, 48, 38, 255],
        "ground": [38, 32, 26, 255],
    },
    "industrial": {
        "floor": [85, 80, 75, 255], "ceiling": [70, 65, 60, 255],
        "wall": [120, 112, 100, 255], "door": [60, 55, 50, 255],
        "window": [180, 200, 210, 160], "internal": [108, 100, 90, 255],
        "ground": [75, 70, 65, 255],
    },
    "industrial_workshop": {
        "floor": [78, 72, 66, 255], "ceiling": [65, 60, 55, 255],
        "wall": [110, 102, 92, 255], "door": [55, 50, 46, 255],
        "window": [170, 190, 200, 150], "internal": [100, 93, 84, 255],
        "ground": [68, 63, 58, 255],
    },
}

# STYLE_SIZES, load_style_params, building_size_fn, make_footprint_fn,
# vary_params_fn, build_ground_plane imported from gen_w3_level above


def layout_metrics(buildings: list[dict]) -> dict:
    """Compute layout quality metrics."""
    if len(buildings) < 2:
        return {}
    pts = np.array([[b["nx"], b["ny"]] for b in buildings])

    # Average nearest-neighbour distance
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    nnd = D.min(axis=1).mean()

    # Orientation std dev
    angles = np.array([b.get("orientation_deg", 0.0) for b in buildings])
    sins, coss = np.sin(np.radians(angles)), np.cos(np.radians(angles))
    R = np.sqrt(sins.mean()**2 + coss.mean()**2)
    ori_std = math.degrees(math.sqrt(-2 * math.log(R + 1e-9)))

    # Type entropy
    from collections import Counter
    import math as _m
    tcounts = Counter(b.get("type", "building") for b in buildings)
    total = sum(tcounts.values())
    entropy = -sum((c/total) * _m.log2(c/total + 1e-12) for c in tcounts.values())

    return {
        "n_buildings": len(buildings),
        "avg_nnd_m": round(nnd, 2),
        "ori_std_deg": round(ori_std, 1),
        "type_entropy": round(entropy, 3),
        "type_dist": dict(tcounts),
    }


def _force_find_free_spot(
        nx: float, ny: float, w: float, d: float, yaw: float,
        placed: list, safety: float, rng: np.random.Generator,
) -> tuple[float, float]:
    """
    强制螺旋扫描找到最近的无碰撞位置。
    以 (nx, ny) 为中心，从小步长开始向外螺旋搜索，
    直到找到安全位置为止（保证终止）。
    """
    step = math.hypot(w, d) / 2 + SAFETY_GAP + OBB_BUFFER  # 每圈步长
    for ring in range(1, 40):
        radius = step * ring
        n_pts  = max(8, ring * 8)
        angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
        rng.shuffle(angles)
        for a in angles:
            cx2 = nx + radius * math.cos(a)
            cy2 = ny + radius * math.sin(a)
            if not _obb_collides(cx2, cy2, w, d, yaw, placed, safety):
                return cx2, cy2
    # 极端情况：移到很远处（不应发生）
    return nx + step * 40, ny


def resolve_overlaps(
        gen_blds: list[dict],
        style_slots: list[str],
        rng: np.random.Generator,
        max_retries: int = 200,
) -> list[dict]:
    """
    对 ML 生成的建筑坐标做 OBB 碰撞解消：
    1. 随机扰动重试 max_retries 次（默认 200）
    2. 仍失败则螺旋扫描强制找到最近空位
    """
    resolved = []
    placed   = []          # list[Polygon]
    moved    = 0
    forced   = 0

    for b, style in zip(gen_blds, style_slots):
        wmin, wmax, dmin, dmax = STYLE_SIZES.get(style, (8.0, 14.0, 6.0, 12.0))
        w = (wmin + wmax) / 2
        d = (dmin + dmax) / 2
        safety = math.hypot(w, d) / 2 + SAFETY_GAP

        nx, ny = b["nx"], b["ny"]
        yaw    = b.get("orientation_deg", 0.0)

        if not _obb_collides(nx, ny, w, d, yaw, placed, safety):
            placed.append(_obb_polygon(nx, ny, w, d, yaw))
            resolved.append(b)
            continue

        # ── 阶段 1：随机扰动，最多 max_retries 次 ──────────────
        diag = math.hypot(w, d)
        search_r = diag * 3          # 扰动搜索半径随重试轮次增大
        placed_ok = False
        for attempt in range(max_retries):
            r = search_r * (1 + attempt / max_retries)   # 线性扩大搜索范围
            a = rng.uniform(0, 2 * math.pi)
            cx2 = nx + r * math.cos(a)
            cy2 = ny + r * math.sin(a)
            if not _obb_collides(cx2, cy2, w, d, yaw, placed, safety):
                new_b = dict(b)
                new_b["nx"] = cx2
                new_b["ny"] = cy2
                placed.append(_obb_polygon(cx2, cy2, w, d, yaw))
                resolved.append(new_b)
                placed_ok = True
                moved += 1
                break

        if not placed_ok:
            # ── 阶段 2：螺旋扫描强制移位 ────────────────────────
            cx2, cy2 = _force_find_free_spot(nx, ny, w, d, yaw, placed, safety, rng)
            new_b = dict(b)
            new_b["nx"] = cx2
            new_b["ny"] = cy2
            placed.append(_obb_polygon(cx2, cy2, w, d, yaw))
            resolved.append(new_b)
            forced += 1

    parts = []
    if moved:  parts.append(f"{moved} 扰动解消")
    if forced: parts.append(f"{forced} 螺旋强制移位")
    if parts:
        print(f"  [OBB] {', '.join(parts)}")
    return resolved


def generate_scene(
        name: str,
        out_glb: Path,
        out_fbx: Path,
        style_slots: list[str],
        n_buildings: int,
        temperature: float = 0.75,
        seed: int = 42,
        variation: float = 0.35,
) -> dict:
    """Generate one scene, export GLB+FBX, return metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    print(f"\n{'='*65}")
    print(f"  Scene: {name}  ({n_buildings} buildings, T={temperature})")
    print(f"{'='*65}")

    model = load_model(MODEL_PATH, device)
    gen_blds = generate_layout(model, device, n_buildings,
                               temperature=temperature, rng=rng)

    # Sort by distance from centroid, assign styles
    for b in gen_blds:
        b["_dist"] = math.hypot(b["nx"], b["ny"])
    gen_blds.sort(key=lambda b: b["_dist"])

    slots = (style_slots * ((n_buildings // len(style_slots)) + 1))[:n_buildings]

    # OBB 碰撞解消：对 ML 布局做最多 50 次重试扰动
    gen_blds = resolve_overlaps(gen_blds, list(slots), rng)

    print(f"\n  {'#':>3}  {'nx':>7}  {'ny':>7}  {'ori':>7}  {'dist':>6}  style")
    print("  " + "-" * 56)
    assignments = []
    for i, (b, style) in enumerate(zip(gen_blds, slots)):
        print(f"  {i+1:>3}  {b['nx']:>7.1f}  {b['ny']:>7.1f}  "
              f"{b['orientation_deg']:>7.1f}  {b['_dist']:>6.1f}  {style}")
        assignments.append((b, style))

    OFFSET = 50.0
    scene = trimesh.Scene()
    scene.add_geometry(
        build_ground_plane(130.0, 130.0, OFFSET - 65.0, OFFSET - 65.0),
        node_name="ground")

    params_cache = {s: load_style_params(s) for s in set(slots)}
    print(f"\n[Mesh] Building {n_buildings} meshes...")
    total_faces = 0

    for i, (b, style) in enumerate(assignments):
        wx = b["nx"] + OFFSET
        wz = b["ny"] + OFFSET
        w, d = building_size_fn(style, rng, variation)
        fp = make_footprint_fn(style, w, d, rng)
        bparams = vary_params_fn(params_cache[style], rng, variation)
        palette = PALETTES.get(style, PALETTES["medieval"])

        try:
            meshes = gl.build_room(bparams, palette,
                                   x_off=0.0, z_off=0.0, footprint=fp)
        except Exception as e:
            print(f"  [warn] building {i+1} ({style}) failed: {e}")
            continue

        yaw = b["orientation_deg"]
        if abs(yaw) > 0.5:
            rot = TF.rotation_matrix(
                math.radians(yaw), [0, 1, 0], point=[w/2, 0.0, d/2])
            for m in meshes:
                m.apply_transform(rot)
        for m in meshes:
            m.apply_translation([wx - w/2, 0.0, wz - d/2])

        faces = sum(len(m.faces) for m in meshes)
        total_faces += faces
        sym = "*" if i == 0 else ("+" if i < 3 else ".")
        print(f"  {sym} {i+1:>2}  {style:<24}  {w:.1f}x{d:.1f}m  "
              f"yaw={yaw:+.0f}  {faces} faces")
        for j, m in enumerate(meshes):
            scene.add_geometry(m, node_name=f"b{i:02d}_{style}_{j:03d}")

    print(f"\n  Total faces: {total_faces:,}")

    # Export GLB
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_glb))
    print(f"[GLB] {out_glb.name}  ({out_glb.stat().st_size/1024:.1f} KB)")

    # Export FBX
    if FBX_SCRIPT.exists():
        res = subprocess.run(
            [sys.executable, str(FBX_SCRIPT), str(out_glb), "--out", str(out_fbx)],
            capture_output=True, text=True)
        if res.returncode == 0 and out_fbx.exists():
            print(f"[FBX] {out_fbx.name}  ({out_fbx.stat().st_size/1024:.1f} KB)")
        else:
            print(f"[FBX] Failed: {res.stderr[:200]}")
    else:
        print("[FBX] Skipped (glb_to_fbx.py not found)")

    # Metrics — use assigned style names for type distribution
    for b, style in zip(gen_blds, slots):
        b["type"] = style
    metrics = layout_metrics(gen_blds)
    return metrics


# ─── Scene definitions ─────────────────────────────────────────────────────────

MEDIEVAL_SLOTS = (
    ["medieval_keep"]       +   # 1  centre landmark
    ["medieval_chapel"] * 2 +   # 2-3 chapels
    ["medieval"] * 12           # 4-15 houses
)

SKELLIGE_SLOTS = (
    ["horror_crypt"]        +   # 1  dark centrepiece
    ["horror"] * 3          +   # 2-4 horror buildings
    ["industrial_workshop"] * 4 +  # 5-8 workshops
    ["industrial"] * 4          # 9-12 industrial
)


def main():
    print("=" * 65)
    print("  gen_two_levels.py  --  Two contrast scenes")
    print("=" * 65)

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found: {MODEL_PATH}\n"
            "Run: python layout_model.py --source combined")
    if not PARAMS_JSON.exists():
        raise SystemExit(f"Style params not found: {PARAMS_JSON}")

    results = {}

    # ── Scene 1: Medieval town ─────────────────────────────────────────────────
    m1 = generate_scene(
        name="Medieval Town v2",
        out_glb=SCRIPT_DIR / "level_medieval_v2.glb",
        out_fbx=SCRIPT_DIR / "level_medieval_v2.fbx",
        style_slots=MEDIEVAL_SLOTS,
        n_buildings=15,
        temperature=0.75,
        seed=2024,
        variation=0.30,
    )
    results["medieval_v2"] = m1

    # ── Scene 2: Skellige village ──────────────────────────────────────────────
    m2 = generate_scene(
        name="Skellige Village v1",
        out_glb=SCRIPT_DIR / "level_skellige_v1.glb",
        out_fbx=SCRIPT_DIR / "level_skellige_v1.fbx",
        style_slots=SKELLIGE_SLOTS,
        n_buildings=12,
        temperature=0.80,
        seed=1984,
        variation=0.40,
    )
    results["skellige_v1"] = m2

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  LAYOUT METRICS SUMMARY")
    print(f"{'='*65}")
    hdr = f"  {'Metric':<26}  {'Medieval v2':>14}  {'Skellige v1':>14}"
    print(hdr)
    print("  " + "-" * 58)
    for metric, label in [
        ("n_buildings",   "Buildings"),
        ("avg_nnd_m",     "Avg Nearest-Neighbour (m)"),
        ("ori_std_deg",   "Orientation Std Dev (deg)"),
        ("type_entropy",  "Type Diversity Entropy"),
    ]:
        v1 = results["medieval_v2"].get(metric, "—")
        v2 = results["skellige_v1"].get(metric, "—")
        print(f"  {label:<26}  {str(v1):>14}  {str(v2):>14}")

    print(f"\n  Type distributions:")
    for scene_key, scene_label in [("medieval_v2", "Medieval v2"),
                                    ("skellige_v1", "Skellige v1")]:
        td = results[scene_key].get("type_dist", {})
        print(f"    {scene_label}: " +
              "  ".join(f"{k}:{v}" for k, v in sorted(td.items(), key=lambda kv: -kv[1])))

    print(f"\n  Output files:")
    for fn in ["level_medieval_v2.glb", "level_medieval_v2.fbx",
               "level_skellige_v1.glb", "level_skellige_v1.fbx"]:
        p = SCRIPT_DIR / fn
        if p.exists():
            print(f"    {fn:<32} ({p.stat().st_size/1024:.1f} KB)")
        else:
            print(f"    {fn:<32} (missing)")


if __name__ == "__main__":
    main()
