"""
gen_w3_level.py
Use the OSM+W3 combined layout model to generate a medieval town
level GLB + FBX, and compare spacing/orientation vs the old OSM-only model.
"""

import json
import math
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR   = Path(__file__).parent
MODEL_NEW    = SCRIPT_DIR / "models" / "layout_model_w3.pt"
MODEL_OLD    = SCRIPT_DIR / "models" / "layout_model.pt"
PARAMS_JSON  = SCRIPT_DIR / "trained_style_params.json"
OUT_GLB      = SCRIPT_DIR / "level_w3_layout.glb"
OUT_FBX      = SCRIPT_DIR / "level_w3_layout.fbx"

sys.path.insert(0, str(SCRIPT_DIR))
import generate_level as gl
import trimesh
import trimesh.transformations as TF
from shapely.geometry import box as sbox

from layout_model import (
    load_model, generate_layout, evaluate_layout,
    TYPES, TYPE2ID,
)

# ─── Style assignment (same as gen_ml_level.py) ──────────────────────────────
STYLE_SLOTS = (
    ["medieval_keep"]       +   # 0     centre
    ["medieval_chapel"] * 2 +   # 1-2   inner chapels
    ["medieval"] * 4        +   # 3-6   inner houses
    ["medieval"] * 8            # 7-14  outer ring
)

STYLE_SIZES = {
    "medieval_keep":   (9.0,  13.0,  9.0,  13.0),
    "medieval_chapel": (8.0,  11.0, 14.0,  20.0),
    "medieval":        (9.0,  14.0,  7.0,  11.0),
}

_PALETTES = {
    "medieval": {
        "floor":  [115, 95, 75, 255], "ceiling": [135, 115, 90, 255],
        "wall":   [162, 146, 122, 255], "door":   [88, 58, 28, 255],
        "window": [155, 195, 215, 180], "internal": [148, 133, 110, 255],
        "ground": [90, 80, 65, 255],
    },
    "medieval_chapel": {
        "floor":  [130, 110, 88, 255], "ceiling": [150, 130, 105, 255],
        "wall":   [175, 160, 138, 255], "door":   [80, 52, 24, 255],
        "window": [170, 210, 230, 200], "internal": [160, 145, 122, 255],
        "ground": [90, 80, 65, 255],
    },
    "medieval_keep": {
        "floor":  [100, 85, 68, 255], "ceiling": [120, 103, 82, 255],
        "wall":   [148, 134, 112, 255], "door":   [72, 46, 20, 255],
        "window": [130, 170, 190, 160], "internal": [138, 123, 100, 255],
        "ground": [78, 68, 54, 255],
    },
}

_WALL_COLORS = {
    "medieval_keep":   [0.52, 0.47, 0.40],
    "medieval_chapel": [0.64, 0.58, 0.50],
    "medieval":        None,
}


def _load_style_params(style: str) -> dict:
    data   = json.loads(PARAMS_JSON.read_text("utf-8"))
    styles = data.get("styles", {})
    if style not in styles:
        raise ValueError(f"Style '{style}' not found. Available: {list(styles)}")
    return dict(styles[style]["params"])


def _building_size(style: str, rng: np.random.Generator,
                   variation: float = 0.35) -> tuple[float, float]:
    wmin, wmax, dmin, dmax = STYLE_SIZES.get(style, (9.0, 14.0, 7.0, 11.0))
    w = rng.uniform(wmin, wmin + (wmax - wmin) * (0.5 + variation * 0.5))
    d = rng.uniform(dmin, dmin + (dmax - dmin) * (0.5 + variation * 0.5))
    return round(w, 1), round(d, 1)


def _make_footprint(style: str, w: float, d: float, rng: np.random.Generator):
    if style == "medieval":
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
            cut  = sbox(w * (0.5 - nx / 2), 0, w * (0.5 + nx / 2), d * nz)
            fp   = rect.difference(cut)
            return fp if not fp.is_empty else sbox(0, 0, w, d)
    return sbox(0, 0, w, d)


def _vary_params(params: dict, rng: np.random.Generator,
                 variation: float = 0.35) -> dict:
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


def build_ground_plane(area_w, area_d, offset_x, offset_z):
    ground = trimesh.creation.box(extents=[area_w, 0.15, area_d])
    color  = np.array([78, 68, 54, 255], dtype=np.uint8)
    ground.visual.face_colors = np.tile(color, (len(ground.faces), 1))
    ground.apply_translation([offset_x + area_w / 2, -0.075, offset_z + area_d / 2])
    return ground


def generate_w3_level(model_path: Path, n_buildings: int = 15,
                      temperature: float = 0.75, seed: int = 2024,
                      variation: float = 0.35) -> tuple[trimesh.Scene, list[dict]]:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng_np  = np.random.default_rng(seed)

    print(f"[Layout model] Loading {model_path.name}...")
    model    = load_model(model_path, device)
    gen_blds = generate_layout(model, device, n_buildings,
                                temperature=temperature, rng=rng_np)

    # Sort by distance, assign styles
    for b in gen_blds:
        b["_dist"] = math.hypot(b["nx"], b["ny"])
    gen_blds.sort(key=lambda b: b["_dist"])

    print(f"\n  {'#':>3}  {'nx':>7}  {'ny':>7}  {'ori':>6}  {'dist':>6}  style")
    print("  " + "-" * 52)
    assignments = []
    for i, (b, style) in enumerate(zip(gen_blds, STYLE_SLOTS)):
        print(f"  {i+1:>3}  {b['nx']:>7.1f}  {b['ny']:>7.1f}  "
              f"{b['orientation_deg']:>6.1f}  {b['_dist']:>6.1f}  {style}")
        assignments.append((b, style))

    OFFSET = 50.0
    scene  = trimesh.Scene()
    scene.add_geometry(
        build_ground_plane(120.0, 120.0, OFFSET - 60.0, OFFSET - 60.0),
        node_name="ground")

    params_cache = {s: _load_style_params(s) for s in set(STYLE_SLOTS)}

    print(f"\n[Mesh] Building {n_buildings} meshes...")
    total_faces = 0
    for i, (b, style) in enumerate(assignments):
        wx = b["nx"] + OFFSET
        wz = b["ny"] + OFFSET
        w, d   = _building_size(style, rng_np, variation)
        fp     = _make_footprint(style, w, d, rng_np)
        bparams = _vary_params(params_cache[style], rng_np, variation)
        wc = _WALL_COLORS.get(style)
        if wc:
            bparams["wall_color"] = wc
        palette = _PALETTES.get(style, _PALETTES["medieval"])
        try:
            meshes = gl.build_room(bparams, palette,
                                   x_off=0.0, z_off=0.0, footprint=fp)
        except Exception as e:
            print(f"  [warn] building {i+1} failed: {e}")
            continue

        yaw = b["orientation_deg"]
        if abs(yaw) > 0.5:
            rot = TF.rotation_matrix(
                math.radians(yaw), [0, 1, 0],
                point=[w / 2, 0.0, d / 2])
            for m in meshes:
                m.apply_transform(rot)
        for m in meshes:
            m.apply_translation([wx - w / 2, 0.0, wz - d / 2])

        faces = sum(len(m.faces) for m in meshes)
        total_faces += faces
        sym = ("*" if style == "medieval_keep"
               else ("+" if "chapel" in style else "."))
        print(f"  {sym} {i+1:>2}  {style:<20}  {w:.1f}x{d:.1f}m  "
              f"yaw={yaw:+.0f}  {faces} faces")
        for j, m in enumerate(meshes):
            scene.add_geometry(m, node_name=f"b{i:02d}_{style}_{j:03d}")

    print(f"\n  Total faces: {total_faces:,}")
    return scene, gen_blds


# ─── Comparison ──────────────────────────────────────────────────────────────

def compare_models(n_buildings: int = 15, temperature: float = 0.75,
                   seed: int = 2024):
    """
    Generate layouts from both old (OSM-only) and new (OSM+W3) models,
    compare spacing and orientation statistics.
    """
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng_old  = np.random.default_rng(seed)
    rng_new  = np.random.default_rng(seed)

    results = {}

    for label, mpath in [("OSM-only", MODEL_OLD), ("OSM+W3", MODEL_NEW)]:
        if not mpath.exists():
            print(f"  [skip] {label}: {mpath.name} not found")
            continue
        model  = load_model(mpath, device)
        blds   = generate_layout(model, device, n_buildings,
                                  temperature=temperature,
                                  rng=(rng_old if label == "OSM-only" else rng_new))
        # Rename orientation_deg for evaluate_layout
        metrics = evaluate_layout(blds, label=label)
        results[label] = {"buildings": blds, "metrics": metrics}

    if len(results) < 2:
        return results

    old_m = results["OSM-only"]["metrics"]
    new_m = results["OSM+W3"]["metrics"]

    print(f"\n{'='*65}")
    print("  Spacing / Orientation Comparison (15 buildings)")
    print(f"{'='*65}")
    print(f"  {'Metric':<22}  {'OSM-only':>12}  {'OSM+W3':>12}  {'Delta':>10}")
    print("  " + "-" * 60)

    metrics_to_show = [
        ("Avg nearest-neighbour", "avg_nnd_m",    "{:.2f} m",  "{:+.2f}"),
        ("Orientation std",       "ori_std_deg",  "{:.1f} deg","{:+.1f}"),
        ("Type diversity (H)",    "type_entropy", "{:.3f}",    "{:+.3f}"),
        ("Avg area",              "avg_area_m2",  "{:.1f} m2", "{:+.1f}"),
    ]
    for name, key, fmt, dfmt in metrics_to_show:
        ov = old_m.get(key, float("nan"))
        nv = new_m.get(key, float("nan"))
        dv = nv - ov if not (math.isnan(ov) or math.isnan(nv)) else float("nan")
        ov_s = fmt.format(ov) if not math.isnan(ov) else "n/a"
        nv_s = fmt.format(nv) if not math.isnan(nv) else "n/a"
        dv_s = dfmt.format(dv) if not math.isnan(dv) else "n/a"
        print(f"  {name:<22}  {ov_s:>12}  {nv_s:>12}  {dv_s:>10}")

    # Per-building position table
    print(f"\n  Old (OSM-only) positions:")
    print(f"  {'#':>3}  {'nx':>7}  {'ny':>7}  {'ori':>6}")
    for i, b in enumerate(results["OSM-only"]["buildings"][:15], 1):
        print(f"  {i:>3}  {b['nx']:>7.1f}  {b['ny']:>7.1f}  {b['orientation_deg']:>6.1f}")

    print(f"\n  New (OSM+W3) positions:")
    print(f"  {'#':>3}  {'nx':>7}  {'ny':>7}  {'ori':>6}")
    for i, b in enumerate(results["OSM+W3"]["buildings"][:15], 1):
        print(f"  {i:>3}  {b['nx']:>7.1f}  {b['ny']:>7.1f}  {b['orientation_deg']:>6.1f}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65)
    print("  gen_w3_level.py  --  OSM+W3 Model -> Level GLB/FBX")
    print("=" * 65)

    if not MODEL_NEW.exists():
        raise SystemExit(
            f"Model not found: {MODEL_NEW}\n"
            "Run first:  python layout_model.py --source combined")
    if not PARAMS_JSON.exists():
        raise SystemExit(f"Style params not found: {PARAMS_JSON}")

    # 1. Compare old vs new model
    print("\n[Model comparison]")
    compare_models(n_buildings=15, temperature=0.75, seed=2024)

    # 2. Generate level with new model
    print(f"\n{'='*65}")
    print("  Generating level_w3_layout.glb...")
    print(f"{'='*65}")
    scene, blds = generate_w3_level(
        model_path=MODEL_NEW,
        n_buildings=15,
        temperature=0.75,
        seed=2024,
        variation=0.35,
    )

    # 3. Export GLB
    scene.export(str(OUT_GLB))
    glb_kb = OUT_GLB.stat().st_size / 1024
    print(f"\n[GLB] Saved: {OUT_GLB.name}  ({glb_kb:.1f} KB)")

    # 4. Export FBX
    fbx_script = SCRIPT_DIR / "glb_to_fbx.py"
    if fbx_script.exists():
        res = subprocess.run(
            [sys.executable, str(fbx_script), str(OUT_GLB), "--out", str(OUT_FBX)],
            capture_output=True, text=True)
        if res.returncode == 0:
            fbx_kb = OUT_FBX.stat().st_size / 1024
            print(f"[FBX] Saved: {OUT_FBX.name}  ({fbx_kb:.1f} KB)")
        else:
            print(f"[FBX] Failed: {res.stderr[:200]}")
    else:
        print("[FBX] Skipped (glb_to_fbx.py not found)")

    # 5. Scene stats
    total_meshes = len(scene.geometry)
    total_faces  = sum(len(g.faces) for g in scene.geometry.values())
    total_verts  = sum(len(g.vertices) for g in scene.geometry.values())
    print(f"\n{'='*65}")
    print(f"  Scene stats")
    print(f"{'='*65}")
    print(f"  Buildings : 15  (keep x1 + chapel x2 + medieval x12)")
    print(f"  Meshes    : {total_meshes}")
    print(f"  Vertices  : {total_verts:,}")
    print(f"  Faces     : {total_faces:,}")
    print(f"  Output    : {OUT_GLB.name}")
    print(f"             {OUT_FBX.name}")
    print(f"\n  Done! Open {OUT_GLB.name} in a 3D viewer.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
