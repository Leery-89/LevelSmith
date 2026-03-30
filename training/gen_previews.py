"""
gen_previews.py
Generate preview GLBs for multiple styles showing mesh complexity effects.
"""
import sys, json, torch
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from model import StyleParamMLP
from style_registry import (
    STYLE_REGISTRY, OUTPUT_DIM, FEATURE_DIM, OUTPUT_KEYS,
    get_feature_vector, denormalize_params,
)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    linear_keys = [k for k in sd if k.endswith(".weight") and len(sd[k].shape) == 2]
    linear_keys.sort(key=lambda k: int(k.split(".")[1]))
    hdims = [sd[k].shape[0] for k in linear_keys[:-1]]
    out_dim = sd[linear_keys[-1]].shape[0]
    model = StyleParamMLP(FEATURE_DIM, out_dim, hdims, 0.2).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def generate_preview(model, device, style_name, output_path):
    import generate_level as gl
    import trimesh
    from shapely.geometry import box as sbox

    fv = torch.from_numpy(get_feature_vector(style_name)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(fv).cpu().numpy()[0]
    params = denormalize_params(pred)

    # Style-specific palettes
    PALETTES = {
        "medieval_keep": {
            "floor": [100, 85, 68, 255], "ceiling": [120, 103, 82, 255],
            "wall": [148, 134, 112, 255], "door": [72, 46, 20, 255],
            "window": [130, 170, 190, 160], "internal": [138, 123, 100, 255],
            "ground": [78, 68, 54, 255],
        },
        "industrial": {
            "floor": [95, 90, 85, 255], "ceiling": [110, 105, 100, 255],
            "wall": [92, 87, 82, 255], "door": [70, 65, 60, 255],
            "window": [140, 160, 175, 180], "internal": [100, 95, 90, 255],
            "ground": [75, 72, 68, 255],
        },
        "japanese_temple": {
            "floor": [140, 110, 70, 255], "ceiling": [160, 130, 85, 255],
            "wall": [210, 195, 160, 255], "door": [100, 55, 20, 255],
            "window": [200, 210, 220, 180], "internal": [170, 145, 100, 255],
            "ground": [95, 85, 65, 255],
        },
    }
    palette = PALETTES.get(style_name, PALETTES["medieval_keep"])

    wc = [params.get("wall_color_r", 0.6), params.get("wall_color_g", 0.55),
          params.get("wall_color_b", 0.48)]
    params["wall_color"] = wc
    params["win_spec"] = {"density": params.get("win_density", 0.3),
                          "width": params.get("win_width", 0.8),
                          "height": params.get("win_height", 1.0)}
    params["door_spec"] = {"width": params.get("door_width", 1.0),
                           "height": params.get("door_height", 2.2)}
    params["height_range"] = [params.get("height_range_min", 3.0),
                              params.get("height_range_max", 6.0)]

    # Vary building size per style
    sizes = {
        "medieval_keep": (14.0, 14.0),
        "industrial": (16.0, 10.0),
        "japanese_temple": (12.0, 10.0),
    }
    w, d = sizes.get(style_name, (12.0, 10.0))
    fp = sbox(0, 0, w, d)
    meshes = gl.build_room(params, palette, x_off=0, z_off=0, footprint=fp)

    scene = trimesh.Scene()
    for i, m in enumerate(meshes):
        scene.add_geometry(m, node_name=f"part_{i:03d}")
    scene.export(str(output_path))
    faces = sum(len(g.faces) for g in scene.geometry.values())
    verts = sum(len(g.vertices) for g in scene.geometry.values())
    return params, faces, verts


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  gen_previews.py  --  Generate style comparison GLBs")
    print("=" * 65)

    ckpt_path = SCRIPT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        sys.exit(f"Model not found: {ckpt_path}")

    model = load_model(ckpt_path, device)

    STYLES = ["medieval_keep", "industrial", "japanese_temple"]
    results = []

    for style in STYLES:
        out = SCRIPT_DIR / f"{style}_preview.glb"
        params, faces, verts = generate_preview(model, device, style, out)
        mc = params.get("mesh_complexity", 0)
        dd = params.get("detail_density", 0)
        sr = params.get("simple_ratio", 0)
        results.append({
            "style": style,
            "mesh_complexity": round(mc, 3),
            "detail_density": round(dd, 3),
            "simple_ratio": round(sr, 3),
            "faces": faces,
            "verts": verts,
            "height": round(params.get("height_range_max", 0), 1),
            "roof_type": params.get("roof_type", 0),
            "has_battlements": params.get("has_battlements", 0),
            "has_arch": params.get("has_arch", 0),
            "column_count": params.get("column_count", 0),
            "win_density": round(params.get("win_density", 0), 3),
        })
        print(f"  {style}: {out.name} ({faces} faces, {verts} verts)")

    # Print comparison table
    print(f"\n{'='*65}")
    print(f"  Mesh Complexity Comparison")
    print(f"{'='*65}")
    print(f"  {'Param':<22} {'medieval_keep':>14} {'industrial':>14} {'jp_temple':>14}")
    print("  " + "-" * 58)

    keys = ["mesh_complexity", "detail_density", "simple_ratio",
            "faces", "verts", "height", "roof_type",
            "has_battlements", "has_arch", "column_count", "win_density"]
    for k in keys:
        vals = [r[k] for r in results]
        fmt = "{:>14}" if isinstance(vals[0], int) else "{:>14.3f}"
        line = f"  {k:<22}"
        for v in vals:
            if isinstance(v, int):
                line += f"{v:>14}"
            else:
                line += f"{v:>14.3f}"
        print(line)

    # Explain effects
    print(f"\n  Effects of mesh_complexity parameters:")
    for r in results:
        style = r["style"]
        mc, dd, sr = r["mesh_complexity"], r["detail_density"], r["simple_ratio"]
        effects = []
        if mc > 0.7:
            effects.append("force arch+columns (mc>0.7)")
        if dd > 0.6:
            effects.append(f"win density boosted (dd>0.6)")
        if sr > 0.6:
            effects.append("simplified geo (sr>0.6)")
        if sr > 0.75:
            effects.append("forced flat roof (sr>0.75)")
        if not effects:
            effects.append("no threshold triggered")
        print(f"  {style}: {', '.join(effects)}")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()
