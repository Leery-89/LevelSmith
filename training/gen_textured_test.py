"""
gen_textured_test.py
Generate textured building test GLBs for medieval and modern styles.
"""
import sys
sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path
from procedural_materials import apply_materials_to_scene

SCRIPT_DIR = Path(__file__).parent

def generate_textured(style, layout, count, output_name, seed=42):
    import level_layout

    print(f"\n{'='*60}")
    print(f"  Generating: {output_name}")
    print(f"  Style: {style}, Layout: {layout}, Count: {count}")
    print(f"{'='*60}")

    scene = level_layout.generate_level(
        style=style,
        layout_type=layout,
        building_count=count,
        seed=seed,
    )

    # Get wall color from trained params
    params_path = SCRIPT_DIR / "trained_style_params.json"
    wc = (0.55, 0.48, 0.40)
    if params_path.exists():
        data = json.loads(params_path.read_text("utf-8"))
        if style in data.get("styles", {}):
            p = data["styles"][style]["params"]
            wc = (p.get("wall_color_r", 0.55),
                  p.get("wall_color_g", 0.48),
                  p.get("wall_color_b", 0.40))

    print(f"\n  Applying PBR materials (wall_color={wc})...")
    scene = apply_materials_to_scene(scene, style, wc)

    out_path = SCRIPT_DIR / output_name
    scene.export(str(out_path))
    size_kb = out_path.stat().st_size / 1024

    n_meshes = len(scene.geometry)
    n_faces = sum(len(g.faces) for g in scene.geometry.values())
    n_textured = sum(1 for g in scene.geometry.values()
                     if hasattr(g.visual, 'material') and
                     hasattr(g.visual.material, 'baseColorTexture'))

    print(f"\n  Output: {out_path.name} ({size_kb:.0f} KB)")
    print(f"  Meshes: {n_meshes}, Faces: {n_faces:,}")
    print(f"  Textured: {n_textured}/{n_meshes}")


if __name__ == "__main__":
    generate_textured("medieval", "street", 6, "medieval_texture_test.glb", seed=42)
    generate_textured("modern", "grid", 4, "modern_texture_test.glb", seed=42)
    print(f"\n{'='*60}")
    print("  Done! Open in gltf-viewer to preview textures.")
    print(f"{'='*60}")
