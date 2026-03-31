"""
procedural_materials.py
Procedural PBR texture generation for LevelSmith buildings.

Generates tileable textures at runtime (no external image files needed):
  - Stone/brick walls (medieval, fantasy, horror)
  - Concrete/plaster walls (modern, industrial)
  - Wood grain (japanese, doors)
  - Roof tiles / shingles
  - Glass (translucent blue)

Each material includes: baseColorTexture + roughness + metallic factors.
Textures are PIL Images that trimesh can embed directly in GLB.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TEX_SIZE = 256  # texture resolution


# ═══════════════════════════════════════════════════════════════════
# Noise helpers
# ═══════════════════════════════════════════════════════════════════

def _noise_layer(w: int, h: int, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Generate a smooth noise field [0,1] using multi-octave value noise."""
    rng = np.random.default_rng(seed)
    result = np.zeros((h, w), dtype=np.float32)
    for octave in range(4):
        freq = 2 ** octave
        amp = 0.5 ** octave
        small = rng.random((max(2, h // (8 // freq)), max(2, w // (8 // freq)))).astype(np.float32)
        # Upsample with bilinear interpolation via PIL
        img = Image.fromarray((small * 255).astype(np.uint8), mode='L')
        img = img.resize((w, h), Image.BILINEAR)
        result += np.array(img, dtype=np.float32) / 255.0 * amp * scale
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


def _add_grain(img_array: np.ndarray, amount: float = 0.03, seed: int = 42) -> np.ndarray:
    """Add subtle grain noise to an RGB array [0,1]."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-amount, amount, img_array.shape).astype(np.float32)
    return np.clip(img_array + noise, 0, 1)


# ═══════════════════════════════════════════════════════════════════
# Texture generators
# ═══════════════════════════════════════════════════════════════════

def tex_stone_wall(base_rgb=(0.55, 0.48, 0.40), mortar_rgb=(0.40, 0.36, 0.32),
                   brick_h=32, brick_w=64, mortar_w=3, variation=0.03, seed=0) -> Image.Image:
    """Stone/brick wall: rows of offset rectangular blocks with mortar lines."""
    sz = TEX_SIZE
    img = np.zeros((sz, sz, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)

    # Fill with mortar color
    img[:] = mortar_rgb

    # Draw bricks
    y = 0
    row = 0
    while y < sz:
        bh = brick_h + rng.integers(-4, 5)
        offset = (brick_w // 2) * (row % 2)
        x = -offset
        while x < sz:
            bw = brick_w + rng.integers(-8, 9)
            # Grey-only variation: same offset for all channels (no color shift)
            grey_shift = rng.uniform(-variation, variation)
            br = np.array(base_rgb) + grey_shift
            br = np.clip(br, 0, 1)
            y0 = max(0, y + mortar_w)
            y1 = min(sz, y + bh)
            x0 = max(0, x + mortar_w)
            x1 = min(sz, x + bw)
            if y1 > y0 and x1 > x0:
                img[y0:y1, x0:x1] = br
            x += bw + mortar_w
        y += bh + mortar_w
        row += 1

    img = _add_grain(img, 0.015, seed)
    return Image.fromarray((img * 255).astype(np.uint8))


def tex_concrete(base_rgb=(0.62, 0.60, 0.58), noise_amp=0.06, seed=0) -> Image.Image:
    """Smooth concrete with subtle surface variation."""
    sz = TEX_SIZE
    noise = _noise_layer(sz, sz, scale=0.10, seed=seed)
    img = np.zeros((sz, sz, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = base_rgb[c] + (noise - 0.5) * noise_amp
    img = _add_grain(img, 0.01, seed)
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def tex_wood(base_rgb=(0.45, 0.30, 0.18), seed=0) -> Image.Image:
    """Wood grain texture: horizontal streaks with knot variation."""
    sz = TEX_SIZE
    img = np.zeros((sz, sz, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)

    # Base color
    img[:] = base_rgb

    # Horizontal grain lines
    for y in range(sz):
        # Grain intensity varies sinusoidally
        grain = 0.5 + 0.5 * np.sin(y * 0.3 + rng.uniform(0, 6.28))
        grain += rng.uniform(-0.1, 0.1)
        darken = grain * 0.08
        img[y, :, :] -= darken

    # Subtle knots (dark spots)
    for _ in range(3):
        ky, kx = rng.integers(20, sz - 20, 2)
        kr = rng.integers(5, 15)
        yy, xx = np.ogrid[-ky:sz - ky, -kx:sz - kx]
        mask = (yy ** 2 + xx ** 2) < kr ** 2
        img[mask] *= 0.85

    img = _add_grain(img, 0.02, seed)
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def tex_roof_tiles(base_rgb=(0.45, 0.40, 0.35), shadow_rgb=None, seed=0) -> Image.Image:
    """Roof tile pattern: overlapping curved rows."""
    sz = TEX_SIZE
    img = np.zeros((sz, sz, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if shadow_rgb is None:
        shadow_rgb = tuple(max(0, c - 0.15) for c in base_rgb)
    img[:] = shadow_rgb  # gaps between tiles use shadow color

    tile_h = 24
    tile_w = 32
    y = 0
    row = 0
    while y < sz:
        offset = (tile_w // 2) * (row % 2)
        x = -offset
        while x < sz:
            shade = rng.uniform(-0.03, 0.03)
            y0 = max(0, y)
            y1 = min(sz, y + tile_h)
            x0 = max(0, x)
            x1 = min(sz, x + tile_w)
            if y1 > y0 and x1 > x0:
                # Fill tile with base color + slight grey variation
                img[y0:y1, x0:x1] = np.array(base_rgb) + shade
                # Bottom edge shadow
                shadow_h = min(3, y1 - y0)
                img[y1 - shadow_h:y1, x0:x1] -= 0.04
            x += tile_w
        y += tile_h - 4  # overlap
        row += 1

    img = _add_grain(img, 0.02, seed)
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def tex_glass(base_rgb=(0.55, 0.72, 0.82), seed=0) -> Image.Image:
    """Simple glass texture: mostly uniform with subtle reflection streaks."""
    sz = TEX_SIZE
    img = np.zeros((sz, sz, 3), dtype=np.float32)
    img[:] = base_rgb
    # Vertical reflection streaks
    rng = np.random.default_rng(seed)
    for _ in range(5):
        x = rng.integers(10, sz - 10)
        w = rng.integers(2, 8)
        img[:, max(0, x - w):min(sz, x + w)] += 0.05
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def tex_rusty_metal(base_rgb=(0.42, 0.35, 0.28), seed=0) -> Image.Image:
    """Rusty/oxidized metal: base metal with rust patches."""
    sz = TEX_SIZE
    metal_rgb = (0.50, 0.48, 0.45)
    rust_rgb = base_rgb

    noise = _noise_layer(sz, sz, scale=1.0, seed=seed)
    img = np.zeros((sz, sz, 3), dtype=np.float32)

    for c in range(3):
        # Blend between metal and rust based on noise
        img[:, :, c] = metal_rgb[c] * (1 - noise) + rust_rgb[c] * noise

    # Scratch lines
    rng = np.random.default_rng(seed + 1)
    for _ in range(8):
        y = rng.integers(0, sz)
        img[y, :] += 0.04

    img = _add_grain(img, 0.025, seed)
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


# ═══════════════════════════════════════════════════════════════════
# Style material definitions
# ═══════════════════════════════════════════════════════════════════

def get_style_materials(style: str, wall_color=(0.55, 0.48, 0.40), seed=0):
    """
    Return dict of PBR materials for each building part.
    Each entry: {"texture": PIL.Image, "roughness": float, "metallic": float}
    """
    s = style.lower()
    wc = wall_color

    if any(k in s for k in ("medieval", "fantasy", "horror")):
        # Neutral grey-white stone, no pink/warm tint
        stone_base = (0.72, 0.70, 0.68)
        stone_mortar = (0.32, 0.30, 0.28)
        return {
            "wall":    {"texture": tex_stone_wall(stone_base, stone_mortar,
                                                   variation=0.025, seed=seed),
                        "roughness": 0.9, "metallic": 0.0},
            "roof":    {"texture": tex_roof_tiles((0.45, 0.40, 0.35), seed=seed+1),
                        "roughness": 0.85, "metallic": 0.0},
            "door":    {"texture": tex_wood((0.35, 0.22, 0.12), seed=seed+2),
                        "roughness": 0.8, "metallic": 0.0},
            "window":  {"texture": tex_glass(seed=seed+3),
                        "roughness": 0.1, "metallic": 0.1},
            "ground":  {"texture": tex_concrete((0.30, 0.28, 0.25), seed=seed+4),
                        "roughness": 0.95, "metallic": 0.0},
        }

    elif any(k in s for k in ("modern", "industrial")):
        is_industrial = "industrial" in s
        # Modern: warm-grey concrete with subtle variation
        mod_wall = (0.82, 0.82, 0.83) if not is_industrial else wc
        return {
            "wall":    {"texture": tex_concrete(mod_wall, seed=seed) if not is_industrial
                                   else tex_rusty_metal(wc, seed=seed),
                        "roughness": 0.3 if not is_industrial else 0.7,
                        "metallic": 0.1 if not is_industrial else 0.8},
            "roof":    {"texture": tex_concrete(tuple(c * 0.85 for c in mod_wall), seed=seed+1),
                        "roughness": 0.4, "metallic": 0.1},
            "door":    {"texture": tex_rusty_metal((0.38, 0.35, 0.32), seed=seed+2),
                        "roughness": 0.5, "metallic": 0.6},
            "window":  {"texture": tex_glass((0.60, 0.75, 0.85), seed=seed+3),
                        "roughness": 0.05, "metallic": 0.2},
            "ground":  {"texture": tex_concrete((0.70, 0.70, 0.70), seed=seed+4),
                        "roughness": 0.5, "metallic": 0.0},
        }

    elif "japanese" in s:
        return {
            "wall":    {"texture": tex_wood(wc, seed=seed),
                        "roughness": 0.8, "metallic": 0.0},
            "roof":    {"texture": tex_roof_tiles(tuple(c * 0.6 for c in wc), seed=seed+1),
                        "roughness": 0.75, "metallic": 0.0},
            "door":    {"texture": tex_wood((0.40, 0.28, 0.15), seed=seed+2),
                        "roughness": 0.7, "metallic": 0.0},
            "window":  {"texture": tex_glass((0.70, 0.78, 0.75), seed=seed+3),
                        "roughness": 0.15, "metallic": 0.0},
            "ground":  {"texture": tex_stone_wall((0.45, 0.45, 0.40), (0.38, 0.38, 0.34),
                                                   brick_h=40, brick_w=60, seed=seed+4),
                        "roughness": 0.85, "metallic": 0.0},
        }

    elif "desert" in s:
        return {
            "wall":    {"texture": tex_concrete(wc, seed=seed),
                        "roughness": 0.9, "metallic": 0.0},
            "roof":    {"texture": tex_concrete(tuple(c * 0.8 for c in wc), seed=seed+1),
                        "roughness": 0.9, "metallic": 0.0},
            "door":    {"texture": tex_wood((0.50, 0.35, 0.20), seed=seed+2),
                        "roughness": 0.85, "metallic": 0.0},
            "window":  {"texture": tex_glass((0.55, 0.68, 0.78), seed=seed+3),
                        "roughness": 0.1, "metallic": 0.05},
            "ground":  {"texture": tex_concrete((0.76, 0.65, 0.40), seed=seed+4),
                        "roughness": 0.95, "metallic": 0.0},
        }

    # Default fallback
    return {
        "wall":    {"texture": tex_stone_wall(wc, seed=seed),
                    "roughness": 0.8, "metallic": 0.0},
        "roof":    {"texture": tex_roof_tiles(tuple(c * 0.7 for c in wc), seed=seed+1),
                    "roughness": 0.8, "metallic": 0.0},
        "door":    {"texture": tex_wood(seed=seed+2),
                    "roughness": 0.8, "metallic": 0.0},
        "window":  {"texture": tex_glass(seed=seed+3),
                    "roughness": 0.1, "metallic": 0.1},
        "ground":  {"texture": tex_concrete(seed=seed+4),
                    "roughness": 0.9, "metallic": 0.0},
    }


# ═══════════════════════════════════════════════════════════════════
# Apply texture to trimesh
# ═══════════════════════════════════════════════════════════════════

_TEX_REPEAT = 3.0   # texture repeats every 3 meters in world space


def apply_pbr_material(mesh, part_type: str, materials: dict):
    """
    Apply PBR texture with world-space UV projection.
    UV = world_coord / 3.0  (texture tiles every 3m).
    All wall meshes share the same UV space so adjacent strips align seamlessly.
    """
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals

    mat_def = materials.get(part_type, materials.get("wall"))
    tex_img = mat_def["texture"]
    roughness = mat_def["roughness"]
    metallic = mat_def["metallic"]

    verts = mesh.vertices
    extents = verts.max(axis=0) - verts.min(axis=0)
    s = 1.0 / _TEX_REPEAT   # UV per meter

    # Choose projection axes based on mesh orientation
    if extents[1] > max(extents[0], extents[2]) * 0.3:
        # Vertical surface: project wider horizontal axis + Y
        if extents[0] >= extents[2]:
            u = verts[:, 0] * s
            v = verts[:, 1] * s
        else:
            u = verts[:, 2] * s
            v = verts[:, 1] * s
    else:
        # Horizontal surface: project XZ
        u = verts[:, 0] * s
        v = verts[:, 2] * s

    uv = np.column_stack([u, v])

    pbr_mat = PBRMaterial(
        baseColorTexture=tex_img,
        roughnessFactor=roughness,
        metallicFactor=metallic,
    )
    mesh.visual = TextureVisuals(uv=uv, material=pbr_mat)
    return mesh


def apply_materials_to_scene(scene, style: str, wall_color=(0.55, 0.48, 0.40)):
    """
    Apply PBR materials to all meshes in a trimesh.Scene.
    Classifies each mesh by its vertex color to determine part type.
    """
    materials = get_style_materials(style, wall_color)

    for name, geom in scene.geometry.items():
        if not hasattr(geom, 'visual') or not hasattr(geom.visual, 'face_colors'):
            continue

        fc = geom.visual.face_colors[0] if len(geom.faces) > 0 else [128, 128, 128, 255]
        r, g, b = fc[0], fc[1], fc[2]

        # Classify by color + geometry
        # Window glass: blue-tinted, AND thin in one axis (glass panel)
        ext = geom.vertices.max(axis=0) - geom.vertices.min(axis=0)
        dims = sorted(ext)
        is_thin_panel = dims[0] < 0.06  # one axis < 6cm = glass/reveal

        if b > 180 and r < 180 and is_thin_panel and dims[1] < 2.0:
            # Blue + thin + small = actual glass pane (not a reveal)
            part = "window"
        elif r < 90 and g < 70 and b < 50 and ext[1] > 1.5:
            # Very dark brown + tall = door panel (door is at least 1.5m high)
            part = "door"
        else:
            mn = geom.vertices.min(axis=0)
            if ext[1] < 0.35 and max(ext[0], ext[2]) > 10 and mn[1] < 0.5:
                # Large flat slab AND near ground level (Y < 0.5m) = ground/road
                part = "ground"
            elif ext[1] < 0.4 and ext[0] * ext[2] > 30:
                # Large flat area = roof slab
                part = "roof"
            else:
                # Everything else: walls, reveals, coping, battlements, columns
                part = "wall"

        try:
            apply_pbr_material(geom, part, materials)
        except Exception:
            pass  # Keep vertex colors if texture application fails

    return scene


# ═══════════════════════════════════════════════════════════════════
# CLI test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Generate texture atlas preview
    for name, gen_fn in [
        ("stone_wall", lambda: tex_stone_wall()),
        ("concrete", lambda: tex_concrete()),
        ("wood", lambda: tex_wood()),
        ("roof_tiles", lambda: tex_roof_tiles()),
        ("glass", lambda: tex_glass()),
        ("rusty_metal", lambda: tex_rusty_metal()),
    ]:
        img = gen_fn()
        img.save(f"_tex_{name}.png")
        print(f"  {name}: {img.size}")

    print("\nTexture previews saved as _tex_*.png")
