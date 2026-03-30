# Levelsmith FBX -> UE5 Setup Guide

## Problem: Imported mesh shows grey/white, no vertex colors

The FBX file contains vertex colors (verified: `LayerElementColor` with `ByPolygonVertex` mapping).
UE5's **default material ignores vertex colors** -- you need to create a custom material.

---

## Step 1: Import Settings

When importing the FBX in UE5:

1. **Mesh** section:
   - Vertex Color Import Option: **Replace** (not Ignore!)
   - Vertex Override Color: leave as default

2. **Transform** section:
   - Import Uniform Scale: **1.0** (coordinates already in cm)
   - Convert Scene: **OFF**
   - Convert Scene Unit: **OFF**

3. **Miscellaneous**:
   - Generate Lightmap UVs: ON (recommended)

---

## Step 2: Create Vertex Color Material

### 2a. Simple Material (vertex color only)

1. Content Browser -> Right-click -> **Material** -> name it `M_VertexColor`
2. Double-click to open Material Editor
3. Create one node: **Vertex Color** (right-click canvas -> search "Vertex Color")
4. Connect:
   ```
   [Vertex Color] RGB  -->  [M_VertexColor] Base Color
   [Vertex Color] A    -->  [M_VertexColor] Opacity  (only if using translucent)
   ```
5. In the Details panel (left side):
   - Blend Mode: **Opaque** (for solid walls/floors)
   - Shading Model: **Default Lit**
6. Click **Apply** and **Save**

### 2b. Enhanced Material (vertex color + basic lighting)

For better visual quality:

```
Material Graph:

  [Vertex Color] RGB ---- [Multiply] ---- Base Color
                           |
  [Constant: 1.1] --------+              (slight brightness boost)

  [Constant: 0.3] ---- Roughness
  [Constant: 0.0] ---- Metallic
  [Constant: 0.0] ---- Specular
```

### 2c. Material with Translucency Support (for windows)

Some meshes (windows) have alpha < 1.0:

1. Create `M_VertexColor_Translucent`
2. Blend Mode: **Translucent**
3. Lighting Mode: **Surface TranslucencyVolume**
4. Connect:
   ```
   [Vertex Color] RGB  -->  Base Color
   [Vertex Color] A    -->  Opacity
   ```

---

## Step 3: Apply Material to All Meshes

### Option A: On Import (easiest)

1. In Import dialog, under **Material**:
   - Search for Material: `M_VertexColor`
   - Material Import Method: **Do Not Create Material**
   - Set all material slots to `M_VertexColor`

### Option B: After Import (batch)

1. Select all imported Static Meshes in Content Browser
2. Right-click -> **Asset Actions** -> **Property Matrix**
3. In the Property Matrix, find **Static Materials** column
4. Set Material to `M_VertexColor` for all rows

### Option C: Blueprint Batch Script

In a Level Blueprint or Editor Utility Widget:

```cpp
// C++ or Blueprint pseudo-code
TArray<UStaticMeshComponent*> Components;
GetAllComponentsOfClass(Components);
UMaterialInterface* VCMat = LoadObject<UMaterialInterface>(
    nullptr, TEXT("/Game/Materials/M_VertexColor.M_VertexColor"));
for (auto* Comp : Components) {
    for (int i = 0; i < Comp->GetNumMaterials(); i++) {
        Comp->SetMaterial(i, VCMat);
    }
}
```

### Option D: Python Editor Script (recommended for batch)

In UE5 Python console (Edit -> Editor Preferences -> Enable Python):

```python
import unreal

mat_path = '/Game/Materials/M_VertexColor.M_VertexColor'
mat = unreal.EditorAssetLibrary.load_asset(mat_path)

# Get all static meshes in a folder
assets = unreal.EditorAssetLibrary.list_assets('/Game/Levels/', recursive=True)
for asset_path in assets:
    asset = unreal.EditorAssetLibrary.load_asset(asset_path)
    if isinstance(asset, unreal.StaticMesh):
        # Set material for all sections
        for i in range(asset.get_num_sections(0)):
            asset.set_material(i, mat)
        unreal.EditorAssetLibrary.save_asset(asset_path)
        print(f'Applied vertex color material to: {asset_path}')
```

---

## Verification

After applying the material, the mesh should display colors like:
- **Walls**: brown/grey tones (R:0.59, G:0.56, B:0.49)
- **Floors**: darker brown
- **Windows**: blue-tinted with transparency
- **Doors**: dark brown
- **Ground pad**: dark earth tone

If still grey: check that UE5 **Vertex Color Import Option** was set to **Replace** on import.

---

## Color Reference (from Levelsmith palettes)

| Element  | R    | G    | B    | A    |
|----------|------|------|------|------|
| Wall     | 0.64 | 0.57 | 0.48 | 1.0  |
| Floor    | 0.45 | 0.37 | 0.29 | 1.0  |
| Ceiling  | 0.53 | 0.45 | 0.35 | 1.0  |
| Door     | 0.35 | 0.23 | 0.11 | 1.0  |
| Window   | 0.61 | 0.76 | 0.84 | 0.71 |
| Ground   | 0.35 | 0.31 | 0.25 | 1.0  |
