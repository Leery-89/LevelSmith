# LevelSmith

**AI-powered procedural level generator for game development.**

Generate complete 3D building layouts from natural language prompts. Export as GLB/FBX for Unreal Engine 5, Unity, or Blender.

> **LevelSmith** — ML 驱动的程序化关卡生成器。输入一句话描述，自动生成 3D 建筑群场景。

---

## Features

- **Natural Language Input** — Describe your level in plain text: *"medieval village, 15 buildings, compact layout"*
- **20 Architectural Styles** — From medieval castles to Japanese temples to industrial factories
- **5 Layout Algorithms** — Street, grid, plaza, random, organic (ML-powered)
- **ML Layout Model** — Trained on 22,363 OSM buildings + 11,903 Witcher 3 layouts
- **Style Parameter MLP** — 16-dim feature vector → 23 structural parameters per building
- **Boolean CSG Walls** — Seamless wall geometry via manifold3d boolean operations
- **Web Agent** — FastAPI backend + Three.js real-time preview
- **UE5 Ready** — FBX export with vertex colors, correct coordinate system, fixed normals
- **Hardware Optimized** — CUDA training (RTX 4070 AMP), DirectML inference, ONNX export

## Architecture

```
User Prompt
    |
    v
[DeepSeek API / Regex Parser]  →  {style, count, layout, min_gap}
    |
    v
[Layout Engine]
    |--- street:  Two-row street layout
    |--- grid:    Regular grid
    |--- plaza:   Central plaza ring
    |--- organic: ML model (layout_model_w3.pt) + Bezier road network
    |--- random:  Poisson disk sampling
    |
    v
[Building Generator]  ←  StyleParamMLP (best_model.pt)
    |                     16-dim features → 23 structural params
    |--- Boolean CSG walls (manifold3d)
    |--- Doors, windows, glass panels
    |--- 5 roof types (flat/gabled/hip/pagoda/turret)
    |--- Battlements, columns, arch trims
    |--- Perimeter walls, street lamps
    |
    v
[Export]
    |--- GLB (trimesh) → Three.js preview
    |--- FBX ASCII 7.4 (glb_to_fbx.py) → UE5 import
```

## Supported Styles

| Category | Styles |
|----------|--------|
| **Medieval** | `medieval`, `medieval_keep`, `medieval_chapel` |
| **Modern** | `modern`, `modern_loft`, `modern_villa` |
| **Industrial** | `industrial`, `industrial_workshop`, `industrial_powerplant` |
| **Fantasy** | `fantasy`, `fantasy_dungeon`, `fantasy_palace` |
| **Horror** | `horror`, `horror_asylum`, `horror_crypt` |
| **Japanese** | `japanese`, `japanese_temple`, `japanese_machiya` |
| **Desert** | `desert`, `desert_palace` |

Each style has a 16-dimensional feature vector and 23 output parameters controlling wall thickness, height, roof type, window density, mesh complexity, and more.

## Quick Start

### Installation

```bash
git clone https://github.com/yourname/levelsmith.git
cd levelsmith/training
pip install -r requirements.txt
```

### Generate a Level (CLI)

```bash
# Medieval street with 10 buildings
python level_layout.py --style medieval --layout street --count 10 --out my_level.glb

# Fantasy plaza with 8 buildings
python level_layout.py --style fantasy --layout plaza --count 8 --out fantasy_level.glb

# Organic village with ML model (requires trained model)
python level_layout.py --style medieval --layout organic --count 15 --out village.glb

# Convert to UE5 FBX
python glb_to_fbx.py my_level.glb --out my_level.fbx
```

### Web Agent

```bash
# Optional: set DeepSeek API key for natural language parsing
# Without it, regex fallback is used automatically
export DEEPSEEK_API_KEY=sk-xxx

# Start server
cd training
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Open browser
# http://localhost:8000
```

### Train Models

```bash
# 1. Train style parameter MLP
python train.py

# 2. Train layout model (requires OSM data)
python fetch_osm_layout.py          # Download OSM building data
python layout_model.py --source combined  # Train with OSM + W3 data

# 3. Fine-tune for mesh complexity
python finetune_complexity.py
```

## UE5 Import

1. Import the `.fbx` file with:
   - Vertex Color Import Option: **Replace**
   - Import Uniform Scale: **1.0**
   - Convert Scene: **OFF**

2. Create a material `M_VertexColor`:
   - Add a **Vertex Color** node
   - Connect **RGB** → **Base Color**

3. Apply to all imported meshes

See `ue5_setup.md` for detailed instructions.

## Project Structure

```
training/
├── api.py                  # FastAPI web server
├── index.html              # Three.js frontend
├── generate_level.py       # Building mesh generator (CSG boolean walls)
├── level_layout.py         # Layout algorithms (street/grid/plaza/organic)
├── style_registry.py       # 20 styles, 23 output parameters
├── model.py                # StyleParamMLP (16→23)
├── train.py                # Training pipeline
├── layout_model.py         # Autoregressive layout Transformer
├── text_encoder.py         # Text → style feature vector
├── hardware_config.py      # GPU/NPU/CPU device routing
├── glb_to_fbx.py           # GLB → UE5 FBX converter
├── parse_w2l.py            # Witcher 3 .w2l binary parser
├── inference.py             # Model inference utilities
├── ue5_setup.md            # UE5 material setup guide
├── gen_ml_level.py         # ML layout → GLB/FBX
├── gen_w3_level.py         # W3+OSM model → GLB/FBX
├── gen_previews.py         # Style comparison preview generator
├── finetune_complexity.py  # Fine-tune 20→23 dim
├── requirements.txt        # Python dependencies
└── models/                 # Trained weights (not in git)
    ├── best_model.pt       # StyleParamMLP checkpoint
    ├── layout_model_w3.pt  # Layout Transformer (OSM+W3)
    └── layout_model_w3.onnx # ONNX for DirectML inference
```

## Training Data Sources

| Source | Buildings | Usage |
|--------|-----------|-------|
| **OpenStreetMap** (4 medieval towns) | 22,363 | Layout model training |
| **Witcher 3** (.w2l level files) | 11,903 | Layout model (yaw != 0 filter) |
| **Witcher 3** (.w2mesh files) | 1,856 | Mesh complexity features |

> Note: Training data is not included in the repository due to licensing. Use the provided scripts to collect your own data.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Any CUDA GPU | RTX 4070+ (AMP FP16) |
| RAM | 8 GB | 16+ GB |
| Storage | 2 GB | 10 GB (with training data) |
| Python | 3.10+ | 3.11 |

## License

MIT License

## Acknowledgments

- Building layout data from [OpenStreetMap](https://www.openstreetmap.org/) (ODbL)
- Mesh complexity reference from The Witcher 3 (CD Projekt RED)
- [trimesh](https://trimesh.org/) for 3D geometry
- [manifold3d](https://github.com/elalish/manifold) for CSG boolean operations
- [Three.js](https://threejs.org/) for web 3D preview
