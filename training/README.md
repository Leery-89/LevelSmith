# LevelSmith — Training Module

See the [project root README](../README.md) for full documentation.

This directory contains the core Python code, trained models, and agent design documents.

---

## Directory overview

| File | Purpose |
|------|---------|
| `api.py` | FastAPI server, DeepSeek LLM integration, web UI serving |
| `index.html` | Web interface (Three.js 3D preview, prompt input, sidebar) |
| `model.py` | `StyleParamMLP` neural network (16 → 128 → 64 → 32 → 23) |
| `style_registry.py` | 20 styles × 23 parameters, base values + variation ranges |
| `generate_level.py` | Single building geometry engine (CSG booleans: walls, doors, windows, roofs) |
| `generate_data.py` | Synthetic training data generation (131K samples) |
| `level_layout.py` | Multi-building layout engine, road-lot system, 5 layout algorithms |
| `layout_model.py` | Autoregressive layout Transformer |
| `text_encoder.py` | Text → 16-dim feature vector (sentence-transformers) |
| `train.py` | Model training script |
| `inference.py` | Model loading + inference utilities |
| `hardware_config.py` | GPU / NPU / CPU device routing |
| `glb_to_fbx.py` | GLB → UE5-compatible FBX converter |
| `procedural_materials.py` | Procedural material / texture generation |
| `compound_layout.py` | Compound layout from layout graphs (e.g. Kaer Morhen) |

### Data directories (not in git)

| Directory | Contents |
|-----------|----------|
| `data/` | Training datasets (synthetic + OSM + Witcher 3) |
| `cache/` | Cached model weights and intermediate files |
| `VGLC/` | Video Game Level Corpus data |

### Design documents

| Doc | Status |
|-----|--------|
| `docs/archetype_planning_agent.md` | Live — drives building roster and spatial rules |
| `docs/style_material_director.md` | Designed — pending UE5 material pipeline |
| `docs/wall_interior_agent_design.md` | Designed — pending Kaer Morhen prototype |
| `docs/model_training_roadmap.md` | Planning — v2 model architecture |
| `docs/training_data_sources.md` | Reference — data lineage |

---

## Running locally

### 1. Install dependencies

```bash
cd training
pip install -r requirements.txt
```

Core dependencies: `torch`, `numpy`, `trimesh`, `shapely`, `scipy`, `manifold3d`, `Pillow`, `fastapi`, `uvicorn`, `python-dotenv`.

Optional: `openai` (for DeepSeek API), `sentence-transformers` (for text encoder), `onnxruntime-directml` (for NPU inference).

### 2. Configure environment

```bash
# Copy and edit .env (optional — system works without it)
echo "DEEPSEEK_API_KEY=sk-your-key-here" > .env
```

### 3. Start the web server

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000`. The web UI provides:
- Text prompt input
- Style and layout dropdowns
- 3D preview (Three.js)
- Archetype plan sidebar display
- GLB/FBX download

### 4. Command-line generation

```python
from level_layout import generate_level

scene = generate_level(
    style="medieval_keep",
    layout_type="organic",
    building_count=10,
    area_size=100.0,
    seed=42
)
scene.export("my_level.glb")
```

### 5. Training the style model

```bash
python train.py
```

This trains the `StyleParamMLP` on synthetic data (131K samples, 20 styles). Output: `best_model.pt` and `train_history.json`.

---

## Debugging tips

- **Debug mode**: Set `DEBUG=1` in `.env` to enable verbose logging of agent plans and fallback reasons.
- **Intermediate artifacts**: The `/generate` API endpoint returns `archetype_plan` in its JSON response — inspect it to see the LLM's building roster and spatial rules.
- **Seed control**: Use `seed` parameter for reproducible generation. Same seed + same parameters = identical output.
- **Without API key**: The system falls back to regex parsing. Look for `[FALLBACK]` in console output to confirm.
- **Common issues**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
