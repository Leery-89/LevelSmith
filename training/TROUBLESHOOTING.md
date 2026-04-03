# Troubleshooting

Common issues and solutions when running LevelSmith.

---

## Installation issues

### `ModuleNotFoundError: No module named 'trimesh'`

Install dependencies from the requirements file:

```bash
cd training
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'manifold3d'`

manifold3d requires a C++ compiler on some platforms. On Windows, install Visual Studio Build Tools first. If installation fails, trimesh will still work but some boolean operations may be slower.

```bash
pip install manifold3d
```

### PyTorch installs CPU-only version

The default `pip install torch` may install CPU-only PyTorch. For GPU acceleration:

```bash
# Visit https://pytorch.org/get-started/locally/ for your exact command
# Example for CUDA 12.1 on Windows:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### `ImportError: cannot import name 'SentenceTransformer'`

sentence-transformers is optional (used by the text encoder for ML layout model). Install it if needed:

```bash
pip install sentence-transformers
```

---

## Runtime issues

### `archetype_plan` is null / archetype agent not executing

The archetype agent requires a DeepSeek API key. Create a `.env` file in `training/`:

```
DEEPSEEK_API_KEY=sk-your-key-here
```

Without an API key, the system falls back to regex keyword matching. All geometry features still work, but building role assignment and spatial rules are bypassed. Look for `[FALLBACK]` in console output to confirm fallback mode.

### Server won't start: `Address already in use`

Another process is using port 8000. Either kill it or use a different port:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8001
```

### `shapely` geometry errors

Ensure shapely >= 2.0:

```bash
pip install --upgrade shapely
```

On Windows, if you get DLL errors, try:

```bash
pip install shapely --no-binary shapely
```

### Generation produces empty or very small scenes

- Check that `building_count` is at least 1
- Check that `area_size` is large enough for the requested building count (minimum ~50m for 5 buildings)
- For `street` and `grid` layouts, buildings are constrained to lots — if the area is too small, lots may be too small for buildings

### GLB file won't open in viewer

- Verify the file is non-empty: check the file size (should be > 10KB for any scene)
- Try opening in Blender (File > Import > glTF) — it handles edge cases better than some viewers
- If exported from web UI, check the browser console for errors

---

## Debugging generation

### Enable verbose output

The generation functions print diagnostic info to stdout. Run the server in a terminal (not as a background service) to see:

- Style parameters selected
- Layout type and building count
- Road network statistics
- Gate-road alignment distances
- Fallback decisions

### Reproduce a specific result

Use the `seed` parameter for deterministic generation:

```python
scene = generate_level(style="medieval", layout_type="organic",
                       building_count=10, seed=42)
```

Same seed + same parameters = identical output.

### Inspect the archetype plan

The `/generate` API endpoint returns `archetype_plan` in its JSON response. You can inspect it to see:
- Building roster (roles, types, sizes)
- Spatial relationships (adjacency, clustering)
- Enclosure configuration (walled / partial / open)

### Check intermediate artifacts

When debugging layout issues, generate with different seeds and compare:

```python
for seed in [42, 123, 777]:
    scene = generate_level(style="medieval", layout_type="organic",
                           building_count=10, seed=seed)
    scene.export(f"debug_seed_{seed}.glb")
```

---

## Performance

### Generation is slow

- Single building generation: < 1 second (CPU)
- Full scene (10 buildings + roads): 2-5 seconds (CPU)
- Full scene with enclosure walls: 3-8 seconds (CPU)
- If significantly slower, check that trimesh is using manifold3d for CSG operations

### Model inference is slow

- Default device routing: CUDA > DirectML (NPU) > CPU
- Force CPU if GPU causes issues: set `FORCE_CPU=1` in environment
- Model is small (13K params) — CPU inference is < 10ms
