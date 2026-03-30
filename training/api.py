"""
LevelSmith Web Agent — FastAPI backend.

Endpoints:
  POST /generate  — prompt → GLB + FBX
  GET  /styles    — list available styles
  GET  /download/{filename} — serve generated files
  GET  /           — serve index.html

Run:
  cd training && uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import sys
import time
import uuid
import hashlib
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── paths ──
SCRIPT_DIR   = Path(__file__).parent
PARAMS_JSON  = SCRIPT_DIR / "trained_style_params.json"
OUTPUT_DIR   = SCRIPT_DIR / "generated"
INDEX_HTML   = SCRIPT_DIR / "index.html"

OUTPUT_DIR.mkdir(exist_ok=True)

# ── add training dir to path for imports ──
sys.path.insert(0, str(SCRIPT_DIR))

app = FastAPI(title="LevelSmith", version="1.0")


# ═══════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════

class GenerateRequest(BaseModel):
    prompt: str = "medieval village, 10 buildings, street layout"
    style: Optional[str] = None
    count: Optional[int] = None
    layout: Optional[str] = None
    min_gap: Optional[float] = None
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    glb_url: str
    fbx_url: str
    stats: dict
    parsed: dict


# ═══════════════════════════════════════════════════════════════════
# Prompt parsing with Claude API
# ═══════════════════════════════════════════════════════════════════

AVAILABLE_STYLES = []
AVAILABLE_LAYOUTS = ["grid", "street", "plaza", "random", "organic"]

def _load_styles():
    global AVAILABLE_STYLES
    if PARAMS_JSON.exists():
        data = json.loads(PARAMS_JSON.read_text("utf-8"))
        AVAILABLE_STYLES = list(data.get("styles", {}).keys())
    return AVAILABLE_STYLES

_load_styles()

PARSE_SYSTEM = f"""You extract level generation parameters from a user prompt.
Available styles: {', '.join(AVAILABLE_STYLES)}
Available layouts: {', '.join(AVAILABLE_LAYOUTS)}

Reply ONLY with a JSON object (no markdown, no explanation):
{{"style": "medieval", "count": 10, "layout": "street", "min_gap": 3.0}}

Rules:
- style: pick the closest match from available styles. Default "medieval".
- count: integer 3-30. Default 10.
- layout: pick from available layouts. "organic" for natural villages, "street" for towns, "grid" for modern, "plaza" for centered. Default "street".
- min_gap: meters between buildings. 1.5-8.0. Compact=1.5, normal=3.0, spacious=6.0. Default 3.0.
"""


def parse_prompt_with_llm(prompt: str) -> dict:
    """Use DeepSeek API (OpenAI-compatible) to parse prompt into generation params."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("[No DEEPSEEK_API_KEY] using regex fallback")
        return parse_prompt_fallback(prompt)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=200,
            temperature=0,
            messages=[
                {"role": "system", "content": PARSE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        print(f"[DeepSeek parse failed: {e}] using regex fallback")
        return parse_prompt_fallback(prompt)


def parse_prompt_fallback(prompt: str) -> dict:
    """Simple regex fallback when Claude API is unavailable."""
    import re
    p = prompt.lower()

    # Style
    style = "medieval"
    for s in AVAILABLE_STYLES:
        if s.replace("_", " ") in p or s in p:
            style = s
            break

    # Count
    m = re.search(r'(\d+)\s*(?:buildings?|houses?|structures?)', p)
    count = int(m.group(1)) if m else 10
    count = max(3, min(30, count))

    # Layout
    layout = "street"
    for l in AVAILABLE_LAYOUTS:
        if l in p:
            layout = l
            break

    # Gap
    min_gap = 3.0
    if any(w in p for w in ["compact", "dense", "tight", "close"]):
        min_gap = 1.5
    elif any(w in p for w in ["spacious", "spread", "wide"]):
        min_gap = 6.0

    return {"style": style, "count": count, "layout": layout, "min_gap": min_gap}


# ═══════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════

def run_generation(style: str, count: int, layout: str,
                   min_gap: float, seed: int) -> dict:
    """Run level_layout.generate_level and export GLB + FBX."""
    import level_layout
    import glb_to_fbx

    # Unique filename based on params
    param_str = f"{style}_{count}_{layout}_{min_gap}_{seed}"
    file_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    base_name = f"{style}_{layout}_{count}_{file_hash}"
    glb_path  = OUTPUT_DIR / f"{base_name}.glb"
    fbx_path  = OUTPUT_DIR / f"{base_name}.fbx"

    t0 = time.time()

    # Generate scene
    scene = level_layout.generate_level(
        style=style,
        layout_type=layout,
        building_count=count,
        seed=seed,
        min_gap=min_gap,
    )

    # Export GLB
    scene.export(str(glb_path))

    # Export FBX
    glb_to_fbx.convert(str(glb_path), str(fbx_path))

    elapsed = time.time() - t0

    total_faces = sum(len(g.faces) for g in scene.geometry.values())
    total_verts = sum(len(g.vertices) for g in scene.geometry.values())
    total_meshes = len(scene.geometry)

    return {
        "glb_url": f"/download/{glb_path.name}",
        "fbx_url": f"/download/{fbx_path.name}",
        "stats": {
            "faces": total_faces,
            "vertices": total_verts,
            "meshes": total_meshes,
            "glb_kb": round(glb_path.stat().st_size / 1024, 1),
            "fbx_kb": round(fbx_path.stat().st_size / 1024, 1),
            "time_s": round(elapsed, 2),
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if INDEX_HTML.exists():
        return HTMLResponse(INDEX_HTML.read_text("utf-8"))
    return HTMLResponse("<h1>LevelSmith</h1><p>index.html not found</p>")


@app.get("/styles")
async def get_styles():
    styles = _load_styles()
    return {"styles": styles, "layouts": AVAILABLE_LAYOUTS}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    # 1. Always parse prompt first (baseline)
    parsed = parse_prompt_with_llm(req.prompt)

    # 2. UI overrides: manual values take priority over prompt parsing
    if req.style:
        parsed["style"] = req.style
    if req.count is not None:
        parsed["count"] = req.count
    if req.layout:
        parsed["layout"] = req.layout
    if req.min_gap is not None:
        parsed["min_gap"] = req.min_gap

    style   = parsed.get("style", "medieval")
    count   = parsed.get("count", 10)
    layout  = parsed.get("layout", "street")
    min_gap = parsed.get("min_gap", 3.0)
    seed    = req.seed or int(time.time()) % 100000

    # Validate
    if style not in AVAILABLE_STYLES:
        raise HTTPException(400, f"Unknown style '{style}'. Available: {AVAILABLE_STYLES}")
    if layout not in AVAILABLE_LAYOUTS:
        raise HTTPException(400, f"Unknown layout '{layout}'. Available: {AVAILABLE_LAYOUTS}")
    count = max(3, min(30, count))
    min_gap = max(1.0, min(10.0, min_gap))

    # Generate
    result = run_generation(style, count, layout, min_gap, seed)
    result["parsed"] = {**parsed, "seed": seed}
    return result


@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "File not found")
    media = "model/gltf-binary" if filename.endswith(".glb") else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=filename)
