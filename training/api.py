"""
LevelSmith Web Agent — FastAPI backend.

Endpoints:
  POST /generate  — prompt → GLB + FBX (supports multi-zone semantic layout)
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
import hashlib
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# ── paths ──
SCRIPT_DIR   = Path(__file__).parent
PARAMS_JSON  = SCRIPT_DIR / "trained_style_params.json"
OUTPUT_DIR   = SCRIPT_DIR / "generated"
INDEX_HTML   = SCRIPT_DIR / "index.html"

OUTPUT_DIR.mkdir(exist_ok=True)

# ── add training dir to path for imports ──
sys.path.insert(0, str(SCRIPT_DIR))

app = FastAPI(title="LevelSmith", version="2.0")


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
# Prompt parsing
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
{{
  "zones": [
    {{"name": "market", "style": "medieval", "layout": "plaza", "count": 8, "density": "normal"}},
    {{"name": "slums", "style": "medieval_chapel", "layout": "organic", "count": 12, "density": "dense"}}
  ],
  "min_gap": 3.0
}}

Rules:
- zones: 1-4 semantic zones. Single-theme prompts produce 1 zone.
- density: "dense"=1.5m gap, "normal"=3.0m, "sparse"=6.0m
- count per zone: 3-15, total across all zones <= 30
- style/layout: pick closest match from available lists
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
            max_tokens=400,
            temperature=0,
            messages=[
                {"role": "system", "content": PARSE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(text)
        # Ensure zones format
        if "zones" not in result and "style" in result:
            result = {
                "zones": [{"name": "main", "style": result["style"],
                           "layout": result.get("layout", "street"),
                           "count": result.get("count", 10),
                           "density": "normal"}],
                "min_gap": result.get("min_gap", 3.0),
            }
        return result
    except Exception as e:
        print(f"[DeepSeek parse failed: {e}] using regex fallback")
        return parse_prompt_fallback(prompt)


def parse_prompt_fallback(prompt: str) -> dict:
    """Regex fallback with multi-zone keyword detection."""
    import re
    p = prompt.lower()

    ZONE_KEYWORDS = {
        "market|bazaar|market place": {"style": "medieval", "layout": "plaza", "density": "normal"},
        "slum|slums|poor|poverty": {"style": "medieval_chapel", "layout": "organic", "density": "dense"},
        "church|cathedral|chapel|temple|religious": {"style": "medieval_chapel", "layout": "plaza", "density": "sparse"},
        "castle|keep|fortress|fort": {"style": "medieval_keep", "layout": "grid", "density": "sparse"},
        "palace|royal": {"style": "fantasy_palace", "layout": "plaza", "density": "sparse"},
        "dungeon|underground": {"style": "fantasy_dungeon", "layout": "grid", "density": "dense"},
        "workshop|forge|industrial": {"style": "industrial_workshop", "layout": "grid", "density": "normal"},
        "port|dock|harbor": {"style": "industrial", "layout": "street", "density": "normal"},
        "residential|house|homes": {"style": "medieval", "layout": "street", "density": "normal"},
        "asylum|horror|abandoned": {"style": "horror_asylum", "layout": "random", "density": "sparse"},
    }

    # Total building count
    m = re.search(r'(\d+)\s*(?:buildings?|houses?|structures?)', p)
    total_count = int(m.group(1)) if m else 20

    # Match zones
    matched_zones = []
    for pattern, zone_cfg in ZONE_KEYWORDS.items():
        if re.search(pattern, p):
            matched_zones.append((pattern, zone_cfg))

    # No multi-zone match → single zone fallback
    if len(matched_zones) <= 1:
        style = "medieval"
        for s in AVAILABLE_STYLES:
            if s.replace("_", " ") in p or s in p:
                style = s
                break
        layout = "street"
        for l in AVAILABLE_LAYOUTS:
            if l in p:
                layout = l
                break
        density = "normal"
        if any(w in p for w in ["compact", "dense", "tight"]):
            density = "dense"
        elif any(w in p for w in ["spacious", "spread", "wide"]):
            density = "sparse"
        return {
            "zones": [{"name": "main", "style": style, "layout": layout,
                       "count": max(3, min(30, total_count)), "density": density}],
            "min_gap": 3.0,
        }

    # Multi-zone: distribute buildings evenly
    per_zone = max(3, min(15, total_count // len(matched_zones)))
    zones = []
    for pattern, zone_cfg in matched_zones:
        name = pattern.split("|")[0]
        zones.append({
            "name": name,
            "style": zone_cfg["style"],
            "layout": zone_cfg["layout"],
            "count": per_zone,
            "density": zone_cfg["density"],
        })

    min_gap = 3.0
    if any(w in p for w in ["compact", "dense", "tight"]):
        min_gap = 1.5
    elif any(w in p for w in ["spacious", "spread", "wide"]):
        min_gap = 6.0

    return {"zones": zones, "min_gap": min_gap}


# ═══════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════

DENSITY_MAP = {"dense": 1.5, "normal": 3.0, "sparse": 6.0}


def run_generation(zones: list, min_gap: float, seed: int) -> dict:
    """Run level generation for one or more zones, merge, export GLB + FBX."""
    import level_layout
    import glb_to_fbx

    zone_scenes = []
    for i, zone in enumerate(zones):
        gap = DENSITY_MAP.get(zone.get("density", "normal"), min_gap)
        scene = level_layout.generate_level(
            style=zone["style"],
            layout_type=zone["layout"],
            building_count=zone["count"],
            seed=seed + i,
            min_gap=gap,
        )
        zone_scenes.append(scene)

    if len(zone_scenes) > 1:
        merged = level_layout.merge_zones(zone_scenes)
    else:
        merged = zone_scenes[0]

    zone_hash = hashlib.md5(str(zones).encode()).hexdigest()[:8]
    base_name = f"level_{zone_hash}_{seed}"
    glb_path = OUTPUT_DIR / f"{base_name}.glb"
    fbx_path = OUTPUT_DIR / f"{base_name}.fbx"

    t0 = time.time()
    merged.export(str(glb_path))
    glb_to_fbx.convert(str(glb_path), str(fbx_path))
    elapsed = time.time() - t0

    total_faces = sum(len(g.faces) for g in merged.geometry.values())
    total_verts = sum(len(g.vertices) for g in merged.geometry.values())

    return {
        "glb_url": f"/download/{glb_path.name}",
        "fbx_url": f"/download/{fbx_path.name}",
        "stats": {
            "faces": total_faces,
            "vertices": total_verts,
            "meshes": len(merged.geometry),
            "zones": len(zones),
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
    # 1. Parse prompt
    parsed = parse_prompt_with_llm(req.prompt)

    zones   = parsed.get("zones", [{"name": "main", "style": "medieval",
                                     "layout": "street", "count": 10,
                                     "density": "normal"}])
    min_gap = parsed.get("min_gap", 3.0)

    # 2. UI overrides (only for single-zone)
    if req.style and len(zones) == 1:
        zones[0]["style"] = req.style
    if req.count is not None and len(zones) == 1:
        zones[0]["count"] = req.count
    if req.layout and len(zones) == 1:
        zones[0]["layout"] = req.layout
    if req.min_gap is not None:
        min_gap = req.min_gap

    # 3. Validate
    for zone in zones:
        if zone["style"] not in AVAILABLE_STYLES:
            zone["style"] = "medieval"
        if zone["layout"] not in AVAILABLE_LAYOUTS:
            zone["layout"] = "street"
        zone["count"] = max(3, min(15, zone.get("count", 10)))

    seed = req.seed or int(time.time()) % 100000

    # 4. Generate
    result = run_generation(zones, min_gap, seed)
    result["parsed"] = {"zones": zones, "min_gap": min_gap, "seed": seed}
    return result


@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "File not found")
    media = "model/gltf-binary" if filename.endswith(".glb") else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=filename)
