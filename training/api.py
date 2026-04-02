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

from dotenv import load_dotenv
load_dotenv()

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
    clusters: list = []
    road_graph: dict = {}
    building_infos: list = []
    archetype_plan: Optional[dict] = None


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

def _load_archetype_prompt() -> str:
    """Load archetype planning agent system prompt from docs, stripping markdown title."""
    doc_path = SCRIPT_DIR / "docs" / "archetype_planning_agent.md"
    if not doc_path.exists():
        return ""
    text = doc_path.read_text("utf-8")
    # Strip the first markdown title line
    lines = text.split("\n")
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()

ARCHETYPE_SYSTEM = _load_archetype_prompt()

# Legacy system prompt (used when archetype doc is missing)
PARSE_SYSTEM_LEGACY = f"""You extract level generation parameters from a user prompt.
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

# ─── Layout type mapping ─────────────────────────────────────────

_ROAD_TO_LAYOUT = {
    "axial": "street", "radial": "organic", "ring": "plaza",
    "organic": "organic", "grid": "grid", "minimal": "random",
}

_DENSITY_TO_GAP = {
    "crowded": "dense", "moderate": "normal",
    "sparse": "sparse", "desolate": "sparse",
}

_STYLE_DEFAULT_LAYOUT = {
    "medieval": "organic", "medieval_keep": "organic", "medieval_chapel": "organic",
    "fantasy": "organic", "fantasy_dungeon": "random", "fantasy_palace": "plaza",
    "horror": "random", "horror_asylum": "random", "horror_crypt": "random",
    "japanese": "organic", "japanese_temple": "organic", "japanese_machiya": "street",
    "desert": "plaza", "desert_palace": "plaza",
    "industrial": "grid", "industrial_workshop": "grid", "industrial_powerplant": "grid",
    "modern": "grid", "modern_loft": "street", "modern_villa": "organic",
}


def plan_to_params(plan: dict) -> tuple[list, float, dict]:
    """
    Translate archetype JSON plan → (zone_list, min_gap, raw_plan).

    The zone_list follows the existing format consumed by run_generation().
    Extra fields (building_roles, enclosure, spatial_rules) are preserved
    in the raw plan for downstream use when level_layout supports them.
    """
    # ── layout type ──
    road_pref = plan.get("road_preference", {})
    lt_raw = road_pref.get("layout_type", "organic")
    layout = _ROAD_TO_LAYOUT.get(lt_raw, "organic")

    # ── density / gap ──
    atmo = plan.get("atmosphere", {})
    density_feel = atmo.get("density_feel", "moderate")
    density = _DENSITY_TO_GAP.get(density_feel, "normal")

    # ── style ──
    primary_style = plan.get("primary_style", "medieval")
    if primary_style not in AVAILABLE_STYLES:
        primary_style = "medieval"

    # ── count (exclude ambient) ──
    buildings = plan.get("buildings", [])
    count = plan.get("total_building_count", 10)
    if not count or count < 3:
        count = sum(b.get("count", 1) for b in buildings
                    if b.get("role") != "ambient")
    count = max(3, min(30, count))

    # ── build zone ──
    zone = {
        "name": plan.get("archetype", "settlement"),
        "style": primary_style,
        "layout": layout,
        "count": count,
        "density": density,
    }

    # ── gap ──
    gap_map = {"dense": 1.5, "normal": 3.0, "sparse": 6.0}
    min_gap = gap_map.get(density, 3.0)

    return [zone], min_gap, plan


def parse_prompt_with_llm(prompt: str) -> dict:
    """
    Use DeepSeek API to parse prompt via Archetype Planning Agent.

    Flow:
      1. DeepSeek + archetype prompt → JSON plan → plan_to_params()
      2. Failure → parse_prompt_fallback() (regex)

    Returns dict with keys: zones, min_gap, archetype_plan (optional).
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("[No DEEPSEEK_API_KEY] using regex fallback")
        return parse_prompt_fallback(prompt)

    # Choose system prompt: archetype if available, else legacy
    system_prompt = ARCHETYPE_SYSTEM if ARCHETYPE_SYSTEM else PARSE_SYSTEM_LEGACY
    use_archetype = bool(ARCHETYPE_SYSTEM)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=4000 if use_archetype else 400,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        raw = json.loads(text)

        if use_archetype and "archetype" in raw:
            # Archetype flow: translate plan → zones
            zones, min_gap, plan = plan_to_params(raw)
            return {"zones": zones, "min_gap": min_gap, "archetype_plan": plan}
        else:
            # Legacy flow or archetype returned legacy format
            if "zones" not in raw and "style" in raw:
                raw = {
                    "zones": [{"name": "main", "style": raw["style"],
                               "layout": raw.get("layout", "street"),
                               "count": raw.get("count", 10),
                               "density": "normal"}],
                    "min_gap": raw.get("min_gap", 3.0),
                }
            return raw

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
        layout = _STYLE_DEFAULT_LAYOUT.get(style, "organic")
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


def run_generation(zones: list, min_gap: float, seed: int,
                   archetype_plan: dict = None,
                   graph_name: str = None) -> dict:
    """Run level generation for one or more zones, merge, export GLB + FBX."""
    import level_layout
    import glb_to_fbx

    zone_scenes = []
    for i, zone in enumerate(zones):
        gap = DENSITY_MAP.get(zone.get("density", "normal"), min_gap)
        extra_kw = {}
        if graph_name:
            extra_kw["graph_name"] = graph_name
        elif archetype_plan:
            extra_kw["building_roles"] = archetype_plan.get("buildings")
            extra_kw["spatial_rules"] = archetype_plan.get("spatial_rules")
            extra_kw["enclosure_config"] = archetype_plan.get("enclosure")
        scene = level_layout.generate_level(
            style=zone["style"],
            layout_type=zone["layout"],
            building_count=zone["count"],
            seed=seed + i,
            min_gap=gap,
            **extra_kw,
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

    # Extract metadata from scene (use last zone's scene for single-zone)
    src_scene = zone_scenes[-1] if zone_scenes else merged
    meta = getattr(src_scene, "metadata", None) or {}

    raw_clusters = meta.get("clusters", [])
    raw_binfos = meta.get("building_infos", [])

    clusters_out = []
    for cl in raw_clusters:
        clusters_out.append({
            "id": cl.get("id", 0),
            "building_indices": cl.get("building_indices", []),
            "main_building_idx": cl.get("main_building_idx", 0),
            "center": cl.get("center", [0, 0]),
            "size": len(cl.get("building_indices", [])),
        })

    binfos_out = []
    for b in raw_binfos:
        binfos_out.append({
            "idx": b.get("idx", 0),
            "x": round(b.get("x", 0), 2),
            "z": round(b.get("z", 0), 2),
            "yaw_deg": round(b.get("yaw_deg", 0), 1),
            "width": round(b.get("w", 0), 2),
            "depth": round(b.get("d", 0), 2),
            "cluster_id": b.get("cluster_id", -1),
            "is_main": b.get("is_main_building", False),
            "role": b.get("role", ""),
            "style_key": b.get("style_key", ""),
        })

    road_out = {
        "nodes": meta.get("road_nodes", []),
        "edges": meta.get("road_edges", []),
        "renderable": meta.get("road_renderable", True),
    }

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
        "clusters": clusters_out,
        "road_graph": road_out,
        "building_infos": binfos_out,
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
            zone["layout"] = _STYLE_DEFAULT_LAYOUT.get(zone["style"], "organic")
        zone["count"] = max(3, min(15, zone.get("count", 10)))

    seed = req.seed or int(time.time()) % 100000

    # 4. Detect layout graph mode
    import re
    _GRAPH_KEYWORDS = {
        r"kaer\s*morhen|凯尔莫汉": "kaer_morhen",
    }
    graph_name = None
    prompt_lower = req.prompt.lower()
    for pattern, gname in _GRAPH_KEYWORDS.items():
        if re.search(pattern, prompt_lower):
            graph_name = gname
            break

    # 5. Generate
    arch_plan = parsed.get("archetype_plan")
    result = run_generation(zones, min_gap, seed, archetype_plan=arch_plan,
                            graph_name=graph_name)
    result["parsed"] = {"zones": zones, "min_gap": min_gap, "seed": seed}
    result["archetype_plan"] = arch_plan
    return result


@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "File not found")
    media = "model/gltf-binary" if filename.endswith(".glb") else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=filename)
