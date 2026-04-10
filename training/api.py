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
    prompt: str = "medieval fortress on a hilltop"
    style: Optional[str] = None
    count: Optional[int] = None
    layout: Optional[str] = None
    min_gap: Optional[float] = None
    seed: Optional[int] = None
    # Reserved for future conversational agent: list of prior turns to give the
    # backend context for delta edits ("make it bigger", "add a chapel", etc.)
    conversation_history: list = []


class GenerateResponse(BaseModel):
    glb_url: str
    json_url: str
    stats: dict
    parsed: dict
    clusters: list = []
    road_graph: dict = {}
    building_infos: list = []
    archetype_plan: Optional[dict] = None
    classification: Optional[dict] = None


class EditRequest(BaseModel):
    instruction: str                           # natural-language edit
    current_scene: dict = {}                   # last /generate response (slim)
    conversation_history: list = []            # prior turns


class EditResponse(GenerateResponse):
    edit_summary: str = ""


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
    """Run level generation for one or more zones, merge, export GLB + JSON."""
    import level_layout

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
    json_path = OUTPUT_DIR / f"{base_name}_placement.json"

    t0 = time.time()
    merged.export(str(glb_path))

    # Export placement JSON for UE5 assembly
    placement = getattr(merged, "metadata", {}).get("placement")
    if placement:
        json_path.write_text(
            json.dumps(placement, indent=2, ensure_ascii=False), encoding="utf-8")
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

    json_kb = round(json_path.stat().st_size / 1024, 1) if json_path.exists() else 0

    # Pull coverage_ratio from placement metadata if available
    placement = meta.get("placement") or {}
    coverage_ratio = (placement.get("metadata") or {}).get("coverage_ratio", 0.0)

    return {
        "glb_url": f"/download/{glb_path.name}",
        "json_url": f"/download/{json_path.name}",
        "stats": {
            "faces": total_faces,
            "vertices": total_verts,
            "meshes": len(merged.geometry),
            "zones": len(zones),
            "building_count": len(binfos_out),
            "coverage": coverage_ratio,
            "glb_kb": round(glb_path.stat().st_size / 1024, 1),
            "json_kb": json_kb,
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

    # 5. Classify prompt → graph_family + intent (optional, model may be missing)
    classification = None
    try:
        from prompt_classifier import classify_prompt
        primary_style = zones[0]["style"] if zones else None
        classification = classify_prompt(req.prompt, style=primary_style)
        if classification:
            print(f"[CLASSIFIER] family={classification['family']} "
                  f"intent={classification['intent']} "
                  f"conf={classification['confidence']:.2f}"
                  + (" (deepseek-confirmed)" if classification.get("deepseek_confirmed") else ""))
    except Exception as e:
        print(f"[CLASSIFIER] skipped: {e}")

    # 6. Generate
    arch_plan = parsed.get("archetype_plan")
    result = run_generation(zones, min_gap, seed, archetype_plan=arch_plan,
                            graph_name=graph_name)
    result["parsed"] = {"zones": zones, "min_gap": min_gap, "seed": seed}
    result["archetype_plan"] = arch_plan
    result["classification"] = classification
    return result


def _regenerate_from_placement(placement: dict) -> "trimesh.Scene":
    """Rebuild mesh from a placement JSON without re-running layout.

    Each building is reconstructed via generate_level.build_room() using
    style-profile params overridden by the placement's per-building values.
    This preserves positions and only changes the geometry that was edited.
    """
    import trimesh
    import numpy as np
    import trimesh.transformations as TF
    import generate_level as gl
    from style_base_profiles import apply_style_profile, get_profile_for_style
    from geometry.materials import STYLE_PALETTES
    from shapely.geometry import box as shapely_box

    scene_info = placement.get("scene", {})
    scene_style = scene_info.get("style_key", "medieval")
    buildings = placement.get("buildings", [])

    all_meshes = []
    for b in buildings:
        build_style = b.get("style_key", scene_style)

        # Seed default params that apply_style_profile expects
        w = float(b.get("width", 8))
        d = float(b.get("depth", 8))
        h = float(b.get("height", 6))
        bparams = {
            "height_range": [h * 0.9, h],
            "wall_thickness": 0.3,
            "floor_thickness": 0.2,
            "subdivision": 1,
        }

        # Get and apply style profile overrides
        profile = get_profile_for_style(build_style)
        if profile:
            bparams = apply_style_profile(bparams, profile,
                                          role=b.get("role", "primary"))

        # Force placement height (profile may have overridden height_range)
        bparams["height_range"] = [h * 0.95, h]
        feat = b.get("features") or {}
        bparams.setdefault("wall_thickness", float(feat.get("wall_thickness", 0.3)))
        bparams.setdefault("floor_thickness", 0.2)
        bparams.setdefault("subdivision", int(feat.get("subdivision", 1)))

        # Roof
        roof = b.get("roof") or {}
        _ROOF_TYPE_MAP = {"flat": 0, "gabled": 1, "hipped": 2, "pagoda": 3, "dome": 4}
        bparams["roof_type"] = _ROOF_TYPE_MAP.get(roof.get("type", "flat"), 0)
        bparams["roof_pitch"] = float(roof.get("pitch", 0.3))
        bparams["eave_overhang"] = float(roof.get("eave_overhang", 0.0))

        # Doors / windows
        doors = b.get("doors") or [{}]
        door0 = doors[0] if doors else {}
        bparams["door_spec"] = {
            "width": float(door0.get("width", 1.2)),
            "height": float(door0.get("height", 2.2)),
        }
        wins = b.get("windows") or [{}]
        win0 = wins[0] if wins else {}
        bparams["win_spec"] = {
            "width": float(win0.get("width", 0.6)),
            "height": float(win0.get("height", 0.8)),
            "density": float(win0.get("density", 0.3)),
        }
        bparams.setdefault("window_shape", int(win0.get("shape_id", 0)))

        # Features
        bparams.setdefault("has_battlements", int(feat.get("has_battlements", 0)))
        bparams.setdefault("has_arch", int(feat.get("has_arch", 0)))
        bparams.setdefault("column_count", int(feat.get("column_count", 0)))
        bparams.pop("_material_variation", None)
        bparams.pop("_symmetry_bias", None)

        # Palette
        base_style = build_style.split("_")[0] if "_" in build_style else build_style
        palette = STYLE_PALETTES.get(base_style,
                 STYLE_PALETTES.get("medieval"))

        # Footprint
        fp = shapely_box(0, 0, w, d)

        try:
            room = gl.build_room(bparams, palette, x_off=0, z_off=0,
                                 footprint=fp)

            # Rotate around building center
            yaw = float(b.get("rotation_deg", 0))
            if abs(yaw) > 0.01:
                cx, cz = w / 2, d / 2
                rot = TF.rotation_matrix(np.radians(yaw), [0, 1, 0])
                for m in (room if isinstance(room, list) else [room]):
                    if isinstance(m, trimesh.Trimesh):
                        m.vertices -= [cx, 0, cz]
                        m.apply_transform(rot)
                        m.vertices += [cx, 0, cz]

            # Translate to world position
            px = float(b["position"]["x"])
            pz = float(b["position"]["z"])
            for m in (room if isinstance(room, list) else [room]):
                if isinstance(m, trimesh.Trimesh):
                    m.apply_translation([px, 0, pz])
                    all_meshes.append(m)

        except Exception as e:
            try:
                print(f"[REGEN] build_room failed for B{b.get('id')}: {e}")
            except Exception:
                pass

    return trimesh.Scene(all_meshes) if all_meshes else trimesh.Scene()


@app.post("/edit", response_model=EditResponse)
async def edit_scene(req: EditRequest):
    """
    Apply a natural-language edit to the current scene.

    Strategy: modify the placement JSON in-place, then rebuild meshes
    from the modified placement (no layout re-roll). Only 'regenerate'
    and 'modify_style' go through full run_generation().
    """
    from edit_parser import (
        parse_edit_intent,
        parse_direct_edit,
        apply_direct_edit,
        apply_edit_to_zone,
        apply_edit_to_placement,
    )

    # 0. Direct UI edits bypass the parser entirely
    direct = parse_direct_edit(req.instruction)
    if direct:
        cs = req.current_scene or {}
        json_url = cs.get("json_url", "")
        json_name = json_url.rsplit("/", 1)[-1] if json_url else ""
        json_path = OUTPUT_DIR / json_name if json_name else None
        if json_path and json_path.exists():
            placement = json.loads(json_path.read_text(encoding="utf-8"))
            placement = apply_direct_edit(direct, placement)
            # Rebuild mesh from modified placement
            t0 = time.time()
            rebuilt_scene = _regenerate_from_placement(placement)
            elapsed = time.time() - t0
            edit_hash = hashlib.md5(req.instruction.encode()).hexdigest()[:6]
            base_name = f"edit_{edit_hash}_{int(time.time()) % 100000}"
            glb_path = OUTPUT_DIR / f"{base_name}.glb"
            new_json_path = OUTPUT_DIR / f"{base_name}_placement.json"
            rebuilt_scene.export(str(glb_path))
            new_json_path.write_text(
                json.dumps(placement, indent=2, ensure_ascii=False), encoding="utf-8")
            binfos = [{"idx": b.get("id",0), "x": round(b["position"]["x"],2),
                       "z": round(b["position"]["z"],2),
                       "yaw_deg": round(b.get("rotation_deg",0),1),
                       "width": round(b.get("width",0),2), "depth": round(b.get("depth",0),2),
                       "cluster_id": -1, "is_main": b.get("role")=="primary",
                       "role": b.get("role",""), "style_key": b.get("style_key","")}
                      for b in placement.get("buildings", [])]
            return {
                "glb_url": f"/download/{glb_path.name}",
                "json_url": f"/download/{new_json_path.name}",
                "stats": {"faces": sum(len(g.faces) for g in rebuilt_scene.geometry.values() if hasattr(g,'faces')),
                          "vertices": sum(len(g.vertices) for g in rebuilt_scene.geometry.values() if hasattr(g,'vertices')),
                          "meshes": len(rebuilt_scene.geometry),
                          "zones": 1, "building_count": len(binfos),
                          "coverage": (placement.get("metadata") or {}).get("coverage_ratio", 0),
                          "glb_kb": round(glb_path.stat().st_size/1024, 1),
                          "json_kb": round(new_json_path.stat().st_size/1024, 1),
                          "time_s": round(elapsed, 2)},
                "parsed": cs.get("parsed", {}),
                "clusters": [], "road_graph": {}, "building_infos": binfos,
                "archetype_plan": None, "classification": cs.get("classification"),
                "edit_summary": direct.get("summary", "direct edit applied"),
            }

    # 1. Parse the instruction (LLM first, keyword fallback)
    edit = parse_edit_intent(req.instruction, req.current_scene)
    try:
        print(f"[EDIT] intent={edit.get('intent')}  source={edit.get('source')}  "
              f"target={edit.get('target')}  direction={edit.get('direction')}  "
              f"amount={edit.get('amount')}  new_style={edit.get('new_style')}")
    except Exception:
        pass

    cs = req.current_scene or {}
    cs_parsed = cs.get("parsed") or {}

    # 2. Regenerate / modify_style → full pipeline (layout changes)
    if edit.get("intent") in ("regenerate", "modify_style"):
        cs_zones = cs_parsed.get("zones") or []
        base_zone = dict(cs_zones[0]) if cs_zones else {
            "name": "main", "style": "medieval", "layout": "organic",
            "count": 10, "density": "normal",
        }
        base_min_gap = cs_parsed.get("min_gap", 3.0)
        base_seed = cs_parsed.get("seed", int(time.time()) % 100000)

        if edit.get("intent") == "regenerate":
            gen_req = GenerateRequest(prompt=req.instruction)
            result = await generate(gen_req)
            if isinstance(result, dict):
                result["edit_summary"] = edit.get("summary", "已重新生成场景")
            return result

        # modify_style: re-run with new style
        new_zone, new_min_gap, new_seed, summary = apply_edit_to_zone(
            edit, base_zone, base_min_gap, base_seed)
        if new_zone["style"] not in AVAILABLE_STYLES:
            new_zone["style"] = base_zone.get("style", "medieval")
        if new_zone.get("layout") not in AVAILABLE_LAYOUTS:
            new_zone["layout"] = _STYLE_DEFAULT_LAYOUT.get(new_zone["style"], "organic")
        result = run_generation([new_zone], new_min_gap, new_seed)
        result["parsed"] = {"zones": [new_zone], "min_gap": new_min_gap, "seed": new_seed}
        result["archetype_plan"] = None
        result["classification"] = None
        result["edit_summary"] = summary
        return result

    # 3. All other intents: modify placement JSON + rebuild mesh from it
    #    Load the existing placement JSON from the json_url
    json_url = cs.get("json_url", "")
    json_name = json_url.rsplit("/", 1)[-1] if json_url else ""
    json_path = OUTPUT_DIR / json_name if json_name else None

    if not json_path or not json_path.exists():
        # Fall back to full re-generation if we can't find the placement
        cs_zones = cs_parsed.get("zones") or []
        base_zone = dict(cs_zones[0]) if cs_zones else {
            "name": "main", "style": "medieval", "layout": "organic",
            "count": 10, "density": "normal"}
        base_min_gap = cs_parsed.get("min_gap", 3.0)
        base_seed = cs_parsed.get("seed", 42)
        new_zone, new_min_gap, new_seed, summary = apply_edit_to_zone(
            edit, base_zone, base_min_gap, base_seed)
        result = run_generation([new_zone], new_min_gap, new_seed)
        result["parsed"] = {"zones": [new_zone], "min_gap": new_min_gap, "seed": new_seed}
        result["edit_summary"] = summary + " (fallback: full regeneration)"
        return result

    # Load existing placement, apply the edit, rebuild mesh
    placement = json.loads(json_path.read_text(encoding="utf-8"))
    _, _, _, summary = apply_edit_to_zone(
        edit,
        (cs_parsed.get("zones") or [{}])[0] if cs_parsed.get("zones") else {},
        cs_parsed.get("min_gap", 3.0),
        cs_parsed.get("seed", 42),
    )
    placement = apply_edit_to_placement(edit, placement)

    # Rebuild mesh from modified placement
    t0 = time.time()
    rebuilt_scene = _regenerate_from_placement(placement)
    elapsed = time.time() - t0

    # Export GLB + updated JSON
    edit_hash = hashlib.md5(req.instruction.encode()).hexdigest()[:6]
    base_name = f"edit_{edit_hash}_{int(time.time()) % 100000}"
    glb_path = OUTPUT_DIR / f"{base_name}.glb"
    new_json_path = OUTPUT_DIR / f"{base_name}_placement.json"

    rebuilt_scene.export(str(glb_path))
    new_json_path.write_text(
        json.dumps(placement, indent=2, ensure_ascii=False), encoding="utf-8")

    total_faces = sum(len(g.faces) for g in rebuilt_scene.geometry.values()
                      if hasattr(g, "faces"))
    total_verts = sum(len(g.vertices) for g in rebuilt_scene.geometry.values()
                      if hasattr(g, "vertices"))
    n_buildings = len(placement.get("buildings", []))
    coverage = (placement.get("metadata") or {}).get("coverage_ratio", 0)

    # Build response matching GenerateResponse shape
    binfos = []
    for b in placement.get("buildings", []):
        binfos.append({
            "idx": b.get("id", 0),
            "x": round(b["position"]["x"], 2),
            "z": round(b["position"]["z"], 2),
            "yaw_deg": round(b.get("rotation_deg", 0), 1),
            "width": round(b.get("width", 0), 2),
            "depth": round(b.get("depth", 0), 2),
            "cluster_id": -1,
            "is_main": b.get("role") == "primary",
            "role": b.get("role", ""),
            "style_key": b.get("style_key", ""),
        })

    return {
        "glb_url": f"/download/{glb_path.name}",
        "json_url": f"/download/{new_json_path.name}",
        "stats": {
            "faces": total_faces,
            "vertices": total_verts,
            "meshes": len(rebuilt_scene.geometry),
            "zones": 1,
            "building_count": n_buildings,
            "coverage": coverage,
            "glb_kb": round(glb_path.stat().st_size / 1024, 1),
            "json_kb": round(new_json_path.stat().st_size / 1024, 1),
            "time_s": round(elapsed, 2),
        },
        "parsed": cs_parsed,
        "clusters": [],
        "road_graph": {},
        "building_infos": binfos,
        "archetype_plan": None,
        "classification": cs.get("classification"),
        "edit_summary": summary,
    }


@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "File not found")
    if filename.endswith(".glb"):
        media = "model/gltf-binary"
    elif filename.endswith(".json"):
        media = "application/json"
    else:
        media = "application/octet-stream"
    return FileResponse(path, media_type=media, filename=filename)
