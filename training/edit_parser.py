"""
LevelSmith — natural-language edit instruction parser.

Two-tier parsing:
  1. DeepSeek LLM (when DEEPSEEK_API_KEY is set) — handles creative phrasing,
     Chinese/English mix, ambiguous instructions.
  2. Keyword matching (fallback) — covers the 5 basic patterns without network.

Supported intents
-----------------
  modify_height    "主建筑再高一点" / "make it much taller"
  modify_count     "建筑少一点" / "add two more towers"
  modify_spacing   "更开阔" / "make it more compact"
  modify_entrance  "入口改到南边" / "gate on the west"
  modify_style     "换成日式" / "change to industrial"
  modify_size      "让主建筑更宽" / "make buildings bigger"
  regenerate       "重新生成" / "start over"
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

# Load .env for DEEPSEEK_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ─── Style vocabulary ────────────────────────────────────────────────

KNOWN_STYLES = {
    "medieval", "medieval_chapel", "medieval_keep",
    "japanese", "japanese_temple", "japanese_machiya",
    "modern", "modern_loft", "modern_villa",
    "industrial", "industrial_workshop", "industrial_powerplant",
    "fantasy", "fantasy_dungeon", "fantasy_palace",
    "horror", "horror_asylum", "horror_crypt",
    "desert", "desert_palace",
}

STYLE_ALIASES = {
    "medieval": "medieval", "中世纪": "medieval", "城堡": "medieval_keep",
    "japanese": "japanese", "日式": "japanese_temple", "和风": "japanese_temple",
    "industrial": "industrial", "工业": "industrial",
    "fantasy": "fantasy_palace", "奇幻": "fantasy_palace",
    "horror": "horror_crypt", "恐怖": "horror_crypt",
    "modern": "modern", "现代": "modern_villa",
    "desert": "desert_palace", "沙漠": "desert_palace",
}


# ─── LLM parser (tier 1) ────────────────────────────────────────────

def _parse_edit_with_llm(instruction: str,
                         current_scene: Optional[dict]) -> Optional[dict]:
    """Call DeepSeek to parse a free-form edit instruction.

    Returns a structured dict or None on any failure (missing key, timeout,
    bad JSON, etc.)
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    # Build a compact scene summary for the LLM context
    scene_summary = {}
    if current_scene:
        scene_info = current_scene.get("scene") or {}
        parsed = current_scene.get("parsed") or {}
        zones = parsed.get("zones") or []
        zone0 = zones[0] if zones else {}
        buildings = current_scene.get("buildings") or []
        roles: dict[str, int] = {}
        for b in buildings:
            r = b.get("role", "unknown")
            roles[r] = roles.get(r, 0) + 1
        scene_summary = {
            "style": zone0.get("style") or scene_info.get("style_key", "unknown"),
            "layout": zone0.get("layout", "unknown"),
            "building_count": len(buildings) or (current_scene.get("stats") or {}).get("building_count", "?"),
            "roles": roles,
        }

    prompt = f"""You are a 3D scene editing assistant. The user has an existing scene:
{json.dumps(scene_summary, ensure_ascii=False)}

The user's edit instruction is: "{instruction}"

Parse it into the following JSON (output ONLY the JSON, no other text):

{{
  "intent": "modify_height | modify_count | modify_spacing | modify_entrance | modify_style | modify_size | modify_position | regenerate | unknown",
  "target": "primary | secondary | tertiary | ambient | tower | all",
  "direction": 1 or -1,
  "amount": <number or null>,
  "new_value": "<new value string or null>",
  "summary": "<one-line Chinese summary of the operation>"
}}

Rules:
- modify_height: change building height. amount = meters to add/remove (default 3).
  "高一点" → amount 3, "高很多" → 6, "稍微矮一点" → 2.
- modify_count: add/remove buildings. amount = number of buildings (default 2).
  "加一栋" → amount 1, direction 1.  "删掉几栋" → amount 3, direction -1.
- modify_spacing: change gap between buildings. direction 1 = wider, -1 = tighter.
- modify_entrance: move the gate. new_value = "south" | "north" | "east" | "west".
- modify_style: change architectural style. new_value = one of {sorted(KNOWN_STYLES)}.
- modify_size: change building width/depth. amount = meters delta (default 2).
- regenerate: user wants a completely new scene from scratch.
- unknown: cannot understand.

For target: "主建筑"/"最大的"/"keep" → primary. "塔楼" → tower.
  "小建筑" → tertiary. "所有建筑" → all. Not specified → all."""

    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        try:
            print(f"[EDIT-LLM] network error: {e}")
        except Exception:
            pass
        return None

    try:
        text = data["choices"][0]["message"]["content"].strip()
        # Strip possible markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        result = json.loads(text)
        if not isinstance(result, dict) or "intent" not in result:
            return None
        return result
    except (KeyError, json.JSONDecodeError, IndexError):
        return None


# ─── Keyword parser (tier 2 — fallback) ─────────────────────────────

EDIT_PATTERNS: list[dict[str, Any]] = [
    {
        "intent": "modify_height",
        "patterns": [
            "更高", "高一点", "高些", "矮一点", "更矮", "矮些",
            "taller", "shorter", "raise", "lower",
            "increase height", "decrease height",
        ],
        "target_keywords": {
            "主建筑": "primary", "main": "primary", "primary": "primary",
            "keep": "primary", "塔楼": "tower", "tower": "tower",
            "所有": "all", "全部": "all", "all": "all",
        },
        "direction_keywords": {
            "更高": +1, "高一点": +1, "高些": +1,
            "taller": +1, "raise": +1, "increase": +1,
            "更矮": -1, "矮一点": -1, "矮些": -1,
            "shorter": -1, "lower": -1, "decrease": -1,
        },
    },
    {
        "intent": "modify_count",
        "patterns": [
            "多一点", "少一点", "增加", "减少", "添加", "删掉", "移除",
            "more buildings", "fewer", "less", "add a", "add another",
            "remove", "delete",
        ],
        "direction_keywords": {
            "多一点": +1, "增加": +1, "添加": +1, "more": +1, "add": +1,
            "少一点": -1, "减少": -1, "删掉": -1, "移除": -1,
            "fewer": -1, "less": -1, "remove": -1, "delete": -1,
        },
    },
    {
        "intent": "modify_spacing",
        "patterns": [
            "更开阔", "更紧凑", "更密", "更疏", "更挤", "间距",
            "spread out", "compact", "tighter", "wider", "denser", "sparser",
        ],
        "direction_keywords": {
            "更开阔": +1, "更疏": +1, "spread out": +1, "wider": +1, "sparser": +1,
            "更紧凑": -1, "更密": -1, "更挤": -1,
            "compact": -1, "tighter": -1, "denser": -1,
        },
    },
    {
        "intent": "modify_entrance",
        "patterns": [
            "入口", "大门", "gate", "entrance", "entry",
        ],
        "direction_keywords": {
            "南": "south", "南边": "south", "south": "south",
            "北": "north", "北边": "north", "north": "north",
            "东": "east", "东边": "east", "east": "east",
            "西": "west", "西边": "west", "west": "west",
        },
    },
    {
        "intent": "modify_style",
        "patterns": [
            "换成", "改成", "风格", "change to", "make it", "switch to",
        ],
    },
    {
        "intent": "regenerate",
        "patterns": [
            "重新生成", "重来", "从头", "start over", "regenerate",
            "completely new", "全新",
        ],
    },
]


def _parse_edit_keyword(instruction: str) -> dict[str, Any]:
    """Pure keyword-based parser — no network, always works."""
    text = instruction.lower().strip()
    if not text:
        return {"intent": "unknown", "raw": instruction, "source": "keyword"}

    # Regenerate check first (simple)
    for p in ("重新生成", "重来", "从头", "start over", "regenerate", "全新"):
        if p in text:
            return {"intent": "regenerate", "raw": instruction, "source": "keyword"}

    # Style alias scan (catches "japanese" even without "change to")
    for alias, target_style in STYLE_ALIASES.items():
        if alias in text:
            return {
                "intent": "modify_style", "raw": instruction,
                "new_style": target_style, "source": "keyword",
            }

    for group in EDIT_PATTERNS:
        if group["intent"] in ("modify_style", "regenerate"):
            continue
        if not any(p in text for p in group["patterns"]):
            continue

        edit: dict[str, Any] = {
            "intent": group["intent"], "raw": instruction, "source": "keyword",
        }

        if "target_keywords" in group:
            for kw, target in group["target_keywords"].items():
                if kw in text:
                    edit["target"] = target
                    break
            edit.setdefault("target", "all")

        if "direction_keywords" in group:
            for kw, val in group["direction_keywords"].items():
                if kw in text:
                    edit["direction"] = val
                    break

        return edit

    return {"intent": "unknown", "raw": instruction, "source": "keyword"}


# ─── Public API: two-tier parse ──────────────────────────────────────

def parse_edit_intent(instruction: str,
                      current_scene: Optional[dict] = None) -> dict[str, Any]:
    """Parse an edit instruction. Tries LLM first, falls back to keywords.

    Returns:
        {
          "intent":    str,   # modify_height | modify_count | modify_spacing
                              # modify_entrance | modify_style | modify_size
                              # regenerate | unknown
          "target":    str,   # "primary"|"tower"|"all" etc.
          "direction": int|str,  # +1/-1 or "south" etc.
          "amount":    float|None,  # magnitude (from LLM)
          "new_value": str|None,    # new style name / direction (from LLM)
          "new_style": str|None,    # resolved style key (keyword path)
          "summary":   str,   # Chinese summary of the operation
          "source":    str,   # "llm" or "keyword"
          "raw":       str,   # original instruction
        }
    """
    # Tier 1: LLM
    llm = _parse_edit_with_llm(instruction, current_scene)
    if llm and llm.get("intent") and llm["intent"] != "unknown":
        edit = {
            "intent": llm["intent"],
            "target": llm.get("target", "all"),
            "direction": llm.get("direction", 1),
            "amount": llm.get("amount"),
            "new_value": llm.get("new_value"),
            "summary": llm.get("summary", ""),
            "raw": instruction,
            "source": "llm",
        }
        # For modify_style, resolve new_value to a known style key
        if edit["intent"] == "modify_style" and edit.get("new_value"):
            nv = edit["new_value"].lower().replace(" ", "_")
            if nv in KNOWN_STYLES:
                edit["new_style"] = nv
            else:
                # Try alias
                edit["new_style"] = STYLE_ALIASES.get(nv)
        return edit

    # Tier 2: keyword fallback
    return _parse_edit_keyword(instruction)


# ─── Zone-level editor (drives run_generation) ───────────────────────

def apply_edit_to_zone(
    edit: dict,
    base_zone: dict,
    base_min_gap: float,
    base_seed: int,
) -> tuple[dict, float, int, str]:
    """Translate an edit into modified generation parameters.

    Returns (new_zone, new_min_gap, new_seed, summary).
    """
    intent = edit.get("intent", "unknown")
    new_zone = dict(base_zone)
    new_min_gap = base_min_gap
    new_seed = (base_seed or 0) + 1
    summary = edit.get("summary", "")

    if intent == "modify_count":
        amount = edit.get("amount") or 2
        delta = int(amount) * (1 if (edit.get("direction") or 1) > 0 else -1)
        old = base_zone.get("count", 10)
        new = max(3, min(20, old + delta))
        new_zone["count"] = new
        summary = summary or (f"建筑数量从 {old} 调整为 {new}"
                              if new != old else f"建筑数量已在边界 ({new})")

    elif intent == "modify_spacing":
        scale = 1.5 if (edit.get("direction") or 1) > 0 else 0.65
        new_min_gap = max(1.0, min(10.0, base_min_gap * scale))
        verb = "扩大" if scale > 1 else "缩小"
        summary = summary or f"间距已{verb} ({base_min_gap:.1f}m -> {new_min_gap:.1f}m)"

    elif intent == "modify_style":
        new_style = edit.get("new_style") or edit.get("new_value")
        if new_style:
            # Resolve alias if needed
            resolved = STYLE_ALIASES.get(new_style.lower(), new_style)
            if resolved in KNOWN_STYLES:
                old_style = base_zone.get("style", "?")
                new_zone["style"] = resolved
                summary = summary or f"风格已从 {old_style} 改为 {resolved}"
            else:
                summary = summary or f"未识别风格: {new_style}"
        else:
            summary = summary or "未识别到目标风格"

    elif intent == "modify_height":
        target = edit.get("target", "all")
        direction = edit.get("direction") or 1
        verb = "升高" if direction > 0 else "降低"
        label = {"primary": "主建筑", "tower": "塔楼", "all": "所有建筑"}.get(target, target)
        summary = summary or f"已{verb}{label}"

    elif intent == "modify_entrance":
        new_dir = edit.get("new_value") or edit.get("direction")
        if new_dir:
            d = {"south": "南", "north": "北", "east": "东", "west": "西"}.get(str(new_dir), str(new_dir))
            summary = summary or f"已将入口改到{d}侧"
        else:
            summary = summary or "未识别到入口方向"

    elif intent == "modify_size":
        target = edit.get("target", "all")
        direction = edit.get("direction") or 1
        verb = "增大" if direction > 0 else "缩小"
        label = {"primary": "主建筑", "tower": "塔楼", "all": "所有建筑"}.get(target, target)
        summary = summary or f"已{verb}{label}尺寸"

    elif intent == "regenerate":
        summary = summary or "将完全重新生成场景"

    else:
        summary = summary or "无法理解的修改指令，请换一种说法"

    return new_zone, new_min_gap, new_seed, summary


# ─── Placement-level editor (post-processes the placement JSON) ──────

def apply_edit_to_placement(edit: dict, placement: dict) -> dict:
    """Apply edits that don't map to generate_level params directly.

    Modifies per-building height, size, and gate entrance side in the
    placement JSON. The mesh approximates; the JSON is the source of truth.
    """
    if not placement or not isinstance(placement, dict):
        return placement

    out = copy.deepcopy(placement)
    intent = edit.get("intent", "unknown")
    buildings = out.get("buildings") or []

    if intent == "modify_height":
        target = edit.get("target", "all")
        direction = edit.get("direction") or 1
        amount = float(edit.get("amount") or 3)
        delta = amount * (1 if direction > 0 else -1)
        for b in buildings:
            if target == "all" or b.get("role") == target:
                old_h = float(b.get("height", 6.0))
                b["height"] = max(3.0, round(old_h + delta, 2))

    elif intent == "modify_size":
        target = edit.get("target", "all")
        direction = edit.get("direction") or 1
        amount = float(edit.get("amount") or 2)
        delta = amount * (1 if direction > 0 else -1)
        for b in buildings:
            if target == "all" or b.get("role") == target:
                b["width"] = max(4.0, round(float(b.get("width", 8)) + delta, 2))
                b["depth"] = max(4.0, round(float(b.get("depth", 8)) + delta, 2))

    elif intent == "modify_entrance":
        new_side = edit.get("new_value") or edit.get("direction")
        if new_side and out.get("gates"):
            for g in out["gates"]:
                g["wall_side"] = str(new_side)

    return out
