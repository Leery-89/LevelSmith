"""
LevelSmith — natural-language edit instruction parser.

Maps short Chinese / English instructions to a structured edit operation,
then translates that operation into modifications of:
  - the generation parameters (style, count, min_gap, seed) used by
    run_generation() to produce a fresh mesh, AND
  - the resulting placement JSON for changes that don't have a direct
    backend knob (per-building height tweaks, entrance direction).

Used by the /edit endpoint in api.py.

Supported intents
-----------------
  modify_height    "主建筑再高一点" / "make it taller"
  modify_count     "建筑少一点" / "add a tower"
  modify_spacing   "更开阔" / "more compact"
  modify_entrance  "入口改到南边" / "gate on the west"
  modify_style     "换成日式" / "change to japanese"
"""

from __future__ import annotations

import copy
from typing import Any

# ─── Style vocabulary (must match style_registry / classifier) ───────

KNOWN_STYLES = {
    "medieval", "medieval_chapel", "medieval_keep",
    "japanese", "japanese_temple", "japanese_machiya",
    "modern", "modern_loft", "modern_villa",
    "industrial", "industrial_workshop", "industrial_powerplant",
    "fantasy", "fantasy_dungeon", "fantasy_palace",
    "horror", "horror_asylum", "horror_crypt",
    "desert", "desert_palace",
}

# Short alias → full style key (so "japanese" matches "japanese_temple")
STYLE_ALIASES = {
    "medieval": "medieval", "中世纪": "medieval", "城堡": "medieval_keep",
    "japanese": "japanese", "日式": "japanese_temple", "和风": "japanese_temple",
    "industrial": "industrial", "工业": "industrial",
    "fantasy": "fantasy_palace", "奇幻": "fantasy_palace",
    "horror": "horror_crypt", "恐怖": "horror_crypt",
    "modern": "modern", "现代": "modern_villa",
    "desert": "desert_palace", "沙漠": "desert_palace",
}


# ─── Pattern groups ──────────────────────────────────────────────────

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
]


# ─── Intent parser ───────────────────────────────────────────────────

def parse_edit_intent(instruction: str) -> dict[str, Any]:
    """Parse an instruction into a structured edit operation.

    Returns:
        {
          "intent":    one of modify_{height,count,spacing,entrance,style}|"unknown",
          "raw":       original instruction,
          "target":    "primary"|"tower"|"all"|... (height edits only)
          "direction": +1|-1 (height/count/spacing) or "south"|... (entrance)
          "new_style": one of KNOWN_STYLES (style edits only)
        }
    """
    text = instruction.lower().strip()
    if not text:
        return {"intent": "unknown", "raw": instruction}

    # modify_style is special — it doesn't always need a "change to" trigger,
    # just the presence of a known style name in the instruction is enough.
    for alias, target_style in STYLE_ALIASES.items():
        if alias in text:
            # Make sure it's not a no-op (current style might equal target)
            return {
                "intent": "modify_style",
                "raw": instruction,
                "new_style": target_style,
            }

    for group in EDIT_PATTERNS:
        if group["intent"] == "modify_style":
            continue  # already handled above
        if not any(p in text for p in group["patterns"]):
            continue

        edit: dict[str, Any] = {"intent": group["intent"], "raw": instruction}

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

    return {"intent": "unknown", "raw": instruction}


# ─── Zone-level editor (drives run_generation) ───────────────────────

def apply_edit_to_zone(
    edit: dict,
    base_zone: dict,
    base_min_gap: float,
    base_seed: int,
) -> tuple[dict, float, int, str]:
    """Translate an edit into modified generation parameters.

    Returns:
        (new_zone, new_min_gap, new_seed, summary)

    The seed is always bumped so even no-op edits produce a fresh mesh.
    """
    intent = edit.get("intent", "unknown")
    new_zone = dict(base_zone)
    new_min_gap = base_min_gap
    new_seed = (base_seed or 0) + 1
    summary = ""

    if intent == "modify_count":
        delta = 2 * (edit.get("direction") or 1)
        old = base_zone.get("count", 10)
        new = max(3, min(20, old + delta))
        new_zone["count"] = new
        if new == old:
            summary = f"建筑数量已经在边界 ({new})"
        else:
            summary = f"建筑数量从 {old} 调整为 {new}"

    elif intent == "modify_spacing":
        scale = 1.5 if (edit.get("direction") or 1) > 0 else 0.65
        new_min_gap = max(1.0, min(10.0, base_min_gap * scale))
        verb = "扩大" if scale > 1 else "缩小"
        summary = f"间距已{verb} ({base_min_gap:.1f}m → {new_min_gap:.1f}m)"

    elif intent == "modify_style":
        new_style = edit.get("new_style")
        if new_style and new_style in KNOWN_STYLES:
            old_style = base_zone.get("style", "?")
            new_zone["style"] = new_style
            summary = f"风格已从 {old_style} 改为 {new_style}"
        else:
            summary = "未识别到目标风格"

    elif intent == "modify_height":
        # No direct knob in run_generation. We bump the seed so the user
        # sees a fresh layout, and post-process the placement JSON to
        # actually carry the height delta (handled by apply_edit_to_placement).
        target = edit.get("target", "all")
        direction = edit.get("direction") or 1
        verb = "升高" if direction > 0 else "降低"
        target_label = {"primary": "主建筑", "tower": "塔楼", "all": "所有建筑"}.get(target, target)
        summary = f"已{verb}{target_label}（mesh 近似，JSON 精确）"

    elif intent == "modify_entrance":
        new_dir = edit.get("direction")
        if new_dir:
            dir_label = {"south": "南", "north": "北", "east": "东", "west": "西"}.get(new_dir, new_dir)
            summary = f"已将入口改到{dir_label}侧（仅 JSON）"
        else:
            summary = "未识别到入口方向"

    else:
        summary = "无法理解的修改指令，请换一种说法"

    return new_zone, new_min_gap, new_seed, summary


# ─── Placement-level editor (post-processes the placement JSON) ──────

def apply_edit_to_placement(edit: dict, placement: dict) -> dict:
    """Apply edits that don't map to generate_level params directly.

    Operates on the placement JSON returned by export_placement_json().
    Modifications: per-building height delta (modify_height) and
    enclosure gate side (modify_entrance).
    """
    if not placement or not isinstance(placement, dict):
        return placement

    out = copy.deepcopy(placement)
    intent = edit.get("intent", "unknown")
    buildings = out.get("buildings") or []

    if intent == "modify_height":
        target = edit.get("target", "all")
        direction = edit.get("direction") or 1
        delta = 3.0 * direction
        for b in buildings:
            if target == "all" or b.get("role") == target:
                old_h = float(b.get("height", 6.0))
                b["height"] = max(3.0, round(old_h + delta, 2))

    elif intent == "modify_entrance":
        new_side = edit.get("direction")
        if new_side and out.get("gates"):
            for g in out["gates"]:
                g["wall_side"] = new_side

    return out
