"""
Augment graph_family_dataset.jsonl from ~40 to ~300 records using the
Anthropic Claude API.

Strategy:
  1. Per-family expansion: 20 new records per family (10 families = 200)
  2. Boundary cases: 50 records spanning ambiguous family pairs
  3. Diverse language: 50 records with varied phrasing styles

The script reads the existing dataset, calls Claude with strict prompts that
encode the actual schema enums, validates each generated record, and writes
the merged result to graph_family_dataset_v2.jsonl.

Usage:
    cd training
    # ANTHROPIC_API_KEY in .env or environment
    python augment_dataset.py

Requirements:
    pip install anthropic python-dotenv

Cost estimate: ~50-80K output tokens at Opus 4.6 rates (~$2-3 total).
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import anthropic

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set. Add it to training/.env or export it.")
    sys.exit(1)


# ─── Paths ──────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
INPUT_PATH = DATA_DIR / "graph_family_dataset.jsonl"
OUTPUT_PATH = DATA_DIR / "graph_family_dataset_v2.jsonl"


# ─── Schema vocabulary (must match validate_graph_families.py) ───────

VALID_INTENTS = {"defense", "residence", "worship", "commerce", "industry",
                 "governance", "mixed"}
VALID_ENTRY_MODES = {"single_gate", "double_gate", "open_approach",
                     "bridge_crossing", "tunnel", "multi_gate", "ramp"}
VALID_PERIMETER = {"curtain_wall", "palisade", "moat_and_wall", "cliff_edge",
                   "open", "partial_wall", "terraced_wall"}
VALID_TOWER_DIST = {"corners_only", "corners_and_midwall", "gate_flanking",
                    "regular_interval", "clustered_at_entry", "none"}
VALID_COURTYARD = {"single_open", "divided", "tiered", "cloistered", "none",
                   "central_plaza", "ring"}
VALID_DAMAGE = {"pristine", "weathered", "battle_scarred", "ruined", "overgrown"}

NODE_VOCABULARY = {
    "central_keep", "great_hall", "throne_room",
    "gatehouse", "defensive_tower", "corner_tower", "watchtower",
    "residential_tower", "barracks", "guard_post",
    "chapel", "temple", "shrine",
    "stable", "smithy", "granary", "storehouse", "armory",
    "market", "tavern", "inn",
    "well", "fountain", "cistern",
    "training_ground", "arena",
    "garden", "courtyard_tree", "statue",
    "workshop", "furnace", "chimney_stack", "warehouse",
    "dock", "crane", "silo",
    "prison", "dungeon",
    "library", "scriptorium",
    "kitchen", "bakery",
    "cemetery", "crypt",
}

VALID_STYLES = [
    "medieval", "medieval_chapel", "medieval_keep",
    "japanese", "japanese_temple", "japanese_machiya",
    "modern", "modern_loft", "modern_villa",
    "industrial", "industrial_workshop", "industrial_powerplant",
    "fantasy", "fantasy_dungeon", "fantasy_palace",
    "horror", "horror_asylum", "horror_crypt",
    "desert", "desert_palace",
]


# ─── Load existing dataset ──────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


existing = load_jsonl(INPUT_PATH)
print(f"Loaded {len(existing)} existing records from {INPUT_PATH.name}")

by_family: dict[str, list[dict]] = {}
for r in existing:
    by_family.setdefault(r["graph_family"], []).append(r)

FAMILIES = sorted(by_family.keys())
print(f"Families ({len(FAMILIES)}): {FAMILIES}")


# ─── Schema reference block (injected into every prompt) ─────────────

SCHEMA_BLOCK = f"""Each record is a flat JSON object with these exact fields and enum values:

REQUIRED FIELDS:
  prompt          (string) — natural language description of the scene
  style           (string) — one of: {sorted(VALID_STYLES)}
  intent          (string) — one of: {sorted(VALID_INTENTS)}
  graph_family    (string) — one of: {FAMILIES}

OPTIONAL FIELDS (include all when possible — they shape the layout):
  entry_mode          (string) — one of: {sorted(VALID_ENTRY_MODES)}
  perimeter_type      (string) — one of: {sorted(VALID_PERIMETER)}
  tower_distribution  (string) — one of: {sorted(VALID_TOWER_DIST)}
  courtyard_mode      (string) — one of: {sorted(VALID_COURTYARD)}
  damage_state        (string) — one of: {sorted(VALID_DAMAGE)}
  support_program     (list of strings) — each item from this vocabulary:
                       {sorted(NODE_VOCABULARY)}

CRITICAL RULES:
  - The structure is FLAT. Do NOT nest fields under "attributes".
  - Use the EXACT enum strings shown above. Do not invent new values.
  - support_program MUST be a list of strings, with 2-6 items per record.
  - Each support_program item MUST come from the vocabulary above.
  - Output ONLY raw JSON, one object per line. No markdown fences, no commentary.
"""


# ─── Anthropic client ───────────────────────────────────────────────

client = anthropic.Anthropic()
MODEL = "claude-opus-4-6"


def call_claude(prompt: str, max_tokens: int = 8000) -> str:
    """Stream a request and return the final text content."""
    with client.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for _ in stream.text_stream:
            pass
        final = stream.get_final_message()

    text_parts = [b.text for b in final.content if b.type == "text"]
    return "".join(text_parts)


# ─── JSON parsing & validation ──────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*$|^```\s*$", re.MULTILINE)


def parse_jsonl_response(text: str) -> list[dict]:
    """Extract JSON objects from a model response, tolerant of markdown fences."""
    text = _FENCE_RE.sub("", text).strip()
    results = []
    for line in text.split("\n"):
        line = line.strip().rstrip(",")
        if not line or line in ("[", "]"):
            continue
        if line.startswith("//") or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            continue
    return results


def validate_record(r: dict) -> tuple[bool, str]:
    """Return (valid, error_message)."""
    for field in ("prompt", "style", "intent", "graph_family"):
        if field not in r or not isinstance(r[field], str):
            return False, f"missing/invalid {field}"

    if r["intent"] not in VALID_INTENTS:
        return False, f"intent={r['intent']!r}"
    if r.get("entry_mode") and r["entry_mode"] not in VALID_ENTRY_MODES:
        return False, f"entry_mode={r['entry_mode']!r}"
    if r.get("perimeter_type") and r["perimeter_type"] not in VALID_PERIMETER:
        return False, f"perimeter_type={r['perimeter_type']!r}"
    if r.get("tower_distribution") and r["tower_distribution"] not in VALID_TOWER_DIST:
        return False, f"tower_distribution={r['tower_distribution']!r}"
    if r.get("courtyard_mode") and r["courtyard_mode"] not in VALID_COURTYARD:
        return False, f"courtyard_mode={r['courtyard_mode']!r}"
    if r.get("damage_state") and r["damage_state"] not in VALID_DAMAGE:
        return False, f"damage_state={r['damage_state']!r}"

    sp = r.get("support_program", [])
    if not isinstance(sp, list):
        return False, "support_program is not a list"
    for item in sp:
        if item not in NODE_VOCABULARY:
            return False, f"support_program item={item!r}"

    return True, ""


# ─── Augmentation rounds ────────────────────────────────────────────

def augment_family(family: str, examples: list[dict], count: int = 20) -> list[dict]:
    """Generate `count` new records for one graph family."""
    sample = examples[:4]
    samples_str = "\n".join(json.dumps(e, ensure_ascii=False) for e in sample)

    prompt = f"""You are a procedural level design data annotator.

{SCHEMA_BLOCK}

Existing examples for graph_family = "{family}":
{samples_str}

Generate exactly {count} NEW records for graph_family = "{family}".

Diversity requirements:
  - 5 short prompts (3-8 words, e.g. "mountain fortress")
  - 5 medium prompts (10-20 words, mention atmosphere or function)
  - 5 long prompts (20-40 words, narrative or stakeholder hooks)
  - 5 intent-only prompts (describe purpose without naming building type)

Other diversity:
  - Vary `style` across at least 4 different style values appropriate for "{family}"
  - Vary `damage_state` (mix pristine/weathered/battle_scarred/ruined/overgrown)
  - Vary `entry_mode`, `perimeter_type`, `tower_distribution`, `courtyard_mode`
  - support_program lists should differ between records (2-6 items each)

Output exactly {count} JSON objects, one per line. NO markdown, NO commentary."""

    print(f"  [{family}] requesting {count} records ... ", end="", flush=True)
    text = call_claude(prompt, max_tokens=8000)
    raw = parse_jsonl_response(text)

    valid = []
    rejected = 0
    for r in raw:
        # Force the family label — sometimes Claude drifts
        r["graph_family"] = family
        ok, _ = validate_record(r)
        if ok:
            valid.append(r)
        else:
            rejected += 1
    print(f"got {len(raw)} parsed, {len(valid)} valid, {rejected} rejected")
    return valid


def augment_boundary_cases(count: int = 50) -> list[dict]:
    """Generate ambiguous-boundary records covering confusing family pairs."""
    family_summary = "\n".join(
        f"  - {f}: {len(by_family[f])} existing examples" for f in FAMILIES
    )

    prompt = f"""You are a procedural level design data annotator.

{SCHEMA_BLOCK}

Available graph families:
{family_summary}

Generate {count} BOUNDARY-CASE records — prompts whose surface description
could plausibly fit multiple families, but where one is clearly correct.

Coverage requirements:
  - At least 8 different graph_family labels across the {count} records
  - Focus especially on these confusable pairs:
      mountain_fortress vs walled_settlement
      temple_complex vs institutional_compound
      manor_estate vs royal_palace
      open_settlement vs walled_settlement
      industrial_district vs institutional_compound
      border_redoubt vs mountain_fortress
      underground_ruin vs mountain_fortress
  - Each record must include all optional fields (entry_mode, perimeter_type,
    tower_distribution, courtyard_mode, damage_state, support_program)
  - The chosen family must be unambiguously the best fit, with attributes
    that match it (e.g. a manor_estate has lower defenses than royal_palace)

Output exactly {count} JSON objects, one per line. NO markdown, NO commentary."""

    print(f"\n[boundary cases] requesting {count} records ... ", end="", flush=True)
    text = call_claude(prompt, max_tokens=10000)
    raw = parse_jsonl_response(text)

    valid = []
    rejected_examples = []
    for r in raw:
        ok, err = validate_record(r)
        if ok and r.get("graph_family") in FAMILIES:
            valid.append(r)
        else:
            if len(rejected_examples) < 3:
                rejected_examples.append(err or "wrong family")

    print(f"got {len(raw)} parsed, {len(valid)} valid")
    if rejected_examples:
        print(f"  rejection samples: {rejected_examples}")
    return valid


def augment_diverse_language(count: int = 50) -> list[dict]:
    """Generate records with varied prompt phrasing styles."""
    prompt = f"""You are a procedural level design data annotator.

{SCHEMA_BLOCK}

Available families: {FAMILIES}

Generate {count} records that test LANGUAGE DIVERSITY in the prompt field.

Phrasing distribution (10 records each):
  - Mixed Chinese-English (e.g. "一个 medieval 风格的山顶城堡")
  - Game/RPG terminology (e.g. "raid dungeon with a boss arena")
  - Minimal commands (1-3 words: "dark castle", "factory", "shrine")
  - Narrative style (1-2 sentences telling a small story about the place)
  - Negation/contrast (e.g. "a settlement with NO walls", "a temple without towers")

For each record include all optional fields. Pick whichever graph_family best
fits the prompt. Distribute across at least 6 different families.

Output exactly {count} JSON objects, one per line. NO markdown, NO commentary."""

    print(f"\n[diverse language] requesting {count} records ... ", end="", flush=True)
    text = call_claude(prompt, max_tokens=10000)
    raw = parse_jsonl_response(text)

    valid = []
    for r in raw:
        ok, _ = validate_record(r)
        if ok and r.get("graph_family") in FAMILIES:
            valid.append(r)
    print(f"got {len(raw)} parsed, {len(valid)} valid")
    return valid


# ─── Main ───────────────────────────────────────────────────────────

def main():
    print(f"\nUsing model: {MODEL}")
    print(f"Output: {OUTPUT_PATH}\n")

    all_new: list[dict] = []

    print("=== Round 1: per-family expansion ===")
    for family in FAMILIES:
        try:
            new = augment_family(family, by_family[family], count=20)
            all_new.extend(new)
        except Exception as e:
            print(f"  [{family}] FAILED: {e}")

    print("\n=== Round 2: boundary cases ===")
    try:
        all_new.extend(augment_boundary_cases(count=50))
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n=== Round 3: diverse language ===")
    try:
        all_new.extend(augment_diverse_language(count=50))
    except Exception as e:
        print(f"  FAILED: {e}")

    # Merge and write
    merged = existing + all_new
    print(f"\nTotal: {len(existing)} existing + {len(all_new)} new = {len(merged)} records")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT_PATH}")

    # Stats
    print("\n=== Distribution ===")
    fam_counts = Counter(r.get("graph_family", "?") for r in merged)
    print("Family:")
    for fam, c in sorted(fam_counts.items(), key=lambda kv: -kv[1]):
        bar = "#" * c
        print(f"  {fam:25s} {c:4d}  {bar}")

    intent_counts = Counter(r.get("intent", "?") for r in merged)
    print("\nIntent:")
    for it, c in sorted(intent_counts.items(), key=lambda kv: -kv[1]):
        bar = "#" * c
        print(f"  {it:15s} {c:4d}  {bar}")


if __name__ == "__main__":
    main()
