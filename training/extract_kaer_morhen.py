"""Extract and classify Kaer Morhen mesh catalogue from W3 Depot."""
import json, re, sys
from pathlib import Path

DEPOT = Path("D:/W3Depot")
OUT = Path(__file__).parent / "data" / "kaer_morhen_catalogue.json"

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Find all kaer_morhen .w2mesh files
    meshes = sorted(set(
        f.name for f in DEPOT.rglob("*kaer_morhen*")
        if f.suffix == ".w2mesh"
    ))
    print(f"Total kaer_morhen meshes found: {len(meshes)}")

    # Classification rules (order matters — first match wins)
    rules = [
        ("entity_proxies", r"entity_\d+_proxy|entity_gate"),
        ("towers", r"tower|barbican|rock_tower|front_tower"),
        ("battlements", r"battlement"),
        ("walls", r"wall|brickwall|brick"),
        ("gates", r"gate"),
        ("columns", r"column"),
        ("stairs", r"stair|steps|under_door_stairs|wide_stairs"),
        ("roofs", r"roof|towerroof"),
        ("floors", r"floor|cokol"),
        ("windows", r"window|cross_window"),
        ("doors", r"door|stonedoor"),
        ("interior_arches", r"interior_arch|interior_vault"),
        ("interior_furniture", r"triss_interior|bed|chair|table|hourglass|carafe|pot|screen|plateau"),
        ("interior_details", r"interior_detail|interior_chimney|interior_painted|interior_shadow|interior_volume|interior_wall|interior_wooden|interior_floor|interior_roof"),
        ("mainkeep_details", r"mainkeep_detail"),
        ("bridges", r"bridge"),
        ("platforms", r"platform|wooden_stand"),
        ("debris_damage", r"debris|crashed|rubble|fake_rubble|damaged|pile"),
        ("props", r"well|stove|fireplace|toilet|swing|guard_post|training|bars|jail|chain|metal_eye|wood_pole|map"),
        ("environment", r"river|water|rock|trees|sand"),
    ]

    categories = {}
    for mesh in meshes:
        name = mesh.replace(".w2mesh", "")
        matched = False
        for cat, pattern in rules:
            if re.search(pattern, name):
                categories.setdefault(cat, []).append(name)
                matched = True
                break
        if not matched:
            categories.setdefault("unclassified", []).append(name)

    # Build catalogue
    catalogue = {
        "source": "W3 REDkit Depot",
        "depot_paths": [
            "D:/W3Depot/environment/architecture/human/kaer_morhen/",
            "D:/W3Depot/dlc/bob/data/environment/terrain_surroundings/",
        ],
        "total_meshes": len(meshes),
        "components": {k: sorted(v) for k, v in sorted(categories.items())},
        "inferred_archetype": {
            "primary": {
                "role": "central_keep",
                "evidence": f"mainkeep_detail x{len(categories.get('mainkeep_details', []))}, "
                            f"entity_proxy x{len(categories.get('entity_proxies', []))}",
                "component_types": ["mainkeep_details", "entity_proxies", "towers"],
            },
            "secondary": [
                {"role": "front_tower", "evidence": "front_tower_triss_* (Triss quarters)"},
                {"role": "yenn_tower", "evidence": "yenn_tower_* (Yennefer quarters)"},
                {"role": "barbican_tower", "evidence": "barbican_tower (gatehouse)"},
                {"role": "guard_post", "evidence": "guard_post"},
            ],
            "infrastructure": {
                "walls": len(categories.get("walls", [])),
                "battlements": len(categories.get("battlements", [])),
                "gates": len(categories.get("gates", [])),
                "bridges": len(categories.get("bridges", [])),
                "stairs": len(categories.get("stairs", [])),
            },
            "ambient": ["well", "training_ground (ciri)", "fireplace"],
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalogue, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print stats
    print()
    print("Category distribution:")
    for cat in sorted(categories, key=lambda c: -len(categories[c])):
        items = categories[cat]
        print(f"  {cat:25s}: {len(items):3d}  (e.g. {items[0]})")

    print()
    infra = catalogue["inferred_archetype"]["infrastructure"]
    print("Inferred archetype:")
    print(f"  primary:       central_keep (mainkeep_detail + entity_proxies)")
    print(f"  secondary:     front_tower (Triss), yenn_tower, barbican, guard_post")
    print(f"  infrastructure: {infra['walls']} walls, {infra['battlements']} battlements, "
          f"{infra['gates']} gates, {infra['bridges']} bridges, {infra['stairs']} stairs")
    print(f"  ambient:       well, training_ground, fireplace")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
