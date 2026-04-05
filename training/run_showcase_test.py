"""
LevelSmith MVP Showcase Stability Test
Generates 3 frozen showcases, runs acceptance checks, prints report.

Usage:  cd training && python run_showcase_test.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))

import level_layout
from style_base_profiles import get_profile_for_style

SHOWCASES = [
    {
        "name": "medieval_keep_courtyard",
        "style": "medieval_keep",
        "layout_type": "organic",
        "building_count": 10,
        "area_size": 120,
        "area_w": 120,    # override organic compaction
        "area_d": 120,
        "seed": 42,
        "min_gap": 4.0,
    },
    {
        "name": "japanese_temple_courtyard",
        "style": "japanese_temple",
        "layout_type": "organic",
        "building_count": 8,
        "area_size": 100,
        "area_w": 100,    # override organic compaction
        "area_d": 100,
        "seed": 42,
        "min_gap": 4.0,
    },
    {
        "name": "industrial_yard",
        "style": "industrial",
        "layout_type": "grid",
        "building_count": 8,
        "area_size": 100,
        "seed": 42,
        "min_gap": 5.0,
    },
]


def run_showcase(sc):
    name = sc["name"]
    r = {
        "generated": False, "crash": None, "building_count": 0,
        "coverage": 0.0, "overlaps": 0, "too_close": 0,
        "narrow": [], "has_primary": False, "roles": {},
        "glb_ok": False, "style_params_ok": False,
        "glb_kb": 0, "faces": 0,
    }

    print(f"\n{'=' * 60}")
    print(f"GENERATING: {name}")
    print(f"  style={sc['style']}  layout={sc['layout_type']}  count={sc['building_count']}  seed={sc['seed']}")
    print(f"{'=' * 60}")

    try:
        scene = level_layout.generate_level(
            style=sc["style"],
            layout_type=sc["layout_type"],
            building_count=sc["building_count"],
            area_size=sc.get("area_size", 100),
            area_w=sc.get("area_w"),
            area_d=sc.get("area_d"),
            seed=sc["seed"],
            min_gap=sc["min_gap"],
        )
        r["generated"] = True

        # Export
        glb_path = f"{name}.glb"
        scene.export(glb_path)
        fsize = os.path.getsize(glb_path)
        r["glb_ok"] = fsize > 1000
        r["glb_kb"] = fsize / 1024
        r["faces"] = sum(len(g.faces) for g in scene.geometry.values())
        print(f"  Export: {glb_path} ({r['glb_kb']:.1f} KB, {r['faces']:,} faces)")

        # Metadata
        meta = scene.metadata if hasattr(scene, "metadata") else {}
        buildings = meta.get("building_infos", [])
        r["building_count"] = len(buildings)
        print(f"  Buildings: {len(buildings)}")

        total_area = 0
        for i, b in enumerate(buildings):
            w = b.get("w", b.get("width", 0))
            d = b.get("d", b.get("depth", 0))
            h = b.get("h", b.get("height", 0))
            role = b.get("role", "")
            skey = b.get("style_key", sc["style"])
            area = w * d
            total_area += area
            tag = ""
            if w < 4.0 or d < 4.0:
                tag = " << NARROW"
                r["narrow"].append({"idx": i, "w": w, "d": d, "role": role})
            if role in ("primary", "anchor", ""):
                r["has_primary"] = True
            r["roles"][role] = r["roles"].get(role, 0) + 1
            print(f"    B{i}: {role:12s} {skey:20s} {w:5.1f}x{d:5.1f}x{h:5.1f}m  {area:6.1f}m2{tag}")

        scene_area = sc["area_size"] ** 2
        r["coverage"] = total_area / scene_area * 100
        print(f"  Coverage: {r['coverage']:.1f}% ({total_area:.0f}/{scene_area}m2)")

        # Overlap check
        for i in range(len(buildings)):
            for j in range(i + 1, len(buildings)):
                bi, bj = buildings[i], buildings[j]
                xi, zi = bi.get("x", 0), bi.get("z", 0)
                xj, zj = bj.get("x", 0), bj.get("z", 0)
                wi = bi.get("w", bi.get("width", 0))
                di_v = bi.get("d", bi.get("depth", 0))
                wj = bj.get("w", bj.get("width", 0))
                dj = bj.get("d", bj.get("depth", 0))
                cxi, czi = xi + wi / 2, zi + di_v / 2
                cxj, czj = xj + wj / 2, zj + dj / 2
                dx = abs(cxi - cxj)
                dz = abs(czi - czj)
                half_w = (wi + wj) / 2
                half_d = (di_v + dj) / 2
                if dx < half_w and dz < half_d:
                    r["overlaps"] += 1
                    print(f"  !! OVERLAP: B{i} x B{j}")
                elif dx < half_w + sc["min_gap"] and dz < half_d + sc["min_gap"]:
                    r["too_close"] += 1

        print(f"  Overlaps: {r['overlaps']}")
        print(f"  Too-close: {r['too_close']}")
        print(f"  Roles: {r['roles']}")

        # Style profile check
        prof = get_profile_for_style(sc["style"])
        r["style_params_ok"] = prof is not None
        if prof:
            rr = prof.get("roof_rules", {})
            print(f"  Style profile: YES  roof={rr.get('type', '?')}  eave={rr.get('eave_overhang', '?')}")
        else:
            print(f"  Style profile: NO (will use registry defaults)")

        print(f"  >>> Generation SUCCESS")

    except Exception as e:
        r["crash"] = str(e)
        print(f"  >>> Generation FAILED: {e}")
        traceback.print_exc()

    return r


def print_report(results):
    print()
    print("=" * 90)
    print("  ACCEPTANCE REPORT")
    print("=" * 90)

    headers = list(results.keys())

    def v(cond):
        return "PASS" if cond else "FAIL"

    checks = [
        ("Generated OK",       [v(results[h]["generated"]) for h in headers]),
        ("No crash",            [v(results[h]["crash"] is None) for h in headers]),
        ("Building count 3-15", [v(3 <= results[h]["building_count"] <= 15) for h in headers]),
        ("Coverage 2-25%",      [v(2.0 <= results[h]["coverage"] <= 25.0) for h in headers]),
        ("No overlap",          [v(results[h]["overlaps"] == 0) for h in headers]),
        ("No too-close",        [v(results[h]["too_close"] == 0) for h in headers]),
        ("No narrow (<4m)",     [v(len(results[h]["narrow"]) == 0) for h in headers]),
        ("Primary exists",      [v(results[h]["has_primary"]) for h in headers]),
        ("GLB export OK",       [v(results[h]["glb_ok"]) for h in headers]),
        ("Style profile OK",    [v(results[h]["style_params_ok"]) for h in headers]),
    ]

    cw = 28
    print(f"{'':32s}", end="")
    for h in headers:
        print(f"{h:>{cw}s}", end="")
    print()
    print("-" * (32 + cw * len(headers)))

    fail_list = []
    for check_name, vals in checks:
        print(f"{check_name:32s}", end="")
        for i, val in enumerate(vals):
            print(f"{val:>{cw}s}", end="")
            if val == "FAIL":
                fail_list.append(f"{headers[i]}: {check_name}")
        print()

    print("-" * (32 + cw * len(headers)))
    # Metrics
    print(f"{'Building count':32s}", end="")
    for h in headers:
        print(f"{results[h]['building_count']:>{cw}d}", end="")
    print()
    print(f"{'Coverage %':32s}", end="")
    for h in headers:
        print(f"{results[h]['coverage']:>{cw}.1f}", end="")
    print()
    print(f"{'Faces':32s}", end="")
    for h in headers:
        print(f"{results[h]['faces']:>{cw},}", end="")
    print()
    print(f"{'GLB size (KB)':32s}", end="")
    for h in headers:
        print(f"{results[h]['glb_kb']:>{cw}.1f}", end="")
    print()

    print()
    total_checks = sum(len(vals) for _, vals in checks)
    total_pass = sum(sum(1 for x in vals if x == "PASS") for _, vals in checks)
    total_fail = total_checks - total_pass
    print(f"  Total: {total_pass}/{total_checks} passed, {total_fail} failed")

    if fail_list:
        print()
        print("  FAILURES:")
        for fd in fail_list:
            print(f"    - {fd}")
    else:
        print("  ALL CHECKS PASSED")
    print()


if __name__ == "__main__":
    all_results = {}
    for sc in SHOWCASES:
        all_results[sc["name"]] = run_showcase(sc)
    print_report(all_results)
