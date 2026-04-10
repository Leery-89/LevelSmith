"""
LevelSmith — Placement JSON validator and auto-fixer.

Checks for structural issues (overlaps, out-of-bounds, invalid sizes,
missing primary, oversized doors) and applies safe fixes.

Called after every /edit to ensure the scene is physically consistent
before mesh rebuild.
"""

from __future__ import annotations

import math


def validate_placement(placement: dict) -> dict:
    """Check and auto-fix a placement JSON.

    Returns:
        {
          "valid":         bool   — True if no issues found (before fixes)
          "issues":        list   — problems detected
          "fixes_applied": list   — corrections made
          "building_count": int
        }
    """
    buildings = placement.get("buildings", [])
    scene = placement.get("scene", {})
    area_w = float(scene.get("area_width", 100))
    area_d = float(scene.get("area_depth", 100))

    issues: list[str] = []
    fixes: list[str] = []

    # ── 1. Building overlap detection + push-apart ──────────────────
    for i in range(len(buildings)):
        for j in range(i + 1, len(buildings)):
            bi, bj = buildings[i], buildings[j]
            xi = float(bi["position"]["x"])
            zi = float(bi["position"]["z"])
            xj = float(bj["position"]["x"])
            zj = float(bj["position"]["z"])

            min_dx = (float(bi["width"]) + float(bj["width"])) / 2
            min_dz = (float(bi["depth"]) + float(bj["depth"])) / 2
            dx = abs(xi - xj)
            dz = abs(zi - zj)

            if dx < min_dx and dz < min_dz:
                issues.append(
                    f"Overlap: B{bi.get('id','?')} and B{bj.get('id','?')}")

                push_x = (min_dx - dx + 1) / 2
                push_z = (min_dz - dz + 1) / 2

                # Don't move primary
                if bi.get("role") == "primary":
                    sign_x = 1 if xj > xi else -1
                    sign_z = 1 if zj > zi else -1
                    bj["position"]["x"] = round(xj + sign_x * push_x, 2)
                    bj["position"]["z"] = round(zj + sign_z * push_z, 2)
                elif bj.get("role") == "primary":
                    sign_x = 1 if xi > xj else -1
                    sign_z = 1 if zi > zj else -1
                    bi["position"]["x"] = round(xi + sign_x * push_x, 2)
                    bi["position"]["z"] = round(zi + sign_z * push_z, 2)
                else:
                    bi["position"]["x"] = round(xi - push_x / 2, 2)
                    bj["position"]["x"] = round(xj + push_x / 2, 2)

                fixes.append(
                    f"Pushed apart B{bi.get('id','?')} and B{bj.get('id','?')}")

    # ── 2. Boundary clamping ────────────────────────────────────────
    margin = 2.0
    for b in buildings:
        x = float(b["position"]["x"])
        z = float(b["position"]["z"])
        hw = float(b["width"]) / 2
        hd = float(b["depth"]) / 2
        clamped = False

        if x - hw < margin:
            b["position"]["x"] = round(margin + hw, 2)
            clamped = True
        if x + hw > area_w - margin:
            b["position"]["x"] = round(area_w - margin - hw, 2)
            clamped = True
        if z - hd < margin:
            b["position"]["z"] = round(margin + hd, 2)
            clamped = True
        if z + hd > area_d - margin:
            b["position"]["z"] = round(area_d - margin - hd, 2)
            clamped = True

        if clamped:
            fixes.append(f"Clamped B{b.get('id','?')} inside boundary")

    # ── 3. Dimension clamping ───────────────────────────────────────
    for b in buildings:
        bid = b.get("id", "?")
        for dim, lo, hi in [("width", 3, 30), ("depth", 3, 30), ("height", 2.5, 25)]:
            val = float(b.get(dim, 6))
            if val < lo:
                b[dim] = lo
                fixes.append(f"B{bid} {dim} clamped to min {lo}m")
            elif val > hi:
                b[dim] = hi
                fixes.append(f"B{bid} {dim} clamped to max {hi}m")

    # ── 4. Primary must exist ───────────────────────────────────────
    has_primary = any(b.get("role") == "primary" for b in buildings)
    if not has_primary and buildings:
        largest = max(buildings,
                      key=lambda b: float(b.get("width", 0)) * float(b.get("depth", 0)))
        largest["role"] = "primary"
        fixes.append(f"Promoted B{largest.get('id','?')} to primary (was missing)")

    # ── 5. Door / window sanity ─────────────────────────────────────
    for b in buildings:
        bid = b.get("id", "?")
        bw = float(b.get("width", 8))
        bh = float(b.get("height", 6))

        for door in b.get("doors", []):
            dw = float(door.get("width", 1.2))
            dh = float(door.get("height", 2.2))
            if dw > bw * 0.8:
                door["width"] = round(bw * 0.4, 2)
                fixes.append(f"B{bid} door width reduced")
            if dh > bh * 0.9:
                door["height"] = round(bh * 0.7, 2)
                fixes.append(f"B{bid} door height reduced")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "fixes_applied": fixes,
        "building_count": len(buildings),
    }
