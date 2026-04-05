"""
LevelSmith → UE5 Scene Assembler

Reads a placement JSON and spawns proxy geometry in the UE5 editor.
Each building is a scaled Cube with role-based material color.
Walls, roads, and ground plane are also placed.

Usage (in UE5 Python console or Output Log > Cmd):
    py "D:/PythonProjects/levelsmith/training/ue5_assemble_scene.py"

Or with a specific JSON:
    py "D:/PythonProjects/levelsmith/training/ue5_assemble_scene.py" --json "path/to/placement.json"

Prerequisites:
    - Edit > Plugins > Python Editor Script Plugin (enabled, restart editor)
    - Edit > Project Settings > Python > Additional Paths:
        add: D:/PythonProjects/levelsmith/training

Coordinate mapping (LevelSmith → UE5):
    LevelSmith: Y-up right-hand (X=right, Y=up, Z=depth)
    UE5:        Z-up left-hand  (X=forward, Y=right, Z=up)
    Mapping:    UE_X = LS_X * 100,  UE_Y = LS_Z * 100,  UE_Z = LS_Y * 100
    Scale:      meters → centimeters (x100)
"""

import json
import math
import sys
import os

# ─── Detect environment ──────────────────────────────────────────

try:
    import unreal
    IN_UE = True
except ImportError:
    IN_UE = False

# Safe script directory — works in both standalone Python and UE5 exec()
if "__file__" in dir():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    SCRIPT_DIR = r"D:/PythonProjects/levelsmith/training"


# ─── Configuration ───────────────────────────────────────────────

# Default JSON path — change this or pass --json argument
DEFAULT_JSON = os.path.join(
    SCRIPT_DIR, "medieval_keep_courtyard_placement.json"
)

# Role → color (linear RGB, 0-1)
ROLE_COLORS = {
    "primary":   (0.8, 0.2, 0.2),   # red
    "secondary": (0.2, 0.5, 0.8),   # blue
    "tertiary":  (0.6, 0.6, 0.3),   # olive
    "ambient":   (0.3, 0.7, 0.3),   # green
    "keep":      (0.9, 0.1, 0.1),   # deep red
    "tower":     (0.4, 0.4, 0.8),   # purple-blue
    "main_hall": (0.8, 0.4, 0.1),   # orange
    "warehouse": (0.5, 0.5, 0.5),   # gray
}

WALL_COLOR = (0.4, 0.35, 0.3)      # dark stone
ROAD_COLOR = (0.35, 0.3, 0.25)     # dark brown
GROUND_COLOR = (0.25, 0.35, 0.2)   # dark green
ROOF_COLOR = (0.35, 0.25, 0.2)     # dark brown-red

ROOF_SLAB_THICKNESS = 0.15         # meters — thickness of slope slabs
FLOOR_THICKNESS = 0.2              # meters
WIN_PROTRUSION = 0.05              # meters — how far window cubes stick out


def m_to_cm(m):
    """Meters to centimeters."""
    return m * 100.0


def _rotate_offset(dx_cm, dy_cm, yaw_deg):
    """Rotate a local (dx, dy) offset by yaw degrees around Z.

    Use for converting building-local offsets to world UE coordinates.
    Input and output are in the same units (cm).
    """
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    return dx_cm * c - dy_cm * s, dx_cm * s + dy_cm * c


def _roof_params(b):
    """Compute roof slope geometry from building data.

    Returns dict:
      rise      ridge height above wall top (m)
      angle_w   slope angle in width direction (deg)
      angle_d   slope angle in depth direction (deg)
      slope_w   slope hypotenuse, width direction (m)
      slope_d   slope hypotenuse, depth direction (m)
      eave      eave overhang per side (m)
      half_w    half building width (m)
      half_d    half building depth (m)
      rtype     roof type string
    """
    roof = b["roof"]
    pitch = roof.get("pitch", 0.1)
    eave = roof.get("eave_overhang", 0.0)
    rtype = roof.get("type", "flat")

    half_w = b["width"] / 2
    half_d = b["depth"] / 2
    rise = half_w * pitch    # ridge height = half-width * pitch

    angle_w = math.degrees(math.atan2(rise, half_w)) if rise > 0.01 else 0
    angle_d = math.degrees(math.atan2(rise, half_d)) if rise > 0.01 else 0
    slope_w = math.hypot(half_w, rise)
    slope_d = math.hypot(half_d, rise)

    return {
        "rise": rise, "angle_w": angle_w, "angle_d": angle_d,
        "slope_w": slope_w, "slope_d": slope_d,
        "eave": eave, "half_w": half_w, "half_d": half_d,
        "rtype": rtype,
    }


def _roof_slabs(b):
    """Return list of roof slab descriptors for one building.

    Each slab is a dict with:
      tag       'flat' / 'L' / 'R' / 'F' / 'B'
      dx_cm     local X offset from building center (cm)
      dy_cm     local Y offset from building center (cm)
      z_cm      world Z position (cm)
      sx, sy, sz  UE scale (meters)
      p, r      pitch and roll (degrees), yaw is always building yaw
    """
    rp = _roof_params(b)
    wall_top = m_to_cm(b["height"])
    rise_cm = m_to_cm(rp["rise"])
    half_w_cm = m_to_cm(rp["half_w"])
    half_d_cm = m_to_cm(rp["half_d"])
    eave = rp["eave"]

    # Flat roof: single slab
    if rp["rtype"] == "flat" or rp["rise"] < 0.3:
        h = max(rp["rise"], 0.3)
        return [{
            "tag": "flat",
            "dx_cm": 0, "dy_cm": 0,
            "z_cm": wall_top + m_to_cm(h) / 2,
            "sx": b["width"] + eave * 2,
            "sy": b["depth"] + eave * 2,
            "sz": h,
            "p": 0, "r": 0,
        }]

    slabs = []
    z_mid = wall_top + rise_cm / 2
    t = ROOF_SLAB_THICKNESS

    # Left / Right slopes (Pitch tilt, ridge along Y/depth)
    sx_lr = rp["slope_w"]
    sy_lr = b["depth"] + eave * 2
    slabs.append({
        "tag": "L",
        "dx_cm": -half_w_cm / 2, "dy_cm": 0,
        "z_cm": z_mid,
        "sx": sx_lr, "sy": sy_lr, "sz": t,
        "p": rp["angle_w"], "r": 0,
    })
    slabs.append({
        "tag": "R",
        "dx_cm": half_w_cm / 2, "dy_cm": 0,
        "z_cm": z_mid,
        "sx": sx_lr, "sy": sy_lr, "sz": t,
        "p": -rp["angle_w"], "r": 0,
    })

    # Hipped / pagoda: add Front / Back slopes
    if rp["rtype"] in ("hipped", "hipped_curved", "pagoda"):
        sx_fb = rp["slope_d"]
        sy_fb = b["width"] + eave * 2
        slabs.append({
            "tag": "F",
            "dx_cm": 0, "dy_cm": -half_d_cm / 2,
            "z_cm": z_mid,
            "sx": sx_fb, "sy": sy_fb, "sz": t,
            "p": rp["angle_d"], "r": 0,
            "yaw_extra": 90,   # perpendicular to main slopes
        })
        slabs.append({
            "tag": "B",
            "dx_cm": 0, "dy_cm": half_d_cm / 2,
            "z_cm": z_mid,
            "sx": sx_fb, "sy": sy_fb, "sz": t,
            "p": -rp["angle_d"], "r": 0,
            "yaw_extra": 90,
        })

    return slabs


def _building_parts(b):
    """Decompose one building into wall segments + floor + windows.

    Returns list of dicts, each with:
      tag      label ('wall_F_L', 'lintel', 'wall_B', 'floor', 'win_B0', ...)
      dx_cm    local X offset from building center (cm, before yaw)
      dy_cm    local Y offset from building center (cm, before yaw)
      z_cm     absolute Z position of part center (cm)
      sx,sy,sz UE scale (meters)

    Building local frame (before yaw rotation):
        Width along X  (-half_w .. +half_w)
        Depth along Y  (-half_d .. +half_d)
        Front face at Y = -half_d  (door wall)
    """
    w = b["width"]
    d = b["depth"]
    h = b["height"]
    t = b["features"]["wall_thickness"]
    half_w_cm = m_to_cm(w / 2)
    half_d_cm = m_to_cm(d / 2)
    h_cm = m_to_cm(h)
    t_cm = m_to_cm(t)

    door = b["doors"][0] if b.get("doors") else None
    door_w = door["width"] if door else 0
    door_h = door["height"] if door else 0

    parts = []

    # ── Floor slab ──
    parts.append({
        "tag": "floor",
        "dx_cm": 0, "dy_cm": 0,
        "z_cm": m_to_cm(FLOOR_THICKNESS) / 2,
        "sx": w, "sy": d, "sz": FLOOR_THICKNESS,
    })

    # ── Front wall (Y- face) — split around door ──
    front_dy = -half_d_cm + t_cm / 2

    if door and door_w > 0.1:
        seg_w = (w - door_w) / 2          # each side segment width (m)
        if seg_w > 0.05:
            # Left of door
            parts.append({
                "tag": "wall_F_L",
                "dx_cm": -m_to_cm((w + door_w) / 4),
                "dy_cm": front_dy,
                "z_cm": h_cm / 2,
                "sx": seg_w, "sy": t, "sz": h,
            })
            # Right of door
            parts.append({
                "tag": "wall_F_R",
                "dx_cm": m_to_cm((w + door_w) / 4),
                "dy_cm": front_dy,
                "z_cm": h_cm / 2,
                "sx": seg_w, "sy": t, "sz": h,
            })
        # Lintel above door
        lintel_h = h - door_h
        if lintel_h > 0.1:
            parts.append({
                "tag": "lintel",
                "dx_cm": 0, "dy_cm": front_dy,
                "z_cm": m_to_cm(door_h) + m_to_cm(lintel_h) / 2,
                "sx": door_w, "sy": t, "sz": lintel_h,
            })
    else:
        # No door — solid front wall
        parts.append({
            "tag": "wall_F",
            "dx_cm": 0, "dy_cm": front_dy,
            "z_cm": h_cm / 2, "sx": w, "sy": t, "sz": h,
        })

    # ── Back wall (Y+ face) ──
    parts.append({
        "tag": "wall_B",
        "dx_cm": 0, "dy_cm": half_d_cm - t_cm / 2,
        "z_cm": h_cm / 2, "sx": w, "sy": t, "sz": h,
    })

    # ── Left wall (X- face) ──
    parts.append({
        "tag": "wall_L",
        "dx_cm": -half_w_cm + t_cm / 2, "dy_cm": 0,
        "z_cm": h_cm / 2, "sx": t, "sy": d, "sz": h,
    })

    # ── Right wall (X+ face) ──
    parts.append({
        "tag": "wall_R",
        "dx_cm": half_w_cm - t_cm / 2, "dy_cm": 0,
        "z_cm": h_cm / 2, "sx": t, "sy": d, "sz": h,
    })

    # ── Windows (surface-mounted thin cubes) ──
    win_data = b["windows"][0] if b.get("windows") else None
    if win_data and win_data.get("density", 0) > 0:
        ww = win_data["width"]
        wh = win_data["height"]
        dens = win_data["density"]
        win_z = h_cm * 0.6       # 60% up the wall
        p = m_to_cm(WIN_PROTRUSION) / 2  # half protrusion in cm

        # Back wall windows (along X)
        n = max(1, round(w * dens * 3))
        for i in range(n):
            parts.append({
                "tag": f"win_B{i}",
                "dx_cm": -half_w_cm + (i + 0.5) / n * m_to_cm(w),
                "dy_cm": half_d_cm + p,
                "z_cm": win_z,
                "sx": ww, "sy": WIN_PROTRUSION, "sz": wh,
            })

        # Left wall windows (along Y)
        n_s = max(1, round(d * dens * 3))
        for i in range(n_s):
            parts.append({
                "tag": f"win_L{i}",
                "dx_cm": -(half_w_cm + p),
                "dy_cm": -half_d_cm + (i + 0.5) / n_s * m_to_cm(d),
                "z_cm": win_z,
                "sx": WIN_PROTRUSION, "sy": ww, "sz": wh,
            })

        # Right wall windows (along Y)
        for i in range(n_s):
            parts.append({
                "tag": f"win_R{i}",
                "dx_cm": half_w_cm + p,
                "dy_cm": -half_d_cm + (i + 0.5) / n_s * m_to_cm(d),
                "z_cm": win_z,
                "sx": WIN_PROTRUSION, "sy": ww, "sz": wh,
            })

    return parts


def _split_walls_at_gates(walls, gates):
    """Split enclosure wall segments at gate positions, leaving gaps.

    Each gate's wall_side determines which wall it cuts.
    Returns a new list of wall dicts — potentially more segments than
    the original (one wall split into left+right around each gate).
    """
    if not gates:
        return walls

    # Group gates by wall side
    by_side = {}
    for g in gates:
        by_side.setdefault(g.get("wall_side", ""), []).append(g)

    result = []
    for w in walls:
        side_gates = by_side.get(w["side"], [])
        if not side_gates:
            result.append(w)
            continue

        sx, sz = w["start"]["x"], w["start"]["z"]
        ex, ez = w["end"]["x"], w["end"]["z"]

        if abs(ez - sz) < 0.01:
            # Horizontal wall (south/north): runs along X
            lo, hi, fixed = min(sx, ex), max(sx, ex), sz
            cuts = sorted(side_gates, key=lambda g: g["position"]["x"])
            cur = lo
            for g in cuts:
                gap_lo = g["position"]["x"] - g["width"] / 2
                gap_hi = g["position"]["x"] + g["width"] / 2
                if gap_lo > cur + 0.1:
                    result.append({**w,
                        "start": {"x": round(cur, 1), "z": round(fixed, 1)},
                        "end":   {"x": round(gap_lo, 1), "z": round(fixed, 1)}})
                cur = gap_hi
            if hi > cur + 0.1:
                result.append({**w,
                    "start": {"x": round(cur, 1), "z": round(fixed, 1)},
                    "end":   {"x": round(hi, 1), "z": round(fixed, 1)}})

        elif abs(ex - sx) < 0.01:
            # Vertical wall (east/west): runs along Z
            lo, hi, fixed = min(sz, ez), max(sz, ez), sx
            cuts = sorted(side_gates, key=lambda g: g["position"]["z"])
            cur = lo
            for g in cuts:
                gap_lo = g["position"]["z"] - g["width"] / 2
                gap_hi = g["position"]["z"] + g["width"] / 2
                if gap_lo > cur + 0.1:
                    result.append({**w,
                        "start": {"x": round(fixed, 1), "z": round(cur, 1)},
                        "end":   {"x": round(fixed, 1), "z": round(gap_lo, 1)}})
                cur = gap_hi
            if hi > cur + 0.1:
                result.append({**w,
                    "start": {"x": round(fixed, 1), "z": round(cur, 1)},
                    "end":   {"x": round(fixed, 1), "z": round(hi, 1)}})
        else:
            result.append(w)   # non-axis-aligned: pass through

    return result


# ─── UE5 Assembly (runs inside Unreal Editor) ────────────────────

def assemble_in_ue(data):
    """Spawn proxy geometry in UE5 from placement JSON."""
    cube_path = "/Engine/BasicShapes/Cube.Cube"
    cube_mesh = unreal.load_asset(cube_path)
    if not cube_mesh:
        unreal.log_error("Could not load Cube mesh")
        return

    folder_name = f"LevelSmith_{data['scene']['style_key']}"

    # ── Ground plane ──
    aw = m_to_cm(data["scene"]["area_width"])
    ad = m_to_cm(data["scene"]["area_depth"])
    ground_loc = unreal.Vector(aw / 2, ad / 2, -15.0)
    ground = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor, ground_loc)
    ground.static_mesh_component.set_static_mesh(cube_mesh)
    ground.set_actor_scale3d(unreal.Vector(aw / 100, ad / 100, 0.3))
    ground.set_actor_label(f"{folder_name}/ground")
    ground.set_folder_path(folder_name)
    unreal.log(f"Ground: {aw:.0f}x{ad:.0f}cm")

    # ── Buildings (walls + floor + windows + roofs) ──
    for b in data["buildings"]:
        px = m_to_cm(b["position"]["x"])
        py = m_to_cm(b["position"]["z"])
        yaw = b.get("rotation_deg", 0)
        bid = b["id"]
        role = b.get("role", "unknown")
        bld_folder = f"{folder_name}/buildings/B{bid}_{role}"

        # Structural parts: walls, floor, windows
        for part in _building_parts(b):
            wx, wy = _rotate_offset(part["dx_cm"], part["dy_cm"], yaw)
            loc = unreal.Vector(px + wx, py + wy, part["z_cm"])
            actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor, loc)
            actor.static_mesh_component.set_static_mesh(cube_mesh)
            actor.set_actor_scale3d(
                unreal.Vector(part["sx"], part["sy"], part["sz"]))
            actor.set_actor_rotation(
                unreal.Rotator(pitch=0, yaw=yaw, roll=0), False)
            actor.set_actor_label(
                f"{folder_name}/B{bid}_{part['tag']}")
            actor.set_folder_path(bld_folder)
            actor.tags.append(role)

        # Roof slabs
        for slab in _roof_slabs(b):
            s_yaw = yaw + slab.get("yaw_extra", 0)
            wx, wy = _rotate_offset(slab["dx_cm"], slab["dy_cm"], yaw)
            r_loc = unreal.Vector(px + wx, py + wy, slab["z_cm"])
            r_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.StaticMeshActor, r_loc)
            r_actor.static_mesh_component.set_static_mesh(cube_mesh)
            r_actor.set_actor_scale3d(
                unreal.Vector(slab["sx"], slab["sy"], slab["sz"]))
            r_actor.set_actor_rotation(
                unreal.Rotator(pitch=slab["p"], yaw=s_yaw,
                               roll=slab["r"]), False)
            r_actor.set_actor_label(
                f"{folder_name}/B{bid}_roof_{slab['tag']}")
            r_actor.set_folder_path(bld_folder)

    unreal.log(f"Placed {len(data['buildings'])} buildings")

    # ── Enclosure walls (split at gate positions) ──
    enc_walls = _split_walls_at_gates(
        data.get("walls", []), data.get("gates", []))
    for w in enc_walls:
        ls_sx, ls_sz = w["start"]["x"], w["start"]["z"]
        ls_ex, ls_ez = w["end"]["x"], w["end"]["z"]
        ue_x = m_to_cm((ls_sx + ls_ex) / 2)
        ue_y = m_to_cm((ls_sz + ls_ez) / 2)
        ue_z = m_to_cm(w["height"]) / 2
        dx = ls_ex - ls_sx
        dz = ls_ez - ls_sz
        length_m = math.hypot(dx, dz)
        yaw = math.degrees(math.atan2(dz, dx))
        loc = unreal.Vector(ue_x, ue_y, ue_z)
        wall = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.StaticMeshActor, loc)
        wall.static_mesh_component.set_static_mesh(cube_mesh)
        wall.set_actor_scale3d(unreal.Vector(
            length_m, w["thickness"], w["height"]))
        wall.set_actor_rotation(
            unreal.Rotator(pitch=0, yaw=yaw, roll=0), False)
        wall.set_actor_label(f"{folder_name}/wall_{w['side']}")
        wall.set_folder_path(f"{folder_name}/walls")
    if enc_walls:
        unreal.log(f"Placed {len(enc_walls)} wall segments "
                   f"({len(data.get('gates', []))} gates)")

    unreal.log(f"=== LevelSmith scene assembled: {folder_name} ===")


# ─── Dry-run mode (outside UE5) ─────────────────────────────────

def dry_run(data):
    """Print placement commands without UE5 — for verification.

    Prints UE coordinates (x, y, z), rotation (pitch, yaw, roll),
    and scale (sx, sy, sz) for every actor.  Checks that all building
    rotations have pitch=0 and roll=0, and that wall segments form
    a closed rectangle.
    """
    scene = data["scene"]
    print(f"\n{'='*70}")
    print(f"  LevelSmith Scene Assembly — Dry Run")
    print(f"  Style: {scene['style_key']}  Layout: {scene['layout_type']}")
    print(f"  Area: {scene['area_width']}x{scene['area_depth']}m"
          f" ({m_to_cm(scene['area_width']):.0f}x"
          f"{m_to_cm(scene['area_depth']):.0f}cm)")
    print(f"{'='*70}")

    print(f"\n  Ground: {m_to_cm(scene['area_width']):.0f}x"
          f"{m_to_cm(scene['area_depth']):.0f}cm at Z=-15cm"
          f"  rot=(0, 0, 0)")

    # ── Buildings ──
    print(f"\n  Buildings ({len(data['buildings'])}):")
    for b in data["buildings"]:
        px = m_to_cm(b["position"]["x"])
        py = m_to_cm(b["position"]["z"])
        yaw = b.get("rotation_deg", 0)
        role = b.get("role", "?")
        bid = b["id"]

        # Header
        door = b["doors"][0] if b.get("doors") else None
        door_str = (f"door {door['width']:.1f}x{door['height']:.1f}m"
                    if door else "no door")
        print(f"\n    [{bid:2d}] {role:12s}  {b['width']:.1f}x{b['depth']:.1f}"
              f"x{b['height']:.1f}m  yaw={yaw:.1f}  "
              f"roof={b['roof']['type']}  {door_str}  "
              f"t={b['features']['wall_thickness']}m")

        # Structural parts
        all_parts = _building_parts(b)
        struct = [p for p in all_parts if not p["tag"].startswith("win")]
        wins = [p for p in all_parts if p["tag"].startswith("win")]

        for p in struct:
            wx, wy = _rotate_offset(p["dx_cm"], p["dy_cm"], yaw)
            print(f"          {p['tag']:12s}  "
                  f"({px+wx:8.0f}, {py+wy:8.0f}, {p['z_cm']:7.0f})  "
                  f"({p['sx']:5.2f}, {p['sy']:5.2f}, {p['sz']:5.2f})")

        # Window summary
        if wins:
            n_b = sum(1 for p in wins if p["tag"].startswith("win_B"))
            n_l = sum(1 for p in wins if p["tag"].startswith("win_L"))
            n_r = sum(1 for p in wins if p["tag"].startswith("win_R"))
            print(f"          + {len(wins)} windows "
                  f"(back={n_b}, left={n_l}, right={n_r})")

        # Roof slabs
        for slab in _roof_slabs(b):
            s_yaw = yaw + slab.get("yaw_extra", 0)
            wx, wy = _rotate_offset(slab["dx_cm"], slab["dy_cm"], yaw)
            print(f"          {'roof_'+slab['tag']:12s}  "
                  f"({px+wx:8.0f}, {py+wy:8.0f}, {slab['z_cm']:7.0f})  "
                  f"({slab['p']:3.0f}, {s_yaw:6.1f}, {slab['r']:3.0f})  "
                  f"({slab['sx']:5.2f}, {slab['sy']:5.2f}, {slab['sz']:5.2f})")

    # ── Gates ──
    raw_gates = data.get("gates", [])
    if raw_gates:
        print(f"\n  Gates ({len(raw_gates)}):")
        for g in raw_gates:
            print(f"    gate_{g['id']} on {g.get('wall_side','?')}  "
                  f"pos=({g['position']['x']:.1f}, {g['position']['z']:.1f})  "
                  f"width={g['width']}m")

    # ── Enclosure walls (split at gates) ──
    enc_walls = _split_walls_at_gates(
        data.get("walls", []), raw_gates)
    if enc_walls:
        n_orig = len(data.get("walls", []))
        extra = f" (split from {n_orig})" if len(enc_walls) != n_orig else ""
        print(f"\n  Enclosure wall segments ({len(enc_walls)}){extra}:")
        print(f"    {'Side':6s}  {'len(m)':>6s}  {'h(m)':>4s}  {'t(m)':>4s}  "
              f"{'UE pos (x,y,z) cm':>28s}  {'rot (p,y,r)':>16s}  "
              f"{'scale (x,y,z)':>18s}")
        print(f"    {'-'*6}  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*28}  {'-'*16}  {'-'*18}")
        for w in enc_walls:
            dx = w["end"]["x"] - w["start"]["x"]
            dz = w["end"]["z"] - w["start"]["z"]
            length_m = math.hypot(dx, dz)
            ue_x = m_to_cm((w["start"]["x"] + w["end"]["x"]) / 2)
            ue_y = m_to_cm((w["start"]["z"] + w["end"]["z"]) / 2)
            ue_z = m_to_cm(w["height"]) / 2
            yaw = math.degrees(math.atan2(dz, dx))
            print(f"    {w['side']:6s}  {length_m:6.1f}  {w['height']:4.1f}  "
                  f"{w['thickness']:4.1f}  "
                  f"({ue_x:8.0f}, {ue_y:8.0f}, {ue_z:7.0f})  "
                  f"(  0, {yaw:6.1f},   0)  "
                  f"({length_m:5.1f}, {w['thickness']:5.1f}, {w['height']:5.1f})")

    print(f"\n  Roads: {len(data.get('roads', []))} segments")
    print(f"  Ground material hint: {data['ground']['material_hint']}")
    print(f"\n  Coverage: {data['metadata']['coverage_ratio']*100:.1f}%")
    print(f"  Roles: {data['metadata']['role_distribution']}")
    print()


# ─── Main ────────────────────────────────────────────────────────

def _resolve_json_path():
    """Find placement JSON path from args or fallback to default."""
    # sys.argv may not exist in UE5 exec() context
    argv = getattr(sys, "argv", [])
    for i, arg in enumerate(argv):
        if arg == "--json" and i + 1 < len(argv):
            return argv[i + 1]
    return DEFAULT_JSON


def main(json_path=None):
    """
    Entry point. Call with json_path to override, or None to auto-detect.

    In UE5 console you can also do:
        import ue5_assemble_scene
        ue5_assemble_scene.main(r"D:/path/to/placement.json")
    """
    if json_path is None:
        json_path = _resolve_json_path()

    if not os.path.exists(json_path):
        msg = f"Placement JSON not found: {json_path}"
        if IN_UE:
            unreal.log_error(msg)
        else:
            print(f"ERROR: {msg}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if IN_UE:
        assemble_in_ue(data)
    else:
        dry_run(data)


# Run on import/exec — works for both `python script.py` and UE5 `py "script.py"`
main()
