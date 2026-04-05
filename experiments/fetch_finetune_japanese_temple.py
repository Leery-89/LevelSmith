"""
fetch_finetune_japanese_temple.py
用边界框抓取京都寺庙/神社数据，然后直接精调并生成 preview GLB。

bbox: west=135.75, south=34.98, east=135.80, north=35.02（京都市区核心）
也会额外尝试几个小区域避免单次超时。
"""

import json
import sys
import time
import importlib
import math
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

try:
    import osmnx as ox
except ImportError:
    raise SystemExit("pip install osmnx geopandas")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "validation_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 寺庙颜色 ─────────────────────────────────────────────────
TEMPLE_COLOR = {
    "wood":    [0.60, 0.44, 0.28],
    "timber":  [0.58, 0.42, 0.26],
    "tile":    [0.30, 0.30, 0.32],
    "plaster": [0.90, 0.86, 0.78],
    "clay":    [0.72, 0.58, 0.42],
    "stone":   [0.60, 0.58, 0.54],
    "yes":     [0.68, 0.52, 0.36],
}

ROOF_SHAPE_MAP = {
    "flat": 0, "yes": 0, "no": 0,
    "gabled": 1, "saltbox": 1, "gambrel": 1, "skillion": 1,
    "hipped": 2, "half-hipped": 2, "pyramidal": 2, "mansard": 2,
    "dome": 3, "cone": 3, "onion": 3, "round": 3,
}

TEMPLE_BUILDING_TAGS = {"temple", "shrine", "pagoda", "monastery", "chapel"}
TEMPLE_HISTORIC_TAGS = {"temple", "shrine", "building", "castle"}
TEMPLE_AMENITY_TAGS  = {"place_of_worship"}


def _str(val) -> str:
    if val is None:
        return ""
    try:
        if math.isnan(float(val)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def parse_height(val):
    s = _str(val).lower().replace(",", ".")
    m = re.match(r"([\d.]+)\s*(m|ft|'|meters?|feet)?", s)
    if not m:
        return None
    v = float(m.group(1))
    if (m.group(2) or "m").strip("'") in ("ft", "feet"):
        v *= 0.3048
    return v if v > 0 else None


def parse_levels(val):
    try:
        return max(1, int(float(_str(val).split(";")[0].strip())))
    except (ValueError, AttributeError):
        return None


def _get_tags(row) -> dict:
    raw = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    tags = {}
    for k, v in raw.items():
        try:
            tags[k] = None if (isinstance(v, float) and math.isnan(v)) else v
        except TypeError:
            tags[k] = v
    return tags


def temple_filter(tags: dict) -> bool:
    building = _str(tags.get("building")).lower()
    historic = _str(tags.get("historic")).lower()
    amenity  = _str(tags.get("amenity")).lower()
    religion = _str(tags.get("religion")).lower()
    return (building in TEMPLE_BUILDING_TAGS or
            historic in TEMPLE_HISTORIC_TAGS or
            (amenity in TEMPLE_AMENITY_TAGS and religion in ("buddhist", "shinto", "")))


def extract_temple(tags: dict, city: str) -> dict | None:
    h_direct = parse_height(_str(tags.get("height")) or None)
    lv_raw   = parse_levels(_str(tags.get("building:levels")) or None)
    if h_direct is not None:
        h = min(h_direct, 20.0)
    elif lv_raw is not None:
        h = float(lv_raw) * 3.5
    else:
        h = 6.0
    if h < 2.5:
        return None

    building = _str(tags.get("building")).lower()
    lv = lv_raw or max(1, int(h / 3.5))

    # 材质颜色
    wc = TEMPLE_COLOR["yes"]
    for key in ("building:material", "building:facade:material"):
        val = _str(tags.get(key)).lower()
        if val in TEMPLE_COLOR:
            wc = TEMPLE_COLOR[val]
            break

    roof_shape = _str(tags.get("roof:shape")).lower()
    roof_type  = ROOF_SHAPE_MAP.get(roof_shape, 3)   # 默认翘角屋顶
    is_pagoda  = building == "pagoda"

    return {
        "source": "osm", "city": city, "osm_id": str(tags.get("osmid", "")),
        "name":   (_str(tags.get("name")) or "")[:80],
        "building_type": building or "temple",
        "style": "japanese_temple",
        "height_range_min":  0.0,
        "height_range_max":  round(h, 2),
        "wall_thickness":    0.20,
        "floor_thickness":   0.22,
        "door_width":        1.20,
        "door_height":       2.20,
        "win_width":         0.90,
        "win_height":        1.50,
        "win_density":       0.30,
        "subdivision":       min(5, lv),
        "roof_type":         roof_type,
        "roof_pitch":        0.60,
        "wall_color":        [round(c, 3) for c in wc],
        "has_battlements":   0,
        "has_arch":          0,
        "eave_overhang":     0.65,
        "column_count":      4 if is_pagoda else 2,
        "window_shape":      0,
    }


# ─── 边界框抓取 ───────────────────────────────────────────────

# 多个小区域，逐个尝试
BBOX_REGIONS = [
    # (north, south, east, west, label)
    (35.02, 34.98, 135.80, 135.75, "Kyoto_core"),
    (35.04, 34.98, 135.78, 135.74, "Kyoto_north"),   # 金阁寺、仁和寺区域
    (34.98, 34.94, 135.80, 135.76, "Kyoto_south"),   # 伏见稻荷区域
    (35.02, 34.99, 135.74, 135.70, "Kyoto_west"),    # 嵐山区域
    (35.04, 35.01, 135.84, 135.80, "Kyoto_east"),    # 祇园/东山区域
]


def fetch_bbox(north, south, east, west, label):
    print(f"    [{label}] bbox ({south:.2f},{west:.2f}) → ({north:.2f},{east:.2f}) ...", flush=True)
    try:
        # osmnx >= 2.0: bbox=(left, bottom, right, top) = (west, south, east, north)
        gdf = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags={"building": True}
        )
    except TypeError:
        # 旧版 osmnx API: (north, south, east, west)
        try:
            gdf = ox.features_from_bbox(north, south, east, west,
                                         tags={"building": True})
        except Exception as e:
            print(f"      [失败] {e}")
            return []
    except Exception as e:
        print(f"      [失败] {e}")
        return []

    print(f"      原始: {len(gdf)} 条")
    results, skipped = [], 0
    for idx, row in gdf.iterrows():
        tags = _get_tags(row.copy())
        tags["osmid"] = idx[1] if isinstance(idx, tuple) else idx
        if not temple_filter(tags):
            skipped += 1
            continue
        rec = extract_temple(tags, label)
        if rec is None:
            skipped += 1
            continue
        results.append(rec)
    print(f"      有效: {len(results)}  跳过: {skipped}")
    return results


# ─── Fine-tune ────────────────────────────────────────────────

BASE_CKPT   = SCRIPT_DIR / "best_model.pt"
OSM_JSON    = OUTPUT_DIR / "japanese_temple_osm.json"

FINETUNE_LR     = 1e-4
FINETUNE_EPOCHS = 50
EARLY_STOP      = 10
BATCH_SIZE      = 256
TRAIN_SPLIT     = 0.80
RANDOM_SEED     = 42

_OSM_SCALAR_KEYS = [
    "height_range_min", "height_range_max", "wall_thickness", "floor_thickness",
    "door_width", "door_height", "win_width", "win_height", "win_density",
    "subdivision", "roof_type", "roof_pitch",
    "has_battlements", "has_arch", "eave_overhang", "column_count", "window_shape",
]

sys.path.insert(0, str(SCRIPT_DIR))
from model import StyleParamLoss, build_model
from style_registry import (
    STYLE_REGISTRY, OUTPUT_KEYS, OUTPUT_PARAMS, FEATURE_DIM, OUTPUT_DIM,
    get_feature_vector, get_param_vector, denormalize_params,
)
from generate_data import generate_dataset


def osm_to_vec(rec):
    y = get_param_vector("japanese_temple").copy()
    for key in _OSM_SCALAR_KEYS:
        if key not in rec or key not in OUTPUT_KEYS:
            continue
        idx = OUTPUT_KEYS.index(key)
        lo, hi = OUTPUT_PARAMS[key]["range"]
        y[idx] = float(np.clip((float(rec[key]) - lo) / (hi - lo + 1e-9), 0.0, 1.0))
    wc = rec.get("wall_color")
    if wc and len(wc) == 3:
        for j, k in enumerate(("wall_color_r", "wall_color_g", "wall_color_b")):
            y[OUTPUT_KEYS.index(k)] = float(np.clip(wc[j], 0.0, 1.0))
    return y.astype(np.float32)


@torch.no_grad()
def evaluate(model, crit, X_t, y_t, device):
    model.eval()
    X_t, y_t = X_t.to(device), y_t.to(device)
    p = model(X_t)
    return crit(p, y_t).item(), torch.mean(torch.abs(p - y_t)).item()


def finetune_temple(buildings, device, base_ckpt):
    style = "japanese_temple"
    print(f"\n{'='*65}", flush=True)
    print(f"  Fine-tune: JAPANESE_TEMPLE  ({len(buildings)} 条 OSM)", flush=True)
    print(f"{'='*65}", flush=True)

    fv    = get_feature_vector(style)
    X_all = np.array([fv for _ in buildings], dtype=np.float32)
    y_all = np.array([osm_to_vec(r) for r in buildings], dtype=np.float32)

    rng   = np.random.default_rng(RANDOM_SEED)
    idx   = rng.permutation(len(X_all))
    n_tr  = max(1, int(len(idx) * TRAIN_SPLIT))
    tr, va = idx[:n_tr], idx[n_tr:] if len(idx) > n_tr else idx[:1]

    X_val_t = torch.from_numpy(X_all[va]).to(device)
    y_val_t = torch.from_numpy(y_all[va]).to(device)

    base_cfg = base_ckpt.get("config", {})
    model = build_model(FEATURE_DIM, OUTPUT_DIM,
                        base_cfg.get("hidden_dims", [128, 64, 32]),
                        base_cfg.get("dropout", 0.2), str(device))
    model.load_state_dict(base_ckpt["model_state_dict"])

    crit = StyleParamLoss(subdiv_weight=0.1).to(device)
    pre_loss, pre_mae = evaluate(model, crit, X_val_t, y_val_t, device)
    synth_base = float(base_cfg.get("_best_val_loss", float("nan")))
    print(f"[基线] Loss={pre_loss:.6f}  MAE={pre_mae:.6f}  泛化比={pre_loss/(synth_base+1e-12):.1f}x", flush=True)

    with torch.no_grad():
        fv_t = torch.from_numpy(fv).unsqueeze(0).to(device)
        pred_before = denormalize_params(model(fv_t).squeeze(0).cpu().numpy())

    # 合成防遗忘数据
    n_per = max(8, len(tr) * 3 // (len(STYLE_REGISTRY) * 2))
    synth = generate_dataset(n_gaussian=n_per, n_interp_per_pair=0,
                             n_perturb=n_per, val_ratio=0.0)
    X_syn = synth["X_train"].astype(np.float32)
    y_syn = synth["y_train"].astype(np.float32)
    print(f"[数据] OSM训练: {len(tr)}  合成: {len(X_syn)}", flush=True)

    mixed = ConcatDataset([
        TensorDataset(torch.from_numpy(X_all[tr]), torch.from_numpy(y_all[tr])),
        TensorDataset(torch.from_numpy(X_syn),     torch.from_numpy(y_syn)),
    ])
    loader = DataLoader(mixed, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=5,
                                                      factor=0.5, min_lr=1e-6)
    best_loss  = pre_loss
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0

    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'MAE':>8}  {'LR':>8}  {'T':>5}")
    print("─" * 55)
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(Xb)
        tr_loss = total / len(mixed)
        val_loss, val_mae = evaluate(model, crit, X_val_t, y_val_t, device)
        sch.step(val_loss)
        lr_now = opt.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_loss:>10.6f}  {val_loss:>10.6f}  "
                  f"{val_mae:>8.6f}  {lr_now:>8.2e}  {time.time()-t0:>4.1f}s", flush=True)

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP:
            print(f"[Early Stop] epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    post_loss, post_mae = evaluate(model, crit, X_val_t, y_val_t, device)

    with torch.no_grad():
        pred_after = denormalize_params(model(fv_t).squeeze(0).cpu().numpy())

    print(f"\n[结果] Loss 前: {pre_loss:.6f}  后: {best_loss:.6f}  "
          f"({'↓' if best_loss < pre_loss else '↑'}{abs(best_loss-pre_loss):.6f})", flush=True)
    print(f"       MAE  前: {pre_mae:.6f}  后: {post_mae:.6f}", flush=True)

    print(f"\n  {'参数':<22}  {'精调前':>12}  {'精调后':>12}  {'变化':>10}")
    print("─" * 62)
    for key in OUTPUT_KEYS:
        bef, aft = pred_before[key], pred_after[key]
        dlt = aft - bef
        flag = "  <-" if abs(dlt) > 0.05 * max(abs(bef), 1e-3) else ""
        print(f"  {key:<22}  {bef:>12.4f}  {aft:>12.4f}  {dlt:>+10.4f}{flag}")

    # 保存模型
    out_ckpt = SCRIPT_DIR / f"finetuned_{style}_model.pt"
    torch.save({
        "model_state_dict": best_state,
        "config": {**base_cfg,
                   f"_{style}_finetune_val_loss": round(best_loss, 8)},
        "output_keys": OUTPUT_KEYS, "output_params": OUTPUT_PARAMS,
        "feature_dim": FEATURE_DIM, "output_dim": OUTPUT_DIM,
    }, out_ckpt)
    print(f"\n[保存] {out_ckpt.name}", flush=True)

    out_params = SCRIPT_DIR / f"finetuned_{style}_style_params.json"
    from train import export_trained_params
    export_trained_params(model, device,
                          {**base_cfg, "_best_val_loss": synth_base},
                          save_path=str(out_params))

    # 生成 preview GLB
    import generate_level as gl
    importlib.reload(gl)
    params_data = json.loads(out_params.read_text(encoding="utf-8"))
    sp = params_data["styles"].get(style, {}).get("params")
    if sp:
        scene = gl.build_scene({style: sp}, use_style_palette=True)
        n = sum(len(g.faces) for g in scene.geometry.values())
        glb_path = SCRIPT_DIR / f"{style}_preview.glb"
        scene.export(str(glb_path))
        print(f"[GLB]  {glb_path.name}  ({n:,} faces)", flush=True)

    return best_loss, post_mae


# ─── 主程序 ────────────────────────────────────────────────────

def main():
    print("=" * 65, flush=True)
    print("  japanese_temple — bbox 抓取 + 精调 + GLB", flush=True)
    print("=" * 65, flush=True)

    # ── 抓取阶段 ──
    all_records = []
    seen_ids = set()

    for north, south, east, west, label in BBOX_REGIONS:
        recs = fetch_bbox(north, south, east, west, label)
        for r in recs:
            uid = r.get("osm_id", "")
            if uid not in seen_ids:
                seen_ids.add(uid)
                all_records.append(r)
        time.sleep(2.0)   # 礼貌延迟

    print(f"\n总去重有效建筑: {len(all_records)} 条", flush=True)

    if len(all_records) == 0:
        print("[错误] 抓取失败，无有效数据", flush=True)
        return

    # 保存 JSON
    by_city = {}
    for r in all_records:
        by_city.setdefault(r["city"], 0)
        by_city[r["city"]] += 1
    OSM_JSON.write_text(json.dumps({
        "total": len(all_records),
        "cities": list(by_city.keys()),
        "buildings": all_records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已保存: {OSM_JSON.name}", flush=True)
    for c, n in by_city.items():
        print(f"  {c:<20}: {n:4d} 条")
    heights = [r["height_range_max"] for r in all_records]
    print(f"高度: min={min(heights):.1f}m  median={sorted(heights)[len(heights)//2]:.1f}m  max={max(heights):.1f}m")

    # ── 精调阶段 ──
    if not BASE_CKPT.exists():
        print(f"[错误] 未找到 {BASE_CKPT}", flush=True)
        return

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
    print(f"[设备] {device}", flush=True)

    finetune_temple(all_records, device, base_ckpt)


if __name__ == "__main__":
    main()
