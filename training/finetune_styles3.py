"""
finetune_styles3.py
对剩余 11 种风格进行 OSM 数据专项精调：
modern_loft / modern_villa / industrial_workshop / industrial_powerplant /
fantasy_dungeon / fantasy_palace / horror_asylum / horror_crypt /
japanese_temple / japanese_machiya / desert_palace
"""

import json
import sys
import time
import importlib
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent))

from model import StyleParamLoss, build_model
from style_registry import (
    STYLE_REGISTRY, OUTPUT_KEYS, OUTPUT_PARAMS, FEATURE_DIM, OUTPUT_DIM,
    get_feature_vector, get_param_vector, denormalize_params,
)
from generate_data import generate_dataset

SCRIPT_DIR      = Path(__file__).parent
BASE_CKPT       = SCRIPT_DIR / "best_model.pt"
OSM_DIR         = SCRIPT_DIR / "validation_data"
OUT_DIR         = SCRIPT_DIR

FINETUNE_LR     = 1e-4
FINETUNE_EPOCHS = 50
EARLY_STOP      = 10
SYNTH_RATIO     = 3
BATCH_SIZE      = 256
TRAIN_SPLIT     = 0.80
RANDOM_SEED     = 42

STYLES_TO_RUN = [
    "modern_loft", "modern_villa",
    "industrial_workshop", "industrial_powerplant",
    "fantasy_dungeon", "fantasy_palace",
    "horror_asylum", "horror_crypt",
    "japanese_temple", "japanese_machiya",
    "desert_palace",
]

_OSM_SCALAR_KEYS = [
    "height_range_min", "height_range_max", "wall_thickness", "floor_thickness",
    "door_width", "door_height", "win_width", "win_height", "win_density",
    "subdivision", "roof_type", "roof_pitch",
    "has_battlements", "has_arch", "eave_overhang", "column_count", "window_shape",
]


def osm_to_param_vector(rec, style_name):
    y = get_param_vector(style_name).copy()
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


def build_osm_tensors(style, json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    buildings = data["buildings"]
    print(f"  OSM 原始: {len(buildings)} 条")
    fv = get_feature_vector(style)
    X_all = np.array([fv for _ in buildings], dtype=np.float32)
    y_all = np.array([osm_to_param_vector(r, style) for r in buildings], dtype=np.float32)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(X_all))
    n_tr = int(len(idx) * TRAIN_SPLIT)
    tr, va = idx[:n_tr], idx[n_tr:]
    print(f"  训练集: {len(tr)}  验证集: {len(va)}")
    return X_all[tr], y_all[tr], X_all[va], y_all[va]


@torch.no_grad()
def evaluate(model, criterion, X_t, y_t, device):
    model.eval()
    X_t, y_t = X_t.to(device), y_t.to(device)
    pred = model(X_t)
    return criterion(pred, y_t).item(), torch.mean(torch.abs(pred - y_t)).item()


def finetune_one(style, device, base_ckpt, criterion):
    print(f"\n{'='*65}", flush=True)
    print(f"  Fine-tune: {style.upper()}", flush=True)
    print(f"{'='*65}", flush=True)

    json_path = OSM_DIR / f"{style}_osm.json"
    X_tr, y_tr, X_val, y_val = build_osm_tensors(style, json_path)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    base_cfg = base_ckpt.get("config", {})
    model = build_model(FEATURE_DIM, OUTPUT_DIM,
                        base_cfg.get("hidden_dims", [128, 64, 32]),
                        base_cfg.get("dropout", 0.2), str(device))
    model.load_state_dict(base_ckpt["model_state_dict"])

    pre_loss, pre_mae = evaluate(model, criterion, X_val_t, y_val_t, device)
    synth_base = float(base_cfg.get("_best_val_loss", float("nan")))
    print(f"[基线] Loss={pre_loss:.6f}  MAE={pre_mae:.6f}  泛化比={pre_loss/(synth_base+1e-12):.1f}x", flush=True)

    with torch.no_grad():
        fv_t = torch.from_numpy(get_feature_vector(style)).unsqueeze(0).to(device)
        pred_before = denormalize_params(model(fv_t).squeeze(0).cpu().numpy())

    # 合成防遗忘数据
    n_synth = max(len(X_tr) * SYNTH_RATIO, 300)
    n_per   = max(8, n_synth // (len(STYLE_REGISTRY) * 2))
    synth_ds = generate_dataset(n_gaussian=n_per, n_interp_per_pair=0,
                                n_perturb=n_per, val_ratio=0.0)
    X_syn = synth_ds["X_train"].astype(np.float32)
    y_syn = synth_ds["y_train"].astype(np.float32)
    print(f"[数据] 合成: {len(X_syn)}  OSM训练: {len(X_tr)}  比例 1:{len(X_syn)/max(1,len(X_tr)):.1f}", flush=True)

    mixed_ds = ConcatDataset([
        TensorDataset(torch.from_numpy(X_tr),  torch.from_numpy(y_tr)),
        TensorDataset(torch.from_numpy(X_syn), torch.from_numpy(y_syn)),
    ])
    loader = DataLoader(mixed_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5, min_lr=1e-6)

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
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(Xb)
        tr_loss = total / len(mixed_ds)
        val_loss, val_mae = evaluate(model, criterion, X_val_t, y_val_t, device)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

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
    post_loss, post_mae = evaluate(model, criterion, X_val_t, y_val_t, device)

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
    out_ckpt = OUT_DIR / f"finetuned_{style}_model.pt"
    torch.save({
        "model_state_dict": best_state,
        "config": {**base_cfg,
                   f"_{style}_finetune_val_loss": round(best_loss, 8),
                   f"_{style}_finetune_val_mae":  round(post_mae,  8)},
        "output_keys": OUTPUT_KEYS, "output_params": OUTPUT_PARAMS,
        "feature_dim": FEATURE_DIM, "output_dim": OUTPUT_DIM,
    }, out_ckpt)
    print(f"\n[保存] {out_ckpt.name}", flush=True)

    out_params = OUT_DIR / f"finetuned_{style}_style_params.json"
    from train import export_trained_params
    export_trained_params(model, device,
                          {**base_cfg, "_best_val_loss": synth_base},
                          save_path=str(out_params))
    return {
        "style": style, "params_path": str(out_params),
        "pre_loss": pre_loss, "post_loss": best_loss,
        "post_mae": post_mae, "synth_base": synth_base,
    }


def generate_preview(style, params_path):
    import generate_level as gl
    importlib.reload(gl)
    data = json.loads(Path(params_path).read_text(encoding="utf-8"))
    sp   = data["styles"].get(style, {}).get("params")
    if sp is None:
        print(f"  [跳过] {style} 参数未找到")
        return None
    scene = gl.build_scene({style: sp}, use_style_palette=True)
    n = sum(len(g.faces) for g in scene.geometry.values())
    out = OUT_DIR / f"{style}_preview.glb"
    scene.export(str(out))
    print(f"  [GLB] {out.name}  ({n:,} faces)", flush=True)
    return str(out)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65, flush=True)
    print("  finetune_styles3.py  —  11 风格 OSM 专项精调", flush=True)
    print("=" * 65, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    if not BASE_CKPT.exists():
        raise FileNotFoundError(f"未找到 {BASE_CKPT}")

    base_ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
    criterion = StyleParamLoss(subdiv_weight=0.1).to(device)

    print("\n[检查 OSM 文件]", flush=True)
    styles_to_run = []
    for style in STYLES_TO_RUN:
        jp = OSM_DIR / f"{style}_osm.json"
        if jp.exists():
            n = json.loads(jp.read_text("utf-8")).get("total", 0)
            status = "ok" if n >= 10 else "X "
            print(f"  {status}  {style:<22}: {n:6d} 条")
            if n >= 10:
                styles_to_run.append(style)
        else:
            print(f"  X   {style:<22}: 未找到 {jp.name}")

    if not styles_to_run:
        raise SystemExit("没有可精调的风格，请先运行 fetch_osm_styles3.py")

    results = []
    for style in styles_to_run:
        results.append(finetune_one(style, device, base_ckpt, criterion))

    print(f"\n{'='*65}", flush=True)
    print("  生成 Preview GLB", flush=True)
    print(f"{'='*65}", flush=True)
    for res in results:
        generate_preview(res["style"], res["params_path"])

    print(f"\n{'='*65}", flush=True)
    print("  精调效果总览", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'风格':<22}  {'OSM条数':>8}  {'前Loss':>10}  {'后Loss':>10}  {'降幅':>8}")
    print("─" * 72)
    for res in results:
        n_osm = json.loads((OSM_DIR/f"{res['style']}_osm.json").read_text("utf-8")).get("total", 0)
        drop = (res["pre_loss"] - res["post_loss"]) / (res["pre_loss"] + 1e-12) * 100
        print(f"  {res['style']:<22}  {n_osm:>8d}  {res['pre_loss']:>10.6f}  "
              f"{res['post_loss']:>10.6f}  {drop:>7.1f}%")
    print()


if __name__ == "__main__":
    main()
