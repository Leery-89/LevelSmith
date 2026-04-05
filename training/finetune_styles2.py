"""
finetune_styles2.py
对 fantasy / horror / medieval_chapel / medieval_keep 4 种风格进行 OSM 数据专项精调。

策略:
  - 每种风格从 best_model.pt 独立精调（避免风格间相互污染）
  - OSM 数据 80/20 分训练集/验证集，混合合成数据（真实训练集 × 3）防遗忘
  - lr=1e-4, epochs=50, early_stop=10
  - 保存 finetuned_{style}_model.pt + {style}_preview.glb
  - 对比 fine-tune 前后验证损失
"""

import json
import sys
import time
import importlib
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
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

STYLES_TO_RUN = ["fantasy", "horror", "medieval_chapel", "medieval_keep"]

_OSM_SCALAR_KEYS = [
    "height_range_min", "height_range_max", "wall_thickness", "floor_thickness",
    "door_width", "door_height", "win_width", "win_height", "win_density",
    "subdivision", "roof_type", "roof_pitch",
    "has_battlements", "has_arch", "eave_overhang", "column_count", "window_shape",
]


def osm_to_param_vector(rec: dict, style_name: str) -> np.ndarray:
    y = get_param_vector(style_name).copy()

    for key in _OSM_SCALAR_KEYS:
        if key not in rec or key not in OUTPUT_KEYS:
            continue
        idx = OUTPUT_KEYS.index(key)
        lo, hi = OUTPUT_PARAMS[key]["range"]
        norm = float(np.clip((float(rec[key]) - lo) / (hi - lo + 1e-9), 0.0, 1.0))
        y[idx] = norm

    wc = rec.get("wall_color")
    if wc and len(wc) == 3:
        for j, key in enumerate(("wall_color_r", "wall_color_g", "wall_color_b")):
            idx = OUTPUT_KEYS.index(key)
            y[idx] = float(np.clip(wc[j], 0.0, 1.0))

    return y.astype(np.float32)


def build_osm_tensors(style: str, json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"未找到 {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    buildings = data["buildings"]
    print(f"  OSM 原始: {len(buildings)} 条")

    fv = get_feature_vector(style)

    X_all, y_all = [], []
    for rec in buildings:
        y_all.append(osm_to_param_vector(rec, style))
        X_all.append(fv)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(X_all))
    n_train = int(len(idx) * TRAIN_SPLIT)
    tr, va  = idx[:n_train], idx[n_train:]
    print(f"  训练集: {len(tr)}  验证集: {len(va)}")
    return X_all[tr], y_all[tr], X_all[va], y_all[va]


@torch.no_grad()
def evaluate(model, criterion, X_t, y_t, device) -> Tuple[float, float]:
    model.eval()
    X_t, y_t = X_t.to(device), y_t.to(device)
    pred = model(X_t)
    loss = criterion(pred, y_t).item()
    mae  = torch.mean(torch.abs(pred - y_t)).item()
    return loss, mae


def finetune_one(style: str, device, base_ckpt: dict, criterion) -> dict:
    print(f"\n{'='*65}", flush=True)
    print(f"  Fine-tune: {style.upper()}", flush=True)
    print(f"{'='*65}", flush=True)

    json_path = OSM_DIR / f"{style}_osm.json"
    X_tr, y_tr, X_val, y_val = build_osm_tensors(style, json_path)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    base_cfg = base_ckpt.get("config", {})
    model = build_model(
        FEATURE_DIM, OUTPUT_DIM,
        base_cfg.get("hidden_dims", [128, 64, 32]),
        base_cfg.get("dropout", 0.2), str(device)
    )
    model.load_state_dict(base_ckpt["model_state_dict"])

    pre_loss, pre_mae = evaluate(model, criterion, X_val_t, y_val_t, device)
    synth_base = float(base_cfg.get("_best_val_loss", float("nan")))
    print(f"\n[基线] OSM 验证集  Loss={pre_loss:.6f}  MAE={pre_mae:.6f}", flush=True)
    print(f"       合成验证集 loss:  {synth_base:.6f}  泛化比: {pre_loss/(synth_base+1e-12):.1f}x", flush=True)

    with torch.no_grad():
        fv_t = torch.from_numpy(get_feature_vector(style)).unsqueeze(0).to(device)
        pred_before = denormalize_params(model(fv_t).squeeze(0).cpu().numpy())

    n_synth = len(X_tr) * SYNTH_RATIO
    n_per   = max(8, n_synth // (len(STYLE_REGISTRY) * 2))
    print(f"\n[数据] 生成合成防遗忘数据 (每风格约 {n_per*2} 条)...", flush=True)
    synth_ds = generate_dataset(n_gaussian=n_per, n_interp_per_pair=0,
                                n_perturb=n_per, val_ratio=0.0)
    X_syn = synth_ds["X_train"].astype(np.float32)
    y_syn = synth_ds["y_train"].astype(np.float32)
    print(f"  合成: {len(X_syn)}  OSM 训练: {len(X_tr)}  "
          f"比例 1:{len(X_syn)/max(1,len(X_tr)):.1f}", flush=True)

    mixed_ds = ConcatDataset([
        TensorDataset(torch.from_numpy(X_tr),  torch.from_numpy(y_tr)),
        TensorDataset(torch.from_numpy(X_syn), torch.from_numpy(y_syn)),
    ])
    train_loader = DataLoader(mixed_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6)

    best_loss  = pre_loss
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0

    print(f"\n[训练] lr={FINETUNE_LR}  epochs={FINETUNE_EPOCHS}  batch={BATCH_SIZE}", flush=True)
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val(OSM)':>10}  {'MAE':>8}  {'LR':>8}  {'T':>5}")
    print("─" * 58)

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(Xb)
        train_loss = total / len(mixed_ds)

        val_loss, val_mae = evaluate(model, criterion, X_val_t, y_val_t, device)
        scheduler.step(val_loss)
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
                  f"{val_mae:>8.6f}  {lr_now:>8.2e}  {elapsed:>4.1f}s", flush=True)

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP:
            print(f"\n[Early Stop] epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)

    post_loss, post_mae = evaluate(model, criterion, X_val_t, y_val_t, device)

    with torch.no_grad():
        pred_after = denormalize_params(model(fv_t).squeeze(0).cpu().numpy())

    print(f"\n{'─'*65}", flush=True)
    print(f"  [结果] Loss  前: {pre_loss:.6f}  后: {best_loss:.6f}  "
          f"({'↓' if best_loss < pre_loss else '↑'}{abs(best_loss - pre_loss):.6f})", flush=True)
    print(f"         MAE   前: {pre_mae:.6f}  后: {post_mae:.6f}", flush=True)

    print(f"\n{'─'*65}")
    print(f"  {'参数':<22}  {'精调前':>12}  {'精调后':>12}  {'变化':>10}")
    print(f"{'─'*65}")
    for key in OUTPUT_KEYS:
        bef = pred_before[key]
        aft = pred_after[key]
        dlt = aft - bef
        flag = "  <-" if abs(dlt) > 0.05 * max(abs(bef), 1e-3) else ""
        print(f"  {key:<22}  {bef:>12.4f}  {aft:>12.4f}  {dlt:>+10.4f}{flag}")

    out_ckpt = OUT_DIR / f"finetuned_{style}_model.pt"
    torch.save({
        "model_state_dict": best_state,
        "config": {
            **base_cfg,
            f"_{style}_finetune_val_loss": round(best_loss, 8),
            f"_{style}_finetune_val_mae":  round(post_mae,  8),
        },
        "output_keys":   OUTPUT_KEYS,
        "output_params": OUTPUT_PARAMS,
        "feature_dim":   FEATURE_DIM,
        "output_dim":    OUTPUT_DIM,
    }, out_ckpt)
    print(f"\n[保存] {out_ckpt.name}", flush=True)

    out_params_path = OUT_DIR / f"finetuned_{style}_style_params.json"
    from train import export_trained_params
    export_trained_params(model, device,
                          {**base_cfg, "_best_val_loss": synth_base},
                          save_path=str(out_params_path))

    return {
        "style":       style,
        "state_dict":  best_state,
        "params_path": str(out_params_path),
        "pre_loss":    pre_loss,
        "post_loss":   best_loss,
        "post_mae":    post_mae,
        "synth_base":  synth_base,
    }


def generate_preview(style: str, params_path: str):
    import generate_level as gl
    importlib.reload(gl)

    params_data = json.loads(Path(params_path).read_text(encoding="utf-8"))
    style_params = params_data["styles"].get(style, {}).get("params")
    if style_params is None:
        print(f"  [跳过] {style} 参数未找到")
        return None

    scene = gl.build_scene({style: style_params}, use_style_palette=True)
    n_faces = sum(len(g.faces) for g in scene.geometry.values())
    out_path = OUT_DIR / f"{style}_preview.glb"
    scene.export(str(out_path))
    print(f"  [生成] {out_path.name}  ({n_faces:,} faces)", flush=True)
    return str(out_path)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65, flush=True)
    print("  finetune_styles2.py  —  4 风格 OSM 专项精调", flush=True)
    print("=" * 65, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    if not BASE_CKPT.exists():
        raise FileNotFoundError(f"未找到 {BASE_CKPT}，请先运行 train.py")

    base_ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
    criterion = StyleParamLoss(subdiv_weight=0.1).to(device)

    styles_to_run = []
    print("\n[检查 OSM 文件]", flush=True)
    for style in STYLES_TO_RUN:
        json_path = OSM_DIR / f"{style}_osm.json"
        if json_path.exists():
            data = json.loads(json_path.read_text("utf-8"))
            n = data.get("total", 0)
            if n >= 10:
                print(f"  ok {style:<18}: {n:5d} 条 OSM 数据")
                styles_to_run.append(style)
            else:
                print(f"  X  {style:<18}: {n:5d} 条（不足，跳过）")
        else:
            print(f"  X  {style:<18}: 未找到 {json_path.name}")

    if not styles_to_run:
        raise SystemExit("没有可精调的风格，请先运行 fetch_osm_styles2.py")

    results = []
    for style in styles_to_run:
        result = finetune_one(style, device, base_ckpt, criterion)
        results.append(result)

    print(f"\n{'='*65}", flush=True)
    print("  生成 Preview GLB", flush=True)
    print(f"{'='*65}", flush=True)
    for res in results:
        generate_preview(res["style"], res["params_path"])

    print(f"\n{'='*65}", flush=True)
    print("  精调效果总览", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'风格':<18}  {'OSM条数':>8}  {'前 Loss':>10}  {'后 Loss':>10}  {'降幅':>8}  {'泛化比':>8}")
    print("─" * 72)

    for res in results:
        json_path = OSM_DIR / f"{res['style']}_osm.json"
        n_osm = json.loads(json_path.read_text("utf-8")).get("total", 0)
        drop_pct = (res["pre_loss"] - res["post_loss"]) / (res["pre_loss"] + 1e-12) * 100
        ratio    = res["post_loss"] / (res["synth_base"] + 1e-12)
        print(f"  {res['style']:<18}  {n_osm:>8d}  {res['pre_loss']:>10.6f}  "
              f"{res['post_loss']:>10.6f}  {drop_pct:>7.1f}%  {ratio:>7.1f}x")
    print()


if __name__ == "__main__":
    main()
