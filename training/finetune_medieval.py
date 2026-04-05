"""
finetune_medieval.py
用 medieval_osm.json（24,064 条 OSM 真实数据）对 medieval 风格专项精调。

策略:
  - OSM 数据 80/20 分成 fine-tune 训练集 / 验证集
  - 混合合成数据（真实训练集 × 3）防止遗忘
  - lr=1e-4, epochs=50, early_stop=10
  - 保存 finetuned_medieval_model.pt
  - 重新生成 medieval_preview.glb
  - 输出20个参数预测值变化对比
"""

import json
import random
import sys
import time
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

SCRIPT_DIR   = Path(__file__).parent
BASE_CKPT    = SCRIPT_DIR / "best_model.pt"
OSM_JSON     = SCRIPT_DIR / "validation_data" / "medieval_osm.json"
OUT_CKPT     = SCRIPT_DIR / "finetuned_medieval_model.pt"
OUT_PARAMS   = SCRIPT_DIR / "finetuned_medieval_style_params.json"
PREVIEW_PATH = SCRIPT_DIR / "medieval_preview.glb"

FINETUNE_LR    = 1e-4
FINETUNE_EPOCHS = 50
EARLY_STOP     = 10
SYNTH_RATIO    = 3      # 合成数据 = 真实训练集 × SYNTH_RATIO
BATCH_SIZE     = 256
TRAIN_SPLIT    = 0.80
RANDOM_SEED    = 42


# ─── OSM 数据 → 归一化参数向量 ─────────────────────────────────

# OSM record 字段 → OUTPUT_KEYS 的直接映射
_OSM_DIRECT = {
    "height_range_min": "height_range_min",
    "height_range_max": "height_range_max",
    "wall_thickness":   "wall_thickness",
    "floor_thickness":  "floor_thickness",
    "door_width":       "door_width",
    "door_height":      "door_height",
    "win_width":        "win_width",
    "win_height":       "win_height",
    "win_density":      "win_density",
    "subdivision":      "subdivision",
    "roof_type":        "roof_type",
    "roof_pitch":       "roof_pitch",
    "has_battlements":  "has_battlements",
    "has_arch":         "has_arch",
    "eave_overhang":    "eave_overhang",
    "column_count":     "column_count",
    "window_shape":     "window_shape",
}

_MED_DEFAULT = get_param_vector("medieval")   # shape (20,), normalized


def osm_to_param_vector(rec: dict) -> np.ndarray:
    """将单条 OSM 建筑记录转换为归一化 20 维参数向量。"""
    y = _MED_DEFAULT.copy()

    # 直接映射字段
    for osm_key, out_key in _OSM_DIRECT.items():
        if osm_key not in rec:
            continue
        idx = OUTPUT_KEYS.index(out_key)
        lo, hi = OUTPUT_PARAMS[out_key]["range"]
        norm = float(np.clip((float(rec[osm_key]) - lo) / (hi - lo + 1e-9), 0.0, 1.0))
        y[idx] = norm

    # wall_color: [r,g,b] 列表字段
    wc = rec.get("wall_color")
    if wc and len(wc) == 3:
        for j, key in enumerate(("wall_color_r", "wall_color_g", "wall_color_b")):
            idx = OUTPUT_KEYS.index(key)
            y[idx] = float(np.clip(wc[j], 0.0, 1.0))

    return y.astype(np.float32)


def build_osm_tensors() -> Tuple[
    np.ndarray, np.ndarray,   # X_train, y_train
    np.ndarray, np.ndarray,   # X_val,   y_val
]:
    """加载 OSM JSON，构建 80/20 训练/验证集。"""
    data = json.loads(OSM_JSON.read_text(encoding="utf-8"))
    buildings = data["buildings"]
    print(f"  OSM 原始: {len(buildings)} 条")

    fv = get_feature_vector("medieval")   # 所有 OSM 样本共享同一特征向量

    X_all, y_all = [], []
    for rec in buildings:
        y_all.append(osm_to_param_vector(rec))
        X_all.append(fv)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)

    # 打乱后分割
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(X_all))
    n_train = int(len(idx) * TRAIN_SPLIT)
    tr, va = idx[:n_train], idx[n_train:]

    print(f"  训练集: {len(tr)}  验证集: {len(va)}")
    return X_all[tr], y_all[tr], X_all[va], y_all[va]


# ─── 评估辅助 ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, criterion, X_t, y_t, device) -> Tuple[float, float]:
    model.eval()
    X_t, y_t = X_t.to(device), y_t.to(device)
    pred = model(X_t)
    loss = criterion(pred, y_t).item()
    mae  = torch.mean(torch.abs(pred - y_t)).item()
    return loss, mae


# ─── 主程序 ────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  finetune_medieval.py  — OSM 数据 medieval 专项精调")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ── 加载基础模型 ──
    if not BASE_CKPT.exists():
        raise FileNotFoundError(f"未找到 {BASE_CKPT}，请先运行 train.py")
    ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
    base_cfg = ckpt.get("config", {})
    model = build_model(FEATURE_DIM, OUTPUT_DIM,
                        base_cfg.get("hidden_dims", [128, 64, 32]),
                        base_cfg.get("dropout", 0.2), str(device))
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[加载] {BASE_CKPT.name}  best_val_loss={base_cfg.get('_best_val_loss','?')}")

    criterion = StyleParamLoss(subdiv_weight=0.1).to(device)

    # ── 构建 OSM 数据集 ──
    print(f"\n[数据] 加载 OSM medieval 数据...")
    X_tr_osm, y_tr_osm, X_val_osm, y_val_osm = build_osm_tensors()

    X_val_t = torch.from_numpy(X_val_osm).to(device)
    y_val_t = torch.from_numpy(y_val_osm).to(device)

    # ── fine-tune 前基线 ──
    pre_loss, pre_mae = evaluate(model, criterion, X_val_t, y_val_t, device)
    synth_base = float(base_cfg.get("_best_val_loss", float("nan")))
    print(f"\n[基线] fine-tune 前 OSM 验证集  Loss={pre_loss:.6f}  MAE={pre_mae:.6f}")
    print(f"       合成验证集 loss (参考):   {synth_base:.6f}")
    print(f"       泛化比: {pre_loss/(synth_base+1e-12):.1f}x")

    # ── 记录 fine-tune 前的 medieval 参数预测值 ──
    with torch.no_grad():
        med_fv_t = torch.from_numpy(get_feature_vector("medieval")).unsqueeze(0).to(device)
        pred_before_norm = model(med_fv_t).squeeze(0).cpu().numpy()
    params_before = denormalize_params(pred_before_norm)

    # ── 生成防遗忘合成数据 ──
    n_synth = len(X_tr_osm) * SYNTH_RATIO
    n_per_style = max(8, n_synth // (len(STYLE_REGISTRY) * 2))
    print(f"\n[数据] 生成防遗忘合成数据 (每风格约 {n_per_style*2} 条)...")
    synth_ds = generate_dataset(
        n_gaussian        = n_per_style,
        n_interp_per_pair = 0,
        n_perturb         = n_per_style,
        val_ratio         = 0.0,
    )
    X_synth = synth_ds["X_train"].astype(np.float32)
    y_synth = synth_ds["y_train"].astype(np.float32)
    print(f"  合成样本: {len(X_synth)}  OSM 训练: {len(X_tr_osm)}")
    print(f"  混合比例: 真实 1 : 合成 {len(X_synth)/len(X_tr_osm):.1f}")

    # ── DataLoader ──
    tr_osm_ds  = TensorDataset(torch.from_numpy(X_tr_osm), torch.from_numpy(y_tr_osm))
    tr_syn_ds  = TensorDataset(torch.from_numpy(X_synth),  torch.from_numpy(y_synth))
    mixed_ds   = ConcatDataset([tr_osm_ds, tr_syn_ds])
    train_loader = DataLoader(mixed_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # ── 训练循环 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6)

    best_loss  = pre_loss
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0

    print(f"\n[训练] lr={FINETUNE_LR}  epochs={FINETUNE_EPOCHS}  batch={BATCH_SIZE}")
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val(OSM)':>10}  {'Val MAE':>9}  {'LR':>8}  {'T':>5}")
    print("─" * 60)

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
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
                  f"{val_mae:>9.6f}  {lr_now:>8.2e}  {elapsed:>4.1f}s")

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP:
            print(f"\n[Early Stop] {EARLY_STOP} epochs 无改善 (epoch {epoch})")
            break

    model.load_state_dict(best_state)

    # ── fine-tune 后评估 ──
    post_loss, post_mae = evaluate(model, criterion, X_val_t, y_val_t, device)

    # ── 20参数对比 ──
    with torch.no_grad():
        pred_after_norm = model(med_fv_t).squeeze(0).cpu().numpy()
    params_after = denormalize_params(pred_after_norm)

    print("\n" + "=" * 65)
    print("  Fine-tune 效果对比")
    print("=" * 65)
    print(f"  合成验证集 loss (参考):         {synth_base:.6f}")
    print(f"  OSM 验证集 loss  前: {pre_loss:.6f}  后: {best_loss:.6f}  "
          f"({'↓' if best_loss < pre_loss else '↑'}{abs(best_loss - pre_loss):.6f})")
    print(f"  OSM 验证集 MAE   前: {pre_mae:.6f}  后: {post_mae:.6f}")
    print(f"  泛化比           前: {pre_loss/(synth_base+1e-12):.1f}x  "
          f"后: {best_loss/(synth_base+1e-12):.1f}x")

    print(f"\n{'─'*65}")
    print(f"  {'参数':<22}  {'fine-tune前':>12}  {'fine-tune后':>12}  {'变化':>10}")
    print(f"{'─'*65}")
    for key in OUTPUT_KEYS:
        before = params_before[key]
        after  = params_after[key]
        delta  = after - before
        flag   = "  ←" if abs(delta) > 0.05 * max(abs(before), 1e-3) else ""
        print(f"  {key:<22}  {before:>12.4f}  {after:>12.4f}  {delta:>+10.4f}{flag}")

    # ── 保存模型 ──
    torch.save({
        "model_state_dict": best_state,
        "config": {**base_cfg,
                   "_medieval_finetune_val_loss": round(best_loss, 8),
                   "_medieval_finetune_val_mae":  round(post_mae,  8)},
        "output_keys":  OUTPUT_KEYS,
        "output_params": OUTPUT_PARAMS,
        "feature_dim":  FEATURE_DIM,
        "output_dim":   OUTPUT_DIM,
    }, OUT_CKPT)
    print(f"\n[保存] {OUT_CKPT.name}")

    # ── 导出全量风格参数 JSON ──
    from train import export_trained_params
    export_trained_params(model, device,
                          {**base_cfg, "_best_val_loss": synth_base},
                          save_path=str(OUT_PARAMS))

    # ── 重新生成 medieval_preview.glb ──
    print(f"\n[生成] 重新生成 {PREVIEW_PATH.name} ...")
    import importlib
    import generate_level as gl
    importlib.reload(gl)   # 确保读到最新代码

    med_params = json.loads(
        Path(OUT_PARAMS).read_text(encoding="utf-8")
    )["styles"]["medieval"]["params"]

    scene = gl.build_scene({"medieval": med_params}, use_style_palette=True)
    n_faces = sum(len(g.faces) for g in scene.geometry.values())
    scene.export(str(PREVIEW_PATH))
    print(f"  => {PREVIEW_PATH}  ({n_faces:,} 面)")

    print("\n完成。")


if __name__ == "__main__":
    main()
