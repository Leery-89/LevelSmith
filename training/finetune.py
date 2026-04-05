"""
finetune.py
用真实 GLB 验证数据对已训练模型进行 fine-tune。

策略:
  - 加载 best_model.pt
  - 真实数据 (32条) + 少量合成数据 (防止灾难性遗忘) 混合训练
  - lr=1e-4, epochs=30, early_stop patience=10
  - 训练集 = 全部真实数据 + 10x 合成数据
  - 验证集 = 全部真实数据 (无留出; 报告训练集拟合情况)
  - 对比 fine-tune 前后真实验证集 Loss
"""

import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent))

from model import StyleParamMLP, StyleParamLoss, build_model
from style_registry import (
    STYLE_REGISTRY, OUTPUT_KEYS, OUTPUT_PARAMS, FEATURE_DIM, OUTPUT_DIM,
    get_feature_vector, get_param_vector,
)
from generate_data import generate_dataset

SCRIPT_DIR    = Path(__file__).parent
CKPT_PATH     = SCRIPT_DIR / "best_model.pt"
REAL_JSON     = SCRIPT_DIR / "validation_data" / "extracted_params.json"
OUT_CKPT      = SCRIPT_DIR / "finetuned_model.pt"
OUT_PARAMS    = SCRIPT_DIR / "finetuned_style_params.json"

FINETUNE_LR      = 1e-4
FINETUNE_EPOCHS  = 30
EARLY_STOP       = 10
SYNTH_MULTIPLIER = 10   # 真实样本数 × 10 = 合成样本数（防遗忘）

# ─── 真实数据构建（与 train.py 保持一致） ──────────────────────

_REAL_PARAM_MAP = {
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
}


def build_real_tensors(json_path: Path) -> Tuple[np.ndarray, np.ndarray, list]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    X_list, y_list, labels = [], [], []
    for m in data.get("models", []):
        style = m.get("style", "")
        if style not in STYLE_REGISTRY:
            continue
        fv       = get_feature_vector(style)
        y_row    = get_param_vector(style).copy()
        for ext_key, out_key in _REAL_PARAM_MAP.items():
            if ext_key not in m:
                continue
            idx = OUTPUT_KEYS.index(out_key)
            lo, hi = OUTPUT_PARAMS[out_key]["range"]
            norm_val = float(np.clip((float(m[ext_key]) - lo) / (hi - lo + 1e-9), 0.0, 1.0))
            y_row[idx] = norm_val
        X_list.append(fv)
        y_list.append(y_row)
        labels.append(f"{style}/{m.get('file','?')}")

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            labels)


# ─── 评估辅助 ──────────────────────────────────────────────────

@torch.no_grad()
def eval_real(model, criterion, X_t, y_t, labels, header="") -> Tuple[float, float]:
    model.eval()
    pred = model(X_t)
    loss = criterion(pred, y_t).item()
    mae  = torch.mean(torch.abs(pred - y_t)).item()
    if header:
        print(f"\n{'─'*60}")
        print(f"  {header}  ({len(labels)} 个真实模型)")
        print(f"{'─'*60}")
        per = torch.mean(torch.abs(pred - y_t), dim=1).cpu().numpy()
        for lbl, m in sorted(zip(labels, per), key=lambda x: x[1], reverse=True)[:8]:
            print(f"  {lbl:<48}  {m:.4f}")
        print(f"{'─'*60}")
        print(f"  Loss: {loss:.6f}  MAE: {mae:.6f}")
    return loss, mae


@torch.no_grad()
def eval_loader(model, criterion, loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_mae, n = 0.0, 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        pred = model(X_b)
        total_loss += criterion(pred, y_b).item() * len(X_b)
        total_mae  += torch.mean(torch.abs(pred - y_b)).item() * len(X_b)
        n += len(X_b)
    return total_loss / n, total_mae / n


# ─── 主程序 ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LevelSmith Fine-tune  (真实数据 + 防遗忘合成数据)")
    print("=" * 60)

    # ── 加载设备 & checkpoint ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"未找到 {CKPT_PATH}，请先运行 train.py")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    base_config = ckpt.get("config", {})
    hidden_dims = base_config.get("hidden_dims", [128, 64, 32])
    dropout     = base_config.get("dropout", 0.2)

    model = build_model(FEATURE_DIM, OUTPUT_DIM, hidden_dims, dropout, str(device))
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[加载] {CKPT_PATH.name}  (best_val_loss={base_config.get('_best_val_loss','?')})")

    criterion = StyleParamLoss(subdiv_weight=0.1).to(device)

    # ── 构建真实数据集 ──
    print(f"\n[数据] 加载真实验证集: {REAL_JSON.name}")
    X_real, y_real, labels = build_real_tensors(REAL_JSON)
    n_real = len(X_real)
    print(f"  真实样本: {n_real} 条  ({len(set(l.split('/')[0] for l in labels))} 个风格)")

    X_real_t = torch.from_numpy(X_real).to(device)
    y_real_t = torch.from_numpy(y_real).to(device)

    # ── fine-tune 前基线 ──
    pre_loss, pre_mae = eval_real(model, criterion, X_real_t, y_real_t, labels,
                                   header="Fine-tune 前 — 真实验证集")
    synth_val_loss = float(base_config.get("_best_val_loss", "nan"))
    print(f"\n  合成验证集 Loss (训练时): {synth_val_loss:.6f}")
    print(f"  泛化比 (fine-tune 前): {pre_loss / (synth_val_loss + 1e-12):.1f}x")

    # ── 构建防遗忘合成数据（小批量） ──
    n_synth = n_real * SYNTH_MULTIPLIER
    print(f"\n[数据] 生成防遗忘合成数据 (~{n_synth} 条)...")
    synth_ds = generate_dataset(
        n_gaussian        = n_synth // (len(STYLE_REGISTRY) * 2),
        n_interp_per_pair = 0,
        n_perturb         = n_synth // (len(STYLE_REGISTRY) * 2),
        val_ratio         = 0.0,   # 全部用于训练
    )
    X_synth = torch.from_numpy(synth_ds["X_train"]).float()
    y_synth = torch.from_numpy(synth_ds["y_train"]).float()
    print(f"  合成样本: {len(X_synth)} 条")

    # ── 构建混合 DataLoader ──
    real_ds  = TensorDataset(torch.from_numpy(X_real), torch.from_numpy(y_real))
    synth_tds = TensorDataset(X_synth, y_synth)
    mixed_ds  = ConcatDataset([real_ds, synth_tds])
    train_loader = DataLoader(mixed_ds, batch_size=64, shuffle=True, num_workers=0)
    print(f"  混合训练集: {len(mixed_ds)} 条  (真实 {n_real} + 合成 {len(X_synth)})")

    # ── fine-tune 训练循环 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6)

    best_real_loss = pre_loss
    best_state     = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve     = 0

    print(f"\n[Fine-tune] lr={FINETUNE_LR}  epochs={FINETUNE_EPOCHS}  early_stop={EARLY_STOP}")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Real Loss':>10}  {'Real MAE':>9}  {'LR':>9}  {'Time':>5}")
    print("─" * 60)

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(X_b)
        train_loss = total / len(mixed_ds)

        real_loss, real_mae = eval_real(model, criterion, X_real_t, y_real_t, labels)
        scheduler.step(real_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"{epoch:>6}  {train_loss:>10.6f}  {real_loss:>10.6f}  {real_mae:>9.6f}  {lr_now:>9.2e}  {elapsed:>4.1f}s")

        if real_loss < best_real_loss:
            best_real_loss = real_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve     = 0
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP:
            print(f"\n[Early Stop] {EARLY_STOP} epochs 无改善 (epoch {epoch})")
            break

    # ── 恢复最优权重 ──
    model.load_state_dict(best_state)

    # ── fine-tune 后评估 ──
    post_loss, post_mae = eval_real(model, criterion, X_real_t, y_real_t, labels,
                                     header="Fine-tune 后 — 真实验证集")

    # ── 对比输出 ──
    print("\n" + "=" * 60)
    print("  Fine-tune 效果对比")
    print("=" * 60)
    print(f"  合成验证集 Loss (不变): {synth_val_loss:.6f}")
    print(f"  真实验证集 Loss  前: {pre_loss:.6f}  后: {post_loss:.6f}  "
          f"({'↓' if post_loss < pre_loss else '↑'}{abs(post_loss - pre_loss):.6f})")
    print(f"  真实验证集 MAE   前: {pre_mae:.6f}   后: {post_mae:.6f}")
    post_ratio = post_loss / (synth_val_loss + 1e-12)
    print(f"  泛化比 前: {pre_loss/(synth_val_loss+1e-12):.1f}x  后: {post_ratio:.1f}x")
    if post_ratio < 10.0:
        print("  判断: 泛化比 < 10x，fine-tune 有效 ✓")
    elif post_ratio < 20.0:
        print("  判断: 轻度改善，可继续训练或扩充真实数据")
    else:
        print("  判断: 改善有限，分布差异较大")

    # ── 保存 ──
    torch.save({
        "model_state_dict": best_state,
        "config": {**base_config, "_finetune_real_loss": round(post_loss, 8),
                   "_finetune_real_mae": round(post_mae, 8)},
        "output_keys":  OUTPUT_KEYS,
        "output_params": OUTPUT_PARAMS,
        "feature_dim":  FEATURE_DIM,
        "output_dim":   OUTPUT_DIM,
    }, OUT_CKPT)
    print(f"\n[保存] {OUT_CKPT.name}")

    # ── 导出风格参数 JSON ──
    from train import export_trained_params
    export_trained_params(model, device,
                          {**base_config, "_best_val_loss": synth_val_loss},
                          save_path=str(OUT_PARAMS))


if __name__ == "__main__":
    main()
