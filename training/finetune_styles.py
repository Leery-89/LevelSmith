"""
finetune_styles.py
对 japanese / desert / modern / industrial 4 种风格进行 OSM 数据专项精调。

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

# ── 风格名到注册名映射（OSM JSON 里的 style 字段） ───────────────
STYLE_MAP = {
    "japanese":   "japanese",
    "desert":     "desert",
    "modern":     "modern",
    "industrial": "industrial",
}

# ── 每个风格的 preview 生成用参数风格名（对应 generate_level.py 中的 palette） ──
PREVIEW_STYLE_ALIAS = {
    "japanese":   "japanese",
    "desert":     "desert",
    "modern":     "modern",
    "industrial": "industrial",
}


# ─── OSM 记录 → 归一化 20 维参数向量 ─────────────────────────────

_OSM_SCALAR_KEYS = [
    "height_range_min", "height_range_max", "wall_thickness", "floor_thickness",
    "door_width", "door_height", "win_width", "win_height", "win_density",
    "subdivision", "roof_type", "roof_pitch",
    "has_battlements", "has_arch", "eave_overhang", "column_count", "window_shape",
]


def osm_to_param_vector(rec: dict, style_name: str) -> np.ndarray:
    """将单条 OSM 建筑记录转换为归一化 20 维参数向量。"""
    y = get_param_vector(style_name).copy()   # 风格默认值作为基底

    # 标量字段
    for key in _OSM_SCALAR_KEYS:
        if key not in rec:
            continue
        if key not in OUTPUT_KEYS:
            continue
        idx = OUTPUT_KEYS.index(key)
        lo, hi = OUTPUT_PARAMS[key]["range"]
        norm = float(np.clip((float(rec[key]) - lo) / (hi - lo + 1e-9), 0.0, 1.0))
        y[idx] = norm

    # wall_color: [r,g,b] 列表
    wc = rec.get("wall_color")
    if wc and len(wc) == 3:
        for j, key in enumerate(("wall_color_r", "wall_color_g", "wall_color_b")):
            idx = OUTPUT_KEYS.index(key)
            y[idx] = float(np.clip(wc[j], 0.0, 1.0))

    return y.astype(np.float32)


def build_osm_tensors(style: str, json_path: Path):
    """加载 OSM JSON，构建 80/20 训练/验证集。"""
    if not json_path.exists():
        raise FileNotFoundError(f"未找到 {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    buildings = data["buildings"]
    print(f"  OSM 原始: {len(buildings)} 条")

    fv = get_feature_vector(style)   # 所有样本共享同一特征向量

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


# ─── 评估辅助 ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, criterion, X_t, y_t, device) -> Tuple[float, float]:
    model.eval()
    X_t, y_t = X_t.to(device), y_t.to(device)
    pred = model(X_t)
    loss = criterion(pred, y_t).item()
    mae  = torch.mean(torch.abs(pred - y_t)).item()
    return loss, mae


# ─── 单风格精调 ────────────────────────────────────────────────

def finetune_one(style: str, device, base_ckpt: dict, criterion) -> dict:
    """对单个风格精调，返回精调后 state_dict 和指标。"""
    print(f"\n{'═'*65}")
    print(f"  Fine-tune: {style.upper()}")
    print(f"{'═'*65}")

    json_path = OSM_DIR / f"{style}_osm.json"
    X_tr, y_tr, X_val, y_val = build_osm_tensors(style, json_path)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    # ── 重建模型（每次都从 best_model 重新加载，风格独立）──
    base_cfg = base_ckpt.get("config", {})
    model = build_model(
        FEATURE_DIM, OUTPUT_DIM,
        base_cfg.get("hidden_dims", [128, 64, 32]),
        base_cfg.get("dropout", 0.2), str(device)
    )
    model.load_state_dict(base_ckpt["model_state_dict"])

    # ── 基线 ──
    pre_loss, pre_mae = evaluate(model, criterion, X_val_t, y_val_t, device)
    synth_base = float(base_cfg.get("_best_val_loss", float("nan")))
    print(f"\n[基线] OSM 验证集  Loss={pre_loss:.6f}  MAE={pre_mae:.6f}")
    print(f"       合成验证集 loss:  {synth_base:.6f}  泛化比: {pre_loss/(synth_base+1e-12):.1f}x")

    # ── 精调前参数预测 ──
    with torch.no_grad():
        fv_t = torch.from_numpy(get_feature_vector(style)).unsqueeze(0).to(device)
        pred_before = denormalize_params(model(fv_t).squeeze(0).cpu().numpy())

    # ── 合成防遗忘数据 ──
    n_synth = len(X_tr) * SYNTH_RATIO
    n_per   = max(8, n_synth // (len(STYLE_REGISTRY) * 2))
    print(f"\n[数据] 生成合成防遗忘数据 (每风格约 {n_per*2} 条)...")
    synth_ds = generate_dataset(n_gaussian=n_per, n_interp_per_pair=0,
                                n_perturb=n_per, val_ratio=0.0)
    X_syn = synth_ds["X_train"].astype(np.float32)
    y_syn = synth_ds["y_train"].astype(np.float32)
    print(f"  合成: {len(X_syn)}  OSM 训练: {len(X_tr)}  "
          f"比例 1:{len(X_syn)/max(1,len(X_tr)):.1f}")

    # ── DataLoader ──
    mixed_ds = ConcatDataset([
        TensorDataset(torch.from_numpy(X_tr),  torch.from_numpy(y_tr)),
        TensorDataset(torch.from_numpy(X_syn), torch.from_numpy(y_syn)),
    ])
    train_loader = DataLoader(mixed_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    # ── 训练循环 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6)

    best_loss  = pre_loss
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0

    print(f"\n[训练] lr={FINETUNE_LR}  epochs={FINETUNE_EPOCHS}  batch={BATCH_SIZE}")
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
                  f"{val_mae:>8.6f}  {lr_now:>8.2e}  {elapsed:>4.1f}s")

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP:
            print(f"\n[Early Stop] epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # ── 精调后评估 ──
    post_loss, post_mae = evaluate(model, criterion, X_val_t, y_val_t, device)

    # ── 参数对比 ──
    with torch.no_grad():
        pred_after = denormalize_params(model(fv_t).squeeze(0).cpu().numpy())

    print(f"\n{'─'*65}")
    print(f"  [结果] Loss  前: {pre_loss:.6f}  后: {best_loss:.6f}  "
          f"({'↓' if best_loss < pre_loss else '↑'}{abs(best_loss - pre_loss):.6f})")
    print(f"         MAE   前: {pre_mae:.6f}  后: {post_mae:.6f}")
    print(f"         泛化比 前: {pre_loss/(synth_base+1e-12):.1f}x  "
          f"后: {best_loss/(synth_base+1e-12):.1f}x")

    print(f"\n{'─'*65}")
    print(f"  {'参数':<22}  {'精调前':>12}  {'精调后':>12}  {'变化':>10}")
    print(f"{'─'*65}")
    for key in OUTPUT_KEYS:
        bef = pred_before[key]
        aft = pred_after[key]
        dlt = aft - bef
        flag = "  ←" if abs(dlt) > 0.05 * max(abs(bef), 1e-3) else ""
        print(f"  {key:<22}  {bef:>12.4f}  {aft:>12.4f}  {dlt:>+10.4f}{flag}")

    # ── 保存模型 ──
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
    print(f"\n[保存] {out_ckpt.name}")

    # ── 导出全量风格参数 JSON ──
    out_params_path = OUT_DIR / f"finetuned_{style}_style_params.json"
    from train import export_trained_params
    export_trained_params(model, device,
                          {**base_cfg, "_best_val_loss": synth_base},
                          save_path=str(out_params_path))

    return {
        "style":      style,
        "state_dict": best_state,
        "params_path": str(out_params_path),
        "pre_loss":   pre_loss,
        "post_loss":  best_loss,
        "post_mae":   post_mae,
        "synth_base": synth_base,
    }


# ─── 生成 Preview GLB ──────────────────────────────────────────

def generate_preview(style: str, params_path: str):
    """从精调后风格参数生成 {style}_preview.glb。"""
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
    print(f"  [生成] {out_path.name}  ({n_faces:,} faces)")
    return str(out_path)


# ─── 主程序 ────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  finetune_styles.py  —  4 风格 OSM 专项精调")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if not BASE_CKPT.exists():
        raise FileNotFoundError(f"未找到 {BASE_CKPT}，请先运行 train.py")

    base_ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
    criterion = StyleParamLoss(subdiv_weight=0.1).to(device)

    # ── 检查 OSM 文件是否存在 ──
    styles_to_run = []
    for style in ["japanese", "desert", "modern", "industrial"]:
        json_path = OSM_DIR / f"{style}_osm.json"
        if json_path.exists():
            data = json.loads(json_path.read_text("utf-8"))
            n = data.get("total", 0)
            if n >= 10:
                print(f"  ✓ {style:<15}: {n:5d} 条 OSM 数据")
                styles_to_run.append(style)
            else:
                print(f"  ✗ {style:<15}: {n:5d} 条（不足，跳过）")
        else:
            print(f"  ✗ {style:<15}: 未找到 {json_path.name}")

    if not styles_to_run:
        raise SystemExit("没有可精调的风格，请先运行 fetch_osm_styles.py")

    # ── 逐风格精调 ──
    results = []
    for style in styles_to_run:
        result = finetune_one(style, device, base_ckpt, criterion)
        results.append(result)

    # ── 生成所有 preview GLB ──
    print(f"\n{'═'*65}")
    print("  生成 Preview GLB")
    print(f"{'═'*65}")
    for res in results:
        generate_preview(res["style"], res["params_path"])

    # ── 最终汇总 ──
    print(f"\n{'═'*65}")
    print("  精调效果总览")
    print(f"{'═'*65}")
    print(f"  {'风格':<15}  {'前 Loss':>10}  {'后 Loss':>10}  {'降幅':>8}  {'泛化比':>8}")
    print("─" * 65)
    for res in results:
        drop_pct = (res["pre_loss"] - res["post_loss"]) / (res["pre_loss"] + 1e-12) * 100
        ratio    = res["post_loss"] / (res["synth_base"] + 1e-12)
        print(f"  {res['style']:<15}  {res['pre_loss']:>10.6f}  "
              f"{res['post_loss']:>10.6f}  {drop_pct:>7.1f}%  {ratio:>7.1f}x")
    print()


if __name__ == "__main__":
    main()
