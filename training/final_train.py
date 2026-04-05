"""
final_train.py
全量最终训练：合成数据 (~131K) + 所有 OSM 真实数据 (~540K)
架构: 16→256→128→64→32→20
early stopping patience=20
完成后生成全部 20 种风格 preview GLB
"""

import json
import sys
import time
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent))

from generate_data import generate_dataset
from model import StyleParamLoss, build_model, count_parameters
from style_registry import (
    STYLE_REGISTRY, OUTPUT_KEYS, OUTPUT_PARAMS, FEATURE_DIM, OUTPUT_DIM,
    get_feature_vector, get_param_vector, denormalize_params,
)
from train import export_trained_params

SCRIPT_DIR = Path(__file__).parent
OSM_DIR    = SCRIPT_DIR / "validation_data"

# ── 超参数 ──────────────────────────────────────────────────────
HIDDEN_DIMS  = [256, 128, 64, 32]
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.2
BATCH_SIZE   = 512
EPOCHS       = 300
EARLY_STOP   = 20
LR_PATIENCE  = 10
LR_FACTOR    = 0.5

# 合成数据量（约 131K 总，取 85% 作训练）
N_GAUSSIAN   = 1000
N_INTERP     = 500
N_PERTURB    = 800
VAL_RATIO    = 0.15

# OSM 文件 → 风格名称映射
OSM_STYLE_MAP = {
    "medieval_osm.json":              "medieval",
    "modern_osm.json":                "modern",
    "industrial_osm.json":            "industrial",
    "japanese_osm.json":              "japanese",
    "desert_osm.json":                "desert",
    "fantasy_osm.json":               "fantasy",
    "horror_osm.json":                "horror",
    "medieval_chapel_osm.json":       "medieval_chapel",
    "medieval_keep_osm.json":         "medieval_keep",
    "modern_loft_osm.json":           "modern_loft",
    "modern_villa_osm.json":          "modern_villa",
    "industrial_workshop_osm.json":   "industrial_workshop",
    "industrial_powerplant_osm.json": "industrial_powerplant",
    "fantasy_dungeon_osm.json":       "fantasy_dungeon",
    "fantasy_palace_osm.json":        "fantasy_palace",
    "horror_asylum_osm.json":         "horror_asylum",
    "horror_crypt_osm.json":          "horror_crypt",
    "japanese_temple_osm.json":       "japanese_temple",
    "japanese_machiya_osm.json":      "japanese_machiya",
    "desert_palace_osm.json":         "desert_palace",
}

_OSM_SCALAR_KEYS = [
    "height_range_min", "height_range_max", "wall_thickness", "floor_thickness",
    "door_width", "door_height", "win_width", "win_height", "win_density",
    "subdivision", "roof_type", "roof_pitch",
    "has_battlements", "has_arch", "eave_overhang", "column_count", "window_shape",
]

# 每个风格 OSM 数据最大样本数（防止极大文件主导训练，但保留多样性）
MAX_PER_STYLE = 20_000


def osm_to_param_vector(rec: dict, style_name: str) -> np.ndarray:
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


def load_all_osm() -> tuple[np.ndarray, np.ndarray]:
    """加载所有 OSM JSON，返回 (X, y) numpy 数组。"""
    X_parts, y_parts = [], []
    total_loaded = 0

    print("\n[OSM] 加载真实建筑数据...")
    print(f"  {'文件':<40}  {'总数':>7}  {'取用':>7}")
    print("  " + "─" * 58)

    rng = np.random.default_rng(42)

    for filename, style in OSM_STYLE_MAP.items():
        path = OSM_DIR / filename
        if not path.exists():
            print(f"  {'  [跳过] ' + filename:<40}  {'—':>7}  {'—':>7}")
            continue
        if style not in STYLE_REGISTRY:
            print(f"  {'  [未知] ' + filename:<40}  {'—':>7}  {'—':>7}")
            continue

        try:
            data = json.loads(path.read_text("utf-8"))
            buildings = data.get("buildings", [])
            total = len(buildings)

            # 采样上限
            if total > MAX_PER_STYLE:
                idx = rng.choice(total, MAX_PER_STYLE, replace=False)
                buildings = [buildings[i] for i in idx]
                used = MAX_PER_STYLE
            else:
                used = total

            fv = get_feature_vector(style)
            X_chunk = np.tile(fv, (used, 1)).astype(np.float32)
            y_chunk = np.array([osm_to_param_vector(r, style) for r in buildings],
                               dtype=np.float32)
            X_parts.append(X_chunk)
            y_parts.append(y_chunk)
            total_loaded += used
            print(f"  {filename:<40}  {total:>7,d}  {used:>7,d}")

        except Exception as e:
            print(f"  [错误] {filename}: {e}")

    print(f"\n  OSM 总计加载: {total_loaded:,d} 条")
    return np.concatenate(X_parts), np.concatenate(y_parts)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae, n = 0.0, 0.0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred = model(Xb)
        loss = criterion(pred, yb)
        mae  = torch.mean(torch.abs(pred - yb))
        total_loss += loss.item() * len(Xb)
        total_mae  += mae.item()  * len(Xb)
        n += len(Xb)
    return total_loss / n, total_mae / n


def generate_all_previews(params_path: Path, out_dir: Path):
    print(f"\n{'='*65}")
    print("  生成 20 种风格 Preview GLB")
    print(f"{'='*65}")

    import generate_level as gl
    importlib.reload(gl)

    data = json.loads(params_path.read_text("utf-8"))
    results = []

    for style in STYLE_REGISTRY:
        sp = data["styles"].get(style, {}).get("params")
        if sp is None:
            print(f"  [跳过] {style} — 参数未找到")
            continue
        try:
            scene = gl.build_scene({style: sp}, use_style_palette=True)
            n_faces = sum(len(g.faces) for g in scene.geometry.values())
            out = out_dir / f"{style}_preview.glb"
            scene.export(str(out))
            print(f"  {style:<25}  {n_faces:>8,d} faces  →  {out.name}")
            results.append(style)
        except Exception as e:
            print(f"  [错误] {style}: {e}")

    return results


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65)
    print("  final_train.py  —  全量最终训练")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ── 1. 合成数据 ────────────────────────────────────────────
    print("\n[合成数据] 生成训练/验证集...")
    synth = generate_dataset(
        n_gaussian=N_GAUSSIAN, n_interp_per_pair=N_INTERP,
        n_perturb=N_PERTURB, val_ratio=VAL_RATIO,
    )
    X_syn_tr = synth["X_train"].astype(np.float32)
    y_syn_tr = synth["y_train"].astype(np.float32)
    X_val    = torch.from_numpy(synth["X_val"].astype(np.float32))
    y_val    = torch.from_numpy(synth["y_val"].astype(np.float32))
    print(f"  合成训练: {len(X_syn_tr):,d}  合成验证: {len(X_val):,d}")

    # ── 2. OSM 真实数据 ────────────────────────────────────────
    X_osm, y_osm = load_all_osm()
    print(f"  OSM 训练: {len(X_osm):,d}")

    # ── 3. 合并训练集 ──────────────────────────────────────────
    X_train_all = np.concatenate([X_syn_tr, X_osm])
    y_train_all = np.concatenate([y_syn_tr, y_osm])
    print(f"\n[数据] 训练集总计: {len(X_train_all):,d}  验证集: {len(X_val):,d}")

    # DataLoader
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_all), torch.from_numpy(y_train_all)),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin, num_workers=0,
    )

    # ── 4. 模型 ────────────────────────────────────────────────
    print(f"\n[模型] 架构: {FEATURE_DIM} → {' → '.join(map(str, HIDDEN_DIMS))} → {OUTPUT_DIM}")
    model = build_model(FEATURE_DIM, OUTPUT_DIM, HIDDEN_DIMS, DROPOUT, str(device))
    print(f"  参数量: {count_parameters(model):,}")

    criterion = StyleParamLoss(subdiv_weight=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=LR_PATIENCE, factor=LR_FACTOR, min_lr=1e-6)

    # ── 5. 训练 ────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    t_start       = time.time()

    print(f"\n[训练] epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}  early_stop={EARLY_STOP}")
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'MAE':>9}  {'LR':>9}  {'T':>6}")
    print("─" * 65)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(Xb)
        tr_loss = total / len(train_loader.dataset)

        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_loss:>10.6f}  {val_loss:>10.6f}  "
                  f"{val_mae:>9.6f}  {lr_now:>9.2e}  {time.time()-t0:>5.1f}s",
                  flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP:
            print(f"\n[Early Stop] epoch {epoch}，{EARLY_STOP} epochs 无改善")
            break

    total_time = time.time() - t_start
    print(f"\n[完成] 耗时: {total_time:.1f}s  最佳验证损失: {best_val_loss:.6f}")
    print(f"  对比基准: 0.001322  →  {'↓改善' if best_val_loss < 0.001322 else '↑退步'}"
          f"  ({abs(best_val_loss - 0.001322):.6f})")

    model.load_state_dict(best_state)

    # ── 6. 保存 ────────────────────────────────────────────────
    config = {
        "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "early_stop": EARLY_STOP,
        "n_gaussian": N_GAUSSIAN, "n_interp_per_pair": N_INTERP, "n_perturb": N_PERTURB,
        "_best_val_loss": round(best_val_loss, 8),
        "_training": "final_full_20styles_osm",
    }
    ckpt_path = SCRIPT_DIR / "best_model.pt"
    torch.save({
        "model_state_dict": best_state,
        "config": config,
        "output_keys": OUTPUT_KEYS,
        "output_params": OUTPUT_PARAMS,
        "feature_dim": FEATURE_DIM,
        "output_dim": OUTPUT_DIM,
    }, ckpt_path)
    print(f"[保存] {ckpt_path.name}")

    params_path = SCRIPT_DIR / "trained_style_params.json"
    export_trained_params(model, device, config, save_path=str(params_path))

    # ── 7. 生成全部 Preview GLB ────────────────────────────────
    generate_all_previews(params_path, SCRIPT_DIR)

    # ── 8. 汇总 ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  最终训练结果汇总")
    print(f"{'='*65}")
    print(f"  架构        : {FEATURE_DIM} → {' → '.join(map(str, HIDDEN_DIMS))} → {OUTPUT_DIM}")
    print(f"  训练样本    : {len(X_train_all):,d}  (合成 {len(X_syn_tr):,d} + OSM {len(X_osm):,d})")
    print(f"  验证样本    : {len(X_val):,d}  (合成)")
    print(f"  最佳Val Loss: {best_val_loss:.6f}")
    print(f"  基准 Loss   : 0.001322")
    delta = best_val_loss - 0.001322
    pct   = delta / 0.001322 * 100
    print(f"  变化        : {delta:+.6f}  ({pct:+.1f}%)")
    print()


if __name__ == "__main__":
    main()
