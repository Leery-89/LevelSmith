"""
LevelSmith 结构参数训练脚本

用法:
    python train.py                    # 使用默认配置
    python train.py --epochs 200       # 自定义 epoch
    python train.py --device cpu       # 强制 CPU

训练完成后自动导出 trained_style_params.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 确保当前目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

from generate_data import generate_dataset
from model import StyleParamMLP, StyleParamLoss, build_model, count_parameters
from style_registry import (
    STYLE_REGISTRY, OUTPUT_KEYS, OUTPUT_PARAMS, FEATURE_DIM, OUTPUT_DIM,
    get_feature_vector, denormalize_params,
)

# ─── 默认超参数 ────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "epochs":           150,
    "batch_size":       256,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "dropout":          0.2,
    "hidden_dims":      [128, 64, 32],
    "lr_patience":      15,       # ReduceLROnPlateau patience
    "lr_factor":        0.5,
    "early_stop":       30,       # early stopping patience (epochs)
    "subdiv_weight":    0.1,      # subdivision 整数惩罚权重
    # 数据生成参数
    "n_gaussian":       1000,
    "n_interp_per_pair":500,
    "n_perturb":        800,
    "val_ratio":        0.15,
}


# ─── 工具函数 ─────────────────────────────────────────────────

def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[设备] 使用 GPU: {torch.cuda.get_device_name(0)}")
        # RTX 4070 启用 TF32 加速
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        dev = torch.device("cpu")
        if preferred == "cuda":
            print("[设备] CUDA 不可用，回退到 CPU")
        else:
            print("[设备] 使用 CPU")
    return dev


def make_dataloaders(
    dataset: Dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    X_train = torch.from_numpy(dataset["X_train"])
    y_train = torch.from_numpy(dataset["y_train"])
    X_val   = torch.from_numpy(dataset["X_val"])
    y_val   = torch.from_numpy(dataset["y_val"])

    # 对于小数据集可以预先移到 GPU，避免每 batch 传输开销
    if device.type == "cuda" and len(X_train) < 50000:
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val,   y_val   = X_val.to(device),   y_val.to(device)
        pin = False
    else:
        pin = (device.type == "cuda")

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              pin_memory=pin, num_workers=0)
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    n = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        mae  = torch.mean(torch.abs(pred - y_batch))
        total_loss += loss.item() * len(X_batch)
        total_mae  += mae.item()  * len(X_batch)
        n += len(X_batch)
    return total_loss / n, total_mae / n


# ─── 导出函数 ─────────────────────────────────────────────────

@torch.no_grad()
def export_trained_params(
    model: nn.Module,
    device: torch.device,
    config: dict,
    save_path: str = "trained_style_params.json",
) -> dict:
    """
    用训练好的模型推理每个已知风格的参数，
    并导出为 trained_style_params.json。

    JSON 结构:
    {
      "metadata": { ... },
      "styles": {
        "medieval":   { "feature_vector": [...], "params": {...}, "normalized": [...] },
        "modern":     { ... },
        "industrial": { ... }
      }
    }
    """
    model.eval()
    result = {
        "metadata": {
            "model":        "StyleParamMLP",
            "input_dim":    FEATURE_DIM,
            "output_dim":   OUTPUT_DIM,
            "output_keys":  OUTPUT_KEYS,
            "output_params": {
                k: {"range": list(v["range"]), "unit": v["unit"]}
                for k, v in OUTPUT_PARAMS.items()
            },
            "hidden_dims":  config["hidden_dims"],
            "epochs_trained": config.get("_epochs_trained", config["epochs"]),
            "best_val_loss":  config.get("_best_val_loss", None),
        },
        "styles": {},
    }

    for name in STYLE_REGISTRY:
        fv = get_feature_vector(name)
        fv_tensor = torch.from_numpy(fv).unsqueeze(0).to(device)
        pred_norm = model(fv_tensor).squeeze(0).cpu().numpy()
        params = denormalize_params(pred_norm)

        result["styles"][name] = {
            "description":   STYLE_REGISTRY[name].description,
            "feature_vector": fv.tolist(),
            "normalized":    pred_norm.tolist(),
            "params": {
                "height_range":    [params["height_range_min"], params["height_range_max"]],
                "wall_thickness":  params["wall_thickness"],
                "floor_thickness": params["floor_thickness"],
                "door_spec":       {"width": params["door_width"], "height": params["door_height"]},
                "win_spec":        {"width": params["win_width"],  "height": params["win_height"],
                                    "density": params["win_density"]},
                "subdivision":     int(params["subdivision"]),
            },
        }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[导出] 训练结果已保存: {save_path}")
    return result


# ─── 训练主函数 ───────────────────────────────────────────────

def train(config: dict, output_dir: str = ".") -> nn.Module:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(config.get("device", "cuda"))

    print("\n[数据] 生成训练数据...")
    dataset = generate_dataset(
        n_gaussian       = config["n_gaussian"],
        n_interp_per_pair= config["n_interp_per_pair"],
        n_perturb        = config["n_perturb"],
        val_ratio        = config["val_ratio"],
    )

    train_loader, val_loader = make_dataloaders(dataset, config["batch_size"], device)

    print("\n[模型] 构建模型...")
    model = build_model(
        input_dim   = FEATURE_DIM,
        output_dim  = OUTPUT_DIM,
        hidden_dims = config["hidden_dims"],
        dropout     = config["dropout"],
        device      = str(device),
    )
    print(f"  参数量: {count_parameters(model):,}")

    criterion = StyleParamLoss(subdiv_weight=config["subdiv_weight"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=config["lr_patience"],
        factor=config["lr_factor"],
        min_lr=1e-6,
    )

    # ── 训练循环 ──
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [], "val_mae": [], "lr": []}

    print(f"\n[训练] 开始训练 {config['epochs']} epochs ...")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val MAE':>9}  {'LR':>9}  {'Time':>6}")
    print("─" * 65)

    t_start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        train_loss          = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(lr_now)

        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch == 1 or epoch == config["epochs"]:
            print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  {val_mae:>9.6f}  {lr_now:>9.2e}  {elapsed:>5.1f}s")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= config["early_stop"]:
            print(f"\n[Early Stop] {config['early_stop']} epochs 无改善，停止训练 (epoch {epoch})")
            config["_epochs_trained"] = epoch
            break
    else:
        config["_epochs_trained"] = config["epochs"]

    total_time = time.time() - t_start
    print(f"\n[完成] 总耗时: {total_time:.1f}s | 最佳验证损失: {best_val_loss:.6f}")

    # 恢复最优权重
    model.load_state_dict(best_state)
    config["_best_val_loss"] = round(best_val_loss, 8)

    # 保存模型权重
    ckpt_path = output_dir / "best_model.pt"
    torch.save({
        "model_state_dict": best_state,
        "config":           config,
        "output_keys":      OUTPUT_KEYS,
        "output_params":    OUTPUT_PARAMS,
        "feature_dim":      FEATURE_DIM,
        "output_dim":       OUTPUT_DIM,
    }, ckpt_path)
    print(f"[保存] 模型权重: {ckpt_path}")

    # 保存训练历史
    hist_path = output_dir / "train_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[保存] 训练历史: {hist_path}")

    # 导出风格参数 JSON
    export_path = output_dir / "trained_style_params.json"
    export_trained_params(model, device, config, save_path=str(export_path))

    return model


# ─── 入口 ─────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LevelSmith 结构参数训练")
    parser.add_argument("--epochs",       type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size",   type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",           type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--dropout",      type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--device",       type=str,   default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir",   type=str,   default=".")
    parser.add_argument("--n-gaussian",   type=int,   default=DEFAULT_CONFIG["n_gaussian"])
    parser.add_argument("--n-interp",     type=int,   default=DEFAULT_CONFIG["n_interp_per_pair"])
    parser.add_argument("--n-perturb",    type=int,   default=DEFAULT_CONFIG["n_perturb"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = {**DEFAULT_CONFIG}
    config.update({
        "epochs":            args.epochs,
        "batch_size":        args.batch_size,
        "lr":                args.lr,
        "dropout":           args.dropout,
        "device":            args.device,
        "n_gaussian":        args.n_gaussian,
        "n_interp_per_pair": args.n_interp,
        "n_perturb":         args.n_perturb,
    })

    print("=" * 65)
    print("  LevelSmith 结构参数训练模块")
    print("=" * 65)
    print(f"配置: {json.dumps({k: v for k, v in config.items() if not k.startswith('_')}, indent=2, ensure_ascii=False)}")

    train(config, output_dir=args.output_dir)
