"""
LevelSmith 模型结构优化脚本

对比三种 MLP 结构:
  方案 A: 16 → 256 → 128 → 64 → 32 → 10
  方案 B: 16 → 128 → 128 → 64 → 32 → 10
  方案 C: 16 → 256 → 128 → 128 → 64 → 32 → 10

输出目录:
  models/          — 各方案最佳模型权重
  training_logs/   — 各方案训练历史 JSON
  results/         — 汇总对比报告

随机种子: 42（全局固定，保证可复现）
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 确保当前目录在路径中
sys.path.insert(0, str(Path(__file__).parent))

from generate_data import generate_dataset
from model import StyleParamLoss, count_parameters
from style_registry import (
    STYLE_REGISTRY, OUTPUT_KEYS, OUTPUT_PARAMS, FEATURE_DIM, OUTPUT_DIM,
    get_feature_vector, denormalize_params,
)

# ─── 随机种子固定 ─────────────────────────────────────────────
SEED = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── 三种候选结构 ──────────────────────────────────────────────
ARCHITECTURES = {
    "A": [256, 128, 64, 32],
    "B": [128, 128, 64, 32],
    "C": [256, 128, 128, 64, 32],
}

BASE_CONFIG = {
    "epochs":            150,
    "batch_size":        256,
    "lr":                1e-3,
    "weight_decay":      1e-4,
    "dropout":           0.2,
    "lr_patience":       15,
    "lr_factor":         0.5,
    "early_stop":        30,
    "subdiv_weight":     0.1,
    "n_gaussian":        1000,
    "n_interp_per_pair": 500,
    "n_perturb":         800,
    "val_ratio":         0.15,
}


# ─── 模型构建（内联，避免与原 build_model 的 device 字符串冲突）────
class StyleParamMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers += [nn.Linear(in_dim, output_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ─── 数据加载 ──────────────────────────────────────────────────
def make_loaders(dataset, batch_size, device):
    X_tr = torch.from_numpy(dataset["X_train"])
    y_tr = torch.from_numpy(dataset["y_train"])
    X_va = torch.from_numpy(dataset["X_val"])
    y_va = torch.from_numpy(dataset["y_val"])

    if device.type == "cuda" and len(X_tr) < 50000:
        X_tr, y_tr = X_tr.to(device), y_tr.to(device)
        X_va, y_va = X_va.to(device), y_va.to(device)
        pin = False
    else:
        pin = (device.type == "cuda")

    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                           shuffle=True, pin_memory=pin, num_workers=0)
    va_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size,
                           shuffle=False, pin_memory=pin, num_workers=0)
    return tr_loader, va_loader


# ─── 单 epoch 训练 / 验证 ──────────────────────────────────────
def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae, n = 0.0, 0.0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(X)
        total_loss += criterion(pred, y).item() * len(X)
        total_mae  += torch.mean(torch.abs(pred - y)).item() * len(X)
        n += len(X)
    return total_loss / n, total_mae / n


# ─── 单方案训练 ────────────────────────────────────────────────
def train_one_arch(name, hidden_dims, dataset, config, device, models_dir, logs_dir):
    print(f"\n{'='*60}")
    print(f"  方案 {name}: 16 → {' → '.join(map(str, hidden_dims))} → 10")
    print(f"{'='*60}")

    set_seed(SEED)  # 每个方案独立固定种子

    model = StyleParamMLP(FEATURE_DIM, OUTPUT_DIM, hidden_dims, config["dropout"]).to(device)
    print(f"  参数量: {count_parameters(model):,}")

    criterion = StyleParamLoss(config["subdiv_weight"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=config["lr_patience"],
        factor=config["lr_factor"],
        min_lr=1e-6,
    )

    tr_loader, va_loader = make_loaders(dataset, config["batch_size"], device)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [], "val_mae": [], "lr": []}

    print(f"\n{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'MAE':>9}  {'LR':>9}  {'Time':>5}")
    print("─" * 60)

    t_start = time.time()
    epochs_trained = config["epochs"]
    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        tr_loss          = train_epoch(model, tr_loader, optimizer, criterion, device)
        va_loss, va_mae  = evaluate(model, va_loader, criterion, device)
        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_mae"].append(va_mae)
        history["lr"].append(lr_now)

        if epoch % 10 == 0 or epoch == 1 or epoch == config["epochs"]:
            print(f"{epoch:>6}  {tr_loss:>10.6f}  {va_loss:>10.6f}  "
                  f"{va_mae:>9.6f}  {lr_now:>9.2e}  {time.time()-t0:>4.1f}s")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if no_improve >= config["early_stop"]:
            print(f"\n[Early Stop] {config['early_stop']} epochs 无改善，停止 (epoch {epoch})")
            epochs_trained = epoch
            break

    total_time = time.time() - t_start
    print(f"\n[完成] 耗时: {total_time:.1f}s | 最佳验证损失: {best_val_loss:.6f}")

    # 恢复最优权重并推理 7 种风格
    model.load_state_dict(best_state)

    # 保存模型
    ckpt_path = models_dir / f"best_model_{name}.pt"
    torch.save({
        "model_state_dict": best_state,
        "hidden_dims": hidden_dims,
        "feature_dim": FEATURE_DIM,
        "output_dim":  OUTPUT_DIM,
        "output_keys": OUTPUT_KEYS,
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs_trained,
    }, ckpt_path)
    print(f"[保存] 模型: {ckpt_path}")

    # 保存训练历史
    log_path = logs_dir / f"train_history_{name}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[保存] 日志: {log_path}")

    # 推理 7 种风格参数
    style_results = infer_styles(model, device)

    return {
        "arch":          name,
        "hidden_dims":   hidden_dims,
        "n_params":      count_parameters(model),
        "best_val_loss": round(best_val_loss, 8),
        "epochs_trained": epochs_trained,
        "styles":        style_results,
        "ckpt_path":     str(ckpt_path),
    }


# ─── 风格推理 ─────────────────────────────────────────────────
@torch.no_grad()
def infer_styles(model, device):
    model.eval()
    results = {}
    for name in STYLE_REGISTRY:
        fv = torch.from_numpy(get_feature_vector(name)).unsqueeze(0).to(device)
        pred_norm = model(fv).squeeze(0).cpu().numpy()
        params = denormalize_params(pred_norm)
        results[name] = params
    return results


# ─── 参数合理性检查 ────────────────────────────────────────────
def check_params_sanity(style_name, params):
    issues = []
    if params["height_range_max"] <= params["height_range_min"]:
        issues.append("height_range_max <= height_range_min")
    for key, meta in OUTPUT_PARAMS.items():
        lo, hi = meta["range"]
        val = params[key]
        if val < lo or val > hi:
            issues.append(f"{key}={val:.3f} 超出范围 [{lo}, {hi}]")
    return issues


# ─── 打印对比表 ───────────────────────────────────────────────
def print_comparison_table(results):
    print("\n" + "=" * 70)
    print("  模型结构对比")
    print("=" * 70)
    print(f"{'方案':<6} {'结构':<30} {'参数量':>8} {'验证损失':>12} {'Epochs':>7}")
    print("─" * 70)
    for r in results:
        arch_str = f"16→{'→'.join(map(str, r['hidden_dims']))}→10"
        print(f"  {r['arch']:<4} {arch_str:<30} {r['n_params']:>8,} "
              f"{r['best_val_loss']:>12.6f} {r['epochs_trained']:>7}")
    print("─" * 70)


def print_style_table(best_result):
    print(f"\n{'='*90}")
    print(f"  最佳模型 方案{best_result['arch']} — 7种风格参数预测结果")
    print(f"{'='*90}")
    print(f"{'风格':<12} {'层高范围(m)':<16} {'墙厚':>6} {'楼板':>6} "
          f"{'门宽':>6} {'门高':>6} {'窗宽':>6} {'窗高':>6} {'窗密度':>7} {'细分':>4}")
    print("─" * 90)
    for sname, p in best_result["styles"].items():
        h = p["height_range_min"]
        H = p["height_range_max"]
        print(f"{sname:<12} {h:.2f}~{H:.2f}{'':>4} "
              f"{p['wall_thickness']:>6.3f} {p['floor_thickness']:>6.3f} "
              f"{p['door_width']:>6.3f} {p['door_height']:>6.3f} "
              f"{p['win_width']:>6.3f} {p['win_height']:>6.3f} "
              f"{p['win_density']:>7.3f} {p['subdivision']:>4}")
    print("─" * 90)


# ─── 主流程 ──────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).parent
    models_dir   = base_dir / "models"
    logs_dir     = base_dir / "training_logs"
    results_dir  = base_dir / "results"
    for d in [models_dir, logs_dir, results_dir]:
        d.mkdir(exist_ok=True)

    # 选择设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[设备] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[设备] CPU")

    # 生成数据集（固定种子，所有方案共享同一份数据）
    set_seed(SEED)
    print("\n[数据] 生成训练数据...")
    dataset = generate_dataset(
        n_gaussian       = BASE_CONFIG["n_gaussian"],
        n_interp_per_pair= BASE_CONFIG["n_interp_per_pair"],
        n_perturb        = BASE_CONFIG["n_perturb"],
        val_ratio        = BASE_CONFIG["val_ratio"],
    )

    # 训练三种结构
    all_results = []
    for arch_name, hidden_dims in ARCHITECTURES.items():
        result = train_one_arch(
            arch_name, hidden_dims, dataset,
            BASE_CONFIG, device, models_dir, logs_dir,
        )
        all_results.append(result)

    # 选最佳
    best = min(all_results, key=lambda r: r["best_val_loss"])

    # 打印对比表
    print_comparison_table(all_results)
    print(f"\n[最佳] 方案 {best['arch']}  验证损失: {best['best_val_loss']:.6f}")

    # 打印风格参数表
    print_style_table(best)

    # 参数合理性检查
    print("\n[合理性检查]")
    any_issue = False
    for sname, params in best["styles"].items():
        issues = check_params_sanity(sname, params)
        if issues:
            print(f"  {sname}: {', '.join(issues)}")
            any_issue = True
    if not any_issue:
        print("  所有风格参数均在合理建筑范围内 ✓")

    # 与原模型对比
    original_val_loss = 0.00175225  # 从 trained_style_params.json 读取
    improvement = (original_val_loss - best["best_val_loss"]) / original_val_loss * 100
    print(f"\n[对比] 原模型验证损失: {original_val_loss:.6f}")
    print(f"       最佳新模型损失:  {best['best_val_loss']:.6f}")
    if improvement > 0:
        print(f"       改进幅度: -{improvement:.1f}%  ✓ 有改善")
    else:
        print(f"       变化幅度: {-improvement:.1f}%  (原模型已较优)")

    # 保存汇总报告
    report = {
        "seed": SEED,
        "original_model": {
            "hidden_dims":   [128, 64, 32],
            "best_val_loss": original_val_loss,
            "epochs_trained": 150,
        },
        "candidates": [
            {
                "arch":          r["arch"],
                "hidden_dims":   r["hidden_dims"],
                "n_params":      r["n_params"],
                "best_val_loss": r["best_val_loss"],
                "epochs_trained": r["epochs_trained"],
                "ckpt_path":     r["ckpt_path"],
            }
            for r in all_results
        ],
        "best_arch":       best["arch"],
        "best_val_loss":   best["best_val_loss"],
        "improvement_pct": round(improvement, 2),
        "style_params":    best["styles"],
    }

    report_path = results_dir / "optimization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[报告] 已保存: {report_path}")

    # 建议
    print("\n[建议]")
    if best["epochs_trained"] < BASE_CONFIG["epochs"]:
        print(f"  最佳模型在 epoch {best['epochs_trained']} 触发 early stopping，")
        print("  说明当前数据量已充分训练，可考虑扩充数据集以进一步提升。")
    else:
        print("  训练跑满 150 epochs，可尝试增加 epochs 或扩充数据集。")


if __name__ == "__main__":
    main()
