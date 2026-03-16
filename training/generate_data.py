"""
LevelSmith 训练数据生成器
基于 style_registry.py 中7个风格，通过3种扩增策略生成训练数据集。

扩增策略:
  1. Gaussian Noise    — 在特征向量和参数上叠加高斯噪声
  2. Interpolation     — 在两个风格之间线性插值 (混合风格)
  3. Param Perturbation— 固定风格特征，对输出参数施加更大扰动 (模拟设计变体)
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from style_registry import (
    STYLE_REGISTRY, FEATURE_DIM, OUTPUT_DIM, OUTPUT_KEYS,
    OUTPUT_PARAMS, get_feature_vector, get_param_vector, denormalize_params,
    get_style_bounds_normalized,
)

RNG = np.random.default_rng(42)


# ─── 风格级边界约束 ───────────────────────────────────────────

def clamp_to_style_bounds(pvs: np.ndarray, style_name: str) -> np.ndarray:
    """
    将归一化参数矩阵 [N, OUTPUT_DIM] 按风格级边界约束。
    比全局 clip(0,1) 更紧：防止扩增数据漂移到其他风格的参数空间。
    """
    bounds = get_style_bounds_normalized(style_name)  # [OUTPUT_DIM, 2]
    return np.clip(pvs, bounds[:, 0], bounds[:, 1])


# ─── 扩增策略 ────────────────────────────────────────────────

def augment_gaussian(
    fv: np.ndarray,
    pv: np.ndarray,
    n: int,
    style_name: str,
    feature_std: float = 0.03,
    param_std: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray]:
    """策略1: 高斯噪声扩增（参数严格限制在风格级边界内）"""
    fv_noise = RNG.normal(0, feature_std, (n, FEATURE_DIM)).astype(np.float32)
    pv_noise = RNG.normal(0, param_std,   (n, OUTPUT_DIM)).astype(np.float32)
    fvs = np.clip(fv + fv_noise, 0.0, 1.0)
    pvs = clamp_to_style_bounds(pv + pv_noise, style_name)
    return fvs, pvs


def augment_interpolation(
    style_names: List[str],
    n_per_pair: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """策略2: 风格插值扩增 — 在两两风格之间随机插值"""
    fvs_list, pvs_list = [], []
    pairs = [
        (style_names[i], style_names[j])
        for i in range(len(style_names))
        for j in range(i + 1, len(style_names))
    ]
    for s1, s2 in pairs:
        fv1, fv2 = get_feature_vector(s1), get_feature_vector(s2)
        pv1, pv2 = get_param_vector(s1),   get_param_vector(s2)
        alphas = RNG.uniform(0.0, 1.0, (n_per_pair, 1)).astype(np.float32)
        fvs_list.append(fv1 * (1 - alphas) + fv2 * alphas)
        pvs_list.append(pv1 * (1 - alphas) + pv2 * alphas)
    return np.vstack(fvs_list), np.vstack(pvs_list)


def augment_param_perturbation(
    fv: np.ndarray,
    pv: np.ndarray,
    n: int,
    style_name: str,
    param_std: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """策略3: 参数扰动扩增 — 特征保持不变，参数施加较大扰动（模拟设计变体）
    参数被约束在该风格的 param_bounds 内，不会越界到其他风格空间。
    """
    fvs = np.tile(fv, (n, 1))
    noise = RNG.normal(0, param_std, (n, OUTPUT_DIM)).astype(np.float32)
    pvs = clamp_to_style_bounds(pv + noise, style_name)
    return fvs.astype(np.float32), pvs


# ─── 主生成函数 ──────────────────────────────────────────────

def generate_dataset(
    n_gaussian: int = 1000,
    n_interp_per_pair: int = 500,
    n_perturb: int = 800,
    val_ratio: float = 0.15,
    save_dir: str = None,
) -> Dict[str, np.ndarray]:
    """
    生成完整训练/验证数据集。

    Returns:
        dict with keys: X_train, y_train, X_val, y_val
        X shape: [N, 16], y shape: [N, 10]
    """
    style_names = list(STYLE_REGISTRY.keys())
    all_fvs, all_pvs = [], []

    print("生成训练数据...")
    for name in style_names:
        fv = get_feature_vector(name)
        pv = get_param_vector(name)

        # 原始样本
        all_fvs.append(fv.reshape(1, -1))
        all_pvs.append(pv.reshape(1, -1))

        # 策略1: 高斯噪声（风格级边界约束）
        fvs_g, pvs_g = augment_gaussian(fv, pv, n_gaussian, style_name=name)
        all_fvs.append(fvs_g)
        all_pvs.append(pvs_g)
        print(f"  [{name}] 高斯噪声: +{n_gaussian} 样本")

        # 策略3: 参数扰动（风格级边界约束）
        fvs_p, pvs_p = augment_param_perturbation(fv, pv, n_perturb, style_name=name)
        all_fvs.append(fvs_p)
        all_pvs.append(pvs_p)
        print(f"  [{name}] 参数扰动: +{n_perturb} 样本")

    # 策略2: 风格插值 (跨风格)
    fvs_i, pvs_i = augment_interpolation(style_names, n_interp_per_pair)
    all_fvs.append(fvs_i)
    all_pvs.append(pvs_i)
    print(f"  [interpolation] 风格插值: +{len(fvs_i)} 样本 ({len(style_names)*(len(style_names)-1)//2} 对 × {n_interp_per_pair})")

    X = np.vstack(all_fvs).astype(np.float32)
    y = np.vstack(all_pvs).astype(np.float32)

    # 打乱顺序
    idx = RNG.permutation(len(X))
    X, y = X[idx], y[idx]

    # 划分训练/验证集
    n_val = int(len(X) * val_ratio)
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    print(f"\n数据集汇总:")
    print(f"  总样本数  : {len(X)}")
    print(f"  训练集    : {len(X_train)}")
    print(f"  验证集    : {len(X_val)}")
    print(f"  特征维度  : {X.shape[1]}")
    print(f"  输出维度  : {y.shape[1]}")

    dataset = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val":   X_val,
        "y_val":   y_val,
    }

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "X_train.npy", X_train)
        np.save(save_path / "y_train.npy", y_train)
        np.save(save_path / "X_val.npy",   X_val)
        np.save(save_path / "y_val.npy",   y_val)
        # 同时保存元数据
        meta = {
            "feature_dim": int(X.shape[1]),
            "output_dim":  int(y.shape[1]),
            "output_keys": OUTPUT_KEYS,
            "output_params": {k: {"range": list(v["range"]), "unit": v["unit"]} for k, v in OUTPUT_PARAMS.items()},
            "styles": list(style_names),
            "augmentation": {
                "gaussian_per_style":   n_gaussian,
                "interp_per_pair":      n_interp_per_pair,
                "perturb_per_style":    n_perturb,
            },
            "n_train": int(len(X_train)),
            "n_val":   int(len(X_val)),
        }
        with open(save_path / "dataset_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\n数据已保存至: {save_path}")

    return dataset


if __name__ == "__main__":
    ds = generate_dataset(save_dir="data")
    # 打印几条样本的反归一化结果作验证
    print("\n── 样本验证 (前3条训练样本) ──")
    for i in range(3):
        params = denormalize_params(ds["y_train"][i])
        print(f"  样本 {i}: {params}")
