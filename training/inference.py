"""
LevelSmith Inference Module — Standalone
=========================================
独立推理模块，无需任何训练代码依赖。
依赖: torch, sentence-transformers, numpy

Pipeline:
  text → SentenceTransformer (384-dim) → StyleProjectionLayer (384→16)
       → StyleParamMLP (16→256→128→64→32→10) → denormalize → dict

Device priority: MPS (Mac Silicon) > CUDA > CPU

Usage:
    from inference import LevelSmithInference

    model = LevelSmithInference()                        # 自动加载
    params = model.predict("一个阴暗压抑的地下城")        # 文本 → 参数
    params = model.predict_style("fantasy_dungeon")      # 风格名 → 参数
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ─── 内联常量（不依赖 style_registry.py）────────────────────────────

# 输出参数元信息: key → (lo, hi, unit, description)
_OUTPUT_META: Dict[str, Tuple] = {
    "height_range_min":  (2.0,   6.0,  "m",  "楼层净高下限 [2.0, 6.0]"),
    "height_range_max":  (3.0,  20.0,  "m",  "楼层净高上限 [3.0, 20.0]"),
    "wall_thickness":    (0.1,   1.5,  "m",  "墙体厚度 [0.1, 1.5]"),
    "floor_thickness":   (0.1,   0.6,  "m",  "楼板厚度 [0.1, 0.6]"),
    "door_width":        (0.6,   3.0,  "m",  "门洞宽度 [0.6, 3.0]"),
    "door_height":       (1.8,   5.0,  "m",  "门洞高度 [1.8, 5.0]"),
    "win_width":         (0.3,   3.0,  "m",  "窗口宽度 [0.3, 3.0]"),
    "win_height":        (0.4,   3.0,  "m",  "窗口高度 [0.4, 3.0]"),
    "win_density":       (0.0,   1.0,  "",   "窗户密度比 [0.0, 1.0]"),
    "subdivision":       (1,     8,    "",   "空间细分等级 [1, 8] 整数"),
}
_OUTPUT_KEYS = list(_OUTPUT_META.keys())

# 20种风格特征向量（16-dim，值域 [0,1]）
# 维度含义: [建筑年代, 墙体密度, 结构复杂度, 装饰程度, 天花类型, 照明类型,
#            对称程度, 热质量, 窗密度, 门正式度, 地板材料, 屋顶类型,
#            内部分割, 气候适应, 安全等级, 垂直强调]
_STYLE_FEATURES: Dict[str, List[float]] = {
    "medieval":              [0.33,1.00,0.70,0.60,0.50,0.20,0.75,1.00,0.15,0.80,0.25,0.50,0.80,0.50,0.90,0.60],
    "modern":                [1.00,0.50,0.40,0.20,0.00,1.00,0.90,0.40,0.75,0.40,1.00,0.00,0.20,0.50,0.20,0.70],
    "industrial":            [0.66,0.70,0.55,0.05,0.00,0.70,0.50,0.75,0.40,0.50,0.75,0.00,0.60,0.50,0.60,0.80],
    "fantasy":               [0.28,0.80,0.88,0.92,0.70,0.30,0.85,0.65,0.45,0.88,0.30,0.75,0.55,0.45,0.65,0.88],
    "horror":                [0.35,0.88,0.60,0.30,0.35,0.05,0.30,0.92,0.10,0.42,0.22,0.40,0.88,0.85,0.80,0.25],
    "japanese":              [0.42,0.35,0.55,0.52,0.10,0.48,0.88,0.22,0.68,0.52,0.48,0.55,0.62,0.62,0.22,0.12],
    "desert":                [0.15,0.88,0.28,0.28,0.02,0.18,0.72,0.98,0.08,0.32,0.12,0.05,0.48,0.02,0.55,0.12],
    "medieval_chapel":       [0.33,0.95,0.50,0.70,0.60,0.25,0.90,0.90,0.25,0.65,0.25,0.65,0.60,0.50,0.60,0.70],
    "medieval_keep":         [0.33,1.00,0.60,0.20,0.40,0.10,0.70,1.00,0.08,0.50,0.25,0.60,0.90,0.60,1.00,0.95],
    "modern_loft":           [0.85,0.65,0.35,0.15,0.00,0.90,0.60,0.55,0.65,0.50,0.75,0.00,0.10,0.50,0.25,0.80],
    "modern_villa":          [1.00,0.35,0.55,0.60,0.00,1.00,0.80,0.30,0.85,0.70,1.00,0.00,0.25,0.40,0.30,0.65],
    "industrial_workshop":   [0.66,0.80,0.45,0.05,0.00,0.60,0.60,0.70,0.30,0.55,0.75,0.10,0.55,0.50,0.50,0.55],
    "industrial_powerplant": [0.75,0.90,0.70,0.02,0.00,0.55,0.65,0.90,0.15,0.60,0.75,0.05,0.35,0.50,0.70,0.95],
    "fantasy_dungeon":       [0.20,1.00,0.65,0.30,0.30,0.05,0.20,1.00,0.05,0.25,0.10,0.15,1.00,0.70,0.95,0.10],
    "fantasy_palace":        [0.25,0.85,0.95,1.00,0.90,0.45,0.95,0.70,0.60,1.00,0.30,0.90,0.20,0.45,0.70,0.95],
    "horror_asylum":         [0.55,0.70,0.45,0.10,0.10,0.30,0.70,0.65,0.30,0.25,0.75,0.15,0.95,0.80,0.85,0.20],
    "horror_crypt":          [0.30,1.00,0.55,0.20,0.25,0.02,0.35,1.00,0.02,0.20,0.20,0.10,1.00,0.90,0.90,0.05],
    "japanese_temple":       [0.38,0.45,0.75,0.80,0.15,0.40,1.00,0.35,0.30,0.85,0.50,0.65,0.50,0.60,0.45,0.55],
    "japanese_machiya":      [0.45,0.30,0.60,0.45,0.10,0.45,0.60,0.25,0.40,0.40,0.50,0.55,0.70,0.60,0.35,0.40],
    "desert_palace":         [0.20,0.85,0.75,0.85,0.70,0.30,0.90,0.95,0.30,0.90,0.20,0.55,0.40,0.05,0.65,0.70],
}

SUPPORTED_STYLES = list(_STYLE_FEATURES.keys())


# ─── 模型架构（内联，与训练脚本完全一致）──────────────────────────────

class _StyleProjectionLayer(nn.Module):
    """384-dim 文本嵌入 → 16-dim 风格特征向量"""
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _StyleParamMLP(nn.Module):
    """16-dim 特征向量 → 10-dim 归一化建筑参数"""
    def __init__(self, hidden_dims: List[int], dropout: float = 0.0):
        super().__init__()
        layers: list = []
        in_dim = 16
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 10), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── 工具函数 ─────────────────────────────────────────────────────────

def _denormalize(normalized: np.ndarray) -> Dict[str, object]:
    """归一化参数向量 → 物理参数字典（含单位和范围注释）"""
    result: Dict[str, object] = {}
    for i, key in enumerate(_OUTPUT_KEYS):
        lo, hi, unit, desc = _OUTPUT_META[key]
        val = float(normalized[i]) * (hi - lo) + lo
        if key == "subdivision":
            val = int(round(max(1, min(8, val))))
        else:
            val = round(val, 4)
        result[key] = {
            "value": val,
            "unit": unit,
            "range": [lo, hi],
            "note": desc,
        }
    return result


def _detect_device() -> torch.device:
    """优先级: MPS (Mac Silicon) > CUDA > CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_model_path(base_dir: Path) -> Path:
    """按优先级查找 MLP 权重文件"""
    candidates = [
        base_dir / "models" / "best_model_A.pt",
        base_dir / "best_model.pt",
        base_dir / "models" / "best_model_B.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"未找到 MLP 权重文件，已搜索:\n" + "\n".join(f"  {p}" for p in candidates)
    )


# ─── 主推理类 ─────────────────────────────────────────────────────────

class LevelSmithInference:
    """
    LevelSmith 独立推理模块

    自动加载 projection.pt 和 MLP 权重，支持 MPS/CUDA/CPU。

    Args:
        model_dir:   包含权重文件的目录（默认为本脚本所在目录）
        encoder_name: sentence-transformers 模型名称
        verbose:     是否打印加载日志

    Example:
        model = LevelSmithInference()
        params = model.predict("一个阴暗压抑的地下城")
        params = model.predict_style("fantasy_dungeon")
    """

    _ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        encoder_name: str = _ENCODER_NAME,
        verbose: bool = True,
    ):
        self._verbose = verbose
        self._base = Path(model_dir) if model_dir else Path(__file__).parent
        self._device = _detect_device()
        self._log(f"Device: {self._device}")

        self._projection: Optional[_StyleProjectionLayer] = None
        self._encoder = None
        self._mlp = self._load_mlp()

    def _log(self, msg: str):
        if self._verbose:
            print(f"[LevelSmith] {msg}")

    # ── 模型加载 ──────────────────────────────────────────────────────

    def _load_mlp(self) -> _StyleParamMLP:
        path = _find_model_path(self._base)
        self._log(f"Loading MLP: {path.name}")
        ckpt = torch.load(str(path), map_location=self._device, weights_only=False)

        hidden_dims = ckpt.get("hidden_dims") or ckpt.get("config", {}).get("hidden_dims", [256, 128, 64, 32])
        mlp = _StyleParamMLP(hidden_dims, dropout=0.0).to(self._device)
        mlp.load_state_dict(ckpt["model_state_dict"])
        mlp.eval()

        val_loss = ckpt.get("best_val_loss", "?")
        n_styles = ckpt.get("n_styles", "?")
        self._log(f"MLP loaded | arch: 16->{hidden_dims}->10 | val_loss={val_loss} | styles={n_styles}")
        return mlp

    def _ensure_text_pipeline(self):
        """懒加载：仅在调用 predict() 时加载 sentence-transformers"""
        if self._encoder is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装。请运行:\n"
                "  pip install sentence-transformers"
            )

        self._log(f"Loading encoder: {self._ENCODER_NAME} ...")
        self._encoder = SentenceTransformer(self._ENCODER_NAME)

        proj_path = self._base / "projection.pt"
        if not proj_path.exists():
            raise FileNotFoundError(f"未找到投影层权重: {proj_path}")
        self._log(f"Loading projection: {proj_path.name}")
        ckpt = torch.load(str(proj_path), map_location=self._device, weights_only=False)
        proj = _StyleProjectionLayer(
            input_dim=ckpt.get("input_dim", 384),
            hidden_dim=ckpt.get("hidden_dim", 128),
            output_dim=ckpt.get("output_dim", 16),
        ).to(self._device)
        proj.load_state_dict(ckpt["model_state_dict"])
        proj.eval()
        self._projection = proj
        self._log(f"Projection loaded | val_loss={ckpt.get('best_val_loss', '?')}")

    # ── 推理函数 ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _fv_to_params(self, fv: np.ndarray) -> Dict[str, object]:
        """特征向量 (16-dim) → 参数字典"""
        x = torch.from_numpy(fv).unsqueeze(0).to(self._device)
        out = self._mlp(x).squeeze(0).cpu().numpy()
        return _denormalize(out)

    def _cosine_similarities(self, query_emb: np.ndarray) -> Dict[str, float]:
        """计算 query 与所有风格的余弦相似度"""
        sims = {}
        for name, fv_list in _STYLE_FEATURES.items():
            # 用风格描述已编码的嵌入不在这里（避免重复编码）
            # 直接用简单特征向量点积作为近似风格匹配（此处不需要精确匹配，仅供参考）
            _ = fv_list  # 特征向量是建筑参数空间，不能和语义嵌入比较
        # 实际相似度由 projection 输出的 fv 与注册特征向量计算
        style_fvs = {n: np.array(v, dtype=np.float32) for n, v in _STYLE_FEATURES.items()}
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        for name, fv in style_fvs.items():
            fv_norm = fv / (np.linalg.norm(fv) + 1e-8)
            sims[name] = float(np.dot(q, fv_norm))
        return sims

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, object]:
        """
        自然语言描述 → 建筑参数

        Args:
            text: 中文或英文建筑描述

        Returns:
            dict with keys:
              - params:          建筑参数字典（含 value/unit/range/note）
              - dominant_style:  最相似的风格名称
              - feature_vector:  16-dim 特征向量（列表）
              - input_text:      原始输入文本

        Example:
            result = model.predict("一个阴暗压抑的地下城")
            print(result["params"]["wall_thickness"]["value"])  # 0.897
            print(result["dominant_style"])                      # "fantasy_dungeon"
        """
        self._ensure_text_pipeline()

        # Step 1: 文本 → 384-dim 语义嵌入
        emb: np.ndarray = self._encoder.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )

        # Step 2: 投影层 384 → 16
        emb_t = torch.from_numpy(emb).unsqueeze(0).to(self._device)
        fv = self._projection(emb_t).squeeze(0).cpu().numpy()
        fv = np.clip(fv, 0.0, 1.0)

        # Step 3: 计算与各风格特征向量的余弦相似度（用于 dominant_style 识别）
        sims = self._cosine_similarities(fv)
        dominant = max(sims, key=sims.get)

        # Step 4: MLP 16 → 10 → 反归一化
        params = self._fv_to_params(fv)

        return {
            "input_text":      text,
            "params":          params,
            "dominant_style":  dominant,
            "style_scores":    {k: round(v, 4) for k, v in sorted(sims.items(), key=lambda x: -x[1])},
            "feature_vector":  fv.tolist(),
        }

    def predict_style(self, style_name: str) -> Dict[str, object]:
        """
        风格名称 → 建筑参数（不需要加载 sentence-transformers）

        Args:
            style_name: 20种风格之一，见 SUPPORTED_STYLES

        Returns:
            dict with keys:
              - params:        建筑参数字典（含 value/unit/range/note）
              - style_name:    风格名称
              - feature_vector: 16-dim 特征向量（列表）

        Supported styles:
            medieval, modern, industrial, fantasy, horror, japanese, desert,
            medieval_chapel, medieval_keep, modern_loft, modern_villa,
            industrial_workshop, industrial_powerplant,
            fantasy_dungeon, fantasy_palace,
            horror_asylum, horror_crypt,
            japanese_temple, japanese_machiya, desert_palace

        Example:
            result = model.predict_style("japanese_temple")
            print(result["params"]["win_density"]["value"])  # 0.313
        """
        if style_name not in _STYLE_FEATURES:
            raise ValueError(
                f"未知风格: '{style_name}'\n"
                f"支持的风格: {SUPPORTED_STYLES}"
            )
        fv = np.array(_STYLE_FEATURES[style_name], dtype=np.float32)
        params = self._fv_to_params(fv)
        return {
            "style_name":    style_name,
            "params":        params,
            "feature_vector": fv.tolist(),
        }

    def predict_raw(self, style_name: str) -> Dict[str, float]:
        """
        返回简洁的纯数值参数字典（不含单位注释），方便下游代码直接使用。

        Args:
            style_name: 风格名称

        Returns:
            {"height_range_min": 2.16, "wall_thickness": 0.897, ..., "subdivision": 8}
        """
        result = self.predict_style(style_name)
        return {k: v["value"] for k, v in result["params"].items()}

    def predict_text_raw(self, text: str) -> Dict[str, float]:
        """
        文本描述 → 简洁纯数值参数字典。

        Args:
            text: 自然语言描述

        Returns:
            {"height_range_min": 2.16, "wall_thickness": 0.897, ..., "subdivision": 8}
        """
        result = self.predict(text)
        return {k: v["value"] for k, v in result["params"].items()}

    # ── 格式化输出 ────────────────────────────────────────────────────

    def print_params(self, result: Dict) -> None:
        """打印参数字典的格式化预览"""
        params = result.get("params", {})
        title = result.get("input_text") or result.get("style_name", "")
        dominant = result.get("dominant_style", "")
        print(f"\n{'='*58}")
        if title:
            print(f"  Input: {title}")
        if dominant:
            print(f"  Dominant style: {dominant}")
        print(f"{'='*58}")
        print(f"  {'Parameter':<22} {'Value':>8}  Unit   Range")
        print(f"  {'-'*52}")
        for key, info in params.items():
            val = info["value"]
            unit = info["unit"] or "-"
            lo, hi = info["range"]
            if isinstance(val, int):
                print(f"  {key:<22} {val:>8d}  {unit:<5}  [{lo}, {hi}]")
            else:
                print(f"  {key:<22} {val:>8.3f}  {unit:<5}  [{lo}, {hi}]")
        print(f"  {'-'*52}")


# ─── 便捷函数（模块级，兼容函数式调用）─────────────────────────────────

_default_model: Optional[LevelSmithInference] = None


def _get_default() -> LevelSmithInference:
    global _default_model
    if _default_model is None:
        _default_model = LevelSmithInference()
    return _default_model


def predict(text: str) -> Dict[str, object]:
    """
    模块级便捷函数：文本 → 建筑参数。

    首次调用时自动初始化模型（懒加载 sentence-transformers）。

    Args:
        text: 中英文建筑描述

    Returns:
        完整结果字典，params 字段含物理单位和范围

    Example:
        from inference import predict
        result = predict("一个阴暗压抑的地下城")
        wall = result["params"]["wall_thickness"]["value"]  # 0.897 m
    """
    return _get_default().predict(text)


def predict_style(style_name: str) -> Dict[str, object]:
    """
    模块级便捷函数：风格名称 → 建筑参数（无需 sentence-transformers）。

    Args:
        style_name: 20种风格之一

    Returns:
        完整结果字典，params 字段含物理单位和范围

    Example:
        from inference import predict_style
        result = predict_style("medieval_keep")
        wall = result["params"]["wall_thickness"]["value"]  # 1.181 m
    """
    return _get_default().predict_style(style_name)


# ─── CLI Demo ─────────────────────────────────────────────────────────

def _demo():
    """运行推理示例，保存结果到 inference_demo.json"""
    print(f"\n{'#'*60}")
    print("  LevelSmith Inference Demo")
    print(f"{'#'*60}")

    model = LevelSmithInference()

    # ── 1. predict_style（不需要 sentence-transformers）─────────────
    print(f"\n{'─'*60}")
    print("  [1] predict_style — 20种风格参数")
    print(f"{'─'*60}")

    style_results = {}
    for style in SUPPORTED_STYLES:
        r = model.predict_style(style)
        style_results[style] = r
        raw = {k: v["value"] for k, v in r["params"].items()}
        h, H = raw["height_range_min"], raw["height_range_max"]
        print(f"  {style:<24} h={h:.2f}~{H:.2f}m  wall={raw['wall_thickness']:.3f}m  "
              f"win_den={raw['win_density']:.3f}  sub={raw['subdivision']}")

    # ── 2. predict（需要 sentence-transformers）──────────────────────
    print(f"\n{'─'*60}")
    print("  [2] predict — 自然语言描述")
    print(f"{'─'*60}")

    test_texts = [
        "一个阴暗压抑的地下城，厚重的石墙，几乎没有光源",
        "明亮通透的现代别墅，落地玻璃幕墙，开放式空间",
        "奇幻魔法宫殿大厅，极高穹顶，华丽彩窗",
        "废弃精神病院，铁格窗，压抑的走廊布局",
        "A rusty industrial warehouse with exposed steel beams",
        "Traditional Japanese tea house with sliding shoji screens",
    ]

    text_results = []
    for text in test_texts:
        r = model.predict(text)
        text_results.append(r)
        model.print_params(r)

    # ── 3. 保存结果 ────────────────────────────────────────────────
    out_path = Path(__file__).parent / "inference_demo.json"
    output = {
        "device": str(model._device),
        "style_predictions": {
            name: {k: v["value"] for k, v in r["params"].items()}
            for name, r in style_results.items()
        },
        "text_predictions": [
            {
                "text": r["input_text"],
                "dominant_style": r["dominant_style"],
                "params": {k: v["value"] for k, v in r["params"].items()},
            }
            for r in text_results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    _demo()
