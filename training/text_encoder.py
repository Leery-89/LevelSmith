"""
LevelSmith Text Encoder
自然语言风格描述 → 结构参数完整 Pipeline。

Pipeline:
  自然语言描述
    → sentence-transformers (384-dim 语义嵌入)
    → StyleProjectionLayer (384→16，监督训练的投影层)
    → StyleParamMLP (16→10，归一化结构参数)
    → denormalize → 物理参数值

StyleProjectionLayer 训练策略:
  - 监督信号: 7 个已知风格的 (描述文本嵌入, 特征向量) 对
  - 数据增强: 嵌入空间高斯噪声 (800条/风格) + 跨风格线性插值 (400条/对)
  - 损失函数: MSE
  - 保存路径: projection.pt

默认编码器: paraphrase-multilingual-MiniLM-L12-v2
  - 同属 MiniLM 轻量系列，支持 50+ 语言含中文
  - all-MiniLM-L6-v2 仅支持英文，不适用于中文描述
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from style_registry import (
    STYLE_REGISTRY, OUTPUT_KEYS, OUTPUT_PARAMS,
    get_feature_vector, get_style_bounds_normalized, denormalize_params,
)
from model import StyleParamMLP, clamp_output_to_style_bounds

# 支持的编码器
ENCODER_MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"  # 中英文均可
ENCODER_ENGLISH_ONLY = "all-MiniLM-L6-v2"                       # 仅英文

PROJECTION_FILE = Path(__file__).parent / "projection.pt"


# ─── 监督投影层 ──────────────────────────────────────────────────

class StyleProjectionLayer(nn.Module):
    """
    384-dim 文本嵌入 → 16-dim 风格特征向量（监督投影层）

    通过 7 个已知风格的 (描述文本嵌入, 特征向量) 对进行监督训练，
    直接学习语义嵌入空间到风格特征空间的映射，
    避免余弦相似度加权平均导致的中心化收敛问题。
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, output_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # 输出归一化到 [0, 1]，与特征向量空间一致
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _augment_projection_data(
    embeddings: np.ndarray,    # [N_styles, 384]
    feature_vecs: np.ndarray,  # [N_styles, 16]
    n_noise: int = 800,
    n_interp: int = 400,
    noise_std: float = 0.012,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 N 个锚点扩增训练样本：
      1. 嵌入空间高斯噪声（每个锚点生成 n_noise 条）
      2. 跨风格线性插值（每对风格随机采样 n_interp 条 alpha）

    Returns: (aug_embs [M, 384], aug_fvs [M, 16])
    """
    rng = np.random.default_rng(seed)
    N = len(embeddings)
    aug_embs, aug_fvs = [embeddings], [feature_vecs]  # 包含原始锚点

    # 1. 高斯噪声增强
    for i in range(N):
        noise = rng.normal(0, noise_std, (n_noise, embeddings.shape[1])).astype(np.float32)
        aug_embs.append(embeddings[i] + noise)
        aug_fvs.append(np.tile(feature_vecs[i], (n_noise, 1)))

    # 2. 跨风格插值（alpha 均匀采样，两端各留 5% 余量）
    alphas = rng.uniform(0.05, 0.95, n_interp).astype(np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            a = alphas
            aug_embs.append(a[:, None] * embeddings[i] + (1 - a[:, None]) * embeddings[j])
            aug_fvs.append(a[:, None] * feature_vecs[i] + (1 - a[:, None]) * feature_vecs[j])

    return (
        np.vstack(aug_embs).astype(np.float32),
        np.vstack(aug_fvs).astype(np.float32),
    )


def train_projection_layer(
    anchor_embeddings: np.ndarray,    # [N_styles, 384]
    anchor_feature_vecs: np.ndarray,  # [N_styles, 16]
    device: str = "cpu",
    epochs: int = 1000,
    lr: float = 1e-3,
    batch_size: int = 256,
    patience: int = 100,
    save_path: Optional[Path] = None,
) -> StyleProjectionLayer:
    """
    监督训练 StyleProjectionLayer。
    Returns: 训练好的投影层（eval 模式）
    """
    print(f"[Projection] 生成增强训练数据 ...")
    aug_embs, aug_fvs = _augment_projection_data(anchor_embeddings, anchor_feature_vecs)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(aug_embs))
    aug_embs = aug_embs[idx]
    aug_fvs  = aug_fvs[idx]

    split = int(len(aug_embs) * 0.8)
    trn_emb, val_emb = aug_embs[:split],  aug_embs[split:]
    trn_fv,  val_fv  = aug_fvs[:split],   aug_fvs[split:]

    print(f"[Projection] 训练样本: {len(trn_emb)} | 验证样本: {len(val_emb)}")

    trn_emb_t = torch.from_numpy(trn_emb).to(device)
    trn_fv_t  = torch.from_numpy(trn_fv).to(device)
    val_emb_t = torch.from_numpy(val_emb).to(device)
    val_fv_t  = torch.from_numpy(val_fv).to(device)

    proj = StyleProjectionLayer().to(device)
    optimizer = optim.AdamW(proj.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    print(f"[Projection] 开始训练 (device={device}, epochs={epochs}) ...")
    for epoch in range(1, epochs + 1):
        proj.train()
        perm = torch.randperm(len(trn_emb_t), device=device)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(trn_emb_t), batch_size):
            bi = perm[start:start + batch_size]
            loss = criterion(proj(trn_emb_t[bi]), trn_fv_t[bi])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        proj.eval()
        with torch.no_grad():
            val_loss = criterion(proj(val_emb_t), val_fv_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 100 == 0 or epoch == 1:
            print(f"  epoch {epoch:5d} | train={total_loss/n_batches:.6f} "
                  f"| val={val_loss:.6f} | best={best_val_loss:.6f}")

        if patience_cnt >= patience:
            print(f"[Projection] 早停 (连续 {patience} epoch 验证损失无改善，已训练 {epoch} epochs)")
            break

    if best_state is not None:
        proj.load_state_dict(best_state)
    proj.eval()

    if save_path:
        torch.save({
            "model_state_dict": proj.state_dict(),
            "input_dim":  384,
            "hidden_dim": 128,
            "output_dim": 16,
            "best_val_loss": best_val_loss,
        }, str(save_path))
        print(f"[Projection] 已保存: {save_path} | 最佳验证损失: {best_val_loss:.6f}")

    return proj


# ─── TextStyleEncoder ─────────────────────────────────────────────

class TextStyleEncoder:
    """
    文本 → 16-dim 风格特征向量

    使用监督训练的 StyleProjectionLayer 将 384-dim 语义嵌入映射到
    风格特征空间，比余弦相似度加权平均更能区分不同风格。
    同时保留余弦相似度用于 dominant_style 识别和调试。
    """

    def __init__(
        self,
        model_name: str = ENCODER_MULTILINGUAL,
        temperature: float = 0.3,
        device: Optional[str] = None,
        force_retrain: bool = False,
    ):
        """
        Args:
            model_name:    sentence-transformers 模型名称
            temperature:   余弦相似度 softmax 温度（用于风格识别信息，不影响投影）
            device:        推理设备
            force_retrain: 强制重新训练投影层（忽略已保存的 projection.pt）
        """
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[TextEncoder] 加载 {model_name} ...")
        self.encoder = SentenceTransformer(model_name)

        self._style_names: List[str] = list(STYLE_REGISTRY.keys())
        self._style_descs: List[str] = [STYLE_REGISTRY[n].description for n in self._style_names]
        self._style_fvs: np.ndarray  = np.stack(
            [get_feature_vector(n) for n in self._style_names]
        )  # [N_styles, 16]

        print(f"[TextEncoder] 预计算 {len(self._style_names)} 个风格嵌入 ...")
        self._style_embeddings: np.ndarray = self.encoder.encode(
            self._style_descs,
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # [N_styles, 384]
        print(f"[TextEncoder] 就绪 | 风格: {self._style_names}")

        self.projection = self._load_or_train_projection(force_retrain)

    def _load_or_train_projection(self, force_retrain: bool) -> StyleProjectionLayer:
        if not force_retrain and PROJECTION_FILE.exists():
            print(f"[Projection] 加载已有模型: {PROJECTION_FILE}")
            ckpt = torch.load(str(PROJECTION_FILE), map_location=self.device, weights_only=False)
            proj = StyleProjectionLayer(
                input_dim=ckpt.get("input_dim", 384),
                hidden_dim=ckpt.get("hidden_dim", 128),
                output_dim=ckpt.get("output_dim", 16),
            ).to(self.device)
            proj.load_state_dict(ckpt["model_state_dict"])
            proj.eval()
            print(f"[Projection] 加载完成 | 历史验证损失: {ckpt.get('best_val_loss', 'N/A'):.6f}")
            return proj
        return train_projection_layer(
            anchor_embeddings=self._style_embeddings,
            anchor_feature_vecs=self._style_fvs,
            device=self.device,
            save_path=PROJECTION_FILE,
        )

    def encode_text(self, text: str) -> np.ndarray:
        """文本 → L2 归一化语义嵌入 (384-dim)"""
        return self.encoder.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )

    def similarity_scores(self, text: str) -> Dict[str, float]:
        """返回输入文本与每个风格的余弦相似度 dict"""
        query = self.encode_text(text)
        sims = self._style_embeddings @ query
        return {name: float(sims[i]) for i, name in enumerate(self._style_names)}

    @torch.no_grad()
    def text_to_feature_vector(
        self, text: str
    ) -> Tuple[np.ndarray, Dict]:
        """
        文本 → (16-dim 特征向量, 调试信息)

        特征向量由监督投影层产生，调试信息包含余弦相似度和 softmax 权重。
        """
        query = self.encode_text(text)  # [384]

        # ① 监督投影：384 → 16
        emb_t = torch.from_numpy(query).unsqueeze(0).to(self.device)
        fv = self.projection(emb_t).squeeze(0).cpu().numpy()  # [16]
        fv = np.clip(fv, 0.0, 1.0)

        # ② 余弦相似度（用于 dominant_style 识别和调试）
        sims = self._style_embeddings @ query  # [N_styles]
        scaled = sims / self.temperature
        scaled -= scaled.max()
        weights = np.exp(scaled)
        weights /= weights.sum()

        sorted_idx = np.argsort(sims)[::-1]
        info = {
            "similarities": {n: round(float(sims[i]),    4) for i, n in enumerate(self._style_names)},
            "weights":      {n: round(float(weights[i]), 4) for i, n in enumerate(self._style_names)},
            "dominant_style":  self._style_names[sorted_idx[0]],
            "top2_styles": [self._style_names[i] for i in sorted_idx[:2]],
        }
        return fv, info


# ─── 完整 Pipeline ────────────────────────────────────────────────

class TextToStructuralParams:
    """
    完整 Pipeline：自然语言描述 → 结构参数

    text → TextStyleEncoder (384→16) → StyleParamMLP (16→10) → params
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        encoder_model: str = ENCODER_MULTILINGUAL,
        temperature: float = 0.3,
        device: Optional[str] = None,
        force_retrain_projection: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        script_dir = Path(__file__).parent

        self.text_encoder = TextStyleEncoder(
            model_name=encoder_model,
            temperature=temperature,
            device=self.device,
            force_retrain=force_retrain_projection,
        )

        ckpt_path = Path(model_path) if model_path else script_dir / "best_model.pt"
        print(f"\n[Pipeline] 加载 MLP: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)

        cfg = ckpt["config"]
        self.mlp = StyleParamMLP(
            input_dim=ckpt.get("feature_dim", 16),
            output_dim=ckpt.get("output_dim", 10),
            hidden_dims=cfg.get("hidden_dims", [128, 64, 32]),
            dropout=0.0,
        ).to(self.device)
        self.mlp.load_state_dict(ckpt["model_state_dict"])
        self.mlp.eval()
        print(f"[Pipeline] 就绪 | 验证损失: {cfg.get('_best_val_loss', 'N/A')}")

    @torch.no_grad()
    def predict(self, text: str, apply_style_clamp: bool = True) -> Dict:
        """
        文本描述 → 结构参数

        Args:
            text:              自然语言描述（中文/英文均可）
            apply_style_clamp: dominant_style 余弦相似度 > 0.35 时施加风格边界约束

        Returns:
            dict 包含 input_text / params / style_info / feature_vector / normalized
        """
        fv, style_info = self.text_encoder.text_to_feature_vector(text)

        fv_t = torch.from_numpy(fv).unsqueeze(0).to(self.device)
        pred = self.mlp(fv_t)

        dominant = style_info["dominant_style"]
        dominant_sim = style_info["similarities"][dominant]
        clamped = apply_style_clamp and dominant_sim > 0.35
        if clamped:
            pred = clamp_output_to_style_bounds(pred, dominant, self.device)

        pred_np = pred.squeeze(0).cpu().numpy()
        raw = denormalize_params(pred_np)

        params = {
            "height_range":    [raw["height_range_min"], raw["height_range_max"]],
            "wall_thickness":  raw["wall_thickness"],
            "floor_thickness": raw["floor_thickness"],
            "door_spec":       {"width": raw["door_width"],  "height": raw["door_height"]},
            "win_spec":        {"width": raw["win_width"],   "height": raw["win_height"],
                                "density": raw["win_density"]},
            "subdivision":     int(raw["subdivision"]),
        }

        return {
            "input_text":          text,
            "params":              params,
            "style_info":          style_info,
            "style_clamp_applied": clamped,
            "feature_vector":      fv.tolist(),
            "normalized":          pred_np.tolist(),
        }

    def predict_and_print(self, text: str, apply_style_clamp: bool = True) -> Dict:
        """推理并打印格式化结果，返回结果 dict"""
        result = self.predict(text, apply_style_clamp=apply_style_clamp)
        style_info = result["style_info"]
        p = result["params"]

        print(f"\n{'═'*62}")
        print(f"  输入: 「{text}」")
        print(f"{'═'*62}")

        print("\n  风格匹配 (余弦相似度 → softmax 权重):")
        sims = style_info["similarities"]
        for name in sorted(sims, key=sims.get, reverse=True):
            w = style_info["weights"][name]
            bar_len = int(w * 36)
            bar = "█" * bar_len + "░" * (36 - bar_len)
            marker = " ◀ 主导" if name == style_info["dominant_style"] else ""
            print(f"  {name:12} {sims[name]:+.3f} │{bar}│ {w:.3f}{marker}")

        clamp_note = (f"（已施加 {style_info['dominant_style']} 边界约束）"
                      if result["style_clamp_applied"] else "")
        print(f"\n  预测结构参数 {clamp_note}")
        print(f"  {'─'*46}")
        print(f"  height_range    {p['height_range'][0]:.2f}m  ~  {p['height_range'][1]:.2f}m")
        print(f"  wall_thickness  {p['wall_thickness']:.3f} m")
        print(f"  floor_thickness {p['floor_thickness']:.3f} m")
        print(f"  door_spec       {p['door_spec']['width']:.2f}m × {p['door_spec']['height']:.2f}m")
        print(f"  win_spec        {p['win_spec']['width']:.2f}m × {p['win_spec']['height']:.2f}m"
              f"  density={p['win_spec']['density']:.2f}")
        print(f"  subdivision     {p['subdivision']}")

        return result


# ─── 入口 / Demo ──────────────────────────────────────────────

def main():
    pipeline = TextToStructuralParams(temperature=0.3)

    # ── 主对比测试 ──────────────────────────────────────────────
    print("\n" + "▓" * 62)
    print("  对比测试：地下城 vs 日式茶室")
    print("▓" * 62)

    comparison_queries = [
        "一个阴暗压抑的地下城",
        "明亮通透的日式茶室",
    ]
    comparison_results = []
    for q in comparison_queries:
        r = pipeline.predict_and_print(q)
        comparison_results.append(r)

    r_a, r_b = comparison_results
    p_a, p_b = r_a["params"], r_b["params"]
    print(f"\n{'─'*62}")
    print(f"  参数对比")
    print(f"  {'─'*58}")
    print(f"  {'参数':<18} {'地下城':>10}   {'日式茶室':>10}   {'差值(茶-地)':>10}")
    print(f"  {'─'*58}")
    rows = [
        ("height_min(m)",    p_a["height_range"][0],     p_b["height_range"][0]),
        ("height_max(m)",    p_a["height_range"][1],     p_b["height_range"][1]),
        ("wall_thick(m)",    p_a["wall_thickness"],      p_b["wall_thickness"]),
        ("floor_thick(m)",   p_a["floor_thickness"],     p_b["floor_thickness"]),
        ("door_h(m)",        p_a["door_spec"]["height"], p_b["door_spec"]["height"]),
        ("win_density",      p_a["win_spec"]["density"], p_b["win_spec"]["density"]),
        ("subdivision",      float(p_a["subdivision"]),  float(p_b["subdivision"])),
    ]
    for name, va, vb in rows:
        print(f"  {name:<18} {va:>10.3f}   {vb:>10.3f}   {vb - va:>+10.3f}")

    # ── 扩展测试 ────────────────────────────────────────────────
    print(f"\n{'▓'*62}")
    print("  扩展测试：更多风格描述")
    print("▓" * 62)

    extra_queries = [
        "高耸入云的奇幻魔法城堡",
        "明亮宽敞的现代办公大楼",
        "炎热沙漠中的古老土坯堡垒",
        "rusty industrial warehouse with exposed steel beams",
        "A dark oppressive dungeon",
    ]
    extra_results = []
    for q in extra_queries:
        r = pipeline.predict_and_print(q)
        extra_results.append(r)

    out_path = Path(__file__).parent / "text_encoder_demo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results + extra_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Demo] 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
