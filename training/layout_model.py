"""
layout_model.py
自回归布局生成模型：基于 OSM 真实城镇数据学习建筑空间分布规律。

模型：Decoder-only Transformer（因果掩码，自回归）
  输入：已放置的建筑序列 → 输出：下一栋建筑的位置/朝向/类型/尺寸
  每个建筑 token = [nx_norm, ny_norm, sin_ori, cos_ori, len_norm, wid_norm, type_id]

数据：training_data/osm_layouts.json（22,363 栋真实建筑，4个中世纪城镇）

用法：
  python layout_model.py              # 训练 + 生成测试布局
  python layout_model.py --gen-only   # 只生成（需已存在模型文件）
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR  = Path(__file__).parent
DATA_PATH   = SCRIPT_DIR / "training_data" / "osm_layouts.json"
W3_DIR      = SCRIPT_DIR / "training_data" / "w3_layouts"
MODEL_DIR   = SCRIPT_DIR / "models"
MODEL_PATH  = MODEL_DIR / "layout_model.pt"
MODEL_W3_PATH = MODEL_DIR / "layout_model_w3.pt"

# Default size imputed for W3 buildings that lack size info
W3_DEFAULT_SIZES = {
    "house":        (10.0, 8.0),
    "building":     (10.0, 8.0),
    "castle":       (14.0, 14.0),
    "church":       (14.0, 9.0),
    "commercial":   (12.0, 9.0),
    "industrial":   (14.0, 10.0),
    "fortification":(12.0, 3.0),
}

# ─── 超参数 ────────────────────────────────────────────────────
PATCH_SIZE   = 24     # 每个训练序列的建筑数量
PATCH_RADIUS = 80.0   # m，采样邻域半径
MAX_PATCHES_PER_TOWN = 3000  # 每城镇最多采多少个 patch

D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 4
DROPOUT  = 0.10

BATCH_SIZE   = 64
EPOCHS       = 120
LR           = 3e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO    = 0.20
EARLY_STOP   = 15
LR_PATIENCE  = 8

# 连续特征归一化常数
POS_SCALE  = 50.0   # nx,ny 范围 ±50m → ÷50 → ±1
ORI_SCALE  = 1.0    # sin/cos 已在 ±1
SIZE_SCALE = 20.0   # 长宽典型值 ~20m → ÷20 → ~1

# 建筑类型
TYPES     = ["other", "house", "commercial", "civic", "church",
             "castle", "industrial"]
N_TYPES   = len(TYPES)
TYPE2ID   = {t: i for i, t in enumerate(TYPES)}

# 生成时输出权重（loss balancing）
W_POS  = 1.0
W_ORI  = 0.5
W_SIZE = 0.3
W_TYPE = 1.2


# ═══════════════════════════════════════════════════════════════
# 1. 数据处理
# ═══════════════════════════════════════════════════════════════

def load_buildings(path: Path) -> dict[str, list[dict]]:
    """返回 {town_name: [building_dict, ...]}"""
    data = json.loads(path.read_text("utf-8"))
    return data["data"]


def load_w3_buildings(w3_dir: Path) -> dict[str, list[dict]]:
    """
    Load W3 layout JSON files and convert to OSM-compatible building dicts.
    W3 buildings lack size info -> imputed from type defaults.
    """
    result: dict[str, list[dict]] = {}
    for jf in sorted(w3_dir.glob("*.json")):
        # Skip prolog_village_winter (near-duplicate of sketch version)
        if "winter" in jf.stem:
            continue
        raw = json.loads(jf.read_text("utf-8"))
        buildings = raw.get("buildings", [])
        normalized: list[dict] = []
        for b in buildings:
            btype = b.get("type", "house")
            l, w = W3_DEFAULT_SIZES.get(btype, (10.0, 8.0))
            normalized.append({
                "nx":              b["nx"],
                "ny":              b["ny"],
                "orientation_deg": b.get("yaw_deg", 0.0),
                "length_m":        l,
                "width_m":         w,
                "area_m2":         round(l * w, 1),
                "type":            btype,
            })
        if normalized:
            result[f"w3_{jf.stem}"] = normalized
    return result


def load_combined(osm_path: Path, w3_dir: Path,
                  w3_weight: int = 2) -> dict[str, list[dict]]:
    """
    Merge OSM and W3 building data.
    w3_weight=2: duplicate each W3 map entry so W3 gets 2x sampling priority.
    """
    combined = load_buildings(osm_path)
    w3_data  = load_w3_buildings(w3_dir)

    total_w3 = sum(len(v) for v in w3_data.values())
    print(f"  W3 maps: {list(w3_data.keys())}")
    print(f"  W3 buildings: {total_w3}  (weight x{w3_weight})")

    # Duplicate W3 entries to achieve 2x weighting
    for key, blds in w3_data.items():
        for rep in range(w3_weight):
            combined[f"{key}_w{rep}"] = blds

    return combined


def _greedy_order(buildings: list[dict]) -> list[dict]:
    """
    从质心出发，贪心最近邻排序。
    使建筑序列具有局部空间连贯性，便于 Transformer 学习。
    """
    if len(buildings) <= 1:
        return buildings
    cx = sum(b["nx"] for b in buildings) / len(buildings)
    cy = sum(b["ny"] for b in buildings) / len(buildings)

    remaining = list(buildings)
    ordered   = []
    # 从最接近质心的建筑开始
    start = min(remaining,
                key=lambda b: (b["nx"] - cx) ** 2 + (b["ny"] - cy) ** 2)
    remaining.remove(start)
    ordered.append(start)

    while remaining:
        last  = ordered[-1]
        lx, ly = last["nx"], last["ny"]
        nxt = min(remaining,
                  key=lambda b: (b["nx"] - lx) ** 2 + (b["ny"] - ly) ** 2)
        remaining.remove(nxt)
        ordered.append(nxt)

    return ordered


def extract_patches(
    town_buildings: dict[str, list[dict]],
    patch_size: int = PATCH_SIZE,
    radius: float    = PATCH_RADIUS,
    max_per_town: int = MAX_PATCHES_PER_TOWN,
    rng: np.random.Generator = None,
) -> list[list[dict]]:
    """
    从每个城镇随机采样局部 patch（空间邻域子集）。
    每个 patch = patch_size 栋建筑，按近邻顺序排列。
    """
    if rng is None:
        rng = np.random.default_rng(42)

    patches = []
    for town, buildings in town_buildings.items():
        arr = np.array([[b["nx"], b["ny"]] for b in buildings])
        n   = len(buildings)
        # 按城镇大小等比例分配 patch 数量
        n_patches = min(max_per_town,
                        max(200, int(n / patch_size * 2)))

        center_idxs = rng.choice(n, size=n_patches, replace=True)

        for ci in center_idxs:
            cx, cy = arr[ci, 0], arr[ci, 1]
            # 找邻域内的建筑
            dists = np.sqrt((arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2)
            nbrs  = np.where(dists <= radius)[0]
            if len(nbrs) < max(4, patch_size // 2):
                continue

            # 采样 patch_size 个邻居（最近的优先，随机扰动）
            sorted_nbrs = nbrs[np.argsort(dists[nbrs])]
            take = sorted_nbrs[:min(len(sorted_nbrs), patch_size)]
            patch_blds = [buildings[i] for i in take]

            # 局部坐标归一化：以 patch 中心为原点
            lcx = sum(b["nx"] for b in patch_blds) / len(patch_blds)
            lcy = sum(b["ny"] for b in patch_blds) / len(patch_blds)
            norm_blds = [
                {**b, "nx": b["nx"] - lcx, "ny": b["ny"] - lcy}
                for b in patch_blds
            ]
            # 近邻排序
            ordered = _greedy_order(norm_blds)
            if len(ordered) >= 3:
                patches.append(ordered)

    rng.shuffle(patches)
    return patches


# ═══════════════════════════════════════════════════════════════
# 2. 数据集
# ═══════════════════════════════════════════════════════════════

def encode_building(b: dict) -> torch.Tensor:
    """
    编码单栋建筑为 7 维连续向量 + 1 维类型 id（共 8 维）。
    返回 float tensor shape (8,)：
      [nx/POS_SCALE, ny/POS_SCALE, sin(ori), cos(ori),
       len/SIZE_SCALE, wid/SIZE_SCALE, sin(ori_coarse), type_id]
    注：type_id 以 float 存储，Dataset 中分开处理。
    """
    ori_rad = math.radians(b["orientation_deg"])
    return torch.tensor([
        b["nx"]  / POS_SCALE,
        b["ny"]  / POS_SCALE,
        math.sin(ori_rad),
        math.cos(ori_rad),
        b["length_m"] / SIZE_SCALE,
        b["width_m"]  / SIZE_SCALE,
    ], dtype=torch.float32)


def decode_building(cont: torch.Tensor, type_id: int) -> dict:
    """
    将连续向量 + 类型 id 转回 building dict。
    cont: shape (6,) — [nx_n, ny_n, sin, cos, len_n, wid_n]
    """
    c = cont.cpu().tolist()
    ori_deg = math.degrees(math.atan2(c[2], c[3]))
    return {
        "nx":              round(c[0] * POS_SCALE, 2),
        "ny":              round(c[1] * POS_SCALE, 2),
        "orientation_deg": round(ori_deg, 1),
        "length_m":        round(max(3.0, c[4] * SIZE_SCALE), 2),
        "width_m":         round(max(2.0, c[5] * SIZE_SCALE), 2),
        "area_m2":         round(max(3.0, c[4] * SIZE_SCALE) *
                                  max(2.0, c[5] * SIZE_SCALE), 1),
        "type":            TYPES[type_id],
    }


class LayoutDataset(Dataset):
    """
    每个样本 = 一个 patch（建筑序列）。
    Teacher-forcing：输入序列[0..k-1]，目标序列[1..k]。
    """
    def __init__(self, patches: list[list[dict]]):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        blds  = self.patches[idx]
        conts = torch.stack([encode_building(b) for b in blds])   # (k, 6)
        types = torch.tensor(
            [TYPE2ID.get(b["type"], 0) for b in blds], dtype=torch.long)

        # 输入：[0..k-2]，目标：[1..k-1]
        src_cont  = conts[:-1]   # (k-1, 6)
        src_types = types[:-1]   # (k-1,)
        tgt_cont  = conts[1:]    # (k-1, 6)
        tgt_types = types[1:]    # (k-1,)

        return src_cont, src_types, tgt_cont, tgt_types


def collate_fn(batch):
    """Pad 变长序列至 batch 内最长。"""
    src_conts, src_types, tgt_conts, tgt_types = zip(*batch)
    max_len = max(s.shape[0] for s in src_conts)

    def pad2d(tensors, fill=0.0):
        out = torch.full((len(tensors), max_len, tensors[0].shape[1]), fill)
        for i, t in enumerate(tensors):
            out[i, :t.shape[0]] = t
        return out

    def pad1d(tensors, fill=-1):
        out = torch.full((len(tensors), max_len), fill, dtype=torch.long)
        for i, t in enumerate(tensors):
            out[i, :t.shape[0]] = t
        return out

    key_mask = torch.zeros(len(src_conts), max_len, dtype=torch.bool)
    for i, t in enumerate(src_conts):
        key_mask[i, t.shape[0]:] = True   # True = 忽略该位置（padding）

    return (
        pad2d(src_conts),       # (B, L, 6)
        pad1d(src_types, 0),    # (B, L)
        pad2d(tgt_conts),       # (B, L, 6)
        pad1d(tgt_types, -1),   # (B, L) -1 = padding，cross_entropy 忽略
        key_mask,               # (B, L)
    )


# ═══════════════════════════════════════════════════════════════
# 3. 模型
# ═══════════════════════════════════════════════════════════════

class BuildingEmbedder(nn.Module):
    """将建筑 token (连续 6 维 + 类型 id) 映射为 d_model 维 embedding。"""
    def __init__(self, d_model: int, n_types: int = N_TYPES):
        super().__init__()
        self.cont_proj  = nn.Linear(6, d_model - 16)  # 连续特征
        self.type_embed = nn.Embedding(n_types, 16)    # 类型 embedding
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, cont, type_ids):
        # cont: (B, L, 6)  type_ids: (B, L)
        c = self.cont_proj(cont)         # (B, L, d-16)
        t = self.type_embed(type_ids)    # (B, L, 16)
        return self.norm(torch.cat([c, t], dim=-1))  # (B, L, d)


class LayoutTransformer(nn.Module):
    """
    Decoder-only Transformer，因果掩码，自回归生成建筑序列。
    输出三个头：
      - pos_head:  (B, L, 4) — [nx_n, ny_n, sin_ori, cos_ori]
      - size_head: (B, L, 2) — [len_n, wid_n]
      - type_head: (B, L, N_TYPES) — logits
    """
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.d_model   = d_model
        self.embedder  = BuildingEmbedder(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,   # Pre-LN，训练更稳
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.pos_head  = nn.Linear(d_model, 4)
        self.size_head = nn.Linear(d_model, 2)
        self.type_head = nn.Linear(d_model, N_TYPES)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """上三角因果掩码 (seq_len, seq_len)，True = 被掩盖。"""
        return torch.triu(torch.ones(seq_len, seq_len, device=device),
                          diagonal=1).bool()

    def forward(self, src_cont, src_types, key_padding_mask=None):
        # src_cont:  (B, L, 6)
        # src_types: (B, L)
        B, L, _ = src_cont.shape
        x    = self.embedder(src_cont, src_types)       # (B, L, d)
        mask = self._causal_mask(L, src_cont.device)    # (L, L)

        h = self.transformer(x,
                             mask=mask,
                             src_key_padding_mask=key_padding_mask,
                             is_causal=True)             # (B, L, d)

        pos  = self.pos_head(h)    # (B, L, 4)
        size = self.size_head(h)   # (B, L, 2)
        types = self.type_head(h)  # (B, L, N_TYPES)
        return pos, size, types


def compute_loss(pos_pred, size_pred, type_pred,
                 tgt_cont, tgt_types, key_mask):
    """
    混合损失：
      pos  = MSE( pred[nx,ny,sin,cos],  tgt[nx,ny,sin,cos] )
      size = MSE( pred[len,wid],         tgt[len,wid] )
      type = CrossEntropy( logits, tgt_types ), ignore_index=-1
    key_mask: True = padding，需忽略。
    """
    # 有效位置掩码
    valid = ~key_mask  # (B, L)

    def masked_mse(pred, target):
        diff = (pred - target) ** 2  # (B, L, k)
        # valid: (B, L) → (B, L, 1)
        m = valid.unsqueeze(-1).float()
        return (diff * m).sum() / (m.sum() * pred.shape[-1] + 1e-8)

    loss_pos  = masked_mse(pos_pred,  tgt_cont[:, :, :4])
    loss_size = masked_mse(size_pred, tgt_cont[:, :, 4:6])

    # type cross-entropy — reshape + mask with -1 for padding
    B, L = tgt_types.shape
    tgt_t = tgt_types.clone()
    tgt_t[~valid] = -1   # padding 位置用 -1，ignore_index 跳过
    loss_type = F.cross_entropy(
        type_pred.reshape(B * L, -1),
        tgt_t.reshape(-1),
        ignore_index=-1,
    )

    total = W_POS * loss_pos + W_ORI * 0  # ori 已包含在 pos 最后2维
    total = W_POS * loss_pos + W_SIZE * loss_size + W_TYPE * loss_type
    return total, loss_pos.item(), loss_size.item(), loss_type.item()


# ═══════════════════════════════════════════════════════════════
# 4. 训练
# ═══════════════════════════════════════════════════════════════

def train_model(patches: list[list[dict]], device: torch.device) -> LayoutTransformer:
    random.shuffle(patches)
    split   = int(len(patches) * (1 - VAL_RATIO))
    tr_pat  = patches[:split]
    va_pat  = patches[split:]

    tr_ds = LayoutDataset(tr_pat)
    va_ds = LayoutDataset(va_pat)

    pin = (device.type == "cuda")
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       collate_fn=collate_fn, pin_memory=pin, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate_fn, pin_memory=pin, num_workers=0)

    model = LayoutTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数: {n_params:,}")
    print(f"  训练 patches: {len(tr_pat):,}  验证 patches: {len(va_pat):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=LR_PATIENCE, factor=0.5, min_lr=1e-6)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    print(f"\n{'Ep':>4}  {'TrLoss':>9}  {'VaLoss':>9}  "
          f"{'pos':>7}  {'size':>7}  {'type':>7}  {'lr':>8}  {'t':>5}")
    print("─" * 72)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── 训练 ──────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for sc, st, tc, tt, km in tr_dl:
            sc = sc.to(device); st = st.to(device)
            tc = tc.to(device); tt = tt.to(device); km = km.to(device)
            optimizer.zero_grad(set_to_none=True)
            pp, sp, tp = model(sc, st, km)
            loss, _, _, _ = compute_loss(pp, sp, tp, tc, tt, km)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_dl)

        # ── 验证 ──────────────────────────────────────────────
        model.eval()
        va_loss = va_pos = va_sz = va_ty = 0.0
        with torch.no_grad():
            for sc, st, tc, tt, km in va_dl:
                sc = sc.to(device); st = st.to(device)
                tc = tc.to(device); tt = tt.to(device); km = km.to(device)
                pp, sp, tp = model(sc, st, km)
                loss, lp, ls, lt = compute_loss(pp, sp, tp, tc, tt, km)
                va_loss += loss.item()
                va_pos  += lp; va_sz += ls; va_ty += lt
        va_loss /= len(va_dl)
        va_pos  /= len(va_dl)
        va_sz   /= len(va_dl)
        va_ty   /= len(va_dl)

        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            dt = time.time() - t0
            print(f"{epoch:>4}  {tr_loss:>9.5f}  {va_loss:>9.5f}  "
                  f"{va_pos:>7.4f}  {va_sz:>7.4f}  {va_ty:>7.4f}  "
                  f"{lr_now:>8.2e}  {dt:>4.1f}s", flush=True)

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP:
            print(f"\n[Early Stop] epoch {epoch}，无改善 {EARLY_STOP} epochs")
            break

    print(f"\n最佳验证损失: {best_val:.5f}")
    model.load_state_dict(best_state)
    model._best_val = best_val   # stash for caller
    return model


# ═══════════════════════════════════════════════════════════════
# 5. 布局生成
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_layout(
    model: LayoutTransformer,
    device: torch.device,
    n_buildings: int = 15,
    seed_buildings: list[dict] = None,
    temperature: float = 0.8,
    rng: np.random.Generator = None,
) -> list[dict]:
    """
    自回归生成建筑布局。
    seed_buildings: 初始锚点建筑（可为空）。
    temperature: 采样温度，越高越随机。
    """
    if rng is None:
        rng = np.random.default_rng(0)

    model.eval()
    buildings = list(seed_buildings) if seed_buildings else []

    # 若无种子建筑，用零向量作为起点（中心位置）
    if not buildings:
        buildings = [{
            "nx": 0.0, "ny": 0.0,
            "orientation_deg": 0.0,
            "length_m": 12.0, "width_m": 8.0,
            "type": "house",
        }]

    generated = []

    while len(generated) < n_buildings:
        # 编码当前序列
        conts = torch.stack(
            [encode_building(b) for b in buildings]).unsqueeze(0).to(device)
        types = torch.tensor(
            [TYPE2ID.get(b["type"], 0) for b in buildings],
            dtype=torch.long).unsqueeze(0).to(device)

        # 前向（只取最后一个时刻的预测）
        pp, sp, tp = model(conts, types)
        pos_pred   = pp[0, -1]   # (4,)
        size_pred  = sp[0, -1]   # (2,)
        type_logits = tp[0, -1]  # (N_TYPES,)

        # ── 连续特征：均值 + 温度扰动 ──
        noise_pos  = torch.randn(4, device=device) * temperature * 0.15
        noise_size = torch.randn(2, device=device) * temperature * 0.1
        pos_s  = (pos_pred  + noise_pos).cpu()
        size_s = (size_pred + noise_size).cpu()

        # ── 类型：温度采样 ──
        probs    = F.softmax(type_logits / max(temperature, 0.1), dim=-1)
        probs_np = probs.cpu().numpy()
        type_id  = int(rng.choice(N_TYPES, p=probs_np))

        cont6 = torch.cat([pos_s, size_s])   # (6,)
        bld   = decode_building(cont6, type_id)
        generated.append(bld)
        buildings.append(bld)

    return generated


# ═══════════════════════════════════════════════════════════════
# 6. 评估 & 对比
# ═══════════════════════════════════════════════════════════════

def evaluate_layout(buildings: list[dict], label: str = "") -> dict:
    """
    计算布局质量指标：
      - 平均最近邻距离（越小 = 越密集）
      - 朝向方差（越小 = 越对齐）
      - 类型熵（越高 = 越多样）
      - 平均建筑面积
    """
    n = len(buildings)
    if n == 0:
        return {}

    xs = np.array([b["nx"] for b in buildings])
    ys = np.array([b["ny"] for b in buildings])

    # 最近邻距离
    nnd = []
    for i in range(n):
        dists = np.sqrt((xs - xs[i]) ** 2 + (ys - ys[i]) ** 2)
        dists[i] = 1e9
        nnd.append(dists.min())
    avg_nnd = float(np.mean(nnd))

    # 朝向方差（circular std via sin/cos mean resultant length）
    oris = np.radians([b["orientation_deg"] for b in buildings])
    R = math.sqrt(np.mean(np.sin(oris)) ** 2 + np.mean(np.cos(oris)) ** 2)
    circ_std_deg = float(math.degrees(math.sqrt(-2 * math.log(R + 1e-9))))

    # 类型分布熵
    type_counts = {}
    for b in buildings:
        type_counts[b["type"]] = type_counts.get(b["type"], 0) + 1
    probs = np.array(list(type_counts.values())) / n
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))

    # 平均面积
    avg_area = float(np.mean([b["area_m2"] for b in buildings]))

    metrics = {
        "n_buildings":    n,
        "avg_nnd_m":      round(avg_nnd, 2),
        "ori_std_deg":    round(circ_std_deg, 1),
        "type_entropy":   round(entropy, 3),
        "avg_area_m2":    round(avg_area, 1),
        "type_dist":      type_counts,
    }

    if label:
        print(f"\n  [{label}]")
        print(f"    建筑数        : {n}")
        print(f"    平均最近邻距离 : {avg_nnd:.2f} m")
        print(f"    朝向离散度    : {circ_std_deg:.1f}°")
        print(f"    类型多样性熵  : {entropy:.3f}")
        print(f"    平均建筑面积  : {avg_area:.1f} m²")
        dist_str = "  ".join(f"{k}:{v}" for k, v in
                             sorted(type_counts.items(), key=lambda x: -x[1]))
        print(f"    类型分布      : {dist_str}")

    return metrics


def osm_reference_stats(town_buildings: dict[str, list[dict]]) -> dict:
    """计算真实 OSM 数据的参考统计（每城镇取前 200 栋建筑的局部 patch）。"""
    rng = np.random.default_rng(99)
    all_blds = []
    for blds in town_buildings.values():
        sample = list(blds)
        rng.shuffle(sample)
        all_blds.extend(sample[:200])
    return evaluate_layout(all_blds[:200], label="OSM真实参考(200栋)")


def print_layout_table(buildings: list[dict], label: str):
    """打印生成布局的建筑列表。"""
    print(f"\n  {label} — {len(buildings)} 栋建筑：")
    print(f"  {'#':>3}  {'nx':>7}  {'ny':>7}  {'ori°':>6}  "
          f"{'len':>6}  {'wid':>6}  {'area':>7}  type")
    print("  " + "─" * 65)
    for i, b in enumerate(buildings, 1):
        print(f"  {i:>3}  {b['nx']:>7.1f}  {b['ny']:>7.1f}  "
              f"{b['orientation_deg']:>6.1f}  {b['length_m']:>6.1f}  "
              f"{b['width_m']:>6.1f}  {b['area_m2']:>7.1f}  {b['type']}")


def generate_rule_based_reference(n: int = 15) -> list[dict]:
    """
    生成规则算法参考布局（网格 + 随机变体）。
    不依赖 level_layout.py，独立实现简单对照。
    """
    rng = np.random.default_rng(7)
    buildings = []
    cols = max(1, round(math.sqrt(n)))
    rows = math.ceil(n / cols)
    spacing = 18.0
    for r in range(rows):
        for c in range(cols):
            if len(buildings) >= n:
                break
            x = (c - cols / 2) * spacing + rng.uniform(-2, 2)
            y = (r - rows / 2) * spacing + rng.uniform(-2, 2)
            ori = rng.choice([0, 90, -90, 180]) + rng.uniform(-5, 5)
            l   = rng.uniform(8, 16)
            w   = rng.uniform(6, 12)
            tp  = rng.choice(["house", "house", "house",
                               "commercial", "church"], p=[0.5, 0.2, 0.1, 0.1, 0.1])
            buildings.append({
                "nx": round(x, 2), "ny": round(y, 2),
                "orientation_deg": round(float(ori), 1),
                "length_m": round(l, 2), "width_m": round(w, 2),
                "area_m2": round(l * w, 1), "type": tp,
            })
    return buildings[:n]


# ═══════════════════════════════════════════════════════════════
# 7. 保存 & 加载
# ═══════════════════════════════════════════════════════════════

def save_model(model: LayoutTransformer, path: Path, best_val: float = float("nan")):
    path.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "d_model": D_MODEL, "n_heads": N_HEADS,
            "n_layers": N_LAYERS, "dropout": DROPOUT,
        },
        "types": TYPES,
        "pos_scale": POS_SCALE,
        "size_scale": SIZE_SCALE,
        "best_val_loss": best_val,
    }, path)
    print(f"[保存] {path}")


def load_model(path: Path, device: torch.device) -> LayoutTransformer:
    ckpt  = torch.load(path, map_location=device)
    cfg   = ckpt["config"]
    model = LayoutTransformer(
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[加载] {path}")
    return model


# ═══════════════════════════════════════════════════════════════
# 8. 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-only",  action="store_true",
                        help="跳过训练，只生成布局（需已存在模型文件）")
    parser.add_argument("--n-gen",     type=int, default=15,
                        help="生成建筑数量（默认15）")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度（默认0.8）")
    parser.add_argument("--source",    default="combined",
                        choices=["osm", "combined"],
                        help="数据来源: osm(仅OSM) | combined(OSM+W3, 默认)")
    parser.add_argument("--w3-weight", type=int, default=2,
                        help="W3数据重复权重（默认2，即加倍）")
    args = parser.parse_args()

    use_combined = (args.source == "combined")
    out_model    = MODEL_W3_PATH if use_combined else MODEL_PATH

    print("=" * 65)
    print(f"  layout_model.py  —  数据源: {args.source.upper()}")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ── 1. 加载数据 ──────────────────────────────────────────
    if use_combined:
        print(f"\n[数据] 合并 OSM + W3 (weight={args.w3_weight})...")
        town_blds = load_combined(DATA_PATH, W3_DIR, w3_weight=args.w3_weight)
    else:
        print(f"\n[数据] 加载 {DATA_PATH.name}...")
        town_blds = load_buildings(DATA_PATH)

    # Print stats only for real (non-duplicated) entries
    base_keys = [k for k in town_blds if not k.endswith("_w1")]
    total = sum(len(town_blds[k]) for k in base_keys)
    for k in base_keys:
        prefix = "W3 " if k.startswith("w3_") else "OSM"
        print(f"  [{prefix}] {k:<30} {len(town_blds[k]):>6,d} 栋")
    print(f"  {'Total (unique)':<35} {total:>6,d} 栋")

    if not args.gen_only:
        # ── 2. 提取 patches ─────────────────────────────────
        print(f"\n[Patches] 提取训练序列 (size={PATCH_SIZE}, r={PATCH_RADIUS}m)...")
        rng = np.random.default_rng(42)
        patches = extract_patches(town_blds, PATCH_SIZE, PATCH_RADIUS,
                                  MAX_PATCHES_PER_TOWN, rng)
        print(f"  总 patches: {len(patches):,}")
        lens = [len(p) for p in patches]
        print(f"  序列长度: min={min(lens)}  avg={sum(lens)/len(lens):.1f}  max={max(lens)}")

        # 空间关系统计
        _print_spatial_stats({k: v for k, v in town_blds.items()
                               if not k.endswith("_w1")})

        # ── 3. 训练 ─────────────────────────────────────────
        print(f"\n[训练] epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}  early_stop={EARLY_STOP}")
        t0    = time.time()
        model = train_model(patches, device)
        elapsed = time.time() - t0
        print(f"[训练完成] 耗时: {elapsed:.1f}s")

        best_val_loss = getattr(model, "_best_val", float("nan"))
        save_model(model, out_model, best_val=best_val_loss)

        # ── 对比旧模型 ──────────────────────────────────────
        if use_combined and MODEL_PATH.exists():
            old_ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            new_ckpt = torch.load(out_model,  map_location="cpu", weights_only=True)
            old_val  = old_ckpt.get("best_val_loss", float("nan"))
            new_val  = new_ckpt.get("best_val_loss", float("nan"))
            print(f"\n[模型对比]")
            print(f"  旧模型 (OSM-only)   最佳验证损失: {old_val:.5f}  → {MODEL_PATH.name}")
            print(f"  新模型 (OSM+W3×{args.w3_weight})  最佳验证损失: {new_val:.5f}  → {out_model.name}")
            if not math.isnan(old_val) and not math.isnan(new_val):
                delta = new_val - old_val
                print(f"  变化: {delta:+.5f} "
                      f"({'改善' if delta < 0 else '下降'})")

    else:
        if not out_model.exists():
            # Fall back to old model for gen-only
            out_model = MODEL_PATH if MODEL_PATH.exists() else None
        if out_model is None or not out_model.exists():
            raise SystemExit(f"模型文件不存在，请先训练")
        model = load_model(out_model, device)

    # ── 4. 生成测试布局 ──────────────────────────────────────
    N  = args.n_gen
    T  = args.temperature
    rng_gen = np.random.default_rng(2024)

    print(f"\n{'='*65}")
    print(f"  布局生成测试  —  {N} 栋建筑  温度={T}")
    print(f"{'='*65}")

    # 学习模型生成
    layout_model = generate_layout(model, device, N, temperature=T, rng=rng_gen)
    print_layout_table(layout_model, "模型生成布局")

    # 规则算法对照
    layout_rule = generate_rule_based_reference(N)
    print_layout_table(layout_rule, "规则算法布局（网格对照）")

    # ── 5. 对比评估 ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  布局质量对比")
    print(f"{'='*65}")

    osm_reference_stats(town_blds)
    m1 = evaluate_layout(layout_model, label="模型生成布局")
    m2 = evaluate_layout(layout_rule,  label="规则算法布局")

    print(f"\n  {'指标':<18}  {'OSM参考':>10}  {'模型生成':>10}  {'规则算法':>10}")
    print("  " + "─" * 55)
    # (简化对比：只打印模型 vs 规则)
    print(f"  {'平均最近邻(m)':<18}  {'—':>10}  "
          f"{m1['avg_nnd_m']:>10.2f}  {m2['avg_nnd_m']:>10.2f}")
    print(f"  {'朝向离散度(°)':<18}  {'—':>10}  "
          f"{m1['ori_std_deg']:>10.1f}  {m2['ori_std_deg']:>10.1f}")
    print(f"  {'类型多样性熵':<18}  {'—':>10}  "
          f"{m1['type_entropy']:>10.3f}  {m2['type_entropy']:>10.3f}")
    print(f"  {'平均面积(m²)':<18}  {'—':>10}  "
          f"{m1['avg_area_m2']:>10.1f}  {m2['avg_area_m2']:>10.1f}")

    # ── 6. 保存生成结果 ─────────────────────────────────────
    suffix = "_w3" if use_combined else ""
    result_path = MODEL_DIR / f"generated_layout{suffix}.json"
    MODEL_DIR.mkdir(exist_ok=True)
    result = {
        "model_layout":      layout_model,
        "rule_layout":       layout_rule,
        "metrics_model":     m1,
        "metrics_rule":      m2,
        "data_source":       args.source,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\n  生成结果: {result_path}")
    print(f"  模型文件: {out_model}")


def _print_spatial_stats(town_blds: dict[str, list[dict]]):
    """输出空间关系统计，帮助理解训练数据特征。"""
    print("\n[空间关系统计]")
    print(f"  {'城镇':<15}  {'平均NND':>9}  {'朝向Std':>9}  {'类型熵':>8}")
    print("  " + "─" * 50)
    all_nnds, all_stds, all_ents = [], [], []
    for town, blds in town_blds.items():
        # 取代表性子集
        sub = blds[:min(500, len(blds))]
        m   = evaluate_layout(sub)
        all_nnds.append(m["avg_nnd_m"])
        all_stds.append(m["ori_std_deg"])
        all_ents.append(m["type_entropy"])
        print(f"  {town:<15}  {m['avg_nnd_m']:>9.2f}  "
              f"{m['ori_std_deg']:>9.1f}  {m['type_entropy']:>8.3f}")
    print(f"  {'─'*50}")
    print(f"  {'平均':<15}  {sum(all_nnds)/len(all_nnds):>9.2f}  "
          f"{sum(all_stds)/len(all_stds):>9.1f}  "
          f"{sum(all_ents)/len(all_ents):>8.3f}")


if __name__ == "__main__":
    main()
