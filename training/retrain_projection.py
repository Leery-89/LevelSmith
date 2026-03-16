"""
LevelSmith 投影层重训练脚本 (retrain_projection.py)

为扩展后的 20 种风格重新训练 StyleProjectionLayer (projection.pt)。

改进策略：
  - 每种新子风格提供 3~5 条中英文描述作为训练锚点
  - 原 7 种风格各补充 2 条额外描述，增强泛化
  - 多描述共享同一特征向量，显式强化风格区分边界

训练完成后自动测试 4 条目标描述。
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from text_encoder import (
    StyleProjectionLayer,
    train_projection_layer,
    ENCODER_MULTILINGUAL,
    PROJECTION_FILE,
)
from style_registry import STYLE_REGISTRY, get_feature_vector

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer


# ─── 多描述锚点定义 ───────────────────────────────────────────────
# 每个风格：原有 description + 额外描述
# 格式：{ style_name: [desc1, desc2, ...] }

STYLE_DESCRIPTIONS = {
    # ── 原始 7 种（各补充 2 条）──────────────────────────────────
    "medieval": [
        STYLE_REGISTRY["medieval"].description,
        "中世纪城堡大厅，厚重石墙，拱形天花，小窗，宏伟木门",
        "A medieval castle with thick stone walls, vaulted ceilings and heavy wooden doors",
    ],
    "modern": [
        STYLE_REGISTRY["modern"].description,
        "现代简约建筑，薄墙大窗，开放空间，极简风格",
        "Modern minimalist architecture with thin walls, large windows and open floor plan",
    ],
    "industrial": [
        STYLE_REGISTRY["industrial"].description,
        "工业仓库厂房，裸露钢铁混凝土，超高空间，工业格窗",
        "Industrial warehouse with exposed steel and concrete, high ceilings and metal frame windows",
    ],
    "fantasy": [
        STYLE_REGISTRY["fantasy"].description,
        "奇幻魔法城堡，高耸尖塔，极致装饰，哥特拱窗，宏伟门洞",
        "A fantasy RPG castle with soaring spires, ornate Gothic windows and grand archways",
    ],
    "horror": [
        STYLE_REGISTRY["horror"].description,
        "哥特废墟鬼屋，压抑低矮，极暗，迷宫走廊，腐朽墙体",
        "A gothic haunted house ruin with oppressive low ceilings and dark labyrinthine corridors",
    ],
    "japanese": [
        STYLE_REGISTRY["japanese"].description,
        "日式传统建筑，低矮水平延伸，障子格窗，轻薄木构，推拉门",
        "Traditional Japanese architecture with horizontal lines, shoji screens and lightweight wood construction",
    ],
    "desert": [
        STYLE_REGISTRY["desert"].description,
        "中东沙漠土坯建筑，极厚隔热土墙，极少小窗，平屋顶",
        "Middle Eastern adobe desert fortress with extremely thick walls and minimal small windows",
    ],

    # ── 中世纪衍生 ──────────────────────────────────────────────
    "medieval_chapel": [
        STYLE_REGISTRY["medieval_chapel"].description,
        "中世纪小礼拜堂，石砌单间，细高尖拱彩窗，宗教氛围",
        "A small medieval chapel with tall lancet stained glass windows and intimate stone interior",
        "礼拜堂，石材墙体，尖拱窗户，低矮亲切的宗教空间",
    ],
    "medieval_keep": [
        STYLE_REGISTRY["medieval_keep"].description,
        "城堡防御主塔，极厚石墙，箭孔微窗，多层楼，要塞级安全",
        "A medieval keep tower with arrow slit windows, massive stone walls and multiple defensive floors",
        "中世纪要塞塔楼，极厚防御石墙，箭孔几乎无窗，高耸入云",
    ],

    # ── 现代衍生 ────────────────────────────────────────────────
    "modern_loft": [
        STYLE_REGISTRY["modern_loft"].description,
        "工业Loft公寓，高挑裸露混凝土天花，开放式大空间，超大工业窗",
        "An industrial loft apartment with exposed concrete ceilings, open plan and oversized factory windows",
        "现代工业风loft，高挑开放，裸露管道与混凝土结构",
    ],
    "modern_villa": [
        STYLE_REGISTRY["modern_villa"].description,
        "现代豪华别墅，近落地超大玻璃幕墙，极薄围护，奢华开敞空间",
        "A modern luxury villa with floor-to-ceiling glass walls, minimal structure and open living spaces",
        "高端现代别墅，极薄玻璃围护，大面积采光，奢华材料",
    ],

    # ── 工业衍生 ────────────────────────────────────────────────
    "industrial_workshop": [
        STYLE_REGISTRY["industrial_workshop"].description,
        "小型工业工坊，砖墙钢框格窗，中等净高，多工作区分割",
        "A small industrial workshop with brick walls, steel frame windows and multiple work zones",
        "机械工坊车间，砖墙结构，工业格窗，功能性多区布局",
    ],
    "industrial_powerplant": [
        STYLE_REGISTRY["industrial_powerplant"].description,
        "重工业发电站厂房，超高净空二十米，极厚混凝土墙，设备级超大门",
        "A heavy industrial power plant with twenty meter ceilings, thick concrete walls and massive equipment doors",
        "发电厂厂房，极高净空，几乎无窗，重型工业设施",
        "超大型工业建筑，钢筋混凝土结构，极高净空，功能性极强",
    ],

    # ── 奇幻衍生 ────────────────────────────────────────────────
    "fantasy_dungeon": [
        STYLE_REGISTRY["fantasy_dungeon"].description,
        "阴暗压抑的地下城地牢，极厚石墙，近乎无窗，最大迷宫分割",
        "A dark oppressive dungeon with thick stone walls, no windows and maximum labyrinthine room division",
        "黑暗地牢，极低天花板，沉重铁门，迷宫般的走廊分割",
        "阴暗地下城，极低矮压迫，石墙厚重，几乎完全封闭",
    ],
    "fantasy_palace": [
        STYLE_REGISTRY["fantasy_palace"].description,
        "宏伟华丽的魔法宫殿大厅，极高穹顶，极致装饰，巨型魔法彩窗",
        "A grand magical palace hall with soaring dome ceiling, extreme ornamentation and giant stained glass",
        "奇幻宫殿，穹顶高耸入云，最华丽装饰，宏伟巨型门洞",
        "魔法城堡宫殿大厅，彩光穿窗，极高空间，庄严宏伟",
    ],

    # ── 恐怖衍生 ────────────────────────────────────────────────
    "horror_asylum": [
        STYLE_REGISTRY["horror_asylum"].description,
        "废弃精神病院，铁格窗走廊，蜂巢格间，冰冷制度性压抑空间",
        "An abandoned asylum with barred iron windows, cell-like rooms and cold institutional corridors",
        "废弃医院精神病院，走廊昏暗，铁门铁窗，蜂巢格间布局",
        "精神病院废墟，制度冷漠，铁格窗，阴冷昏暗的走廊",
    ],
    "horror_crypt": [
        STYLE_REGISTRY["horror_crypt"].description,
        "地下墓穴，极低矮天花，极厚石墙，近乎完全封闭，迷宫墓室",
        "An underground stone crypt with extremely low ceiling, thick walls and almost no light or windows",
        "墓室地下空间，极低极暗，迷宫式布局，几乎无窗无光",
    ],

    # ── 和风衍生 ────────────────────────────────────────────────
    "japanese_temple": [
        STYLE_REGISTRY["japanese_temple"].description,
        "日式神社大殿，宏伟重木构斗拱，高台基，翘脊屋顶，庄严宗教对称",
        "A Japanese Shinto shrine with heavy timber bracketing, raised platform and sweeping ceremonial roof",
        "佛寺大殿，庄严对称，重木构，高台翘脊，宗教性空间",
    ],
    "japanese_machiya": [
        STYLE_REGISTRY["japanese_machiya"].description,
        "京都町家，窄面深进，多小间布局，极薄木壁，城市商住连排",
        "A Kyoto machiya townhouse with narrow street facade, deep floor plan and multiple small rooms",
        "京町家，城市连排，窄深多间，极薄木壁障子格窗",
        "传统日式城市商住宅，窄面深进，多室布局，轻薄木构",
    ],

    # ── 沙漠衍生 ────────────────────────────────────────────────
    "desert_palace": [
        STYLE_REGISTRY["desert_palace"].description,
        "伊斯兰沙漠宫殿，极厚隔热土墙，装饰性尖拱窗，高大庭院，华丽几何纹样",
        "An Islamic desert palace with thick insulating walls, ornate pointed arch windows and grand courtyard",
        "伊斯兰宫殿，穹顶高耸，几何装饰纹样，细高尖拱窗，庄严对称",
        "沙漠宫殿，伊斯兰轴对称，极厚隔热，装饰性拱门拱窗",
    ],
}


# ─── 构建多描述锚点 ───────────────────────────────────────────────

def build_anchors(encoder: SentenceTransformer) -> tuple:
    """
    将每种风格的多条描述都编码，每条描述对应该风格的特征向量。
    Returns: (embeddings [N, 384], feature_vecs [N, 16])
    """
    all_embs, all_fvs = [], []
    total = 0
    for style_name, descs in STYLE_DESCRIPTIONS.items():
        fv = get_feature_vector(style_name)  # [16]
        embs = encoder.encode(
            descs,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )  # [len(descs), 384]
        for emb in embs:
            all_embs.append(emb)
            all_fvs.append(fv)
        total += len(descs)
        print(f"  [{style_name}] {len(descs)} 条描述已编码")

    print(f"\n锚点总数: {total} 条（涵盖 {len(STYLE_DESCRIPTIONS)} 种风格）")
    return (
        np.stack(all_embs).astype(np.float32),
        np.stack(all_fvs).astype(np.float32),
    )


# ─── 测试函数 ─────────────────────────────────────────────────────

@torch.no_grad()
def test_queries(
    projection: StyleProjectionLayer,
    encoder: SentenceTransformer,
    style_embeddings: np.ndarray,   # [20, 384] 每种风格一条代表性嵌入
    style_names: list,
    queries: list,
    device: str,
):
    print("\n" + "=" * 70)
    print("  测试查询结果")
    print("=" * 70)

    projection.eval()
    for q in queries:
        q_emb = encoder.encode(q, normalize_embeddings=True, show_progress_bar=False)

        # 余弦相似度（对所有20种风格）
        sims = style_embeddings @ q_emb  # [20]
        sorted_idx = np.argsort(sims)[::-1]
        dominant = style_names[sorted_idx[0]]
        top3 = [(style_names[i], float(sims[i])) for i in sorted_idx[:3]]

        # 投影层输出（仅作参考，不直接决定dominant）
        emb_t = torch.from_numpy(q_emb).unsqueeze(0).to(device)
        fv_pred = projection(emb_t).squeeze(0).cpu().numpy()

        print(f"\n  输入: 「{q}」")
        print(f"  dominant_style: {dominant}  (sim={sims[sorted_idx[0]]:.4f})")
        print(f"  Top-3 匹配:")
        for rank, (name, sim) in enumerate(top3, 1):
            bar = "█" * int(max(0, sim + 1) * 20)
            print(f"    {rank}. {name:<25} {sim:+.4f}  {bar}")
        print(f"  投影输出特征向量(前8维): {[round(float(x),3) for x in fv_pred[:8]]}")
    print()


# ─── 主流程 ──────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {'GPU: ' + torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    # 加载句子编码器
    print(f"\n[编码器] 加载 {ENCODER_MULTILINGUAL} ...")
    encoder = SentenceTransformer(ENCODER_MULTILINGUAL)

    # 构建多描述锚点
    print("\n[锚点] 编码所有描述 ...")
    anchor_embs, anchor_fvs = build_anchors(encoder)

    # 训练投影层
    # 注意：n_noise/n_interp 在 train_projection_layer 内部的 _augment_projection_data 使用。
    # 原默认值(n_noise=800, n_interp=400)为7个锚点设计，79个锚点会产生~130万样本。
    # 这里通过 monkey-patch 覆盖默认值，将总样本控制在 ~3万 以内。
    import text_encoder as _te
    _orig_augment = _te._augment_projection_data

    def _fast_augment(embeddings, feature_vecs, n_noise=800, n_interp=400,
                      noise_std=0.012, seed=42):
        # 79锚点：n_noise=30, n_interp=8 → ~79+2370+~24600 ≈ 27000样本
        return _orig_augment(embeddings, feature_vecs,
                             n_noise=30, n_interp=8,
                             noise_std=noise_std, seed=seed)

    _te._augment_projection_data = _fast_augment

    print("\n[投影层] 开始训练 ...")
    projection = train_projection_layer(
        anchor_embeddings=anchor_embs,
        anchor_feature_vecs=anchor_fvs,
        device=device,
        epochs=1500,
        lr=1e-3,
        batch_size=256,
        patience=150,
        save_path=PROJECTION_FILE,
    )

    _te._augment_projection_data = _orig_augment  # 恢复原函数

    # 为测试准备每种风格的代表性嵌入（取第一条描述）
    style_names = list(STYLE_DESCRIPTIONS.keys())
    repr_descs = [STYLE_DESCRIPTIONS[n][0] for n in style_names]
    style_embeddings = encoder.encode(
        repr_descs,
        normalize_embeddings=True,
        show_progress_bar=False,
    )  # [20, 384]

    # 测试4条目标描述
    test_queries_list = [
        "阴暗压抑的地下牢房",
        "宏伟华丽的魔法宫殿",
        "窄面深进的京町家",
        "废弃精神病院的铁窗走廊",
    ]
    test_queries(
        projection, encoder,
        style_embeddings, style_names,
        test_queries_list, device,
    )

    # 保存测试结果到 results/
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = []
    projection.eval()
    for q in test_queries_list:
        q_emb = encoder.encode(q, normalize_embeddings=True, show_progress_bar=False)
        sims = style_embeddings @ q_emb
        sorted_idx = np.argsort(sims)[::-1]
        output.append({
            "query": q,
            "dominant_style": style_names[sorted_idx[0]],
            "top3": [
                {"style": style_names[i], "similarity": round(float(sims[i]), 4)}
                for i in sorted_idx[:3]
            ],
        })

    report_path = results_dir / "projection_test_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[报告] 测试结果已保存: {report_path}")


if __name__ == "__main__":
    main()
