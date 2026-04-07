"""
Train + compare three Graph Family classifier configurations:

  v1: Linear head, all-MiniLM-L6-v2 (384-d), no style features  (baseline)
  v2: MLP head,    all-MiniLM-L6-v2 (384-d), style one-hot
  v3: MLP head,    all-mpnet-base-v2 (768-d), style one-hot

Same train/val/test splits for fair comparison. Online style dropout (10% of
training samples zero their style vector each epoch) so the model learns to
work with or without style at inference time.

Usage:
    cd training
    python train_classifier_v2.py
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# ─── Paths ───────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "graph_family_dataset.jsonl"
MODEL_DIR = SCRIPT_DIR / "classifier"
MODEL_DIR.mkdir(exist_ok=True)


# ─── Style vocabulary (must match style_registry order doesn't matter) ──

STYLES = [
    "medieval", "medieval_chapel", "medieval_keep",
    "japanese", "japanese_temple", "japanese_machiya",
    "modern", "modern_loft", "modern_villa",
    "industrial", "industrial_workshop", "industrial_powerplant",
    "fantasy", "fantasy_dungeon", "fantasy_palace",
    "horror", "horror_asylum", "horror_crypt",
    "desert", "desert_palace",
]
style_to_idx = {s: i for i, s in enumerate(STYLES)}
NUM_STYLES = len(STYLES)


def encode_style(style_key: str) -> torch.Tensor:
    """One-hot a style key. Unknown style → all zeros (matches dropout aug)."""
    vec = torch.zeros(NUM_STYLES)
    if style_key in style_to_idx:
        vec[style_to_idx[style_key]] = 1.0
    return vec


# ─── Load data ───────────────────────────────────────────────────────

data = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print(f"Loaded {len(data)} records")

families = sorted({d["graph_family"] for d in data})
intents = sorted({d["intent"] for d in data})
family_to_idx = {f: i for i, f in enumerate(families)}
intent_to_idx = {it: i for i, it in enumerate(intents)}

print(f"Families ({len(families)}): {families}")
print(f"Intents  ({len(intents)}): {intents}")

# Stratified split — same indices reused across all 3 configs
indices = list(range(len(data)))
strat = [d["graph_family"] for d in data]
train_idx, temp_idx = train_test_split(
    indices, test_size=0.2, stratify=strat, random_state=42)
temp_strat = [strat[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=temp_strat, random_state=42)
print(f"Split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

# Labels (shared across configs)
family_labels = torch.tensor([family_to_idx[d["graph_family"]] for d in data])
intent_labels = torch.tensor([intent_to_idx[d["intent"]] for d in data])

# Style vectors (shared across configs)
style_vecs = torch.stack([encode_style(d.get("style", "")) for d in data])
print(f"Style vector shape: {tuple(style_vecs.shape)}")


# ─── Model ───────────────────────────────────────────────────────────

class GraphFamilyClassifier(nn.Module):
    def __init__(self, embed_dim, style_dim, num_families, num_intents,
                 head_type="mlp"):
        super().__init__()
        self.use_style = style_dim > 0
        input_dim = embed_dim + style_dim

        if head_type == "linear":
            self.shared = nn.Identity()
            self.family_head = nn.Linear(input_dim, num_families)
            self.intent_head = nn.Linear(input_dim, num_intents)
        else:
            # Lower dropout for small dataset (272 samples in full batch)
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.10),
            )
            self.family_head = nn.Linear(64, num_families)
            self.intent_head = nn.Linear(64, num_intents)

    def forward(self, text_emb, style_onehot=None):
        if self.use_style:
            x = torch.cat([text_emb, style_onehot], dim=-1)
        else:
            x = text_emb
        x = self.shared(x)
        return {
            "family": self.family_head(x),
            "intent": self.intent_head(x),
        }


# ─── Encoder cache ───────────────────────────────────────────────────

_encoder_cache: dict[str, SentenceTransformer] = {}


def get_embeddings(model_name: str, prompts: list[str]) -> torch.Tensor:
    """Encode prompts, caching the encoder across calls."""
    if model_name not in _encoder_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n  Loading encoder: {model_name} ({device})")
        _encoder_cache[model_name] = SentenceTransformer(model_name, device=device)
    enc = _encoder_cache[model_name]
    arr = enc.encode(prompts, show_progress_bar=False, convert_to_numpy=True,
                     batch_size=64)
    return torch.tensor(arr, dtype=torch.float32)


# ─── Train one config ────────────────────────────────────────────────

def train_config(name: str, encoder_name: str, head_type: str,
                 use_style: bool, epochs: int = 300, patience: int = 25,
                 lr: float = 5e-4, family_weight: float = 3.0,
                 style_dropout: float = 0.10, seed: int = 42) -> dict:
    """Train one configuration. Returns metrics dict."""
    print(f"\n{'='*70}")
    print(f"  {name}: encoder={encoder_name}  head={head_type}  "
          f"style={use_style}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Encode prompts (cached encoder)
    prompts = [d["prompt"] for d in data]
    embeddings = get_embeddings(encoder_name, prompts)
    print(f"  Embedding shape: {tuple(embeddings.shape)}")

    train_emb = embeddings[train_idx]
    val_emb = embeddings[val_idx]
    test_emb = embeddings[test_idx]
    train_fam = family_labels[train_idx]
    val_fam = family_labels[val_idx]
    test_fam = family_labels[test_idx]
    train_intent = intent_labels[train_idx]
    val_intent = intent_labels[val_idx]
    test_intent = intent_labels[test_idx]
    train_style_full = style_vecs[train_idx]
    val_style = style_vecs[val_idx]
    test_style = style_vecs[test_idx]

    style_dim = NUM_STYLES if use_style else 0
    model = GraphFamilyClassifier(
        embed_dim=embeddings.shape[1],
        style_dim=style_dim,
        num_families=len(families),
        num_intents=len(intents),
        head_type=head_type,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_fam_acc = 0.0
    best_epoch = 0
    bad_epochs = 0
    best_state = None

    for epoch in range(epochs):
        model.train()

        # Online style dropout: randomly mask 10% of training style vectors
        if use_style and style_dropout > 0:
            train_style = train_style_full.clone()
            mask = torch.rand(len(train_idx)) < style_dropout
            train_style[mask] = 0.0
        else:
            train_style = train_style_full if use_style else None

        preds = model(train_emb, train_style)
        loss = (family_weight * criterion(preds["family"], train_fam)
                + criterion(preds["intent"], train_intent))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_in_style = val_style if use_style else None
            vp = model(val_emb, val_in_style)
            val_fam_acc = (vp["family"].argmax(-1) == val_fam).float().mean().item()

        if val_fam_acc > best_val_fam_acc:
            best_val_fam_acc = val_fam_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

        if epoch % 30 == 0:
            print(f"    epoch {epoch:3d}  loss={loss.item():6.3f}  "
                  f"val_fam={val_fam_acc:.3f}  best={best_val_fam_acc:.3f}")

    print(f"  Trained {epoch+1} epochs, best val_fam_acc={best_val_fam_acc:.3f} "
          f"@ epoch {best_epoch}")

    # Load best and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_in_style = test_style if use_style else None
        tp = model(test_emb, test_in_style)
        test_fam_pred = tp["family"].argmax(-1)
        test_int_pred = tp["intent"].argmax(-1)

        top1 = (test_fam_pred == test_fam).float().mean().item()
        top2_idx = tp["family"].topk(2, dim=-1).indices
        top2 = (top2_idx == test_fam.unsqueeze(1)).any(dim=1).float().mean().item()
        top3_idx = tp["family"].topk(3, dim=-1).indices
        top3 = (top3_idx == test_fam.unsqueeze(1)).any(dim=1).float().mean().item()
        intent_top1 = (test_int_pred == test_intent).float().mean().item()

    # Per-family F1
    f1_per = f1_score(
        test_fam.numpy(), test_fam_pred.numpy(),
        labels=list(range(len(families))), average=None, zero_division=0)
    f1_dict = dict(zip(families, f1_per.tolist()))
    macro_f1 = float(f1_per.mean())

    # Critical: f1 of the two formerly-failing families
    institutional_f1 = f1_dict.get("institutional_compound", 0.0)
    walled_f1 = f1_dict.get("walled_settlement", 0.0)

    # Find worst family
    worst_fam = min(f1_dict.items(), key=lambda kv: kv[1])

    # Save model + metadata
    save_path = MODEL_DIR / f"{name}.pt"
    torch.save({
        "state_dict": best_state,
        "encoder": encoder_name,
        "head_type": head_type,
        "use_style": use_style,
        "embed_dim": embeddings.shape[1],
        "style_dim": style_dim,
        "families": families,
        "intents": intents,
        "styles": STYLES,
    }, save_path)

    return {
        "name": name,
        "encoder": encoder_name,
        "head_type": head_type,
        "use_style": use_style,
        "params": n_params,
        "best_epoch": best_epoch,
        "trained_epochs": epoch + 1,
        "val_fam_acc": best_val_fam_acc,
        "test_top1": top1,
        "test_top2": top2,
        "test_top3": top3,
        "test_intent_top1": intent_top1,
        "macro_f1": macro_f1,
        "f1_per_family": f1_dict,
        "institutional_f1": institutional_f1,
        "walled_f1": walled_f1,
        "worst_family": worst_fam[0],
        "worst_f1": worst_fam[1],
        "test_predictions": test_fam_pred.tolist(),
        "test_truth": test_fam.tolist(),
    }


# ─── Run all three configs ───────────────────────────────────────────

results = []

results.append(train_config(
    name="v1_linear_minilm",
    encoder_name="all-MiniLM-L6-v2",
    head_type="linear",
    use_style=False,
    epochs=200, patience=15, lr=1e-3,
))

results.append(train_config(
    name="v2_mlp_minilm_style",
    encoder_name="all-MiniLM-L6-v2",
    head_type="mlp",
    use_style=True,
    epochs=300, patience=30, lr=1e-3,
))

results.append(train_config(
    name="v3_mlp_mpnet_style",
    encoder_name="all-mpnet-base-v2",
    head_type="mlp",
    use_style=True,
    epochs=300, patience=30, lr=1e-3,
))


# ─── Comparison table ────────────────────────────────────────────────

print(f"\n\n{'='*78}")
print(f"  COMPARISON")
print(f"{'='*78}")

hdr = f"  {'Config':<28} {'Top-1':>7} {'Top-2':>7} {'Top-3':>7} {'Macro-F1':>9} {'Inst':>6} {'Wall':>6}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in results:
    print(f"  {r['name']:<28} "
          f"{r['test_top1']:>7.3f} "
          f"{r['test_top2']:>7.3f} "
          f"{r['test_top3']:>7.3f} "
          f"{r['macro_f1']:>9.3f} "
          f"{r['institutional_f1']:>6.2f} "
          f"{r['walled_f1']:>6.2f}")

print(f"\n  Worst family per config:")
for r in results:
    print(f"    {r['name']:<28} {r['worst_family']:<25} F1={r['worst_f1']:.2f}")

print(f"\n  Per-family F1:")
fam_hdr = f"    {'Family':<25}" + "".join(f"{r['name'][:14]:>16}" for r in results)
print(fam_hdr)
print("    " + "-" * (len(fam_hdr) - 4))
for fam in families:
    line = f"    {fam:<25}"
    for r in results:
        line += f"{r['f1_per_family'][fam]:>16.2f}"
    print(line)


# ─── 5-fold cross-validation for robust numbers ──────────────────────

print(f"\n\n{'='*78}")
print(f"  5-FOLD CROSS-VALIDATION (each fold uses ~272 train, 68 test)")
print(f"{'='*78}")


def cv_one_config(name: str, encoder_name: str, head_type: str,
                  use_style: bool, n_splits: int = 5) -> dict:
    """Run k-fold CV for one config. Returns averaged metrics."""
    print(f"\n  {name}: encoder={encoder_name} head={head_type} style={use_style}")

    prompts = [d["prompt"] for d in data]
    embeddings = get_embeddings(encoder_name, prompts)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_top1 = []
    fold_top2 = []
    fold_top3 = []
    fold_macro_f1 = []
    fold_inst_f1 = []
    fold_wall_f1 = []
    fold_f1_per_family = {f: [] for f in families}

    for fold_i, (tr_idx, te_idx) in enumerate(
        skf.split(np.zeros(len(data)), strat)
    ):
        torch.manual_seed(42 + fold_i)
        np.random.seed(42 + fold_i)

        tr_emb = embeddings[tr_idx]
        te_emb = embeddings[te_idx]
        tr_fam = family_labels[tr_idx]
        te_fam = family_labels[te_idx]
        tr_int = intent_labels[tr_idx]
        te_int = intent_labels[te_idx]
        tr_style_full = style_vecs[tr_idx]
        te_style = style_vecs[te_idx]

        style_dim = NUM_STYLES if use_style else 0
        model = GraphFamilyClassifier(
            embed_dim=embeddings.shape[1],
            style_dim=style_dim,
            num_families=len(families),
            num_intents=len(intents),
            head_type=head_type,
        )

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()

        # Use last 15% of training as in-fold val for early stopping
        n_val = max(10, len(tr_idx) // 7)
        in_val_idx = tr_idx[:n_val]
        in_train_idx = tr_idx[n_val:]
        in_tr_emb = embeddings[in_train_idx]
        in_val_emb = embeddings[in_val_idx]
        in_tr_fam = family_labels[in_train_idx]
        in_val_fam = family_labels[in_val_idx]
        in_tr_int = intent_labels[in_train_idx]
        in_val_int = intent_labels[in_val_idx]
        in_tr_style_full = style_vecs[in_train_idx]
        in_val_style = style_vecs[in_val_idx]

        best_val = 0.0
        best_state = None
        bad = 0
        for epoch in range(150):
            model.train()
            if use_style:
                in_tr_style = in_tr_style_full.clone()
                mask = torch.rand(len(in_train_idx)) < 0.10
                in_tr_style[mask] = 0
            else:
                in_tr_style = None
            preds = model(in_tr_emb, in_tr_style)
            loss = 3.0 * crit(preds["family"], in_tr_fam) + crit(preds["intent"], in_tr_int)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                vp = model(in_val_emb, in_val_style if use_style else None)
                v_acc = (vp["family"].argmax(-1) == in_val_fam).float().mean().item()
            if v_acc > best_val:
                best_val = v_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 25:
                    break

        # Eval on this fold's test
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            tp = model(te_emb, te_style if use_style else None)
            te_pred = tp["family"].argmax(-1)
            top1 = (te_pred == te_fam).float().mean().item()
            top2 = (tp["family"].topk(2, -1).indices == te_fam.unsqueeze(1)).any(1).float().mean().item()
            top3 = (tp["family"].topk(3, -1).indices == te_fam.unsqueeze(1)).any(1).float().mean().item()

        f1_per = f1_score(te_fam.numpy(), te_pred.numpy(),
                          labels=list(range(len(families))),
                          average=None, zero_division=0)
        f1d = dict(zip(families, f1_per.tolist()))

        fold_top1.append(top1)
        fold_top2.append(top2)
        fold_top3.append(top3)
        fold_macro_f1.append(float(f1_per.mean()))
        fold_inst_f1.append(f1d.get("institutional_compound", 0.0))
        fold_wall_f1.append(f1d.get("walled_settlement", 0.0))
        for f in families:
            fold_f1_per_family[f].append(f1d[f])

        print(f"    fold {fold_i+1}/{n_splits}  top1={top1:.3f}  "
              f"inst={f1d.get('institutional_compound', 0):.2f}  "
              f"wall={f1d.get('walled_settlement', 0):.2f}")

    return {
        "name": name,
        "top1_mean": float(np.mean(fold_top1)),
        "top1_std": float(np.std(fold_top1)),
        "top2_mean": float(np.mean(fold_top2)),
        "top3_mean": float(np.mean(fold_top3)),
        "macro_f1_mean": float(np.mean(fold_macro_f1)),
        "inst_f1_mean": float(np.mean(fold_inst_f1)),
        "wall_f1_mean": float(np.mean(fold_wall_f1)),
        "f1_per_family": {f: float(np.mean(v)) for f, v in fold_f1_per_family.items()},
    }


cv_results = []
cv_results.append(cv_one_config("v1_linear_minilm",      "all-MiniLM-L6-v2", "linear", False))
cv_results.append(cv_one_config("v2_mlp_minilm_style",   "all-MiniLM-L6-v2", "mlp",    True))
cv_results.append(cv_one_config("v3_mlp_mpnet_style",    "all-mpnet-base-v2", "mlp",   True))

print(f"\n\n{'='*78}")
print(f"  CV COMPARISON (5-fold averages)")
print(f"{'='*78}")
hdr = (f"  {'Config':<28} {'Top-1':>10} {'Top-2':>7} {'Top-3':>7} "
       f"{'Macro-F1':>9} {'Inst':>6} {'Wall':>6}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in cv_results:
    print(f"  {r['name']:<28} "
          f"{r['top1_mean']:>5.3f}±{r['top1_std']:.2f} "
          f"{r['top2_mean']:>7.3f} "
          f"{r['top3_mean']:>7.3f} "
          f"{r['macro_f1_mean']:>9.3f} "
          f"{r['inst_f1_mean']:>6.2f} "
          f"{r['wall_f1_mean']:>6.2f}")

print(f"\n  Per-family F1 (CV averages):")
fam_hdr = f"    {'Family':<25}" + "".join(f"{r['name'][:14]:>16}" for r in cv_results)
print(fam_hdr)
print("    " + "-" * (len(fam_hdr) - 4))
for fam in families:
    line = f"    {fam:<25}"
    for r in cv_results:
        line += f"{r['f1_per_family'][fam]:>16.2f}"
    print(line)


# ─── Pick best config and write summary ──────────────────────────────

best = max(results, key=lambda r: r["test_top1"])
print(f"\n  Best config: {best['name']}  (top-1={best['test_top1']:.3f})")

# Save comparison json
comparison = {
    "single_split_results": [{k: v for k, v in r.items()
                              if k not in ("test_predictions", "test_truth")}
                             for r in results],
    "cv_results": cv_results,
    "best_config": best["name"],
}
(MODEL_DIR / "comparison.json").write_text(
    json.dumps(comparison, indent=2), encoding="utf-8")

# Update label_maps with style vocab
label_maps = {
    "families": families,
    "intents": intents,
    "styles": STYLES,
    "family_to_idx": family_to_idx,
    "intent_to_idx": intent_to_idx,
    "style_to_idx": style_to_idx,
}
(MODEL_DIR / "label_maps.json").write_text(
    json.dumps(label_maps, indent=2), encoding="utf-8")

# Copy best to classifier_best.pt
import shutil
shutil.copy(MODEL_DIR / f"{best['name']}.pt", MODEL_DIR / "classifier_best.pt")

print(f"\n  Saved files in {MODEL_DIR}:")
for p in sorted(MODEL_DIR.glob("*")):
    print(f"    {p.name}")
