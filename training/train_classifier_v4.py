"""
v4 Fine-tuned classifier: unfreeze MPNet last 2 transformer layers.

Compared to v3 (frozen encoder + MLP head, top1=0.576 CV):
  - Unfreezes encoder layers 10-11 + pooler (~7M params)
  - Differential LR: encoder 2e-5, head 5e-4
  - Re-encodes every epoch (encoder weights change)
  - Gradient clipping to prevent instability
  - 5-fold CV for robust comparison

Usage:  cd training && python train_classifier_v4.py
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "graph_family_dataset.jsonl"
MODEL_DIR = SCRIPT_DIR / "classifier"
MODEL_DIR.mkdir(exist_ok=True)

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

# ── Load data ────────────────────────────────────────────────

data = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

families = sorted({d["graph_family"] for d in data})
intents = sorted({d["intent"] for d in data})
family_to_idx = {f: i for i, f in enumerate(families)}
intent_to_idx = {it: i for i, it in enumerate(intents)}

prompts = [d["prompt"] for d in data]
strat = [d["graph_family"] for d in data]
family_labels = torch.tensor([family_to_idx[d["graph_family"]] for d in data])
intent_labels = torch.tensor([intent_to_idx[d["intent"]] for d in data])
style_vecs = torch.stack([
    torch.zeros(NUM_STYLES).scatter_(0, torch.tensor(style_to_idx.get(d.get("style", ""), 0)), 1.0)
    for d in data
])

print(f"Loaded {len(data)} records, {len(families)} families, {len(intents)} intents")


# ── Model ────────────────────────────────────────────────────

class FinetuneClassifier(nn.Module):
    def __init__(self, encoder_name="all-mpnet-base-v2", style_dim=20,
                 num_families=10, num_intents=7, device="cpu"):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.embed_dim = self.encoder.get_sentence_embedding_dimension()

        # Freeze all, then unfreeze last 2 layers + pooler
        for p in self.encoder.parameters():
            p.requires_grad = False
        for name, p in self.encoder.named_parameters():
            if "layer.10" in name or "layer.11" in name or "pooler" in name:
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder.parameters())
        print(f"  Encoder: {trainable:,} / {total:,} trainable ({trainable*100//total}%)")

        input_dim = self.embed_dim + style_dim
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

    def encode(self, texts):
        return self.encoder.encode(texts, convert_to_tensor=True,
                                   show_progress_bar=False, batch_size=64)

    def forward(self, text_emb, style_onehot):
        x = torch.cat([text_emb, style_onehot], dim=-1)
        x = self.shared(x)
        return {"family": self.family_head(x), "intent": self.intent_head(x)}


# ── CV training ──────────────────────────────────────────────

def cv_finetune(n_splits=5, max_epochs=50, patience=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Running {n_splits}-fold CV with fine-tuning...\n")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(data)), strat)):
        print(f"  Fold {fold_i+1}/{n_splits}")
        torch.manual_seed(42 + fold_i)
        np.random.seed(42 + fold_i)

        # Split indices further: tr_idx → actual train + in-fold val
        n_val = max(10, len(tr_idx) // 7)
        val_idx = tr_idx[:n_val]
        train_idx = tr_idx[n_val:]

        tr_prompts = [prompts[i] for i in train_idx]
        val_prompts_list = [prompts[i] for i in val_idx]
        te_prompts = [prompts[i] for i in te_idx]

        tr_fam = family_labels[train_idx]
        val_fam = family_labels[val_idx]
        te_fam = family_labels[te_idx]
        tr_int = intent_labels[train_idx]
        tr_style = style_vecs[train_idx]
        val_style = style_vecs[val_idx]
        te_style = style_vecs[te_idx]

        model = FinetuneClassifier(
            num_families=len(families), num_intents=len(intents),
            device=device)
        # Move MLP head to same device as encoder
        model.shared = model.shared.to(device)
        model.family_head = model.family_head.to(device)
        model.intent_head = model.intent_head.to(device)

        # Differential LR
        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in model.encoder.named_parameters() if p.requires_grad],
             "lr": 2e-5},
            {"params": list(model.shared.parameters()) +
                       list(model.family_head.parameters()) +
                       list(model.intent_head.parameters()),
             "lr": 5e-4},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()

        best_val = 0.0
        best_state = None
        bad = 0

        for epoch in range(max_epochs):
            model.train()

            # Re-encode every epoch (encoder is changing)
            tr_emb = model.encode(tr_prompts)
            # Style dropout 10%
            s = tr_style.clone()
            mask = torch.rand(len(train_idx)) < 0.10
            s[mask] = 0

            preds = model(tr_emb, s.to(tr_emb.device))
            loss = (3.0 * criterion(preds["family"], tr_fam.to(tr_emb.device)) +
                    criterion(preds["intent"], tr_int.to(tr_emb.device)))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_emb = model.encode(val_prompts_list)
                vp = model(val_emb, val_style.to(val_emb.device))
                v_acc = (vp["family"].argmax(-1) == val_fam.to(val_emb.device)).float().mean().item()

            if v_acc > best_val:
                best_val = v_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

            if epoch % 10 == 0:
                print(f"    epoch {epoch}: loss={loss.item():.3f} val={v_acc:.3f} best={best_val:.3f}")

        # Eval on test fold
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            te_emb = model.encode(te_prompts)
            tp = model(te_emb, te_style.to(te_emb.device))
            te_pred = tp["family"].argmax(-1).cpu()
            top1 = (te_pred == te_fam).float().mean().item()
            top2 = (tp["family"].cpu().topk(2, -1).indices == te_fam.unsqueeze(1)).any(1).float().mean().item()
            top3 = (tp["family"].cpu().topk(3, -1).indices == te_fam.unsqueeze(1)).any(1).float().mean().item()

        f1_per = f1_score(te_fam.numpy(), te_pred.numpy(),
                          labels=list(range(len(families))), average=None, zero_division=0)
        f1d = dict(zip(families, f1_per.tolist()))
        macro = float(f1_per.mean())

        fold_metrics.append({
            "top1": top1, "top2": top2, "top3": top3,
            "macro_f1": macro,
            "inst": f1d.get("institutional_compound", 0),
            "wall": f1d.get("walled_settlement", 0),
            "f1_per_family": f1d,
        })
        print(f"    test: top1={top1:.3f} top2={top2:.3f} top3={top3:.3f} "
              f"inst={f1d.get('institutional_compound',0):.2f} "
              f"wall={f1d.get('walled_settlement',0):.2f}")

        # Save last fold's model as the candidate best
        if fold_i == n_splits - 1:
            torch.save({
                "state_dict": best_state,
                "encoder": "all-mpnet-base-v2",
                "head_type": "mlp",
                "use_style": True,
                "embed_dim": model.embed_dim,
                "style_dim": NUM_STYLES,
                "families": families,
                "intents": intents,
                "styles": STYLES,
                "finetuned_layers": ["layer.10", "layer.11", "pooler"],
            }, MODEL_DIR / "v4_finetune.pt")
            model.encoder.save(str(MODEL_DIR / "finetuned_encoder"))

    # Aggregate
    avg = {k: float(np.mean([m[k] for m in fold_metrics]))
           for k in ["top1", "top2", "top3", "macro_f1", "inst", "wall"]}
    std = float(np.std([m["top1"] for m in fold_metrics]))
    avg["top1_std"] = std
    avg["f1_per_family"] = {f: float(np.mean([m["f1_per_family"][f] for m in fold_metrics]))
                            for f in families}
    return avg


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    v4 = cv_finetune()
    elapsed = time.time() - t0

    # Load v3 baseline
    with open(MODEL_DIR / "comparison.json") as f:
        comp = json.load(f)
    v3 = next((r for r in comp.get("cv_results", []) if "mpnet" in r["name"]), {})

    print(f"\n{'='*70}")
    print(f"  COMPARISON: v3 (frozen) vs v4 (fine-tuned)")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'Top-1':>10} {'Top-2':>7} {'Top-3':>7} {'MacroF1':>8} {'Inst':>6} {'Wall':>6}")
    print(f"  {'-'*70}")
    print(f"  {'v3_frozen_mpnet':<25} "
          f"{v3.get('top1_mean',0):>5.3f}+-{v3.get('top1_std',0):.2f}  "
          f"{v3.get('top2_mean',0):>6.3f}  {v3.get('top3_mean',0):>6.3f}  "
          f"{v3.get('macro_f1_mean',0):>6.3f}  "
          f"{v3.get('inst_f1_mean',0):>6.2f} {v3.get('wall_f1_mean',0):>6.2f}")
    print(f"  {'v4_finetune_mpnet':<25} "
          f"{v4['top1']:>5.3f}+-{v4['top1_std']:.2f}  "
          f"{v4['top2']:>6.3f}  {v4['top3']:>6.3f}  "
          f"{v4['macro_f1']:>6.3f}  "
          f"{v4['inst']:>6.2f} {v4['wall']:>6.2f}")

    delta = v4["top1"] - v3.get("top1_mean", 0)
    print(f"\n  Delta top-1: {delta:+.3f}  ({'improved' if delta > 0 else 'no improvement'})")
    print(f"  Training time: {elapsed:.0f}s")

    if delta > 0.02:
        print(f"\n  v4 is better. Updating classifier_best.pt...")
        import shutil
        shutil.copy(MODEL_DIR / "v4_finetune.pt", MODEL_DIR / "classifier_best.pt")
        print(f"  Done.")
    else:
        print(f"\n  v4 is NOT significantly better. Keeping v3 as default.")

    # Save v4 results
    comp["v4_cv"] = v4
    (MODEL_DIR / "comparison.json").write_text(
        json.dumps(comp, indent=2), encoding="utf-8")
    print(f"\n  Results saved to classifier/comparison.json")
