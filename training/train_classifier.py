"""
Train a multi-task classifier for graph_family + intent prediction.

Inputs:  training/data/graph_family_dataset.jsonl  (~340 records)
Outputs: training/classifier/
           classifier_best.pt   PyTorch state dict
           label_maps.json      Family/intent index mappings
           training_log.json    Per-epoch metrics

Architecture:
  prompt -> SentenceTransformer (all-MiniLM-L6-v2, 384-d)
         -> two linear heads (family, intent)

Loss: family CE * 3.0 + intent CE * 1.0  (family is the harder task)

Usage:
    cd training
    python train_classifier.py
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ─── Paths (anchored to this script, CWD-independent) ────────────────

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "graph_family_dataset.jsonl"
MODEL_DIR = SCRIPT_DIR / "classifier"
MODEL_DIR.mkdir(exist_ok=True)


# ─── 1. Load data ───────────────────────────────────────────────────

data = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print(f"Loaded {len(data)} records from {DATA_PATH.name}")

families = sorted({d["graph_family"] for d in data})
intents = sorted({d["intent"] for d in data})
family_to_idx = {f: i for i, f in enumerate(families)}
intent_to_idx = {it: i for i, it in enumerate(intents)}

label_maps = {
    "families": families,
    "intents": intents,
    "family_to_idx": family_to_idx,
    "intent_to_idx": intent_to_idx,
}
(MODEL_DIR / "label_maps.json").write_text(
    json.dumps(label_maps, indent=2), encoding="utf-8")

print(f"Families ({len(families)}): {families}")
print(f"Intents ({len(intents)}): {intents}")
print(f"Family distribution: {Counter(d['graph_family'] for d in data)}")


# ─── 2. Encode prompts ──────────────────────────────────────────────

print("\nEncoding prompts with all-MiniLM-L6-v2 ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
prompts = [d["prompt"] for d in data]
embeddings = encoder.encode(prompts, show_progress_bar=True, convert_to_numpy=True)
embeddings = torch.tensor(embeddings, dtype=torch.float32)

family_labels = torch.tensor([family_to_idx[d["graph_family"]] for d in data])
intent_labels = torch.tensor([intent_to_idx[d["intent"]] for d in data])

print(f"Embedding shape: {tuple(embeddings.shape)}")


# ─── 3. Train / val / test split (stratified by family) ─────────────

indices = list(range(len(data)))
strat = [d["graph_family"] for d in data]

train_idx, temp_idx = train_test_split(
    indices, test_size=0.2, stratify=strat, random_state=42)
temp_strat = [strat[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=temp_strat, random_state=42)

print(f"\nSplit: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

train_emb = embeddings[train_idx]
val_emb = embeddings[val_idx]
test_emb = embeddings[test_idx]

train_fam = family_labels[train_idx]
val_fam = family_labels[val_idx]
test_fam = family_labels[test_idx]

train_intent = intent_labels[train_idx]
val_intent = intent_labels[val_idx]
test_intent = intent_labels[test_idx]


# ─── 4. Model ───────────────────────────────────────────────────────

class GraphFamilyClassifier(nn.Module):
    def __init__(self, embed_dim=384, num_families=10, num_intents=7):
        super().__init__()
        self.family_head = nn.Linear(embed_dim, num_families)
        self.intent_head = nn.Linear(embed_dim, num_intents)

    def forward(self, x):
        return {
            "family": self.family_head(x),
            "intent": self.intent_head(x),
        }


model = GraphFamilyClassifier(
    embed_dim=embeddings.shape[1],
    num_families=len(families),
    num_intents=len(intents),
)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")


# ─── 5. Training loop ───────────────────────────────────────────────

print("\nTraining ...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200, eta_min=1e-5)
criterion = nn.CrossEntropyLoss()

FAMILY_WEIGHT = 3.0
INTENT_WEIGHT = 1.0
MAX_EPOCHS = 200
MAX_PATIENCE = 15

best_val_fam_acc = 0.0
best_epoch = 0
patience = 0
log = []

for epoch in range(MAX_EPOCHS):
    model.train()
    preds = model(train_emb)
    loss = (FAMILY_WEIGHT * criterion(preds["family"], train_fam)
            + INTENT_WEIGHT * criterion(preds["intent"], train_intent))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        train_preds = model(train_emb)
        train_fam_acc = (train_preds["family"].argmax(-1) == train_fam).float().mean().item()
        val_preds = model(val_emb)
        val_fam_acc = (val_preds["family"].argmax(-1) == val_fam).float().mean().item()
        val_int_acc = (val_preds["intent"].argmax(-1) == val_intent).float().mean().item()

    log.append({
        "epoch": epoch,
        "loss": float(loss.item()),
        "train_fam_acc": train_fam_acc,
        "val_fam_acc": val_fam_acc,
        "val_int_acc": val_int_acc,
        "lr": optimizer.param_groups[0]["lr"],
    })

    if epoch % 10 == 0 or epoch == MAX_EPOCHS - 1:
        print(f"  epoch {epoch:3d}  loss={loss.item():6.4f}  "
              f"train_fam={train_fam_acc:.3f}  val_fam={val_fam_acc:.3f}  "
              f"val_intent={val_int_acc:.3f}")

    if val_fam_acc > best_val_fam_acc:
        best_val_fam_acc = val_fam_acc
        best_epoch = epoch
        torch.save(model.state_dict(), MODEL_DIR / "classifier_best.pt")
        patience = 0
    else:
        patience += 1
        if patience >= MAX_PATIENCE:
            print(f"  early stopping at epoch {epoch} (no improvement for {MAX_PATIENCE} epochs)")
            break

print(f"\nBest val_fam_acc: {best_val_fam_acc:.3f} at epoch {best_epoch}")

(MODEL_DIR / "training_log.json").write_text(
    json.dumps(log, indent=2), encoding="utf-8")


# ─── 6. Test set evaluation ─────────────────────────────────────────

print("\n=== Test set evaluation ===")
model.load_state_dict(torch.load(MODEL_DIR / "classifier_best.pt"))
model.eval()

with torch.no_grad():
    test_preds = model(test_emb)
    test_fam_pred = test_preds["family"].argmax(-1)
    test_int_pred = test_preds["intent"].argmax(-1)

    test_fam_acc = (test_fam_pred == test_fam).float().mean().item()
    test_int_acc = (test_int_pred == test_intent).float().mean().item()

    top2 = test_preds["family"].topk(2, dim=-1).indices
    top2_acc = (top2 == test_fam.unsqueeze(1)).any(dim=1).float().mean().item()
    top3 = test_preds["family"].topk(3, dim=-1).indices
    top3_acc = (top3 == test_fam.unsqueeze(1)).any(dim=1).float().mean().item()

print(f"Family top-1 accuracy: {test_fam_acc:.3f}")
print(f"Family top-2 accuracy: {top2_acc:.3f}")
print(f"Family top-3 accuracy: {top3_acc:.3f}")
print(f"Intent top-1 accuracy: {test_int_acc:.3f}")

print("\n--- Family classification report ---")
print(classification_report(
    test_fam.numpy(), test_fam_pred.numpy(),
    target_names=families, zero_division=0))

print("--- Confusion matrix (rows=true, cols=pred) ---")
cm = confusion_matrix(test_fam.numpy(), test_fam_pred.numpy(),
                      labels=list(range(len(families))))
header = " " * 25 + "".join(f"{f[:8]:>9}" for f in families)
print(header)
for i, row in enumerate(cm):
    print(f"{families[i][:24]:>25}" + "".join(f"{v:>9}" for v in row))


# ─── 7. Confidence analysis ─────────────────────────────────────────

print("\n--- Confidence analysis ---")
probs = torch.softmax(test_preds["family"], dim=-1)
max_probs = probs.max(dim=-1).values
correct_mask = test_fam_pred == test_fam

if correct_mask.sum() > 0:
    correct_conf = max_probs[correct_mask].mean().item()
    print(f"Mean confidence on correct preds: {correct_conf:.3f}")
if (~correct_mask).sum() > 0:
    wrong_conf = max_probs[~correct_mask].mean().item()
    print(f"Mean confidence on wrong preds:   {wrong_conf:.3f}")
    n_wrong = int((~correct_mask).sum())
    print(f"Wrong predictions: {n_wrong}/{len(test_fam)}")


# ─── 8. Inference latency ───────────────────────────────────────────

print("\n--- Inference latency (CPU, batch=1) ---")
test_prompt = "a medieval fortress on a mountaintop"
encoder_cpu = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
model_cpu = GraphFamilyClassifier(
    embed_dim=embeddings.shape[1],
    num_families=len(families),
    num_intents=len(intents),
)
model_cpu.load_state_dict(torch.load(MODEL_DIR / "classifier_best.pt"))
model_cpu.eval()

# Warm-up
for _ in range(5):
    with torch.no_grad():
        emb = torch.tensor(encoder_cpu.encode([test_prompt]), dtype=torch.float32)
        _ = model_cpu(emb)

start = time.time()
N = 100
for _ in range(N):
    with torch.no_grad():
        emb = torch.tensor(encoder_cpu.encode([test_prompt]), dtype=torch.float32)
        _ = model_cpu(emb)
elapsed_ms = (time.time() - start) / N * 1000
print(f"End-to-end (encode + classify): {elapsed_ms:.1f} ms / prompt")


# ─── 9. Save final summary ──────────────────────────────────────────

summary = {
    "dataset_size": len(data),
    "n_families": len(families),
    "n_intents": len(intents),
    "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
    "best_epoch": best_epoch,
    "metrics": {
        "best_val_fam_acc": best_val_fam_acc,
        "test_fam_top1": test_fam_acc,
        "test_fam_top2": top2_acc,
        "test_fam_top3": top3_acc,
        "test_intent_top1": test_int_acc,
    },
    "inference_ms_cpu": elapsed_ms,
    "model_params": n_params,
}
(MODEL_DIR / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")

print(f"\nModel saved to: {MODEL_DIR}")
print("Files:", sorted(p.name for p in MODEL_DIR.glob("*")))
