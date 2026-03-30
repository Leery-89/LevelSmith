"""
finetune_complexity.py
Fine-tune existing 20-dim model to 23-dim (adding mesh_complexity, detail_density, simple_ratio).

Strategy:
  1. Load best_model.pt (20-dim output)
  2. Extend final Linear layer from 20 → 23
  3. Initialize new 3 dims from style base values
  4. Fine-tune with lr=0.0001, 30 epochs (frozen earlier layers for first 5 epochs)
  5. Save as best_model.pt (overwrite)
  6. Generate preview GLBs for medieval_keep and industrial
"""

import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from model import StyleParamMLP, StyleParamLoss, count_parameters
from style_registry import (
    STYLE_REGISTRY, OUTPUT_KEYS, OUTPUT_PARAMS, OUTPUT_DIM,
    FEATURE_DIM, get_param_vector, get_feature_vector, denormalize_params,
)
from train import generate_dataset, make_dataloaders, train_one_epoch, evaluate

OLD_DIM = 20
NEW_DIM = OUTPUT_DIM  # 23


def extend_model(old_ckpt_path: Path, device: torch.device) -> StyleParamMLP:
    """Load 20-dim model and extend output layer to 23-dim."""
    ckpt = torch.load(old_ckpt_path, map_location=device, weights_only=False)
    old_state = ckpt["model_state_dict"]

    # Build new 23-dim model
    hidden_dims = ckpt.get("config", {}).get("hidden_dims", [128, 64, 32])
    dropout = ckpt.get("config", {}).get("dropout", 0.2)
    model = StyleParamMLP(
        input_dim=FEATURE_DIM,
        output_dim=NEW_DIM,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    # Copy old weights for all layers except the final Linear
    new_state = model.state_dict()
    final_w_key = None
    final_b_key = None

    for key in new_state:
        if key in old_state and new_state[key].shape == old_state[key].shape:
            new_state[key] = old_state[key]
        elif "weight" in key and new_state[key].shape[0] == NEW_DIM:
            # This is the final Linear layer (32 → 23 vs 32 → 20)
            final_w_key = key
            # Copy first 20 rows from old weights
            new_state[key][:OLD_DIM] = old_state[key]
            # Initialize new 3 rows with small random values
            nn.init.kaiming_normal_(new_state[key][OLD_DIM:].unsqueeze(0))
            print(f"  Extended {key}: {old_state[key].shape} -> {new_state[key].shape}")
        elif "bias" in key and new_state[key].shape[0] == NEW_DIM:
            final_b_key = key
            new_state[key][:OLD_DIM] = old_state[key]
            # Bias for new dims: initialize to inverse-sigmoid of target values
            # mesh_complexity ~0.35, detail_density ~0.55, simple_ratio ~0.40
            targets = [0.35, 0.55, 0.40]
            for i, t in enumerate(targets):
                # inverse sigmoid: log(t / (1-t))
                new_state[key][OLD_DIM + i] = np.log(t / (1 - t + 1e-7))
            print(f"  Extended {key}: {old_state[key].shape} -> {new_state[key].shape}")

    model.load_state_dict(new_state)
    return model.to(device)


def finetune(model, device, lr=0.0001, epochs=30, early_stop=10):
    """Fine-tune with new 23-dim targets."""
    config = {
        "n_gaussian": 60,
        "n_interp_per_pair": 10,
        "n_perturb": 8,
        "val_ratio": 0.2,
        "batch_size": 64,
        "hidden_dims": [128, 64, 32],
        "dropout": 0.2,
        "lr": lr,
        "weight_decay": 1e-4,
        "lr_patience": 5,
        "lr_factor": 0.5,
        "subdiv_weight": 0.1,
        "early_stop": early_stop,
        "epochs": epochs,
    }

    print("\n[Data] Generating 23-dim training data...")
    dataset = generate_dataset(
        n_gaussian=config["n_gaussian"],
        n_interp_per_pair=config["n_interp_per_pair"],
        n_perturb=config["n_perturb"],
        val_ratio=config["val_ratio"],
    )
    train_loader, val_loader = make_dataloaders(dataset, config["batch_size"], device)
    print(f"  Train: {len(dataset['X_train'])} | Val: {len(dataset['X_val'])}")

    criterion = StyleParamLoss(subdiv_weight=config["subdiv_weight"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6)

    # Phase 1: freeze early layers for 5 epochs (only train last layer + new dims)
    all_params = list(model.named_parameters())
    early_params = [(n, p) for n, p in all_params if "net.12" not in n]  # last Linear is net.12
    for n, p in early_params:
        p.requires_grad = False
    print(f"\n[Phase 1] Frozen early layers, training only final layer (5 epochs)")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    print(f"\n{'Ep':>4}  {'TrLoss':>10}  {'VaLoss':>10}  {'VaMAE':>9}  {'LR':>9}")
    print("-" * 52)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # Unfreeze all after phase 1
        if epoch == 6:
            for n, p in early_params:
                p.requires_grad = True
            # Reset optimizer with full params
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.5, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n[Phase 2] Unfroze all layers, trainable: {trainable:,}")
            print(f"{'Ep':>4}  {'TrLoss':>10}  {'VaLoss':>10}  {'VaMAE':>9}  {'LR':>9}")
            print("-" * 52)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch <= 5 or epoch % 5 == 0 or epoch == epochs:
            print(f"{epoch:>4}  {train_loss:>10.6f}  {val_loss:>10.6f}  {val_mae:>9.6f}  {lr_now:>9.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= early_stop:
            print(f"\n[Early Stop] epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"\n[Done] {elapsed:.1f}s | Best val loss: {best_val_loss:.6f}")
    model.load_state_dict(best_state)
    return model, best_val_loss


def save_model(model, best_val_loss, output_dir):
    """Save the fine-tuned model."""
    ckpt_path = output_dir / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": FEATURE_DIM,
            "output_dim": NEW_DIM,
            "hidden_dims": [128, 64, 32],
            "dropout": 0.2,
            "finetune_lr": 0.0001,
            "finetune_epochs": 30,
            "best_val_loss": best_val_loss,
        },
        "output_keys": OUTPUT_KEYS,
        "output_params": {k: v for k, v in OUTPUT_PARAMS.items()},
        "feature_dim": FEATURE_DIM,
        "output_dim": NEW_DIM,
    }, ckpt_path)
    print(f"[Save] {ckpt_path}")

    # Also export style params
    from train import export_trained_params
    device = next(model.parameters()).device
    ft_config = {
        "styles": list(STYLE_REGISTRY.keys()),
        "hidden_dims": [256, 128, 64, 32],
        "dropout": 0.2,
        "epochs": 30,
        "_epochs_trained": 30,
        "_best_val_loss": best_val_loss,
    }
    export_trained_params(model, device, ft_config,
                          save_path=str(output_dir / "trained_style_params.json"))


def generate_preview(model, device, style_name, output_path):
    """Generate a preview GLB for a given style."""
    import generate_level as gl
    import trimesh
    from shapely.geometry import box as sbox

    fv = torch.from_numpy(get_feature_vector(style_name)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(fv).cpu().numpy()[0]
    params = denormalize_params(pred)

    # Build a simple rectangular building
    palette = {
        "floor":   [120, 100, 80, 255], "ceiling": [140, 120, 95, 255],
        "wall":    [160, 145, 120, 255], "door":    [85, 55, 25, 255],
        "window":  [155, 195, 215, 180], "internal":[145, 130, 108, 255],
        "ground":  [90, 80, 65, 255],
    }
    # Apply wall color from params
    wc = [params.get("wall_color_r", 0.6), params.get("wall_color_g", 0.55),
          params.get("wall_color_b", 0.48)]
    params["wall_color"] = wc
    params["win_spec"] = {"density": params.get("win_density", 0.3),
                          "width": params.get("win_width", 0.8),
                          "height": params.get("win_height", 1.0)}
    params["door_spec"] = {"width": params.get("door_width", 1.0),
                           "height": params.get("door_height", 2.2)}
    params["height_range"] = [params.get("height_range_min", 3.0),
                              params.get("height_range_max", 6.0)]

    fp = sbox(0, 0, 12.0, 10.0)
    meshes = gl.build_room(params, palette, x_off=0, z_off=0, footprint=fp)

    scene = trimesh.Scene()
    for i, m in enumerate(meshes):
        scene.add_geometry(m, node_name=f"part_{i:03d}")
    scene.export(str(output_path))
    faces = sum(len(g.faces) for g in scene.geometry.values())
    print(f"  {style_name}: {output_path.name} ({faces} faces)")
    return params


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  finetune_complexity.py  --  20-dim -> 23-dim fine-tune")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Old dim: {OLD_DIM} -> New dim: {NEW_DIM}")
    print(f"  New params: mesh_complexity, detail_density, simple_ratio")

    old_ckpt = SCRIPT_DIR / "best_model.pt"
    if not old_ckpt.exists():
        sys.exit(f"Model not found: {old_ckpt}")

    # 1. Check if model is already 23-dim (from previous run)
    ckpt = torch.load(old_ckpt, map_location=device, weights_only=False)
    existing_dim = ckpt.get("output_dim", OLD_DIM)
    if existing_dim == NEW_DIM:
        print(f"\n[1/4] Model already {NEW_DIM}-dim, loading directly...")
        cfg = ckpt.get("config", {})
        # Infer hidden_dims from state_dict Linear weight shapes
        sd = ckpt["model_state_dict"]
        linear_keys = [k for k in sd if k.endswith(".weight") and len(sd[k].shape) == 2]
        # Sort numerically by layer index: net.0.weight, net.4.weight, ...
        linear_keys.sort(key=lambda k: int(k.split(".")[1]))
        hdims = [sd[k].shape[0] for k in linear_keys[:-1]]  # all except output layer
        drp = cfg.get("dropout", 0.2)
        model = StyleParamMLP(FEATURE_DIM, NEW_DIM, hdims, drp).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val = cfg.get("best_val_loss", cfg.get("_best_val_loss", 0.014))
        print(f"  Architecture: {FEATURE_DIM} -> {hdims} -> {NEW_DIM}")
        print(f"  Loaded existing model (val_loss={best_val:.6f})")
        print(f"  Skipping fine-tune (already done)")
    else:
        print(f"\n[1/4] Extending model {existing_dim} -> {NEW_DIM}...")
        model = extend_model(old_ckpt, device)
        print(f"  Total params: {count_parameters(model):,}")

        # 2. Fine-tune
        print("\n[2/4] Fine-tuning (lr=0.0001, 30 epochs)...")
        model, best_val = finetune(model, device, lr=0.0001, epochs=30, early_stop=10)

    # 3. Save
    print("\n[3/4] Saving model...")
    save_model(model, best_val, SCRIPT_DIR)

    # 4. Generate preview GLBs
    print("\n[4/4] Generating preview GLBs...")
    for style in ["medieval_keep", "industrial"]:
        out = SCRIPT_DIR / f"{style}_preview.glb"
        params = generate_preview(model, device, style, out)
        # Show the 3 new params
        mc = params.get("mesh_complexity", "?")
        dd = params.get("detail_density", "?")
        sr = params.get("simple_ratio", "?")
        print(f"    mesh_complexity={mc:.3f}  detail_density={dd:.3f}  simple_ratio={sr:.3f}")

    print("\n" + "=" * 60)
    print("  Done! Compare medieval_keep_preview.glb vs industrial_preview.glb")
    print("  medieval_keep: high complexity, many details, low simple_ratio")
    print("  industrial: low complexity, few details, high simple_ratio")
    print("=" * 60)


if __name__ == "__main__":
    main()
