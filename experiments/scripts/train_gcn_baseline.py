"""
Train GCN Baseline (M1) on Elliptic Dataset

This is the first experiment to run. Expected AUC-ROC: 0.93-0.97.
If AUC > 0.99, there is likely a data leakage bug.

Usage:
    python experiments/scripts/train_gcn_baseline.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.elliptic_loader import load_elliptic_csv, temporal_split
from src.models.baselines.gcn import GCNBaseline
from src.evaluation.metrics import compute_metrics, print_report

# ============================================================
# Configuration
# ============================================================
SEED = 42
HIDDEN_CHANNELS = 128
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 20  # early stopping
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, data, train_mask, optimizer, class_weight):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss = F.binary_cross_entropy_with_logits(
        logits[train_mask],
        data.y[train_mask].float(),  # illicit=1, licit=0
        pos_weight=class_weight,
    )
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluate model on a given mask."""
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    # For metrics: illicit=1 is the positive class
    metrics = compute_metrics(labels, probs)
    return metrics, probs, labels


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n" + "=" * 60)
    print("Loading Elliptic Dataset...")
    print("=" * 60)
    data = load_elliptic_csv(DATA_DIR)
    train_mask, val_mask, test_mask = temporal_split(data)

    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Compute class weight for imbalanced data
    n_illicit = (data.y[train_mask] == 1).sum().float()
    n_licit = (data.y[train_mask] == 0).sum().float()
    class_weight = (n_licit / n_illicit).to(device)
    print(f"Class weight (licit/illicit): {class_weight:.2f}")

    # Model
    model = GCNBaseline(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Training GCN Baseline (M1)")
    print("=" * 60)

    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, data, train_mask, optimizer, class_weight)

        if epoch % 10 == 0 or epoch == 1:
            val_metrics, _, _ = evaluate(model, data, val_mask)
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")

            # Early stopping
            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 10
                if patience_counter >= PATIENCE * 10:
                    print(f"Early stopping at epoch {epoch}")
                    break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\n" + "=" * 60)
    print("Test Set Results (M1: GCN Baseline)")
    print("=" * 60)
    test_metrics, test_probs, test_labels = evaluate(model, data, test_mask)
    print_report(test_labels, test_probs)

    # Sanity check
    if test_metrics["auc_roc"] > 0.99:
        print("\n⚠️  WARNING: AUC > 0.99 — possible data leakage!")
    elif test_metrics["auc_roc"] < 0.80:
        print("\n⚠️  WARNING: AUC < 0.80 — model may have a bug.")
    else:
        print(f"\n✅ Results look reasonable (AUC={test_metrics['auc_roc']:.4f})")

    return test_metrics


if __name__ == "__main__":
    metrics = main()
