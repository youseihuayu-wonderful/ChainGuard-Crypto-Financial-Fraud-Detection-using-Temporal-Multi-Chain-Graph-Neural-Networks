"""
Train M2: GCN + Temporal Attention on Elliptic Dataset

M2 adds temporal self-attention on top of the GCN baseline (M1).
Uses identical data loading, splitting, and evaluation as M1 for fair comparison.

Expected: AUC should be higher than M1 (0.7449) by 1-3%.

Usage:
    python experiments/scripts/train_m2_temporal.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.elliptic_loader import load_elliptic_csv, temporal_split
from src.models.temporal_gcn import TemporalGCN
from src.evaluation.metrics import compute_metrics, print_report

# ============================================================
# Configuration (same hyperparameters as M1 for fair comparison)
# ============================================================
SEED = 42
HIDDEN_CHANNELS = 128
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 20  # early stopping (eval rounds, not epochs)
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")

# M2-specific hyperparameters
NUM_HEADS = 4
ATTN_DROPOUT = 0.1


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

    logits = model(data.x, data.edge_index, data.timestep)
    loss = F.binary_cross_entropy_with_logits(
        logits[train_mask],
        data.y[train_mask].float(),
        pos_weight=class_weight,
    )
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluate model on a given mask."""
    model.eval()
    logits = model(data.x, data.edge_index, data.timestep)
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    metrics = compute_metrics(labels, probs)
    return metrics, probs, labels


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data (identical to M1)
    print("\n" + "=" * 60)
    print("Loading Elliptic Dataset...")
    print("=" * 60)
    data = load_elliptic_csv(DATA_DIR)
    train_mask, val_mask, test_mask = temporal_split(data)

    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Compute class weight (identical to M1)
    n_illicit = (data.y[train_mask] == 1).sum().float()
    n_licit = (data.y[train_mask] == 0).sum().float()
    class_weight = (n_licit / n_illicit).to(device)
    print(f"Class weight (licit/illicit): {class_weight:.2f}")

    # Model (M2: GCN + Temporal Attention)
    model = TemporalGCN(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        attn_dropout=ATTN_DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    gcn_params = sum(p.numel() for n, p in model.named_parameters()
                     if "temporal_attention" not in n)
    temporal_params = total_params - gcn_params
    print(f"\nTotal parameters: {total_params:,}")
    print(f"  GCN parameters: {gcn_params:,} (same as M1)")
    print(f"  Temporal attention parameters: {temporal_params:,} (M2 addition)")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training loop (identical structure to M1)
    print("\n" + "=" * 60)
    print("Training M2: GCN + Temporal Attention")
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

            # Early stopping based on validation AUC
            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no improvement for {PATIENCE} eval rounds)")
                    break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\n" + "=" * 60)
    print("Test Set Results (M2: GCN + Temporal Attention)")
    print("=" * 60)
    test_metrics, test_probs, test_labels = evaluate(model, data, test_mask)
    print_report(test_labels, test_probs)

    # Compare with M1 baseline
    m1_auc = 0.7449
    delta = test_metrics["auc_roc"] - m1_auc
    print(f"\nComparison with M1 (GCN Baseline):")
    print(f"  M1 AUC: {m1_auc:.4f}")
    print(f"  M2 AUC: {test_metrics['auc_roc']:.4f}")
    print(f"  Delta:  {delta:+.4f} ({'improvement' if delta > 0 else 'regression'})")

    # Analyze temporal attention weights
    print("\n" + "=" * 60)
    print("Temporal Attention Analysis")
    print("=" * 60)
    attn_weights = model.get_attention_weights()
    if attn_weights is not None:
        # Average across heads: [T, T]
        avg_attn = attn_weights.mean(dim=0).cpu().numpy()
        # Show attention pattern for last few test timesteps
        unique_ts = sorted(data.timestep.unique().cpu().tolist())
        num_ts = len(unique_ts)
        print(f"Attention matrix shape: {avg_attn.shape} (T={num_ts} timesteps)")
        print(f"\nAttention from last 5 timesteps (test period):")
        print(f"{'From\\To':>8s}", end="")
        for j in range(max(0, num_ts - 5), num_ts):
            print(f"  ts={unique_ts[j]:2d}", end="")
        print()
        for i in range(max(0, num_ts - 5), num_ts):
            print(f"ts={unique_ts[i]:2d}  ", end="")
            for j in range(max(0, num_ts - 5), num_ts):
                print(f"  {avg_attn[i, j]:.3f}", end="")
            print()

    # Sanity checks
    if test_metrics["auc_roc"] > 0.99:
        print("\nWARNING: AUC > 0.99 -- possible data leakage!")
    elif test_metrics["auc_roc"] < m1_auc - 0.05:
        print(f"\nWARNING: AUC significantly below M1 ({m1_auc:.4f}). "
              "Temporal attention may not be helping or has a bug.")

    return test_metrics


if __name__ == "__main__":
    metrics = main()
