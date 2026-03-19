"""
Train M3: R-GCN + Heterogeneous Edge Types on Elliptic Dataset

M3 augments the Elliptic graph with temporal k-NN edges between adjacent
timesteps, then uses R-GCN to learn separate weights for:
    - Type 0: Original payment flow edges (intra-timestep)
    - Type 1: Temporal continuity edges (cross-timestep k-NN)

Key design: Elliptic has isolated timestep subgraphs (0 cross-timestep edges).
We add temporal edges to enable heterogeneous edge modeling. R-GCN learns
that original payment edges and temporal similarity edges carry different
information for fraud detection.

Expected: AUC should be higher than M1 (0.7449).

Usage:
    python experiments/scripts/train_m3_hetero.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.elliptic_loader import load_elliptic_csv, temporal_split, add_temporal_edges
from src.models.hetero_gcn import HeteroGCN
from src.evaluation.metrics import compute_metrics, print_report

# ============================================================
# Configuration
# ============================================================
SEED = 42
HIDDEN_CHANNELS = 80  # Reduced from 128 to match M1 param count (~38K)
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 20
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")

# M3-specific
NUM_RELATIONS = 2  # original edges + temporal edges
TEMPORAL_K = 5  # k nearest neighbors for temporal edges


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, data, train_mask, optimizer, class_weight):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_type)
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
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_type)
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    metrics = compute_metrics(labels, probs)
    return metrics, probs, labels


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n" + "=" * 60)
    print("Loading Elliptic Dataset + Temporal Edge Augmentation...")
    print("=" * 60)
    data = load_elliptic_csv(DATA_DIR)
    train_mask, val_mask, test_mask = temporal_split(data)

    # Add temporal k-NN edges (M3 key step)
    data = add_temporal_edges(data, k=TEMPORAL_K)

    n_original = (data.edge_type == 0).sum().item()
    n_temporal = (data.edge_type == 1).sum().item()
    print(f"\nEdge type distribution:")
    print(f"  Original (type 0): {n_original:,} ({n_original/(n_original+n_temporal)*100:.1f}%)")
    print(f"  Temporal (type 1): {n_temporal:,} ({n_temporal/(n_original+n_temporal)*100:.1f}%)")

    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Compute class weight (identical to M1)
    n_illicit = (data.y[train_mask] == 1).sum().float()
    n_licit = (data.y[train_mask] == 0).sum().float()
    class_weight = (n_licit / n_illicit).to(device)
    print(f"Class weight (licit/illicit): {class_weight:.2f}")

    # Model
    model = HeteroGCN(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        num_relations=NUM_RELATIONS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,} (M1 has 37,889)")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Training M3: R-GCN + Heterogeneous Edges")
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

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\n" + "=" * 60)
    print("Test Set Results (M3: R-GCN + Heterogeneous Edges)")
    print("=" * 60)
    test_metrics, test_probs, test_labels = evaluate(model, data, test_mask)
    print_report(test_labels, test_probs)

    # Compare
    m1_auc = 0.7449
    m2_auc = 0.7937
    delta_m1 = test_metrics["auc_roc"] - m1_auc
    print(f"\nComparison:")
    print(f"  M1 AUC: {m1_auc:.4f} (GCN, original graph)")
    print(f"  M2 AUC: {m2_auc:.4f} (GCN + temporal attention, original graph)")
    print(f"  M3 AUC: {test_metrics['auc_roc']:.4f} (R-GCN, augmented graph)")
    print(f"  Delta vs M1: {delta_m1:+.4f}")

    if test_metrics["auc_roc"] > 0.99:
        print("\nWARNING: AUC > 0.99 -- possible data leakage!")

    return test_metrics


if __name__ == "__main__":
    metrics = main()
