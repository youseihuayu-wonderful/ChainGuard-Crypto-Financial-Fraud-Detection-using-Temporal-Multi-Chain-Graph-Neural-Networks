"""
Train M4: Full TH-GNN (Temporal + Heterogeneous) on Elliptic Dataset

M4 combines M2 (temporal attention) and M3 (heterogeneous edges) into
the complete TH-GNN model. This tests whether both components together
provide greater benefit than either alone.

Expected: AUC > M2 (0.7937) and AUC > M3 (0.8678).

Usage:
    python experiments/scripts/train_m4_thgnn.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.elliptic_loader import load_elliptic_csv, temporal_split, add_temporal_edges
from src.models.th_gnn import THGNN
from src.evaluation.metrics import compute_metrics, print_report

# ============================================================
# Configuration
# ============================================================
SEED = 42
HIDDEN_CHANNELS = 80  # Match M3 for fair comparison
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 20
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")

# M4-specific (combines M2 + M3)
NUM_RELATIONS = 2
NUM_HEADS = 4
ATTN_DROPOUT = 0.1
TEMPORAL_K = 5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, data, train_mask, optimizer, class_weight):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_type, data.timestep)
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
    logits = model(data.x, data.edge_index, data.edge_type, data.timestep)
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    metrics = compute_metrics(labels, probs)
    return metrics, probs, labels


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data with temporal augmentation (same as M3)
    print("\n" + "=" * 60)
    print("Loading Elliptic Dataset + Temporal Edge Augmentation...")
    print("=" * 60)
    data = load_elliptic_csv(DATA_DIR)
    train_mask, val_mask, test_mask = temporal_split(data)
    data = add_temporal_edges(data, k=TEMPORAL_K)

    n_original = (data.edge_type == 0).sum().item()
    n_temporal = (data.edge_type == 1).sum().item()
    print(f"\nEdge types: original={n_original:,}, temporal={n_temporal:,}")

    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    n_illicit = (data.y[train_mask] == 1).sum().float()
    n_licit = (data.y[train_mask] == 0).sum().float()
    class_weight = (n_licit / n_illicit).to(device)
    print(f"Class weight (licit/illicit): {class_weight:.2f}")

    # Model (M4: full TH-GNN)
    model = THGNN(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        num_relations=NUM_RELATIONS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        attn_dropout=ATTN_DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    rgcn_params = sum(p.numel() for n, p in model.named_parameters()
                      if "temporal_attention" not in n and "classifier" not in n)
    temporal_params = sum(p.numel() for n, p in model.named_parameters()
                         if "temporal_attention" in n)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"  R-GCN: {rgcn_params:,} | Temporal: {temporal_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Training M4: Full TH-GNN (Temporal + Heterogeneous)")
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
    print("Test Set Results (M4: Full TH-GNN)")
    print("=" * 60)
    test_metrics, test_probs, test_labels = evaluate(model, data, test_mask)
    print_report(test_labels, test_probs)

    # Compare all
    m1_auc, m2_auc, m3_auc = 0.7449, 0.7937, 0.8678
    m4_auc = test_metrics["auc_roc"]
    print(f"\nAblation Summary So Far:")
    print(f"  M1 (GCN):           {m1_auc:.4f}")
    print(f"  M2 (+Temporal):     {m2_auc:.4f}  (delta: {m2_auc-m1_auc:+.4f})")
    print(f"  M3 (+Hetero):       {m3_auc:.4f}  (delta: {m3_auc-m1_auc:+.4f})")
    print(f"  M4 (TH-GNN):        {m4_auc:.4f}  (delta: {m4_auc-m1_auc:+.4f})")

    if m4_auc < m3_auc:
        print(f"\n  Note: M4 < M3. Temporal attention may not help on augmented graph.")
    if m4_auc > 0.99:
        print("\nWARNING: AUC > 0.99 -- possible data leakage!")

    return test_metrics


if __name__ == "__main__":
    metrics = main()
