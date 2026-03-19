"""
Train M5: TH-GNN + Cross-Chain Label Propagation on Elliptic Dataset

M5 extends M4 (TH-GNN) with semi-supervised label propagation.
In Elliptic, 77% of nodes are unlabeled. Label propagation spreads
known label signals through the graph (especially temporal edges)
to influence predictions on unlabeled nodes.

Training loss = supervised_loss + lambda * consistency_loss
- supervised_loss: BCE on labeled nodes (same as M1-M4)
- consistency_loss: MSE between GNN predictions and LP soft labels
  on unlabeled nodes (encourages GNN to agree with graph structure)

This simulates the cross-chain scenario where:
- One chain has labels (Elliptic/Bitcoin)
- Connected chains have no labels (cross-chain transactions)
- LP propagates labels through bridge/temporal edges

Expected: AUC >= M4.

Usage:
    python experiments/scripts/train_m5_crosschain.py
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
from src.models.modules.label_propagation import LabelPropagation, consistency_loss
from src.evaluation.metrics import compute_metrics, print_report

# ============================================================
# Configuration
# ============================================================
SEED = 42
HIDDEN_CHANNELS = 80
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 20
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")

# M5-specific
NUM_RELATIONS = 2
NUM_HEADS = 4
ATTN_DROPOUT = 0.1
TEMPORAL_K = 5
LP_ITERATIONS = 10
LP_ALPHA = 0.5
CONSISTENCY_LAMBDA = 0.1  # Weight for consistency loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, data, train_mask, unlabeled_mask, optimizer,
                class_weight, lp_soft_labels, consistency_lambda):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index, data.edge_type, data.timestep)

    # Supervised loss on labeled training nodes (same as M1-M4)
    sup_loss = F.binary_cross_entropy_with_logits(
        logits[train_mask],
        data.y[train_mask].float(),
        pos_weight=class_weight,
    )

    # Consistency loss on unlabeled nodes (M5 addition)
    gnn_probs = torch.sigmoid(logits)
    cons_loss = consistency_loss(gnn_probs, lp_soft_labels, unlabeled_mask)

    total_loss = sup_loss + consistency_lambda * cons_loss
    total_loss.backward()
    optimizer.step()

    return sup_loss.item(), cons_loss.item(), total_loss.item()


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

    # Load data with temporal augmentation
    print("\n" + "=" * 60)
    print("Loading Elliptic Dataset + Temporal Edge Augmentation...")
    print("=" * 60)
    data = load_elliptic_csv(DATA_DIR)
    train_mask, val_mask, test_mask = temporal_split(data)
    data = add_temporal_edges(data, k=TEMPORAL_K)

    # Identify unlabeled nodes
    labeled_mask = data.y >= 0
    unlabeled_mask = data.y < 0
    print(f"\nLabeled: {labeled_mask.sum():,}, Unlabeled: {unlabeled_mask.sum():,}")

    # Run label propagation (M5 key step)
    print("\n" + "=" * 60)
    print("Running Label Propagation...")
    print("=" * 60)
    lp = LabelPropagation(num_iterations=LP_ITERATIONS, alpha=LP_ALPHA)

    data_device = data.to(device)
    lp_soft_labels = lp(
        data_device.edge_index,
        data_device.y,
        labeled_mask.to(device),
        data_device.num_nodes,
    )

    # Analyze LP results
    lp_unlabeled = lp_soft_labels[unlabeled_mask.to(device)]
    print(f"LP soft labels on unlabeled nodes:")
    print(f"  Mean: {lp_unlabeled.mean():.4f}")
    print(f"  Std:  {lp_unlabeled.std():.4f}")
    print(f"  Predicted illicit (>0.5): {(lp_unlabeled > 0.5).sum():,}")
    print(f"  Predicted licit (<0.5):   {(lp_unlabeled < 0.5).sum():,}")
    print(f"  Uncertain (~0.5):         {((lp_unlabeled > 0.45) & (lp_unlabeled < 0.55)).sum():,}")

    data = data_device
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    unlabeled_mask = unlabeled_mask.to(device)

    n_illicit = (data.y[train_mask] == 1).sum().float()
    n_licit = (data.y[train_mask] == 0).sum().float()
    class_weight = (n_licit / n_illicit).to(device)
    print(f"\nClass weight (licit/illicit): {class_weight:.2f}")

    # Model (same TH-GNN as M4)
    model = THGNN(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        num_relations=NUM_RELATIONS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        attn_dropout=ATTN_DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (same architecture as M4)")
    print(f"Consistency lambda: {CONSISTENCY_LAMBDA}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Training M5: TH-GNN + Label Propagation")
    print("=" * 60)

    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        sup_loss, cons_loss, total_loss = train_epoch(
            model, data, train_mask, unlabeled_mask, optimizer,
            class_weight, lp_soft_labels, CONSISTENCY_LAMBDA,
        )

        if epoch % 10 == 0 or epoch == 1:
            val_metrics, _, _ = evaluate(model, data, val_mask)
            print(f"Epoch {epoch:3d} | Sup: {sup_loss:.4f} | "
                  f"Cons: {cons_loss:.4f} | Total: {total_loss:.4f} | "
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
    print("Test Set Results (M5: TH-GNN + Label Propagation)")
    print("=" * 60)
    test_metrics, test_probs, test_labels = evaluate(model, data, test_mask)
    print_report(test_labels, test_probs)

    # Full ablation comparison
    m1_auc, m2_auc, m3_auc, m4_auc = 0.7449, 0.7937, 0.8678, None
    m5_auc = test_metrics["auc_roc"]
    print(f"\n{'='*60}")
    print(f"FULL ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"  M1 (GCN):             {m1_auc:.4f}")
    print(f"  M2 (+Temporal):       {m2_auc:.4f}  (delta: {m2_auc-m1_auc:+.4f})")
    print(f"  M3 (+Hetero):         {m3_auc:.4f}  (delta: {m3_auc-m1_auc:+.4f})")
    print(f"  M4 (TH-GNN):          (see M4 results)")
    print(f"  M5 (+LP):             {m5_auc:.4f}  (delta: {m5_auc-m1_auc:+.4f})")

    if test_metrics["auc_roc"] > 0.99:
        print("\nWARNING: AUC > 0.99 -- possible data leakage!")

    return test_metrics


if __name__ == "__main__":
    metrics = main()
