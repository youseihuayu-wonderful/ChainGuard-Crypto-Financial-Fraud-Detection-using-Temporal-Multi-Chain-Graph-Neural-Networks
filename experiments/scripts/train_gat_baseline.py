"""
Train GAT Baseline on Elliptic Dataset

Graph Attention Network (Veličković et al., ICLR 2018).
Uses original graph only (no temporal k-NN edges) for fair comparison with M1 (GCN).

Architecture: 2-layer GAT, 8 heads, hidden=16 per head (total 128), ~38K params.

Usage:
    python experiments/scripts/train_gat_baseline.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.elliptic_loader import load_elliptic_csv, temporal_split
from src.evaluation.metrics import compute_metrics, print_report

# ============================================================
# Configuration
# ============================================================
SEED = 42
HEADS = 8
HIDDEN_PER_HEAD = 16  # 16 * 8 = 128 total hidden
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 20
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")


class GATBaseline(torch.nn.Module):
    """
    2-layer GAT for binary node classification.
    Layer 1: multi-head attention (concat), Layer 2: single-head (mean).
    """
    def __init__(self, in_channels, hidden_per_head=16, heads=8, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_per_head, heads=heads, dropout=dropout)
        # Layer 2: single head, takes concatenated output from layer 1
        self.conv2 = GATConv(hidden_per_head * heads, hidden_per_head * heads, heads=1,
                             concat=False, dropout=dropout)
        self.classifier = torch.nn.Linear(hidden_per_head * heads, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.classifier(x).squeeze(-1)
        return out


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, data, train_mask, optimizer, class_weight):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
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
    logits = model(data.x, data.edge_index)
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    metrics = compute_metrics(labels, probs)
    return metrics, probs, labels


def main(seed=SEED):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\n{'='*60}")
    print("Loading Elliptic Dataset...")
    print(f"{'='*60}")
    data = load_elliptic_csv(DATA_DIR)
    train_mask, val_mask, test_mask = temporal_split(data)

    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    n_illicit = (data.y[train_mask] == 1).sum().float()
    n_licit = (data.y[train_mask] == 0).sum().float()
    class_weight = (n_licit / n_illicit).to(device)
    print(f"Class weight (licit/illicit): {class_weight:.2f}")

    model = GATBaseline(
        in_channels=data.x.shape[1],
        hidden_per_head=HIDDEN_PER_HEAD,
        heads=HEADS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"\n{'='*60}")
    print("Training GAT Baseline")
    print(f"{'='*60}")

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
                    print(f"Early stopping at epoch {epoch}")
                    break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n{'='*60}")
    print("Test Set Results (GAT)")
    print(f"{'='*60}")
    test_metrics, test_probs, test_labels = evaluate(model, data, test_mask)
    print_report(test_labels, test_probs)

    return test_metrics


if __name__ == "__main__":
    metrics = main()
