"""
Train EvolveGCN-H Baseline on Elliptic Dataset

EvolveGCN-H (Pareja et al., AAAI 2020): Uses GRU to evolve GCN weight matrices
across timesteps. The GCN weight matrix is the hidden state of the GRU.

This is a temporal GNN baseline that processes graph snapshots sequentially.
Uses original graph only (no temporal k-NN edges).

Usage:
    python experiments/scripts/train_evolvegcn_baseline.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.elliptic_loader import load_elliptic_csv, temporal_split
from src.evaluation.metrics import compute_metrics, print_report

# ============================================================
# Configuration
# ============================================================
SEED = 42
HIDDEN_CHANNELS = 64  # Smaller to keep GRU hidden manageable (166*64=10,624)
DROPOUT = 0.5
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
EPOCHS = 100  # Fewer epochs since each epoch processes 49 snapshots
PATIENCE = 15
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")


class EvolveGCNH(nn.Module):
    """
    EvolveGCN-H: GRU evolves the first GCN layer's weight matrix across timesteps.

    Architecture per timestep:
        1. GCN layer 1 with evolved weights W_t
        2. Static GCN layer 2
        3. Classifier MLP
        4. GRU updates: W_{t+1} = GRU(summary(H_t), flatten(W_t))
    """

    def __init__(self, in_channels, hidden_channels=64, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Initial weights for GCN layer 1 (evolved by GRU)
        self.conv1_weight_init = nn.Parameter(torch.empty(in_channels, hidden_channels))
        self.conv1_bias = nn.Parameter(torch.zeros(hidden_channels))
        nn.init.xavier_uniform_(self.conv1_weight_init)

        # GRU to evolve conv1 weight matrix
        gru_hidden_size = in_channels * hidden_channels  # 166*64=10,624
        self.gru = nn.GRUCell(
            input_size=hidden_channels,
            hidden_size=gru_hidden_size,
        )

        # Static second GCN layer + classifier
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)

    def _gcn_manual(self, x, edge_index, weight, bias):
        """GCN layer with given weight matrix (for evolved weights)."""
        num_nodes = x.size(0)
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_sl

        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Transform: X @ W + b
        out = x @ weight + bias

        # Aggregate with normalization
        msg = out[col] * norm.unsqueeze(-1)
        agg = torch.zeros(num_nodes, out.size(1), device=x.device)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)
        return agg

    def forward_snapshot(self, x, edge_index, gru_hidden):
        """Process one timestep snapshot, evolve weights via GRU."""
        weight = gru_hidden.view(self.in_channels, self.hidden_channels)

        h = self._gcn_manual(x, edge_index, weight, self.conv1_bias)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Evolve weights via GRU
        node_summary = h.mean(dim=0, keepdim=True)  # [1, hidden]
        new_hidden = self.gru(node_summary, gru_hidden)

        return h, new_hidden

    def init_hidden(self, device):
        return self.conv1_weight_init.flatten().unsqueeze(0).to(device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_snapshots(data):
    """Build per-timestep graph snapshots (vectorized)."""
    timesteps = data.timestep.numpy()
    unique_ts = sorted(np.unique(timesteps))
    edge_src = data.edge_index[0].numpy()
    edge_dst = data.edge_index[1].numpy()

    snapshots = []
    for ts in unique_ts:
        ts_mask = timesteps == ts
        node_indices = np.where(ts_mask)[0]
        node_set = set(node_indices.tolist())

        # Create old->new index mapping
        idx_map = np.full(data.num_nodes, -1, dtype=np.int64)
        idx_map[node_indices] = np.arange(len(node_indices))

        # Vectorized edge filtering: both endpoints must be in this timestep
        src_in = np.isin(edge_src, node_indices)
        dst_in = np.isin(edge_dst, node_indices)
        edge_mask = src_in & dst_in

        local_src = idx_map[edge_src[edge_mask]]
        local_dst = idx_map[edge_dst[edge_mask]]
        local_ei = torch.tensor(np.stack([local_src, local_dst]), dtype=torch.long)

        snapshots.append({
            "x": data.x[node_indices],
            "edge_index": local_ei,
            "y": data.y[node_indices],
            "global_indices": node_indices,
            "timestep": ts,
        })

    return snapshots


def main(seed=SEED):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Loading Elliptic Dataset...", flush=True)
    print(f"{'='*60}", flush=True)
    data = load_elliptic_csv(DATA_DIR)
    train_mask, val_mask, test_mask = temporal_split(data)

    print("Building per-timestep graph snapshots...", flush=True)
    snapshots = build_snapshots(data)
    print(f"Built {len(snapshots)} snapshots", flush=True)

    # Pre-compute local masks for each snapshot
    train_mask_np = train_mask.numpy()
    val_mask_np = val_mask.numpy()
    for snap in snapshots:
        gi = snap["global_indices"]
        snap["local_train_mask"] = torch.tensor(train_mask_np[gi], dtype=torch.bool)
        snap["local_val_mask"] = torch.tensor(val_mask_np[gi], dtype=torch.bool)

    n_illicit = (data.y[train_mask] == 1).sum().float()
    n_licit = (data.y[train_mask] == 0).sum().float()
    class_weight = (n_licit / n_illicit).to(device)
    print(f"Class weight: {class_weight:.2f}", flush=True)

    model = EvolveGCNH(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"\n{'='*60}", flush=True)
    print("Training EvolveGCN-H", flush=True)
    print(f"{'='*60}", flush=True)

    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        optimizer.zero_grad()
        gru_hidden = model.init_hidden(device)
        total_loss = 0.0
        train_count = 0

        for snap in snapshots:
            x = snap["x"].to(device)
            ei = snap["edge_index"].to(device)
            y = snap["y"].to(device)
            local_train = snap["local_train_mask"].to(device)

            if x.size(0) == 0:
                continue

            h, gru_hidden = model.forward_snapshot(x, ei, gru_hidden)
            gru_hidden = gru_hidden.detach()  # Truncate BPTT to save memory

            if local_train.any():
                logits = model.classifier(h).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(
                    logits[local_train], y[local_train].float(), pos_weight=class_weight
                )
                loss.backward()
                train_count += local_train.sum().item()
                total_loss += loss.item() * local_train.sum().item()

        if train_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            avg_loss = total_loss / train_count

        # ---- Evaluate ----
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                gru_h = model.init_hidden(device)
                val_probs_list = []
                val_labels_list = []

                for snap in snapshots:
                    x = snap["x"].to(device)
                    ei = snap["edge_index"].to(device)
                    local_val = snap["local_val_mask"].to(device)

                    if x.size(0) == 0:
                        continue

                    h, gru_h = model.forward_snapshot(x, ei, gru_h)

                    if local_val.any():
                        logits = model.classifier(h).squeeze(-1)
                        probs = torch.sigmoid(logits[local_val]).cpu().numpy()
                        labels = snap["y"][snap["local_val_mask"]].numpy()
                        val_probs_list.append(probs)
                        val_labels_list.append(labels)

                if val_probs_list:
                    val_probs = np.concatenate(val_probs_list)
                    val_labels = np.concatenate(val_labels_list)
                    val_metrics = compute_metrics(val_labels, val_probs)

                    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                          f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                          f"Val F1: {val_metrics['f1']:.4f}", flush=True)

                    if val_metrics["auc_roc"] > best_val_auc:
                        best_val_auc = val_metrics["auc_roc"]
                        patience_counter = 0
                        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        patience_counter += 1
                        if patience_counter >= PATIENCE:
                            print(f"Early stopping at epoch {epoch}", flush=True)
                            break

    # ---- Test ----
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_mask_np = test_mask.numpy()
    model.eval()
    with torch.no_grad():
        gru_h = model.init_hidden(device)
        test_probs_list = []
        test_labels_list = []

        for snap in snapshots:
            x = snap["x"].to(device)
            ei = snap["edge_index"].to(device)
            gi = snap["global_indices"]

            if x.size(0) == 0:
                continue

            h, gru_h = model.forward_snapshot(x, ei, gru_h)

            local_test = torch.tensor(test_mask_np[gi], dtype=torch.bool)
            if local_test.any():
                logits = model.classifier(h).squeeze(-1)
                probs = torch.sigmoid(logits[local_test.to(device)]).cpu().numpy()
                labels = snap["y"][local_test].numpy()
                test_probs_list.append(probs)
                test_labels_list.append(labels)

        test_probs = np.concatenate(test_probs_list)
        test_labels = np.concatenate(test_labels_list)

    print(f"\n{'='*60}", flush=True)
    print("Test Set Results (EvolveGCN-H)", flush=True)
    print(f"{'='*60}", flush=True)
    print_report(test_labels, test_probs)

    test_metrics = compute_metrics(test_labels, test_probs)
    return test_metrics


if __name__ == "__main__":
    metrics = main()
