"""
Generate t-SNE node embedding visualizations.

Extracts last-layer embeddings from M1 (GCN) and M3 (R-GCN+Hetero),
then uses t-SNE to project to 2D. Colored by label (illicit=red, licit=blue).

Output: figures/tsne_embeddings.pdf, figures/tsne_embeddings.png
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.elliptic_loader import load_elliptic_csv, temporal_split, add_temporal_edges
from src.models.baselines.gcn import GCNBaseline
from src.models.hetero_gcn import HeteroGCN

SEED = 42
EPOCHS_GCN = 100
EPOCHS_M3 = 50  # Fewer epochs for M3 (2M edges on CPU)
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "../../figures")

sns.set_theme(style="white", font_scale=1.1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_and_extract_gcn(data, train_mask, test_mask):
    """Train M1 (GCN) and extract test node embeddings."""
    set_seed(SEED)
    model = GCNBaseline(in_channels=data.x.shape[1], hidden_channels=128, dropout=0.5)
    class_weight = (data.y[train_mask] == 0).sum().float() / (data.y[train_mask] == 1).sum().float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1, EPOCHS_GCN + 1):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], data.y[train_mask].float(), pos_weight=class_weight)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  GCN epoch {epoch}/{EPOCHS_GCN}, loss={loss.item():.4f}", flush=True)

    # Extract embeddings before classifier
    model.eval()
    with torch.no_grad():
        x = model.conv1(data.x, data.edge_index)
        x = torch.relu(x)
        x = model.conv2(x, data.edge_index)
        x = torch.relu(x)
    return x[test_mask].numpy()


def train_and_extract_m3(data_aug, train_mask, test_mask):
    """Train M3 (R-GCN+Hetero) and extract test node embeddings."""
    set_seed(SEED)
    model = HeteroGCN(in_channels=data_aug.x.shape[1], hidden_channels=80, num_relations=2, dropout=0.5)
    class_weight = (data_aug.y[train_mask] == 0).sum().float() / (data_aug.y[train_mask] == 1).sum().float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1, EPOCHS_M3 + 1):
        optimizer.zero_grad()
        logits = model(data_aug.x, data_aug.edge_index, data_aug.edge_type)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], data_aug.y[train_mask].float(), pos_weight=class_weight)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"  M3 epoch {epoch}/{EPOCHS_M3}, loss={loss.item():.4f}", flush=True)

    # Extract embeddings before classifier (using conv1 + conv2 directly)
    model.eval()
    with torch.no_grad():
        h = model.conv1(data_aug.x, data_aug.edge_index, data_aug.edge_type)
        h = torch.relu(h)
        h = model.conv2(h, data_aug.edge_index, data_aug.edge_type)
        h = torch.relu(h)
    return h[test_mask].numpy()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data...", flush=True)
    data = load_elliptic_csv(DATA_DIR)
    train_mask, _, test_mask = temporal_split(data)

    labels = data.y[test_mask].numpy()
    print(f"Test: {test_mask.sum()} nodes ({(labels==1).sum()} illicit, {(labels==0).sum()} licit)", flush=True)

    # --- M1: GCN ---
    print("\nTraining M1 (GCN)...", flush=True)
    emb_gcn = train_and_extract_gcn(data, train_mask, test_mask)
    print(f"  GCN embeddings: {emb_gcn.shape}", flush=True)

    # --- M3: R-GCN ---
    print("\nLoading augmented graph for M3...", flush=True)
    data2 = load_elliptic_csv(DATA_DIR)
    data_aug = add_temporal_edges(data2, k=5)
    train_mask2, _, test_mask2 = temporal_split(data_aug)

    print("Training M3 (R-GCN+Hetero)...", flush=True)
    emb_m3 = train_and_extract_m3(data_aug, train_mask2, test_mask2)
    print(f"  M3 embeddings: {emb_m3.shape}", flush=True)

    # --- t-SNE ---
    print("\nRunning t-SNE on GCN embeddings...", flush=True)
    tsne1 = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    proj_gcn = tsne1.fit_transform(emb_gcn)

    print("Running t-SNE on M3 embeddings...", flush=True)
    tsne2 = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    proj_m3 = tsne2.fit_transform(emb_m3)

    # --- Plot ---
    print("Generating figure...", flush=True)
    licit = labels == 0
    illicit = labels == 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, proj, title in [
        (axes[0], proj_gcn, 'M1: GCN (AUC=0.7449)'),
        (axes[1], proj_m3, 'M3: R-GCN+Hetero (AUC=0.8678)'),
    ]:
        ax.scatter(proj[licit, 0], proj[licit, 1],
                  c='#2196F3', alpha=0.3, s=8, label='Licit', rasterized=True)
        ax.scatter(proj[illicit, 0], proj[illicit, 1],
                  c='#D32F2F', alpha=0.8, s=25, label='Illicit',
                  edgecolors='black', linewidths=0.3, rasterized=True)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', markerscale=2)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('t-SNE Node Embeddings: GCN vs TH-GNN (R-GCN+Hetero)',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(os.path.join(FIGURES_DIR, 'tsne_embeddings.pdf'), dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'tsne_embeddings.png'), dpi=150)
    plt.close(fig)
    print(f"\nSaved: figures/tsne_embeddings.pdf", flush=True)


if __name__ == "__main__":
    main()
