"""
Case Study: High-Risk Node Analysis

Compares GCN (M1) vs TH-GNN (M3) predictions on test set to identify:
1. Nodes where M3 correctly detects illicit but M1 misses
2. Feature patterns of these high-risk nodes
3. Statistical analysis of detection improvements

Output: figures/case_study_detection.pdf, figures/case_study_detection.png
"""

import os
import sys
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.elliptic_loader import load_elliptic_csv, temporal_split, add_temporal_edges
from src.models.baselines.gcn import GCNBaseline
from src.models.hetero_gcn import HeteroGCN
from src.evaluation.metrics import compute_metrics

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "../../figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../experiments/results")

sns.set_theme(style="whitegrid", font_scale=1.1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model_gcn(data, train_mask):
    set_seed(SEED)
    model = GCNBaseline(in_channels=data.x.shape[1], hidden_channels=128, dropout=0.5)
    cw = (data.y[train_mask] == 0).sum().float() / (data.y[train_mask] == 1).sum().float()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for _ in range(200):
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        F.binary_cross_entropy_with_logits(logits[train_mask], data.y[train_mask].float(), pos_weight=cw).backward()
        opt.step()
    return model


def train_model_m3(data_aug, train_mask):
    set_seed(SEED)
    model = HeteroGCN(in_channels=data_aug.x.shape[1], hidden_channels=80, num_relations=2, dropout=0.5)
    cw = (data_aug.y[train_mask] == 0).sum().float() / (data_aug.y[train_mask] == 1).sum().float()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for _ in range(50):
        opt.zero_grad()
        logits = model(data_aug.x, data_aug.edge_index, data_aug.edge_type)
        F.binary_cross_entropy_with_logits(logits[train_mask], data_aug.y[train_mask].float(), pos_weight=cw).backward()
        opt.step()
    return model


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...", flush=True)
    data = load_elliptic_csv(DATA_DIR)
    train_mask, _, test_mask = temporal_split(data)

    # Train M1
    print("Training M1 (GCN)...", flush=True)
    model_gcn = train_model_gcn(data, train_mask)
    model_gcn.eval()
    with torch.no_grad():
        probs_gcn = torch.sigmoid(model_gcn(data.x, data.edge_index)[test_mask]).numpy()

    # Train M3
    print("Loading augmented data for M3...", flush=True)
    data2 = load_elliptic_csv(DATA_DIR)
    data_aug = add_temporal_edges(data2, k=5)
    train_mask2, _, test_mask2 = temporal_split(data_aug)

    print("Training M3 (R-GCN)...", flush=True)
    model_m3 = train_model_m3(data_aug, train_mask2)
    model_m3.eval()
    with torch.no_grad():
        probs_m3 = torch.sigmoid(model_m3(data_aug.x, data_aug.edge_index, data_aug.edge_type)[test_mask2]).numpy()

    labels = data.y[test_mask].numpy()
    preds_gcn = (probs_gcn >= 0.5).astype(int)
    preds_m3 = (probs_m3 >= 0.5).astype(int)

    # Analysis
    illicit_mask = labels == 1
    n_illicit = illicit_mask.sum()

    gcn_detected = preds_gcn[illicit_mask] == 1
    m3_detected = preds_m3[illicit_mask] == 1

    # Key: nodes M3 catches but GCN misses
    m3_only = m3_detected & ~gcn_detected
    both_catch = m3_detected & gcn_detected
    gcn_only = gcn_detected & ~m3_detected
    neither = ~m3_detected & ~gcn_detected

    print(f"\n{'='*60}", flush=True)
    print("CASE STUDY: Detection Comparison on Illicit Nodes", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total illicit in test set: {n_illicit}", flush=True)
    print(f"  Both detect:    {both_catch.sum():4d} ({both_catch.sum()/n_illicit*100:.1f}%)", flush=True)
    print(f"  M3 only:        {m3_only.sum():4d} ({m3_only.sum()/n_illicit*100:.1f}%)", flush=True)
    print(f"  GCN only:       {gcn_only.sum():4d} ({gcn_only.sum()/n_illicit*100:.1f}%)", flush=True)
    print(f"  Neither:        {neither.sum():4d} ({neither.sum()/n_illicit*100:.1f}%)", flush=True)

    # Feature analysis of M3-only detected nodes
    test_features = data.x[test_mask].numpy()
    illicit_features = test_features[illicit_mask]

    if m3_only.sum() > 0:
        m3_only_features = illicit_features[m3_only]
        missed_features = illicit_features[neither]

        print(f"\nFeature analysis (M3-only detected vs missed):", flush=True)
        print(f"  M3-only mean feature norm:  {np.linalg.norm(m3_only_features, axis=1).mean():.2f}", flush=True)
        print(f"  Missed mean feature norm:   {np.linalg.norm(missed_features, axis=1).mean():.2f}", flush=True)

    # Confidence analysis
    m3_high_conf = probs_m3[illicit_mask] > 0.9
    gcn_high_conf = probs_gcn[illicit_mask] > 0.9
    print(f"\nHigh-confidence detections (p > 0.9):", flush=True)
    print(f"  GCN: {gcn_high_conf.sum()} / {n_illicit}", flush=True)
    print(f"  M3:  {m3_high_conf.sum()} / {n_illicit}", flush=True)

    # Save results
    case_results = {
        "total_illicit_test": int(n_illicit),
        "both_detect": int(both_catch.sum()),
        "m3_only": int(m3_only.sum()),
        "gcn_only": int(gcn_only.sum()),
        "neither": int(neither.sum()),
        "gcn_high_conf": int(gcn_high_conf.sum()),
        "m3_high_conf": int(m3_high_conf.sum()),
        "gcn_metrics": compute_metrics(labels, probs_gcn),
        "m3_metrics": compute_metrics(labels, probs_m3),
    }
    with open(os.path.join(RESULTS_DIR, "case_study_results.json"), "w") as f:
        json.dump(case_results, f, indent=2)
    print(f"\nSaved: experiments/results/case_study_results.json", flush=True)

    # --- Figure 1: Venn-style detection comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Detection breakdown
    categories = ['Both\nDetect', 'M3 Only\n(TH-GNN gain)', 'GCN Only', 'Neither\nDetect']
    counts = [both_catch.sum(), m3_only.sum(), gcn_only.sum(), neither.sum()]
    colors = ['#4CAF50', '#D32F2F', '#2196F3', '#9E9E9E']
    bars = axes[0].bar(categories, counts, color=colors, edgecolor='white')
    for bar, c in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(c), ha='center', fontweight='bold')
    axes[0].set_ylabel('Number of Illicit Nodes')
    axes[0].set_title('Detection Comparison\n(408 illicit nodes in test set)', fontweight='bold')

    # Panel 2: Probability distributions
    axes[1].hist(probs_gcn[illicit_mask], bins=30, alpha=0.6, label='GCN', color='#2196F3', density=True)
    axes[1].hist(probs_m3[illicit_mask], bins=30, alpha=0.6, label='M3 (TH-GNN)', color='#D32F2F', density=True)
    axes[1].axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
    axes[1].set_xlabel('Predicted P(illicit)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Prediction Distribution\n(illicit nodes only)', fontweight='bold')
    axes[1].legend()

    # Panel 3: Per-timestep recall
    timesteps = data.timestep[test_mask].numpy()
    test_ts = sorted(set(timesteps))
    gcn_recall_ts = []
    m3_recall_ts = []
    for ts in test_ts:
        ts_mask = (timesteps == ts) & illicit_mask
        if ts_mask.sum() == 0:
            gcn_recall_ts.append(0)
            m3_recall_ts.append(0)
            continue
        gcn_recall_ts.append(preds_gcn[ts_mask].sum() / ts_mask.sum())
        m3_recall_ts.append(preds_m3[ts_mask].sum() / ts_mask.sum())

    axes[2].plot(test_ts, gcn_recall_ts, 'o-', label='GCN', color='#2196F3', markersize=6)
    axes[2].plot(test_ts, m3_recall_ts, 's-', label='M3 (TH-GNN)', color='#D32F2F', markersize=6)
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('Recall (illicit)')
    axes[2].set_title('Per-Timestep Recall\n(test timesteps 42-49)', fontweight='bold')
    axes[2].legend()
    axes[2].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'case_study_detection.pdf'), dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'case_study_detection.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: figures/case_study_detection.pdf", flush=True)


if __name__ == "__main__":
    main()
