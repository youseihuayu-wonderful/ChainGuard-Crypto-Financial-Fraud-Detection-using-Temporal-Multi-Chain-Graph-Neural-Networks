"""
Generate all paper figures for ChainGuard.

Outputs:
    figures/ablation_bar_chart.pdf     - Ablation study results (M1-M5)
    figures/baseline_comparison.pdf    - Baseline comparison chart
    figures/combined_results.pdf       - All methods ranked by AUC
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Style
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "../../figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_ablation_bar_chart():
    """Figure 2: Ablation study results (M1-M5)."""
    models = ['M1\nGCN', 'M2\n+Temporal', 'M3\n+Hetero', 'M4\nTH-GNN', 'M5\n+LP']
    auc = [0.7449, 0.7937, 0.8678, 0.8535, 0.8435]
    f1 = [0.2812, 0.3663, 0.5110, 0.4927, 0.4741]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, auc, width, label='AUC-ROC', color='#2196F3', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, f1, width, label='F1 (illicit)', color='#FF9800', edgecolor='white', linewidth=0.5)

    # Highlight M3 (best)
    bars1[2].set_edgecolor('#D32F2F')
    bars1[2].set_linewidth(2.5)
    bars2[2].set_edgecolor('#D32F2F')
    bars2[2].set_linewidth(2.5)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    # Delta annotations
    deltas = ['baseline', '+4.9%', '+12.3%', '+10.9%', '+9.9%']
    for i, d in enumerate(deltas):
        if i > 0:
            ax.annotate(d, xy=(x[i] - width/2, auc[i] + 0.04),
                       fontsize=8, color='#D32F2F', ha='center', fontstyle='italic')

    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Incremental Component Contribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    path = os.path.join(FIGURES_DIR, 'ablation_bar_chart.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

    # Also save PNG for preview
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars1 = ax2.bar(x - width/2, auc, width, label='AUC-ROC', color='#2196F3', edgecolor='white', linewidth=0.5)
    bars2 = ax2.bar(x + width/2, f1, width, label='F1 (illicit)', color='#FF9800', edgecolor='white', linewidth=0.5)
    bars1[2].set_edgecolor('#D32F2F'); bars1[2].set_linewidth(2.5)
    bars2[2].set_edgecolor('#D32F2F'); bars2[2].set_linewidth(2.5)
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for i, d in enumerate(deltas):
        if i > 0:
            ax2.annotate(d, xy=(x[i] - width/2, auc[i] + 0.04),
                       fontsize=8, color='#D32F2F', ha='center', fontstyle='italic')
    ax2.set_ylabel('Score')
    ax2.set_title('Ablation Study: Incremental Component Contribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(models)
    ax2.legend(loc='upper left'); ax2.set_ylim(0, 1.05)
    fig2.savefig(os.path.join(FIGURES_DIR, 'ablation_bar_chart.png'))
    plt.close(fig2)


def plot_baseline_comparison():
    """Figure: All methods ranked by AUC-ROC."""
    methods = [
        'GCN (M1)', 'EvolveGCN-H', 'GAT',
        'Grad. Boost', 'LR',
        'RF', 'GraphSAGE',
        'TH-GNN (Ours)'
    ]
    auc = [0.7449, 0.7994, 0.8047, 0.8429, 0.8546, 0.8601, 0.8624, 0.8678]
    types = ['GNN', 'Temporal GNN', 'GNN', 'Non-graph', 'Non-graph', 'Non-graph', 'GNN', 'TH-GNN']

    color_map = {
        'Non-graph': '#9E9E9E',
        'GNN': '#2196F3',
        'Temporal GNN': '#9C27B0',
        'TH-GNN': '#D32F2F',
    }
    colors = [color_map[t] for t in types]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(methods)), auc, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    # Highlight our method
    bars[-1].set_edgecolor('#B71C1C')
    bars[-1].set_linewidth(2.5)

    for i, (bar, v) in enumerate(zip(bars, auc)):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=10,
                fontweight='bold' if i == len(methods)-1 else 'normal')

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel('AUC-ROC')
    ax.set_title('Baseline Comparison (Temporal Split)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.7, 0.90)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9E9E9E', label='Non-graph ML'),
        Patch(facecolor='#2196F3', label='Standard GNN'),
        Patch(facecolor='#9C27B0', label='Temporal GNN'),
        Patch(facecolor='#D32F2F', label='TH-GNN (Ours)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    path = os.path.join(FIGURES_DIR, 'baseline_comparison.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

    # PNG for preview
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars = ax2.barh(range(len(methods)), auc, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    bars[-1].set_edgecolor('#B71C1C'); bars[-1].set_linewidth(2.5)
    for i, (bar, v) in enumerate(zip(bars, auc)):
        ax2.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=10,
                fontweight='bold' if i == len(methods)-1 else 'normal')
    ax2.set_yticks(range(len(methods))); ax2.set_yticklabels(methods)
    ax2.set_xlabel('AUC-ROC')
    ax2.set_title('Baseline Comparison (Temporal Split)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0.7, 0.90)
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
    fig2.savefig(os.path.join(FIGURES_DIR, 'baseline_comparison.png'))
    plt.close(fig2)


def plot_precision_recall_comparison():
    """Figure: Precision vs Recall scatter for all methods."""
    methods = ['LR', 'RF', 'GB', 'GCN', 'GAT', 'SAGE', 'EvolveGCN', 'TH-GNN']
    precision = [0.1260, 0.9688, 0.6154, 0.2782, 0.2084, 0.7511, 0.1185, 0.7168]
    recall = [0.7647, 0.4559, 0.4902, 0.2843, 0.4632, 0.4216, 0.5221, 0.3971]
    types = ['Non-graph', 'Non-graph', 'Non-graph', 'GNN', 'GNN', 'GNN', 'Temporal', 'TH-GNN']

    color_map = {'Non-graph': '#9E9E9E', 'GNN': '#2196F3', 'Temporal': '#9C27B0', 'TH-GNN': '#D32F2F'}
    colors = [color_map[t] for t in types]
    sizes = [80] * 7 + [200]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(methods)):
        ax.scatter(recall[i], precision[i], c=colors[i], s=sizes[i],
                  edgecolors='black' if i == 7 else 'none', linewidths=2 if i == 7 else 0,
                  zorder=10 if i == 7 else 5)
        offset = (0.01, 0.02) if methods[i] != 'RF' else (0.01, -0.05)
        ax.annotate(methods[i], (recall[i] + offset[0], precision[i] + offset[1]), fontsize=9)

    ax.set_xlabel('Recall (illicit)')
    ax.set_ylabel('Precision (illicit)')
    ax.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 1.05)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9E9E9E', label='Non-graph'),
        Patch(facecolor='#2196F3', label='GNN'),
        Patch(facecolor='#9C27B0', label='Temporal GNN'),
        Patch(facecolor='#D32F2F', label='TH-GNN (Ours)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    path = os.path.join(FIGURES_DIR, 'precision_recall_scatter.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i in range(len(methods)):
        ax2.scatter(recall[i], precision[i], c=colors[i], s=sizes[i],
                   edgecolors='black' if i == 7 else 'none', linewidths=2 if i == 7 else 0,
                   zorder=10 if i == 7 else 5)
        offset = (0.01, 0.02) if methods[i] != 'RF' else (0.01, -0.05)
        ax2.annotate(methods[i], (recall[i] + offset[0], precision[i] + offset[1]), fontsize=9)
    ax2.set_xlabel('Recall (illicit)'); ax2.set_ylabel('Precision (illicit)')
    ax2.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 0.85); ax2.set_ylim(0, 1.05)
    ax2.legend(handles=legend_elements, loc='upper right')
    fig2.savefig(os.path.join(FIGURES_DIR, 'precision_recall_scatter.png'))
    plt.close(fig2)


if __name__ == "__main__":
    print("Generating paper figures...")
    plot_ablation_bar_chart()
    plot_baseline_comparison()
    plot_precision_recall_comparison()
    print("\nAll figures generated!")
