# ChainGuard

### Temporal Heterogeneous Graph Neural Networks for Cross-Chain Cryptocurrency Fraud Detection

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

Cross-chain bridges have become the primary vector for cryptocurrency money laundering, with over $28 billion in cross-chain token transfers in 2024 alone. Existing graph neural network (GNN)-based fraud detection methods operate exclusively on single-chain data, failing to trace transaction trails that fragment across multiple blockchains. The only prior work on cross-chain graph-based detection (GMM-CCT, BSCI 2024) achieved just 57% precision and 43% recall — insufficient for practical deployment.

We propose **ChainGuard**, a **Temporal Heterogeneous Graph Neural Network (TH-GNN)** designed specifically for cross-chain fraud detection. Our method makes three contributions:

1. **Cross-Chain Heterogeneous Graph Construction** — A unified multi-chain transaction graph where native transactions and cross-chain bridge operations are modeled as distinct edge types with different message-passing schemes.
2. **Temporal Attention over Graph Snapshots** — A dual attention mechanism that captures both structural neighborhood patterns and temporal evolution of fraud networks across time windows.
3. **Ablation-Driven Analysis of Cross-Chain Information Value** — Systematic experiments quantifying the marginal contribution of cross-chain signals to fraud detection performance.

We evaluate on the XChainDataGen dataset (11.28M cross-chain transactions, 8 bridges, 11 blockchains) combined with labeled attack data from BridgeGuard (203 attack + 40K normal transactions) and Elliptic2 (50M nodes, 200M edges). Our ablation study answers a key open question: **does cross-chain information actually improve fraud detection, and by how much?**

## Research Questions

- **RQ1**: How should cross-chain bridge transactions be represented in a graph neural network to preserve inter-chain relationships?
- **RQ2**: Does incorporating cross-chain bridge information significantly improve fraud detection compared to single-chain GNN methods?
- **RQ3**: What is the relative contribution of temporal modeling vs. cross-chain topology vs. heterogeneous edge types to detection performance?

## Method: TH-GNN Architecture

```
Input: Multi-chain transaction graph G = (V, E_native ∪ E_bridge, X, T)

┌─────────────────────────────────────────────────────┐
│  1. Cross-Chain Heterogeneous Graph Construction     │
│     • Native edges: intra-chain transactions         │
│     • Bridge edges: cross-chain bridge operations    │
│     • Node features: tx amount, gas, degree, age     │
│     • Edge features: value, timestamp, bridge type   │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  2. Heterogeneous Message Passing                    │
│     • Type-specific transformation: W_native, W_bridge│
│     • Graph attention over typed neighborhoods       │
│     • Bridge-aware aggregation (separate W for       │
│       cross-chain vs intra-chain neighbors)          │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  3. Temporal Attention Module                        │
│     • Graph snapshots at time windows [t-k, ..., t] │
│     • Multi-head attention over temporal embeddings  │
│     • Captures fraud pattern evolution               │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  4. Classification Head                              │
│     • Node-level: fraud/normal binary classification │
│     • Attention weights → explainability             │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Design Choice | Motivation |
|--------------|-----------|
| Heterogeneous edge types | Bridge edges have fundamentally different semantics than native transfers — mixing them loses information |
| Type-specific W matrices | Cross-chain neighbors should be weighted differently from intra-chain neighbors |
| Temporal snapshots | Fraud networks evolve; static graphs miss time-dependent laundering patterns |
| Attention-based explainability | Regulatory compliance requires reasoning chains, not just scores |

## Experiments

### Core Experiment: Ablation Study (Primary Contribution)

The central scientific question: **Does cross-chain information help?**

| Model Variant | Description | Tests |
|--------------|-------------|-------|
| **M1**: Single-chain GNN | GNN on ETH-only data | Baseline |
| **M2**: Single-chain GNN + bridge features | M1 + bridge tx count/volume as node features | RQ2 (feature-level) |
| **M3**: Multi-chain GNN (homogeneous) | Unified graph, all edges treated equally | RQ1 |
| **M4**: Multi-chain GNN (heterogeneous) | Separate W for native vs bridge edges | RQ1 |
| **M5**: M4 + temporal attention | Full TH-GNN | RQ2, RQ3 |

This ablation isolates the contribution of each component:
- **M1→M2**: Does bridge metadata alone help? (feature augmentation)
- **M1→M3**: Does multi-chain graph structure help? (topology)
- **M3→M4**: Does heterogeneous edge typing matter? (edge semantics)
- **M4→M5**: Does temporal modeling add value? (dynamics)

### Baseline Comparison

| Method | Type | Paper |
|--------|------|-------|
| Logistic Regression | Traditional ML | — |
| Random Forest | Traditional ML | — |
| XGBoost | Gradient Boosting | Chen & Guestrin, 2016 |
| Isolation Forest | Anomaly Detection | Liu et al., 2008 |
| GCN | GNN | Kipf & Welling, 2017 |
| GraphSAGE | GNN | Hamilton et al., 2017 |
| GAT | GNN (attention) | Veličković et al., 2018 |
| EvolveGCN | Temporal GNN | Pareja et al., AAAI 2020 |
| GMM-CCT | Cross-chain + GCN | BSCI 2024 |
| **TH-GNN (ours)** | Temporal heterogeneous GNN | This work |

### Case Study: Ronin Bridge Hack ($625M)

Using the Nomad/Ronin attack data from XChainWatcher, we analyze:
- Whether TH-GNN assigns high risk scores to known attack addresses
- Which attention heads activate on bridge edges vs. native edges
- How early in the attack timeline the model detects anomalous patterns

## Datasets

| Dataset | Source | Scale | Labels | Usage |
|---------|--------|-------|--------|-------|
| [XChainDataGen](https://zenodo.org/records/15341722) | Zenodo (2025) | 11.28M cross-chain txs, 8 bridges, 11 chains | ❌ No fraud labels | Graph structure |
| [BridgeGuard](https://arxiv.org/abs/2410.14493) | ACM WWW 2025 | 203 attack + 40K normal txs | ✅ Attack/normal | Cross-chain labels |
| [XChainWatcher](https://github.com/AndreAugusto11/XChainWatcher) | GitHub (2024) | 81K+ cross-chain events | ✅ Attack events | Case study |
| [Elliptic2](https://arxiv.org/abs/2404.19109) | arXiv (2024) | 50M nodes, 200M edges, 122K subgraphs | ✅ Licit/illicit | Single-chain baseline |
| [ETH Fraud Detection](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset) | Kaggle | 9.8K addresses | ✅ Fraud/normal | Single-chain baseline |
| [Nomad Hack Data](https://github.com/nomad-xyz/hack-data) | GitHub (2022) | Raw attack txs + address labels | ✅ White hat/attacker | Case study |

### Data Strategy for Label Scarcity

Cross-chain fraud labels are scarce (~200 from BridgeGuard). We address this via:
- **Semi-supervised learning**: Use labeled single-chain data (Elliptic2) + unlabeled cross-chain data
- **Cross-chain label propagation**: Propagate labels from labeled chains through bridge edges
- **Contrastive pre-training**: Self-supervised pre-training on unlabeled cross-chain graph, then fine-tune on labeled data

## Project Structure

```
chainguard/
├── data/
│   ├── raw/                    # Downloaded datasets
│   ├── processed/              # Graph-format data (PyG)
│   └── splits/                 # Train/val/test splits
├── src/
│   ├── data/                   # Data loading & graph construction
│   │   ├── graph_builder.py    # Multi-chain graph construction
│   │   ├── bridge_parser.py    # Cross-chain bridge tx parsing
│   │   └── feature_extract.py  # Node/edge feature extraction
│   ├── models/
│   │   ├── baselines/          # LR, RF, XGBoost, IF, GCN, GAT, GraphSAGE
│   │   ├── evolve_gcn.py       # EvolveGCN baseline
│   │   ├── th_gnn.py           # TH-GNN (proposed model)
│   │   └── modules/
│   │       ├── hetero_conv.py  # Heterogeneous message passing
│   │       ├── temporal_attn.py # Temporal attention module
│   │       └── classifier.py   # Classification head
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   ├── semi_supervised.py  # Semi-supervised methods
│   │   └── contrastive.py      # Contrastive pre-training
│   └── evaluation/
│       ├── metrics.py          # AUC, F1, Precision, Recall
│       ├── ablation.py         # Ablation experiment runner
│       └── case_study.py       # Ronin/Nomad case analysis
├── experiments/
│   ├── configs/                # Experiment configurations
│   └── scripts/                # Run scripts for each experiment
├── notebooks/                  # Analysis & visualization
├── paper/                      # LaTeX write-up
└── tests/                      # Unit tests
```

## Tech Stack

| Category | Tools |
|----------|-------|
| **Core ML** | PyTorch, PyTorch Geometric, scikit-learn, XGBoost |
| **Graph** | NetworkX, PyG HeteroData |
| **Experiment Tracking** | MLflow, Weights & Biases |
| **Data** | Pandas, NumPy, Web3.py |
| **Visualization** | Matplotlib, Seaborn, t-SNE/UMAP |
| **Reproducibility** | DVC, fixed seeds, config files |

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| AUC-ROC | Overall discrimination ability |
| F1 Score | Balanced precision-recall |
| Precision@k | Top-k ranking quality |
| Recall | Fraud coverage |
| Ablation Δ | Per-component contribution |

## Related Work

| Category | Key Papers | Limitation Addressed |
|----------|-----------|---------------------|
| Single-chain GNN | GCN, GAT, GraphSAGE on Elliptic | Single-chain only |
| Temporal GNN | EvolveGCN (AAAI 2020), MDST-GNN (2025) | Single-chain only |
| Cross-chain detection | GMM-CCT (BSCI 2024), BridgeGuard (WWW 2025) | No temporal modeling; low precision |
| Cross-chain tracing | ABCTracer (2025), XChainWatcher (2024) | Rule-based, not learned |

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Data acquisition & graph construction | Weeks 1-4 | Multi-chain PyG dataset |
| Baseline implementation & evaluation | Weeks 5-8 | Baseline results table |
| TH-GNN implementation | Weeks 9-12 | Core model code |
| Ablation experiments & analysis | Weeks 13-16 | Ablation study results |
| Case study & write-up | Weeks 17-20 | Technical report (arXiv) |

## Documentation

- [English Research Proposal (DOCX)](docs/ChainGuard-Research-Proposal-EN.docx)
- [Chinese Research Proposal (DOCX)](docs/ChainGuard-Research-Proposal-CN.docx)
- [English Research Proposal (Markdown)](docs/ChainGuard-Research-Proposal-EN.md)
- [Chinese Research Proposal (Markdown)](docs/ChainGuard-Research-Proposal-CN.md)

## Keywords

`cross-chain fraud detection` · `heterogeneous graph neural networks` · `temporal graph learning` · `cryptocurrency AML` · `explainable AI` · `blockchain analytics` · `graph attention networks` · `semi-supervised learning`

## License

MIT License

## Citation

```bibtex
@misc{chainguard2026,
  title={ChainGuard: Temporal Heterogeneous Graph Neural Networks for Cross-Chain Cryptocurrency Fraud Detection},
  author={},
  year={2026},
  url={https://github.com/youseihuayu-wonderful/ChainGuard-Crypto-Financial-Fraud-Detection-using-Temporal-Multi-Chain-Graph-Neural-Networks}
}
```
