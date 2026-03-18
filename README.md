# ChainGuard

### Crypto Financial Fraud Detection using Temporal Multi-Chain Graph Neural Networks

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**ChainGuard** is a research-driven fraud detection system designed to combat the growing threat of cross-chain cryptocurrency financial crimes. As criminals increasingly exploit cross-chain bridges to launder funds across Ethereum, BSC, Polygon, and other networks — generating over $11 billion in suspicious cross-chain transfers in 2025 alone — existing single-chain detection methods fail to trace fragmented transaction trails.

ChainGuard addresses this gap by constructing a **unified multi-chain transaction graph** where addresses are nodes, transactions are edges, and cross-chain bridge operations form special inter-chain links. A novel **Temporal Multi-Chain Graph Neural Network (TM-GNN)** learns both the structural patterns of fraud networks and their temporal evolution, while built-in attention mechanisms provide inherent explainability — critical for regulatory compliance (MiCA, Travel Rule, FinCEN).

The system delivers real-time risk scoring (P99 < 100ms), automated SAR generation, and an interactive investigation dashboard, bridging the gap between academic research and production-grade AML compliance.

## Key Features

- **Cross-chain unified graph** — ETH, BSC, Polygon transactions in one graph with bridge-aware edges
- **TM-GNN model** — temporal + structural + feature fusion via dual attention mechanism
- **Explainable by design** — attention-based reasoning chains, not post-hoc SHAP
- **Real-time scoring** — P99 < 100ms risk assessment API
- **Compliance ready** — automated SAR generation, OFAC screening, audit trails
- **Four-dimensional features** — transaction, wallet, graph structural, and temporal features

## Architecture

```
=================================================================
|                    Application Layer                           |
|   Risk Dashboard | Analyst Workbench | REST API | Reports     |
=================================================================
|                Decision & Compliance Layer                     |
|   Rule Engine | SAR Generator | Case Management | Alerts      |
=================================================================
|                      Model Layer                               |
|   Anomaly Detection | Risk Scoring | TM-GNN | Behavior        |
=================================================================
|                  Data Processing Layer                         |
|   Stream (Kafka+Flink) | Batch (Spark) | Feature Engineering  |
=================================================================
|                   Data Ingestion Layer                         |
|   On-chain Collectors | Exchange APIs | Compliance Data        |
=================================================================
```

## Core Innovation: TM-GNN

The **Temporal Multi-Chain Graph Neural Network** is the core algorithmic contribution:

1. **Graph Attention Layer** — learns which counterparty addresses matter most for fraud prediction
2. **Temporal Attention Layer** — learns which time windows are most indicative of fraud
3. **Cross-Dimension Fusion** — adaptively fuses graph structure, temporal dynamics, and node features
4. **Bridge-Aware Message Passing** — special message passing across cross-chain bridge edges

**Key advantages over existing methods:**
- Cross-chain awareness (vs single-chain GraphSAGE/GAT)
- Temporal evolution modeling (vs static graph snapshots)
- Inherent dual-layer explainability (vs post-hoc SHAP)
- Semi-supervised cross-chain contrastive learning

## Project Structure

```
chainguard/
├── config/                     # Configuration management
│   ├── experiments/            # Experiment configs
│   └── rules/                  # Risk control rules
├── data/                       # Data management
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Processed data
│   └── splits/                 # Train/val/test splits
├── src/
│   ├── data/                   # Data collection & preprocessing
│   ├── features/               # Four-dimensional feature engineering
│   ├── models/
│   │   ├── baselines/          # LR, RF, XGBoost, IF, Autoencoder
│   │   ├── gnn/                # GCN, GraphSAGE, GAT
│   │   ├── tfgnn/              # TM-GNN (proposed)
│   │   ├── ensemble/           # Multi-model fusion
│   │   └── explainability/     # SHAP + attention visualization
│   ├── engine/                 # Decision engine (rules, scoring, alerts)
│   ├── compliance/             # SAR, sanctions screening, case mgmt
│   └── api/                    # REST API service
├── experiments/                # Experiment scripts & analysis
├── notebooks/                  # Jupyter notebooks (EDA, experiments)
├── dashboard/                  # React frontend
├── paper/                      # LaTeX paper & defense slides
├── tests/                      # Unit, integration, e2e tests
├── infrastructure/             # Docker, K8s, monitoring
└── docs/                       # Documentation & proposals
```

## Datasets

| Dataset | Source | Size | Usage |
|---------|--------|------|-------|
| [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) | Kaggle | 203K BTC transactions | Primary evaluation |
| [Ethereum Fraud](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset) | Kaggle | 9.8K ETH addresses | Cross-chain validation |
| [Bitcoin Trust](http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) | Stanford SNAP | 24K trust edges | Graph analysis |
| Self-collected | Etherscan/BSCScan/PolygonScan | 1M+ transactions | Cross-chain experiments |

## Tech Stack

**ML & Data Science:** PyTorch, PyTorch Geometric, scikit-learn, XGBoost, SHAP, Optuna, NetworkX

**Data Processing:** Apache Kafka, Apache Flink, Apache Spark, Apache Airflow, Feast

**Blockchain:** Web3.py, Etherscan API, BSCScan API, PolygonScan API, Alchemy

**Backend:** FastAPI, PostgreSQL, Neo4j, TimescaleDB, Redis

**Frontend:** React, TypeScript, D3.js, ECharts

**MLOps:** MLflow, DVC, Docker, Kubernetes, GitHub Actions

**Monitoring:** Prometheus, Grafana, ELK Stack

## Experiments

| Experiment | Purpose | Validates |
|-----------|---------|-----------|
| E1: Main Comparison | TM-GNN vs 9 baselines on 3 datasets | H2 |
| E2: Feature Ablation | Four-dimension feature contributions | H1 |
| E3: Architecture Ablation | TM-GNN component contributions | H2 |
| E4: Model Fusion | Multi-model ensemble benefit | H4 |
| E5: Explainability | Explanation quality & performance impact | H3, H5 |
| E6: Efficiency | Real-time inference feasibility | H6 |
| E7: Robustness | Adversarial stability | Extra |
| E8: Case Study | Qualitative fraud case analysis | H3 |

## Target Performance

| Metric | Target |
|--------|--------|
| AUC-ROC | > 0.94 |
| F1 Score | > 0.87 |
| Precision | > 0.80 |
| Recall | > 0.85 |
| Inference Latency (P99) | < 100ms |
| Throughput | > 10K TPS |

## Documentation

- [English Project Proposal (DOCX)](docs/ChainGuard-Project-Proposal-EN.docx)
- [Chinese Project Proposal (DOCX)](docs/ChainGuard-Project-Proposal-CN.docx)
- [English Project Proposal (Markdown)](docs/ChainGuard-Project-Proposal-EN.md)
- [Chinese Project Proposal (Markdown)](docs/ChainGuard-Project-Proposal-CN.md)

## Keywords

`cross-chain fraud detection` · `graph neural networks` · `temporal graph learning` · `cryptocurrency AML` · `explainable AI` · `risk scoring` · `anti-money laundering` · `blockchain analytics` · `DeFi security` · `financial compliance`

## License

MIT License

## Citation

If you use ChainGuard in your research, please cite:

```bibtex
@misc{chainguard2026,
  title={ChainGuard: Crypto Financial Fraud Detection using Temporal Multi-Chain Graph Neural Networks},
  author={},
  year={2026},
  url={https://github.com/youseihuayu-wonderful/ChainGuard-Crypto-Financial-Fraud-Detection-using-Temporal-Multi-Chain-Graph-Neural-Networks}
}
```
