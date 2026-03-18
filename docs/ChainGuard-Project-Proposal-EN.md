# ChainGuard: Crypto Financial Fraud Detection using Temporal Multi-Chain Graph Neural Networks

# Complete Project Proposal

---

GitHub: https://github.com/youseihuayu-wonderful/ChainGuard-Crypto-Financial-Fraud-Detection-using-Temporal-Multi-Chain-Graph-Neural-Networks

Author: PhD Candidate, Expected Graduation 2026

Date: March 18, 2026

---

## Table of Contents

1. Project Overview & Research Motivation
2. Research Methodology (PhD Framework)
3. Research Questions, Hypotheses & Contributions
4. Unique Selling Points & Competitive Analysis
5. System Technical Architecture
6. Core Innovation: TM-GNN Model
7. Four-Dimensional Feature Engineering System
8. Experiment Design Framework
9. Data Strategy
10. Project Management Framework (20-Week Timeline)
11. Five-Engineer Division of Labor
12. Complete Technology Stack
13. Key Performance Indicators (KPIs)
14. Risk Management
15. References & Datasets

---

## 1. Project Overview & Research Motivation

### 1.1 Background

The cryptocurrency market has experienced explosive growth, with total market capitalization exceeding $3 trillion. However, this growth has been accompanied by a surge in financial crimes. In 2025 alone, actors sent approximately USD 35 billion in cryptocurrency to fraud schemes. Cross-chain suspicious transfers exceeded $11 billion, with final estimates potentially surpassing $75 billion once complex cross-chain investigations mature.

Traditional anti-money laundering (AML) methods, primarily rule-based systems, face significant limitations when dealing with the decentralized, pseudonymous, and cross-chain nature of cryptocurrency transactions. Existing machine learning approaches operate on single-chain data, failing to trace fragmented transaction trails that span multiple blockchains.

### 1.2 Problem Statement

ChainGuard addresses the following core problems:

- P1 (Cross-Chain Blind Spot): Criminals exploit cross-chain bridges to launder funds across Ethereum, BSC, Polygon, and other networks. Assets converted through bridges become new tokens with no on-chain reference to the original asset, breaking single-chain detection methods.

- P2 (Temporal Dynamics): Fraud patterns evolve over time. Static graph analysis misses the temporal evolution of criminal networks, such as gradual fund layering and burst transaction patterns.

- P3 (Explainability Gap): Regulators (MiCA, FinCEN, Travel Rule) require auditable reasoning chains for every flagged transaction. Existing GNN models are black boxes that cannot satisfy compliance requirements.

- P4 (Real-Time Requirements): AML compliance demands real-time risk assessment. Current academic models prioritize accuracy over inference speed, making them impractical for production deployment.

### 1.3 Project Value

- Academic Value: Novel contribution in cross-chain graph learning for fraud detection, an area with minimal existing academic research
- Industrial Value: Production-grade risk scoring platform for exchanges and compliance providers
- Social Value: Helping regulators combat cryptocurrency financial crime and protect investors

---

## 2. Research Methodology (PhD Framework)

### 2.1 Design Science Research (DSR) Paradigm

This project adopts the Design Science Research paradigm, producing both a research artifact (TM-GNN model + ChainGuard system) and theoretical contributions (cross-chain graph learning methodology).

DSR Seven-Step Application:

| Step | DSR Principle | Application in ChainGuard |
|------|--------------|--------------------------|
| 1 | Problem Identification | Cross-chain fraud detection blind spot in existing methods |
| 2 | Objectives of Solution | Real-time, explainable, high-accuracy multi-chain fraud detection |
| 3 | Design & Development | TM-GNN model + unified multi-chain graph + end-to-end system |
| 4 | Demonstration | Experiments on Elliptic, ETH Fraud, and self-collected multi-chain data |
| 5 | Evaluation | Quantitative (AUC/F1) + qualitative (case study) + efficiency (latency) |
| 6 | Communication | Paper submission to KDD/WWW/Financial Crypto + open-source code |
| 7 | Iteration | Model refinement based on experimental feedback |

### 2.2 Hypothesis-Driven Development

Every model module starts from a hypothesis, validated through rigorous experimentation:

- H1: Multi-dimensional feature representation (transaction + wallet + graph + temporal) significantly outperforms single-dimension features
- H2: Temporal Multi-Chain GNN outperforms static GNN by >= 5% AUC on cross-chain fraud detection
- H3: Attention mechanisms automatically learn meaningful fraud patterns, providing node-level explainability
- H4: Multi-model fusion (GNN + anomaly detection + rules) improves F1 by >= 10% over the best single model
- H5: SHAP explainability constraints do not significantly degrade model performance (AUC drop < 2%)
- H6: The system can complete end-to-end risk scoring for a single transaction within 100ms

### 2.3 Experimental Design Principles

- Baseline Comparison: Every model compared against 8+ baselines
- Ablation Studies: Validate each component's contribution independently
- Statistical Significance: 5-fold cross-validation + paired t-test (p < 0.05)
- Reproducibility: Fixed random seeds, all hyperparameters logged in MLflow, open-source code

---

## 3. Research Questions, Hypotheses & Contributions

### 3.1 Research Questions

- RQ (Main): How can we build an efficient, explainable cross-chain cryptocurrency fraud detection and risk scoring system?
  - RQ1 (Representation): How to effectively represent multi-chain transaction features across transaction, wallet, graph, and temporal dimensions?
  - RQ2 (Detection): Does a temporal multi-chain GNN outperform single-chain methods and static GNNs for cross-chain fraud detection?
  - RQ3 (Risk Quantification): How to fuse multi-model outputs into a unified, explainable risk score?
  - RQ4 (Practicality): How to maintain high accuracy while satisfying real-time and compliance requirements?

### 3.2 Academic Contributions

- Contribution 1 (Methodology): Propose TM-GNN (Temporal Multi-Chain Graph Neural Network), a novel end-to-end framework that fuses temporal, cross-chain structural, and transaction features for fraud detection.

- Contribution 2 (Cross-Chain Graph): Design a unified multi-chain transaction graph where cross-chain bridge operations form heterogeneous inter-chain edges, enabling cross-chain fund flow tracing.

- Contribution 3 (Feature Engineering): Design a four-dimensional feature system specifically for cryptocurrency fraud detection: transaction features, wallet features, graph structural features, and temporal features.

- Contribution 4 (System): Build an end-to-end explainable risk scoring platform with real-time inference, automated SAR generation, and regulatory compliance support.

- Contribution 5 (Empirical): Extensive experiments on 3 datasets with 10 baseline methods, demonstrating state-of-the-art performance.

---

## 4. Unique Selling Points & Competitive Analysis

### 4.1 Why ChainGuard is Unique

ChainGuard's uniqueness comes from addressing gaps that existing academic and industrial solutions have not adequately covered:

UNIQUE POINT 1: CROSS-CHAIN FRAUD DETECTION (Academic Blue Ocean)

Current State: Almost ALL academic papers on crypto fraud detection operate on a SINGLE chain (BTC or ETH). Cross-chain fund tracing is done only by commercial companies (Chainalysis, TRM Labs, Elliptic) with proprietary methods.

ChainGuard's Innovation: Constructs a unified multi-chain transaction graph where:
- Nodes = addresses across ETH, BSC, Polygon
- Intra-chain edges = normal transactions
- Inter-chain edges = bridge operations (novel heterogeneous edge type)
- TM-GNN performs bridge-aware message passing across chains

Why This Matters: In 2025, cross-chain suspicious transfers exceeded $11 billion. Chain-hopping and layering significantly delay attribution. This is the biggest unsolved problem in crypto AML.

Research Gap Evidence: Searching for "cross-chain fraud detection GNN" in academic databases yields virtually no results. This represents a massive publication opportunity.

---

UNIQUE POINT 2: INHERENT EXPLAINABILITY (Not Post-hoc)

Current State: Most GNN papers apply SHAP or GNNExplainer as post-hoc explanation tools. These are "bolted-on" explanations that may not faithfully represent the model's actual reasoning.

ChainGuard's Innovation: TM-GNN has DUAL built-in attention mechanisms:
- Graph Attention: explains "which neighbors (counterparties) matter most"
- Temporal Attention: explains "which time windows are most suspicious"
- Together they produce a human-readable reasoning chain: "Address X is flagged because it received 85% of funds from known mixer Y (graph attention = 0.92) during an abnormal burst at 3AM UTC (temporal attention = 0.87)"

Why This Matters: In 2026, 90% of financial institutions are expected to use AI for AML. Regulators explicitly require auditable reasoning chains. "Black box" AI is not acceptable. ChainGuard's inherent explainability directly satisfies MiCA, FinCEN, and Travel Rule requirements.

---

UNIQUE POINT 3: DeFi-SPECIFIC FRAUD PATTERN DETECTION

Current State: ACM Computing Surveys 2025 explicitly states: "For DeFi-related threats such as rug pulls and flash-loan attacks, constructing representative labeled datasets remains technically challenging." Very few academic papers address DeFi fraud.

ChainGuard's Innovation:
- Designs DeFi-specific features (liquidity change rate, contract interaction patterns, flash loan indicators)
- Cross-chain bridge attack detection (exploiting bridge vulnerabilities)
- Potential to contribute the first DeFi fraud labeled dataset (a publishable contribution on its own)

---

UNIQUE POINT 4: END-TO-END SYSTEM (Academic Research + Production System)

Current State: Academic papers publish models only. No system, no API, no dashboard, no compliance module. Industrial products are proprietary with no published methodology.

ChainGuard's Innovation: Bridges the gap between academia and industry:
- Research: Novel TM-GNN model with rigorous experiments
- System: Real-time risk scoring API (P99 < 100ms)
- Compliance: Automated SAR generation, OFAC screening
- Visualization: Interactive investigation dashboard with graph exploration

This combination is extremely compelling for a PhD defense.

---

UNIQUE POINT 5: SEMI-SUPERVISED CROSS-CHAIN CONTRASTIVE LEARNING

Current State: The Elliptic dataset has 77% unlabeled transactions. Most methods only use labeled data, wasting the majority of available information.

ChainGuard's Innovation: Uses cross-chain contrastive learning to leverage:
- Unlabeled transaction data (majority of blockchain data)
- Cross-chain correspondence (same entity on different chains)
- This improves detection performance especially for new/unseen fraud patterns

### 4.2 Competitive Landscape

| Competitor | Valuation | Focus | ChainGuard's Advantage |
|-----------|-----------|-------|----------------------|
| Chainalysis | $8.6B | On-chain analytics + law enforcement | Proprietary methods, no academic transparency |
| Elliptic | $800M+ | Transaction screening + compliance | DeFi coverage still early |
| TRM Labs | $1.2B | Multi-chain intelligence | Limited real-time detection |
| Merkle Science | $200M+ | Asia-Pacific compliance | Regional focus, limited ML depth |
| Academic SOTA (MDST-GNN, ATGAT, etc.) | N/A | Single-chain GNN models | No cross-chain, no system, no compliance |

### 4.3 Comparison with Existing Academic Methods (2025-2026)

| Method | Year | Approach | Limitation | ChainGuard Advantage |
|--------|------|----------|-----------|---------------------|
| MDST-GNN | 2025 | Multi-distance spatial-temporal GNN | Single dataset (Elliptic), no cross-chain | Multi-chain unified graph |
| ATGAT | 2025 | Temporal-aware graph attention | AUC 0.913, limited explainability | Inherent dual-attention explainability |
| CoSemiGNN | 2025 | Semi-supervised dynamic GNN | No DeFi scenarios | DeFi-specific features + cross-chain |
| ChronoWave-GNN | 2026 | Wavelet + temporal GNN | Frequency domain, poor explainability | Human-readable attention explanations |
| EvolveGCN | 2020 | RNN-based evolving GCN | No cross-chain capability | Bridge-aware message passing |

---

## 5. System Technical Architecture

### 5.1 Five-Layer Architecture

```
=================================================================
|                    Layer 5: Application                        |
|   Risk Dashboard | Analyst Workbench | REST API | Reports     |
=================================================================
|                Layer 4: Decision & Compliance                  |
|   Rule Engine | SAR Generator | Case Management | Alerts      |
=================================================================
|                   Layer 3: Model Layer                         |
|   Anomaly Detection | Risk Scoring | TM-GNN | Behavior        |
=================================================================
|                Layer 2: Data Processing                        |
|   Stream Processing (Kafka+Flink) | Batch (Spark) | Features  |
=================================================================
|                 Layer 1: Data Ingestion                        |
|   On-chain Collectors | Exchange APIs | Compliance Data        |
=================================================================
```

### 5.2 Data Ingestion Layer
- On-chain data collectors: Web3.py / Etherscan API / Alchemy for ETH; BSCScan for BSC; PolygonScan for Polygon
- Exchange data: Binance / Coinbase API for CEX transaction records
- Enrichment: IP geolocation, OFAC sanctions list, known blacklisted addresses
- Bridge data: Cross-chain bridge transaction monitoring (Wormhole, Multichain, Stargate)

### 5.3 Data Processing & Feature Engineering Layer
- Real-time stream processing: Kafka + Flink for live transaction feature computation
- Batch processing: Spark for historical data analysis and model training data preparation
- Feature Store: Feast for unified online/offline feature management
- Graph Builder: Constructs unified multi-chain transaction graph with temporal snapshots

### 5.4 Model Layer (Core)
- Baselines: LR, RF, XGBoost, Isolation Forest, Autoencoder
- Static GNNs: GCN, GraphSAGE, GAT
- TM-GNN (Proposed): Temporal Multi-Chain Graph Neural Network
- Ensemble: Multi-model fusion scoring engine

### 5.5 Decision & Compliance Layer
- Rule Engine: Configurable if-then rules (large transactions, high-frequency transfers, blacklist matching)
- Risk Scoring: Multi-model weighted fusion -> unified risk score (0-100)
- Alert Tiers: Low (0-25) / Medium (25-50) / High (50-75) / Critical (75-100)
- SAR Generator: Automated Suspicious Activity Report generation
- Case Management: Analyst review workflow, case tracking, audit logs

### 5.6 Application Layer
- Risk Dashboard: Real-time risk posture awareness, key metrics visualization
- Analyst Workbench: Case details, transaction graph exploration, decision support
- REST API: External risk scoring query service
- Reporting: Periodic regulatory and statistical reports

---

## 6. Core Innovation: TM-GNN (Temporal Multi-Chain Graph Neural Network)

### 6.1 Architecture Overview

The TM-GNN is the core algorithmic contribution of this thesis. It consists of three key modules:

Module A: Graph Attention Layer
- Performs neighborhood aggregation with attention weights
- Learns which counterparty addresses are most relevant for fraud prediction
- Attention weights serve as explanations: "which neighbors matter"

Module B: Temporal Attention Layer
- Takes graph snapshots from multiple time windows (1h, 6h, 24h, 7d)
- Learns which time periods are most indicative of fraud
- Attention weights explain: "which time window is most suspicious"

Module C: Cross-Dimension Fusion Module
- Fuses graph structural information, temporal dynamics, and node features
- Uses adaptive attention to weight different information sources
- Outputs final node classification (fraud probability) and risk score

### 6.2 Innovation Points vs Existing Methods

1. Cross-Chain Awareness: Unlike GraphSAGE/GAT that only see single-chain graphs, TM-GNN models cross-chain bridge operations as heterogeneous edges with bridge-aware message passing.

2. Temporal Modeling: Unlike static GNNs, TM-GNN captures the temporal evolution of fraud networks through sequential attention over graph snapshots.

3. Dual-Layer Explainability: Graph Attention explains "which neighbors are important"; Temporal Attention explains "which time window is important". Together they produce compliance-ready reasoning chains.

4. Cross-Chain Contrastive Learning: Leverages unlabeled data through cross-chain correspondence for semi-supervised learning.

### 6.3 Mathematical Formulation

Graph Attention (simplified):
  alpha_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j || e_ij]))
  where e_ij encodes edge type (intra-chain transaction vs cross-chain bridge)

Temporal Attention (simplified):
  beta_t = softmax(v^T tanh(W_t * z_t + b))
  where z_t is the graph-level representation at time step t

Fusion:
  h_final = FFN(concat(h_graph_attn, h_temporal_attn, x_node_features))
  y = sigmoid(W_out * h_final)

---

## 7. Four-Dimensional Feature Engineering System

### 7.1 Transaction Features

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| tx_amount_mean | mean(amount) over window | Average transaction amount; abnormal values may indicate fraud |
| tx_amount_std | std(amount) over window | Amount volatility; sudden changes may be suspicious |
| tx_frequency | count(tx) / time_window | Transaction frequency; high frequency may indicate automated laundering |
| tx_counterparty_entropy | entropy(counterparty_dist) | Counterparty diversity; low entropy may indicate circular transactions |
| tx_gas_anomaly | (gas_price - median) / MAD | Gas price anomaly score |
| tx_value_percentile | percentile_rank(amount) | Transaction amount percentile in network |

### 7.2 Wallet Features

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| wallet_age | now - first_tx_time | Wallet age; new wallets with large transactions are more suspicious |
| wallet_balance_velocity | d(balance) / dt | Balance change rate |
| wallet_in_out_ratio | sum(in) / sum(out) | Fund inflow/outflow ratio |
| wallet_hhi | Herfindahl Index of sources | Fund source concentration; too high may indicate directed transfers |
| wallet_active_days | count(distinct active days) | Active days count |
| wallet_dormancy_ratio | inactive_days / total_days | Dormancy ratio; sudden reactivation is suspicious |

### 7.3 Graph Structural Features

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| node_in_degree | count(incoming edges) | Number of received transactions |
| node_out_degree | count(outgoing edges) | Number of sent transactions |
| node_pagerank | PageRank algorithm | Node importance/centrality |
| node_clustering_coeff | 2*triangles / (deg*(deg-1)) | Clustering coefficient; high values may indicate fraud rings |
| node_fraud_neighbor_ratio | count(fraud_neighbors) / degree | Association with known fraud nodes |
| node_bridge_usage_count | count(bridge transactions) | Cross-chain bridge usage frequency |

### 7.4 Temporal Features

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| temp_burst_score | Kleinberg burst detection | Transaction burstiness detection |
| temp_periodicity | autocorrelation analysis | Periodic patterns (bot trading signature) |
| temp_trend_slope | linear regression slope | Amount trend (increasing may be progressive fraud) |
| temp_window_stats | rolling window aggregations | Multi-window statistics (1h/6h/24h/7d) |
| temp_time_of_day | hour_of_day encoding | Time-of-day pattern |
| temp_cross_chain_delay | time between bridge operations | Cross-chain transfer timing patterns |

---

## 8. Experiment Design Framework

### 8.1 Experiment Matrix

| ID | Experiment | Purpose | Methods Compared | Metrics | Dataset | Validates |
|----|-----------|---------|-----------------|---------|---------|-----------|
| E1 | Main Comparison | Validate TM-GNN overall superiority | LR, RF, XGBoost, IF, GCN, GraphSAGE, GAT, EvolveGCN, TM-GNN | AUC, F1, Precision, Recall, AP | All 3 datasets | H2 |
| E2 | Feature Ablation | Validate four-dimension feature contributions | TM-GNN (full) vs removing each dimension | Delta AUC | Elliptic | H1 |
| E3 | Architecture Ablation | Validate TM-GNN component contributions | No Temporal Attn / No Graph Attn / No Fusion / No Bridge Edges | AUC, F1 | Elliptic | H2 |
| E4 | Model Fusion | Validate multi-model ensemble benefit | Single models vs pairwise vs full fusion | Delta F1 | All 3 datasets | H4 |
| E5 | Explainability | Validate explanation quality and performance impact | SHAP + Attention visualization + Fidelity evaluation | Fidelity, Sparsity, AUC diff | Elliptic | H3, H5 |
| E6 | Efficiency | Validate real-time inference feasibility | All models inference latency and throughput | P50/P99 latency, TPS | Self-collected | H6 |
| E7 | Robustness | Validate adversarial stability | Add noise edges/nodes, feature perturbation | AUC degradation rate | Elliptic | Extra |
| E8 | Case Study | Qualitative analysis of detection capability | Select typical fraud cases, analyze attention weights | Qualitative | Elliptic + Self-collected | H3 |

### 8.2 Baseline Methods

| Category | Method | Reference | Why Selected |
|----------|--------|-----------|-------------|
| Traditional ML | Logistic Regression | - | Most basic linear baseline |
| Traditional ML | Random Forest | - | Classic nonlinear baseline |
| Traditional ML | XGBoost | Chen & Guestrin, 2016 | Strong industry baseline |
| Anomaly Detection | Isolation Forest | Liu et al., 2008 | Classic unsupervised anomaly detection |
| Anomaly Detection | Autoencoder | - | Deep anomaly detection baseline |
| GNN | GCN | Kipf & Welling, 2017 | Most basic GNN |
| GNN | GraphSAGE | Hamilton et al., 2017 | Inductive GNN, industry standard |
| GNN | GAT | Velickovic et al., 2018 | Attention-based GNN baseline |
| Temporal GNN | EvolveGCN | Pareja et al., 2020 | Temporal GNN baseline |
| Ours | TM-GNN | This paper | Temporal Multi-Chain GNN |

---

## 9. Data Strategy

### 9.1 Dataset Plan

| Dataset | Source | Size | Labels | Usage | Priority |
|---------|--------|------|--------|-------|----------|
| Elliptic | Kaggle | 203,769 BTC transactions, 166 features | 4,545 illicit / 42,019 licit / 157,205 unlabeled | Primary dataset #1: core comparison and ablation experiments | P0 |
| Ethereum Fraud Detection | Kaggle | 9,841 ETH addresses | 2,179 fraud / 7,662 normal | Primary dataset #2: cross-chain generalization validation | P0 |
| Bitcoin Alpha + OTC | Stanford SNAP | 24,186 trust edges | Trust scores (-10 to +10) | Auxiliary: graph analysis and community detection | P1 |
| Self-collected Multi-Chain | Etherscan + BSCScan + PolygonScan | Target 1M+ transactions | Semi-labeled (blacklist matching) | Cross-chain experiments + large-scale performance test | P1 |
| ERC-20 Token Transfers | Google BigQuery | Hundreds of millions | Unlabeled | Large-scale graph construction + unsupervised experiments | P2 |
| OFAC SDN List | U.S. Treasury | ~12,000 entities | Sanctioned entities | Compliance module validation | P1 |

### 9.2 Data Processing Pipeline

Step 1: Collection -> Raw data from APIs and public datasets
Step 2: Preprocessing -> Deduplication, missing value handling, address normalization, timestamp alignment
Step 3: Graph Construction -> Nodes (addresses), edges (transactions), temporal snapshots (hourly/daily/weekly)
Step 4: Feature Computation -> Four-dimension features computed per node
Step 5: Data Splitting -> TEMPORAL split (not random!) to prevent data leakage: Train 60% / Val 20% / Test 20%

CRITICAL: Data must be split by TIME ORDER to prevent temporal data leakage.

---

## 10. Project Management Framework

### 10.1 Overall Timeline (20 Weeks = 5 Months)

Phase 1: Foundation (Weeks 1-3)
- Literature review, data collection, infrastructure setup
- Deliverables: Literature survey, data pipeline, development environment

Phase 2: Features & Baselines (Weeks 4-7)
- Four-dimension feature engineering, baseline experiments
- Deliverables: Feature pipeline, baseline results table

Phase 3: Core Model Development (Weeks 8-12) -- PROJECT CORE
- TM-GNN design, implementation, main experiments, ablation studies
- Deliverables: TM-GNN model, E1/E2/E3 experiment results

Phase 4: System & Advanced Experiments (Weeks 13-15)
- Decision engine, API, fusion experiments, explainability, robustness
- Deliverables: Risk scoring API, E4/E5/E6/E7 results, dashboard MVP

Phase 5: Paper & Defense (Weeks 16-20)
- Paper writing, defense preparation, system deployment
- Deliverables: Complete paper, defense slides, deployed system

### 10.2 Week-by-Week Plan

Week 1:
- Engineer A: GitHub repo + CI/CD + Docker environment setup
- Engineer B: Elliptic dataset download + EDA + data schema definition
- Engineer C (PhD): Systematic literature review (30+ papers) + SOTA method summary
- Engineer D: PostgreSQL + Neo4j + TimescaleDB + Redis deployment
- Engineer E: React project skeleton + UI prototype design

Week 2:
- Engineer A: ETH/BTC/BSC on-chain data collectors + Kafka setup
- Engineer B: Data cleaning pipeline + format standardization
- Engineer C (PhD): Research gap analysis + RQ/hypothesis definition + experiment design
- Engineer D: Database schema + migration + basic CRUD API
- Engineer E: Dashboard basic layout + routing + WebSocket communication

Week 3:
- Engineer A: Data pipeline integration (collection -> Kafka -> storage) + monitoring
- Engineer B: Elliptic graph construction + temporal snapshot splitting + data partitioning
- Engineer C (PhD): Baseline code preparation (LR/RF/XGBoost/IF) + initial Elliptic runs
- Engineer D: Neo4j graph data import + Cypher query API
- Engineer E: Real-time data display components + basic charts

Weeks 4-5:
- Engineer A: Feature Store (Feast) setup + online/offline feature management
- Engineer B: Transaction features + wallet features implementation + unit tests
- Engineer C (PhD): Graph features + temporal features implementation + feature importance analysis
- Engineer D: Feature API development + Redis caching layer
- Engineer E: Feature distribution visualization dashboard

Weeks 6-7:
- Engineer A: MLflow experiment tracking + model training pipeline (Airflow)
- Engineer B: XGBoost risk scoring model + SHAP analysis + hyperparameter tuning (Optuna)
- Engineer C (PhD): 6 baseline models complete experiments + cross-validation + statistical tests + results tables
- Engineer D: Model inference API framework (FastAPI) + Swagger docs
- Engineer E: Model comparison results visualization

Weeks 8-9:
- Engineer A: GPU training environment optimization + distributed training support
- Engineer B: GCN/GraphSAGE/GAT implementation + Elliptic training + comparison results
- Engineer C (PhD): TM-GNN architecture design + Temporal Attention + Graph Attention implementation
- Engineer D: Graph data API optimization + batch query interface
- Engineer E: Transaction graph visualization (D3.js force graph)

Week 10:
- Engineer A: Experiment results auto-collection + comparison table generation scripts
- Engineer B: EvolveGCN baseline + temporal GNN comparison experiments
- Engineer C (PhD): TM-GNN Fusion Module implementation + complete model training and tuning
- Engineer D: Rule engine DSL design + basic rule implementation
- Engineer E: Attention weight visualization component

Weeks 11-12:
- Engineer A: ETH Fraud dataset pipeline + cross-dataset experiment automation
- Engineer B: Feature ablation experiment (E2) execution + results compilation
- Engineer C (PhD): Main comparison E1 (3 datasets x 10 methods) + architecture ablation E3 + statistical significance tests
- Engineer D: Ensemble scoring engine + model output normalization + fusion API
- Engineer E: Experiment results dashboard (dynamic charts)

Weeks 13-14:
- Engineer A: Real-time scoring pipeline (Kafka -> features -> inference) + performance load testing
- Engineer B: Multi-model fusion experiment E4 + score calibration (Platt Scaling)
- Engineer C (PhD): Explainability experiment E5 (SHAP + Attention) + Case Study E8
- Engineer D: Alert system + SAR template + compliance API + case management
- Engineer E: Alert center UI + SAR report page + analyst workbench

Week 15:
- Engineer A: System performance optimization (ONNX export + caching) + efficiency experiment E6
- Engineer B: Robustness experiment E7 (adversarial samples + noise injection)
- Engineer C (PhD): Paper Introduction + Methodology first draft
- Engineer D: Security hardening + API authentication + audit logs
- Engineer E: Dashboard feature completion + responsive design

Weeks 16-17:
- Engineer A: K8s deployment + canary release + monitoring (Prometheus + Grafana)
- Engineer B: Supplementary experiments + final results tables + paper figures
- Engineer C (PhD): Paper Experiments + Results + Discussion chapters
- Engineer D: Runbook + API documentation final version
- Engineer E: Demo video recording + defense PPT visual materials

Weeks 18-19:
- Engineer A: System stability testing + production validation
- Engineer B: Paper review + data/experiment reproducibility verification
- Engineer C (PhD): Complete paper draft + advisor review + revisions
- Engineer D: End-to-end integration testing + bug fixes
- Engineer E: Defense PPT creation (40-50 slides) + demo environment setup

Week 20:
- Engineer A: Final system deployment + documentation archive
- Engineer B: Open-source code cleanup + README
- Engineer C (PhD): Defense rehearsal (3 times) + paper final version + submission preparation
- Engineer D: Code archive + license
- Engineer E: Defense day technical support + demo assurance

---

## 11. Five-Engineer Division of Labor

### Engineer A: Data Infrastructure Engineer / DevOps

Core Responsibilities:
- Data collection pipeline development and maintenance
- Kafka / Flink / Airflow infrastructure setup
- Feature Store setup and management
- CI/CD, Docker, Kubernetes deployment
- System monitoring and alerting

Key Skills: Python, Docker, Kubernetes, Kafka, Flink/Spark, Airflow, Web3.py

Phase Workload: P1 30% | P2 15% | P3 20% | P4 15% | P5 20%

### Engineer B: Data Scientist / Feature Engineer

Core Responsibilities:
- Feature engineering design and implementation (transaction + wallet features)
- XGBoost risk scoring model development
- Data processing pipelines (Flink/Spark jobs)
- Multi-model fusion scoring engine
- Data quality monitoring

Key Skills: Python, pandas, scikit-learn, XGBoost, Spark, statistics

Phase Workload: P1 20% | P2 35% | P3 20% | P4 15% | P5 10%

### Engineer C: ML Research Engineer (PhD - YOU - Project Lead)

Core Responsibilities:
- TM-GNN model design and implementation
- Anomaly detection models (Isolation Forest, Autoencoder, LSTM)
- Graph neural network models (GCN, GraphSAGE, GAT)
- Experiment design, comparison experiments, ablation studies
- Paper writing
- Model explainability analysis

Key Skills: PyTorch, PyTorch Geometric, deep learning theory, academic writing, experiment design

Phase Workload: P1 15% | P2 40% | P3 15% | P4 15% | P5 15%

PhD-Specific Tasks:
- Weeks 1-2: Systematic literature review, identify research gaps
- Weeks 6-12: Core experiments (validate hypotheses H1-H6)
- Week 15: Paper Introduction + Methodology draft
- Weeks 16-17: Paper Experiments + Results + Discussion
- Weeks 18-20: Complete paper + defense preparation

### Engineer D: Backend Engineer / System Architect

Core Responsibilities:
- REST API / gRPC service development
- Rule engine design and implementation
- Compliance module (SAR, sanctions screening, case management)
- Database management and optimization
- Security

Key Skills: Python (FastAPI), SQL, PostgreSQL, Neo4j, Redis, API design

Phase Workload: P1 20% | P2 15% | P3 25% | P4 25% | P5 15%

### Engineer E: Frontend Engineer / Visualization

Core Responsibilities:
- Risk dashboard development (React + TypeScript)
- Data visualization (graph visualization, trend charts, heatmaps)
- Analyst workbench UI
- Rule management / alert center / SAR report UI
- User experience optimization

Key Skills: React, TypeScript, D3.js, WebSocket, UI/UX design

Phase Workload: P1 15% | P2 15% | P3 25% | P4 25% | P5 20%

### Collaboration Matrix

| Task | Eng A | Eng B | Eng C | Eng D | Eng E |
|------|-------|-------|-------|-------|-------|
| Data Pipeline | Lead | Assist | - | - | - |
| Feature Engineering | Assist (Store) | Lead | Assist (graph/temporal) | - | - |
| TM-GNN Model | - | - | Lead | - | - |
| Baseline Models | - | Assist | Lead | - | - |
| Risk Scoring | - | Lead | Review | Assist (API) | - |
| Rule Engine | - | - | - | Lead | Assist (UI) |
| Compliance | Assist | Assist | - | Lead | Assist (UI) |
| Dashboard | - | - | - | Assist (API) | Lead |
| Deployment | Lead | Assist | - | Assist | - |
| Paper Writing | - | Assist | Lead | - | - |

---

## 12. Complete Technology Stack

### 12.1 Programming Languages

| Language | Purpose | Version |
|----------|---------|---------|
| Python | ML models, data processing, API, experiments | 3.11+ |
| TypeScript | Frontend dashboard | 5.0+ |
| SQL / Cypher | Data queries | - |
| LaTeX | Paper writing | - |

### 12.2 Machine Learning & Data Science

| Tool | Purpose | Priority |
|------|---------|----------|
| PyTorch 2.0+ | Deep learning (Autoencoder, LSTM, TM-GNN) | P0 |
| PyTorch Geometric (PyG) | Graph neural networks (GCN, GraphSAGE, GAT, TM-GNN) | P0 |
| scikit-learn | Traditional ML (LR, RF, IF, clustering, preprocessing) | P0 |
| XGBoost | Risk scoring model | P0 |
| SHAP | Model explainability | P0 |
| Optuna | Hyperparameter optimization | P1 |
| imbalanced-learn | Imbalanced data handling (SMOTE) | P1 |
| pandas / polars | Data analysis and feature computation | P0 |
| NumPy / SciPy | Numerical computation and statistical analysis | P0 |
| NetworkX | Graph analysis algorithms (PageRank, clustering coefficient) | P0 |
| matplotlib / seaborn | Paper figure generation | P0 |

### 12.3 MLOps & Model Management

| Tool | Purpose | Priority |
|------|---------|----------|
| MLflow | Experiment tracking + model registry + versioning | P0 |
| DVC | Data version management | P1 |
| Weights & Biases | Experiment visualization (alternative) | P2 |

### 12.4 Blockchain & Data Collection

| Tool | Purpose | Priority |
|------|---------|----------|
| Web3.py | Ethereum node interaction | P0 |
| Etherscan API | ETH transaction/address queries | P0 |
| BSCScan API | BSC transaction data | P0 |
| PolygonScan API | Polygon transaction data | P1 |
| Alchemy | Node service + enhanced API | P0 |
| CoinGecko API | Cryptocurrency price data | P1 |

### 12.5 Data Processing & Streaming

| Tool | Purpose | Priority |
|------|---------|----------|
| Apache Kafka | Message queue / event streaming | P0 |
| Apache Flink | Real-time stream processing | P1 |
| Apache Spark | Batch data processing | P0 |
| Apache Airflow | Workflow orchestration / scheduling | P0 |
| Feast | Feature Store (online/offline unified) | P1 |

### 12.6 Databases & Storage

| Database | Purpose | Priority |
|----------|---------|----------|
| PostgreSQL | Business data (users, cases, rules, alerts) | P0 |
| Neo4j | Address relationship graph | P0 |
| TimescaleDB | Transaction time-series data | P0 |
| Redis | Cache, real-time features, rate limiting | P0 |
| MinIO / S3 | Object storage (model files, raw data) | P1 |

### 12.7 Backend & API

| Tool | Purpose | Priority |
|------|---------|----------|
| FastAPI | REST API service | P0 |
| Pydantic | Data validation and serialization | P0 |
| SQLAlchemy + Alembic | ORM + database migration | P0 |
| Celery | Async task queue | P1 |
| gRPC + Protobuf | High-performance internal communication | P1 |

### 12.8 Frontend & Visualization

| Tool | Purpose | Priority |
|------|---------|----------|
| React 18+ | Frontend framework | P0 |
| TypeScript | Type safety | P0 |
| D3.js | Custom graph visualization | P0 |
| ECharts | Statistical charts | P1 |
| Ant Design | UI component library | P1 |

### 12.9 Infrastructure & DevOps

| Tool | Purpose | Priority |
|------|---------|----------|
| Docker + Docker Compose | Containerization / local orchestration | P0 |
| Kubernetes + Helm | Production orchestration | P1 |
| GitHub Actions | CI/CD pipeline | P0 |
| Prometheus + Grafana | Monitoring + alerting | P1 |
| ELK Stack | Log collection and analysis | P1 |

### 12.10 Testing & Code Quality

| Tool | Purpose | Priority |
|------|---------|----------|
| pytest + pytest-cov | Unit/integration tests + coverage | P0 |
| Locust | Load testing | P1 |
| ruff + mypy | Lint + type checking | P0 |
| pre-commit | Git hooks automation | P0 |

### 12.11 Academic Tools

| Tool | Purpose | Priority |
|------|---------|----------|
| Overleaf / LaTeX | Paper writing and collaboration | P0 |
| Zotero | Reference management | P0 |
| Jupyter Notebook | Exploratory analysis / experiment records | P0 |
| Google Scholar Alerts | Track latest related papers | P1 |

---

## 13. Key Performance Indicators (KPIs)

### 13.1 Model Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Anomaly Detection AUC-ROC | > 0.94 | 5-fold cross-validation |
| Precision | > 0.80 | Test set evaluation |
| Recall | > 0.85 | Test set evaluation |
| F1 Score | > 0.87 | Test set evaluation |
| TM-GNN vs Best Baseline AUC improvement | > 5% | Comparison experiment |
| Multi-model fusion vs single model F1 improvement | > 10% | Ablation experiment |

### 13.2 System Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Real-time scoring latency P50 | < 20ms | Locust load test |
| Real-time scoring latency P99 | < 100ms | Locust load test |
| Data processing throughput | > 10,000 TPS | Peak load test |
| System availability | > 99.9% | Monitoring statistics |

### 13.3 Academic Output

| Metric | Target |
|--------|--------|
| Paper submissions | >= 1 (target: KDD / WWW / Financial Crypto) |
| Open-source repository | Complete code + documentation + reproducibility |
| Test coverage | > 80% |

---

## 14. Risk Management

| ID | Risk | Impact | Probability | Mitigation |
|----|------|--------|-------------|-----------|
| R1 | Insufficient labeled data | High | High | Semi-supervised / unsupervised methods; Elliptic pre-training; synthetic augmentation |
| R2 | High false positive rate | High | Medium | Human feedback loop; tiered alerting; continuous threshold calibration |
| R3 | Real-time performance shortfall | High | Medium | Model lightweighting (ONNX/TensorRT); feature pre-computation; multi-level caching |
| R4 | Regulatory requirement changes | Medium | High | Fully configurable rule engine; extensible compliance interfaces |
| R5 | On-chain API instability | Medium | Medium | Multi-source redundancy (Alchemy + Infura); retry + degradation strategy |
| R6 | Adversarial adaptation | High | Medium | Monthly model retraining; adversarial training; dual rule+model engine |
| R7 | Cross-chain data alignment challenges | Medium | Medium | Bridge transaction tracking; heuristic address matching; incremental improvement |
| R8 | Team member unavailability | Medium | Low | Comprehensive documentation; code review ensures knowledge sharing |

---

## 15. References & Datasets

### 15.1 Key References

1. Weber, M., et al. (2019). "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics." KDD Workshop.
2. Rossi, E., et al. (2020). "Temporal Graph Networks for Deep Learning on Dynamic Graphs." ICML Workshop.
3. Hamilton, W., et al. (2017). "Inductive Representation Learning on Large Graphs." NeurIPS. (GraphSAGE)
4. Velickovic, P., et al. (2018). "Graph Attention Networks." ICLR. (GAT)
5. Liu, F.T., et al. (2008). "Isolation Forest." ICDM.
6. Lundberg, S. & Lee, S. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS. (SHAP)
7. Pareja, A., et al. (2020). "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs." AAAI.
8. Kipf, T. & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR.
9. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." KDD.
10. Alarab, I., et al. (2020). "Competence of Graph Convolutional Networks for Anti-Money Laundering in Bitcoin Blockchain." MLCS.

### 15.2 Public Datasets

| Dataset | Source | Content | Usage |
|---------|--------|---------|-------|
| Elliptic | Kaggle (Elliptic Co.) | 203,769 BTC transactions, labeled licit/illicit | Core training and evaluation |
| Ethereum Fraud Detection | Kaggle | ETH fraud address data | Supplementary training |
| Bitcoin Alpha + OTC Trust | Stanford SNAP | BTC trust scoring network | Graph analysis |
| Ethereum Token Transfers | Google BigQuery | ETH ERC-20 token transfers | Large-scale graph construction |
| OFAC SDN List | U.S. Treasury | Sanctions list | Compliance screening |

---

Document Version: v2.0
Created: March 18, 2026
Author: PhD ML Research Engineer
Project: ChainGuard
