# ChainGuard: Temporal Heterogeneous Graph Neural Networks for Cross-Chain Cryptocurrency Fraud Detection

# Research Proposal

---

Author: NYU Tandon School of Engineering, MS Candidate

Date: March 18, 2026

---

## Table of Contents

1. Introduction & Motivation
2. Related Work
3. Research Questions
4. Proposed Method: TH-GNN
5. Datasets & Data Strategy
6. Experiment Design
7. Expected Contributions
8. Timeline
9. References

---

## 1. Introduction & Motivation

### 1.1 The Cross-Chain Money Laundering Problem

The cryptocurrency ecosystem has evolved from isolated blockchains to an interconnected multi-chain landscape. Cross-chain bridges — protocols that enable asset transfers between blockchains — processed over $28 billion in token transfers across 11 blockchains in the latter half of 2024 alone (André Augusto et al., 2025). While bridges enable legitimate interoperability, they have also become the primary infrastructure for money laundering: criminals exploit bridges to fragment transaction trails across multiple chains, breaking the traceability that single-chain analytics depend on.

The scale of the problem is significant:
- Cross-chain bridge attacks caused over $2.8 billion in losses between 2021-2024 (Wu et al., 2025)
- Cross-chain laundering volumes increased fivefold since 2022 (Chainalysis, 2025)
- 49 documented cross-chain bridge attack incidents between June 2021 and September 2024 (Wu et al., 2025)

### 1.2 Limitations of Existing Approaches

Current fraud detection methods fall into three categories, each with significant limitations:

**Single-chain GNN methods** (GCN, GAT, GraphSAGE on Elliptic dataset): These achieve strong performance on single-chain benchmarks (AUC > 0.95 on Elliptic) but are fundamentally blind to cross-chain activity. When a criminal bridges funds from Ethereum to BSC, the trail goes cold for any single-chain detector.

**Rule-based cross-chain tools** (XChainWatcher, ABCTracer): These use hand-crafted Datalog rules or heuristics to trace cross-chain transactions. While useful for known attack patterns, they cannot generalize to novel fraud schemes and require manual rule updates.

**Cross-chain ML methods** (GMM-CCT, BSCI 2024): The only published work combining graph-based methods with cross-chain fraud detection. GMM-CCT uses Node2vec + GCN + LR + XGBoost fusion, achieving 57% precision and 43% recall — far below the threshold for practical deployment. Critically, GMM-CCT uses a basic GCN that treats all edges identically (no distinction between native transactions and bridge operations) and has no temporal modeling.

### 1.3 Research Gap

The gap we address is precise:

| Capability | Single-chain GNN | Rule-based | GMM-CCT | **TH-GNN (ours)** |
|-----------|-----------------|------------|---------|-------------------|
| Cross-chain awareness | ❌ | ✅ | ✅ | ✅ |
| Learned representations | ✅ | ❌ | ✅ | ✅ |
| Heterogeneous edge types | ❌ | N/A | ❌ | ✅ |
| Temporal modeling | ❌ | ❌ | ❌ | ✅ |
| Explainability | ❌ | ✅ (rules) | ❌ | ✅ (attention) |

No existing method combines learned graph representations, heterogeneous edge modeling for bridge vs. native transactions, and temporal dynamics. This is the specific gap ChainGuard fills.

---

## 2. Related Work

### 2.1 GNN-Based Blockchain Fraud Detection (Single-Chain)

The Elliptic dataset (Weber et al., 2019) catalyzed a wave of GNN-based Bitcoin fraud detection research. Key developments:

- **GCN/GAT/GraphSAGE** on Elliptic: Baseline GNN methods achieve AUC 0.93-0.97 on the binary classification task (Alarab et al., 2020; Pareja et al., 2020).
- **EvolveGCN** (Pareja et al., AAAI 2020): Introduces temporal modeling by using an RNN to evolve GCN parameters across time steps. Validated on Bitcoin OTC trust network.
- **Elliptic2** (Bellei et al., 2024): Extends the Elliptic dataset to 50M nodes and 200M edges with subgraph-level labels, enabling subgraph classification tasks.
- **MDST-GNN** (Chen et al., 2025): Multi-distance spatial-temporal GNN that captures local and global dependencies; +1.5% AUC-ROC over prior SOTA on Elliptic.
- **CoSemiGNN** (2025): Dynamic GNN with semi-supervised co-association to address label scarcity in blockchain fraud detection.
- **ChronoWave-GNN** (2025): Wavelet-temporal graph transformer for Bitcoin anomaly detection.

**Common limitation**: All methods operate on a single blockchain. They cannot detect or leverage cross-chain patterns.

### 2.2 Cross-Chain Security and Detection

- **XChainWatcher** (André Augusto et al., 2024): Logic-driven anomaly detector for cross-chain bridges using Datalog rules. Provides the first open dataset of 81K+ cross-chain events across Ronin and Nomad bridges. Rule-based, not learned.
- **BridgeGuard** (Wu et al., ACM WWW 2025): Detects cross-chain bridge attacks using graph motif mining. Collected 203 attack transactions + 40K normal from 49 attack incidents. Uses graph representation but not deep GNN.
- **ABCTracer** (2025): Cross-chain transaction tracing across 12 bridges with F1 > 91% for address linkage. Focuses on tracing, not fraud classification.
- **XChainDataGen** (André Augusto et al., 2025): Cross-chain dataset generation framework. Extracted 11.28M cross-chain transactions from 8 bridges across 11 blockchains. No fraud labels, but provides the largest public cross-chain transaction dataset.

### 2.3 Cross-Chain Graph-Based Anomaly Detection

- **GMM-CCT** (ACM BSCI 2024): The closest prior work. Uses Node2vec for graph embedding, then fuses GCN, LR, and XGBoost predictions. Achieves 57% precision and 43% recall on cross-chain anomaly detection. **Limitations**: (1) Homogeneous GCN — does not distinguish bridge edges from native edges; (2) No temporal modeling; (3) Node2vec is transductive and cannot generalize to unseen nodes.
- **"Anomaly Detection in Cross-Chain Bridges: A Data Analytics Study"** (2025): Uses traditional ML (Random Forest, Gradient Boosting) on tabular features from Wormhole bridge transactions. Not graph-based.

### 2.4 Positioning

ChainGuard builds on the datasets and problem formulation established by this body of work, while addressing the specific technical limitations of GMM-CCT through heterogeneous edge modeling, temporal attention, and inductive learning.

---

## 3. Research Questions

**RQ1 (Graph Construction)**: How should cross-chain bridge transactions be represented in a heterogeneous graph to preserve inter-chain relationships while enabling GNN message passing?

**RQ2 (Cross-Chain Value)**: Does incorporating cross-chain bridge information significantly improve fraud detection performance compared to single-chain GNN methods? If so, by how much?

**RQ3 (Component Analysis)**: What is the relative contribution of (a) heterogeneous edge typing, (b) temporal attention, and (c) cross-chain topology to overall detection performance?

---

## 4. Proposed Method: TH-GNN

### 4.1 Problem Formulation

**Input**: A temporal multi-chain transaction graph G = (V, E, X, T) where:
- V = set of wallet addresses across all chains
- E = E_native ∪ E_bridge (native intra-chain transactions + cross-chain bridge operations)
- X ∈ R^{|V| × d} = node feature matrix
- T = sequence of graph snapshots {G_1, G_2, ..., G_t} at discrete time windows

**Output**: For each node v ∈ V, a probability p(v) ∈ [0,1] indicating fraud risk.

**Task**: Semi-supervised node classification under extreme label imbalance and cross-chain label scarcity.

### 4.2 Cross-Chain Heterogeneous Graph Construction

We construct a unified multi-chain graph where:

1. **Nodes** represent wallet addresses. Addresses active on multiple chains via bridges are represented as a single node with cross-chain connectivity.

2. **Native edges** (e ∈ E_native) represent intra-chain transactions (ETH→ETH, BSC→BSC, etc.) with features: [amount, gas_fee, timestamp, token_type].

3. **Bridge edges** (e ∈ E_bridge) represent cross-chain bridge operations with features: [amount, source_chain, target_chain, bridge_protocol, timestamp, fee].

4. **Address alignment**: For cross-chain bridge transactions, the deposit address on chain A and the withdrawal address on chain B are linked through the bridge event logs (following ABCTracer methodology, F1 > 91%).

### 4.3 Heterogeneous Message Passing

Standard GNN message passing treats all edges equally:

h_v^{(l+1)} = AGG({h_u^{(l)} : u ∈ N(v)})

We introduce type-specific transformations:

h_v^{(l+1)} = σ( Σ_{r∈R} Σ_{u∈N_r(v)} α_{vu}^r · W_r^{(l)} · h_u^{(l)} )

Where:
- R = {native, bridge} are edge types
- W_native, W_bridge are type-specific weight matrices
- α_{vu}^r are attention coefficients computed per edge type
- N_r(v) is the set of neighbors connected via edge type r

**Motivation**: Bridge edges carry fundamentally different semantics than native transfers. A bridge operation involves protocol-level mechanics (locking, minting, burning) that native transfers do not. Treating them identically (as GMM-CCT does) discards this structural information.

### 4.4 Temporal Attention Module

Fraud patterns evolve over time: layering phases involve rapid small transactions, integration phases involve consolidation. We model temporal dynamics through:

1. **Graph snapshots**: Partition the temporal graph into T discrete snapshots {G_1, ..., G_T} at fixed time windows (e.g., 1-hour blocks).

2. **Per-snapshot embedding**: Apply the heterogeneous message passing layer to each snapshot independently, producing h_v^{(t)} for each node v at time t.

3. **Temporal attention**: Compute attention over the sequence of embeddings:

z_v = Σ_t β_t · h_v^{(t)}

Where β_t = softmax(w^T · tanh(W_T · h_v^{(t)} + b_T))

The attention weights β_t are interpretable: high weights indicate which time windows are most predictive of fraud for each node.

### 4.5 Classification and Training

**Classification head**: MLP with sigmoid output for binary fraud/normal classification.

**Loss function**: Binary cross-entropy with class weights to handle imbalance:

L = -Σ_v [w_1 · y_v · log(p_v) + w_0 · (1-y_v) · log(1-p_v)]

**Semi-supervised training**: For nodes without fraud labels (the majority of cross-chain nodes), we add a self-supervised contrastive loss:

L_contrast = -log(exp(sim(z_v, z_v^+)/τ) / Σ_k exp(sim(z_v, z_k^-)/τ))

Where z_v^+ is an augmented view of node v (e.g., edge dropout), encouraging the model to learn meaningful representations even without labels.

**Total loss**: L_total = L_supervised + λ · L_contrast

### 4.6 Explainability via Attention

Both the graph attention weights (α) and temporal attention weights (β) provide built-in explainability:

- **Graph attention α**: "Which neighbors most influenced this node's risk score?" — identifies the most suspicious connected addresses.
- **Temporal attention β**: "Which time period was most suspicious?" — identifies when the fraud pattern was most active.

This dual-layer explainability produces natural reasoning chains:

> "Address 0x1a2b was flagged because (1) it received funds from known bridge attacker 0x3c4d [graph attention], (2) its transaction volume spiked anomalously during hours 14-16 on March 5 [temporal attention]."

---

## 5. Datasets & Data Strategy

### 5.1 Primary Datasets

| Dataset | Role | Scale | Labels |
|---------|------|-------|--------|
| **XChainDataGen** | Cross-chain graph structure | 11.28M txs, 8 bridges, 11 chains | No fraud labels |
| **BridgeGuard** | Cross-chain fraud labels | 203 attack + 40K normal | Attack/normal |
| **XChainWatcher** | Case study data | 81K+ cross-chain events | Attack events |
| **Elliptic2** | Single-chain baseline | 50M nodes, 200M edges | Licit/illicit |
| **ETH Fraud Detection** | Single-chain baseline | 9.8K addresses | Fraud/normal |
| **Nomad Hack Data** | Case study | Raw attack txs | Attacker/white-hat |

### 5.2 Addressing Label Scarcity

The central data challenge: BridgeGuard provides only ~200 labeled cross-chain attack transactions. We address this through three complementary strategies:

**Strategy 1: Cross-Chain Label Propagation**
Use labeled single-chain nodes (from Elliptic2) and propagate labels through bridge edges to unlabeled cross-chain nodes. If a known illicit Bitcoin address bridges funds to Ethereum, the receiving Ethereum address inherits a soft label.

**Strategy 2: Contrastive Pre-Training**
Pre-train node embeddings on the full unlabeled XChainDataGen graph using graph contrastive learning (GraphCL). This learns structural representations without requiring labels, which can then be fine-tuned on the small labeled set.

**Strategy 3: Few-Shot Transfer**
Train a strong single-chain model on Elliptic2 (abundant labels), then transfer learned parameters to initialize the cross-chain model. Fine-tune on the small BridgeGuard labeled set.

### 5.3 Data Processing Pipeline

1. **Download & parse**: XChainDataGen (Zenodo), BridgeGuard (ACM), Elliptic2 (Kaggle)
2. **Address alignment**: Link cross-chain addresses using bridge event logs (ABCTracer methodology)
3. **Feature extraction**: Node features (tx count, avg amount, degree, age, chain ID) + edge features (amount, timestamp, type)
4. **Graph construction**: Build PyG HeteroData objects with native and bridge edge types
5. **Temporal slicing**: Partition into time-windowed snapshots
6. **Train/val/test split**: Temporal split (earlier data for training, later for testing) to prevent information leakage

---

## 6. Experiment Design

### 6.1 Experiment 1: Ablation Study (Primary)

**Purpose**: Quantify the marginal contribution of each TH-GNN component.

| Variant | Cross-chain graph? | Hetero edges? | Temporal? | Tests |
|---------|-------------------|---------------|-----------|-------|
| M1: Single-chain GCN | ❌ | ❌ | ❌ | Baseline |
| M2: Single-chain + bridge features | ❌ | ❌ | ❌ | Feature augmentation |
| M3: Multi-chain GCN (homogeneous) | ✅ | ❌ | ❌ | Graph topology |
| M4: Multi-chain Hetero-GNN | ✅ | ✅ | ❌ | Edge type distinction |
| M5: Full TH-GNN | ✅ | ✅ | ✅ | Complete model |

**Key comparisons**:
- M1 vs M2: Does bridge metadata help as features? (Δ_feature)
- M1 vs M3: Does cross-chain graph structure help? (Δ_topology)
- M3 vs M4: Does heterogeneous edge typing matter? (Δ_hetero)
- M4 vs M5: Does temporal attention help? (Δ_temporal)

**Statistical rigor**: 5-fold cross-validation, paired t-test (p < 0.05), report mean ± std.

### 6.2 Experiment 2: Baseline Comparison

**Purpose**: Compare TH-GNN against 9 established baselines.

**Baselines**: LR, RF, XGBoost, Isolation Forest, GCN, GraphSAGE, GAT, EvolveGCN, GMM-CCT

**Datasets**: (a) Elliptic2 single-chain, (b) cross-chain BridgeGuard + XChainDataGen

**Metrics**: AUC-ROC, F1, Precision, Recall, Precision@100

### 6.3 Experiment 3: Label Scarcity Analysis

**Purpose**: Evaluate performance degradation as labeled cross-chain data decreases.

Test with 100%, 50%, 25%, 10% of available cross-chain labels. Compare:
- Supervised-only training
- + Contrastive pre-training
- + Cross-chain label propagation
- + Few-shot transfer from Elliptic2

### 6.4 Experiment 4: Attention Explainability Analysis

**Purpose**: Validate that attention weights are meaningful, not random.

- Visualize graph attention on known attack subgraphs (Ronin, Nomad)
- Compare attention distributions: attack nodes vs. normal nodes
- Measure alignment between high-attention edges and known laundering paths

### 6.5 Experiment 5: Case Study — Ronin Bridge Hack

**Purpose**: Qualitative analysis demonstrating practical utility.

Using XChainWatcher's Ronin Bridge data:
1. Reconstruct the attack transaction graph
2. Run TH-GNN inference
3. Analyze: Does the model flag the attacker address? When in the timeline? Which attention heads activate?
4. Compare with XChainWatcher's rule-based detection

---

## 7. Expected Contributions

### 7.1 Methodological Contributions

1. **TH-GNN architecture**: A temporal heterogeneous graph neural network with type-specific message passing for cross-chain fraud detection — the first to model bridge edges as a distinct edge type in a GNN framework.

2. **Cross-chain label propagation**: A method to transfer fraud labels from labeled chains to unlabeled chains through bridge edges, addressing the fundamental label scarcity problem in cross-chain settings.

3. **Ablation-driven cross-chain analysis**: Systematic quantification of whether and how much cross-chain information improves fraud detection — answering a key open question in the field.

### 7.2 Empirical Contributions

4. **Benchmark results**: First comprehensive comparison of single-chain vs. multi-chain GNN methods for fraud detection on public cross-chain datasets.

5. **Case study**: Detailed analysis of TH-GNN behavior on real-world bridge attacks (Ronin, Nomad).

### 7.3 Positioning for PhD Application

This project demonstrates:
- **Graph learning depth**: Heterogeneous message passing, temporal attention, type-specific transformations
- **Probabilistic thinking**: Semi-supervised learning, contrastive objectives, uncertainty in label propagation
- **Experimental rigor**: Ablation studies, statistical testing, case studies
- **Cross-domain adaptability**: Applying graph ML methodology (also used in molecular graphs, protein interactions) to financial networks

---

## 8. Timeline (20 Weeks)

| Phase | Weeks | Activities | Deliverable |
|-------|-------|-----------|-------------|
| **Phase 1: Data** | 1-4 | Download datasets, parse bridge events, build multi-chain graph, extract features, create PyG HeteroData | Multi-chain dataset ready |
| **Phase 2: Baselines** | 5-8 | Implement all 9 baselines, run on Elliptic2 and cross-chain data, establish benchmark | Baseline results table |
| **Phase 3: TH-GNN** | 9-12 | Implement heterogeneous message passing, temporal attention, training loop, semi-supervised components | Core model code + initial results |
| **Phase 4: Experiments** | 13-16 | Run all 5 experiments, ablation study, label scarcity analysis, case study | Complete experimental results |
| **Phase 5: Write-up** | 17-20 | Write arXiv technical report (8-10 pages), create figures, finalize code for open-source release | arXiv preprint + GitHub release |

---

## 9. References

### Cross-Chain Datasets & Tools
- André Augusto et al. (2025). "XChainDataGen: A Cross-Chain Dataset Generation Framework." arXiv:2503.13637.
- André Augusto et al. (2024). "XChainWatcher: Monitoring and Identifying Attacks in Cross-Chain Bridges." arXiv:2410.02029.
- Wu et al. (2025). "Safeguarding Blockchain Ecosystem: Understanding and Detecting Attack Transactions on Cross-chain Bridges." ACM Web Conference 2025.
- ABCTracer (2025). "Track and Trace: Automatically Uncovering Cross-chain Transactions in the Multi-blockchain Ecosystems." arXiv:2504.01822.

### Cross-Chain Anomaly Detection
- GMM-CCT (2024). "Cross-chain Abnormal Transaction Detection via Graph-based Multi-model Fusion." ACM BSCI 2024.
- "Anomaly Detection in Cross-Chain Bridges: A Data Analytics Study." (2025).

### Single-Chain GNN Fraud Detection
- Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017.
- Hamilton et al. (2017). "Inductive Representation Learning on Large Graphs." NeurIPS 2017.
- Veličković et al. (2018). "Graph Attention Networks." ICLR 2018.
- Pareja et al. (2020). "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs." AAAI 2020.
- Bellei et al. (2024). "Elliptic2: A Dataset for Financial Network Analysis." arXiv:2404.19109.
- Chen et al. (2025). "MDST-GNN: Multi-Distance Spatial-Temporal Graph Neural Network for Anomaly Detection in Blockchain Transactions." Advanced Intelligent Systems.

### Foundational Methods
- Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System." KDD 2016.
- Liu et al. (2008). "Isolation Forest." ICDM 2008.
- You et al. (2020). "Graph Contrastive Learning with Augmentations." NeurIPS 2020.
