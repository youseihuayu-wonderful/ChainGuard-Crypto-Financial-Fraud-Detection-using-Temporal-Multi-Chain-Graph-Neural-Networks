# ChainGuard Experiment Phases

> **项目定位**: NYU Tandon MS 论文 + PhD 申请 research sample
> **核心命题**: "跨链异构时序 GNN 在欺诈检测中的有效性验证"
> **方法**: 消融实验驱动 (Ablation-driven research)

---

## Phase 1: 数据获取与基线建立 (Data & Baseline)

### 目标
拿到数据、理解数据、跑通最简单的模型，建立实验基准线。

### 任务清单

| # | 任务 | 状态 | 产出物 | 验证标准 |
|---|------|------|--------|----------|
| 1.1 | 下载 Elliptic 原版数据集 | ✅ 完成 | `data/raw/elliptic_bitcoin_dataset/` | 203,769 节点、234,355 边、4,545 illicit |
| 1.2 | 下载 XChainDataGen 跨链数据 | ✅ 完成 | `data/raw/xchaindatagen/` | 4 个桥文件，1,768,281 笔跨链交易 |
| 1.3 | 搭建项目骨架 | ✅ 完成 | `src/`, `experiments/`, `notebooks/` | 模块化结构，pyproject.toml 依赖管理 |
| 1.4 | 数据探索 (EDA) | ✅ 完成 | `notebooks/01_data_exploration.py` | 统计数字与官方论文一致 |
| 1.5 | GCN Baseline (M1) | ✅ 完成 | `experiments/scripts/train_gcn_baseline.py` | AUC=0.7449 (时序划分) |

### M1 实验结果 (基准线)

| 指标 | 值 | 说明 |
|------|-----|------|
| AUC-ROC | **0.7449** | 时序划分 (train: ts 1-34, val: 35-41, test: 42-49) |
| F1 (illicit) | 0.2812 | |
| Precision | 0.2782 | |
| Recall | 0.2843 | |
| Seed | 42 | 可复现 |

> **注意**: 文献报告 AUC 0.93-0.97 使用随机划分。我们的时序划分更严格但更真实，AUC 较低是预期内的。

### 已知数据质量问题

| 问题 | 数据集 | 严重程度 | 处理方案 |
|------|--------|----------|----------|
| connext/orbitChain 的 srcUSD 全为 NaN | XChainDataGen | 中 | 使用 srcAmount 替代，或从链上查询 USD 价格 |
| orbitChain 部分行 fromChain/toChain 为 NaN | XChainDataGen | 中 | 使用 fromChainRaw/toChainRaw 回填 |
| 无欺诈标签 | XChainDataGen | 高 | 通过 Elliptic 标签 + 桥边传播 (M5 解决) |

### Phase 1 完成日期: 2026-03-19

---

## Phase 2: 消融实验 (Ablation Study) — M2 到 M5

### 目标
逐步加入创新组件，通过消融实验证明每个组件的贡献。这是论文的**核心章节**。

### 模型变体总览

```
M1 (GCN)  →  M2 (+时序)  →  M3 (+异构边)  →  M4 (完整TH-GNN)  →  M5 (+跨链传播)
  基线          时序有用吗？     异构有用吗？      两者结合更好吗？     跨链信息有用吗？
```

### 任务清单

| # | 任务 | 状态 | 产出物 | 验证标准 |
|---|------|------|--------|----------|
| **数据准备** | | | | |
| 2.1 | XChainDataGen 数据清洗 (处理 NaN) | 待开始 | `src/data/xchain_loader.py` | 所有行有有效的 fromChain/toChain |
| 2.2 | 构建异构图 (PyG HeteroData) | 待开始 | `src/data/graph_builder.py` | Elliptic + 跨链桥边的统一异构图 |
| **M2: GCN + 时序注意力** | | | | |
| 2.3 | 实现时序注意力模块 | 待开始 | `src/models/modules/temporal_attention.py` | 注意力权重可视化合理 |
| 2.4 | 训练 M2 并记录结果 | 待开始 | `experiments/scripts/train_m2_temporal.py` | AUC > M1 (0.7449) |
| **M3: GCN + 异构边建模** | | | | |
| 2.5 | 实现异构消息传递 (W_native vs W_bridge) | 待开始 | `src/models/modules/hetero_conv.py` | 两套独立权重矩阵 |
| 2.6 | 训练 M3 并记录结果 | 待开始 | `experiments/scripts/train_m3_hetero.py` | AUC > M1 (0.7449) |
| **M4: 完整 TH-GNN (时序 + 异构)** | | | | |
| 2.7 | 合并 M2 + M3 为完整 TH-GNN | 待开始 | `src/models/th_gnn.py` | 同时包含时序注意力和异构边 |
| 2.8 | 训练 M4 并记录结果 | 待开始 | `experiments/scripts/train_m4_thgnn.py` | AUC > M2 且 > M3 |
| **M5: TH-GNN + 跨链标签传播** | | | | |
| 2.9 | 实现跨链标签传播模块 | 待开始 | `src/models/modules/label_propagation.py` | 标签通过桥边传播到无标签链 |
| 2.10 | 构建 Elliptic + XChainDataGen 统一图 | 待开始 | 更新 `graph_builder.py` | 多链节点 + 桥边连接 |
| 2.11 | 训练 M5 并记录结果 | 待开始 | `experiments/scripts/train_m5_crosschain.py` | AUC >= M4 |
| **汇总** | | | | |
| 2.12 | 编写一键复现脚本 | 待开始 | `experiments/scripts/run_all_ablation.py` | 一个命令跑完 M1-M5 |
| 2.13 | 生成消融实验结果表 | 待开始 | `experiments/results/ablation_results.json` | 包含所有指标 + 标准差 |

### 建议执行顺序

```
Week 1:  2.3 → 2.4 (M2: 时序注意力，仅需 Elliptic 数据)
Week 1:  2.1 → 2.2 (并行: 数据清洗 + 异构图构建)
Week 2:  2.5 → 2.6 (M3: 异构边建模)
Week 2:  2.7 → 2.8 (M4: 合并为完整 TH-GNN)
Week 3:  2.9 → 2.10 → 2.11 (M5: 跨链标签传播)
Week 3:  2.12 → 2.13 (汇总 + 一键复现)
```

### 消融实验结果表 (待填写)

| 变体 | AUC-ROC | F1 | Precision | Recall | Δ AUC vs M1 | 说明 |
|------|---------|-----|-----------|--------|-------------|------|
| M1: GCN | 0.7449 | 0.2812 | 0.2782 | 0.2843 | — | 基线 |
| M2: +Temporal | — | — | — | — | — | 待实验 |
| M3: +Hetero | — | — | — | — | — | 待实验 |
| M4: TH-GNN | — | — | — | — | — | 待实验 |
| M5: +CrossChain | — | — | — | — | — | 待实验 |

### 质量检查标准

- [ ] M1 < M2 ≤ M3 < M4 ≤ M5 (大致单调递增)
- [ ] 每步提升在 +0.5% 到 +3% 之间 (合理范围)
- [ ] 5 个变体使用完全相同的数据和划分方式
- [ ] seed=42, 两次运行结果完全一致
- [ ] 每个变体包含标准差 (跑 3-5 次不同 seed)
- [ ] 如果某步提升 > 5%, 需要排查是否有 bug

### 关键技术细节

**时序注意力 (M2)**:
- 将 49 个时间步切成图快照 (graph snapshots)
- 用 attention 机制对快照加权聚合
- 核心公式: `h_t = Σ_i α_i * GCN(G_i)`, α 为学到的注意力权重

**异构消息传递 (M3)**:
- 链内交易边: `h_v = W_native * Σ_{u∈N_native(v)} h_u`
- 跨链桥边: `h_v = W_bridge * Σ_{u∈N_bridge(v)} h_u`
- W_native 和 W_bridge 是两套独立参数

**跨链标签传播 (M5)**:
- Elliptic 有标签 (4,545 illicit)，跨链数据无标签
- 通过桥边将 Elliptic 的标签"传播"到跨链节点
- 半监督学习: labeled loss + consistency regularization

---

## Phase 3: Baseline 对比实验 (待规划)

### 目标
将 TH-GNN 与已有方法进行公平对比，证明我们的方法更优。

### 任务清单 (初步)

| # | 任务 | 状态 | 说明 |
|---|------|------|------|
| 3.1 | 实现传统 ML baselines (LR, RF, XGBoost) | 待开始 | 非图方法对比 |
| 3.2 | 实现 GAT baseline | 待开始 | 图注意力网络对比 |
| 3.3 | 实现 GraphSAGE baseline | 待开始 | 采样聚合对比 |
| 3.4 | 复现 GMM-CCT (BSCI 2024) 结果 | 待开始 | 主要竞争对手: 57% precision, 43% recall |
| 3.5 | 生成对比实验结果表 | 待开始 | 所有方法统一评估 |

### 对比实验结果表 (待填写)

| 方法 | 类型 | AUC-ROC | F1 | Precision | Recall |
|------|------|---------|-----|-----------|--------|
| Logistic Regression | 非图 | — | — | — | — |
| Random Forest | 非图 | — | — | — | — |
| XGBoost | 非图 | — | — | — | — |
| GCN (M1) | 图-单链 | 0.7449 | 0.2812 | 0.2782 | 0.2843 |
| GAT | 图-单链 | — | — | — | — |
| GraphSAGE | 图-单链 | — | — | — | — |
| GMM-CCT | 图-跨链 | — | — | ~0.57 | ~0.43 |
| **TH-GNN (Ours)** | **图-跨链** | — | — | — | — |

---

## Phase 4: Case Study & 论文撰写 (待规划)

### 目标
用真实攻击案例验证模型，完成 arXiv technical report (8-10 页)。

### 任务清单 (初步)

| # | 任务 | 状态 | 说明 |
|---|------|------|------|
| 4.1 | Ronin Bridge Hack 案例分析 | 待开始 | 用模型检测已知攻击，展示注意力可视化 |
| 4.2 | Nomad Bridge Hack 案例分析 | 待开始 | 第二个案例 (数据已有: nomad-xyz/hack-data) |
| 4.3 | 可解释性分析 | 待开始 | 图注意力 + 时序注意力权重可视化 |
| 4.4 | 撰写论文 Introduction + Related Work | 待开始 | |
| 4.5 | 撰写论文 Method 章节 | 待开始 | TH-GNN 架构描述 |
| 4.6 | 撰写论文 Experiments 章节 | 待开始 | 消融 + 对比 + Case Study |
| 4.7 | 撰写论文 Conclusion | 待开始 | |
| 4.8 | 上传 arXiv | 待开始 | 目标: 8-10 页 technical report |

### 论文结构

```
1. Introduction (1 页)
2. Related Work (1 页)
   - 单链欺诈检测
   - 跨链分析
   - 时序图神经网络
3. Method: TH-GNN (2 页)
   - 问题定义
   - 异构消息传递
   - 时序注意力
   - 跨链标签传播
4. Experiments (3 页)
   - 数据集描述
   - 消融实验 (M1-M5)
   - Baseline 对比
   - Case Study
5. Conclusion (0.5 页)
References (1 页)
```

---

## 时间线总览

| 阶段 | 时间 | 核心产出 | 状态 |
|------|------|----------|------|
| **Phase 1**: 数据 & 基线 | Week 1-2 | Elliptic 数据 + M1 baseline | ✅ 完成 |
| **Phase 2**: 消融实验 | Week 3-5 | M2-M5 + 消融结果表 | 进行中 |
| **Phase 3**: 对比实验 | Week 6-7 | LR/RF/XGB/GAT/GraphSAGE/GMM-CCT 对比 | 待开始 |
| **Phase 4**: Case Study & 论文 | Week 8-10 | arXiv technical report (8-10页) | 待开始 |

---

## 关键配置

| 项目 | 值 |
|------|-----|
| Python | 3.12.13 |
| PyTorch Geometric | latest |
| 随机种子 | 42 |
| 数据划分 | 时序划分 (train: ts 1-34, val: 35-41, test: 42-49) |
| 训练 epochs | 200 (带早停, patience=20) |
| 类权重 | licit/illicit = 7.63x |
| GitHub | [ChainGuard Repo](https://github.com/youseihuayu-wonderful/ChainGuard-Crypto-Financial-Fraud-Detection-using-Temporal-Multi-Chain-Graph-Neural-Networks) |
