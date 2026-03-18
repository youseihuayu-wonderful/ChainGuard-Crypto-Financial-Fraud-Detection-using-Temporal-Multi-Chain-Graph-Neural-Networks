# ChainGuard：基于时序多链图神经网络的加密货币金融欺诈检测

# 完整项目方案书

---

GitHub: https://github.com/youseihuayu-wonderful/ChainGuard-Crypto-Financial-Fraud-Detection-using-Temporal-Multi-Chain-Graph-Neural-Networks

作者：博士研究生，预计 2026 年毕业

日期：2026 年 3 月 18 日

---

## 目录

1. 项目概述与研究动机
2. 研究方法论（PhD 框架）
3. 研究问题、假设与贡献
4. 项目独特亮点与竞争力分析
5. 系统技术架构
6. 核心创新：TM-GNN 模型
7. 四维特征工程体系
8. 实验设计框架
9. 数据策略
10. 项目管理框架（20 周时间线）
11. 五名工程师分工方案
12. 完整技术栈
13. 关键成功指标（KPIs）
14. 风险管理
15. 参考文献与数据集

---

## 1. 项目概述与研究动机

### 1.1 背景

加密货币市场经历了爆发式增长，总市值超过 3 万亿美元。然而，伴随增长的是金融犯罪的激增。仅 2025 年，流向欺诈计划的加密货币就约达 350 亿美元。跨链可疑转账超过 110 亿美元，随着复杂的跨链调查成熟，最终估计可能超过 750 亿美元。

传统反洗钱（AML）方法——主要是基于规则的系统——在面对加密货币交易的去中心化、匿名性和跨链特性时存在显著局限性。现有的机器学习方法仅在单链数据上运行，无法追踪跨越多条区块链的碎片化交易轨迹。

### 1.2 问题定义

ChainGuard 解决以下核心问题：

- P1（跨链盲区）：犯罪分子利用跨链桥在 Ethereum、BSC、Polygon 等网络间洗钱。通过桥转换的资产变成新代币，与原始资产没有链上引用关系，破坏了单链检测方法。

- P2（时序动态）：欺诈模式随时间演变。静态图分析遗漏了犯罪网络的时序演化，如渐进式资金分层和突发交易模式。

- P3（可解释性缺口）：监管机构（MiCA、FinCEN、Travel Rule）要求每笔被标记交易都有可审计的推理链。现有 GNN 模型是黑盒，无法满足合规要求。

- P4（实时性要求）：AML 合规要求实时风险评估。当前学术模型优先考虑准确性而非推理速度，使其不适合生产部署。

### 1.3 项目价值

- 学术价值：在跨链图学习用于欺诈检测领域做出新贡献，该领域现有学术研究极少
- 工业价值：为交易所和合规服务商提供生产级风险评分平台
- 社会价值：帮助监管机构打击加密货币金融犯罪，保护投资者

---

## 2. 研究方法论（PhD 框架）

### 2.1 设计科学研究（DSR）范式

本项目采用设计科学研究范式，同时产出研究制品（TM-GNN 模型 + ChainGuard 系统）和理论贡献（跨链图学习方法论）。

DSR 七步框架应用：

| 步骤 | DSR 原则 | 在 ChainGuard 中的应用 |
|------|----------|----------------------|
| 1 | 问题识别 | 现有方法在跨链欺诈检测中的盲区 |
| 2 | 解决方案目标 | 实时、可解释、高精度的多链欺诈检测 |
| 3 | 设计与开发 | TM-GNN 模型 + 统一多链图 + 端到端系统 |
| 4 | 演示 | 在 Elliptic、ETH Fraud 和自采集多链数据上实验 |
| 5 | 评估 | 定量（AUC/F1）+ 定性（案例研究）+ 效率（延迟） |
| 6 | 沟通 | 投稿 KDD/WWW/Financial Crypto + 开源代码 |
| 7 | 迭代 | 基于实验反馈优化模型 |

### 2.2 假设驱动开发

每个模型模块都从假设出发，通过严格实验验证：

- H1：多维特征表征（交易+钱包+图+时序）显著优于单一维度特征
- H2：时序多链图神经网络（TM-GNN）在跨链欺诈检测中优于静态 GNN >= 5% AUC
- H3：注意力机制能自动学习有意义的欺诈模式，提供节点级可解释性
- H4：多模型融合（GNN+异常检测+规则）的 F1 比最优单模型提升 >= 10%
- H5：SHAP 可解释性约束不会显著降低模型性能（AUC 下降 < 2%）
- H6：系统可在 100ms 内完成单笔交易的端到端风险评分

### 2.3 实验设计原则

- Baseline 对比：每个模型与 8+ 个 baseline 对比
- 消融实验：独立验证每个组件的贡献
- 统计显著性：5 折交叉验证 + 配对 t 检验（p < 0.05）
- 可复现性：固定随机种子、MLflow 记录所有超参数、开源代码

---

## 3. 研究问题、假设与贡献

### 3.1 研究问题

- RQ（总问题）：如何构建高效、可解释的跨链加密货币欺诈检测与风险评分系统？
  - RQ1（表征学习）：如何有效表征跨交易、钱包、图和时序四个维度的多链交易特征？
  - RQ2（检测方法）：时序多链 GNN 是否优于单链方法和静态 GNN？
  - RQ3（风险量化）：如何将多模型输出融合为统一的可解释风险评分？
  - RQ4（实用性）：如何在保持高精度的同时满足实时性和合规要求？

### 3.2 学术贡献

- 贡献 1（方法论）：提出 TM-GNN（时序多链图神经网络），一种融合时序、跨链结构和交易特征的端到端欺诈检测框架。

- 贡献 2（跨链图）：设计统一多链交易图，跨链桥操作形成异构链间边，实现跨链资金流追踪。

- 贡献 3（特征工程）：设计加密货币欺诈检测专用的四维特征体系：交易特征、钱包特征、图结构特征和时序特征。

- 贡献 4（系统）：构建端到端可解释风险评分平台，包含实时推理、SAR 自动生成和监管合规支持。

- 贡献 5（实证）：在 3 个数据集上与 10 种 baseline 方法进行大规模实验，证明方法的有效性。

---

## 4. 项目独特亮点与竞争力分析

### 4.1 五大独特亮点

独特亮点 1：跨链欺诈检测 — 学术蓝海

现状：几乎所有加密货币欺诈检测的学术论文都在单链（BTC 或 ETH）上运行。跨链资金追踪仅由商业公司（Chainalysis、TRM Labs、Elliptic）使用专有方法完成。

ChainGuard 的创新：构建统一多链交易图：
- 节点 = ETH、BSC、Polygon 上的地址
- 链内边 = 普通交易
- 链间边 = 桥接操作（新型异构边类型）
- TM-GNN 执行桥接感知的跨链消息传递

为什么重要：2025 年跨链可疑转移超过 110 亿美元。链跳和分层严重延迟归因。这是加密货币 AML 中最大的未解决问题。

研究空白证据：在学术数据库中搜索"cross-chain fraud detection GNN"几乎没有结果。这代表着巨大的发表机会。

---

独特亮点 2：内嵌可解释性（非事后解释）

现状：大多数 GNN 论文将 SHAP 或 GNNExplainer 作为事后解释工具。这些是"外挂"解释，可能无法忠实反映模型的实际推理过程。

ChainGuard 的创新：TM-GNN 具有双重内置注意力机制：
- 图注意力：解释"哪些邻居（交易对手）最重要"
- 时序注意力：解释"哪个时间窗口最可疑"
- 两者共同产生人类可读的推理链

为什么重要：2026 年 90% 的金融机构预计将使用 AI 进行 AML。监管机构明确要求可审计的推理链。"黑盒" AI 是不可接受的。ChainGuard 的内嵌可解释性直接满足 MiCA、FinCEN 和 Travel Rule 要求。

---

独特亮点 3：DeFi 特定欺诈模式检测

现状：ACM Computing Surveys 2025 明确指出："对于 DeFi 相关威胁（如 rug pull 和闪电贷攻击），构建有代表性的标注数据集在技术上仍然具有挑战性。"极少有学术论文涉及 DeFi 欺诈。

ChainGuard 的创新：
- 设计 DeFi 特定特征（流动性变化率、合约交互模式、闪电贷指标）
- 跨链桥攻击检测
- 潜在贡献首个 DeFi 欺诈标注数据集

---

独特亮点 4：端到端系统（学术研究 + 生产系统）

现状：学术论文只发布模型。没有系统、没有 API、没有仪表盘、没有合规模块。工业产品是专有的，没有公开方法论。

ChainGuard 的创新：弥合学术和工业之间的鸿沟：
- 研究：新颖的 TM-GNN 模型 + 严谨实验
- 系统：实时风险评分 API（P99 < 100ms）
- 合规：SAR 自动生成、OFAC 筛查
- 可视化：交互式调查仪表盘

---

独特亮点 5：半监督跨链对比学习

现状：Elliptic 数据集中 77% 的交易没有标注。大多数方法只使用标注数据，浪费了大部分可用信息。

ChainGuard 的创新：使用跨链对比学习利用：
- 无标注交易数据（区块链数据的大部分）
- 跨链对应关系（不同链上的同一实体）

### 4.2 竞争格局

| 竞争对手 | 估值 | 焦点 | ChainGuard 的优势 |
|---------|------|------|-------------------|
| Chainalysis | $86 亿 | 链上分析 + 执法合作 | 专有方法，无学术透明度 |
| Elliptic | $8 亿+ | 交易筛查 + 合规 | DeFi 覆盖仍处于早期 |
| TRM Labs | $12 亿 | 多链情报 | 有限的实时检测能力 |
| Merkle Science | $2 亿+ | 亚太合规 | 区域性，ML 深度有限 |
| 学术 SOTA | 不适用 | 单链 GNN 模型 | 无跨链、无系统、无合规 |

### 4.3 与现有学术方法对比（2025-2026）

| 方法 | 年份 | 技术路线 | 局限性 | ChainGuard 优势 |
|------|------|---------|--------|----------------|
| MDST-GNN | 2025 | 多距离时空 GNN | 单数据集、无跨链 | 多链统一图 |
| ATGAT | 2025 | 时序感知图注意力 | 可解释性未深入 | 内嵌双重注意力可解释性 |
| CoSemiGNN | 2025 | 半监督动态 GNN | 无 DeFi 场景 | DeFi 特定特征 + 跨链 |
| ChronoWave-GNN | 2026 | 小波+时序 GNN | 频域方法，可解释性差 | 人类可读的注意力解释 |
| EvolveGCN | 2020 | RNN 演化 GCN | 无跨链能力 | 桥接感知消息传递 |

---

## 5. 系统技术架构

### 5.1 五层架构

```
=================================================================
|                    第五层：应用层                               |
|   风控仪表盘 | 分析师工作台 | REST API | 报告系统               |
=================================================================
|                第四层：决策与合规层                              |
|   规则引擎 | SAR 生成器 | 案件管理 | 实时告警                   |
=================================================================
|                   第三层：模型层                                |
|   异常检测 | 风险评分 | TM-GNN | 行为分析                     |
=================================================================
|                第二层：数据处理层                               |
|   实时流处理(Kafka+Flink) | 批处理(Spark) | 特征工程引擎       |
=================================================================
|                 第一层：数据采集层                              |
|   链上数据采集 | 交易所 API | 合规数据源 | 跨链桥监控           |
=================================================================
```

### 5.2 数据采集层
- 链上数据采集器：Web3.py / Etherscan API / Alchemy (ETH)；BSCScan (BSC)；PolygonScan (Polygon)
- 交易所数据：Binance / Coinbase API
- 数据增强：IP 地理位置、OFAC 制裁名单、已知黑名单地址库
- 桥接数据：跨链桥交易监控（Wormhole、Multichain、Stargate）

### 5.3 数据处理与特征工程层
- 实时流处理：Kafka + Flink 实时交易特征计算
- 批处理：Spark 历史数据分析和模型训练数据准备
- Feature Store：Feast 在线/离线特征统一管理
- 图构建器：构建统一多链交易图 + 时序快照

### 5.4 模型层（核心）
- Baselines：LR、RF、XGBoost、Isolation Forest、Autoencoder
- 静态 GNN：GCN、GraphSAGE、GAT
- TM-GNN（本文提出）：时序多链图神经网络
- 融合引擎：多模型融合评分

### 5.5 决策与合规层
- 规则引擎：可配置的 if-then 规则
- 风险评分：多模型加权融合 -> 统一风险分数 (0-100)
- 告警分级：Low (0-25) / Medium (25-50) / High (50-75) / Critical (75-100)
- SAR 生成器：可疑活动报告自动生成
- 案件管理：分析师审核工作流、案件跟踪、审计日志

### 5.6 应用层
- 风控仪表盘：实时风险态势感知
- 分析师工作台：案件详情、交易图谱探索
- REST API：对外风险评分查询服务
- 报告系统：定期监管和统计报告

---

## 6. 核心创新：TM-GNN（时序多链图神经网络）

### 6.1 架构概述

TM-GNN 是本论文的核心算法贡献，包含三个关键模块：

模块 A：图注意力层
- 执行带注意力权重的邻域聚合
- 学习哪些交易对手地址与欺诈预测最相关
- 注意力权重作为解释："哪些邻居重要"

模块 B：时序注意力层
- 输入多个时间窗口的图快照（1h、6h、24h、7d）
- 学习哪些时间段最能指示欺诈
- 注意力权重解释："哪个时间窗口最可疑"

模块 C：跨维度融合模块
- 融合图结构信息、时序动态和节点特征
- 使用自适应注意力加权不同信息源
- 输出最终节点分类（欺诈概率）和风险评分

### 6.2 创新点

1. 跨链感知：不同于只能看到单链图的 GraphSAGE/GAT，TM-GNN 将跨链桥操作建模为异构边，执行桥接感知的消息传递
2. 时序建模：不同于静态 GNN，TM-GNN 通过图快照序列上的注意力机制捕捉欺诈网络的时序演化
3. 双层可解释性：图注意力解释"哪些邻居重要"；时序注意力解释"哪个时间窗口重要"，共同产出合规级推理链
4. 跨链对比学习：利用多链对应关系进行半监督学习

---

## 7. 四维特征工程体系

### 7.1 交易特征

| 特征 | 计算方式 | 业务含义 |
|------|---------|---------|
| tx_amount_mean | 窗口内金额均值 | 平均交易金额，异常值可能指示欺诈 |
| tx_amount_std | 窗口内金额标准差 | 金额波动性，突然变化可能是可疑行为 |
| tx_frequency | 交易数/时间窗口 | 交易频率，高频可能是自动化洗钱 |
| tx_counterparty_entropy | 交易对手分布熵 | 交易对手多样性，过低可能是循环交易 |
| tx_gas_anomaly | (gas_price - 中位数) / MAD | Gas 价格异常度 |

### 7.2 钱包特征

| 特征 | 计算方式 | 业务含义 |
|------|---------|---------|
| wallet_age | 当前时间 - 首笔交易时间 | 钱包年龄，新钱包大额交易更可疑 |
| wallet_balance_velocity | 余额变化率 | 余额变化速率 |
| wallet_in_out_ratio | 流入总额/流出总额 | 资金流入/流出比 |
| wallet_hhi | 来源 Herfindahl 指数 | 资金来源集中度 |
| wallet_active_days | 活跃天数 | 活跃天数 |

### 7.3 图结构特征

| 特征 | 计算方式 | 业务含义 |
|------|---------|---------|
| node_in_degree | 入边数 | 接收交易数 |
| node_out_degree | 出边数 | 发出交易数 |
| node_pagerank | PageRank 算法 | 节点重要性/中心性 |
| node_clustering_coeff | 聚类系数公式 | 高值可能指示欺诈团伙 |
| node_fraud_neighbor_ratio | 欺诈邻居数/度 | 与已知欺诈节点的关联程度 |
| node_bridge_usage_count | 桥接交易数 | 跨链桥使用频率 |

### 7.4 时序特征

| 特征 | 计算方式 | 业务含义 |
|------|---------|---------|
| temp_burst_score | Kleinberg 突发检测 | 交易突发性 |
| temp_periodicity | 自相关分析 | 周期性模式（机器人交易特征） |
| temp_trend_slope | 线性回归斜率 | 金额趋势 |
| temp_window_stats | 滚动窗口聚合 | 多窗口统计（1h/6h/24h/7d） |
| temp_cross_chain_delay | 桥接操作时间间隔 | 跨链转移时间模式 |

---

## 8. 实验设计框架

### 8.1 实验矩阵

| 编号 | 实验 | 目的 | 对比方法 | 指标 | 数据集 | 验证假设 |
|------|------|------|---------|------|--------|---------|
| E1 | 主对比实验 | 验证 TM-GNN 整体优越性 | LR, RF, XGBoost, IF, GCN, GraphSAGE, GAT, EvolveGCN, TM-GNN | AUC, F1, Precision, Recall | 全部 3 个数据集 | H2 |
| E2 | 特征消融 | 验证四维特征各自贡献 | 完整 vs 去掉各维度 | AUC 变化量 | Elliptic | H1 |
| E3 | 架构消融 | 验证各组件贡献 | 无时序注意力/无图注意力/无融合/无桥接边 | AUC, F1 | Elliptic | H2 |
| E4 | 模型融合 | 验证融合增益 | 单模型 vs 两两 vs 全融合 | F1 提升量 | 全部 3 个数据集 | H4 |
| E5 | 可解释性 | 验证解释质量和性能影响 | SHAP + 注意力可视化 + Fidelity | Fidelity, Sparsity, AUC diff | Elliptic | H3, H5 |
| E6 | 效率 | 验证实时推理可行性 | 各模型推理延迟和吞吐 | P50/P99 延迟, TPS | 自采集 | H6 |
| E7 | 鲁棒性 | 验证对抗稳定性 | 添加噪声边/节点、特征扰动 | AUC 下降率 | Elliptic | 额外 |
| E8 | 案例研究 | 定性分析检测能力 | 选取典型欺诈案例 | 定性评价 | Elliptic+自采集 | H3 |

### 8.2 Baseline 方法

| 类别 | 方法 | 参考论文 | 选择原因 |
|------|------|---------|---------|
| 传统 ML | Logistic Regression | - | 最基础线性 baseline |
| 传统 ML | Random Forest | - | 经典非线性 baseline |
| 传统 ML | XGBoost | Chen & Guestrin, 2016 | 工业界强 baseline |
| 异常检测 | Isolation Forest | Liu et al., 2008 | 经典无监督异常检测 |
| 异常检测 | Autoencoder | - | 深度异常检测 baseline |
| GNN | GCN | Kipf & Welling, 2017 | 最基础 GNN |
| GNN | GraphSAGE | Hamilton et al., 2017 | 归纳式 GNN |
| GNN | GAT | Velickovic et al., 2018 | 注意力 GNN baseline |
| 时序 GNN | EvolveGCN | Pareja et al., 2020 | 时序 GNN baseline |
| 本文 | TM-GNN | 本文提出 | 时序多链图神经网络 |

---

## 9. 数据策略

### 9.1 数据集规划

| 数据集 | 来源 | 规模 | 标注情况 | 用途 | 优先级 |
|--------|------|------|---------|------|--------|
| Elliptic | Kaggle | 203,769 笔 BTC 交易 | 4,545 非法/42,019 合法/157,205 未标注 | 主数据集 #1 | P0 |
| Ethereum Fraud Detection | Kaggle | 9,841 个 ETH 地址 | 2,179 欺诈/7,662 正常 | 主数据集 #2 | P0 |
| Bitcoin Alpha + OTC | Stanford SNAP | 24,186 条信任边 | 信任评分 | 辅助数据集 | P1 |
| 自采集多链数据 | Etherscan+BSCScan+PolygonScan | 目标 100 万+ 笔交易 | 半标注 | 跨链实验 | P1 |
| ERC-20 Token Transfers | Google BigQuery | 数亿级 | 无标注 | 大规模图构建 | P2 |
| OFAC SDN List | 美国财政部 | ~12,000 实体 | 制裁实体 | 合规模块验证 | P1 |

### 9.2 数据处理流水线

步骤 1：采集 -> 从 API 和公开数据集获取原始数据
步骤 2：预处理 -> 去重、缺失值处理、地址标准化、时间戳对齐
步骤 3：图构建 -> 节点（地址）、边（交易）、时序快照（按小时/天/周）
步骤 4：特征计算 -> 为每个节点计算四维特征
步骤 5：数据划分 -> 按时间顺序划分（防止数据泄露）：训练 60% / 验证 20% / 测试 20%

关键：数据必须按时间顺序划分，不能随机划分，否则会造成时序数据泄露。

---

## 10. 项目管理框架

### 10.1 总体时间线（20 周 = 5 个月）

第一阶段：基础建设（第 1-3 周）
第二阶段：特征与 Baseline（第 4-7 周）
第三阶段：核心模型开发（第 8-12 周）-- 项目核心
第四阶段：系统与高级实验（第 13-15 周）
第五阶段：论文与答辩（第 16-20 周）

### 10.2 每周详细计划

第 1 周：
- 工程师 A：GitHub 仓库 + CI/CD + Docker 环境搭建
- 工程师 B：Elliptic 数据集下载 + EDA + 数据 Schema 定义
- 工程师 C（PhD/你）：系统性文献综述（30+ 篇）+ SOTA 方法整理
- 工程师 D：PostgreSQL + Neo4j + TimescaleDB + Redis 部署
- 工程师 E：React 项目骨架 + UI 原型设计

第 2 周：
- 工程师 A：ETH/BTC/BSC 链上数据采集器 + Kafka 搭建
- 工程师 B：数据清洗管道 + 格式标准化
- 工程师 C（PhD/你）：Research Gap 分析 + RQ/假设定义 + 实验设计
- 工程师 D：数据库 Schema + Migration + 基础 CRUD API
- 工程师 E：仪表盘基础布局 + WebSocket 通信

第 3 周：
- 工程师 A：数据管道联调 + 监控
- 工程师 B：Elliptic 图构建 + 时序快照切分 + 数据划分
- 工程师 C（PhD/你）：Baseline 代码准备 + Elliptic 初步运行
- 工程师 D：Neo4j 图数据导入 + Cypher 查询 API
- 工程师 E：实时数据展示组件

第 4-5 周：
- 工程师 A：Feature Store (Feast) 搭建
- 工程师 B：交易特征 + 钱包特征实现
- 工程师 C（PhD/你）：图特征 + 时序特征实现 + 特征重要性分析
- 工程师 D：特征 API + Redis 缓存层
- 工程师 E：特征分布可视化

第 6-7 周：
- 工程师 A：MLflow 实验跟踪 + 模型训练 Pipeline
- 工程师 B：XGBoost 风险评分 + SHAP + 调参
- 工程师 C（PhD/你）：6 个 Baseline 完整实验 + 交叉验证 + 统计检验
- 工程师 D：模型推理 API 框架
- 工程师 E：模型对比结果可视化

第 8-9 周：
- 工程师 A：GPU 训练环境优化
- 工程师 B：GCN/GraphSAGE/GAT 实现和训练
- 工程师 C（PhD/你）：TM-GNN 架构设计 + 时序注意力 + 图注意力实现
- 工程师 D：图数据 API 优化
- 工程师 E：交易图谱可视化（D3.js）

第 10 周：
- 工程师 A：实验结果自动收集脚本
- 工程师 B：EvolveGCN baseline + 时序 GNN 对比
- 工程师 C（PhD/你）：TM-GNN 融合模块 + 完整模型训练调优
- 工程师 D：规则引擎 DSL 设计
- 工程师 E：注意力权重可视化组件

第 11-12 周：
- 工程师 A：ETH Fraud 数据集管道 + 跨数据集实验自动化
- 工程师 B：特征消融实验 E2
- 工程师 C（PhD/你）：主对比实验 E1 + 架构消融 E3 + 统计显著性检验
- 工程师 D：综合评分引擎 + 融合 API
- 工程师 E：实验结果 Dashboard

第 13-14 周：
- 工程师 A：实时评分管道 + 性能压测
- 工程师 B：模型融合实验 E4 + 评分校准
- 工程师 C（PhD/你）：可解释性实验 E5 + 案例研究 E8
- 工程师 D：告警系统 + SAR + 案件管理
- 工程师 E：告警中心 + 分析师工作台 UI

第 15 周：
- 工程师 A：性能优化（ONNX）+ 效率实验 E6
- 工程师 B：鲁棒性实验 E7
- 工程师 C（PhD/你）：论文 Introduction + Methodology 初稿
- 工程师 D：安全加固 + 审计日志
- 工程师 E：仪表盘功能完善

第 16-17 周：
- 工程师 A：K8s 部署 + 监控
- 工程师 B：补充实验 + 论文图表
- 工程师 C（PhD/你）：论文 Experiments + Results + Discussion
- 工程师 D：Runbook + API 文档
- 工程师 E：演示视频 + 答辩 PPT 素材

第 18-19 周：
- 工程师 A：系统稳定性测试
- 工程师 B：论文审阅 + 可复现性验证
- 工程师 C（PhD/你）：论文完整初稿 + 导师审阅 + 修改
- 工程师 D：集成测试 + Bug 修复
- 工程师 E：答辩 PPT（40-50 页）+ Demo 环境

第 20 周：
- 工程师 A：最终部署 + 文档归档
- 工程师 B：开源代码整理
- 工程师 C（PhD/你）：答辩演练 x3 + 论文终稿 + 投稿准备
- 工程师 D：代码归档
- 工程师 E：答辩现场技术支持

---

## 11. 五名工程师分工方案

### 工程师 A：数据基础设施工程师 / DevOps
核心职责：数据采集管道、Kafka/Flink/Airflow 基础设施、Feature Store、CI/CD、Docker、K8s 部署、监控
关键技能：Python, Docker, Kubernetes, Kafka, Flink/Spark, Airflow, Web3.py

### 工程师 B：数据科学家 / 特征工程师
核心职责：特征工程（交易+钱包特征）、XGBoost 风险评分、数据处理管道、模型融合评分、数据质量
关键技能：Python, pandas, scikit-learn, XGBoost, Spark, 统计学

### 工程师 C：ML 研究工程师（PhD/你/项目负责人）
核心职责：TM-GNN 模型设计与实现、异常检测模型、GNN 模型、实验设计与执行、论文写作、可解释性分析
关键技能：PyTorch, PyTorch Geometric, 深度学习理论, 学术写作, 实验设计

### 工程师 D：后端开发工程师
核心职责：REST API/gRPC 服务、规则引擎、合规模块（SAR/制裁名单/案件管理）、数据库、安全
关键技能：Python (FastAPI), SQL, PostgreSQL, Neo4j, Redis, API 设计

### 工程师 E：前端开发工程师
核心职责：风控仪表盘（React）、数据可视化（图谱/图表）、分析师工作台、规则管理/告警中心 UI
关键技能：React, TypeScript, D3.js, WebSocket, UI/UX

### 协作矩阵

| 任务 | 工程师A | 工程师B | 工程师C | 工程师D | 工程师E |
|------|---------|---------|---------|---------|---------|
| 数据管道 | 主导 | 协助 | - | - | - |
| 特征工程 | 协助 | 主导 | 协助 | - | - |
| TM-GNN 模型 | - | - | 主导 | - | - |
| Baseline 模型 | - | 协助 | 主导 | - | - |
| 风险评分 | - | 主导 | 审核 | 协助 | - |
| 规则引擎 | - | - | - | 主导 | 协助 |
| 合规模块 | 协助 | 协助 | - | 主导 | 协助 |
| 仪表盘 | - | - | - | 协助 | 主导 |
| 部署 | 主导 | 协助 | - | 协助 | - |
| 论文 | - | 协助 | 主导 | - | - |

---

## 12. 完整技术栈

### 12.1 编程语言
Python 3.11+（ML/数据/API）、TypeScript（前端）、SQL/Cypher（数据查询）、LaTeX（论文）

### 12.2 机器学习与数据科学
PyTorch 2.0+、PyTorch Geometric (PyG)、scikit-learn、XGBoost、SHAP、Optuna、imbalanced-learn、pandas/polars、NumPy/SciPy、NetworkX、matplotlib/seaborn

### 12.3 MLOps
MLflow、DVC、Weights & Biases（可选）

### 12.4 区块链与数据采集
Web3.py、Etherscan API、BSCScan API、PolygonScan API、Alchemy、CoinGecko API

### 12.5 数据处理与流计算
Apache Kafka、Apache Flink、Apache Spark、Apache Airflow、Feast

### 12.6 数据库与存储
PostgreSQL、Neo4j、TimescaleDB、Redis、MinIO/S3

### 12.7 后端与 API
FastAPI、Pydantic、SQLAlchemy + Alembic、Celery、gRPC + Protobuf

### 12.8 前端与可视化
React 18+、TypeScript、D3.js、ECharts、Ant Design

### 12.9 基础设施与 DevOps
Docker + Docker Compose、Kubernetes + Helm、GitHub Actions、Prometheus + Grafana、ELK Stack

### 12.10 测试与代码质量
pytest + pytest-cov、Locust、ruff + mypy、pre-commit

### 12.11 学术工具
Overleaf/LaTeX、Zotero、Jupyter Notebook、Google Scholar Alerts

---

## 13. 关键成功指标（KPIs）

### 模型性能
| 指标 | 目标值 |
|------|--------|
| AUC-ROC | > 0.94 |
| Precision | > 0.80 |
| Recall | > 0.85 |
| F1 Score | > 0.87 |
| TM-GNN vs 最优 Baseline AUC 提升 | > 5% |
| 多模型融合 vs 单模型 F1 提升 | > 10% |

### 系统性能
| 指标 | 目标值 |
|------|--------|
| 实时评分 P99 延迟 | < 100ms |
| 数据处理吞吐 | > 10,000 TPS |
| 系统可用性 | > 99.9% |

### 学术产出
| 指标 | 目标值 |
|------|--------|
| 论文投稿 | >= 1 篇（目标：KDD/WWW/Financial Crypto）|
| 开源仓库 | 完整代码+文档+可复现性 |
| 测试覆盖率 | > 80% |

---

## 14. 风险管理

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 标注数据不足 | 高 | 高 | 半监督/无监督方法；Elliptic 预训练；合成数据增强 |
| 误报率过高 | 高 | 中 | 人工反馈闭环；分级告警；阈值校准 |
| 实时性能不达标 | 高 | 中 | ONNX 轻量化；特征预计算；多级缓存 |
| 合规要求变更 | 中 | 高 | 可配置规则引擎；预留扩展接口 |
| 链上 API 不稳定 | 中 | 中 | 多源冗余；重试+降级策略 |
| 对抗性攻击 | 高 | 中 | 月度再训练；对抗训练；规则+模型双引擎 |
| 跨链数据对齐难度 | 中 | 中 | 桥接交易追踪；启发式地址匹配；渐进改进 |

---

## 15. 参考文献与数据集

### 核心参考文献
1. Weber, M., et al. (2019). "Anti-Money Laundering in Bitcoin." KDD Workshop.
2. Rossi, E., et al. (2020). "Temporal Graph Networks." ICML Workshop.
3. Hamilton, W., et al. (2017). "Inductive Representation Learning on Large Graphs." NeurIPS.
4. Velickovic, P., et al. (2018). "Graph Attention Networks." ICLR.
5. Liu, F.T., et al. (2008). "Isolation Forest." ICDM.
6. Lundberg, S. & Lee, S. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
7. Pareja, A., et al. (2020). "EvolveGCN." AAAI.
8. Kipf, T. & Welling, M. (2017). "Semi-Supervised Classification with GCN." ICLR.
9. Chen, T. & Guestrin, C. (2016). "XGBoost." KDD.
10. Alarab, I., et al. (2020). "GCN for Anti-Money Laundering in Bitcoin." MLCS.

### 公开数据集
| 数据集 | 来源 | 内容 | 用途 |
|--------|------|------|------|
| Elliptic | Kaggle | 203,769 笔 BTC 交易 | 核心训练和评估 |
| Ethereum Fraud Detection | Kaggle | ETH 欺诈地址 | 补充训练 |
| Bitcoin Alpha + OTC | Stanford SNAP | BTC 信任网络 | 图分析 |
| ERC-20 Token Transfers | Google BigQuery | ETH Token 转账 | 大规模图构建 |
| OFAC SDN List | 美国财政部 | 制裁名单 | 合规筛查 |

---

文档版本: v2.0
创建日期: 2026-03-18
作者: PhD ML Research Engineer
项目: ChainGuard
