"""
Data Exploration Script for Elliptic Dataset
=============================================
Run this FIRST to understand the data before writing any model code.

Outputs:
    - Dataset statistics
    - Label distribution
    - Feature distributions
    - Temporal distribution of labels
    - Graph statistics

Usage:
    python notebooks/01_data_exploration.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/raw/elliptic_bitcoin_dataset")

print("=" * 60)
print("Elliptic Bitcoin Dataset - Data Exploration")
print("=" * 60)

# ============================================================
# 1. Load raw CSV files
# ============================================================
print("\n1. Loading CSV files...")
features_df = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_features.csv"), header=None)
classes_df = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_classes.csv"))
edges_df = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv"))

# ============================================================
# 2. Basic statistics
# ============================================================
print("\n2. Basic Statistics")
print("-" * 40)
print(f"Number of nodes (transactions): {len(features_df):,}")
print(f"Number of edges (tx links):     {len(edges_df):,}")
print(f"Number of features per node:    {features_df.shape[1] - 2}")  # minus txId and timestep
print(f"Number of timesteps:            {features_df.iloc[:, 1].nunique()}")
print(f"Timestep range:                 {features_df.iloc[:, 1].min()} - {features_df.iloc[:, 1].max()}")

# ============================================================
# 3. Label distribution
# ============================================================
print("\n3. Label Distribution")
print("-" * 40)
label_counts = classes_df["class"].value_counts()
for label, count in label_counts.items():
    pct = count / len(classes_df) * 100
    label_name = {"1": "illicit", "2": "licit", "unknown": "unknown"}.get(str(label), str(label))
    print(f"  {label_name:10s} ({label}): {count:>8,} ({pct:.1f}%)")

n_illicit = label_counts.get("1", label_counts.get(1, 0))
n_licit = label_counts.get("2", label_counts.get(2, 0))
print(f"\n  Imbalance ratio (licit/illicit): {n_licit/n_illicit:.1f}x")

# ============================================================
# 4. Temporal distribution of labels
# ============================================================
print("\n4. Temporal Distribution of Labels")
print("-" * 40)

# Merge features (for timestep) with classes
tx_ids = features_df.iloc[:, 0]
timesteps = features_df.iloc[:, 1]
temp_df = pd.DataFrame({"txId": tx_ids, "timestep": timesteps})
temp_df = temp_df.merge(classes_df, on="txId", how="left")

for ts in sorted(temp_df["timestep"].unique()):
    ts_data = temp_df[temp_df["timestep"] == ts]
    n_ill = (ts_data["class"].astype(str) == "1").sum()
    n_lic = (ts_data["class"].astype(str) == "2").sum()
    n_unk = (ts_data["class"].astype(str) == "unknown").sum()
    total = len(ts_data)
    bar = "█" * max(1, n_ill * 50 // max(total, 1))
    print(f"  ts={ts:2d}: total={total:5d} | illicit={n_ill:4d} | licit={n_lic:4d} | unknown={n_unk:5d} | {bar}")

# ============================================================
# 5. Graph statistics
# ============================================================
print("\n5. Graph Statistics")
print("-" * 40)

# Degree distribution
all_nodes = pd.concat([edges_df["txId1"], edges_df["txId2"]])
degree = all_nodes.value_counts()
print(f"  Avg degree:    {degree.mean():.2f}")
print(f"  Median degree: {degree.median():.0f}")
print(f"  Max degree:    {degree.max()}")
print(f"  Min degree:    {degree.min()}")

# Isolated nodes (not in edge list)
nodes_in_edges = set(edges_df["txId1"]) | set(edges_df["txId2"])
all_node_ids = set(features_df.iloc[:, 0])
isolated = all_node_ids - nodes_in_edges
print(f"  Isolated nodes (no edges): {len(isolated):,} ({len(isolated)/len(all_node_ids)*100:.1f}%)")

# ============================================================
# 6. Feature statistics
# ============================================================
print("\n6. Feature Statistics (first 10 features)")
print("-" * 40)
feature_cols = features_df.iloc[:, 2:12]  # first 10 features
stats = feature_cols.describe().round(4)
print(stats.to_string())

print("\n" + "=" * 60)
print("Data exploration complete.")
print("=" * 60)
print("\nKey takeaways for engineers:")
print("  1. Labels are HIGHLY imbalanced — use class weighting")
print("  2. 77% of data is unlabeled — consider semi-supervised methods")
print("  3. Split by TIMESTEP, not random — prevents data leakage")
print("  4. Check for isolated nodes before building graph")
