"""
Elliptic Bitcoin Dataset Loader

Loads the Elliptic dataset from CSV files and converts to PyTorch Geometric format.
Supports temporal splitting (train on earlier timesteps, test on later ones).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def load_elliptic_csv(data_dir: str) -> Data:
    """
    Load Elliptic dataset from CSV files into a PyG Data object.

    Args:
        data_dir: Path to directory containing the 3 CSV files.

    Returns:
        PyG Data object with:
            - x: node features [num_nodes, 166]
            - edge_index: edge list [2, num_edges]
            - y: labels (0=illicit, 1=licit, -1=unknown)
            - timestep: timestep for each node [num_nodes]
    """
    features_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    edges_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")

    # Load features: first column is txId, second is timestep, rest are features
    print("Loading features...")
    features_df = pd.read_csv(features_path, header=None)
    tx_ids = features_df.iloc[:, 0].values
    timesteps = features_df.iloc[:, 1].values.astype(int)
    features = features_df.iloc[:, 2:].values.astype(np.float32)

    # Create txId -> index mapping
    tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

    # Load labels
    print("Loading labels...")
    classes_df = pd.read_csv(classes_path)
    label_map = {"1": 0, "2": 1, "unknown": -1}  # 1=illicit->0, 2=licit->1
    labels = np.full(len(tx_ids), -1, dtype=np.int64)
    for _, row in classes_df.iterrows():
        tx_id = row["txId"]
        if tx_id in tx_to_idx:
            labels[tx_to_idx[tx_id]] = label_map.get(str(row["class"]), -1)

    # Load edges
    print("Loading edges...")
    edges_df = pd.read_csv(edges_path)
    edge_list = []
    for _, row in edges_df.iterrows():
        src, dst = row["txId1"], row["txId2"]
        if src in tx_to_idx and dst in tx_to_idx:
            edge_list.append([tx_to_idx[src], tx_to_idx[dst]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create PyG Data object
    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
    )
    data.timestep = torch.tensor(timesteps, dtype=torch.long)
    data.num_classes = 2

    print(f"Loaded: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{(labels == 0).sum()} illicit, {(labels == 1).sum()} licit, "
          f"{(labels == -1).sum()} unknown")

    return data


def temporal_split(data: Data, train_ratio: float = 0.7) -> tuple:
    """
    Split data by timestep (NOT random) to prevent data leakage.

    Earlier timesteps -> train, later timesteps -> test.

    Args:
        data: PyG Data object with timestep attribute.
        train_ratio: Fraction of timesteps for training.

    Returns:
        (train_mask, val_mask, test_mask) as boolean tensors.
    """
    timesteps = data.timestep.numpy()
    unique_ts = sorted(np.unique(timesteps))
    num_ts = len(unique_ts)

    train_end = int(num_ts * train_ratio)
    val_end = int(num_ts * (train_ratio + 0.15))

    train_ts = set(unique_ts[:train_end])
    val_ts = set(unique_ts[train_end:val_end])
    test_ts = set(unique_ts[val_end:])

    labeled_mask = data.y >= 0  # only labeled nodes

    train_mask = torch.tensor([ts in train_ts for ts in timesteps]) & labeled_mask
    val_mask = torch.tensor([ts in val_ts for ts in timesteps]) & labeled_mask
    test_mask = torch.tensor([ts in test_ts for ts in timesteps]) & labeled_mask

    print(f"Split: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    print(f"Train timesteps: {min(train_ts)}-{max(train_ts)}, "
          f"Val: {min(val_ts)}-{max(val_ts)}, "
          f"Test: {min(test_ts)}-{max(test_ts)}")

    return train_mask, val_mask, test_mask


if __name__ == "__main__":
    # Quick test
    data_dir = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")
    data = load_elliptic_csv(data_dir)
    train_mask, val_mask, test_mask = temporal_split(data)
