"""
Graph Builder for Multi-Chain Fraud Detection.

Strictly fuses separate blockchain datasets and cross-chain bridge events
into a PyTorch Geometric `HeteroData` object. This creates the true multigraph
described in the ChainGuard architecture, completely replacing the artificial
single-chain "temporal edges" simulation.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List

from src.data.bridge_parser import BridgeParser


class MultiChainGraphBuilder:
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: The root directory containing chain subfolders (e.g., Bitcoin/ Ethereum/)
                      and `cross_chain_bridges.csv`.
        """
        self.data_dir = data_dir
        self.bridge_csv = os.path.join(data_dir, "cross_chain_bridges.csv")
        self.hetero_data = HeteroData()

        # Local caches for tx_id -> node index mappings
        self.chain_tx_maps: Dict[str, Dict[str, int]] = {}

    def _load_single_chain(self, chain_name: str):
        """Loads a single blockchain's intra-chain data."""
        print(f"[GraphBuilder] Loading chain topology: {chain_name}...")
        chain_path = os.path.join(self.data_dir, chain_name)

        features_path = os.path.join(chain_path, "features.csv")
        edges_path = os.path.join(chain_path, "edgelist.csv")
        classes_path = os.path.join(chain_path, "classes.csv")

        # Load Features
        features_df = pd.read_csv(features_path, header=None)
        tx_ids = features_df.iloc[:, 0].astype(str).values
        # Note: timestep in column 1, features in columns [2:]
        timesteps = features_df.iloc[:, 1].values.astype(int)
        features = features_df.iloc[:, 2:].values.astype(np.float32)

        # Build mapping
        tx_map = {tx: idx for idx, tx in enumerate(tx_ids)}
        self.chain_tx_maps[chain_name] = tx_map

        # Map Labels
        classes_df = pd.read_csv(classes_path)
        label_map_dict = {"1": 1, "2": 0, "unknown": -1}
        labels = np.full(len(tx_ids), -1, dtype=np.int64)

        for _, row in classes_df.iterrows():
            tx = str(row["txId"])
            if tx in tx_map:
                labels[tx_map[tx]] = label_map_dict.get(str(row["class"]), -1)

        # Map Intra-Chain Edges
        edges_df = pd.read_csv(edges_path)
        edge_list = []
        for _, row in edges_df.iterrows():
            src = str(row["txId1"])
            dst = str(row["txId2"])
            if src in tx_map and dst in tx_map:
                edge_list.append([tx_map[src], tx_map[dst]])

        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Update HeteroData
        self.hetero_data[chain_name].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data[chain_name].y = torch.tensor(labels, dtype=torch.long)
        self.hetero_data[chain_name].timestep = torch.tensor(timesteps, dtype=torch.long)
        self.hetero_data[chain_name].num_nodes = len(tx_ids)

        self.hetero_data[chain_name, 'intra_chain', chain_name].edge_index = edge_index

    def build(self, active_chains: List[str] = ["Bitcoin", "Ethereum"]) -> HeteroData:
        """
        Constructs the entire cross-chain multigraph.
        """
        print("\n" + "=" * 50)
        print("Constructing True Multi-Chain HeteroData Object")
        print("=" * 50)

        # 1. Load each isolated blockchain subgraph
        for chain in active_chains:
            self._load_single_chain(chain)

        # 2. Inject Cross-Chain Bridge Edges
        print("\n[GraphBuilder] Fusing subgraphs with cross-chain bridge edges...")
        bridge_parser = BridgeParser(self.bridge_csv)

        for src_chain in active_chains:
            for dst_chain in active_chains:
                if src_chain == dst_chain:
                    continue  # We handled intra_chain already
                
                edge_idx, edge_attr = bridge_parser.get_bridge_edges(
                    src_chain=src_chain,
                    dst_chain=dst_chain,
                    src_tx_to_idx=self.chain_tx_maps[src_chain],
                    dst_tx_to_idx=self.chain_tx_maps[dst_chain]
                )

                if edge_idx.shape[1] > 0:
                    edge_type_name = ('bridge', f"to_{dst_chain}")
                    self.hetero_data[src_chain, 'bridge', dst_chain].edge_index = edge_idx
                    self.hetero_data[src_chain, 'bridge', dst_chain].edge_attr = edge_attr

        # Final structural validation
        print("\n[GraphBuilder] Validation: HeteroData schema generated successfully!")
        print(self.hetero_data.metadata())
        
        return self.hetero_data

if __name__ == "__main__":
    mock_dir = os.path.join(os.path.dirname(__file__), "../../data/processed/xchain_mock")
    builder = MultiChainGraphBuilder(mock_dir)
    try:
        data = builder.build()
        print("\nHeteroData Object:\n", data)
    except Exception as e:
        print(f"Build failed (likely because you haven't run generate_mock_xchain.py yet): {e}")
