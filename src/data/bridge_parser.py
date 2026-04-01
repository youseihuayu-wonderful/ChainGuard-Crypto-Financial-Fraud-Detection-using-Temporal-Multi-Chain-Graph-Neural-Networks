"""
Bridge Data Parser for True Cross-Chain Heterogeneous Graphs

This module explicitly parses actual cross-chain event logs (e.g., from XChainDataGen
or BridgeGuard) representing funds moving across decentralized bridge protocols.

Unlike the previous "cheating" method in the paper repo where temporal edges
on the same chain were pretended to be cross-chain edges, this parser ensures
strict mapping between the source chain's transaction/address space and the
destination chain's transaction/address space.

Typical Bridge Event Schema (XChainDataGen format):
- source_chain: e.g., 'Bitcoin', 'Ethereum', 'BSC'
- source_tx_id: Transaction ID on the source chain
- dest_chain: e.g., 'Bitcoin', 'Ethereum', 'BSC'
- dest_tx_id: Transaction ID on the destination chain
- bridge_protocol: e.g., 'AnySwap', 'Ronin', 'Synapse'
- amount: Token amount transferred (normalized)
- timestamp: Unix timestamp of the bridge event
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple


class BridgeParser:
    """
    Parses real cross-chain bridge transactions and converts them
    to edge indices and edge attributes for PyTorch Geometric HeteroData.
    """
    def __init__(self, bridge_csv_path: str):
        """
        Args:
            bridge_csv_path: Path to the bridge events CSV.
        """
        print(f"[BridgeParser] Loading real bridge data from {bridge_csv_path}...")
        self.bridge_df = pd.read_csv(bridge_csv_path)
        required_cols = {'source_chain', 'source_tx_id', 'dest_chain', 'dest_tx_id'}
        if not required_cols.issubset(self.bridge_df.columns):
            raise ValueError(f"Bridge CSV must contain columns: {required_cols}")

    def get_bridge_edges(
        self,
        src_chain: str,
        dst_chain: str,
        src_tx_to_idx: Dict[str, int],
        dst_tx_to_idx: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract directed cross-chain edges from src_chain to dst_chain.

        Args:
            src_chain: Name of the source chain (e.g., 'BTC')
            dst_chain: Name of the destination chain (e.g., 'ETH')
            src_tx_to_idx: Mapping of source chain txIDs to PyG node indices.
            dst_tx_to_idx: Mapping of dest chain txIDs to PyG node indices.

        Returns:
            edge_index: [2, num_bridge_edges] torch tensor
            edge_attr: [num_bridge_edges, num_features] torch tensor (amount, timestamp)
        """
        # Filter for the specific directed cross-chain relation
        mask = (self.bridge_df['source_chain'] == src_chain) & (self.bridge_df['dest_chain'] == dst_chain)
        subset_df = self.bridge_df[mask]

        edge_list = []
        attr_list = []

        valid_edges = 0
        missing_nodes = 0

        for _, row in subset_df.iterrows():
            src_tx = str(row['source_tx_id'])
            dst_tx = str(row['dest_tx_id'])

            # Only add the edge if BOTH transactions exist in our loaded chain subgraphs
            if src_tx in src_tx_to_idx and dst_tx in dst_tx_to_idx:
                src_idx = src_tx_to_idx[src_tx]
                dst_idx = dst_tx_to_idx[dst_tx]
                edge_list.append([src_idx, dst_idx])

                # Optional: Extract bridge features (amount, timestamp)
                amt = float(row.get('amount', 0.0))
                ts = float(row.get('timestamp', 0.0))
                attr_list.append([amt, ts])
                valid_edges += 1
            else:
                missing_nodes += 1

        print(f"[BridgeParser] {src_chain}->{dst_chain}: Extracted {valid_edges} valid bridge edges "
              f"({missing_nodes} ignored due to missing tx nodes in subgraphs).")

        if valid_edges == 0:
            # Return empty tensors of correct shape
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 2), dtype=torch.float)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attr_list, dtype=torch.float)

        return edge_index, edge_attr

    def get_supported_chains(self):
        """Returns the unique chains present in the parsed bridge dataset."""
        chains = set(self.bridge_df['source_chain'].unique()).union(
            set(self.bridge_df['dest_chain'].unique())
        )
        return list(chains)
