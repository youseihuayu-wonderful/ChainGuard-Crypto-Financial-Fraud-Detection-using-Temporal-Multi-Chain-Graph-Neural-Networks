"""
Heterogeneous Edge Convolution Module

Implements message passing with separate weight matrices for different
edge types. This is based on R-GCN (Schlichtkrull et al., ESWC 2018):
"Modeling Relational Data with Graph Convolutional Networks"

In our ablation study:
- M3 uses 2 edge types: intra-timestep (type 0) vs cross-timestep (type 1)
- M5 extends to 3 types: native (type 0), cross-timestep (type 1), bridge (type 2)

The key hypothesis: different edge types carry different information for
fraud detection, and learning separate parameters per type captures this.

Design choices:
- Uses basis decomposition (num_bases) for parameter efficiency when
  number of edge types is small (R-GCN default)
- Supports variable number of edge types for extensibility (M3 → M5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class HeteroEdgeConv(nn.Module):
    """
    2-layer R-GCN for heterogeneous edge types.

    Uses separate weight matrices per edge type, implementing the core
    heterogeneous message passing from our TH-GNN proposal:
        h_v = W_type * AGG({h_u : u in N_type(v)})
    where type ∈ {intra-timestep, cross-timestep} for M3,
    extended to {native, cross-timestep, bridge} for M5.

    Args:
        in_channels: Number of input features.
        hidden_channels: Hidden dimension.
        num_relations: Number of edge types (2 for M3, 3 for M5).
        num_bases: Number of basis matrices for R-GCN decomposition.
                   None means no decomposition (full separate weights).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_relations: int = 2,
        num_bases: int = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = RGCNConv(
            in_channels,
            hidden_channels,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.conv2 = RGCNConv(
            hidden_channels,
            hidden_channels,
            num_relations=num_relations,
            num_bases=num_bases,
        )
        self.dropout = dropout
        self.num_relations = num_relations

    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x: [N, in_channels] node features.
            edge_index: [2, E] edge indices.
            edge_type: [E] edge type for each edge (0, 1, ..., num_relations-1).

        Returns:
            h: [N, hidden_channels] node embeddings.
        """
        h = self.conv1(x, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h


def assign_edge_types(edge_index: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """
    Assign edge types based on temporal relationship between endpoints.

    Type 0 (intra-timestep): both nodes in the same timestep.
        These represent direct payment flows within a time window.
    Type 1 (cross-timestep): nodes in different timesteps.
        These represent longer-range dependencies across time.

    Args:
        edge_index: [2, E] edge indices.
        timesteps: [N] timestep for each node.

    Returns:
        edge_type: [E] tensor of edge types (0 or 1).
    """
    src_ts = timesteps[edge_index[0]]
    dst_ts = timesteps[edge_index[1]]
    edge_type = (src_ts != dst_ts).long()
    return edge_type
