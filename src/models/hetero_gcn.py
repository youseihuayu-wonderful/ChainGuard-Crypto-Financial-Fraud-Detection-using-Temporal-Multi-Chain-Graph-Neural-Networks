"""
M3: R-GCN with Heterogeneous Edge Types

Extends M1 (plain GCN) by using separate weight matrices for different
edge types via R-GCN (Schlichtkrull et al., ESWC 2018).

The Elliptic dataset has isolated timestep subgraphs (0 cross-timestep edges).
We augment the graph with temporal k-NN edges between adjacent timesteps:
    - Type 0 (original): Payment flow edges within a timestep
    - Type 1 (temporal): Feature-similarity edges across adjacent timesteps

R-GCN learns separate weight matrices for each type, testing whether
distinguishing edge types improves fraud detection.

Difference from M1:
    M1: Single GCN weight matrix, original edges only
    M3: R-GCN with per-type weights, augmented graph with temporal edges
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class HeteroGCN(torch.nn.Module):
    """
    R-GCN with heterogeneous edge types (M3 in ablation study).

    Args:
        in_channels: Number of input features per node.
        hidden_channels: Hidden dimension (adjusted for parameter parity with M1).
        num_relations: Number of edge types (2: original + temporal).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 80,
        num_relations: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = RGCNConv(
            in_channels, hidden_channels,
            num_relations=num_relations,
            num_bases=None,
        )
        self.conv2 = RGCNConv(
            hidden_channels, hidden_channels,
            num_relations=num_relations,
            num_bases=None,
        )
        self.classifier = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x: [N, in_channels] node features.
            edge_index: [2, E] edge indices (original + temporal).
            edge_type: [E] edge type (0=original, 1=temporal).

        Returns:
            logits: [N] raw logits for binary classification.
        """
        h = self.conv1(x, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.classifier(h).squeeze(-1)
        return out
