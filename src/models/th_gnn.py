"""
M4: TH-GNN (Temporal Heterogeneous Graph Neural Network)

The complete model combining:
- R-GCN with heterogeneous edge types (from M3)
- Temporal self-attention over graph snapshots (from M2)

This is the main proposed model of the ChainGuard paper.

Architecture:
    Input features
    → R-GCN Layer 1 (separate weights per edge type) → ReLU → Dropout
    → R-GCN Layer 2 (separate weights per edge type) → ReLU → Dropout
    → Temporal Attention (pool per timestep → self-attention → gate back)
    → Linear Classifier → Sigmoid

Ablation positioning:
    M1: GCN                         (no temporal, no heterogeneous)
    M2: GCN + Temporal              (temporal only)
    M3: R-GCN + Heterogeneous       (heterogeneous only)
    M4: R-GCN + Temporal + Hetero   (both → this model)
    M5: M4 + Cross-chain propagation
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from src.models.modules.temporal_attention import TemporalAttentionLayer


class THGNN(torch.nn.Module):
    """
    Temporal Heterogeneous Graph Neural Network (M4 in ablation study).

    Combines R-GCN (heterogeneous edges) with temporal self-attention.

    Args:
        in_channels: Number of input features per node.
        hidden_channels: Hidden dimension.
        num_relations: Number of edge types.
        num_heads: Number of attention heads for temporal attention.
        dropout: Dropout rate for R-GCN layers.
        attn_dropout: Dropout rate for attention weights.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 80,
        num_relations: int = 2,
        num_heads: int = 4,
        dropout: float = 0.5,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        # R-GCN layers (from M3: separate weights per edge type)
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
        self.dropout = dropout

        # Temporal attention (from M2)
        self.temporal_attention = TemporalAttentionLayer(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            dropout=attn_dropout,
            max_timesteps=100,
        )

        # Classifier
        self.classifier = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_type, timesteps):
        """
        Args:
            x: [N, in_channels] node features.
            edge_index: [2, E] edge indices.
            edge_type: [E] edge type for each edge.
            timesteps: [N] timestep index for each node.

        Returns:
            logits: [N] raw logits for binary classification.
        """
        # R-GCN encoding with heterogeneous edges (M3 component)
        h = self.conv1(x, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Temporal attention (M2 component)
        h, self._attn_weights = self.temporal_attention(h, timesteps)

        # Classification
        out = self.classifier(h).squeeze(-1)
        return out

    def get_attention_weights(self):
        """Return the last computed temporal attention weights."""
        return self._attn_weights
