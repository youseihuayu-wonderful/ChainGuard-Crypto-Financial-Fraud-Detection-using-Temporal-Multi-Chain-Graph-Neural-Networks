"""
M2: GCN + Temporal Attention

Extends M1 (plain GCN) by adding temporal self-attention over graph
snapshots. This model tests whether temporal dynamics improve fraud detection.

Architecture:
    Input features → GCN Layer 1 → ReLU → Dropout
    → GCN Layer 2 → ReLU → Dropout
    → Temporal Attention (pool per timestep → self-attention → gate back)
    → Linear Classifier → Sigmoid

Difference from M1:
    M1: GCN → classify (ignores temporal ordering)
    M2: GCN → temporal attention → classify (leverages temporal patterns)

The temporal attention module:
1. Pools GCN node embeddings by timestep (mean pooling)
2. Applies causal self-attention over the timestep sequence
3. Uses a learnable gate to blend temporal context into node embeddings
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.models.modules.temporal_attention import TemporalAttentionLayer


class TemporalGCN(torch.nn.Module):
    """
    GCN with Temporal Self-Attention (M2 in ablation study).

    Args:
        in_channels: Number of input features per node.
        hidden_channels: Hidden dimension for GCN and attention.
        num_heads: Number of attention heads for temporal attention.
        dropout: Dropout rate for GCN layers.
        attn_dropout: Dropout rate for attention weights.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 4,
        dropout: float = 0.5,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        # GCN layers (same architecture as M1 for fair comparison)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout

        # Temporal attention (the M2 addition)
        self.temporal_attention = TemporalAttentionLayer(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            dropout=attn_dropout,
            max_timesteps=100,  # Elliptic has 49 timesteps
        )

        # Classifier (same as M1)
        self.classifier = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, timesteps):
        """
        Args:
            x: [N, in_channels] node features.
            edge_index: [2, E] edge indices.
            timesteps: [N] timestep index for each node.

        Returns:
            logits: [N] raw logits for binary classification.
        """
        # GCN encoding (identical to M1)
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Temporal attention (M2 addition)
        h, self._attn_weights = self.temporal_attention(h, timesteps)

        # Classification
        out = self.classifier(h).squeeze(-1)
        return out

    def get_attention_weights(self):
        """Return the last computed temporal attention weights for analysis."""
        return self._attn_weights
