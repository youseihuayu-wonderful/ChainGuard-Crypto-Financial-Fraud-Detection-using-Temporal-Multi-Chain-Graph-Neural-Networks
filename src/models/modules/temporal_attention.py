"""
Temporal Self-Attention Module for Graph Snapshots

Computes attention over graph-level timestep representations to capture
temporal dynamics. Inspired by DySAT (Sankar et al., WSDM 2020) and
Transformer temporal encoding.

Design rationale:
- In Elliptic, each node belongs to exactly ONE timestep (49 total).
- We run a shared GCN on the full graph, then pool node embeddings per
  timestep to get graph-level representations.
- Temporal self-attention learns which timestep patterns are most relevant.
- The attended temporal context is added back to each node's embedding,
  enriching it with information about temporal dynamics.

Architecture:
    GCN node embeddings → mean pool per timestep → temporal self-attention
    → context vector per timestep → add back to node embeddings → classify
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for timesteps.
    Same formulation as Vaswani et al. (2017) "Attention Is All You Need".
    """

    def __init__(self, d_model: int, max_timesteps: int = 100):
        super().__init__()
        pe = torch.zeros(max_timesteps, d_model)
        position = torch.arange(0, max_timesteps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (not a parameter, but moves with .to(device))
        self.register_buffer("pe", pe)

    def forward(self, timestep_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep_indices: [T] tensor of timestep indices (0-based).
        Returns:
            [T, d_model] positional encodings.
        """
        return self.pe[timestep_indices]


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention over a sequence of timestep representations.

    Given graph-level representations for T timesteps, computes attention
    to capture which timesteps are most informative for each other.

    Args:
        d_model: Dimension of input representations.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal: bool = True) -> tuple:
        """
        Args:
            x: [T, d_model] sequence of timestep representations.
            causal: If True, apply causal mask (each timestep can only
                    attend to itself and earlier timesteps). This prevents
                    information leakage from future timesteps.

        Returns:
            output: [T, d_model] attention-enriched representations.
            attn_weights: [num_heads, T, T] attention weight matrix.
        """
        T = x.size(0)

        # Compute Q, K, V and reshape for multi-head attention
        Q = self.W_Q(x).view(T, self.num_heads, self.d_k).transpose(0, 1)  # [H, T, d_k]
        K = self.W_K(x).view(T, self.num_heads, self.d_k).transpose(0, 1)  # [H, T, d_k]
        V = self.W_V(x).view(T, self.num_heads, self.d_k).transpose(0, 1)  # [H, T, d_k]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [H, T, T]

        # Causal mask: prevent attending to future timesteps
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # [H, T, T]
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, V)  # [H, T, d_k]
        context = context.transpose(0, 1).contiguous().view(T, self.d_model)  # [T, d_model]

        # Output projection + residual connection + layer norm
        output = self.layer_norm(x + self.W_O(context))

        return output, attn_weights.detach()


class TemporalAttentionLayer(nn.Module):
    """
    Complete temporal attention layer that:
    1. Pools node embeddings per timestep (mean pooling)
    2. Adds positional encoding for temporal ordering
    3. Applies multi-head self-attention over timesteps
    4. Broadcasts attended context back to individual nodes

    This is the key module that differentiates M2 from M1.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_timesteps: int = 100,
    ):
        super().__init__()
        self.pos_encoding = TemporalPositionalEncoding(hidden_channels, max_timesteps)
        self.self_attention = TemporalSelfAttention(hidden_channels, num_heads, dropout)
        # Learnable gate to control how much temporal context is added
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid(),
        )

    def forward(
        self, node_embeddings: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple:
        """
        Args:
            node_embeddings: [N, hidden] node embeddings from GCN.
            timesteps: [N] timestep index for each node.

        Returns:
            enriched: [N, hidden] temporally enriched node embeddings.
            attn_weights: [num_heads, T, T] temporal attention weights.
        """
        unique_ts = torch.unique(timesteps, sorted=True)
        T = len(unique_ts)

        # 1. Mean pool node embeddings per timestep → graph-level representations
        ts_reps = []
        ts_to_idx = {}
        for i, t in enumerate(unique_ts):
            mask = timesteps == t
            ts_reps.append(node_embeddings[mask].mean(dim=0))
            ts_to_idx[t.item()] = i

        ts_reps = torch.stack(ts_reps)  # [T, hidden]

        # 2. Add positional encoding (0-based index)
        pos_indices = torch.arange(T, device=node_embeddings.device)
        ts_reps = ts_reps + self.pos_encoding(pos_indices)

        # 3. Temporal self-attention (causal: no future information leakage)
        temporal_context, attn_weights = self.self_attention(ts_reps, causal=True)

        # 4. Broadcast temporal context back to each node with learnable gate
        node_ts_indices = torch.tensor(
            [ts_to_idx[t.item()] for t in timesteps],
            device=node_embeddings.device,
        )
        node_temporal_context = temporal_context[node_ts_indices]  # [N, hidden]

        # Gated addition: learn how much temporal info to incorporate
        gate_input = torch.cat([node_embeddings, node_temporal_context], dim=-1)
        gate_value = self.gate(gate_input)  # [N, hidden], values in [0, 1]
        enriched = node_embeddings + gate_value * node_temporal_context

        return enriched, attn_weights
