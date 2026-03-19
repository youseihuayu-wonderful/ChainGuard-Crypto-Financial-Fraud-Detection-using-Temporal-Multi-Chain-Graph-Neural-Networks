"""
GCN Baseline Model (M1 in ablation study)

Standard Graph Convolutional Network for node classification.
This is the primary baseline against which TH-GNN will be compared.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNBaseline(torch.nn.Module):
    """
    2-layer GCN for binary node classification.

    Architecture:
        Input (166) -> GCN (hidden) -> ReLU -> Dropout -> GCN (hidden) -> MLP -> Sigmoid
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Classification
        out = self.classifier(x).squeeze(-1)
        return out

    def predict_proba(self, x, edge_index):
        """Return probability of being illicit (class 0)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return torch.sigmoid(logits)
