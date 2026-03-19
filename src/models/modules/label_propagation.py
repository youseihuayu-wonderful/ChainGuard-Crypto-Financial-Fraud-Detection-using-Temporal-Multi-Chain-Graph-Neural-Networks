"""
Label Propagation Module for Semi-Supervised Graph Learning

Implements label propagation (LP) that spreads known labels through
the graph structure to influence predictions on unlabeled nodes.
This is the M5 addition on top of TH-GNN (M4).

In the ChainGuard context:
- Elliptic has labels on ~23% of nodes (4,545 illicit + 42,019 licit)
- 77% of nodes are unlabeled
- Temporal edges connect different timestep communities
- LP propagates label signals through these edges to reach unlabeled nodes

The module combines GNN predictions with propagated labels via:
    final_pred = alpha * gnn_pred + (1 - alpha) * lp_pred

Additionally, a consistency regularization loss encourages the GNN
to agree with label propagation on unlabeled nodes:
    L_total = L_supervised + lambda * L_consistency

References:
- Zhu & Ghahramani (2002), "Learning from Labeled and Unlabeled Data"
- Zhou et al. (2003), "Learning with Local and Global Consistency"
- Iscen et al. (2019), "Label Propagation for Deep Semi-supervised Learning"
"""

import torch
import torch.nn.functional as F


class LabelPropagation(torch.nn.Module):
    """
    Iterative label propagation on a graph.

    Propagates soft labels through the graph adjacency matrix:
        Y^(t+1) = (1 - alpha) * Y^(0) + alpha * A_norm * Y^(t)

    where A_norm is the symmetrically normalized adjacency matrix
    and Y^(0) are the initial labels (known labels + zeros for unknown).

    Args:
        num_iterations: Number of propagation iterations.
        alpha: Propagation weight (0=only initial labels, 1=only propagated).
    """

    def __init__(self, num_iterations: int = 10, alpha: float = 0.5):
        super().__init__()
        self.num_iterations = num_iterations
        self.alpha = alpha

    @torch.no_grad()
    def forward(
        self,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        labeled_mask: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Propagate labels through the graph.

        Args:
            edge_index: [2, E] edge indices.
            y: [N] node labels (1=illicit, 0=licit, -1=unknown).
            labeled_mask: [N] boolean mask for labeled nodes.
            num_nodes: Total number of nodes.

        Returns:
            soft_labels: [N] propagated soft labels (probability of illicit).
        """
        device = edge_index.device

        # Initialize soft labels from known labels
        y_init = torch.zeros(num_nodes, device=device)
        y_init[labeled_mask & (y == 1)] = 1.0  # illicit
        y_init[labeled_mask & (y == 0)] = 0.0  # licit
        # Unknown nodes start at 0.5 (uninformative prior)
        y_init[~labeled_mask] = 0.5

        # Build normalized adjacency matrix
        row, col = edge_index
        # Add self-loops
        loop_idx = torch.arange(num_nodes, device=device)
        row = torch.cat([row, loop_idx])
        col = torch.cat([col, loop_idx])

        # Compute degree for normalization
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        # Normalized edge weights: D^{-1/2} A D^{-1/2}
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Iterative propagation
        y_prop = y_init.clone()
        for _ in range(self.num_iterations):
            # Message passing: A_norm * y_prop
            out = torch.zeros(num_nodes, device=device)
            out.scatter_add_(0, row, edge_weight * y_prop[col])

            # Combine with initial labels
            y_prop = (1 - self.alpha) * y_init + self.alpha * out

            # Clamp to [0, 1]
            y_prop = y_prop.clamp(0, 1)

            # Fix known labels (don't change them)
            y_prop[labeled_mask & (y == 1)] = 1.0
            y_prop[labeled_mask & (y == 0)] = 0.0

        return y_prop


def consistency_loss(
    gnn_probs: torch.Tensor,
    lp_probs: torch.Tensor,
    unlabeled_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute consistency regularization loss between GNN predictions
    and label propagation on unlabeled nodes.

    Uses MSE loss (simpler and more stable than KL divergence for
    single-output binary predictions).

    Args:
        gnn_probs: [N] GNN predicted probabilities.
        lp_probs: [N] label propagation soft labels.
        unlabeled_mask: [N] boolean mask for unlabeled nodes.

    Returns:
        Scalar consistency loss.
    """
    if unlabeled_mask.sum() == 0:
        return torch.tensor(0.0, device=gnn_probs.device)

    gnn_unlabeled = gnn_probs[unlabeled_mask]
    lp_unlabeled = lp_probs[unlabeled_mask]

    return F.mse_loss(gnn_unlabeled, lp_unlabeled)
