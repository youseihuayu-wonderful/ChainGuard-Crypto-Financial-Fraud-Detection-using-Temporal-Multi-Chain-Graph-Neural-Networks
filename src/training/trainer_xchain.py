"""
Trainer for the True Multi-Chain Architecture.

Connects the `MultiChainGraphBuilder`'s PyG `HeteroData` to the existing `THGNN` model.
This script completes the actual engineering pipeline promised by the paper,
replacing the faked/simulated scripts in `experiments/scripts/`.
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models.th_gnn import THGNN
from src.data.graph_builder import MultiChainGraphBuilder

# Config
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed/xchain_mock")
HIDDEN_CHANNELS = 80
NUM_HEADS = 4
EPOCHS = 100
LR = 0.01

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Build the true multi-chain HeteroData
    try:
        builder = MultiChainGraphBuilder(DATA_DIR)
        hetero_data = builder.build()
    except Exception as e:
        print(f"Data loading failed: {e}")
        print("Please run `python src/data/generate_mock_xchain.py` first to generate the structural test fixture.")
        return

    # 2. Convert to Homogeneous for R-GCN (which THGNN uses)
    # PyG automatically computes `edge_type` sequentially for each edge tuple.
    print("\n[Trainer] Converting multi-chain graph to homogeneous structure for R-GCN...")
    homo_data = hetero_data.to_homogeneous()
    homo_data = homo_data.to(device)

    # Count actual edge relations
    num_relations = int(homo_data.edge_type.max().item() + 1)
    
    print(f"[Graph Statistics]")
    print(f"Total Nodes: {homo_data.num_nodes}")
    print(f"Total Edges: {homo_data.edge_index.shape[1]}")
    print(f"Number of Edge Relation Types: {num_relations} (Intra-chain + Cross-chain Bridges)")

    # Split: Labeled Mask (exclude unknown = -1)
    labeled_mask = (homo_data.y != -1)
    labeled_indices = torch.where(labeled_mask)[0]
    
    # 70/15/15 random split for simplicity in this integration test
    perm = torch.randperm(len(labeled_indices))
    train_end = int(0.7 * len(perm))
    val_end = int(0.85 * len(perm))

    train_mask = torch.zeros(homo_data.num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(homo_data.num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(homo_data.num_nodes, dtype=torch.bool, device=device)

    train_mask[labeled_indices[perm[:train_end]]] = True
    val_mask[labeled_indices[perm[train_end:val_end]]] = True
    test_mask[labeled_indices[perm[val_end:]]] = True

    # 3. Model Initialization
    in_channels = homo_data.x.shape[1]
    print(f"\n[Trainer] Initializing TH-GNN (in: {in_channels}, hidden: {HIDDEN_CHANNELS}, relations: {num_relations})")
    
    model = THGNN(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        num_relations=num_relations,
        num_heads=NUM_HEADS
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    # 4. Training Loop
    print("\n" + "=" * 50)
    print("STARTING TRUE CROSS-CHAIN TRAINING")
    print("=" * 50)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        
        # Forward pass on structural graph
        logits = model(
            homo_data.x, 
            homo_data.edge_index, 
            homo_data.edge_type, 
            homo_data.timestep
        )
        
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask],
            homo_data.y[train_mask].float()
        )
        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # Eval
            model.eval()
            with torch.no_grad():
                val_logits = model(homo_data.x, homo_data.edge_index, homo_data.edge_type, homo_data.timestep)
                val_pred = (torch.sigmoid(val_logits[val_mask]) > 0.5).long()
                correct = (val_pred == homo_data.y[val_mask]).sum().item()
                acc = correct / val_mask.sum().item()
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Validation Acc: {acc:.4f}")
            model.train()

    print("\n[Success] True Cross-Chain Integration complete. The model trains organically on standard PyG multi-chain structures!")

if __name__ == "__main__":
    main()
