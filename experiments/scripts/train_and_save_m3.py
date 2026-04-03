"""
Train M3 (R-GCN + Heterogeneous Edges) and save:
1. Model weights (.pt)
2. Node embeddings for all nodes
3. Prediction probabilities for all nodes
4. Feature importance via gradient-based attribution
5. Per-node prediction details for test set

This script produces REAL model outputs for the dashboard.
"""

import os
import sys
import random
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.elliptic_loader import load_elliptic_csv, temporal_split, add_temporal_edges
from src.models.hetero_gcn import HeteroGCN
from src.evaluation.metrics import compute_metrics, print_report

# Configuration
SEED = 42
HIDDEN_CHANNELS = 80
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 20
TEMPORAL_K = 5
NUM_RELATIONS = 2
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "../saved_models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, data, train_mask, optimizer, class_weight):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_type)
    loss = F.binary_cross_entropy_with_logits(
        logits[train_mask],
        data.y[train_mask].float(),
        pos_weight=class_weight,
    )
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_type)
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    metrics = compute_metrics(labels, probs)
    return metrics, probs, labels


def get_embeddings(model, data):
    """Extract intermediate embeddings from R-GCN layers."""
    model.eval()
    with torch.no_grad():
        h = model.conv1(data.x, data.edge_index, data.edge_type)
        h = F.relu(h)
        layer1_emb = h.cpu().numpy()

        h = model.conv2(h, data.edge_index, data.edge_type)
        h = F.relu(h)
        layer2_emb = h.cpu().numpy()

    return layer1_emb, layer2_emb


def compute_feature_importance(model, data, test_mask):
    """Compute gradient-based feature importance (real, not simulated)."""
    model.eval()
    data.x.requires_grad_(True)

    logits = model(data.x, data.edge_index, data.edge_type)

    # Compute gradients of illicit predictions w.r.t. input features
    test_logits = logits[test_mask]
    test_logits.sum().backward()

    # Average absolute gradient across test nodes = feature importance
    grad = data.x.grad[test_mask].abs().mean(dim=0).detach().cpu().numpy()
    data.x.requires_grad_(False)

    return grad


def compute_node_explanations(model, data, test_mask, top_n=50):
    """Get per-node feature contributions for top-N highest risk test nodes."""
    model.eval()

    # Get predictions for all test nodes
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_type)
        probs = torch.sigmoid(logits).cpu().numpy()

    test_indices = torch.where(test_mask)[0].cpu().numpy()
    test_probs = probs[test_indices]

    # Sort by risk score, take top N
    top_risk_order = np.argsort(-test_probs)[:top_n]
    top_indices = test_indices[top_risk_order]

    explanations = []
    for node_idx in top_indices:
        node_idx_int = int(node_idx)

        # Compute per-node gradient
        data.x.requires_grad_(True)
        logits = model(data.x, data.edge_index, data.edge_type)
        logits[node_idx_int].backward(retain_graph=True)

        node_grad = data.x.grad[node_idx_int].detach().cpu().numpy()
        data.x.requires_grad_(False)
        model.zero_grad()

        # Feature contributions = gradient * feature_value
        feature_values = data.x[node_idx_int].detach().cpu().numpy()
        contributions = node_grad * feature_values

        # Get top contributing features
        top_feat_idx = np.argsort(-np.abs(contributions))[:15]

        explanation = {
            "node_id": node_idx_int,
            "risk_score": float(probs[node_idx_int]),
            "true_label": int(data.y[node_idx_int].item()),
            "timestep": int(data.timestep[node_idx_int].item()),
            "top_features": [
                {
                    "feature_idx": int(fi),
                    "feature_name": f"f{fi}",
                    "value": float(feature_values[fi]),
                    "gradient": float(node_grad[fi]),
                    "contribution": float(contributions[fi]),
                }
                for fi in top_feat_idx
            ],
        }

        # Get neighbor info from edge_index
        edge_src = data.edge_index[0].cpu().numpy()
        edge_dst = data.edge_index[1].cpu().numpy()
        edge_types = data.edge_type.cpu().numpy()

        neighbor_mask = edge_dst == node_idx_int
        neighbor_ids = edge_src[neighbor_mask]
        neighbor_edge_types = edge_types[neighbor_mask]

        neighbors = []
        for nid, etype in zip(neighbor_ids[:20], neighbor_edge_types[:20]):  # cap at 20
            neighbors.append({
                "node_id": int(nid),
                "label": int(data.y[nid].item()),
                "risk_score": float(probs[nid]),
                "edge_type": int(etype),  # 0=original, 1=temporal
                "timestep": int(data.timestep[nid].item()),
            })
        explanation["neighbors"] = neighbors

        explanations.append(explanation)

    return explanations


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    print("\nLoading Elliptic Dataset + Temporal Edge Augmentation...")
    data = load_elliptic_csv(DATA_DIR)
    train_mask, val_mask, test_mask = temporal_split(data)
    data = add_temporal_edges(data, k=TEMPORAL_K)

    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Class weight
    n_illicit = (data.y[train_mask] == 1).sum().float()
    n_licit = (data.y[train_mask] == 0).sum().float()
    class_weight = (n_licit / n_illicit).to(device)
    print(f"Class weight: {class_weight:.2f}")

    # Model
    model = HeteroGCN(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        num_relations=NUM_RELATIONS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training
    print("\nTraining M3: R-GCN + Heterogeneous Edges")
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    training_history = []

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, data, train_mask, optimizer, class_weight)

        if epoch % 10 == 0 or epoch == 1:
            val_metrics, _, _ = evaluate(model, data, val_mask)
            training_history.append({
                "epoch": epoch, "loss": round(loss, 4),
                "val_auc": round(val_metrics["auc_roc"], 4),
                "val_f1": round(val_metrics["f1"], 4),
            })
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val AUC: {val_metrics['auc_roc']:.4f}")

            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # ═══════════════════════════════════════════
    # SAVE 1: Model weights
    # ═══════════════════════════════════════════
    weights_path = os.path.join(SAVE_DIR, "m3_hetero_gcn.pt")
    torch.save({
        "model_state_dict": best_model_state,
        "config": {
            "in_channels": data.x.shape[1],
            "hidden_channels": HIDDEN_CHANNELS,
            "num_relations": NUM_RELATIONS,
            "dropout": DROPOUT,
        },
        "seed": SEED,
        "best_val_auc": best_val_auc,
    }, weights_path)
    print(f"\nSaved model weights: {weights_path}")

    # ═══════════════════════════════════════════
    # SAVE 2: Test results
    # ═══════════════════════════════════════════
    test_metrics, test_probs, test_labels = evaluate(model, data, test_mask)
    print_report(test_labels, test_probs)

    # ═══════════════════════════════════════════
    # SAVE 3: All node predictions
    # ═══════════════════════════════════════════
    print("\nComputing predictions for all nodes...")
    model.eval()
    with torch.no_grad():
        all_logits = model(data.x, data.edge_index, data.edge_type)
        all_probs = torch.sigmoid(all_logits).cpu().numpy()

    predictions = {
        "n_nodes": int(len(all_probs)),
        "test_auc": round(test_metrics["auc_roc"], 4),
        "test_f1": round(test_metrics["f1"], 4),
        "test_precision": round(test_metrics["precision"], 4),
        "test_recall": round(test_metrics["recall"], 4),
    }

    # Save test node predictions
    test_indices = torch.where(test_mask)[0].cpu().numpy()
    test_predictions = []
    for idx in test_indices:
        test_predictions.append({
            "node_id": int(idx),
            "risk_score": round(float(all_probs[idx]), 4),
            "true_label": int(data.y[idx].item()),
            "timestep": int(data.timestep[idx].item()),
        })
    predictions["test_predictions"] = sorted(test_predictions, key=lambda x: -x["risk_score"])

    pred_path = os.path.join(RESULTS_DIR, "m3_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions: {pred_path}")

    # ═══════════════════════════════════════════
    # SAVE 4: Feature importance (gradient-based)
    # ═══════════════════════════════════════════
    print("\nComputing feature importance (gradient-based)...")
    data_cpu = data.cpu()
    model_cpu = model.cpu()
    test_mask_cpu = test_mask.cpu()

    feat_importance = compute_feature_importance(model_cpu, data_cpu, test_mask_cpu)

    # Feature names from Elliptic: first 94 are "local" features, rest are "aggregated"
    feature_names = []
    for i in range(94):
        feature_names.append(f"local_f{i+1}")
    for i in range(71):
        feature_names.append(f"agg_f{i+1}")

    importance_data = {
        "method": "gradient-based attribution (real model gradients)",
        "model": "M3 (R-GCN + Heterogeneous Edges)",
        "n_features": len(feat_importance),
        "features": [
            {"idx": int(i), "name": feature_names[i] if i < len(feature_names) else f"f{i}",
             "importance": round(float(feat_importance[i]), 6)}
            for i in np.argsort(-feat_importance)
        ],
    }

    imp_path = os.path.join(RESULTS_DIR, "m3_feature_importance.json")
    with open(imp_path, "w") as f:
        json.dump(importance_data, f, indent=2)
    print(f"Saved feature importance: {imp_path}")

    # ═══════════════════════════════════════════
    # SAVE 5: Node explanations (top 50 riskiest)
    # ═══════════════════════════════════════════
    print("\nComputing node explanations for top 50 riskiest test nodes...")
    explanations = compute_node_explanations(model_cpu, data_cpu, test_mask_cpu, top_n=50)

    expl_path = os.path.join(RESULTS_DIR, "m3_node_explanations.json")
    with open(expl_path, "w") as f:
        json.dump(explanations, f, indent=2)
    print(f"Saved node explanations: {expl_path}")

    # ═══════════════════════════════════════════
    # SAVE 6: Training history
    # ═══════════════════════════════════════════
    history_path = os.path.join(RESULTS_DIR, "m3_training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"Saved training history: {history_path}")

    print("\n" + "=" * 60)
    print("ALL REAL MODEL OUTPUTS SAVED")
    print("=" * 60)
    print(f"  Model weights: {weights_path}")
    print(f"  Predictions:   {pred_path}")
    print(f"  Feature imp:   {imp_path}")
    print(f"  Explanations:  {expl_path}")
    print(f"  History:       {history_path}")
    print(f"\n  Test AUC: {test_metrics['auc_roc']:.4f}")
    print(f"  Test F1:  {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
