"""
Train Random Forest Baseline on Elliptic Dataset

Non-graph baseline: uses only node features (166-dim), no graph structure.
Random Forest was used in the original Elliptic paper (Weber et al., 2019).

Usage:
    python experiments/scripts/train_rf_baseline.py
"""

import os
import sys
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.elliptic_loader import load_elliptic_csv, temporal_split
from src.evaluation.metrics import compute_metrics, print_report

# ============================================================
# Configuration
# ============================================================
SEED = 42
N_ESTIMATORS = 300
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def main(seed=SEED):
    set_seed(seed)

    print(f"\n{'='*60}")
    print("Loading Elliptic Dataset...")
    print(f"{'='*60}")
    data = load_elliptic_csv(DATA_DIR)
    train_mask, val_mask, test_mask = temporal_split(data)

    X = data.x.numpy()
    y = data.y.numpy()

    X_train, y_train = X[train_mask.numpy()], y[train_mask.numpy()]
    X_val, y_val = X[val_mask.numpy()], y[val_mask.numpy()]
    X_test, y_test = X[test_mask.numpy()], y[test_mask.numpy()]

    print(f"\nTrain: {len(y_train)} samples ({(y_train==1).sum()} illicit)")
    print(f"Val:   {len(y_val)} samples ({(y_val==1).sum()} illicit)")
    print(f"Test:  {len(y_test)} samples ({(y_test==1).sum()} illicit)")

    # Train Random Forest
    print(f"\n{'='*60}")
    print(f"Training Random Forest (n_estimators={N_ESTIMATORS})")
    print(f"{'='*60}")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"Trained {N_ESTIMATORS} trees")

    # Evaluate on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, val_probs)
    print(f"Val AUC: {val_metrics['auc_roc']:.4f} | Val F1: {val_metrics['f1']:.4f}")

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Test Set Results (Random Forest)")
    print(f"{'='*60}")
    test_probs = model.predict_proba(X_test)[:, 1]
    print_report(y_test, test_probs)

    test_metrics = compute_metrics(y_test, test_probs)
    return test_metrics


if __name__ == "__main__":
    metrics = main()
