"""
Train Logistic Regression Baseline on Elliptic Dataset

Non-graph baseline: uses only node features (166-dim), no graph structure.
This establishes the lower bound for what features alone can achieve.

Reference: Weber et al., 2019 (Elliptic original paper)

Usage:
    python experiments/scripts/train_lr_baseline.py
"""

import os
import sys
import random
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.elliptic_loader import load_elliptic_csv, temporal_split
from src.evaluation.metrics import compute_metrics, print_report

# ============================================================
# Configuration
# ============================================================
SEED = 42
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

    # Extract numpy arrays (no graph structure needed)
    X = data.x.numpy()
    y = data.y.numpy()

    X_train, y_train = X[train_mask.numpy()], y[train_mask.numpy()]
    X_val, y_val = X[val_mask.numpy()], y[val_mask.numpy()]
    X_test, y_test = X[test_mask.numpy()], y[test_mask.numpy()]

    print(f"\nTrain: {len(y_train)} samples ({(y_train==1).sum()} illicit)")
    print(f"Val:   {len(y_val)} samples ({(y_val==1).sum()} illicit)")
    print(f"Test:  {len(y_test)} samples ({(y_test==1).sum()} illicit)")

    # Train Logistic Regression
    print(f"\n{'='*60}")
    print("Training Logistic Regression")
    print(f"{'='*60}")

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    print(f"Converged in {model.n_iter_[0]} iterations")

    # Evaluate on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, val_probs)
    print(f"Val AUC: {val_metrics['auc_roc']:.4f} | Val F1: {val_metrics['f1']:.4f}")

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Test Set Results (Logistic Regression)")
    print(f"{'='*60}")
    test_probs = model.predict_proba(X_test)[:, 1]
    print_report(y_test, test_probs)

    test_metrics = compute_metrics(y_test, test_probs)
    return test_metrics


if __name__ == "__main__":
    metrics = main()
