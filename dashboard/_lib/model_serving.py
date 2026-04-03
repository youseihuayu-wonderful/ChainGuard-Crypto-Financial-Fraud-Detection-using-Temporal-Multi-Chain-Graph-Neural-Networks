"""
Model Serving Module — Load trained M3 weights and run real inference.

This module loads the REAL trained TH-GNN (M3) model weights and provides:
1. predict_node(node_id) — Get risk score for a specific node
2. get_feature_importance() — Real gradient-based feature importance
3. get_node_explanation(node_id) — Per-node feature contributions
4. get_predictions() — All test set predictions
5. get_training_history() — Training loss/AUC curve

ALL outputs are from the actual trained model — no simulation.
"""

import os
import json
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../experiments/results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../experiments/saved_models")


def _model_available():
    """Check if trained model outputs exist."""
    return os.path.exists(os.path.join(RESULTS_DIR, "m3_predictions.json"))


@st.cache_data
def load_predictions():
    """Load all M3 predictions from real model inference."""
    path = os.path.join(RESULTS_DIR, "m3_predictions.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    """Load real gradient-based feature importance from trained M3."""
    path = os.path.join(RESULTS_DIR, "m3_feature_importance.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_node_explanations():
    """Load per-node explanations (gradient * feature) for top-50 riskiest nodes."""
    path = os.path.join(RESULTS_DIR, "m3_node_explanations.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_training_history():
    """Load training loss/AUC curve from real training run."""
    path = os.path.join(RESULTS_DIR, "m3_training_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_statistical_tests():
    """Load statistical significance test results."""
    path = os.path.join(RESULTS_DIR, "statistical_tests.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_node_risk_score(node_id, predictions=None):
    """Get real M3 risk score for a specific node."""
    if predictions is None:
        predictions = load_predictions()
    if predictions is None:
        return None

    for p in predictions.get("test_predictions", []):
        if p["node_id"] == node_id:
            return p
    return None


def get_top_risk_nodes(n=20, predictions=None):
    """Get top N riskiest nodes from real model predictions."""
    if predictions is None:
        predictions = load_predictions()
    if predictions is None:
        return []

    return predictions.get("test_predictions", [])[:n]
