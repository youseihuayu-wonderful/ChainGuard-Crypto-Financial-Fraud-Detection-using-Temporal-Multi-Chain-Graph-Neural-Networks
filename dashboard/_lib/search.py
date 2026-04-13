"""
Node Search — Search Across Real Model Predictions
WHO: Investigators, Analysts
WHAT: Search for specific nodes by ID and view predictions, explanations, neighbors.

Uses REAL data from m3_predictions.json, m3_node_explanations.json, graph_data.json.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os

from shared import CHART_LAYOUT, COLORS
from _lib.model_serving import load_predictions, load_node_explanations
from _lib.i18n import t

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../experiments/results")


@st.cache_data
def _load_graph_data():
    """Load graph topology data."""
    path = os.path.join(RESULTS_DIR, "graph_data.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _find_node_in_predictions(node_id, predictions):
    """Find a specific node in predictions list."""
    for p in predictions.get("test_predictions", []):
        if p["node_id"] == node_id:
            return p
    return None


def _find_node_explanation(node_id, explanations):
    """Find explanation for a specific node."""
    if not explanations:
        return None
    for exp in explanations:
        if exp.get("node_id") == node_id:
            return exp
    return None


def _find_node_neighbors(node_id, graph_data, timestep):
    """Find neighbors of a node from graph topology data."""
    ts_key = str(timestep)
    if ts_key not in graph_data:
        return []

    ts_data = graph_data[ts_key]
    edges = ts_data.get("edges", [])
    labels = ts_data.get("labels", [])
    n_nodes = ts_data.get("n_nodes", 0)

    neighbors = set()
    for edge in edges:
        if len(edge) >= 2:
            src, dst = edge[0], edge[1]
            if src == node_id:
                neighbors.add(dst)
            elif dst == node_id:
                neighbors.add(src)

    # Get neighbor info
    neighbor_info = []
    for nid in neighbors:
        label = "unknown"
        if 0 <= nid < len(labels):
            label = labels[nid]
        neighbor_info.append({"node_id": nid, "label": label})

    return neighbor_info


def render(DATA, navigate_to):
    """Render the Node Search page."""
    st.markdown("# :mag: Node Search")
    st.markdown("Search across real M3 model predictions for specific nodes.")
    st.caption("All data from real M3 model predictions and Elliptic dataset.")

    st.markdown("---")

    # Load data
    predictions = load_predictions()
    explanations = load_node_explanations()
    graph_data = _load_graph_data()

    if not predictions:
        st.error(
            "M3 predictions not found. Run `train_and_save_m3.py` to generate "
            "model predictions before using Node Search."
        )
        return

    # Search interface
    search_col, info_col = st.columns([2, 1])

    with search_col:
        st.markdown("### Search by Node ID")
        node_id_input = st.text_input(
            "Enter Node ID",
            placeholder="e.g., 168060",
            key="search_node_id",
            help="Enter a numeric node ID from the Elliptic dataset.",
        )

        search_btn = st.button("Search", type="primary", use_container_width=True, key="search_btn")

    with info_col:
        st.markdown("### Dataset Info")
        n_predictions = len(predictions.get("test_predictions", []))
        st.markdown(
            f'<div class="glass-card">'
            f'<p style="color:#E5E7EB; margin:0">'
            f'<strong style="color:#00D4AA">{n_predictions:,}</strong> nodes in test set<br>'
            f'<strong style="color:#3B82F6">AUC: {predictions.get("test_auc", 0):.4f}</strong><br>'
            f'<span style="color:#9CA3AF">Timesteps 42-49</span>'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quick-access: show some high-risk nodes
    if not search_btn or not node_id_input:
        st.markdown("### Quick Access: Top Risk Nodes")
        st.markdown("Click any node ID below, then paste it in the search box above.")

        top_nodes = predictions.get("test_predictions", [])[:20]
        rows = []
        for n in top_nodes:
            label_text = "ILLICIT" if n["true_label"] == 1 else "LICIT"
            rows.append({
                "Node ID": n["node_id"],
                "Risk Score": f"{n['risk_score']:.4f}",
                "True Label": label_text,
                "Timestep": n["timestep"],
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=350)

        return

    # Parse node ID
    try:
        node_id = int(node_id_input.strip())
    except ValueError:
        st.error("Please enter a valid numeric node ID.")
        return

    # Search for the node
    node = _find_node_in_predictions(node_id, predictions)

    if not node:
        st.warning(
            f"Node {node_id} not found in M3 test predictions. "
            f"The test set contains nodes from timesteps 42-49. "
            f"Try a node ID from the quick access table below."
        )

        # Show nearby nodes
        st.markdown("### Nearby Node IDs in Test Set")
        test_preds = predictions.get("test_predictions", [])
        all_ids = sorted([p["node_id"] for p in test_preds])

        # Find closest IDs
        import bisect
        idx = bisect.bisect_left(all_ids, node_id)
        nearby = all_ids[max(0, idx - 5):min(len(all_ids), idx + 5)]

        if nearby:
            nearby_rows = []
            for nid in nearby:
                p = _find_node_in_predictions(nid, predictions)
                if p:
                    nearby_rows.append({
                        "Node ID": p["node_id"],
                        "Risk Score": f"{p['risk_score']:.4f}",
                        "True Label": "ILLICIT" if p["true_label"] == 1 else "LICIT",
                        "Timestep": p["timestep"],
                    })
            st.dataframe(pd.DataFrame(nearby_rows), use_container_width=True, hide_index=True)

        return

    # Display node information
    st.markdown(f"### Node {node_id}")

    # Risk level
    risk = node["risk_score"]
    level = "HIGH" if risk > 0.7 else ("MEDIUM" if risk > 0.4 else "LOW")
    color = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#00D4AA"}[level]
    css_class = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[level]

    st.markdown(
        f'<div class="{css_class}" style="text-align:center; padding:20px">'
        f'<h2 style="color:{color}; margin:0">Node {node_id} — {level} RISK</h2>'
        f'<h1 style="color:{color}; margin:0; font-size:3rem">{risk:.2%}</h1>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Score", f"{risk:.4f}")
    m2.metric("True Label", "ILLICIT" if node["true_label"] == 1 else "LICIT")
    m3.metric("Timestep", node["timestep"])

    # Model confidence
    confidence = risk if risk > 0.5 else (1 - risk)
    m4.metric("Model Confidence", f"{confidence:.2%}")

    # Prediction correctness
    predicted_illicit = risk > 0.5
    actual_illicit = node["true_label"] == 1
    correct = predicted_illicit == actual_illicit

    if correct:
        st.markdown(
            '<div class="risk-low">'
            '<strong style="color:#10B981">Correct Prediction</strong><br>'
            '<span style="color:#E5E7EB">Model prediction matches ground truth label.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        error_type = "False Positive" if predicted_illicit and not actual_illicit else "False Negative"
        st.markdown(
            f'<div class="risk-high">'
            f'<strong style="color:#EF4444">Incorrect Prediction ({error_type})</strong><br>'
            f'<span style="color:#E5E7EB">Model prediction does not match ground truth label.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Node Explanation
    explanation = _find_node_explanation(node_id, explanations)

    if explanation:
        st.markdown("### Node Explanation")
        st.caption("Real gradient-based feature contributions from trained M3 model.")

        # Top features
        top_features = explanation.get("top_features", [])
        if top_features:
            st.markdown("#### Top Feature Contributions")

            feat_names = [f["feature_name"] for f in top_features[:10]]
            contributions = [f["contribution"] for f in top_features[:10]]
            colors = ["#EF4444" if c > 0 else "#10B981" for c in contributions]

            fig_feat = go.Figure()
            fig_feat.add_trace(go.Bar(
                y=feat_names[::-1],
                x=contributions[::-1],
                orientation="h",
                marker_color=colors[::-1],
                text=[f"{c:.3f}" for c in contributions[::-1]],
                textposition="outside",
                textfont=dict(color="#E5E7EB", size=10),
            ))
            fig_feat.update_layout(
                **CHART_LAYOUT,
                height=350,
                title=dict(text="Feature Contributions (gradient x value)", font=dict(size=14)),
                xaxis_title="Contribution",
            )
            st.plotly_chart(fig_feat, use_container_width=True)

            # Feature details table
            with st.expander("Feature Details"):
                feat_rows = []
                for f in top_features:
                    feat_rows.append({
                        "Feature": f["feature_name"],
                        "Value": f"{f['value']:.4f}",
                        "Gradient": f"{f['gradient']:.4f}",
                        "Contribution": f"{f['contribution']:.4f}",
                        "Direction": "Risk-increasing" if f["contribution"] > 0 else "Risk-decreasing",
                    })
                st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

        # Neighbor info from explanation
        neighbor_info = explanation.get("neighbor_info", {})
        if neighbor_info:
            st.markdown("#### Neighbor Influence")
            ni1, ni2, ni3 = st.columns(3)
            ni1.metric("Total Neighbors", neighbor_info.get("n_neighbors", "N/A"))
            ni2.metric("Illicit Neighbors", neighbor_info.get("n_illicit", "N/A"))
            pct = neighbor_info.get("illicit_pct", 0)
            ni3.metric("Illicit %", f"{pct:.1f}%")

    else:
        st.info(
            f"No detailed explanation available for node {node_id}. "
            f"Explanations are generated for the top 50 riskiest nodes."
        )

    st.markdown("---")

    # Graph Neighbors
    st.markdown("### Graph Neighbors")
    neighbors = _find_node_neighbors(node_id, graph_data, node["timestep"])

    if neighbors:
        st.markdown(f"Found **{len(neighbors)}** neighbors at timestep {node['timestep']}")

        neighbor_rows = []
        n_illicit_neighbors = 0
        n_licit_neighbors = 0
        n_unknown_neighbors = 0

        for nb in neighbors[:50]:  # Limit display
            label = nb["label"]
            if label == "illicit" or label == 1:
                label_text = "ILLICIT"
                n_illicit_neighbors += 1
            elif label == "licit" or label == 0:
                label_text = "LICIT"
                n_licit_neighbors += 1
            else:
                label_text = "UNKNOWN"
                n_unknown_neighbors += 1

            # Check if neighbor has a prediction
            nb_pred = _find_node_in_predictions(nb["node_id"], predictions)
            risk_str = f"{nb_pred['risk_score']:.4f}" if nb_pred else "N/A"

            neighbor_rows.append({
                "Node ID": nb["node_id"],
                "Label": label_text,
                "Risk Score": risk_str,
            })

        # Neighbor summary
        ns1, ns2, ns3 = st.columns(3)
        ns1.metric("Illicit Neighbors", n_illicit_neighbors)
        ns2.metric("Licit Neighbors", n_licit_neighbors)
        ns3.metric("Unknown Neighbors", n_unknown_neighbors)

        st.dataframe(pd.DataFrame(neighbor_rows), use_container_width=True, hide_index=True, height=300)

        if len(neighbors) > 50:
            st.caption(f"Showing 50 of {len(neighbors)} neighbors.")
    else:
        st.info(
            f"No neighbor data found for node {node_id} at timestep {node['timestep']}. "
            f"Graph data may not be available for this timestep."
        )

    # Navigation
    st.markdown("---")
    st.markdown("### Navigate")
    nav_col1, nav_col2, nav_col3 = st.columns(3)

    with nav_col1:
        if st.button(
            f"View Network (t={node['timestep']})",
            key="search_to_network",
        ):
            navigate_to("Network", selected_timestep=node["timestep"])
            st.rerun()

    with nav_col2:
        if st.button("Explainability", key="search_to_explain"):
            navigate_to("Explainability")
            st.rerun()

    with nav_col3:
        if st.button("Alert Center", key="search_to_alerts"):
            navigate_to("Alerts")
            st.rerun()
