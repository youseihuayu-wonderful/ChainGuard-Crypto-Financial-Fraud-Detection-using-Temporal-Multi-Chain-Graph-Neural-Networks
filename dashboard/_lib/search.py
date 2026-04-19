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
    st.markdown(f"# :mag: {t('search_title')}")
    st.markdown(t("search_subtitle"))
    st.caption(t("search_caption"))

    st.markdown("---")

    # Load data
    predictions = load_predictions()
    explanations = load_node_explanations()
    graph_data = _load_graph_data()

    if not predictions:
        st.error(t("search_predictions_missing"))
        return

    # Search interface
    search_col, info_col = st.columns([2, 1])

    with search_col:
        st.markdown(f"### {t('search_by_node_id')}")
        node_id_input = st.text_input(
            t("enter_node_id"),
            placeholder="e.g., 168060",
            key="search_node_id",
            help=t("enter_node_id_help"),
        )

        search_btn = st.button(t("search"), type="primary", use_container_width=True, key="search_btn")

    with info_col:
        st.markdown(f"### {t('dataset_info')}")
        n_predictions = len(predictions.get("test_predictions", []))
        st.markdown(
            f'<div class="glass-card">'
            f'<p style="color:#E5E7EB; margin:0">'
            f'<strong style="color:#00D4AA">{n_predictions:,}</strong> {t("nodes_in_test")}<br>'
            f'<strong style="color:#3B82F6">AUC: {predictions.get("test_auc", 0):.4f}</strong><br>'
            f'<span style="color:#9CA3AF">{t("timesteps_range")}</span>'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quick-access: show some high-risk nodes
    if not search_btn or not node_id_input:
        st.markdown(f"### {t('quick_access_top')}")
        st.markdown(t("quick_access_hint"))

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
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=350)

        return

    # Parse node ID
    try:
        node_id = int(node_id_input.strip())
    except ValueError:
        st.error(t("invalid_node_id"))
        return

    # Search for the node
    node = _find_node_in_predictions(node_id, predictions)

    if not node:
        st.warning(t("node_not_found").format(node_id=node_id))

        # Show nearby nodes
        st.markdown(f"### {t('nearby_node_ids')}")
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
            st.dataframe(pd.DataFrame(nearby_rows), width="stretch", hide_index=True)

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
    m1.metric(t("risk_score_label"), f"{risk:.4f}")
    m2.metric(t("true_label_label"), "ILLICIT" if node["true_label"] == 1 else "LICIT")
    m3.metric(t("timestep"), node["timestep"])

    # Model confidence
    confidence = risk if risk > 0.5 else (1 - risk)
    m4.metric(t("model_confidence"), f"{confidence:.2%}")

    # Prediction correctness
    predicted_illicit = risk > 0.5
    actual_illicit = node["true_label"] == 1
    correct = predicted_illicit == actual_illicit

    if correct:
        st.markdown(
            '<div class="risk-low">'
            f'<strong style="color:#10B981">{t("correct_prediction")}</strong><br>'
            f'<span style="color:#E5E7EB">{t("correct_pred_desc")}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        error_type = t("false_positive") if predicted_illicit and not actual_illicit else t("false_negative")
        st.markdown(
            f'<div class="risk-high">'
            f'<strong style="color:#EF4444">{t("incorrect_prediction").format(error_type=error_type)}</strong><br>'
            f'<span style="color:#E5E7EB">{t("incorrect_pred_desc")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Node Explanation
    explanation = _find_node_explanation(node_id, explanations)

    if explanation:
        st.markdown(f"### {t('node_explanation_title')}")
        st.caption(t("node_expl_caption"))

        # Top features
        top_features = explanation.get("top_features", [])
        if top_features:
            st.markdown(f"#### {t('top_feature_contributions')}")

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
            st.plotly_chart(fig_feat, width="stretch")

            # Feature details table
            with st.expander(t("feature_details_expander")):
                feat_rows = []
                for f in top_features:
                    feat_rows.append({
                        t("feature_col"): f["feature_name"],
                        t("value_col"): f"{f['value']:.4f}",
                        t("gradient_col"): f"{f['gradient']:.4f}",
                        t("contribution_col"): f"{f['contribution']:.4f}",
                        t("direction_label"): t("risk_increasing") if f["contribution"] > 0 else t("risk_decreasing"),
                    })
                st.dataframe(pd.DataFrame(feat_rows), width="stretch", hide_index=True)

        # Neighbor info from explanation
        neighbor_info = explanation.get("neighbor_info", {})
        if neighbor_info:
            st.markdown(f"#### {t('neighbor_influence_title')}")
            ni1, ni2, ni3 = st.columns(3)
            ni1.metric(t("total_neighbors"), neighbor_info.get("n_neighbors", "N/A"))
            ni2.metric(t("illicit_neighbors"), neighbor_info.get("n_illicit", "N/A"))
            pct = neighbor_info.get("illicit_pct", 0)
            ni3.metric(t("illicit_pct_label"), f"{pct:.1f}%")

    else:
        st.info(t("no_explanation").format(node_id=node_id))

    st.markdown("---")

    # Graph Neighbors
    st.markdown(f"### {t('graph_neighbors_title')}")
    neighbors = _find_node_neighbors(node_id, graph_data, node["timestep"])

    if neighbors:
        st.markdown(t("found_neighbors").format(n=len(neighbors), ts=node['timestep']))

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
        ns1.metric(t("illicit_neighbors_metric"), n_illicit_neighbors)
        ns2.metric(t("licit_neighbors_metric"), n_licit_neighbors)
        ns3.metric(t("unknown_neighbors_metric"), n_unknown_neighbors)

        st.dataframe(pd.DataFrame(neighbor_rows), width="stretch", hide_index=True, height=300)

        if len(neighbors) > 50:
            st.caption(t("showing_neighbors").format(n=len(neighbors)))
    else:
        st.info(t("no_neighbors").format(node_id=node_id, ts=node['timestep']))

    # Navigation
    st.markdown("---")
    st.markdown(f"### {t('navigate')}")
    nav_col1, nav_col2, nav_col3 = st.columns(3)

    with nav_col1:
        if st.button(
            t("view_network_ts").format(ts=node['timestep']),
            key="search_to_network",
        ):
            navigate_to("Network", selected_timestep=node["timestep"])
            st.rerun()

    with nav_col2:
        if st.button(t("explainability_btn"), key="search_to_explain"):
            navigate_to("Explainability")
            st.rerun()

    with nav_col3:
        if st.button(t("alert_center_btn"), key="search_to_alerts"):
            navigate_to("Alerts")
            st.rerun()
