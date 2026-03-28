"""Page 1: Executive Dashboard - High-level KPIs and risk overview."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


def render(ablation, baseline, case_study):
    st.markdown("# 📊 Executive Dashboard")
    st.markdown("Real-time fraud detection overview powered by **Temporal Heterogeneous Graph Neural Network**")
    st.markdown("---")

    # ---- KPI Row ----
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Model AUC-ROC", "0.8678", "+12.3% vs baseline")
    with k2:
        st.metric("Illicit Detected", "213 / 408", "52.2% recall")
    with k3:
        st.metric("False Positive Rate", "0.93%", "-2.1%", delta_color="inverse")
    with k4:
        st.metric("Transactions Monitored", "203,769", "49 timesteps")

    st.markdown("---")

    # ---- Risk Distribution & Model Comparison ----
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### Fraud Detection Coverage")

        total = case_study["total_illicit_test"]
        both = case_study["both_detect"]
        m3_only = case_study["m3_only"]
        gcn_only = case_study["gcn_only"]
        neither = case_study["neither"]

        fig = go.Figure(data=[go.Funnel(
            y=["Total Illicit (Test Set)",
               "Detected by Any Model",
               "TH-GNN Unique Detections",
               "High Confidence (>0.9)"],
            x=[total, both + m3_only + gcn_only, m3_only, case_study["m3_high_conf"]],
            textinfo="value+percent initial",
            marker=dict(color=["#ff5252", "#ff9800", "#64ffda", "#00e676"]),
            connector=dict(line=dict(color="#333", width=1)),
        )])
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccd6f6"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Risk Alert Summary")

        st.markdown(
            f'<div class="risk-high">'
            f'<strong style="color:#ff5252">HIGH RISK</strong><br>'
            f'<span style="color:#ccd6f6">{case_study["m3_high_conf"]} transactions</span> flagged with >90% fraud probability<br>'
            f'<small>Requires immediate investigation</small>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="risk-medium">'
            f'<strong style="color:#ff9800">MEDIUM RISK</strong><br>'
            f'<span style="color:#ccd6f6">{m3_only - case_study["m3_high_conf"] + both} transactions</span> flagged with 50-90% probability<br>'
            f'<small>Scheduled for review</small>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="risk-low">'
            f'<strong style="color:#64ffda">UNDETECTED</strong><br>'
            f'<span style="color:#ccd6f6">{neither} illicit transactions</span> missed by current model<br>'
            f'<small>Known limitation — future work targets</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---- Model Superiority ----
    st.markdown("### TH-GNN vs Traditional Methods")

    methods = list(baseline["results"].keys())
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "gcn_m1": "GCN (Baseline)",
        "gat": "GAT",
        "graphsage": "GraphSAGE",
        "evolvegcn_h": "EvolveGCN-H",
        "thgnn_m3_ours": "TH-GNN (Ours)",
    }
    colors = {
        "non-graph ML": "#9E9E9E",
        "GNN": "#2196F3",
        "temporal GNN": "#9C27B0",
        "temporal heterogeneous GNN": "#64ffda",
    }

    names = [display_names.get(m, m) for m in methods]
    aucs = [baseline["results"][m]["auc_roc"] for m in methods]
    types = [baseline["results"][m]["type"] for m in methods]
    bar_colors = [colors.get(t, "#666") for t in types]

    # Sort by AUC
    sorted_data = sorted(zip(names, aucs, bar_colors), key=lambda x: x[1])
    names_s, aucs_s, colors_s = zip(*sorted_data)

    fig = go.Figure(go.Bar(
        x=list(aucs_s),
        y=list(names_s),
        orientation='h',
        marker=dict(color=list(colors_s), line=dict(width=0)),
        text=[f"{a:.4f}" for a in aucs_s],
        textposition='outside',
        textfont=dict(color="#ccd6f6"),
    ))
    fig.update_layout(
        height=400,
        xaxis=dict(range=[0.7, 0.92], title="AUC-ROC",
                   gridcolor="rgba(255,255,255,0.05)", color="#8892b0"),
        yaxis=dict(color="#ccd6f6"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccd6f6"),
        margin=dict(l=20, r=80, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    leg1, leg2, leg3, leg4 = st.columns(4)
    leg1.markdown("⬤ <span style='color:#9E9E9E'>Non-graph ML</span>", unsafe_allow_html=True)
    leg2.markdown("⬤ <span style='color:#2196F3'>Standard GNN</span>", unsafe_allow_html=True)
    leg3.markdown("⬤ <span style='color:#9C27B0'>Temporal GNN</span>", unsafe_allow_html=True)
    leg4.markdown("⬤ <span style='color:#64ffda'>TH-GNN (Ours)</span>", unsafe_allow_html=True)
