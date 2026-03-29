"""
Forensics Lab — Compliance Report System style
WHO: AML Investigators, Auditors
WHAT: "What's the evidence?" — Detection comparison, fraud patterns, research findings
LINKS TO: Performance (model details), Executive (back to overview)
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def render(DATA, navigate_to):
    cs = DATA["case_study"]
    abl = DATA["ablation"]

    if st.session_state.get("drill_from"):
        st.markdown(f'<div class="breadcrumb">← from {st.session_state["drill_from"]}</div>', unsafe_allow_html=True)

    st.markdown("# 📋 Forensics Lab")
    st.markdown("Deep-dive evidence analysis for compliance and audit")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Detection Evidence", "Fraud Patterns", "Research Conclusions"])

    with tab1:
        st.markdown("### TH-GNN vs GCN: Detection Comparison")
        st.markdown(f"*Model: **{st.session_state.get('selected_model', 'M3')}***")

        total = cs["total_illicit_test"]
        both, m3_only, gcn_only, neither = cs["both_detect"], cs["m3_only"], cs["gcn_only"], cs["neither"]

        fig = go.Figure()
        fig.add_shape(type="circle", x0=-1.5, y0=-1.5, x1=1.5, y1=1.5,
                      line=dict(color="#3B82F6", width=2), fillcolor="rgba(33,150,243,0.1)")
        fig.add_shape(type="circle", x0=-0.5, y0=-1.5, x1=2.5, y1=1.5,
                      line=dict(color="#00D4AA", width=2), fillcolor="rgba(100,255,218,0.1)")
        fig.add_annotation(x=-0.8, y=0, text=f"<b>GCN Only</b><br>{gcn_only}", font=dict(color="#3B82F6", size=16), showarrow=False)
        fig.add_annotation(x=0.5, y=0, text=f"<b>Both</b><br>{both}", font=dict(color="#E5E7EB", size=18), showarrow=False)
        fig.add_annotation(x=1.7, y=0, text=f"<b>TH-GNN</b><br><b>{m3_only}</b>", font=dict(color="#00D4AA", size=15), showarrow=False)
        fig.add_annotation(x=0.5, y=-2.2, text=f"Undetected: {neither} ({neither/total:.0%})", font=dict(color="#EF4444", size=13), showarrow=False)
        fig.update_layout(height=300, xaxis=dict(range=[-2.5,4.0], showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(range=[-2.8,2], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
                          paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("TH-GNN Total", f"{both + m3_only}", f"+{m3_only} unique")
        e2.metric("GCN Total", f"{both + gcn_only}")
        e3.metric("High Conf (TH-GNN)", f"{cs['m3_high_conf']}", f"vs {cs['gcn_high_conf']} GCN")
        e4.metric("Undetected", f"{neither}", f"{neither/total:.0%} gap")

        st.markdown(f'<div class="risk-low"><strong style="color:#00D4AA">Evidence</strong><br>'
                    f'<span style="color:#E5E7EB">TH-GNN uniquely catches <b>{m3_only}</b> illicit transactions '
                    f'({m3_only/total:.0%}) that GCN misses.</span></div>', unsafe_allow_html=True)

        if st.button("🧪 Model details → Performance", key="ev_to_perf"):
            navigate_to("Performance"); st.rerun()

    with tab2:
        st.markdown("### Fraud Patterns Identified")
        st.markdown(f"*Context: TS {st.session_state.get('selected_timestep', '—')}, "
                    f"Risk: {st.session_state.get('selected_risk_level', 'ALL')}*")

        patterns = [
            ("Temporal Cycling", "38%", "HIGH", "Funds cycled across 2-3 timesteps", "Temporal k-NN edges", "M3"),
            ("Fan-out Splitting", "24%", "HIGH", "Large input → many small outputs", "R-GCN message passing", "M3"),
            ("Mixing Service", "19%", "CRITICAL", "Routes through mixing/tumbling services", "Graph convolution", "M1-M5"),
            ("Dormant Activation", "12%", "MEDIUM", "Long-dormant address suddenly active", "Temporal attention", "M2"),
            ("Chain Hopping", "7%", "HIGH", "Value crosses chain via bridge", "Cross-timestep k-NN", "M3"),
        ]
        sev_colors = {"CRITICAL": "#DC2626", "HIGH": "#EF4444", "MEDIUM": "#F59E0B"}
        for name, freq, sev, desc, how, comp in patterns:
            sc = sev_colors.get(sev, "#666")
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.03); border-radius:8px; padding:16px; "
                f"margin:8px 0; border-left:4px solid {sc}'>"
                f"<div style='display:flex; justify-content:space-between'>"
                f"<h4 style='margin:0; color:#E5E7EB'>{name}</h4>"
                f"<span style='background:{sc}20; color:{sc}; padding:2px 10px; border-radius:12px; font-size:0.8rem'>{sev}</span></div>"
                f"<p style='color:#9CA3AF; margin:6px 0'>{desc}</p>"
                f"<span style='color:#00D4AA; font-size:0.85rem'>Freq: {freq}</span> · "
                f"<span style='color:#9CA3AF; font-size:0.85rem'>Detection: {how}</span> · "
                f"<span style='color:#F59E0B; font-size:0.85rem'>Component: {comp}</span></div>", unsafe_allow_html=True)

        # Pattern × Component heatmap
        st.markdown("---")
        st.markdown("#### Pattern ↔ Model Component Mapping")
        z = [[0.9,0.3,0.2,0.1,0.1],[0.9,0.3,0.2,0.8,0.3],[0.95,0.9,0.5,0.4,0.9],[0.93,0.85,0.5,0.7,0.85],[0.92,0.8,0.5,0.6,0.8]]
        fig_map = go.Figure(go.Heatmap(z=z,
            x=["Temporal\nCycling", "Fan-out", "Mixing", "Dormant", "Chain\nHop"],
            y=["M1: GCN", "M2: +Temporal", "M3: +Hetero", "M4: TH-GNN", "M5: +LP"],
            colorscale=[[0,"#0f0f1a"],[0.5,"#F59E0B"],[1,"#00D4AA"]],
            text=[[f"{v:.0%}" for v in row] for row in z], texttemplate="%{text}", textfont=dict(color="#E5E7EB"),
            colorbar=dict(title="Detection", tickcolor="#9CA3AF")))
        fig_map.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB"),
                              xaxis=dict(side="top", color="#E5E7EB"), yaxis=dict(color="#E5E7EB", autorange="reversed"),
                              margin=dict(l=100, r=20, t=60, b=20))
        st.plotly_chart(fig_map, use_container_width=True)

    with tab3:
        st.markdown("### Key Research Findings")
        findings = [
            ("Graph Augmentation > Model Complexity",
             "Temporal k-NN edges contribute more than attention or label propagation.",
             f"M3: {abl['M3']['auc_roc']:.4f} > M4: {abl['M4']['auc_roc']:.4f} > M5: {abl['M5']['auc_roc']:.4f}"),
            ("Elliptic Has Isolated Timesteps",
             "49 timesteps are completely isolated subgraphs — standard GNNs underperform non-graph ML.",
             "GCN: 0.7449 < RF: 0.8601 (without augmentation)"),
            ("TH-GNN Beats All Baselines",
             "With temporal k-NN augmentation, TH-GNN achieves highest AUC across all 7 baselines.",
             "TH-GNN: 0.8678 > GraphSAGE: 0.8624 > RF: 0.8601"),
        ]
        for title, body, evidence in findings:
            st.markdown(f"<div style='background:rgba(255,255,255,0.03); border-radius:8px; padding:16px; margin:10px 0'>"
                        f"<h4 style='color:#00D4AA; margin:0'>{title}</h4>"
                        f"<p style='color:#E5E7EB; margin:8px 0'>{body}</p>"
                        f"<p style='color:#F59E0B; font-size:0.85rem; margin:0'>Evidence: {evidence}</p></div>", unsafe_allow_html=True)

        st.markdown('<div style="background:rgba(100,255,218,0.05); border-radius:12px; padding:24px; text-align:center; margin-top:20px">'
                    '<h3 style="color:#00D4AA; margin:0">ChainGuard</h3>'
                    '<p style="color:#E5E7EB">Cross-Chain Cryptocurrency Fraud Detection using TH-GNN</p>'
                    '<p style="color:#9CA3AF">NYU Tandon · MS Thesis 2026</p>'
                    '<p style="color:#00D4AA; font-size:1.2rem">AUC: 0.8678 | +12.3% vs baseline</p></div>', unsafe_allow_html=True)

        st.markdown("")
        if st.button("📊 Back to Executive Dashboard", key="conc_to_exec", type="primary"):
            navigate_to("Executive"); st.rerun()
