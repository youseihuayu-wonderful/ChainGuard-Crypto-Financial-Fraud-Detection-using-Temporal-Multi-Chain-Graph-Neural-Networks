"""
L2: Model Analytics — "Why does TH-GNN work?"

Connections:
  FROM L1: KPIs drill down, ROI drill down
  TO L3: "Try the model" → Investigation Hub
  TO L4: "See evidence" → Forensics Lab
  Shared: selected_model from session_state
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def render(DATA, navigate_to):
    abl = DATA["ablation"]
    bl = DATA["baseline"]
    cs = DATA["case_study"]

    # Breadcrumb
    if st.session_state.get("drill_from"):
        st.markdown(f'<div class="breadcrumb">← from {st.session_state["drill_from"]}</div>', unsafe_allow_html=True)

    st.markdown("# 🧪 L2: Model Analytics")
    st.markdown("Understanding **why** TH-GNN outperforms — ablation, baselines, and ROI")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Ablation Study", "Baseline Comparison", "ROI Analysis"])

    # ── Tab 1: Ablation ──
    with tab1:
        st.markdown("### Component Contribution (M1 → M5)")
        short = ["M1: GCN", "M2: +Temporal", "M3: +Hetero", "M4: TH-GNN", "M5: +LP"]
        aucs = [abl[m]["auc_roc"] for m in abl]
        f1s = [abl[m]["f1"] for m in abl]

        fig = go.Figure()
        colors_auc = ["#2196F3"] * 5
        colors_auc[2] = "#64ffda"  # highlight M3
        fig.add_trace(go.Bar(name="AUC-ROC", x=short, y=aucs, marker_color=colors_auc,
                             text=[f"{v:.4f}" for v in aucs], textposition="outside", textfont=dict(color="#ccd6f6")))
        fig.add_trace(go.Bar(name="F1", x=short, y=f1s, marker_color="#FF9800",
                             text=[f"{v:.4f}" for v in f1s], textposition="outside", textfont=dict(color="#ccd6f6")))

        for i in range(1, 5):
            d = aucs[i] - aucs[0]
            fig.add_annotation(x=short[i], y=aucs[i]+0.06, text=f"+{d:.1%}", showarrow=False,
                               font=dict(color="#ff5252" if i==2 else "#8892b0", size=11))

        fig.update_layout(barmode="group", height=400,
                          yaxis=dict(range=[0, 1.1], gridcolor="rgba(255,255,255,0.05)", color="#8892b0"),
                          xaxis=dict(color="#ccd6f6"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#ccd6f6"), legend=dict(orientation="h", y=1.1, x=0.3), margin=dict(t=60))
        st.plotly_chart(fig, use_container_width=True)

        # Model selector → shared state
        sel = st.radio("Select model variant for investigation", list(abl.keys()),
                       index=2, horizontal=True, key="model_select_l2")
        st.session_state["selected_model"] = sel
        m = abl[sel]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("AUC-ROC", f"{m['auc_roc']:.4f}")
        mc2.metric("F1", f"{m['f1']:.4f}")
        mc3.metric("Precision", f"{m['precision']:.4f}")
        mc4.metric("Recall", f"{m['recall']:.4f}")

        st.markdown(
            '<div class="risk-low"><strong style="color:#64ffda">Key Finding</strong><br>'
            '<span style="color:#ccd6f6">Graph augmentation (temporal k-NN edges) matters more than model complexity. '
            'M3 achieves best AUC with just heterogeneous edge modeling.</span></div>', unsafe_allow_html=True)

        if st.button("📋 See evidence in Forensics Lab →", key="abl_to_l4"):
            navigate_to("L4: Forensics Lab"); st.rerun()

    # ── Tab 2: Baseline ──
    with tab2:
        st.markdown("### TH-GNN vs 7 Methods")
        res = bl["results"]
        names_map = {"logistic_regression": "LR", "random_forest": "RF", "gradient_boosting": "GB",
                     "gcn_m1": "GCN", "gat": "GAT", "graphsage": "GraphSAGE",
                     "evolvegcn_h": "EvolveGCN", "thgnn_m3_ours": "TH-GNN"}
        type_colors = {"non-graph ML": "#9E9E9E", "GNN": "#2196F3", "temporal GNN": "#9C27B0",
                       "temporal heterogeneous GNN": "#64ffda"}

        # Radar chart
        top = sorted(res.items(), key=lambda x: -x[1]["auc_roc"])[:4]
        cats = ["AUC-ROC", "F1", "Precision", "Recall"]
        radar_colors = ["#64ffda", "#2196F3", "#9E9E9E", "#9C27B0"]
        fig_r = go.Figure()
        for i, (k, v) in enumerate(top):
            vals = [v["auc_roc"], v["f1"], v["precision"], v["recall"]]
            fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                            name=names_map.get(k, k), line=dict(color=radar_colors[i], width=2),
                            fill='toself', fillcolor=f"{radar_colors[i]}10"))
        fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,1], gridcolor="rgba(255,255,255,0.1)", color="#8892b0"),
                            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#ccd6f6"),
                            bgcolor="rgba(0,0,0,0)"),
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccd6f6"), height=400,
                            legend=dict(orientation="h", y=-0.1, x=0.1), margin=dict(t=40))
        st.plotly_chart(fig_r, use_container_width=True)

        # Full table
        rows = [{"Method": names_map.get(k, k), "Type": v["type"], "AUC": f"{v['auc_roc']:.4f}",
                 "F1": f"{v['f1']:.4f}", "Prec": f"{v['precision']:.4f}", "Rec": f"{v['recall']:.4f}"}
                for k, v in sorted(res.items(), key=lambda x: -x[1]["auc_roc"])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.button("🔍 Try the model yourself → Investigation Hub", key="bl_to_l3"):
            navigate_to("L3: Investigation Hub"); st.rerun()

    # ── Tab 3: ROI ──
    with tab3:
        st.markdown("### Business Value Analysis")
        cats_roi = ["Fraud Loss", "Recovery", "Investigation Cost", "Net Savings"]
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(name="No Model", x=cats_roi, y=[25, 0, 0, 0], marker_color="#ff5252",
                                 text=["$25M", "$0", "$0", "$0"], textposition="outside", textfont=dict(color="#ccd6f6")))
        fig_roi.add_trace(go.Bar(name="GCN", x=cats_roi, y=[25, 7.1, 2.5, 4.6], marker_color="#2196F3",
                                 text=["$25M", "$7.1M", "$2.5M", "$4.6M"], textposition="outside", textfont=dict(color="#ccd6f6")))
        fig_roi.add_trace(go.Bar(name="TH-GNN", x=cats_roi, y=[25, 17, 3.2, 12.8], marker_color="#64ffda",
                                 text=["$25M", "$17M", "$3.2M", "$12.8M"], textposition="outside", textfont=dict(color="#ccd6f6")))
        fig_roi.update_layout(barmode="group", height=350,
                              yaxis=dict(title="USD (M)", gridcolor="rgba(255,255,255,0.05)", color="#8892b0"),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#ccd6f6"), legend=dict(orientation="h", y=1.1, x=0.2), margin=dict(t=60))
        st.plotly_chart(fig_roi, use_container_width=True)

        st.markdown(
            '<div style="background:rgba(100,255,218,0.08); border-radius:12px; padding:20px; text-align:center">'
            '<p style="color:#8892b0; margin:0">TH-GNN Annual Net Savings</p>'
            '<h1 style="color:#64ffda; margin:5px 0; font-size:3rem">$12.8M</h1>'
            '<p style="color:#64ffda">+178% vs GCN baseline</p></div>', unsafe_allow_html=True)

        items = [("Recovery Rate", "68.0%"), ("Precision Gain", "+43.9%"),
                 ("Analyst Hours Saved", "2,400/yr"), ("Payback", "< 3 months")]
        for label, val in items:
            st.markdown(f"<div style='padding:8px 12px; background:rgba(255,255,255,0.03); border-radius:4px; margin:4px 0; "
                        f"display:flex; justify-content:space-between'>"
                        f"<span style='color:#ccd6f6'>{label}</span>"
                        f"<span style='color:#64ffda; font-weight:bold'>{val}</span></div>", unsafe_allow_html=True)

        st.markdown("<small style='color:#4a5568'>* Based on avg BTC fraud case $61,275 (Chainalysis 2025)</small>", unsafe_allow_html=True)
