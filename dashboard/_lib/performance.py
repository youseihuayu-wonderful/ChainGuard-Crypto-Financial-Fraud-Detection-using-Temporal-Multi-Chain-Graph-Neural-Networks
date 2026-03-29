"""
Model Performance — Internal ML Platform style
WHO: Data Scientists, Technical Team
WHAT: "Why does TH-GNN work?" — Ablation, baselines, ROI
LINKS TO: Scanner (try model), Forensics (see evidence)
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render(DATA, navigate_to):
    abl = DATA["ablation"]
    bl = DATA["baseline"]

    if st.session_state.get("drill_from"):
        st.markdown(f'<div class="breadcrumb">← from {st.session_state["drill_from"]}</div>', unsafe_allow_html=True)

    st.markdown("# 🧪 Model Performance")
    st.markdown("Understanding **why** TH-GNN outperforms — ablation, baselines, ROI")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Ablation Study", "Baseline Comparison", "ROI Analysis"])

    with tab1:
        st.markdown("### Component Contribution (M1 → M5)")
        short = ["M1: GCN", "M2: +Temporal", "M3: +Hetero", "M4: TH-GNN", "M5: +LP"]
        aucs = [abl[m]["auc_roc"] for m in abl]
        f1s = [abl[m]["f1"] for m in abl]

        fig = go.Figure()
        colors_auc = ["#3B82F6"] * 5; colors_auc[2] = "#00D4AA"
        fig.add_trace(go.Bar(name="AUC-ROC", x=short, y=aucs, marker_color=colors_auc,
                             text=[f"{v:.4f}" for v in aucs], textposition="outside", textfont=dict(color="#E5E7EB")))
        fig.add_trace(go.Bar(name="F1", x=short, y=f1s, marker_color="#F59E0B",
                             text=[f"{v:.4f}" for v in f1s], textposition="outside", textfont=dict(color="#E5E7EB")))
        for i in range(1, 5):
            d = aucs[i] - aucs[0]
            fig.add_annotation(x=short[i], y=aucs[i]+0.06, text=f"+{d:.1%}", showarrow=False,
                               font=dict(color="#EF4444" if i==2 else "#9CA3AF", size=11))
        fig.update_layout(barmode="group", height=400,
                          yaxis=dict(range=[0, 1.1], gridcolor="rgba(75,85,99,0.3)", color="#9CA3AF"),
                          xaxis=dict(color="#E5E7EB"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                          font=dict(color="#E5E7EB"), legend=dict(orientation="h", y=1.1, x=0.3), margin=dict(t=60))
        st.plotly_chart(fig, use_container_width=True)

        sel = st.radio("Select model variant", list(abl.keys()), index=2, horizontal=True, key="model_perf")
        st.session_state["selected_model"] = sel
        m = abl[sel]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("AUC-ROC", f"{m['auc_roc']:.4f}"); mc2.metric("F1", f"{m['f1']:.4f}")
        mc3.metric("Precision", f"{m['precision']:.4f}"); mc4.metric("Recall", f"{m['recall']:.4f}")

        st.markdown('<div class="risk-low"><strong style="color:#00D4AA">Key Finding</strong><br>'
                    '<span style="color:#E5E7EB">Graph augmentation (temporal k-NN edges) > model complexity.</span></div>', unsafe_allow_html=True)

        if st.button("📋 See evidence → Forensics", key="abl_to_for"):
            navigate_to("Forensics"); st.rerun()

    with tab2:
        st.markdown("### TH-GNN vs 7 Methods")
        res = bl["results"]
        names_map = {"logistic_regression": "LR", "random_forest": "RF", "gradient_boosting": "GB",
                     "gcn_m1": "GCN", "gat": "GAT", "graphsage": "GraphSAGE",
                     "evolvegcn_h": "EvolveGCN", "thgnn_m3_ours": "TH-GNN"}
        radar_colors = ["#00D4AA", "#3B82F6", "#9E9E9E", "#8B5CF6"]
        top = sorted(res.items(), key=lambda x: -x[1]["auc_roc"])[:4]
        cats = ["AUC-ROC", "F1", "Precision", "Recall"]
        fig_r = go.Figure()
        for i, (k, v) in enumerate(top):
            vals = [v["auc_roc"], v["f1"], v["precision"], v["recall"]]
            fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                            name=names_map.get(k, k), line=dict(color=radar_colors[i], width=2),
                            fill='toself', fillcolor=f"{radar_colors[i]}10"))
        fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,1], gridcolor="rgba(75,85,99,0.3)", color="#9CA3AF"),
                            angularaxis=dict(gridcolor="rgba(75,85,99,0.3)", color="#E5E7EB"), bgcolor="rgba(0,0,0,0)"),
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB"), height=400,
                            legend=dict(orientation="h", y=-0.1, x=0.1), margin=dict(t=40))
        st.plotly_chart(fig_r, use_container_width=True)

        rows = [{"Method": names_map.get(k, k), "Type": v["type"], "AUC": f"{v['auc_roc']:.4f}",
                 "F1": f"{v['f1']:.4f}", "Prec": f"{v['precision']:.4f}", "Rec": f"{v['recall']:.4f}"}
                for k, v in sorted(res.items(), key=lambda x: -x[1]["auc_roc"])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.button("🔍 Try the model → Scanner", key="bl_to_scan"):
            navigate_to("Scanner"); st.rerun()

    with tab3:
        st.markdown("### Business Value Analysis")
        cats_roi = ["Fraud Loss", "Recovery", "Invest. Cost", "Net Savings"]
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(name="No Model", x=cats_roi, y=[25, 0, 0, 0], marker_color="#EF4444",
                                 text=["$25M", "$0", "$0", "$0"], textposition="outside", textfont=dict(color="#E5E7EB")))
        fig_roi.add_trace(go.Bar(name="GCN", x=cats_roi, y=[25, 7.1, 2.5, 4.6], marker_color="#3B82F6",
                                 text=["$25M", "$7.1M", "$2.5M", "$4.6M"], textposition="outside", textfont=dict(color="#E5E7EB")))
        fig_roi.add_trace(go.Bar(name="TH-GNN", x=cats_roi, y=[25, 17, 3.2, 12.8], marker_color="#00D4AA",
                                 text=["$25M", "$17M", "$3.2M", "$12.8M"], textposition="outside", textfont=dict(color="#E5E7EB")))
        fig_roi.update_layout(barmode="group", height=350,
                              yaxis=dict(title="USD (M)", gridcolor="rgba(75,85,99,0.3)", color="#9CA3AF"),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                              font=dict(color="#E5E7EB"), legend=dict(orientation="h", y=1.1, x=0.2), margin=dict(t=60))
        st.plotly_chart(fig_roi, use_container_width=True)

        st.markdown('<div style="background:rgba(100,255,218,0.08); border-radius:12px; padding:20px; text-align:center">'
                    '<p style="color:#9CA3AF; margin:0">TH-GNN Annual Net Savings</p>'
                    '<h1 style="color:#00D4AA; margin:5px 0; font-size:3rem">$12.8M</h1>'
                    '<p style="color:#00D4AA">+178% vs GCN</p></div>', unsafe_allow_html=True)
