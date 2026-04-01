"""
Model Performance — Internal ML Platform style
WHO: Data Scientists, Technical Team
WHAT: "Why does TH-GNN work?" — Ablation, baselines, ROI
LINKS TO: Scanner (try model), Forensics (see evidence)
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from _lib.i18n import t


def render(DATA, navigate_to):
    abl = DATA["ablation"]
    bl = DATA["baseline"]
    cs = DATA.get("case_study", {})

    if st.session_state.get("drill_from"):
        st.markdown(f'<div class="breadcrumb">\u2190 from {st.session_state["drill_from"]}</div>', unsafe_allow_html=True)

    st.markdown(f"# \U0001f9ea {t('perf_title')}")
    st.markdown(t("perf_subtitle"))
    st.markdown("---")

    # ── Top-level metrics from real experiment data ──
    best_model = "M3"
    best = abl[best_model]
    gcn = abl["M1"]
    auc_delta = (best["auc_roc"] - gcn["auc_roc"]) / gcn["auc_roc"]
    prec_delta = (best["precision"] - gcn["precision"]) / gcn["precision"]

    km1, km2, km3, km4 = st.columns(4)
    km1.metric(t("best_auc"), f"{best['auc_roc']:.4f}", f"+{auc_delta:.1%} vs GCN")
    km2.metric(t("best_f1"), f"{best['f1']:.4f}")
    km3.metric(t("precision"), f"{best['precision']:.4f}", f"+{prec_delta:.1%} vs GCN")
    km4.metric(t("models_compared"), f"{len(bl['results'])}", t("ablation_baselines"))
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([t("ablation_study"), t("baseline_comparison"), "Metrics Deep-Dive"])

    with tab1:
        st.markdown(f"### {t('component_contribution')}")
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

        sel = st.radio(t("select_model"), list(abl.keys()), index=2, horizontal=True, key="model_perf")
        st.session_state["selected_model"] = sel
        m = abl[sel]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("AUC-ROC", f"{m['auc_roc']:.4f}"); mc2.metric("F1", f"{m['f1']:.4f}")
        mc3.metric(t("precision"), f"{m['precision']:.4f}"); mc4.metric("Recall", f"{m['recall']:.4f}")

        # Highlight which component contributed most
        max_jump_idx = max(range(1, 5), key=lambda i: aucs[i] - aucs[i-1])
        max_jump = aucs[max_jump_idx] - aucs[max_jump_idx - 1]
        st.markdown(f'<div class="risk-low"><strong style="color:#00D4AA">{t("key_finding")}</strong><br>'
                    f'<span style="color:#E5E7EB">Largest single improvement: {short[max_jump_idx]} '
                    f'(+{max_jump:.1%} AUC-ROC). '
                    f'Graph augmentation (temporal k-NN edges) > model complexity.</span></div>', unsafe_allow_html=True)

        if st.button(f"\U0001f4cb {t('see_evidence')}", key="abl_to_for"):
            navigate_to("Forensics"); st.rerun()

    with tab2:
        st.markdown(f"### {t('thgnn_vs_methods')}")
        res = bl["results"]
        names_map = {"logistic_regression": "LR", "random_forest": "RF", "gradient_boosting": "GB",
                     "gcn_m1": "GCN", "gat": "GAT", "graphsage": "GraphSAGE",
                     "evolvegcn_h": "EvolveGCN", "thgnn_m3_ours": "TH-GNN"}
        radar_colors = ["#00D4AA", "#3B82F6", "#9E9E9E", "#8B5CF6"]
        top = sorted(res.items(), key=lambda x: -x[1]["auc_roc"])[:4]
        cats = ["AUC-ROC", "F1", t("precision"), "Recall"]
        fig_r = go.Figure()
        for i, (k, v) in enumerate(top):
            vals = [v["auc_roc"], v["f1"], v["precision"], v["recall"]]
            fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                            name=names_map.get(k, k), line=dict(color=radar_colors[i], width=2),
                            fill='toself', fillcolor=f"rgba({int(radar_colors[i][1:3],16)},{int(radar_colors[i][3:5],16)},{int(radar_colors[i][5:7],16)},0.06)"))
        fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,1], gridcolor="rgba(75,85,99,0.3)", color="#9CA3AF"),
                            angularaxis=dict(gridcolor="rgba(75,85,99,0.3)", color="#E5E7EB"), bgcolor="rgba(0,0,0,0)"),
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB"), height=400,
                            legend=dict(orientation="h", y=-0.1, x=0.1), margin=dict(t=40))
        st.plotly_chart(fig_r, use_container_width=True)

        rows = [{"Method": names_map.get(k, k), "Type": v["type"], "AUC": f"{v['auc_roc']:.4f}",
                 "F1": f"{v['f1']:.4f}", "Prec": f"{v['precision']:.4f}", "Rec": f"{v['recall']:.4f}"}
                for k, v in sorted(res.items(), key=lambda x: -x[1]["auc_roc"])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Rank position
        rank = next(i for i, (k, _) in enumerate(sorted(res.items(), key=lambda x: -x[1]["auc_roc"]), 1) if k == "thgnn_m3_ours")
        st.markdown(f'<div class="risk-low"><strong style="color:#00D4AA">{t("ranking")}</strong><br>'
                    f'<span style="color:#E5E7EB">TH-GNN ranks #{rank} out of {len(res)} methods on AUC-ROC. '
                    f'Tied with GraphSAGE on AUC but +{best["precision"] - res["graphsage"]["precision"]:.1%} higher precision.</span></div>',
                    unsafe_allow_html=True)

        if st.button(f"\U0001f50d {t('try_model')}", key="bl_to_scan"):
            navigate_to("Scanner"); st.rerun()

    with tab3:
        st.markdown("### All Models — Full Metrics Comparison")
        st.caption("All values from real experiment results (ablation_results.json, baseline_comparison.json)")

        # Complete metrics table for all methods
        all_methods = []
        for k, v in sorted(res.items(), key=lambda x: -x[1]["auc_roc"]):
            name = names_map.get(k, k)
            all_methods.append({
                "Model": name,
                "Type": v["type"],
                "AUC-ROC": v["auc_roc"],
                "F1": v["f1"],
                "Precision": v["precision"],
                "Recall": v["recall"],
                "FDR (1-Prec)": round(1 - v["precision"], 4),
            })
        df_all = pd.DataFrame(all_methods)
        st.dataframe(df_all, use_container_width=True, hide_index=True)

        # Precision vs Recall tradeoff
        st.markdown("#### Precision-Recall Tradeoff")
        fig_pr = go.Figure()
        for k, v in res.items():
            name = names_map.get(k, k)
            color = "#00D4AA" if k == "thgnn_m3_ours" else ("#3B82F6" if "gcn" in k.lower() else "#9CA3AF")
            size = 16 if k == "thgnn_m3_ours" else 10
            fig_pr.add_trace(go.Scatter(
                x=[v["recall"]], y=[v["precision"]],
                mode="markers+text", text=[name], textposition="top center",
                marker=dict(size=size, color=color),
                textfont=dict(color="#E5E7EB", size=10),
                name=name, showlegend=False))
        fig_pr.update_layout(height=350, xaxis=dict(title="Recall", color="#9CA3AF", gridcolor="rgba(75,85,99,0.3)"),
                             yaxis=dict(title="Precision", color="#9CA3AF", gridcolor="rgba(75,85,99,0.3)"),
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                             font=dict(color="#E5E7EB"), margin=dict(l=40, r=20, t=20, b=40))
        st.plotly_chart(fig_pr, use_container_width=True)

        # Key insight
        st.markdown(
            '<div class="risk-low"><strong style="color:#00D4AA">Key Observation</strong><br>'
            '<span style="color:#E5E7EB">'
            f'TH-GNN (M3) achieves the best AUC-ROC ({best["auc_roc"]:.4f}) with highest precision ({best["precision"]:.4f}). '
            f'This means fewer false alarms per detection — critical for operational teams.</span></div>',
            unsafe_allow_html=True)
