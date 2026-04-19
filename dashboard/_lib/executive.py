"""
Executive Dashboard — TRM Labs style
WHO: CEO, CRO, Management
WHAT: "What's happening?" — KPIs, trends, detection results
LINKS TO: Performance (why), Scanner (investigate TX), Network (graph), Forensics (evidence)

DATA SOURCE: All metrics from real experiment results. Timeline from actual Elliptic dataset.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from _lib.i18n import t


def render(DATA, navigate_to):
    cs = DATA["case_study"]
    bl = DATA["baseline"]
    ts_risk = DATA["timestep_risk"]

    st.markdown(f"# \U0001f4ca {t('exec_title')}")
    st.markdown(t("exec_subtitle"))

    # ── Section Navigation (TOC) ──
    st.markdown(
        '<div style="display:flex; gap:6px; flex-wrap:wrap; padding:12px 0">'
        f'<a href="#kpi-overview" style="padding:6px 14px; background:#111827; border:1px solid #1F2937; '
        f'border-radius:6px; color:#9CA3AF; text-decoration:none; font-size:0.8rem; font-weight:500; '
        f'transition:all 0.2s">\U0001f4ca {t("kpis")}</a>'
        f'<a href="#detection-funnel" style="padding:6px 14px; background:#111827; border:1px solid #1F2937; '
        f'border-radius:6px; color:#9CA3AF; text-decoration:none; font-size:0.8rem; font-weight:500">\U0001f53b {t("detection")}</a>'
        f'<a href="#risk-timeline" style="padding:6px 14px; background:#111827; border:1px solid #1F2937; '
        f'border-radius:6px; color:#9CA3AF; text-decoration:none; font-size:0.8rem; font-weight:500">\U0001f4c8 {t("timeline")}</a>'
        f'<a href="#dataset-overview" style="padding:6px 14px; background:#111827; border:1px solid #1F2937; '
        f'border-radius:6px; color:#9CA3AF; text-decoration:none; font-size:0.8rem; font-weight:500">\U0001f4ca {t("dataset_label")}</a>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── KPIs (derived from experiment data) ──
    st.markdown('<div id="kpi-overview"></div>', unsafe_allow_html=True)
    abl = DATA["ablation"]
    best = abl["M3"]
    gcn = abl["M1"]
    auc_delta = (best["auc_roc"] - gcn["auc_roc"]) / gcn["auc_roc"]
    prec_delta = (best["precision"] - gcn["precision"]) / gcn["precision"]
    fdr = 1 - best["precision"]
    gcn_fdr = 1 - gcn["precision"]

    k1, k2, k3, k4 = st.columns(4)
    total_detected = cs["both_detect"] + cs["m3_only"]
    k1.metric(t("auc_roc"), f"{best['auc_roc']:.4f}", f"+{auc_delta:.1%} vs GCN")
    k2.metric(t("detected"), f"{total_detected}/{cs['total_illicit_test']}", f"{total_detected/cs['total_illicit_test']:.0%}")
    k3.metric(t("precision"), f"{best['precision']:.2%}", f"+{prec_delta:.1%} vs GCN")
    k4.metric(t("false_alarm"), f"{fdr:.1%}", f"-{gcn_fdr - fdr:.1%} vs GCN", delta_color="inverse")

    if st.button(f"\U0001f9ea {t('why_numbers')}", key="kpi_drill"):
        navigate_to("Performance"); st.rerun()

    st.markdown("---")

    # ── Detection Funnel ──
    st.markdown('<div id="detection-funnel"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(f"### {t('detection_funnel')}")
        fig = go.Figure(data=[go.Funnel(
            y=[t("total_illicit"), t("any_model_detects"), t("thgnn_unique"), t("high_conf")],
            x=[cs["total_illicit_test"], total_detected + cs["gcn_only"], cs["m3_only"], cs["m3_high_conf"]],
            textinfo="value+percent initial",
            marker=dict(color=["#EF4444", "#F59E0B", "#00D4AA", "#10B981"]),
        )])
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=5, b=5),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB"))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown(f"### {t('risk_alerts')}")
        st.markdown(f'<div class="risk-high"><strong>HIGH — {cs["m3_high_conf"]}</strong><br>'
                    f'<span style="color:#9CA3AF">{t("high_confidence_desc")}</span></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="risk-medium"><strong>DETECTED — {total_detected}</strong><br>'
                    f'<span style="color:#9CA3AF">{t("total_detected_desc")}</span></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="risk-low"><strong>UNDETECTED — {cs["neither"]}</strong><br>'
                    f'<span style="color:#9CA3AF">{t("undetected_miss_desc").format(neither=cs["neither"], total=cs["total_illicit_test"])}</span></div>',
                    unsafe_allow_html=True)

        st.caption(t("exec_real_data_caption"))

    st.markdown("---")

    # ── Risk Timeline (REAL Elliptic data) ──
    st.markdown('<div id="risk-timeline"></div>', unsafe_allow_html=True)
    st.markdown(f"### \U0001f4c8 {t('risk_timeline')}")
    st.caption(t("exec_timeline_caption"))

    timesteps = list(range(1, 50))
    rates = [ts_risk[t_val]["risk_rate"] for t_val in timesteps]
    illicit_counts = [ts_risk[t_val]["illicit"] for t_val in timesteps]
    node_counts = [ts_risk[t_val]["nodes"] for t_val in timesteps]
    colors = ["rgba(100,255,218,0.6)" if ts_risk[t_val]["zone"] == "train"
              else ("rgba(255,152,0,0.6)" if ts_risk[t_val]["zone"] == "val"
                    else "rgba(255,82,82,0.6)") for t_val in timesteps]

    fig_tl = go.Figure(go.Bar(x=timesteps, y=rates, marker_color=colors,
        hovertext=[f"TS{t_val}: {r:.2f}% illicit\n{illicit_counts[i]} illicit / {node_counts[i]} nodes"
                   for i, (t_val, r) in enumerate(zip(timesteps, rates))],
        hoverinfo="text"))
    fig_tl.update_layout(height=250, xaxis=dict(title=t("timestep"), color="#9CA3AF", gridcolor="rgba(75,85,99,0.2)"),
                         yaxis=dict(title=t("illicit_rate"), color="#9CA3AF", gridcolor="rgba(75,85,99,0.2)"),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                         font=dict(color="#E5E7EB"), margin=dict(l=40, r=20, t=10, b=40), showlegend=False)
    st.plotly_chart(fig_tl, width="stretch")

    tc1, tc2 = st.columns([1, 1])
    with tc1:
        sel_ts = st.slider(t("timestep"), 1, 49, st.session_state.get("selected_timestep", 25), key="ts_exec")
    with tc2:
        ti = ts_risk[sel_ts]
        st.markdown(f"**TS {sel_ts}** | {ti['nodes']:,} nodes | {ti['illicit']} illicit ({ti['risk_rate']:.2f}%) | {ti['zone']}")
        if ti.get('edges'):
            st.markdown(f"Edges: {ti['edges']:,}")
        if st.button(f"\U0001f578\ufe0f {t('explore_ts_network').format(ts=sel_ts)}", key="ts_drill"):
            navigate_to("Network", selected_timestep=sel_ts); st.rerun()

    st.markdown("---")

    # ── Dataset Overview (REAL statistics) ──
    st.markdown('<div id="dataset-overview"></div>', unsafe_allow_html=True)
    st.markdown(f"### {t('dataset_overview')}")
    st.caption(t("dataset_caption"))

    total_nodes = sum(ts_risk[ts]["nodes"] for ts in range(1, 50))
    total_illicit = sum(ts_risk[ts]["illicit"] for ts in range(1, 50))
    total_licit = sum(ts_risk[ts].get("licit", 0) for ts in range(1, 50))
    total_unknown = sum(ts_risk[ts].get("unknown", 0) for ts in range(1, 50))
    total_edges = sum(ts_risk[ts].get("edges", 0) for ts in range(1, 50))

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric(t("total_nodes"), f"{total_nodes:,}")
    d2.metric(t("total_edges"), f"{total_edges:,}")
    d3.metric(t("illicit_label"), f"{total_illicit:,}", f"{total_illicit/total_nodes:.2%}")
    d4.metric(t("licit_label"), f"{total_licit:,}", f"{total_licit/total_nodes:.2%}")
    d5.metric(t("unknown_label"), f"{total_unknown:,}", f"{total_unknown/total_nodes:.2%}")

    # Split info
    train_ill = sum(ts_risk[ts]["illicit"] for ts in range(1, 35))
    val_ill = sum(ts_risk[ts]["illicit"] for ts in range(35, 42))
    test_ill = sum(ts_risk[ts]["illicit"] for ts in range(42, 50))
    sp1, sp2, sp3 = st.columns(3)
    sp1.metric(t("train_split"), f"{train_ill} {t('illicit_suffix')}", f"{sum(ts_risk[ts]['nodes'] for ts in range(1,35)):,} {t('nodes_suffix')}")
    sp2.metric(t("val_split"), f"{val_ill} {t('illicit_suffix')}", f"{sum(ts_risk[ts]['nodes'] for ts in range(35,42)):,} {t('nodes_suffix')}")
    sp3.metric(t("test_split"), f"{test_ill} {t('illicit_suffix')}", f"{sum(ts_risk[ts]['nodes'] for ts in range(42,50)):,} {t('nodes_suffix')}")

    st.markdown("---")

    # Cross-links
    cl1, cl2, cl3 = st.columns(3)
    with cl1:
        if st.button(f"\U0001f9ea {t('full_model_comparison')}", key="roi_drill"):
            navigate_to("Performance"); st.rerun()
    with cl2:
        if st.button(f"\U0001f578\ufe0f {t('explore_graph')}", key="net_drill"):
            navigate_to("Network"); st.rerun()
    with cl3:
        if st.button(f"\U0001f4cb {t('evidence_forensics')}", key="flow_drill"):
            navigate_to("Forensics"); st.rerun()
