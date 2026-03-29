"""
Executive Dashboard — TRM Labs style
WHO: CEO, CRO, Management
WHAT: "What's happening?" — KPIs, trends, alerts, fund flows, ROI
LINKS TO: Performance (why), Scanner (investigate TX), Network (graph), Forensics (evidence)
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def render(DATA, navigate_to):
    cs = DATA["case_study"]
    bl = DATA["baseline"]
    ts_risk = DATA["timestep_risk"]
    alerts = DATA["alerts"]

    st.markdown("# 📊 Executive Dashboard")
    st.markdown("Enterprise fraud monitoring — **click any section to drill down**")

    # ── Section Navigation (TOC) ──
    st.markdown(
        '<div style="display:flex; gap:6px; flex-wrap:wrap; padding:12px 0">'
        '<a href="#kpi-overview" style="padding:6px 14px; background:#111827; border:1px solid #1F2937; '
        'border-radius:6px; color:#9CA3AF; text-decoration:none; font-size:0.8rem; font-weight:500; '
        'transition:all 0.2s">📊 KPIs</a>'
        '<a href="#detection-funnel" style="padding:6px 14px; background:#111827; border:1px solid #1F2937; '
        'border-radius:6px; color:#9CA3AF; text-decoration:none; font-size:0.8rem; font-weight:500">🔻 Detection</a>'
        '<a href="#risk-timeline" style="padding:6px 14px; background:#111827; border:1px solid #1F2937; '
        'border-radius:6px; color:#9CA3AF; text-decoration:none; font-size:0.8rem; font-weight:500">📈 Timeline</a>'
        '<a href="#alert-queue" style="padding:6px 14px; background:#111827; border:1px solid #1F2937; '
        'border-radius:6px; color:#9CA3AF; text-decoration:none; font-size:0.8rem; font-weight:500">🚨 Alerts</a>'
        '<a href="#fund-flow" style="padding:6px 14px; background:#111827; border:1px solid #1F2937; '
        'border-radius:6px; color:#9CA3AF; text-decoration:none; font-size:0.8rem; font-weight:500">💰 Fund Flow</a>'
        '<a href="#roi-summary" style="padding:6px 14px; background:#111827; border:1px solid #1F2937; '
        'border-radius:6px; color:#9CA3AF; text-decoration:none; font-size:0.8rem; font-weight:500">💹 ROI</a>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── KPIs ──
    st.markdown('<div id="kpi-overview"></div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    total_detected = cs["both_detect"] + cs["m3_only"]
    k1.metric("AUC-ROC", "0.8678", "+12.3% vs GCN")
    k2.metric("Detected", f"{total_detected}/{cs['total_illicit_test']}", f"{total_detected/cs['total_illicit_test']:.0%}")
    k3.metric("Precision", "71.68%", "+43.9%")
    k4.metric("FP Rate", "0.93%", "-2.1%", delta_color="inverse")
    k5.metric("Savings", "$12.8M/yr", "vs $4.6M GCN")

    if st.button("🧪 Why these numbers? → Performance", key="kpi_drill"):
        navigate_to("Performance"); st.rerun()

    st.markdown("---")

    # ── Detection Funnel + Risk Alerts ──
    st.markdown('<div id="detection-funnel"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Detection Funnel")
        fig = go.Figure(data=[go.Funnel(
            y=["Total Illicit", "Any Model Detects", "TH-GNN Unique", "High Conf (>0.9)"],
            x=[cs["total_illicit_test"], total_detected + cs["gcn_only"], cs["m3_only"], cs["m3_high_conf"]],
            textinfo="value+percent initial",
            marker=dict(color=["#EF4444", "#F59E0B", "#00D4AA", "#10B981"]),
        )])
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=5, b=5),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Risk Alerts → Investigate")
        for level, count, css in [
            ("HIGH", cs["m3_high_conf"], "risk-high"),
            ("MEDIUM", cs["m3_only"] - cs["m3_high_conf"] + cs["both_detect"], "risk-medium"),
            ("LOW", 8355, "risk-low"),
        ]:
            st.markdown(f'<div class="{css}"><strong>{level} — {count}</strong></div>', unsafe_allow_html=True)

        risk_pick = st.selectbox("Filter risk level", ["HIGH", "MEDIUM", "ALL"], key="risk_pick_exec")
        if st.button("🔍 Open Scanner →", key="risk_drill", type="primary"):
            navigate_to("Scanner", selected_risk_level=risk_pick); st.rerun()

    st.markdown("---")

    # ── Risk Timeline ──
    st.markdown('<div id="risk-timeline"></div>', unsafe_allow_html=True)
    st.markdown("### 📈 Risk Timeline → Select timestep to explore")
    timesteps = list(range(1, 50))
    rates = [ts_risk[t]["risk_rate"] for t in timesteps]
    colors = ["rgba(100,255,218,0.6)" if ts_risk[t]["zone"] == "train"
              else ("rgba(255,152,0,0.6)" if ts_risk[t]["zone"] == "val"
                    else "rgba(255,82,82,0.6)") for t in timesteps]

    fig_tl = go.Figure(go.Bar(x=timesteps, y=rates, marker_color=colors,
        hovertext=[f"TS{t}: {r:.1f}% ({ts_risk[t]['illicit']} illicit)" for t, r in zip(timesteps, rates)],
        hoverinfo="text"))
    fig_tl.update_layout(height=220, xaxis=dict(title="Timestep", color="#9CA3AF", gridcolor="rgba(75,85,99,0.2)"),
                         yaxis=dict(title="Risk %", color="#9CA3AF", gridcolor="rgba(75,85,99,0.2)"),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                         font=dict(color="#E5E7EB"), margin=dict(l=40, r=20, t=10, b=40), showlegend=False)
    st.plotly_chart(fig_tl, use_container_width=True)

    tc1, tc2 = st.columns([1, 1])
    with tc1:
        sel_ts = st.slider("Timestep", 1, 49, st.session_state.get("selected_timestep", 25), key="ts_exec")
    with tc2:
        ti = ts_risk[sel_ts]
        st.markdown(f"**TS {sel_ts}** | {ti['nodes']} nodes | {ti['illicit']} illicit ({ti['risk_rate']:.1f}%) | {ti['zone']}")
        if st.button(f"🕸️ Explore TS {sel_ts} in Network →", key="ts_drill"):
            navigate_to("Network", selected_timestep=sel_ts); st.rerun()

    st.markdown("---")

    # ── Alert Queue ──
    st.markdown('<div id="alert-queue"></div>', unsafe_allow_html=True)
    st.markdown("### 🚨 Alert Queue → Select transaction to investigate")
    icons = {"New": "🔴", "In Review": "🟡", "Resolved": "🟢", "Dismissed": "⚪"}
    df = pd.DataFrame([{
        "P": f"P{a['priority']}", "TX": a["tx_id"][:14] + "...",
        "Risk": f"{a['risk_score']:.0%}", "BTC": f"{a['amount_btc']:.3f}",
        "TS": a["timestep"], "Pattern": a["pattern"],
        "Status": f"{icons.get(a['status'], '')} {a['status']}",
    } for a in alerts[:12]])
    st.dataframe(df, use_container_width=True, hide_index=True, height=300)

    ac1, ac2 = st.columns(2)
    with ac1:
        aidx = st.selectbox("Select alert", range(min(12, len(alerts))),
            format_func=lambda i: f"P{alerts[i]['priority']} | {alerts[i]['tx_id'][:12]}... | {alerts[i]['risk_score']:.0%}",
            key="alert_exec")
    with ac2:
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("🔍 Scan TX →", key="alert_to_scan"):
                a = alerts[aidx]
                navigate_to("Scanner", selected_alert_tx=a["tx_id"], selected_timestep=a["timestep"]); st.rerun()
        with bc2:
            if st.button("🕸️ View Network →", key="alert_to_net"):
                a = alerts[aidx]
                navigate_to("Network", selected_alert_tx=a["tx_id"], selected_timestep=a["timestep"]); st.rerun()

    st.markdown("---")

    # ── Fund Flow Sankey ──
    st.markdown('<div id="fund-flow"></div>', unsafe_allow_html=True)
    st.markdown("### 💰 Fund Flow → Drill to Forensics")
    fig_sk = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20,
            label=["Illicit Source", "Direct", "Mixing", "Chain Hop", "Layering",
                   "Exchange", "Darknet", "Unidentified", "Seized"],
            color=["#EF4444", "#F59E0B", "#8B5CF6", "#3B82F6", "#F59E0B",
                   "#10B981", "#DC2626", "#666", "#00D4AA"]),
        link=dict(source=[0,0,0,0,1,2,2,3,4,4,4], target=[1,2,3,4,5,4,6,5,5,7,8],
                  value=[85,120,45,158,60,55,65,30,80,48,30],
                  color=["rgba(255,152,0,0.25)"]*11),
    ))
    fig_sk.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                         font=dict(color="#E5E7EB", size=11), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_sk, use_container_width=True)

    if st.button("📋 Deep-dive patterns → Forensics", key="flow_drill"):
        navigate_to("Forensics"); st.rerun()

    st.markdown("---")

    # ── ROI ──
    st.markdown('<div id="roi-summary"></div>', unsafe_allow_html=True)
    st.markdown("### 💹 ROI Summary")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("TH-GNN Savings", "$12.8M/yr")
    r2.metric("vs GCN", "+$8.2M", "+178%")
    r3.metric("Recovery Rate", "68.0%")
    r4.metric("Payback", "< 3 months")

    if st.button("📊 Full ROI & model comparison → Performance", key="roi_drill"):
        navigate_to("Performance"); st.rerun()
