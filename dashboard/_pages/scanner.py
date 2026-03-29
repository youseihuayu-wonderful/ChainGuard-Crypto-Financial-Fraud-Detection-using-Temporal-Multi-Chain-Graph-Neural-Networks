"""
L3: Investigation Hub — "Who is suspicious?"

Combines Transaction Scanner + Network Explorer in one page.

Connections:
  FROM L1: receives selected_timestep, selected_risk_level, selected_alert_tx
  FROM L2: "Try the model" link
  TO L4: "Submit to Forensics" for deep analysis
  Shared: selected_timestep, selected_alert_tx
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random


def render(DATA, navigate_to):
    alerts = DATA["alerts"]
    ts_risk = DATA["timestep_risk"]

    # Breadcrumb
    drill = st.session_state.get("drill_from")
    if drill:
        st.markdown(f'<div class="breadcrumb">← from {drill}</div>', unsafe_allow_html=True)

    st.markdown("# 🔍 L3: Investigation Hub")

    # Show incoming context
    context_parts = []
    if st.session_state.get("selected_alert_tx"):
        context_parts.append(f"**Alert TX:** `{st.session_state['selected_alert_tx'][:16]}...`")
    if st.session_state.get("selected_risk_level") != "ALL":
        context_parts.append(f"**Risk Filter:** {st.session_state['selected_risk_level']}")
    context_parts.append(f"**Timestep:** {st.session_state.get('selected_timestep', 25)}")

    st.markdown(" | ".join(context_parts))
    st.markdown("---")

    tab1, tab2 = st.tabs(["🔍 Transaction Scanner", "🕸️ Network Explorer"])

    # ── Tab 1: Scanner ──
    with tab1:
        sc1, sc2 = st.columns([1, 1.5])

        with sc1:
            st.markdown("### Transaction Parameters")

            # Pre-fill from alert if drilled from L1
            alert_tx = st.session_state.get("selected_alert_tx")
            alert_data = None
            if alert_tx:
                alert_data = next((a for a in alerts if a["tx_id"] == alert_tx), None)

            tx_amount = st.slider("Amount (BTC)", 0.01, 100.0,
                                  alert_data["amount_btc"] if alert_data else 1.5, 0.01, key="scan_amt")
            in_degree = st.number_input("Input Count", 1, 50,
                                        np.random.randint(2, 8) if alert_data else 3, key="scan_in")
            out_degree = st.number_input("Output Count", 1, 50,
                                         np.random.randint(2, 6) if alert_data else 2, key="scan_out")
            timestep = st.slider("Timestep", 1, 49,
                                 st.session_state.get("selected_timestep", 25), key="scan_ts")

            mixing = st.checkbox("Mixing service pattern",
                                 value=alert_data["pattern"] == "Mixing" if alert_data else False)
            rapid = st.checkbox("Rapid succession (<10 min)",
                                value=alert_data["pattern"] == "Rapid Cycling" if alert_data else False)
            chain_hop = st.checkbox("Cross-chain bridge",
                                    value=alert_data["pattern"] == "Chain Hop" if alert_data else False)

            analyze = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True, key="scan_go")

        with sc2:
            if analyze or alert_data:
                np.random.seed(42)
                risk = 0.15
                if mixing: risk += 0.25
                if rapid: risk += 0.15
                if chain_hop: risk += 0.10
                if tx_amount > 10: risk += 0.08
                if in_degree > 10: risk += 0.10
                if alert_data: risk = alert_data["risk_score"]
                risk = min(0.98, max(0.02, risk))

                level = "HIGH" if risk > 0.7 else ("MEDIUM" if risk > 0.4 else "LOW")
                color = {"HIGH": "#ff5252", "MEDIUM": "#ff9800", "LOW": "#64ffda"}[level]
                css = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[level]

                st.markdown(f'<div class="{css}" style="text-align:center; padding:20px">'
                            f'<h2 style="color:{color}; margin:0">{level} RISK</h2>'
                            f'<h1 style="color:{color}; margin:0; font-size:3rem">{risk:.0%}</h1>'
                            f'</div>', unsafe_allow_html=True)

                # Gauge
                fig_g = go.Figure(go.Indicator(mode="gauge+number", value=risk*100,
                    number=dict(suffix="%", font=dict(color="#ccd6f6")),
                    gauge=dict(axis=dict(range=[0,100]), bar=dict(color=color), bgcolor="rgba(255,255,255,0.05)",
                               steps=[dict(range=[0,40], color="rgba(100,255,218,0.15)"),
                                      dict(range=[40,70], color="rgba(255,152,0,0.15)"),
                                      dict(range=[70,100], color="rgba(255,82,82,0.15)")],
                               threshold=dict(line=dict(color="#ff5252", width=3), thickness=0.8, value=70))))
                fig_g.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccd6f6"),
                                    margin=dict(l=30, r=30, t=30, b=10))
                st.plotly_chart(fig_g, use_container_width=True)

                # Actions
                if risk > 0.5:
                    if st.button("📋 Submit to Forensics Lab for deep analysis →", key="scan_to_l4", type="primary"):
                        navigate_to("L4: Forensics Lab"); st.rerun()
                    if st.button("🕸️ View network neighborhood ↓", key="scan_to_net"):
                        st.session_state["selected_timestep"] = timestep
            else:
                st.markdown("### Configure parameters and click **Analyze**")
                st.markdown("Or select an alert from **L1: Command Center** to auto-fill.")

    # ── Tab 2: Network Explorer ──
    with tab2:
        nc1, nc2 = st.columns([1, 2])

        with nc1:
            st.markdown("### Network Parameters")
            net_ts = st.selectbox("Timestep", list(range(1, 50)),
                                  index=st.session_state.get("selected_timestep", 25) - 1, key="net_ts")
            n_nodes = st.slider("Display nodes", 20, 100, 50, key="net_nodes")
            show_temporal = st.checkbox("Show temporal k-NN edges", True, key="net_temporal")

            ti = ts_risk[net_ts]
            st.markdown(f"**TS {net_ts}**: {ti['nodes']} nodes, {ti['illicit']} illicit, {ti['zone']} zone")

            st.markdown("---")
            st.markdown("### Legend")
            st.markdown("🔴 Illicit &nbsp; 🔵 Licit &nbsp; ⚪ Unknown")
            if show_temporal:
                st.markdown("━━ Original &nbsp; ┄┄ Temporal k-NN")

        with nc2:
            np.random.seed(42 + net_ts)
            random.seed(42 + net_ts)
            n = n_nodes
            pos = np.random.randn(n, 2) * 2
            labels = np.zeros(n)
            n_ill = max(3, int(n * 0.1))
            ill_idx = random.sample(range(n), n_ill)
            for i in ill_idx:
                labels[i] = 1
                pos[i] = pos[i] * 0.4 + np.array([1, 1])

            edges = [(i, random.randint(0, n-1)) for i in range(n) for _ in range(random.randint(1, 3)) if random.randint(0, n-1) != i]
            t_edges = [(i, random.randint(0, n-1)) for i in range(n) for _ in range(random.randint(0, 1))] if show_temporal else []

            fig_net = go.Figure()
            for i, j in edges:
                fig_net.add_trace(go.Scatter(x=[pos[i,0], pos[j,0], None], y=[pos[i,1], pos[j,1], None],
                    mode='lines', line=dict(color='rgba(100,100,100,0.2)', width=0.5), hoverinfo='skip', showlegend=False))
            if show_temporal:
                for i, j in t_edges:
                    fig_net.add_trace(go.Scatter(x=[pos[i,0], pos[j,0], None], y=[pos[i,1], pos[j,1], None],
                        mode='lines', line=dict(color='rgba(100,255,218,0.15)', width=0.8, dash='dot'), hoverinfo='skip', showlegend=False))

            # Highlight selected alert TX
            alert_node = None
            if st.session_state.get("selected_alert_tx"):
                alert_node = random.randint(0, n-1)
                labels[alert_node] = 1

            for lv, color, name, sz in [(-1, '#666', 'Unknown', 7), (0, '#2196F3', 'Licit', 9), (1, '#ff5252', 'Illicit', 13)]:
                mask = labels == lv
                if not mask.any(): continue
                fig_net.add_trace(go.Scatter(
                    x=pos[mask,0], y=pos[mask,1], mode='markers',
                    marker=dict(size=sz, color=color, line=dict(width=1 if lv==1 else 0, color='white')),
                    name=name, hovertext=[f"Node {i}" for i in np.where(mask)[0]], hoverinfo='text'))

            if alert_node is not None:
                fig_net.add_trace(go.Scatter(
                    x=[pos[alert_node, 0]], y=[pos[alert_node, 1]], mode='markers',
                    marker=dict(size=22, color='rgba(0,0,0,0)', line=dict(color='#64ffda', width=3)),
                    name="Selected Alert", hovertext="Alert TX", hoverinfo='text'))

            fig_net.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,26,1)",
                font=dict(color="#ccd6f6"), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                legend=dict(orientation="h", y=-0.05, x=0.2), margin=dict(l=10, r=10, t=10, b=40))
            st.plotly_chart(fig_net, use_container_width=True)

        if st.button("📋 Send findings to Forensics Lab →", key="net_to_l4"):
            navigate_to("L4: Forensics Lab"); st.rerun()
