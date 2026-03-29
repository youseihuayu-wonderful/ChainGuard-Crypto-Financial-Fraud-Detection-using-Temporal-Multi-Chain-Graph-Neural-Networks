"""
Transaction Scanner — Elliptic Navigator style
WHO: Operations Analysts
WHAT: "Is this transaction suspicious?" — Risk scoring, factor breakdown
LINKS TO: Network (see neighborhood), Forensics (submit evidence)
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def render(DATA, navigate_to):
    alerts = DATA["alerts"]

    if st.session_state.get("drill_from"):
        st.markdown(f'<div class="breadcrumb">← from {st.session_state["drill_from"]}</div>', unsafe_allow_html=True)

    st.markdown("# 🔍 Transaction Scanner")
    st.markdown("Analyze individual Bitcoin transactions for fraud risk")

    # Show context from Executive
    ctx = []
    if st.session_state.get("selected_alert_tx"):
        ctx.append(f"**Alert TX:** `{st.session_state['selected_alert_tx'][:16]}...`")
    if st.session_state.get("selected_risk_level", "ALL") != "ALL":
        ctx.append(f"**Risk Filter:** {st.session_state.get('selected_risk_level', 'ALL')}")
    if ctx:
        st.markdown(" | ".join(ctx))
    st.markdown("---")

    sc1, sc2 = st.columns([1, 1.5])

    with sc1:
        st.markdown("### Transaction Parameters")

        alert_tx = st.session_state.get("selected_alert_tx")
        alert_data = next((a for a in alerts if a["tx_id"] == alert_tx), None) if alert_tx else None

        tx_amount = st.slider("Amount (BTC)", 0.01, 100.0,
                              alert_data["amount_btc"] if alert_data else 1.5, 0.01, key="scan_amt")
        in_degree = st.number_input("Input Count", 1, 50, 3, key="scan_in")
        out_degree = st.number_input("Output Count", 1, 50, 2, key="scan_out")
        timestep = st.slider("Timestep", 1, 49,
                             st.session_state.get("selected_timestep", 25), key="scan_ts")

        st.markdown("#### Behavior Flags")
        mixing = st.checkbox("Mixing service pattern", value=alert_data["pattern"] == "Mixing" if alert_data else False)
        rapid = st.checkbox("Rapid succession (<10 min)", value=alert_data["pattern"] == "Rapid Cycling" if alert_data else False)
        chain_hop = st.checkbox("Cross-chain bridge", value=alert_data["pattern"] == "Chain Hop" if alert_data else False)

        analyze = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True, key="scan_go")

    with sc2:
        if analyze or alert_data:
            risk = 0.15
            if mixing: risk += 0.25
            if rapid: risk += 0.15
            if chain_hop: risk += 0.10
            if tx_amount > 10: risk += 0.08
            if in_degree > 10: risk += 0.10
            if out_degree > 10: risk += 0.05
            if alert_data: risk = alert_data["risk_score"]
            risk = min(0.98, max(0.02, risk))

            level = "HIGH" if risk > 0.7 else ("MEDIUM" if risk > 0.4 else "LOW")
            color = {"HIGH": "#ff5252", "MEDIUM": "#ff9800", "LOW": "#64ffda"}[level]
            css = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[level]

            st.markdown(f'<div class="{css}" style="text-align:center; padding:20px">'
                        f'<h2 style="color:{color}; margin:0">{level} RISK</h2>'
                        f'<h1 style="color:{color}; margin:0; font-size:3rem">{risk:.0%}</h1></div>', unsafe_allow_html=True)

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

            # Risk factors
            st.markdown("### Risk Factor Breakdown")
            factors = []
            if mixing: factors.append(("Mixing service pattern", "+25%", "HIGH"))
            if rapid: factors.append(("Rapid succession", "+15%", "MEDIUM"))
            if chain_hop: factors.append(("Cross-chain bridge", "+10%", "MEDIUM"))
            if tx_amount > 10: factors.append(("High value (>10 BTC)", "+8%", "LOW"))
            if in_degree > 10: factors.append(("High input count", "+10%", "MEDIUM"))
            if not factors: factors.append(("No significant risk factors", "0%", "LOW"))

            for name, weight, sev in factors:
                sc = {"HIGH": "#ff5252", "MEDIUM": "#ff9800", "LOW": "#64ffda"}[sev]
                st.markdown(f"<div style='display:flex; justify-content:space-between; padding:8px 12px; "
                            f"background:rgba(255,255,255,0.03); border-radius:4px; margin:4px 0;'>"
                            f"<span style='color:#ccd6f6'>{name}</span>"
                            f"<span style='color:{sc}; font-weight:bold'>{weight}</span></div>", unsafe_allow_html=True)

            # Cross-links
            st.markdown("---")
            st.markdown("### Next Steps")
            n1, n2 = st.columns(2)
            with n1:
                if st.button("🕸️ View neighborhood → Network", key="scan_to_net", type="primary"):
                    navigate_to("Network", selected_timestep=timestep); st.rerun()
            with n2:
                if st.button("📋 Submit to → Forensics", key="scan_to_for"):
                    navigate_to("Forensics"); st.rerun()
        else:
            st.markdown("### 👈 Configure parameters and click **Analyze**")
            st.markdown("Or select an alert from **Executive Dashboard** to auto-fill.")
            st.markdown("---")
            st.markdown("#### How TH-GNN Scores")
            st.markdown("""
            1. **166 features** extracted from transaction
            2. **Graph construction** with temporal k-NN edges
            3. **R-GCN** processes heterogeneous edges
            4. **Risk score** output: [0, 1]
            """)
