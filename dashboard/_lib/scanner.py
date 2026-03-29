"""
Transaction Scanner — Elliptic Navigator style
WHO: Operations Analysts
WHAT: "Is this transaction suspicious?" — Multi-layer risk scoring

Risk Engine v2.0 — Anti-evasion enhancements:
  1. Address aggregation (24h cumulative volume)
  2. Frequency detection (tx count per hour)
  3. Boosted cross-chain weight (+20%)
  4. Multiplicative combo bonuses (rapid+fan-out, mixing+chain-hop)
  5. Graph-level neighborhood risk propagation
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def compute_risk(tx_amount, in_degree, out_degree, mixing, rapid, chain_hop,
                 addr_24h_volume, addr_24h_count, neighbor_illicit_pct):
    """
    Multi-layer risk scoring engine.

    Layer 1: Individual transaction features
    Layer 2: Address-level aggregation (24h window)
    Layer 3: Behavioral combo detection (multiplicative)
    Layer 4: Graph neighborhood propagation
    """
    factors = []
    risk = 0.08  # lower base — let factors drive the score

    # ── Layer 1: Transaction Features ──
    if mixing:
        risk += 0.25
        factors.append(("Mixing service pattern", 0.25, "HIGH"))
    if rapid:
        risk += 0.15
        factors.append(("Rapid succession (<10 min)", 0.15, "MEDIUM"))
    if chain_hop:
        risk += 0.20  # V3: boosted from 0.10
        factors.append(("Cross-chain bridge (boosted)", 0.20, "HIGH"))
    if tx_amount > 10:
        risk += 0.08
        factors.append(("High value (>10 BTC)", 0.08, "MEDIUM"))
    if in_degree > 10:
        risk += 0.10
        factors.append(("High input count (>10)", 0.10, "MEDIUM"))
    if out_degree > 10:
        risk += 0.08
        factors.append(("High output count (>10)", 0.08, "MEDIUM"))
    elif out_degree > 4:
        risk += 0.05
        factors.append(("Fan-out pattern (>4 outputs)", 0.05, "LOW"))

    # ── Layer 2: Address Aggregation (V1, V2) ──
    if addr_24h_volume > 50:
        bonus = 0.25
        risk += bonus
        factors.append(("24h cumulative volume >50 BTC", bonus, "HIGH"))
    elif addr_24h_volume > 20:
        bonus = 0.15
        risk += bonus
        factors.append(("24h cumulative volume >20 BTC", bonus, "MEDIUM"))

    if addr_24h_count > 10:
        bonus = 0.22
        risk += bonus
        factors.append(("Frequency: >10 tx/24h", bonus, "HIGH"))
    elif addr_24h_count > 5:
        bonus = 0.12
        risk += bonus
        factors.append(("Frequency: >5 tx/24h", bonus, "MEDIUM"))

    # ── Layer 3: Combo Detection (V4 — multiplicative) ──
    combo_multiplier = 1.0
    combo_reasons = []

    if rapid and out_degree > 3:
        combo_multiplier *= 1.6
        combo_reasons.append("Rapid + Fan-out")
    if mixing and chain_hop:
        combo_multiplier *= 1.4
        combo_reasons.append("Mixing + Cross-chain")
    if rapid and addr_24h_count > 5:
        combo_multiplier *= 1.2
        combo_reasons.append("Rapid + High frequency")
    if tx_amount < 1 and addr_24h_count > 10:
        combo_multiplier *= 1.3
        combo_reasons.append("Structuring (small amounts + high freq)")

    if combo_multiplier > 1.0:
        old_risk = risk
        risk *= combo_multiplier
        bonus = risk - old_risk
        factors.append((f"Combo: {' + '.join(combo_reasons)}", bonus, "HIGH"))

    # ── Layer 4: Graph Neighborhood (V5) ──
    if neighbor_illicit_pct > 30:
        bonus = 0.15
        risk += bonus
        factors.append(("Neighborhood: >30% illicit neighbors", bonus, "HIGH"))
    elif neighbor_illicit_pct > 15:
        bonus = 0.08
        risk += bonus
        factors.append(("Neighborhood: >15% illicit neighbors", bonus, "MEDIUM"))

    risk = min(0.98, max(0.02, risk))

    if not factors:
        factors.append(("No significant risk factors", 0.0, "LOW"))

    return risk, factors


def render(DATA, navigate_to):
    alerts = DATA["alerts"]

    if st.session_state.get("drill_from"):
        st.markdown(f'<div class="breadcrumb">← from {st.session_state["drill_from"]}</div>', unsafe_allow_html=True)

    st.markdown("# 🔍 Transaction Scanner")
    st.markdown("Multi-layer fraud risk assessment with anti-evasion detection")
    st.caption("⚠️ Transaction data below is simulated for demo purposes. Model metrics are from real experiments.")

    # Show auto-loaded alert context
    alert_tx = st.session_state.get("selected_alert_tx")
    if alert_tx:
        alert_match = next((a for a in alerts if a["tx_id"] == alert_tx), None)
        if alert_match:
            st.markdown(
                f'<div class="risk-high" style="display:flex; align-items:center; gap:16px">'
                f'<div style="font-size:2rem">🚨</div>'
                f'<div>'
                f'<strong style="color:#EF4444; font-size:1rem">Alert Auto-Loaded from Executive</strong><br>'
                f'<span style="color:#E5E7EB">TX: <code>{alert_tx}</code></span><br>'
                f'<span style="color:#F59E0B">Risk: {alert_match["risk_score"]:.0%}</span> · '
                f'<span style="color:#9CA3AF">Pattern: {alert_match["pattern"]}</span> · '
                f'<span style="color:#9CA3AF">TS: {alert_match["timestep"]}</span> · '
                f'<span style="color:#9CA3AF">{alert_match["amount_btc"]:.3f} BTC</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    elif st.session_state.get("selected_risk_level", "ALL") != "ALL":
        st.markdown(f"**Risk Filter:** {st.session_state.get('selected_risk_level', 'ALL')}")
    st.markdown("---")

    alert_data = next((a for a in alerts if a["tx_id"] == alert_tx), None) if alert_tx else None

    sc1, sc2 = st.columns([1, 1.5])

    with sc1:
        st.markdown("### Layer 1: Transaction Features")
        tx_amount = st.slider("Amount (BTC)", 0.01, 100.0,
                              alert_data["amount_btc"] if alert_data else 1.5, 0.01, key="scan_amt")
        in_degree = st.number_input("Input Count", 1, 50, 3, key="scan_in")
        out_degree = st.number_input("Output Count", 1, 50, 2, key="scan_out")
        timestep = st.slider("Timestep", 1, 49,
                             st.session_state.get("selected_timestep", 25), key="scan_ts")

        st.markdown("#### Behavior Flags")
        mixing = st.checkbox("Mixing service pattern",
                             value=alert_data["pattern"] == "Mixing" if alert_data else False)
        rapid = st.checkbox("Rapid succession (<10 min)",
                            value=alert_data["pattern"] == "Rapid Cycling" if alert_data else False)
        chain_hop = st.checkbox("Cross-chain bridge",
                                value=alert_data["pattern"] == "Chain Hop" if alert_data else False)

        st.markdown("---")
        st.markdown("### Layer 2: Address History (24h)")
        addr_24h_volume = st.slider("24h cumulative volume (BTC)", 0.0, 200.0, tx_amount, 0.1, key="scan_vol")
        addr_24h_count = st.slider("24h transaction count", 1, 50, 1, key="scan_freq")

        st.markdown("---")
        st.markdown("### Layer 4: Graph Context")
        neighbor_illicit_pct = st.slider("Neighborhood illicit %", 0, 100, 5, key="scan_neigh")

        analyze = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True, key="scan_go")

    with sc2:
        if analyze or alert_data:
            # Use alert_data's risk if auto-loaded, otherwise compute
            if alert_data and not analyze:
                risk = alert_data["risk_score"]
                factors = [("Alert pre-scored by TH-GNN", alert_data["risk_score"], "HIGH")]
            else:
                risk, factors = compute_risk(
                    tx_amount, in_degree, out_degree, mixing, rapid, chain_hop,
                    addr_24h_volume, addr_24h_count, neighbor_illicit_pct,
                )

            level = "HIGH" if risk > 0.7 else ("MEDIUM" if risk > 0.4 else "LOW")
            color = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#00D4AA"}[level]
            css = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[level]

            st.markdown(f'<div class="{css}" style="text-align:center; padding:20px">'
                        f'<h2 style="color:{color}; margin:0">{level} RISK</h2>'
                        f'<h1 style="color:{color}; margin:0; font-size:3rem">{risk:.0%}</h1>'
                        f'<p style="color:#9CA3AF; margin:4px 0 0 0; font-size:0.8rem">'
                        f'Engine v2.0 | {len(factors)} factor{"s" if len(factors) != 1 else ""} detected</p>'
                        f'</div>', unsafe_allow_html=True)

            # Gauge
            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=risk*100,
                number=dict(suffix="%", font=dict(color="#E5E7EB")),
                gauge=dict(axis=dict(range=[0,100]), bar=dict(color=color), bgcolor="rgba(255,255,255,0.05)",
                           steps=[dict(range=[0,40], color="rgba(16,185,129,0.1)"),
                                  dict(range=[40,70], color="rgba(245,158,11,0.1)"),
                                  dict(range=[70,100], color="rgba(239,68,68,0.1)")],
                           threshold=dict(line=dict(color="#EF4444", width=3), thickness=0.8, value=70))))
            fig_g.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB"),
                                margin=dict(l=30, r=30, t=30, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

            # Risk factor breakdown
            st.markdown("### Risk Factor Breakdown")
            st.markdown(
                '<div style="display:flex; gap:4px; margin-bottom:8px">'
                '<span class="badge badge-blue">L1: Transaction</span>'
                '<span class="badge badge-amber">L2: Address</span>'
                '<span class="badge badge-red">L3: Combo</span>'
                '<span class="badge badge-green">L4: Graph</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            for name, weight, sev in sorted(factors, key=lambda x: -x[1]):
                sc_color = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#00D4AA"}[sev]
                pct = f"+{weight:.0%}" if weight > 0 else "0%"
                st.markdown(
                    f'<div class="stat-row">'
                    f'<span style="color:#E5E7EB; font-size:0.9rem">{name}</span>'
                    f'<span style="color:{sc_color}; font-weight:700; font-family:JetBrains Mono,monospace">{pct}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Anti-evasion verdict
            combo_active = any("Combo" in f[0] for f in factors)
            addr_active = any("24h" in f[0] or "Frequency" in f[0] for f in factors)
            graph_active = any("Neighborhood" in f[0] for f in factors)

            if combo_active or addr_active or graph_active:
                st.markdown(
                    '<div class="risk-high" style="margin-top:12px">'
                    '<strong style="color:#EF4444">Anti-Evasion Detection Active</strong><br>'
                    '<span style="color:#E5E7EB">'
                    + ("Address aggregation triggered. " if addr_active else "")
                    + ("Behavioral combo detected. " if combo_active else "")
                    + ("Graph neighborhood flagged. " if graph_active else "")
                    + '</span></div>',
                    unsafe_allow_html=True,
                )

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
            st.markdown("#### Risk Engine v2.0 — 4 Detection Layers")
            st.markdown("""
| Layer | What it detects | Anti-evasion |
|-------|----------------|-------------|
| **L1: Transaction** | Mixing, rapid, cross-chain, high value | Individual flags |
| **L2: Address** | 24h cumulative volume, tx frequency | Catches structuring (small amount splitting) |
| **L3: Combo** | Multiplicative bonuses for combined tactics | Catches sophisticated multi-technique laundering |
| **L4: Graph** | Neighborhood illicit % from TH-GNN | Catches guilt-by-association |
            """)
            st.markdown(
                '<div class="risk-low" style="margin-top:12px">'
                '<strong style="color:#10B981">Key upgrade from v1.0</strong><br>'
                '<span style="color:#E5E7EB">'
                'v1.0 used additive single-transaction rules — a launderer could split into small amounts '
                'and bypass detection. v2.0 adds address-level aggregation, frequency analysis, '
                'multiplicative combo detection, and graph-level neighborhood risk propagation.</span>'
                '</div>',
                unsafe_allow_html=True,
            )
