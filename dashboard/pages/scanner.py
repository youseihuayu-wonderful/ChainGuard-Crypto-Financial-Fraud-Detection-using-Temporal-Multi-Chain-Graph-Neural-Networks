"""Page 3: Transaction Scanner - Interactive fraud risk assessment demo."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random


def render():
    st.markdown("# 🔍 Transaction Scanner")
    st.markdown("Analyze individual Bitcoin transactions for fraud risk using TH-GNN")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("### Input Transaction Features")

        tx_amount = st.slider("Transaction Amount (BTC)", 0.001, 100.0, 1.5, 0.01)
        tx_fee = st.slider("Transaction Fee (BTC)", 0.0001, 0.1, 0.005, 0.0001)
        in_degree = st.number_input("Input Count (in-degree)", 1, 50, 3)
        out_degree = st.number_input("Output Count (out-degree)", 1, 50, 2)
        timestep = st.slider("Timestep (1-49)", 1, 49, 25)

        st.markdown("#### Transaction Behavior")
        mixing = st.checkbox("Mixing service pattern detected")
        rapid = st.checkbox("Rapid succession transactions (< 10 min)")
        cross_chain = st.checkbox("Cross-chain bridge interaction")
        high_value = tx_amount > 10.0

        analyze = st.button("🔍 Analyze Transaction", use_container_width=True, type="primary")

    with col2:
        if analyze:
            # Simulated risk scoring (weighted heuristic for demo)
            np.random.seed(42)
            base_risk = 0.15

            # Risk factors
            if mixing:
                base_risk += 0.25
            if rapid:
                base_risk += 0.15
            if cross_chain:
                base_risk += 0.10
            if high_value:
                base_risk += 0.08
            if in_degree > 10:
                base_risk += 0.10
            if out_degree > 10:
                base_risk += 0.05
            if tx_fee < 0.001:
                base_risk += 0.05

            # Add some noise
            risk_score = min(0.98, max(0.02, base_risk + random.gauss(0, 0.05)))

            # Determine risk level
            if risk_score > 0.7:
                level, color, icon = "HIGH", "#ff5252", "🔴"
                css_class = "risk-high"
            elif risk_score > 0.4:
                level, color, icon = "MEDIUM", "#ff9800", "🟡"
                css_class = "risk-medium"
            else:
                level, color, icon = "LOW", "#64ffda", "🟢"
                css_class = "risk-low"

            st.markdown("### Analysis Result")

            st.markdown(
                f'<div class="{css_class}" style="text-align:center; padding:20px;">'
                f'<h2 style="color:{color}; margin:0">{icon} {level} RISK</h2>'
                f'<h1 style="color:{color}; margin:0; font-size:3rem">{risk_score:.1%}</h1>'
                f'<p style="color:#ccd6f6">Fraud Probability Score</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                number=dict(suffix="%", font=dict(color="#ccd6f6")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#8892b0"),
                    bar=dict(color=color),
                    bgcolor="rgba(255,255,255,0.05)",
                    steps=[
                        dict(range=[0, 40], color="rgba(100,255,218,0.15)"),
                        dict(range=[40, 70], color="rgba(255,152,0,0.15)"),
                        dict(range=[70, 100], color="rgba(255,82,82,0.15)"),
                    ],
                    threshold=dict(line=dict(color="#ff5252", width=3), thickness=0.8, value=70),
                ),
            ))
            fig.update_layout(
                height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccd6f6"),
                margin=dict(l=30, r=30, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Risk breakdown
            st.markdown("### Risk Factor Breakdown")
            factors = []
            if mixing:
                factors.append(("Mixing service pattern", 0.25, "HIGH"))
            if rapid:
                factors.append(("Rapid succession", 0.15, "MEDIUM"))
            if cross_chain:
                factors.append(("Cross-chain bridge", 0.10, "MEDIUM"))
            if high_value:
                factors.append(("High value (>10 BTC)", 0.08, "LOW"))
            if in_degree > 10:
                factors.append(("High input count", 0.10, "MEDIUM"))
            if not factors:
                factors.append(("No significant risk factors", 0.0, "LOW"))

            for name, weight, severity in factors:
                sev_color = {"HIGH": "#ff5252", "MEDIUM": "#ff9800", "LOW": "#64ffda"}[severity]
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"padding:8px 12px; background:rgba(255,255,255,0.03); "
                    f"border-radius:4px; margin:4px 0;'>"
                    f"<span style='color:#ccd6f6'>{name}</span>"
                    f"<span style='color:{sev_color}; font-weight:bold'>+{weight:.0%}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown(
                "<small style='color:#4a5568'>"
                "Note: This is a demonstration scanner using simplified heuristics. "
                "In production, the full TH-GNN model processes 166-dimensional node features "
                "and 2.19M graph edges for accurate prediction."
                "</small>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("### 👈 Configure transaction parameters and click **Analyze**")
            st.markdown("---")
            st.markdown("#### How TH-GNN Scores Transactions")
            st.markdown("""
            1. **Feature Extraction** — 166 transaction features (amount, fee, degree, timing...)
            2. **Graph Construction** — Build transaction graph + temporal k-NN edges
            3. **R-GCN Processing** — Separate message passing for original vs temporal edges
            4. **Risk Scoring** — Output probability of illicit activity [0, 1]
            """)

            st.markdown("#### Risk Thresholds")
            st.markdown("""
            | Level | Probability | Action |
            |-------|------------|--------|
            | 🟢 LOW | < 40% | Normal processing |
            | 🟡 MEDIUM | 40-70% | Enhanced monitoring |
            | 🔴 HIGH | > 70% | Immediate review |
            """)
