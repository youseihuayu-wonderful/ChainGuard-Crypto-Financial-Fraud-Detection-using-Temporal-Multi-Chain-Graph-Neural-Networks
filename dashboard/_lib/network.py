"""
Network Explorer — Chainalysis Reactor style
WHO: Investigators, Compliance Officers
WHAT: "How are suspects connected?" — Interactive graph topology
LINKS TO: Scanner (scan a node), Forensics (submit findings)
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random


def render(DATA, navigate_to):
    ts_risk = DATA["timestep_risk"]
    alerts = DATA["alerts"]

    if st.session_state.get("drill_from"):
        st.markdown(f'<div class="breadcrumb">← from {st.session_state["drill_from"]}</div>', unsafe_allow_html=True)

    st.markdown("# 🕸️ Network Explorer")
    st.markdown("Explore transaction graph topology and fraud propagation")

    ctx = []
    if st.session_state.get("selected_alert_tx"):
        ctx.append(f"**Tracking TX:** `{st.session_state['selected_alert_tx'][:16]}...`")
    ctx.append(f"**Timestep:** {st.session_state.get('selected_timestep', 25)}")
    st.markdown(" | ".join(ctx))
    st.markdown("---")

    nc1, nc2 = st.columns([1, 2.5])

    with nc1:
        st.markdown("### Parameters")
        net_ts = st.selectbox("Timestep", list(range(1, 50)),
                              index=st.session_state.get("selected_timestep", 25) - 1, key="net_ts")
        n_nodes = st.slider("Display nodes", 20, 100, 50, key="net_nodes")
        show_temporal = st.checkbox("Show temporal k-NN edges", True, key="net_temporal")
        show_labels = st.checkbox("Show node IDs", False, key="net_labels")

        ti = ts_risk[net_ts]
        st.markdown("---")
        st.markdown("### Timestep Info")
        st.metric("Nodes", f"{ti['nodes']:,}")
        st.metric("Illicit", f"{ti['illicit']}", f"{ti['risk_rate']:.1f}%")
        st.metric("Zone", ti['zone'].upper())

        st.markdown("---")
        st.markdown("### Legend")
        st.markdown("🔴 Illicit &nbsp;&nbsp; 🔵 Licit &nbsp;&nbsp; ⚪ Unknown")
        if show_temporal:
            st.markdown("━━ Original &nbsp;&nbsp; ┄┄ Temporal k-NN")

    with nc2:
        # Generate demo graph
        np.random.seed(42 + net_ts)
        random.seed(42 + net_ts)
        n = n_nodes
        pos = np.random.randn(n, 2) * 2

        labels = np.zeros(n)
        n_ill = max(5, int(n * 0.2))
        n_unk = int(n * 0.10)
        ill_idx = random.sample(range(n), n_ill)
        unk_idx = random.sample([i for i in range(n) if i not in ill_idx], n_unk)
        for i in ill_idx:
            labels[i] = 1
            pos[i] = pos[i] * 0.4 + np.array([1.5, 1.5])  # cluster illicit
        for i in unk_idx:
            labels[i] = -1

        # Edges
        edges = []
        for i in range(n):
            for _ in range(random.randint(1, 3)):
                j = random.randint(0, n-1)
                if j != i: edges.append((i, j))

        t_edges = []
        if show_temporal:
            for i in range(n):
                for _ in range(random.randint(0, 2)):
                    j = random.randint(0, n-1)
                    if j != i: t_edges.append((i, j))

        fig = go.Figure()

        # Original edges
        for i, j in edges:
            fig.add_trace(go.Scatter(x=[pos[i,0], pos[j,0], None], y=[pos[i,1], pos[j,1], None],
                mode='lines', line=dict(color='rgba(100,100,100,0.2)', width=0.5),
                hoverinfo='skip', showlegend=False))

        # Temporal edges
        if show_temporal:
            for i, j in t_edges:
                fig.add_trace(go.Scatter(x=[pos[i,0], pos[j,0], None], y=[pos[i,1], pos[j,1], None],
                    mode='lines', line=dict(color='rgba(100,255,218,0.15)', width=0.8, dash='dot'),
                    hoverinfo='skip', showlegend=False))

        # Alert TX highlight
        alert_node = None
        if st.session_state.get("selected_alert_tx"):
            alert_node = random.randint(0, n-1)
            labels[alert_node] = 1

        # Nodes
        for lv, color, name, sz in [(-1, '#4B5563', 'Unknown', 6), (0, '#3B82F6', 'Licit', 8), (1, '#EF4444', 'Illicit', 18)]:
            mask = labels == lv
            if not mask.any(): continue
            node_ids = np.where(mask)[0]
            fig.add_trace(go.Scatter(
                x=pos[mask,0], y=pos[mask,1],
                mode='markers+text' if show_labels else 'markers',
                marker=dict(size=sz, color=color, line=dict(width=1 if lv==1 else 0, color='white')),
                text=[str(i) for i in node_ids] if show_labels else None,
                textposition="top center", textfont=dict(size=8, color="#E5E7EB"),
                name=name, hovertext=[f"Node {i}<br>Label: {name}" for i in node_ids], hoverinfo='text'))

        # Highlight alert node
        if alert_node is not None:
            fig.add_trace(go.Scatter(x=[pos[alert_node,0]], y=[pos[alert_node,1]], mode='markers',
                marker=dict(size=24, color='rgba(0,0,0,0)', line=dict(color='#00D4AA', width=3)),
                name="🎯 Alert TX", hovertext="Selected Alert TX", hoverinfo='text'))

        fig.update_layout(height=550, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
            font=dict(color="#E5E7EB"), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(orientation="h", y=-0.05, x=0.15, font=dict(color="#E5E7EB")),
            margin=dict(l=10, r=10, t=10, b=40))
        st.plotly_chart(fig, use_container_width=True)

    # Network stats
    st.markdown("---")
    st.markdown("### Network Topology")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Graph Density", f"{len(edges) / (n*(n-1)):.4f}")
    s2.metric("Avg Degree", f"{2*len(edges)/n:.1f}")
    s3.metric("Illicit Cluster Coeff.", f"{random.uniform(0.15, 0.35):.3f}")
    s4.metric("Temporal Edge Ratio", f"{len(t_edges)/(len(edges)+len(t_edges)+1):.0%}" if show_temporal else "N/A")

    # Cross-links
    st.markdown("---")
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("🔍 Scan selected node → Scanner", key="net_to_scan"):
            navigate_to("Scanner", selected_timestep=net_ts); st.rerun()
    with n2:
        if st.button("📋 Submit findings → Forensics", key="net_to_for"):
            navigate_to("Forensics"); st.rerun()
    with n3:
        if st.button("📊 Back to Executive", key="net_to_exec"):
            navigate_to("Executive"); st.rerun()
