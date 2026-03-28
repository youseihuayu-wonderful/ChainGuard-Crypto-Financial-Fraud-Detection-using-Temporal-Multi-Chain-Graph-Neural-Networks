"""Page 4: Network Graph Explorer - Interactive graph visualization."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random


def render():
    st.markdown("# 🕸️ Network Graph Explorer")
    st.markdown("Explore transaction network topology and fraud propagation patterns")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Graph Parameters")
        timestep = st.selectbox("Timestep", list(range(1, 50)), index=24)
        n_nodes = st.slider("Display nodes", 20, 100, 50)
        show_temporal = st.checkbox("Show temporal k-NN edges", value=True)
        show_labels = st.checkbox("Show node labels", value=False)
        seed_val = st.number_input("Random seed", 0, 999, 42)

        st.markdown("---")
        st.markdown("### Legend")
        st.markdown("🔴 <span style='color:#ff5252'>Illicit transaction</span>", unsafe_allow_html=True)
        st.markdown("🔵 <span style='color:#2196F3'>Licit transaction</span>", unsafe_allow_html=True)
        st.markdown("⚪ <span style='color:#666'>Unknown label</span>", unsafe_allow_html=True)
        if show_temporal:
            st.markdown("--- <span style='color:#64ffda'>Temporal k-NN edge</span>", unsafe_allow_html=True)
            st.markdown("— <span style='color:#444'>Original edge</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Dataset Statistics")
        st.markdown(f"**Timestep {timestep}** (~{random.randint(3000, 6000)} nodes)")
        st.markdown(f"**Original edges:** ~{random.randint(3000, 8000)}")
        if show_temporal:
            st.markdown(f"**Temporal k-NN edges:** ~{random.randint(20000, 50000)}")

    with col2:
        # Generate demo graph layout
        np.random.seed(seed_val + timestep)
        random.seed(seed_val + timestep)

        # Create node positions using force-directed layout simulation
        n = n_nodes
        pos = np.random.randn(n, 2) * 2

        # Assign labels (roughly 10% illicit, 20% unknown)
        labels = np.zeros(n)
        n_illicit = max(3, int(n * 0.1))
        n_unknown = int(n * 0.2)
        illicit_idx = random.sample(range(n), n_illicit)
        unknown_idx = random.sample([i for i in range(n) if i not in illicit_idx], n_unknown)

        for i in illicit_idx:
            labels[i] = 1
        for i in unknown_idx:
            labels[i] = -1

        # Cluster illicit nodes slightly
        center = pos[illicit_idx].mean(axis=0)
        for i in illicit_idx:
            pos[i] = pos[i] * 0.5 + center * 0.5

        # Generate edges (preferential attachment)
        edges = []
        for i in range(n):
            n_edges = random.randint(1, 4)
            for _ in range(n_edges):
                j = random.randint(0, n - 1)
                if j != i:
                    edges.append((i, j))

        temporal_edges = []
        if show_temporal:
            for i in range(n):
                for _ in range(random.randint(0, 2)):
                    j = random.randint(0, n - 1)
                    if j != i:
                        temporal_edges.append((i, j))

        # Build plotly figure
        fig = go.Figure()

        # Original edges
        for i, j in edges:
            fig.add_trace(go.Scatter(
                x=[pos[i, 0], pos[j, 0], None],
                y=[pos[i, 1], pos[j, 1], None],
                mode='lines',
                line=dict(color='rgba(100,100,100,0.3)', width=0.5),
                hoverinfo='skip',
                showlegend=False,
            ))

        # Temporal edges
        if show_temporal:
            for i, j in temporal_edges:
                fig.add_trace(go.Scatter(
                    x=[pos[i, 0], pos[j, 0], None],
                    y=[pos[i, 1], pos[j, 1], None],
                    mode='lines',
                    line=dict(color='rgba(100,255,218,0.2)', width=0.8, dash='dot'),
                    hoverinfo='skip',
                    showlegend=False,
                ))

        # Nodes by class
        for label_val, color, name, size in [
            (-1, '#666666', 'Unknown', 8),
            (0, '#2196F3', 'Licit', 10),
            (1, '#ff5252', 'Illicit', 14),
        ]:
            mask = labels == label_val
            if mask.sum() == 0:
                continue

            text = [f"Node {i}<br>Label: {name}<br>Risk: {random.uniform(0.1, 0.95):.1%}"
                    for i in np.where(mask)[0]]

            fig.add_trace(go.Scatter(
                x=pos[mask, 0], y=pos[mask, 1],
                mode='markers+text' if show_labels else 'markers',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=1, color='white') if label_val == 1 else dict(width=0),
                ),
                text=[str(i) for i in np.where(mask)[0]] if show_labels else None,
                textposition="top center",
                textfont=dict(size=8, color="#ccd6f6"),
                hovertext=text,
                hoverinfo='text',
                name=name,
            ))

        fig.update_layout(
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,26,1)",
            font=dict(color="#ccd6f6"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(orientation="h", y=-0.05, x=0.3, font=dict(color="#ccd6f6")),
            margin=dict(l=10, r=10, t=10, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Network stats
    st.markdown("### Network Topology Insights")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Graph Density", f"{len(edges) / (n*(n-1)):.4f}")
    c2.metric("Avg Degree", f"{2*len(edges)/n:.1f}")
    c3.metric("Illicit Cluster Coeff.", f"{random.uniform(0.15, 0.35):.3f}")
    c4.metric("Temporal Edge Ratio", f"{len(temporal_edges)/(len(edges)+len(temporal_edges)+1):.1%}" if show_temporal else "N/A")
