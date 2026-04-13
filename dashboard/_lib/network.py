"""
Network Explorer — Chainalysis Reactor style
WHO: Investigators, Compliance Officers
WHAT: "How are suspects connected?" — Interactive graph topology using REAL Elliptic data
LINKS TO: Scanner (scan a node), Forensics (submit findings)

DATA SOURCE: All graph data is from the actual Elliptic Bitcoin Transaction Dataset.
Node labels and edges reflect the real transaction network.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from _lib.i18n import t


def render(DATA, navigate_to):
    ts_risk = DATA["timestep_risk"]
    graph_data = DATA.get("graph_data", {})

    if st.session_state.get("drill_from"):
        st.markdown(f'<div class="breadcrumb">\u2190 from {st.session_state["drill_from"]}</div>', unsafe_allow_html=True)

    st.markdown(f"# \U0001f578\ufe0f {t('network_title')}")
    st.markdown(t("network_subtitle"))

    ctx = []
    ctx.append(f"**{t('timestep')}:** {st.session_state.get('selected_timestep', 25)}")
    st.markdown(" | ".join(ctx))
    st.markdown("---")

    nc1, nc2 = st.columns([1, 2.5])

    with nc1:
        st.markdown(f"### {t('parameters')}")
        net_ts = st.selectbox(t("timestep"), list(range(1, 50)),
                              index=st.session_state.get("selected_timestep", 25) - 1, key="net_ts")
        n_display = st.slider(t("display_nodes"), 20, 200, 80, key="net_nodes")
        show_labels = st.checkbox(t("show_labels"), False, key="net_labels")

        ti = ts_risk[net_ts]
        st.markdown("---")
        st.markdown(f"### {t('timestep_info')}")
        st.metric(t("nodes"), f"{ti['nodes']:,}")
        st.metric(t("illicit"), f"{ti['illicit']}", f"{ti['risk_rate']:.1f}%")
        st.metric(t("licit_label"), f"{ti.get('licit', 0):,}")
        st.metric(t("unknown_label"), f"{ti.get('unknown', 0):,}")
        st.metric(t("zone"), ti['zone'].upper())
        if ti.get('edges'):
            st.metric(t("edges"), f"{ti['edges']:,}")

        st.markdown("---")
        st.markdown(f"### {t('legend')}")
        st.markdown(t("legend_text"))

    with nc2:
        ts_key = str(net_ts)
        if ts_key not in graph_data:
            st.warning(t("graph_not_available"))
        else:
            gd = graph_data[ts_key]
            all_labels = gd["labels"]  # real labels from Elliptic
            all_edges = gd["edges"]    # real edges from Elliptic
            total_n = gd["n_nodes"]

            # Select a subgraph to display (prioritize illicit nodes)
            n = min(n_display, total_n)

            # Find illicit, licit, unknown indices
            illicit_idx = [i for i, l in enumerate(all_labels) if l == "illicit"]
            licit_idx = [i for i, l in enumerate(all_labels) if l == "licit"]
            unknown_idx = [i for i, l in enumerate(all_labels) if l == "unknown"]

            # Include all illicit nodes, then fill with licit/unknown
            selected = []
            selected.extend(illicit_idx[:n])
            remaining = n - len(selected)
            if remaining > 0:
                # Add proportional licit and unknown
                n_licit_add = min(len(licit_idx), int(remaining * 0.6))
                n_unknown_add = min(len(unknown_idx), remaining - n_licit_add)
                np.random.seed(42)  # reproducible selection
                if n_licit_add > 0:
                    selected.extend(np.random.choice(licit_idx, n_licit_add, replace=False).tolist())
                if n_unknown_add > 0:
                    selected.extend(np.random.choice(unknown_idx, n_unknown_add, replace=False).tolist())

            selected_set = set(selected)
            idx_map = {old: new for new, old in enumerate(selected)}
            n_actual = len(selected)

            # Get labels for selected nodes
            node_labels = [all_labels[i] for i in selected]

            # Filter edges to selected subgraph
            sub_edges = []
            for s, d in all_edges:
                if s in selected_set and d in selected_set:
                    sub_edges.append((idx_map[s], idx_map[d]))

            # Layout using force-directed approximation (spring layout)
            np.random.seed(42 + net_ts)
            pos = np.random.randn(n_actual, 2) * 3

            # Simple force-directed refinement (10 iterations)
            for _ in range(15):
                # Repulsion between all nodes
                for i in range(n_actual):
                    for j in range(i + 1, min(n_actual, i + 50)):  # limit for speed
                        diff = pos[i] - pos[j]
                        dist = max(np.linalg.norm(diff), 0.1)
                        force = diff / (dist ** 2) * 0.5
                        pos[i] += force
                        pos[j] -= force
                # Attraction along edges
                for s, d in sub_edges[:500]:  # limit for speed
                    diff = pos[d] - pos[s]
                    dist = np.linalg.norm(diff)
                    if dist > 0.5:
                        force = diff * 0.02
                        pos[s] += force
                        pos[d] -= force

            fig = go.Figure()

            # Draw edges (batch as single trace for performance)
            edge_x, edge_y = [], []
            for s, d in sub_edges:
                edge_x.extend([pos[s, 0], pos[d, 0], None])
                edge_y.extend([pos[s, 1], pos[d, 1], None])
            if edge_x:
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                    mode='lines', line=dict(color='rgba(100,100,100,0.15)', width=0.5),
                    hoverinfo='skip', showlegend=False))

            # Draw nodes by label type
            label_config = {
                "unknown": ('#4B5563', 'Unknown', 5),
                "licit": ('#3B82F6', 'Licit', 7),
                "illicit": ('#EF4444', 'Illicit', 14),
            }
            for label_val, (color, name, sz) in label_config.items():
                mask = [i for i, l in enumerate(node_labels) if l == label_val]
                if not mask:
                    continue
                x_vals = [pos[i, 0] for i in mask]
                y_vals = [pos[i, 1] for i in mask]
                real_ids = [selected[i] for i in mask]
                hover = [f"Node {rid}<br>Label: {name} (real)" for rid in real_ids]
                text_labels = [str(rid) for rid in real_ids] if show_labels else None

                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='markers+text' if show_labels else 'markers',
                    marker=dict(size=sz, color=color,
                                line=dict(width=1 if label_val == "illicit" else 0, color='white')),
                    text=text_labels,
                    textposition="top center", textfont=dict(size=7, color="#E5E7EB"),
                    name=name, hovertext=hover, hoverinfo='text'))

            fig.update_layout(height=550, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
                font=dict(color="#E5E7EB"), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                legend=dict(orientation="h", y=-0.05, x=0.15, font=dict(color="#E5E7EB")),
                margin=dict(l=10, r=10, t=10, b=40))
            st.plotly_chart(fig, use_container_width=True)

            # Data source note
            st.caption(f"Showing {n_actual} of {total_n} nodes | {len(sub_edges)} of {gd['n_edges']} edges | Real Elliptic data")

    # Network stats (computed from real data)
    st.markdown("---")
    st.markdown(f"### {t('network_topology')}")

    ts_key = str(net_ts)
    if ts_key in graph_data:
        gd = graph_data[ts_key]
        total_n = gd["n_nodes"]
        total_e = gd["n_edges"]
        n_illicit = sum(1 for l in gd["labels"] if l == "illicit")
        n_licit = sum(1 for l in gd["labels"] if l == "licit")

        s1, s2, s3, s4 = st.columns(4)
        density = total_e / (total_n * (total_n - 1)) if total_n > 1 else 0
        avg_degree = 2 * total_e / total_n if total_n > 0 else 0
        illicit_ratio = n_illicit / total_n if total_n > 0 else 0
        s1.metric(t("graph_density"), f"{density:.6f}")
        s2.metric(t("avg_degree"), f"{avg_degree:.2f}")
        s3.metric(t("illicit_ratio"), f"{illicit_ratio:.2%}")
        s4.metric(t("licit_unknown"), f"{n_licit} / {total_n - n_illicit - n_licit}")

    # Cross-links
    st.markdown("---")
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button(f"\U0001f50d {t('scan_node')}", key="net_to_scan"):
            navigate_to("Scanner", selected_timestep=net_ts); st.rerun()
    with n2:
        if st.button(f"\U0001f4cb {t('submit_findings')}", key="net_to_for"):
            navigate_to("Forensics"); st.rerun()
    with n3:
        if st.button(f"\U0001f4ca {t('back_executive')}", key="net_to_exec"):
            navigate_to("Executive"); st.rerun()
