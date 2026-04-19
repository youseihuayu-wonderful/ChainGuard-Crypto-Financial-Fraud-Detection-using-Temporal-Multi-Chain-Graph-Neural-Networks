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
from _lib.model_serving import load_predictions, get_top_risk_nodes


def _get_risk_scores():
    """Build a node_id -> risk_score lookup from M3 predictions."""
    predictions = load_predictions()
    if not predictions or "test_predictions" not in predictions:
        return {}
    return {p["node_id"]: p["risk_score"] for p in predictions["test_predictions"]}


def _risk_color(score):
    """Map a 0-1 risk score to a hex color (green → yellow → red)."""
    if score > 0.8:
        return "#EF4444"
    elif score > 0.6:
        return "#F59E0B"
    elif score > 0.4:
        return "#FCD34D"
    else:
        return "#10B981"


def render(DATA, navigate_to):
    ts_risk = DATA["timestep_risk"]
    graph_data = DATA.get("graph_data", {})
    risk_scores = _get_risk_scores()

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
        color_by_risk = st.checkbox(t("net_color_by_risk"), True, key="net_risk_color")

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
        if color_by_risk:
            st.markdown(
                '<div style="font-size:0.8rem; line-height:1.8">'
                '<span style="color:#EF4444">\u25cf</span> High Risk (>80%) &nbsp;'
                '<span style="color:#F59E0B">\u25cf</span> Medium (60-80%) &nbsp;<br>'
                '<span style="color:#FCD34D">\u25cf</span> Low-Med (40-60%) &nbsp;'
                '<span style="color:#10B981">\u25cf</span> Low (<40%) &nbsp;<br>'
                '<span style="color:#4B5563">\u25cf</span> No prediction</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(t("legend_text"))

    with nc2:
        ts_key = str(net_ts)
        if ts_key not in graph_data:
            st.warning(t("graph_not_available"))
        else:
            gd = graph_data[ts_key]
            all_labels = gd["labels"]
            all_edges = gd["edges"]
            total_n = gd["n_nodes"]

            n = min(n_display, total_n)

            illicit_idx = [i for i, l in enumerate(all_labels) if l == "illicit"]
            licit_idx = [i for i, l in enumerate(all_labels) if l == "licit"]
            unknown_idx = [i for i, l in enumerate(all_labels) if l == "unknown"]

            selected = []
            selected.extend(illicit_idx[:n])
            remaining = n - len(selected)
            if remaining > 0:
                n_licit_add = min(len(licit_idx), int(remaining * 0.6))
                n_unknown_add = min(len(unknown_idx), remaining - n_licit_add)
                np.random.seed(42)
                if n_licit_add > 0:
                    selected.extend(np.random.choice(licit_idx, n_licit_add, replace=False).tolist())
                if n_unknown_add > 0:
                    selected.extend(np.random.choice(unknown_idx, n_unknown_add, replace=False).tolist())

            selected_set = set(selected)
            idx_map = {old: new for new, old in enumerate(selected)}
            n_actual = len(selected)

            node_labels = [all_labels[i] for i in selected]

            sub_edges = []
            for s, d in all_edges:
                if s in selected_set and d in selected_set:
                    sub_edges.append((idx_map[s], idx_map[d]))

            # Force-directed layout
            np.random.seed(42 + net_ts)
            pos = np.random.randn(n_actual, 2) * 3

            for _ in range(15):
                for i in range(n_actual):
                    for j in range(i + 1, min(n_actual, i + 50)):
                        diff = pos[i] - pos[j]
                        dist = max(np.linalg.norm(diff), 0.1)
                        force = diff / (dist ** 2) * 0.5
                        pos[i] += force
                        pos[j] -= force
                for s, d in sub_edges[:500]:
                    diff = pos[d] - pos[s]
                    dist = np.linalg.norm(diff)
                    if dist > 0.5:
                        force = diff * 0.02
                        pos[s] += force
                        pos[d] -= force

            fig = go.Figure()

            # Draw edges
            edge_x, edge_y = [], []
            for s, d in sub_edges:
                edge_x.extend([pos[s, 0], pos[d, 0], None])
                edge_y.extend([pos[s, 1], pos[d, 1], None])
            if edge_x:
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                    mode='lines', line=dict(color='rgba(100,100,100,0.15)', width=0.5),
                    hoverinfo='skip', showlegend=False))

            if color_by_risk and risk_scores:
                # Color by risk score (gradient)
                x_vals = [pos[i, 0] for i in range(n_actual)]
                y_vals = [pos[i, 1] for i in range(n_actual)]
                real_ids = [selected[i] for i in range(n_actual)]
                scores = [risk_scores.get(rid, -1) for rid in real_ids]
                colors = []
                sizes = []
                hover_texts = []
                for i, (rid, score, label) in enumerate(zip(real_ids, scores, node_labels)):
                    if score >= 0:
                        colors.append(_risk_color(score))
                        sizes.append(max(6, int(score * 18)))
                        hover_texts.append(f"Node {rid}<br>Risk: {score:.2%}<br>Label: {label}")
                    else:
                        colors.append('#4B5563')
                        sizes.append(5)
                        hover_texts.append(f"Node {rid}<br>Label: {label}<br>No prediction")

                text_labels = [str(rid) for rid in real_ids] if show_labels else None

                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='markers+text' if show_labels else 'markers',
                    marker=dict(size=sizes, color=colors,
                                line=dict(width=0.5, color='rgba(255,255,255,0.3)')),
                    text=text_labels,
                    textposition="top center", textfont=dict(size=7, color="#E5E7EB"),
                    name="Nodes", hovertext=hover_texts, hoverinfo='text',
                    showlegend=False))
            else:
                # Original label-based coloring
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
            st.plotly_chart(fig, width="stretch")

            st.caption(f"Showing {n_actual} of {total_n} nodes | {len(sub_edges)} of {gd['n_edges']} edges | Real Elliptic data")

    # ── Node Detail Panel ──
    st.markdown("---")
    st.markdown(f"### \U0001f50e {t('net_node_detail')}")

    top_nodes = get_top_risk_nodes(20)
    if top_nodes:
        node_options = [f"Node {n['node_id']} (Risk: {n['risk_score']:.2%})" for n in top_nodes]
        selected_node_idx = st.selectbox(
            t("net_select_node"), range(len(node_options)),
            format_func=lambda i: node_options[i],
            key="net_node_select",
        )
        node = top_nodes[selected_node_idx]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric(t("node_id"), node["node_id"])
        score_val = node["risk_score"]
        severity = "CRITICAL" if score_val > 0.95 else ("HIGH" if score_val > 0.85 else "MEDIUM")
        d2.metric(t("risk_score_label"), f"{score_val:.2%}")
        d3.metric(t("severity_label"), severity)
        d4.metric(t("true_label_label"), "ILLICIT" if node["true_label"] == 1 else "LICIT")

        # Action buttons for selected node
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button(f"\U0001f50d {t('scan_node')}", key="net_detail_scan"):
                navigate_to("Scanner"); st.rerun()
        with b2:
            if st.button(f"\U0001f50e {t('search_node_btn')}", key="net_detail_search"):
                st.session_state["search_prefill_node"] = node["node_id"]
                navigate_to("Search"); st.rerun()
        with b3:
            if st.button(f"\U0001f4c1 {t('net_create_case')}", key="net_detail_case"):
                st.session_state["prefill_case_node"] = node["node_id"]
                st.session_state["prefill_case_score"] = node["risk_score"]
                navigate_to("Cases"); st.rerun()
    else:
        st.info(t("net_no_predictions"))

    # Network topology stats
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
