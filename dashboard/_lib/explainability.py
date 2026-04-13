"""
GNN Explainability — Real Model Outputs
WHO: Data Scientists, Researchers
WHAT: "Why did the model flag this node?" — Feature importance, node explanations

DATA SOURCE: ALL visualizations use REAL outputs from the trained M3 model:
- Feature importance: gradient-based attribution (∂loss/∂input)
- Node explanations: gradient × feature_value per node
- Neighbor influence: actual edge connections and predictions from the model
- Training history: real loss/AUC curves from training

NO simulated or mock data.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from _lib.i18n import t
from _lib.model_serving import (
    _model_available,
    load_feature_importance,
    load_node_explanations,
    load_predictions,
    load_training_history,
    load_statistical_tests,
)


def render(DATA, navigate_to):
    if st.session_state.get("drill_from"):
        st.markdown(f'<div class="breadcrumb">← from {st.session_state["drill_from"]}</div>',
                    unsafe_allow_html=True)

    st.markdown(f"# 🧠 {t('explain_title')}")
    st.markdown(t("expl_real_model_subtitle"))

    if not _model_available():
        st.error(t("expl_model_not_available"))
        st.code("cd chainguard-repo && .venv/bin/python experiments/scripts/train_and_save_m3.py", language="bash")
        return

    st.caption(t("expl_all_real_caption"))
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        t("feature_importance"), t("node_explanations_tab"), t("training_history_tab"), t("statistical_tests_tab")
    ])

    # ══════════════════════════════════════════
    # Tab 1: Feature Importance (gradient-based)
    # ══════════════════════════════════════════
    with tab1:
        feat_imp = load_feature_importance()
        if feat_imp is None:
            st.warning(t("feature_imp_not_found"))
            return

        st.markdown(f"### {t('gradient_feature_importance')}")
        st.caption(f"Method: {feat_imp['method']} | Model: {feat_imp['model']} | {feat_imp['n_features']} features")

        # Top 20 features
        top_n = 25
        features = feat_imp["features"][:top_n]
        names = [f["name"] for f in features]
        importances = [f["importance"] for f in features]

        fig = go.Figure(go.Bar(
            y=names[::-1], x=importances[::-1],
            orientation='h',
            marker_color=["#00D4AA" if i < 5 else "#3B82F6" if i < 10 else "#6B7280"
                          for i in range(len(names))][::-1],
            text=[f"{v:.4f}" for v in importances[::-1]],
            textposition="outside",
            textfont=dict(color="#E5E7EB", size=10),
        ))
        fig.update_layout(
            height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
            font=dict(color="#E5E7EB"), margin=dict(l=120, r=60, t=10, b=40),
            xaxis=dict(title="Mean |Gradient|", color="#9CA3AF", gridcolor="rgba(75,85,99,0.3)"),
            yaxis=dict(color="#E5E7EB"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        top3 = features[:3]
        st.markdown(
            f'<div class="risk-low"><strong style="color:#00D4AA">{t("interpretation")}</strong><br>'
            f'<span style="color:#E5E7EB">The top 3 most important features are: '
            f'<b>{top3[0]["name"]}</b> ({top3[0]["importance"]:.4f}), '
            f'<b>{top3[1]["name"]}</b> ({top3[1]["importance"]:.4f}), '
            f'<b>{top3[2]["name"]}</b> ({top3[2]["importance"]:.4f}). '
            f'These features have the highest average absolute gradient across all test nodes, '
            f'meaning changes to these features most strongly affect the model\'s fraud predictions.</span></div>',
            unsafe_allow_html=True)

        # Local vs Aggregated breakdown
        n_local_in_top = sum(1 for f in features[:15] if f["name"].startswith("local_"))
        n_agg_in_top = sum(1 for f in features[:15] if f["name"].startswith("agg_"))
        st.markdown(t("top15_breakdown").format(n_local=n_local_in_top, n_agg=n_agg_in_top))
        st.caption(t("local_features_desc"))

    # ══════════════════════════════════════════
    # Tab 2: Node Explanations
    # ══════════════════════════════════════════
    with tab2:
        explanations = load_node_explanations()
        if explanations is None:
            st.warning(t("node_expl_not_found"))
            return

        st.markdown(f"### {t('per_node_explanations')}")
        st.caption(t("per_node_caption").format(n=len(explanations)))

        # Node selector
        node_options = [
            f"Node {e['node_id']} | Risk: {e['risk_score']:.2%} | "
            f"{'ILLICIT' if e['true_label'] == 1 else 'LICIT'} | TS {e['timestep']}"
            for e in explanations
        ]
        selected_idx = st.selectbox(t("select_node_explain"), range(len(node_options)),
                                     format_func=lambda i: node_options[i], key="expl_node")

        expl = explanations[selected_idx]

        # Node info
        c1, c2, c3, c4 = st.columns(4)
        label_color = "#EF4444" if expl["true_label"] == 1 else "#10B981"
        label_text = "ILLICIT" if expl["true_label"] == 1 else "LICIT"
        c1.metric(t("node_id"), expl["node_id"])
        c2.metric(t("risk_score_label"), f"{expl['risk_score']:.2%}")
        c3.metric(t("true_label_label"), label_text)
        c4.metric(t("timestep"), expl["timestep"])

        # Feature contributions (waterfall chart)
        st.markdown(f"#### {t('feature_contributions_gradient')}")
        st.caption(t("positive_illicit_desc"))

        top_feats = expl["top_features"]
        feat_names = [f["feature_name"] for f in top_feats]
        contributions = [f["contribution"] for f in top_feats]
        colors = ["#EF4444" if c > 0 else "#10B981" for c in contributions]

        fig_wf = go.Figure(go.Bar(
            y=feat_names[::-1], x=contributions[::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f"{c:+.4f}" for c in contributions[::-1]],
            textposition="outside",
            textfont=dict(color="#E5E7EB", size=10),
        ))
        fig_wf.update_layout(
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
            font=dict(color="#E5E7EB"), margin=dict(l=100, r=60, t=10, b=40),
            xaxis=dict(title="Contribution (gradient × value)", color="#9CA3AF",
                       gridcolor="rgba(75,85,99,0.3)", zeroline=True,
                       zerolinecolor="rgba(255,255,255,0.2)"),
            yaxis=dict(color="#E5E7EB"),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        # Feature details table
        st.markdown(f"#### {t('feature_details')}")
        feat_df = pd.DataFrame([{
            "Feature": f["feature_name"],
            "Value": f"{f['value']:.4f}",
            "Gradient": f"{f['gradient']:.6f}",
            "Contribution": f"{f['contribution']:+.6f}",
        } for f in top_feats])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # Neighbor influence
        if expl.get("neighbors"):
            st.markdown(f"#### {t('neighbor_influence_real')}")
            st.caption(t("neighbor_caption"))

            neighbors = expl["neighbors"]
            n_illicit = sum(1 for n in neighbors if n["label"] == 1)
            n_licit = sum(1 for n in neighbors if n["label"] == 0)
            n_unknown = sum(1 for n in neighbors if n["label"] == -1)
            n_orig = sum(1 for n in neighbors if n["edge_type"] == 0)
            n_temp = sum(1 for n in neighbors if n["edge_type"] == 1)

            nc1, nc2, nc3, nc4, nc5 = st.columns(5)
            nc1.metric(t("total_neighbors"), len(neighbors))
            nc2.metric(t("illicit"), n_illicit, f"{n_illicit/max(len(neighbors),1):.0%}")
            nc3.metric(t("licit_label"), n_licit)
            nc4.metric(t("original_edges"), n_orig)
            nc5.metric(t("temporal_edges"), n_temp)

            # Neighbor bar chart
            fig_nb = go.Figure()
            for n in neighbors:
                color = "#EF4444" if n["label"] == 1 else ("#10B981" if n["label"] == 0 else "#6B7280")
                edge_label = "original" if n["edge_type"] == 0 else "temporal"
                label_text = "illicit" if n["label"] == 1 else ("licit" if n["label"] == 0 else "unknown")
                fig_nb.add_trace(go.Bar(
                    x=[f"N{n['node_id']}"], y=[n["risk_score"]],
                    marker_color=color, showlegend=False,
                    hovertext=f"Node {n['node_id']}<br>Label: {label_text}<br>"
                              f"Risk: {n['risk_score']:.2%}<br>Edge: {edge_label}<br>TS: {n['timestep']}",
                    hoverinfo="text",
                ))

            fig_nb.update_layout(
                height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                font=dict(color="#E5E7EB"), margin=dict(l=40, r=20, t=10, b=40),
                yaxis=dict(title="Risk Score", range=[0, 1], color="#9CA3AF",
                           gridcolor="rgba(75,85,99,0.3)"),
                xaxis=dict(color="#9CA3AF"),
            )
            st.plotly_chart(fig_nb, use_container_width=True)

    # ══════════════════════════════════════════
    # Tab 3: Training History
    # ══════════════════════════════════════════
    with tab3:
        history = load_training_history()
        if history is None:
            st.warning(t("training_hist_not_found"))
            return

        st.markdown(f"### {t('training_curves')}")
        st.caption(f"Seed: 42 | {len(history)} evaluation checkpoints | Early stopping with patience=20")

        epochs = [h["epoch"] for h in history]
        losses = [h["loss"] for h in history]
        val_aucs = [h["val_auc"] for h in history]
        val_f1s = [h["val_f1"] for h in history]

        # Loss curve
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=losses, mode='lines+markers',
                                       name='Training Loss', line=dict(color="#EF4444", width=2),
                                       marker=dict(size=4)))
        fig_loss.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
            font=dict(color="#E5E7EB"), margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(title="Epoch", color="#9CA3AF", gridcolor="rgba(75,85,99,0.3)"),
            yaxis=dict(title="BCE Loss", color="#9CA3AF", gridcolor="rgba(75,85,99,0.3)"),
            title=dict(text=t("training_loss"), font=dict(size=14, color="#E5E7EB")),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # AUC + F1 curves
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Scatter(x=epochs, y=val_aucs, mode='lines+markers',
                                          name='Val AUC-ROC', line=dict(color="#00D4AA", width=2),
                                          marker=dict(size=4)))
        fig_metrics.add_trace(go.Scatter(x=epochs, y=val_f1s, mode='lines+markers',
                                          name='Val F1', line=dict(color="#3B82F6", width=2),
                                          marker=dict(size=4)))
        best_epoch = epochs[np.argmax(val_aucs)]
        best_auc = max(val_aucs)
        fig_metrics.add_annotation(x=best_epoch, y=best_auc,
                                    text=f"Best: {best_auc:.4f}", showarrow=True,
                                    arrowcolor="#00D4AA", font=dict(color="#00D4AA", size=11))
        fig_metrics.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
            font=dict(color="#E5E7EB"), margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(title="Epoch", color="#9CA3AF", gridcolor="rgba(75,85,99,0.3)"),
            yaxis=dict(title="Score", color="#9CA3AF", gridcolor="rgba(75,85,99,0.3)", range=[0, 1]),
            title=dict(text=t("validation_metrics"), font=dict(size=14, color="#E5E7EB")),
            legend=dict(orientation="h", y=1.15, x=0.3),
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

        # Summary stats
        st.markdown(f"#### {t('training_summary')}")
        ts1, ts2, ts3, ts4 = st.columns(4)
        ts1.metric(t("best_val_auc"), f"{best_auc:.4f}")
        ts2.metric(t("best_epoch"), best_epoch)
        ts3.metric(t("final_loss"), f"{losses[-1]:.4f}")
        ts4.metric(t("total_epochs"), epochs[-1])

    # ══════════════════════════════════════════
    # Tab 4: Statistical Tests
    # ══════════════════════════════════════════
    with tab4:
        stat_tests = load_statistical_tests()
        if stat_tests is None:
            st.warning(t("stat_tests_not_found"))
            return

        st.markdown(f"### {t('stat_tests_title')}")
        st.caption(f"Method: {stat_tests['method']} | Seeds: {stat_tests['seeds']} | n={stat_tests['n_seeds']}")

        comparisons = stat_tests["comparisons"]

        # Results table
        rows = []
        for method, r in comparisons.items():
            sig_marker = "✅" if r["significant_005"] else ("⚠️" if r["significant_010"] else "❌")
            rows.append({
                "Comparison": f"M3 vs {method}",
                "M3 Mean AUC": f"{r['m3_mean']:.4f}",
                "Other Mean AUC": f"{r['other_mean']:.4f}",
                "Diff": f"{r['mean_diff']:+.4f}",
                "p-value": f"{r['p_value_ttest']:.4f}",
                "Significant (p<0.05)": sig_marker,
                "Effect Size": r["effect_size"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Visualize p-values
        fig_p = go.Figure()
        methods = list(comparisons.keys())
        p_values = [comparisons[m]["p_value_ttest"] for m in methods]
        colors_p = ["#10B981" if p < 0.05 else ("#F59E0B" if p < 0.10 else "#EF4444") for p in p_values]

        fig_p.add_trace(go.Bar(x=methods, y=p_values, marker_color=colors_p,
                                text=[f"p={p:.4f}" for p in p_values],
                                textposition="outside", textfont=dict(color="#E5E7EB")))
        fig_p.add_hline(y=0.05, line_dash="dash", line_color="#F59E0B",
                         annotation_text="α = 0.05", annotation_font_color="#F59E0B")
        fig_p.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
            font=dict(color="#E5E7EB"), margin=dict(l=40, r=20, t=30, b=40),
            yaxis=dict(title="p-value", color="#9CA3AF", gridcolor="rgba(75,85,99,0.3)"),
            xaxis=dict(color="#E5E7EB"),
            title=dict(text="Statistical Significance (paired t-test)", font=dict(size=14, color="#E5E7EB")),
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # Key finding
        sig_comparisons = [m for m, r in comparisons.items() if r["significant_005"]]
        non_sig = [m for m, r in comparisons.items() if not r["significant_005"]]

        st.markdown(
            f'<div class="risk-low"><strong style="color:#00D4AA">{t("key_findings")}</strong><br>'
            f'<span style="color:#E5E7EB">'
            f'{t("stat_sig_better").format(sig=", ".join(sig_comparisons) if sig_comparisons else "none")}<br>'
            f'{t("not_significant_vs").format(non_sig=", ".join(non_sig) if non_sig else "none")}<br>'
            f'Note: {stat_tests["note"]}</span></div>',
            unsafe_allow_html=True)

    # Cross-links
    st.markdown("---")
    n1, n2 = st.columns(2)
    with n1:
        if st.button(f"🔍 {t('try_scanner')}", key="expl_to_scan"):
            navigate_to("Scanner"); st.rerun()
    with n2:
        if st.button(f"📋 {t('evidence_to_forensics')}", key="expl_to_for"):
            navigate_to("Forensics"); st.rerun()
