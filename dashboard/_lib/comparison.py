"""
Model Comparison Dashboard — Side-by-Side Analysis
WHO: Data Scientists, Researchers
WHAT: Compare any two models from ablation study with real experiment data.

Uses ONLY real data from ablation_results.json and baseline_comparison.json.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from shared import CHART_LAYOUT, COLORS
from _lib.i18n import t


METRICS = ["auc_roc", "f1", "precision", "recall"]
METRIC_LABELS = {
    "auc_roc": "AUC-ROC",
    "f1": "F1 Score",
    "precision": "Precision",
    "recall": "Recall",
}


def render(DATA, navigate_to):
    """Render the Model Comparison Dashboard page."""
    st.markdown(f"# :bar_chart: {t('comparison_title')}")
    st.markdown(t("comparison_subtitle"))
    st.caption(t("comparison_caption"))

    st.markdown("---")

    ablation = DATA["ablation"]
    baseline = DATA["baseline"]

    # Build combined model dictionary
    all_models = {}

    # Ablation models (M1-M5)
    for key, val in ablation.items():
        all_models[f"{key}: {val['name']}"] = {
            "auc_roc": val["auc_roc"],
            "f1": val["f1"],
            "precision": val["precision"],
            "recall": val["recall"],
            "source": "ablation",
        }

    # Baseline models
    for key, val in baseline.get("results", {}).items():
        if key != "thgnn_m3_ours":  # Skip duplicate
            display_name = key.replace("_", " ").title()
            all_models[display_name] = {
                "auc_roc": val["auc_roc"],
                "f1": val["f1"],
                "precision": val["precision"],
                "recall": val["recall"],
                "source": "baseline",
            }

    model_names = list(all_models.keys())

    # Model selection
    st.markdown(f"### {t('select_models_compare')}")
    sel_col1, sel_col2 = st.columns(2)

    with sel_col1:
        model_a_name = st.selectbox(
            t("model_a"),
            model_names,
            index=model_names.index("M3: R-GCN + Heterogeneous Edges") if "M3: R-GCN + Heterogeneous Edges" in model_names else 0,
            key="compare_model_a",
        )

    with sel_col2:
        default_b = model_names.index("M1: GCN Baseline") if "M1: GCN Baseline" in model_names else 1
        model_b_name = st.selectbox(
            t("model_b"),
            model_names,
            index=default_b,
            key="compare_model_b",
        )

    model_a = all_models[model_a_name]
    model_b = all_models[model_b_name]

    st.markdown("---")

    # Side-by-side metrics
    st.markdown(f"### {t('head_to_head')}")

    for metric in METRICS:
        label = METRIC_LABELS[metric]
        val_a = model_a[metric]
        val_b = model_b[metric]
        delta = val_a - val_b
        pct_delta = delta / max(val_b, 1e-8) * 100

        c1, c2, c3 = st.columns([1, 0.5, 1])

        # Determine winner
        winner_a = val_a >= val_b
        color_a = "#00D4AA" if winner_a else "#9CA3AF"
        color_b = "#00D4AA" if not winner_a else "#9CA3AF"
        arrow = "+" if delta > 0 else ""

        with c1:
            st.markdown(
                f'<div class="stat-row">'
                f'<span style="color:#9CA3AF; font-size:0.8rem">{label}</span>'
                f'<span style="color:{color_a}; font-weight:700; font-family:JetBrains Mono,monospace; font-size:1.1rem">'
                f'{val_a:.4f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c2:
            delta_color = "#10B981" if delta > 0 else ("#EF4444" if delta < 0 else "#6B7280")
            st.markdown(
                f'<div style="text-align:center; padding:10px">'
                f'<div style="color:{delta_color}; font-weight:700; font-family:JetBrains Mono,monospace">'
                f'{arrow}{delta:.4f}</div>'
                f'<div style="color:#6B7280; font-size:0.75rem">{arrow}{pct_delta:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f'<div class="stat-row">'
                f'<span style="color:{color_b}; font-weight:700; font-family:JetBrains Mono,monospace; font-size:1.1rem">'
                f'{val_b:.4f}</span>'
                f'<span style="color:#9CA3AF; font-size:0.8rem">{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Winner summary
    wins_a = sum(1 for m in METRICS if model_a[m] > model_b[m])
    wins_b = sum(1 for m in METRICS if model_b[m] > model_a[m])
    ties = 4 - wins_a - wins_b

    winner_text = model_a_name if wins_a > wins_b else (model_b_name if wins_b > wins_a else t("tie"))
    winner_color = "#00D4AA" if wins_a > wins_b else ("#3B82F6" if wins_b > wins_a else "#F59E0B")

    st.markdown(
        f'<div class="glass-card" style="text-align:center">'
        f'<div style="color:#6B7280; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em">{t("overall_winner")}</div>'
        f'<div style="color:{winner_color}; font-size:1.5rem; font-weight:700; margin:4px 0">{winner_text}</div>'
        f'<div style="color:#9CA3AF">{model_a_name}: {wins_a} {t("wins")} | {model_b_name}: {wins_b} {t("wins")} | {t("ties")}: {ties}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Bar chart comparison
    st.markdown(f"### {t('bar_chart_comparison')}")

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[METRIC_LABELS[m] for m in METRICS],
        y=[model_a[m] for m in METRICS],
        name=model_a_name,
        marker_color="#00D4AA",
        text=[f"{model_a[m]:.4f}" for m in METRICS],
        textposition="outside",
        textfont=dict(color="#E5E7EB", size=11),
    ))
    fig_bar.add_trace(go.Bar(
        x=[METRIC_LABELS[m] for m in METRICS],
        y=[model_b[m] for m in METRICS],
        name=model_b_name,
        marker_color="#3B82F6",
        text=[f"{model_b[m]:.4f}" for m in METRICS],
        textposition="outside",
        textfont=dict(color="#E5E7EB", size=11),
    ))
    fig_bar.update_layout(
        barmode="group",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.5)",
        font=dict(color="#E5E7EB"),
        title=dict(text=t("metric_comparison"), font=dict(size=14)),
        yaxis_title=t("score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown("---")

    # Radar chart
    st.markdown(f"### {t('radar_chart_overlay')}")

    categories = [METRIC_LABELS[m] for m in METRICS]
    # Close the radar polygon
    values_a = [model_a[m] for m in METRICS] + [model_a[METRICS[0]]]
    values_b = [model_b[m] for m in METRICS] + [model_b[METRICS[0]]]
    categories_closed = categories + [categories[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values_a,
        theta=categories_closed,
        fill="toself",
        name=model_a_name,
        line=dict(color="#00D4AA", width=2),
        fillcolor="rgba(0,212,170,0.15)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=values_b,
        theta=categories_closed,
        fill="toself",
        name=model_b_name,
        line=dict(color="#3B82F6", width=2),
        fillcolor="rgba(59,130,246,0.15)",
    ))
    fig_radar.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
        polar=dict(
            bgcolor="rgba(17,24,39,0.5)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(75,85,99,0.3)",
                color="#9CA3AF",
            ),
            angularaxis=dict(
                gridcolor="rgba(75,85,99,0.3)",
                color="#E5E7EB",
            ),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        title=dict(text=t("model_performance_profile"), font=dict(size=14)),
    )
    st.plotly_chart(fig_radar, width="stretch")

    st.markdown("---")

    # Delta analysis table
    st.markdown(f"### {t('delta_analysis')}")

    delta_rows = []
    for metric in METRICS:
        val_a = model_a[metric]
        val_b = model_b[metric]
        diff = val_a - val_b
        pct = diff / max(val_b, 1e-8) * 100
        winner = model_a_name if diff > 0 else (model_b_name if diff < 0 else t("tie"))

        delta_rows.append({
            t("metric_label"): METRIC_LABELS[metric],
            f"{model_a_name}": f"{val_a:.4f}",
            f"{model_b_name}": f"{val_b:.4f}",
            t("difference"): f"{diff:+.4f}",
            t("pct_change"): f"{pct:+.1f}%",
            t("winner"): winner,
        })

    st.dataframe(pd.DataFrame(delta_rows), width="stretch", hide_index=True)

    st.markdown("---")

    # Full leaderboard
    st.markdown(f"### {t('full_model_leaderboard')}")

    leaderboard_rows = []
    for name, metrics in sorted(all_models.items(), key=lambda x: -x[1]["auc_roc"]):
        leaderboard_rows.append({
            t("rank_col"): len(leaderboard_rows) + 1,
            t("model"): name,
            "AUC-ROC": f"{metrics['auc_roc']:.4f}",
            "F1": f"{metrics['f1']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            t("source"): metrics["source"],
        })

    st.dataframe(pd.DataFrame(leaderboard_rows), width="stretch", hide_index=True)

    # Navigation
    st.markdown("---")
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button(t("detailed_performance"), key="comp_to_perf"):
            navigate_to("Performance")
            st.rerun()
    with nav_col2:
        if st.button(t("view_explainability"), key="comp_to_explain"):
            navigate_to("Explainability")
            st.rerun()
