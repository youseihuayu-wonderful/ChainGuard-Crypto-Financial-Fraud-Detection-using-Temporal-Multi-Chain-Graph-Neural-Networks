"""
SQL Analytics — Operational Intelligence Dashboard
WHO: Head of Compliance, CRO, Data Engineers
WHAT: Production-grade analytical queries demonstrating advanced SQL on fraud detection data.

Demonstrates: Window functions, CTEs, multi-table JOINs, correlated subqueries,
              conditional aggregation, time-series spike detection, Pareto analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from _lib.i18n import t
from _lib import analytics_db as adb
from shared import CHART_LAYOUT


def render(DATA, navigate_to):
    st.markdown(f"# \U0001f4ca {t('sql_title')}")
    st.markdown(t("sql_subtitle"))
    st.caption(t("sql_caption"))

    counts = adb.get_table_counts()
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(t("sql_predictions"), f"{counts['predictions']:,}")
    k2.metric(t("sql_timesteps"), counts["timestep_stats"])
    k3.metric(t("sql_models"), counts["model_results"])
    k4.metric(t("sql_features"), counts["feature_importance"])
    k5.metric(t("sql_explanations"), counts["node_explanations"])
    k6.metric(t("sql_feedback_count"), counts["analyst_feedback"])

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        t("sql_tab_monitoring"), t("sql_tab_risk"),
        t("sql_tab_features"), t("sql_tab_compliance"),
    ])

    # ═══════════════════════════════════════════
    # TAB 1: Model Monitoring
    # ═══════════════════════════════════════════
    with tab1:
        st.markdown(f"### {t('sql_tier_title')}")
        st.caption(t("sql_tier_caption"))

        with st.spinner(t("loading_charts")):
            tiers = adb.query_detection_tiers()

        if tiers:
            df_tiers = pd.DataFrame(tiers)
            st.dataframe(df_tiers, width="stretch", hide_index=True)

            fig_tier = go.Figure()
            tier_colors = {"Critical": "#EF4444", "High": "#F59E0B", "Medium": "#3B82F6", "Low": "#10B981"}
            fig_tier.add_trace(go.Bar(
                x=[r["tier"] for r in tiers],
                y=[r["precision_pct"] for r in tiers],
                name="Precision %",
                marker_color=[tier_colors.get(r["tier"], "#6B7280") for r in tiers],
                text=[f"{r['precision_pct']}%" for r in tiers],
                textposition="outside", textfont=dict(color="#E5E7EB"),
            ))
            fig_tier.add_trace(go.Scatter(
                x=[r["tier"] for r in tiers],
                y=[r["cumulative_recall_pct"] for r in tiers],
                name="Cumulative Recall %",
                mode="lines+markers+text",
                line=dict(color="#00D4AA", width=3),
                text=[f"{r['cumulative_recall_pct']}%" for r in tiers],
                textposition="top center", textfont=dict(color="#00D4AA"),
            ))
            fig_tier.update_layout(
                **CHART_LAYOUT, height=350, barmode="group",
                title=dict(text=t("sql_tier_chart_title"), font=dict(size=14)),
                yaxis=dict(title="%", range=[0, 110]),
            )
            st.plotly_chart(fig_tier, width="stretch")

        with st.expander(f"\U0001f4dd {t('sql_show_query')}: CTE + Cumulative Window"):
            st.code(adb.Q2_SQL, language="sql")

        st.markdown("---")

        # Analyst feedback analysis (3-table JOIN)
        st.markdown(f"### {t('sql_feedback_title')}")
        st.caption(t("sql_feedback_caption"))

        feedback = adb.query_feedback_analysis()
        if feedback:
            df_fb = pd.DataFrame(feedback)
            st.dataframe(df_fb, width="stretch", hide_index=True)

            outcomes = {}
            for r in feedback:
                o = r.get("outcome", "Unknown")
                outcomes[o] = outcomes.get(o, 0) + 1

            if outcomes:
                labels = list(outcomes.keys())
                values = list(outcomes.values())
                colors = {"True Positive": "#10B981", "False Positive": "#F59E0B",
                          "Missed Fraud": "#EF4444", "True Negative": "#3B82F6"}
                fig_fb = go.Figure(go.Pie(
                    labels=labels, values=values,
                    marker=dict(colors=[colors.get(l, "#6B7280") for l in labels]),
                    hole=0.5, textinfo="label+value",
                    textfont=dict(color="#E5E7EB"),
                ))
                fig_fb.update_layout(
                    **CHART_LAYOUT, height=300,
                    title=dict(text=t("sql_feedback_chart"), font=dict(size=14)),
                )
                st.plotly_chart(fig_fb, width="stretch")
        else:
            st.info(t("sql_no_feedback"))

        with st.expander(f"\U0001f4dd {t('sql_show_query')}: 3-Table JOIN + CASE WHEN"):
            st.code(adb.Q3_SQL, language="sql")

    # ═══════════════════════════════════════════
    # TAB 2: Risk Analysis
    # ═══════════════════════════════════════════
    with tab2:
        st.markdown(f"### {t('sql_trend_title')}")
        st.caption(t("sql_trend_caption"))

        with st.spinner(t("loading_charts")):
            trends = adb.query_risk_trends()

        if trends:
            df_trend = pd.DataFrame(trends)

            colors_bar = []
            for r in trends:
                if r["anomaly_flag"] == "SPIKE":
                    colors_bar.append("#EF4444")
                elif r["anomaly_flag"] == "DROP":
                    colors_bar.append("#3B82F6")
                else:
                    colors_bar.append("rgba(100,255,218,0.5)")

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(
                x=[r["timestep"] for r in trends],
                y=[r["risk_rate"] for r in trends],
                name="Risk Rate",
                marker_color=colors_bar,
            ))
            fig_trend.add_trace(go.Scatter(
                x=[r["timestep"] for r in trends],
                y=[r["moving_avg_5"] for r in trends],
                name="5-Period Moving Avg",
                mode="lines",
                line=dict(color="#F59E0B", width=2, dash="dash"),
            ))
            fig_trend.update_layout(
                **CHART_LAYOUT, height=350,
                title=dict(text=t("sql_trend_chart"), font=dict(size=14)),
                xaxis=dict(title="Timestep"),
                yaxis=dict(title="Risk Rate (%)"),
            )
            st.plotly_chart(fig_trend, width="stretch")

            spikes = [r for r in trends if r["anomaly_flag"] == "SPIKE"]
            if spikes:
                spike_labels = ", ".join(f"TS {s['timestep']}" for s in spikes)
                st.warning(f"{t('sql_spikes_detected')}: {spike_labels}")

            st.dataframe(df_trend, width="stretch", hide_index=True, height=300)

        with st.expander(f"\U0001f4dd {t('sql_show_query')}: Window Functions + Spike Detection"):
            st.code(adb.Q6_SQL, language="sql")

        st.markdown("---")

        # Risk ranking (window functions)
        st.markdown(f"### {t('sql_ranking_title')}")
        st.caption(t("sql_ranking_caption"))

        with st.spinner(t("loading_charts")):
            rankings = adb.query_risk_ranking()

        if rankings:
            df_rank = pd.DataFrame(rankings)
            st.dataframe(df_rank, width="stretch", hide_index=True, height=350)

        with st.expander(f"\U0001f4dd {t('sql_show_query')}: RANK, NTILE, LAG Window Functions"):
            st.code(adb.Q1_SQL, language="sql")

        st.markdown("---")

        # Blind spots (correlated subquery)
        st.markdown(f"### {t('sql_blind_title')}")
        st.caption(t("sql_blind_caption"))

        with st.spinner(t("loading_charts")):
            blind = adb.query_blind_spots()

        if blind:
            df_blind = pd.DataFrame(blind)
            st.dataframe(df_blind, width="stretch", hide_index=True)

            st.markdown(
                f'<div class="risk-high" style="padding:16px">'
                f'<strong style="color:#EF4444">{t("sql_blind_insight")}</strong><br>'
                f'<span style="color:#E5E7EB">{t("sql_blind_detail").format(n=len(blind))}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.success(t("sql_no_blind"))

        with st.expander(f"\U0001f4dd {t('sql_show_query')}: Correlated Subquery"):
            st.code(adb.Q4_SQL, language="sql")

    # ═══════════════════════════════════════════
    # TAB 3: Feature Intelligence
    # ═══════════════════════════════════════════
    with tab3:
        st.markdown(f"### {t('sql_pareto_title')}")
        st.caption(t("sql_pareto_caption"))

        with st.spinner(t("loading_charts")):
            pareto = adb.query_feature_pareto()

        if pareto:
            df_par = pd.DataFrame(pareto)

            group_colors = {"Top 50%": "#EF4444", "Top 80%": "#F59E0B", "Top 95%": "#3B82F6", "Tail": "#6B7280"}

            fig_par = go.Figure()
            fig_par.add_trace(go.Bar(
                y=[r["feature_name"] for r in reversed(pareto)],
                x=[r["importance"] for r in reversed(pareto)],
                orientation="h",
                marker_color=[group_colors.get(r["pareto_group"], "#6B7280") for r in reversed(pareto)],
                name="Importance",
                text=[f"{r['importance']}" for r in reversed(pareto)],
                textposition="outside", textfont=dict(color="#E5E7EB", size=9),
            ))
            fig_par.add_trace(go.Scatter(
                y=[r["feature_name"] for r in reversed(pareto)],
                x=[r["cumulative_pct"] / 100 * max(r2["importance"] for r2 in pareto) for r in reversed(pareto)],
                mode="lines+markers",
                name="Cumulative %",
                line=dict(color="#00D4AA", width=2),
                marker=dict(size=4),
            ))
            fig_par.update_layout(
                **CHART_LAYOUT, height=700,
                title=dict(text=t("sql_pareto_chart"), font=dict(size=14)),
                xaxis=dict(title="Importance Score"),
                margin=dict(l=120, r=60, t=40, b=40),
            )
            st.plotly_chart(fig_par, width="stretch")

            st.dataframe(df_par, width="stretch", hide_index=True)

        with st.expander(f"\U0001f4dd {t('sql_show_query')}: ROW_NUMBER + Running SUM Window"):
            st.code(adb.Q5_SQL, language="sql")

    # ═══════════════════════════════════════════
    # TAB 4: Compliance Report
    # ═══════════════════════════════════════════
    with tab4:
        st.markdown(f"### {t('sql_compare_title')}")
        st.caption(t("sql_compare_caption"))

        with st.spinner(t("loading_charts")):
            comparison = adb.query_model_comparison()

        if comparison:
            df_cmp = pd.DataFrame(comparison)
            st.dataframe(df_cmp, width="stretch", hide_index=True)

            fig_cmp = go.Figure()
            for r in comparison:
                fig_cmp.add_trace(go.Bar(
                    x=[r["model_type"]],
                    y=[r["best_auc"]],
                    name=f"{r['top_model']} (best)",
                    text=[f"{r['best_auc']}"],
                    textposition="outside", textfont=dict(color="#E5E7EB"),
                    marker_color="#00D4AA" if r["model_type"] == "ablation" else "#3B82F6",
                ))
            fig_cmp.update_layout(
                **CHART_LAYOUT, height=300, showlegend=True,
                title=dict(text=t("sql_compare_chart"), font=dict(size=14)),
                yaxis=dict(title="AUC-ROC", range=[0.6, 1.0]),
            )
            st.plotly_chart(fig_cmp, width="stretch")

        with st.expander(f"\U0001f4dd {t('sql_show_query')}: GROUP BY + HAVING + Scalar Subquery"):
            st.code(adb.Q7_SQL, language="sql")

        st.markdown("---")

        # Schema overview
        st.markdown(f"### {t('sql_schema_title')}")
        st.caption(t("sql_schema_caption"))

        counts = adb.get_table_counts()
        schema_rows = [
            {"Table": "predictions", "Rows": counts["predictions"],
             "Purpose": "All M3 model scored nodes", "Key Columns": "node_id, risk_score, true_label, timestep, predicted(gen)"},
            {"Table": "timestep_stats", "Rows": counts["timestep_stats"],
             "Purpose": "Per-timestep aggregated statistics", "Key Columns": "timestep, n_nodes, n_illicit, risk_rate, zone"},
            {"Table": "model_results", "Rows": counts["model_results"],
             "Purpose": "Experiment results (ablation + baselines)", "Key Columns": "model_id, model_type, auc_roc, f1, precision_, recall_"},
            {"Table": "feature_importance", "Rows": counts["feature_importance"],
             "Purpose": "Gradient-based feature importance", "Key Columns": "model_id, feature_idx, feature_name, importance"},
            {"Table": "node_explanations", "Rows": counts["node_explanations"],
             "Purpose": "Per-node feature contributions (top 50)", "Key Columns": "node_id, feature_idx, gradient, contribution"},
            {"Table": "analyst_feedback", "Rows": counts["analyst_feedback"],
             "Purpose": "Analyst corrections (Confirm/FP)", "Key Columns": "node_id, feedback_type, analyst, created_at"},
        ]
        st.dataframe(pd.DataFrame(schema_rows), width="stretch", hide_index=True)

        st.markdown(f"#### {t('sql_techniques_title')}")
        techniques = [
            {"#": 1, "Technique": "Window Functions", "Functions": "RANK(), NTILE(), LAG(), AVG() OVER", "Query": "Q1 — Risk Ranking"},
            {"#": 2, "Technique": "CTEs (Common Table Expressions)", "Functions": "WITH ... AS", "Query": "Q2 — Detection Tiers"},
            {"#": 3, "Technique": "Multi-table JOINs", "Functions": "JOIN ... ON (3 tables)", "Query": "Q3 — Feedback Analysis"},
            {"#": 4, "Technique": "Correlated Subqueries", "Functions": "WHERE col > (SELECT ...)", "Query": "Q4 — Blind Spots"},
            {"#": 5, "Technique": "Running Aggregation", "Functions": "SUM() OVER (ORDER BY ...)", "Query": "Q5 — Pareto Analysis"},
            {"#": 6, "Technique": "Time-Series Analysis", "Functions": "LAG(), Moving AVG, CASE", "Query": "Q6 — Spike Detection"},
            {"#": 7, "Technique": "Conditional Aggregation", "Functions": "GROUP BY, HAVING, SUM(CASE)", "Query": "Q7 — Model Comparison"},
        ]
        st.dataframe(pd.DataFrame(techniques), width="stretch", hide_index=True)

    # Navigation
    st.markdown("---")
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button(t("sql_nav_exec"), key="sql_to_exec"):
            navigate_to("Executive"); st.rerun()
    with nav2:
        if st.button(t("sql_nav_alerts"), key="sql_to_alerts"):
            navigate_to("Alerts"); st.rerun()
    with nav3:
        if st.button(t("sql_nav_perf"), key="sql_to_perf"):
            navigate_to("Performance"); st.rerun()
