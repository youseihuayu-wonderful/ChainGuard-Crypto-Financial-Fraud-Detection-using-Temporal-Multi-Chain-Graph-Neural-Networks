"""
Data Upload & Analysis — CSV Upload with Rule-Based Risk Scoring
WHO: Analysts, Operations
WHAT: Upload transaction CSV, get risk scores from rule-based engine.

Risk scoring uses the same rule-based engine as the Transaction Scanner.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

from _lib.i18n import t


def _score_row(row, numeric_cols):
    """
    Apply rule-based risk scoring to a single row.
    Adapts the scanner.py engine for arbitrary CSV data.
    """
    risk = 0.08
    factors = []

    vals = {}
    for col in numeric_cols:
        vals[col] = float(row.get(col, 0))

    # Heuristic: look for columns that suggest transaction features
    # Try to find amount-like columns
    amount_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in
                   ["amount", "value", "btc", "eth", "volume", "total", "sum"])]
    degree_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in
                   ["degree", "in_degree", "out_degree", "count", "input", "output"])]

    # Feature-based scoring
    n_features = len(numeric_cols)
    if n_features == 0:
        return 0.1, [("No numeric features to analyze", 0.0, "LOW")]

    # Statistical outlier detection across all numeric features
    # We flag values that are extreme relative to column name heuristics
    high_value_count = 0
    negative_count = 0

    for col in numeric_cols:
        v = vals[col]
        if abs(v) > 10:
            high_value_count += 1
        if v < -2:
            negative_count += 1

    # Amount-based rules
    for col in amount_cols:
        v = vals[col]
        if v > 100:
            bonus = 0.20
            risk += bonus
            factors.append((f"High {col}: {v:.2f}", bonus, "HIGH"))
        elif v > 10:
            bonus = 0.10
            risk += bonus
            factors.append((f"Elevated {col}: {v:.2f}", bonus, "MEDIUM"))

    # Degree-based rules
    for col in degree_cols:
        v = vals[col]
        if v > 20:
            bonus = 0.15
            risk += bonus
            factors.append((f"High {col}: {v:.0f}", bonus, "HIGH"))
        elif v > 5:
            bonus = 0.07
            risk += bonus
            factors.append((f"Moderate {col}: {v:.0f}", bonus, "MEDIUM"))

    # General outlier detection (for Elliptic-style features f1..f166)
    general_cols = [c for c in numeric_cols if c not in amount_cols and c not in degree_cols]
    if general_cols:
        general_vals = [vals[c] for c in general_cols]
        mean_abs = np.mean([abs(v) for v in general_vals]) if general_vals else 0

        if mean_abs > 5:
            bonus = 0.18
            risk += bonus
            factors.append((f"High mean feature magnitude: {mean_abs:.2f}", bonus, "HIGH"))
        elif mean_abs > 2:
            bonus = 0.08
            risk += bonus
            factors.append((f"Elevated feature magnitude: {mean_abs:.2f}", bonus, "MEDIUM"))

        # Check for extreme outlier features
        extreme_count = sum(1 for v in general_vals if abs(v) > 10)
        if extreme_count > 5:
            bonus = 0.15
            risk += bonus
            factors.append((f"{extreme_count} extreme feature values (|v|>10)", bonus, "HIGH"))
        elif extreme_count > 2:
            bonus = 0.06
            risk += bonus
            factors.append((f"{extreme_count} elevated feature values", bonus, "LOW"))

        # Negative feature pattern (common in illicit transactions)
        neg_count = sum(1 for v in general_vals if v < -3)
        if neg_count > 3:
            bonus = 0.10
            risk += bonus
            factors.append((f"{neg_count} strongly negative features", bonus, "MEDIUM"))

    # Cap risk
    risk = min(0.98, max(0.02, risk))

    if not factors:
        factors.append(("No significant risk indicators", 0.0, "LOW"))

    return risk, factors


def render(DATA, navigate_to):
    """Render the Data Upload & Analysis page."""
    st.markdown(f"# :outbox_tray: {t('upload_title')}")
    st.markdown(t("upload_subtitle"))
    st.caption(t("upload_caption"))

    st.markdown("---")

    # File upload
    st.markdown(f"### {t('upload_tx_data')}")
    uploaded_file = st.file_uploader(
        t("upload_csv_file"),
        type=["csv"],
        key="upload_csv",
        help=t("upload_csv_help"),
    )

    if uploaded_file is None:
        # Show instructions
        st.markdown(f"### {t('instructions')}")
        st.markdown(
            '<div class="glass-card">'
            f'<h4 style="color:#00D4AA; margin-top:0">{t("how_it_works")}</h4>'
            '<ol style="color:#E5E7EB">'
            f'<li>{t("step1")}</li>'
            f'<li>{t("step2")}</li>'
            f'<li>{t("step3")}</li>'
            f'<li>{t("step4")}</li>'
            f'<li>{t("step5")}</li>'
            '</ol>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"### {t('expected_format')}")
        st.markdown(
            '<div class="glass-card">'
            f'<p style="color:#E5E7EB; margin-top:0">{t("csv_should_have")}</p>'
            '<ul style="color:#9CA3AF">'
            f'<li><strong style="color:#E5E7EB">{t("numeric_columns_desc")}</strong> — {t("numeric_columns_detail")}</li>'
            f'<li><strong style="color:#E5E7EB">{t("optional_id_desc")}</strong> — {t("optional_id_detail")}</li>'
            f'<li><strong style="color:#E5E7EB">{t("optional_label_desc")}</strong> — {t("optional_label_detail")}</li>'
            '</ul>'
            f'<p style="color:#6B7280; margin-bottom:0">{t("supports_formats")}</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Sample data generator
        st.markdown(f"### {t('generate_sample')}")
        if st.button(t("download_sample_csv"), key="download_sample"):
            sample_data = {
                "node_id": [1001, 1002, 1003, 1004, 1005],
                "amount": [0.5, 15.2, 0.1, 85.3, 2.1],
                "in_degree": [2, 8, 1, 25, 3],
                "out_degree": [1, 12, 1, 3, 2],
                "f1": [0.3, 2.1, -0.5, 5.8, 0.1],
                "f2": [-0.1, 3.4, 0.2, -4.2, 0.8],
                "f3": [0.5, -1.2, 0.3, 8.1, -0.2],
            }
            sample_df = pd.DataFrame(sample_data)
            csv_buf = io.StringIO()
            sample_df.to_csv(csv_buf, index=False)
            st.download_button(
                t("download"),
                csv_buf.getvalue(),
                "chainguard_sample.csv",
                "text/csv",
                key="sample_download_btn",
            )

        return

    # Process uploaded file
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(t("error_reading_csv").format(error=e))
        return

    if df.empty:
        st.warning(t("empty_csv"))
        return

    st.success(t("loaded_rows").format(rows=f"{len(df):,}", cols=len(df.columns)))

    # Data preview
    st.markdown(f"### {t('data_preview')}")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error(t("no_numeric_cols"))
        return

    st.markdown(t("numeric_cols_identified").format(n=len(numeric_cols)))

    # Column summary
    with st.expander(t("column_details")):
        col_info = []
        for col in numeric_cols:
            col_info.append({
                "Column": col,
                "Mean": f"{df[col].mean():.4f}",
                "Std": f"{df[col].std():.4f}",
                "Min": f"{df[col].min():.4f}",
                "Max": f"{df[col].max():.4f}",
                "NaN Count": df[col].isna().sum(),
            })
        st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Score all rows
    st.markdown(f"### {t('risk_analysis')}")

    if st.button(t("analyze_all_rows"), type="primary", use_container_width=True, key="analyze_upload"):
        progress = st.progress(0, text=t("scoring_transactions"))
        results = []

        for idx, row in df.iterrows():
            risk_score, factors = _score_row(row, numeric_cols)
            level = "HIGH" if risk_score > 0.7 else ("MEDIUM" if risk_score > 0.4 else "LOW")

            result = {
                "row_index": idx,
                "risk_score": risk_score,
                "risk_level": level,
                "n_factors": len([f for f in factors if f[1] > 0]),
                "top_factor": factors[0][0] if factors else "None",
            }

            # Include ID column if present
            for id_col in ["node_id", "transaction_id", "tx_id", "id", "ID"]:
                if id_col in df.columns:
                    result["id"] = row[id_col]
                    break

            # Include label column if present
            for label_col in ["true_label", "label", "class", "is_illicit"]:
                if label_col in df.columns:
                    result["true_label"] = row[label_col]
                    break

            results.append(result)

            if idx % max(1, len(df) // 100) == 0:
                progress.progress(min(1.0, (idx + 1) / len(df)), text=t("scoring_progress").format(current=idx + 1, total=len(df)))

        progress.progress(1.0, text=t("scoring_complete"))

        results_df = pd.DataFrame(results)
        st.session_state["upload_results"] = results_df

    # Display results
    if "upload_results" in st.session_state:
        results_df = st.session_state["upload_results"]

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)

        n_high = (results_df["risk_level"] == "HIGH").sum()
        n_medium = (results_df["risk_level"] == "MEDIUM").sum()
        n_low = (results_df["risk_level"] == "LOW").sum()
        avg_risk = results_df["risk_score"].mean()

        m1.metric(t("average_risk"), f"{avg_risk:.2%}")
        m2.metric(t("high_risk_label"), f"{n_high:,}", help="Risk score > 70%")
        m3.metric(t("medium_risk_label"), f"{n_medium:,}", help="Risk score 40-70%")
        m4.metric(t("low_risk_label"), f"{n_low:,}", help="Risk score < 40%")

        # Risk distribution
        st.markdown(f"### {t('risk_distribution')}")

        import plotly.graph_objects as go
        from shared import CHART_LAYOUT

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=results_df["risk_score"],
            nbinsx=50,
            marker_color="#00D4AA",
            opacity=0.8,
        ))
        fig.add_vline(x=0.7, line_dash="dash", line_color="#EF4444", annotation_text="High Risk (0.7)")
        fig.add_vline(x=0.4, line_dash="dash", line_color="#F59E0B", annotation_text="Medium Risk (0.4)")
        fig.update_layout(
            **CHART_LAYOUT,
            height=300,
            title=dict(text=t("risk_score_distribution"), font=dict(size=14)),
            xaxis_title=t("risk_score_axis"),
            yaxis_title=t("count_axis"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Flagged transactions
        st.markdown(f"### {t('flagged_transactions')}")
        flagged = results_df[results_df["risk_level"].isin(["HIGH", "MEDIUM"])].sort_values(
            "risk_score", ascending=False
        )

        if not flagged.empty:
            display_cols = ["row_index", "risk_score", "risk_level", "n_factors", "top_factor"]
            if "id" in flagged.columns:
                display_cols.insert(0, "id")
            if "true_label" in flagged.columns:
                display_cols.append("true_label")

            display_df = flagged[display_cols].head(100).copy()
            display_df["risk_score"] = display_df["risk_score"].apply(lambda x: f"{x:.2%}")

            st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
            st.markdown(f"*{t('showing_flagged').format(shown=min(100, len(flagged)), total=len(flagged))}*")

            # Accuracy check if labels are available
            if "true_label" in results_df.columns:
                st.markdown(f"### {t('validation_vs_labels')}")
                try:
                    labels = results_df["true_label"].astype(int)
                    preds = (results_df["risk_score"] > 0.5).astype(int)

                    tp = ((preds == 1) & (labels == 1)).sum()
                    fp = ((preds == 1) & (labels == 0)).sum()
                    fn = ((preds == 0) & (labels == 1)).sum()
                    tn = ((preds == 0) & (labels == 0)).sum()

                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

                    v1, v2, v3, v4 = st.columns(4)
                    v1.metric(t("precision"), f"{precision:.2%}")
                    v2.metric(t("recall"), f"{recall:.2%}")
                    v3.metric(t("f1_score"), f"{f1:.2%}")
                    v4.metric(t("accuracy"), f"{(tp + tn) / len(results_df):.2%}")

                    st.caption(t("validation_caption"))
                except Exception:
                    st.caption(t("validation_error"))
        else:
            st.info(t("no_flagged"))

        # Download results
        st.markdown("---")
        st.markdown(f"### {t('download_results')}")

        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)

        st.download_button(
            label=t("download_scored_results").format(n=f"{len(results_df):,}"),
            data=csv_buf.getvalue(),
            file_name="chainguard_risk_scores.csv",
            mime="text/csv",
            type="primary",
            key="download_results",
        )
