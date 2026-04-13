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
    st.markdown("# :outbox_tray: Data Upload & Analysis")
    st.markdown("Upload a CSV file with transaction features and get risk scores from the rule-based engine.")
    st.caption("Risk scoring uses rule-based engine. Upload your own transaction data for analysis.")

    st.markdown("---")

    # File upload
    st.markdown("### Upload Transaction Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="upload_csv",
        help="Upload a CSV with numeric transaction features. Each row is scored independently.",
    )

    if uploaded_file is None:
        # Show instructions
        st.markdown("### Instructions")
        st.markdown(
            '<div class="glass-card">'
            '<h4 style="color:#00D4AA; margin-top:0">How it works</h4>'
            '<ol style="color:#E5E7EB">'
            '<li>Upload a CSV file with transaction features</li>'
            '<li>The system identifies numeric columns for analysis</li>'
            '<li>Each row is scored using the rule-based risk engine</li>'
            '<li>Results show risk scores, levels, and flagged transactions</li>'
            '<li>Download scored results as CSV</li>'
            '</ol>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Expected Format")
        st.markdown(
            '<div class="glass-card">'
            '<p style="color:#E5E7EB; margin-top:0">The CSV should have:</p>'
            '<ul style="color:#9CA3AF">'
            '<li><strong style="color:#E5E7EB">Numeric columns</strong> — transaction features (amount, degree, f1..f166, etc.)</li>'
            '<li><strong style="color:#E5E7EB">Optional ID column</strong> — node_id, transaction_id, etc.</li>'
            '<li><strong style="color:#E5E7EB">Optional label column</strong> — true_label, label, class (for validation)</li>'
            '</ul>'
            '<p style="color:#6B7280; margin-bottom:0">Supports Elliptic dataset format (166 features) and custom formats.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Sample data generator
        st.markdown("### Generate Sample Data")
        if st.button("Download Sample CSV", key="download_sample"):
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
                "Download",
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
        st.error(f"Error reading CSV: {e}")
        return

    if df.empty:
        st.warning("The uploaded CSV is empty.")
        return

    st.success(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    # Data preview
    st.markdown("### Data Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found. The risk engine requires at least some numeric features to analyze.")
        return

    st.markdown(f"**Numeric columns identified:** {len(numeric_cols)}")

    # Column summary
    with st.expander("Column Details"):
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
    st.markdown("### Risk Analysis")

    if st.button("Analyze All Rows", type="primary", use_container_width=True, key="analyze_upload"):
        progress = st.progress(0, text="Scoring transactions...")
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
                progress.progress(min(1.0, (idx + 1) / len(df)), text=f"Scoring... {idx + 1}/{len(df)}")

        progress.progress(1.0, text="Scoring complete!")

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

        m1.metric("Average Risk", f"{avg_risk:.2%}")
        m2.metric("High Risk", f"{n_high:,}", help="Risk score > 70%")
        m3.metric("Medium Risk", f"{n_medium:,}", help="Risk score 40-70%")
        m4.metric("Low Risk", f"{n_low:,}", help="Risk score < 40%")

        # Risk distribution
        st.markdown("### Risk Distribution")

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
            title=dict(text="Risk Score Distribution", font=dict(size=14)),
            xaxis_title="Risk Score",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Flagged transactions
        st.markdown("### Flagged Transactions")
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
            st.markdown(f"*Showing {min(100, len(flagged))} of {len(flagged)} flagged transactions*")

            # Accuracy check if labels are available
            if "true_label" in results_df.columns:
                st.markdown("### Validation (vs. True Labels)")
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
                    v1.metric("Precision", f"{precision:.2%}")
                    v2.metric("Recall", f"{recall:.2%}")
                    v3.metric("F1 Score", f"{f1:.2%}")
                    v4.metric("Accuracy", f"{(tp + tn) / len(results_df):.2%}")

                    st.caption("Validation metrics using threshold=0.5 on rule-based risk scores.")
                except Exception:
                    st.caption("Could not compute validation metrics. Ensure true_label column contains 0/1 values.")
        else:
            st.info("No transactions flagged as HIGH or MEDIUM risk.")

        # Download results
        st.markdown("---")
        st.markdown("### Download Results")

        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)

        st.download_button(
            label=f"Download Scored Results ({len(results_df):,} rows)",
            data=csv_buf.getvalue(),
            file_name="chainguard_risk_scores.csv",
            mime="text/csv",
            type="primary",
            key="download_results",
        )
