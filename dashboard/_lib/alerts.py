"""
Alert Notification System — Real M3 Predictions
WHO: Operations, Compliance
WHAT: Alert dashboard for high-risk nodes from real M3 model predictions.

Uses REAL data from m3_predictions.json — no mock data.
"""

import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime

from _lib.i18n import t
from _lib.model_serving import load_predictions


def _get_high_risk_nodes(threshold=0.7):
    """Get nodes with risk score above threshold from real M3 predictions."""
    predictions = load_predictions()
    if not predictions:
        return []

    high_risk = []
    for p in predictions.get("test_predictions", []):
        if p["risk_score"] >= threshold:
            high_risk.append(p)

    return high_risk


def _generate_email_html(alerts, threshold):
    """Generate styled HTML email preview for alert notifications."""
    n_alerts = len(alerts)
    n_illicit = sum(1 for a in alerts if a["true_label"] == 1)
    top_5 = alerts[:5]

    rows_html = ""
    for a in top_5:
        label_color = "#EF4444" if a["true_label"] == 1 else "#10B981"
        label_text = "ILLICIT" if a["true_label"] == 1 else "LICIT"
        rows_html += (
            f'<tr style="border-bottom:1px solid #E5E7EB">'
            f'<td style="padding:8px; font-family:monospace">{a["node_id"]}</td>'
            f'<td style="padding:8px; font-weight:bold; color:#EF4444">{a["risk_score"]:.2%}</td>'
            f'<td style="padding:8px; color:{label_color}">{label_text}</td>'
            f'<td style="padding:8px">t={a["timestep"]}</td>'
            f'</tr>'
        )

    html = f"""
    <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:24px; max-width:600px; font-family:Arial,sans-serif">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px">
            <div style="width:40px; height:40px; background:linear-gradient(135deg,#00D4AA,#3B82F6);
                        border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:20px">
                &#128737;
            </div>
            <div>
                <div style="font-size:18px; font-weight:700; color:#111827">ChainGuard Alert</div>
                <div style="font-size:12px; color:#6B7280">Fraud Detection Platform</div>
            </div>
        </div>

        <div style="background:#FEF2F2; border:1px solid #FECACA; border-radius:6px; padding:16px; margin-bottom:16px">
            <div style="color:#991B1B; font-weight:600; font-size:16px">
                &#9888; {n_alerts} High-Risk Nodes Detected
            </div>
            <div style="color:#7F1D1D; font-size:14px; margin-top:4px">
                Threshold: {threshold:.0%} | Known illicit: {n_illicit} | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
            </div>
        </div>

        <div style="font-size:14px; color:#374151; margin-bottom:12px">
            <strong>Top 5 highest-risk nodes:</strong>
        </div>

        <table style="width:100%; border-collapse:collapse; font-size:13px">
            <thead>
                <tr style="background:#F3F4F6; border-bottom:2px solid #D1D5DB">
                    <th style="padding:8px; text-align:left; color:#374151">Node ID</th>
                    <th style="padding:8px; text-align:left; color:#374151">Risk Score</th>
                    <th style="padding:8px; text-align:left; color:#374151">True Label</th>
                    <th style="padding:8px; text-align:left; color:#374151">Timestep</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <div style="margin-top:16px; padding:12px; background:#F0FDF4; border:1px solid #BBF7D0; border-radius:6px">
            <div style="color:#166534; font-size:13px">
                <strong>Action Required:</strong> Review flagged nodes in ChainGuard dashboard.
                Navigate to Alert Center for full details.
            </div>
        </div>

        <div style="margin-top:24px; padding-top:16px; border-top:1px solid #E5E7EB; text-align:center; color:#9CA3AF; font-size:11px">
            ChainGuard Fraud Detection Platform | NYU Tandon | This is a preview, not a real email.
        </div>
    </div>
    """
    return html


def _generate_slack_payload(alerts, threshold):
    """Generate Slack webhook JSON payload for alert notifications."""
    n_alerts = len(alerts)
    n_illicit = sum(1 for a in alerts if a["true_label"] == 1)
    top_3 = alerts[:3]

    fields = []
    for a in top_3:
        label = "ILLICIT" if a["true_label"] == 1 else "LICIT"
        fields.append({
            "type": "mrkdwn",
            "text": f"*Node {a['node_id']}*\nRisk: {a['risk_score']:.2%} | {label} | t={a['timestep']}"
        })

    payload = {
        "channel": "#fraud-alerts",
        "username": "ChainGuard Bot",
        "icon_emoji": ":shield:",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Alert: {n_alerts} High-Risk Nodes Detected",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Threshold:* {threshold:.0%} | *Known Illicit:* {n_illicit} | "
                        f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
                    )
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Top 3 Highest-Risk Nodes:*"
                },
                "fields": fields
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Open ChainGuard Dashboard"
                        },
                        "style": "primary",
                        "url": "https://chainguard.example.com/alerts"
                    }
                ]
            }
        ]
    }
    return payload


def render(DATA, navigate_to):
    """Render the Alert Center page."""
    st.markdown(f"# :bell: {t('alert_center_title')}")
    st.markdown(t("alert_center_subtitle"))
    st.caption(t("alert_center_caption"))

    st.markdown("---")

    # Check if predictions are available
    predictions = load_predictions()
    if not predictions:
        st.error(t("alert_predictions_missing"))
        return

    # Settings panel
    st.markdown(f"### {t('alert_settings')}")
    settings_col1, settings_col2, settings_col3 = st.columns(3)

    with settings_col1:
        threshold = st.slider(
            t("risk_score_threshold"),
            min_value=0.5,
            max_value=0.99,
            value=0.7,
            step=0.01,
            key="alert_threshold",
            help=t("threshold_help"),
        )

    with settings_col2:
        notify_email = st.checkbox(t("email_notifications"), value=True, key="alert_email")
        notify_slack = st.checkbox(t("slack_notifications"), value=True, key="alert_slack")

    with settings_col3:
        severity_filter = st.selectbox(
            t("severity_filter"),
            [t("all_severity"), t("critical_severity"), t("high_severity"), t("medium_severity")],
            key="alert_severity",
        )

    st.markdown("---")

    # Get high-risk nodes
    high_risk = _get_high_risk_nodes(threshold)

    # Apply severity filter
    if severity_filter == t("critical_severity"):
        filtered = [n for n in high_risk if n["risk_score"] > 0.95]
    elif severity_filter == t("high_severity"):
        filtered = [n for n in high_risk if n["risk_score"] > 0.85]
    elif severity_filter == t("medium_severity"):
        filtered = [n for n in high_risk if n["risk_score"] > 0.7]
    else:
        filtered = high_risk

    # Summary metrics
    st.markdown(f"### {t('alert_dashboard')}")
    m1, m2, m3, m4 = st.columns(4)

    n_total = len(filtered)
    n_illicit = sum(1 for n in filtered if n["true_label"] == 1)
    n_licit = n_total - n_illicit
    n_critical = sum(1 for n in filtered if n["risk_score"] > 0.95)

    m1.metric(t("total_alerts"), f"{n_total:,}")
    m2.metric(t("confirmed_illicit"), f"{n_illicit:,}", help=t("confirmed_illicit_help"))
    m3.metric(t("false_positives"), f"{n_licit:,}", help=t("false_positives_help"))
    m4.metric(t("critical_95"), f"{n_critical:,}")

    st.markdown("---")

    # Alert table
    st.markdown(f"### {t('alert_queue_title')}")

    if not filtered:
        st.info(t("no_alerts_msg").format(threshold=f"{threshold:.0%}"))
        return

    # Build DataFrame
    rows = []
    for i, node in enumerate(filtered[:500]):  # Limit to 500 for performance
        label_text = "ILLICIT" if node["true_label"] == 1 else "LICIT"
        if node["risk_score"] > 0.95:
            severity = "CRITICAL"
        elif node["risk_score"] > 0.85:
            severity = "HIGH"
        elif node["risk_score"] > 0.7:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        rows.append({
            t("node_id"): node["node_id"],
            t("risk_score_label"): f"{node['risk_score']:.4f}",
            t("risk_pct"): f"{node['risk_score']:.2%}",
            t("severity_label"): severity,
            t("true_label_label"): label_text,
            t("timestep"): node["timestep"],
            t("correct_label"): "Yes" if (
                (node["risk_score"] > 0.5 and node["true_label"] == 1) or
                (node["risk_score"] <= 0.5 and node["true_label"] == 0)
            ) else "No",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=400)

    st.markdown(f"*{t('showing_alerts').format(shown=len(rows), total=n_total, threshold=f'{threshold:.0%}')}*")

    # Timestep distribution
    st.markdown(f"### {t('alerts_by_timestep')}")
    ts_counts = {}
    for node in filtered:
        ts = node["timestep"]
        if ts not in ts_counts:
            ts_counts[ts] = {"total": 0, "illicit": 0}
        ts_counts[ts]["total"] += 1
        if node["true_label"] == 1:
            ts_counts[ts]["illicit"] += 1

    if ts_counts:
        import plotly.graph_objects as go
        from shared import CHART_LAYOUT

        timesteps = sorted(ts_counts.keys())
        totals = [ts_counts[ts]["total"] for ts in timesteps]
        illicits = [ts_counts[ts]["illicit"] for ts in timesteps]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"t={ts}" for ts in timesteps],
            y=totals,
            name=t("total_alerts_legend"),
            marker_color="#3B82F6",
        ))
        fig.add_trace(go.Bar(
            x=[f"t={ts}" for ts in timesteps],
            y=illicits,
            name=t("confirmed_illicit_legend"),
            marker_color="#EF4444",
        ))
        fig.update_layout(
            **CHART_LAYOUT,
            barmode="overlay",
            height=300,
            title=dict(text=t("alert_dist_by_ts"), font=dict(size=14)),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Export & Notifications
    st.markdown(f"### {t('export_notifications_title')}")

    tab_csv, tab_email, tab_slack = st.tabs([t("csv_export"), t("email_preview_tab"), t("slack_preview_tab")])

    with tab_csv:
        st.markdown(t("csv_export_desc"))

        # Build export DataFrame
        export_rows = []
        for node in filtered:
            export_rows.append({
                "node_id": node["node_id"],
                "risk_score": node["risk_score"],
                "true_label": node["true_label"],
                "timestep": node["timestep"],
                "severity": "CRITICAL" if node["risk_score"] > 0.95 else (
                    "HIGH" if node["risk_score"] > 0.85 else "MEDIUM"
                ),
                "alert_threshold": threshold,
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        export_df = pd.DataFrame(export_rows)

        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label=t("download_n_alerts").format(n=len(export_rows)),
            data=csv_data,
            file_name=f"chainguard_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
            key="download_alerts_csv",
        )

        st.markdown(f"#### {t('preview')}")
        st.dataframe(export_df.head(10), use_container_width=True, hide_index=True)

    with tab_email:
        st.markdown(t("email_preview_desc"))
        if notify_email:
            st.markdown(
                '<div class="stat-row">'
                f'<span style="color:#9CA3AF">{t("email_subject")}</span>'
                f'<span style="color:#E5E7EB">ChainGuard Alert: {t("high_risk_detected").format(n=n_total)}</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="stat-row">'
                f'<span style="color:#9CA3AF">{t("email_from")}</span>'
                '<span style="color:#E5E7EB">alerts@chainguard.nyu.edu</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="stat-row">'
                f'<span style="color:#9CA3AF">{t("email_to")}</span>'
                '<span style="color:#E5E7EB">compliance-team@example.com</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.expander("View Email HTML", expanded=True):
                email_html = _generate_email_html(filtered, threshold)
                st.markdown(email_html, unsafe_allow_html=True)
        else:
            st.info(t("email_disabled"))

    with tab_slack:
        st.markdown(t("slack_preview_desc"))
        if notify_slack:
            payload = _generate_slack_payload(filtered, threshold)
            st.json(payload)
        else:
            st.info(t("slack_disabled"))

    # Navigation
    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button(t("investigate_scanner"), key="alert_to_scanner"):
            navigate_to("Scanner")
            st.rerun()
    with nav_col2:
        if st.button(t("view_network_btn"), key="alert_to_network"):
            navigate_to("Network")
            st.rerun()
    with nav_col3:
        if st.button(t("search_node_btn"), key="alert_to_search"):
            navigate_to("Search")
            st.rerun()
