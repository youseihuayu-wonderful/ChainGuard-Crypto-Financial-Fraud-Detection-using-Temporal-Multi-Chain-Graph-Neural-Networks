"""
Activity Log — Team usage history and analyst activity tracking
WHO: Team Leads, Compliance Officers
WHAT: "Who investigated what?" — User activity timeline, scan history, case notes

NOTE: User profiles are demo data for showcase purposes. Activity entries
demonstrate the kind of usage tracking a production system would have.
All referenced model metrics and node IDs are from real experiment data.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

from _lib.i18n import t


# ═══════════════════════════════════════════
# Demo analyst profiles (clearly labeled)
# ═══════════════════════════════════════════
ANALYSTS = [
    {
        "id": "analyst_01",
        "name": "Sarah Chen",
        "role": "Senior AML Investigator",
        "avatar": "\U0001f469\u200d\U0001f4bb",
        "department": "Financial Crime Unit",
    },
    {
        "id": "analyst_02",
        "name": "Marcus Williams",
        "role": "Compliance Officer",
        "avatar": "\U0001f468\u200d\U0001f4bc",
        "department": "Regulatory Compliance",
    },
    {
        "id": "analyst_03",
        "name": "Dr. Yuki Tanaka",
        "role": "Data Scientist",
        "avatar": "\U0001f469\u200d\U0001f52c",
        "department": "ML Engineering",
    },
    {
        "id": "analyst_04",
        "name": "Alex Rivera",
        "role": "Junior Analyst",
        "avatar": "\U0001f9d1\u200d\U0001f4bb",
        "department": "Transaction Monitoring",
    },
]


def _generate_activity_log(DATA):
    """Generate realistic activity entries using real model data."""
    predictions = None
    results_dir = os.path.join(os.path.dirname(__file__), "../../experiments/results")
    try:
        with open(os.path.join(results_dir, "m3_predictions.json")) as f:
            predictions = json.load(f)
    except FileNotFoundError:
        pass

    # Get real high-risk nodes from predictions
    real_nodes = []
    if predictions and "test_predictions" in predictions:
        for p in predictions["test_predictions"][:30]:
            real_nodes.append(p)

    cs = DATA.get("case_study", {})
    abl = DATA.get("ablation", {})

    # Base time: use a fixed reference for reproducibility
    base_time = datetime(2026, 3, 28, 9, 0, 0)
    np.random.seed(42)

    activities = []

    # Day 1: Initial model deployment
    activities.append({
        "timestamp": base_time,
        "analyst": ANALYSTS[2],  # Dr. Tanaka
        "action": "Model Deployed",
        "module": "Performance",
        "detail": f"Deployed TH-GNN M3 model (AUC: {abl.get('M3', {}).get('auc_roc', 0.8678):.4f}). "
                  f"Passed validation on {cs.get('total_illicit_test', 408)} test transactions.",
        "severity": "info",
    })

    activities.append({
        "timestamp": base_time + timedelta(hours=1, minutes=15),
        "analyst": ANALYSTS[2],
        "action": "Baseline Comparison",
        "module": "Performance",
        "detail": f"Confirmed M3 outperforms {len(abl)} ablation variants and GCN baseline "
                  f"(+{(abl.get('M3', {}).get('auc_roc', 0.8678) - abl.get('M1', {}).get('auc_roc', 0.7449)):.4f} AUC).",
        "severity": "info",
    })

    # Day 1-2: Initial scans by Sarah
    for i, node in enumerate(real_nodes[:5]):
        hours_offset = 3 + i * 0.5
        risk_level = "HIGH" if node["risk_score"] > 0.7 else ("MEDIUM" if node["risk_score"] > 0.4 else "LOW")
        correct = (node["risk_score"] > 0.5 and node["true_label"] == 1) or \
                  (node["risk_score"] <= 0.5 and node["true_label"] == 0)
        true_label = "ILLICIT" if node["true_label"] == 1 else "LICIT"
        correct_str = "Correct \u2705" if correct else "Misclass \u274c"
        activities.append({
            "timestamp": base_time + timedelta(hours=hours_offset),
            "analyst": ANALYSTS[0],  # Sarah
            "action": f"Scanned Node {node['node_id']}",
            "module": "Scanner",
            "detail": f"Risk: {node['risk_score']:.1%} ({risk_level}) | True: {true_label} | TS: {node['timestep']} | {correct_str}",
            "severity": "critical" if risk_level == "HIGH" else ("warning" if risk_level == "MEDIUM" else "low"),
        })

    # Day 2: Marcus reviews detection evidence
    activities.append({
        "timestamp": base_time + timedelta(days=1, hours=2),
        "analyst": ANALYSTS[1],  # Marcus
        "action": "Reviewed Detection Evidence",
        "module": "Forensics",
        "detail": f"Analyzed Venn overlap: {cs.get('both_detect', 113)} detected by both models, "
                  f"{cs.get('m3_only', 100)} unique to TH-GNN, {cs.get('neither', 192)} undetected.",
        "severity": "info",
    })

    activities.append({
        "timestamp": base_time + timedelta(days=1, hours=3, minutes=30),
        "analyst": ANALYSTS[1],
        "action": "Generated SAR Report",
        "module": "Forensics",
        "detail": "Downloaded PDF report for regulatory filing. "
                  f"Report covers {cs.get('total_illicit_test', 408)} test-set transactions.",
        "severity": "info",
    })

    # Day 2-3: Network exploration by Alex
    for ts in [42, 43, 44, 45]:
        ts_data = DATA.get("timestep_risk", {}).get(ts, {})
        activities.append({
            "timestamp": base_time + timedelta(days=1, hours=5 + (ts - 42) * 1.5),
            "analyst": ANALYSTS[3],  # Alex
            "action": f"Explored Network TS {ts}",
            "module": "Network",
            "detail": f"Nodes: {ts_data.get('nodes', '?'):,} | "
                      f"Illicit: {ts_data.get('illicit', '?')} ({ts_data.get('risk_rate', 0):.2f}%) | "
                      f"Edges: {ts_data.get('edges', '?'):,}",
            "severity": "low",
        })

    # Day 3: Sarah investigates more high-risk nodes
    for i, node in enumerate(real_nodes[5:12]):
        activities.append({
            "timestamp": base_time + timedelta(days=2, hours=1 + i * 0.4),
            "analyst": ANALYSTS[0],
            "action": f"Investigated Node {node['node_id']}",
            "module": "Scanner",
            "detail": f"Risk: {node['risk_score']:.1%} | True: {'ILLICIT' if node['true_label'] == 1 else 'LICIT'} | "
                      f"TS: {node['timestep']}",
            "severity": "critical" if node["risk_score"] > 0.7 else "warning",
        })

    # Day 3: Explainability review
    activities.append({
        "timestamp": base_time + timedelta(days=2, hours=5),
        "analyst": ANALYSTS[2],
        "action": "Reviewed Feature Importance",
        "module": "Explainability",
        "detail": "Analyzed gradient-based feature attribution. "
                  "Top features: local transaction features dominate over aggregated features.",
        "severity": "info",
    })

    # Day 4: Executive review
    activities.append({
        "timestamp": base_time + timedelta(days=3, hours=1),
        "analyst": ANALYSTS[1],
        "action": "Executive Dashboard Review",
        "module": "Executive",
        "detail": f"Reviewed KPIs: AUC {abl.get('M3', {}).get('auc_roc', 0.8678):.4f}, "
                  f"Detected {cs.get('both_detect', 113) + cs.get('m3_only', 100)}/{cs.get('total_illicit_test', 408)}, "
                  f"Precision {abl.get('M3', {}).get('precision', 0.7168):.1%}.",
        "severity": "info",
    })

    # Sort by timestamp descending (most recent first)
    activities.sort(key=lambda x: x["timestamp"], reverse=True)

    return activities


def render(DATA, navigate_to):
    st.markdown(f"# \U0001f4dc {t('activity_title')}")
    st.markdown(t("activity_subtitle"))
    st.caption(t("activity_caption"))
    st.markdown("---")

    activities = _generate_activity_log(DATA)

    # ── Team Overview ──
    st.markdown(f"### \U0001f465 {t('team')}")
    cols = st.columns(len(ANALYSTS))
    for col, analyst in zip(cols, ANALYSTS):
        with col:
            # Count activities per analyst
            count = sum(1 for a in activities if a["analyst"]["id"] == analyst["id"])
            st.markdown(
                f'<div style="background:#111827; border:1px solid #1F2937; border-radius:8px; '
                f'padding:16px; text-align:center">'
                f'<div style="font-size:2rem">{analyst["avatar"]}</div>'
                f'<div style="color:#F9FAFB; font-weight:600; font-size:0.9rem; margin-top:4px">{analyst["name"]}</div>'
                f'<div style="color:#00D4AA; font-size:0.75rem">{analyst["role"]}</div>'
                f'<div style="color:#6B7280; font-size:0.7rem">{analyst["department"]}</div>'
                f'<div style="color:#9CA3AF; font-size:0.8rem; margin-top:8px">'
                f'<span style="font-family:JetBrains Mono,monospace; font-weight:600">{count}</span> {t("actions_suffix")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Activity Stats ──
    st.markdown(f"### \U0001f4ca {t('activity_summary')}")
    module_counts = {}
    severity_counts = {"critical": 0, "warning": 0, "info": 0, "low": 0}
    for a in activities:
        module_counts[a["module"]] = module_counts.get(a["module"], 0) + 1
        severity_counts[a["severity"]] = severity_counts.get(a["severity"], 0) + 1

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(t("total_actions"), len(activities))
    s2.metric(t("high_risk_scans"), severity_counts["critical"])
    s3.metric(t("medium_risk_label"), severity_counts["warning"])
    s4.metric(t("modules_used"), len(module_counts))

    # Module usage chart
    fig_mod = go.Figure(go.Bar(
        x=list(module_counts.values()),
        y=list(module_counts.keys()),
        orientation='h',
        marker_color=["#00D4AA", "#3B82F6", "#F59E0B", "#8B5CF6", "#EF4444", "#10B981"][:len(module_counts)],
        text=list(module_counts.values()),
        textposition="outside",
    ))
    fig_mod.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)",
                          font=dict(color="#E5E7EB"), margin=dict(l=100, r=40, t=10, b=10),
                          xaxis=dict(showgrid=False, color="#9CA3AF"),
                          yaxis=dict(color="#E5E7EB"))
    st.plotly_chart(fig_mod, use_container_width=True)

    st.markdown("---")

    # ── Filters ──
    st.markdown(f"### \U0001f50d {t('filter_activity')}")
    f1, f2, f3 = st.columns(3)
    with f1:
        filter_analyst = st.selectbox(t("analyst_label"), [t("all_filter")] + [a["name"] for a in ANALYSTS], key="act_analyst")
    with f2:
        filter_module = st.selectbox(t("module_label"), [t("all_filter")] + list(module_counts.keys()), key="act_module")
    with f3:
        filter_severity = st.selectbox(t("severity_label_filter"), [t("all_filter"), t("critical_filter"), t("warning_filter"), t("info_filter"), t("low_filter")], key="act_sev")

    # Apply filters
    filtered = activities
    if filter_analyst != t("all_filter"):
        filtered = [a for a in filtered if a["analyst"]["name"] == filter_analyst]
    if filter_module != t("all_filter"):
        filtered = [a for a in filtered if a["module"] == filter_module]
    if filter_severity != t("all_filter"):
        sev_map = {t("critical_filter"): "critical", t("warning_filter"): "warning", t("info_filter"): "info", t("low_filter"): "low"}
        sev_val = sev_map.get(filter_severity, filter_severity.lower())
        filtered = [a for a in filtered if a["severity"] == sev_val]

    st.markdown(t("showing_activities").format(shown=len(filtered), total=len(activities)))
    st.markdown("---")

    # ── Activity Timeline ──
    st.markdown(f"### \U0001f4c5 {t('timeline_title')}")

    severity_colors = {
        "critical": "#EF4444",
        "warning": "#F59E0B",
        "info": "#3B82F6",
        "low": "#10B981",
    }
    severity_css = {
        "critical": "risk-high",
        "warning": "risk-medium",
        "info": "glass-card",
        "low": "risk-low",
    }
    module_icons = {
        "Executive": "\U0001f4ca",
        "Performance": "\U0001f9ea",
        "Scanner": "\U0001f50d",
        "Network": "\U0001f578\ufe0f",
        "Forensics": "\U0001f4cb",
        "Explainability": "\U0001f9e0",
    }

    current_date = None
    for activity in filtered:
        act_date = activity["timestamp"].strftime("%Y-%m-%d")
        if act_date != current_date:
            current_date = act_date
            day_name = activity["timestamp"].strftime("%A, %B %d")
            st.markdown(
                f'<div style="background:#111827; border:1px solid #1F2937; border-radius:6px; '
                f'padding:8px 16px; margin:16px 0 8px 0; display:inline-block">'
                f'<span style="color:#00D4AA; font-weight:600; font-size:0.85rem">{day_name}</span></div>',
                unsafe_allow_html=True,
            )

        color = severity_colors.get(activity["severity"], "#6B7280")
        icon = module_icons.get(activity["module"], "\U0001f4cc")
        time_str = activity["timestamp"].strftime("%H:%M")

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02); border-left:3px solid {color}; '
            f'border-radius:0 8px 8px 0; padding:12px 16px; margin:6px 0">'
            f'<div style="display:flex; justify-content:space-between; align-items:center">'
            f'<div style="display:flex; align-items:center; gap:10px">'
            f'<span style="font-size:1.2rem">{activity["analyst"]["avatar"]}</span>'
            f'<div>'
            f'<span style="color:#F9FAFB; font-weight:600; font-size:0.9rem">{activity["analyst"]["name"]}</span>'
            f'<span style="color:#6B7280; font-size:0.8rem; margin-left:8px">{activity["analyst"]["role"]}</span>'
            f'</div></div>'
            f'<div style="display:flex; align-items:center; gap:8px">'
            f'<span style="color:{color}; font-size:0.75rem; background:{color}20; padding:2px 8px; '
            f'border-radius:12px; font-weight:600">{activity["severity"].upper()}</span>'
            f'<span style="color:#6B7280; font-family:JetBrains Mono,monospace; font-size:0.8rem">{time_str}</span>'
            f'</div></div>'
            f'<div style="margin-top:6px">'
            f'<span style="font-size:0.9rem">{icon}</span> '
            f'<span style="color:#00D4AA; font-weight:500; font-size:0.85rem">{activity["action"]}</span>'
            f'<span style="color:#6B7280; font-size:0.8rem; margin-left:8px">in {activity["module"]}</span>'
            f'</div>'
            f'<p style="color:#9CA3AF; font-size:0.82rem; margin:4px 0 0 0">{activity["detail"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Navigation
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(f"\U0001f4ca {t('back_to_exec')}", key="act_to_exec"):
            navigate_to("Executive"); st.rerun()
    with c2:
        if st.button(f"\U0001f50d {t('go_to_scanner')}", key="act_to_scan"):
            navigate_to("Scanner"); st.rerun()
    with c3:
        if st.button(f"\U0001f4cb {t('go_to_forensics')}", key="act_to_for"):
            navigate_to("Forensics"); st.rerun()
