"""
Case Management — Investigation Workflow & Ticketing System
WHO: AML Investigators, Compliance Officers, Team Leads
WHAT: Ties alerts, investigations, and findings into a complete case lifecycle.

Workflow: Alert → Create Case → Assign Analyst → Investigate → Record Findings → Close

Cases are stored in st.session_state for this demo/thesis project.
Analyst profiles reuse the demo profiles from activity.py.
Alert data comes from real M3 predictions.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta

from _lib.i18n import t
from _lib.model_serving import load_predictions

DETECTION_TYPES = ["M3 Model", "Rule-based", "Manual", "Etherscan"]
DETECTION_TYPE_COLORS = {
    "M3 Model": "#00D4AA",
    "Rule-based": "#F59E0B",
    "Manual": "#8B5CF6",
    "Etherscan": "#3B82F6",
}

STATUSES = ["Open", "In Progress", "Escalated", "Resolved", "Closed"]
STATUS_COLORS = {
    "Open": "#3B82F6",
    "In Progress": "#F59E0B",
    "Escalated": "#EF4444",
    "Resolved": "#10B981",
    "Closed": "#6B7280",
}
PRIORITIES = ["Critical", "High", "Medium", "Low"]
PRIORITY_COLORS = {
    "Critical": "#EF4444",
    "High": "#F59E0B",
    "Medium": "#3B82F6",
    "Low": "#10B981",
}

ANALYSTS = [
    {"id": "analyst_01", "name": "Sarah Chen", "role": "Senior AML Investigator", "avatar": "\U0001f469\u200d\U0001f4bb"},
    {"id": "analyst_02", "name": "Marcus Williams", "role": "Compliance Officer", "avatar": "\U0001f468\u200d\U0001f4bc"},
    {"id": "analyst_03", "name": "Dr. Yuki Tanaka", "role": "Data Scientist", "avatar": "\U0001f469\u200d\U0001f52c"},
    {"id": "analyst_04", "name": "Alex Rivera", "role": "Junior Analyst", "avatar": "\U0001f9d1\u200d\U0001f4bb"},
]


def _init_cases():
    """Initialize case storage in session state with seed demo cases."""
    if "cases" in st.session_state:
        return

    predictions = load_predictions()
    high_risk = []
    if predictions and "test_predictions" in predictions:
        high_risk = [p for p in predictions["test_predictions"] if p["risk_score"] >= 0.90][:10]

    base_time = datetime(2026, 3, 29, 10, 0, 0)
    seed_cases = []

    if len(high_risk) >= 4:
        seed_cases = [
            {
                "id": "CASE-001",
                "title": f"High-risk cluster at timestep {high_risk[0]['timestep']}",
                "detection_type": "M3 Model",
                "status": "Resolved",
                "priority": "Critical",
                "assignee": ANALYSTS[0],
                "created_at": base_time,
                "updated_at": base_time + timedelta(days=2, hours=5),
                "linked_nodes": [high_risk[0]["node_id"], high_risk[1]["node_id"]],
                "description": f"Multiple high-risk nodes detected in timestep {high_risk[0]['timestep']} "
                               f"with risk scores {high_risk[0]['risk_score']:.2%} and {high_risk[1]['risk_score']:.2%}. "
                               "Potential coordinated illicit activity pattern.",
                "findings": "Confirmed illicit cluster. Both nodes show strong graph-neighborhood risk propagation. "
                            "Feature analysis shows mixing-service patterns (high input count, rapid succession).",
                "timeline": [
                    {"time": base_time, "action": "Case created from Alert Center", "by": "Sarah Chen"},
                    {"time": base_time + timedelta(hours=2), "action": "Assigned to Sarah Chen", "by": "System"},
                    {"time": base_time + timedelta(hours=4), "action": "Status changed to In Progress", "by": "Sarah Chen"},
                    {"time": base_time + timedelta(days=1), "action": "Network analysis completed — identified 3 connected illicit nodes", "by": "Sarah Chen"},
                    {"time": base_time + timedelta(days=1, hours=6), "action": "SAR report generated", "by": "Marcus Williams"},
                    {"time": base_time + timedelta(days=2, hours=5), "action": "Case resolved — SAR filed", "by": "Marcus Williams"},
                ],
            },
            {
                "id": "CASE-002",
                "title": f"Anomalous transaction volume node {high_risk[2]['node_id']}",
                "detection_type": "M3 Model",
                "status": "In Progress",
                "priority": "High",
                "assignee": ANALYSTS[3],
                "created_at": base_time + timedelta(days=1),
                "updated_at": base_time + timedelta(days=3, hours=2),
                "linked_nodes": [high_risk[2]["node_id"]],
                "description": f"Node {high_risk[2]['node_id']} flagged with risk score {high_risk[2]['risk_score']:.2%}. "
                               "Unusually high transaction volume relative to peer group.",
                "findings": "Investigation ongoing. Initial graph analysis shows connections to known illicit addresses.",
                "timeline": [
                    {"time": base_time + timedelta(days=1), "action": "Case created from Alert Center", "by": "Alex Rivera"},
                    {"time": base_time + timedelta(days=1, hours=1), "action": "Assigned to Alex Rivera", "by": "System"},
                    {"time": base_time + timedelta(days=2), "action": "Status changed to In Progress", "by": "Alex Rivera"},
                    {"time": base_time + timedelta(days=3, hours=2), "action": "Added finding: connections to known illicit addresses", "by": "Alex Rivera"},
                ],
            },
            {
                "id": "CASE-003",
                "title": f"Cross-timestep pattern nodes {high_risk[3]['node_id']}",
                "detection_type": "M3 Model",
                "status": "Escalated",
                "priority": "Critical",
                "assignee": ANALYSTS[1],
                "created_at": base_time + timedelta(days=2),
                "updated_at": base_time + timedelta(days=3, hours=8),
                "linked_nodes": [high_risk[3]["node_id"]],
                "description": f"Node {high_risk[3]['node_id']} (risk: {high_risk[3]['risk_score']:.2%}) "
                               "shows temporal pattern crossing multiple timesteps. "
                               "Escalated due to potential cross-chain laundering indicators.",
                "findings": "Escalated to Compliance. Temporal k-NN edges reveal connections to nodes in prior timesteps "
                            "with confirmed illicit labels. Possible layering scheme.",
                "timeline": [
                    {"time": base_time + timedelta(days=2), "action": "Case created", "by": "Dr. Yuki Tanaka"},
                    {"time": base_time + timedelta(days=2, hours=3), "action": "Assigned to Marcus Williams", "by": "Dr. Yuki Tanaka"},
                    {"time": base_time + timedelta(days=3), "action": "Status changed to In Progress", "by": "Marcus Williams"},
                    {"time": base_time + timedelta(days=3, hours=8), "action": "Escalated — cross-chain laundering indicators", "by": "Marcus Williams"},
                ],
            },
            {
                "id": "CASE-004",
                "title": "False positive review — licit exchange node",
                "detection_type": "Rule-based",
                "status": "Closed",
                "priority": "Low",
                "assignee": ANALYSTS[2],
                "created_at": base_time + timedelta(days=1, hours=5),
                "updated_at": base_time + timedelta(days=2, hours=1),
                "linked_nodes": [],
                "description": "Review of false positive alerts from exchange-related addresses. "
                               "High volume triggered risk score but addresses belong to known exchange.",
                "findings": "Confirmed false positive. Added exchange address to allowlist. "
                            "Recommended threshold adjustment for exchange-pattern addresses.",
                "timeline": [
                    {"time": base_time + timedelta(days=1, hours=5), "action": "Case created for FP review", "by": "Dr. Yuki Tanaka"},
                    {"time": base_time + timedelta(days=1, hours=8), "action": "Confirmed false positive — exchange address", "by": "Dr. Yuki Tanaka"},
                    {"time": base_time + timedelta(days=2, hours=1), "action": "Case closed — allowlist updated", "by": "Dr. Yuki Tanaka"},
                ],
            },
        ]

    st.session_state["cases"] = seed_cases
    st.session_state["case_counter"] = len(seed_cases)


def _get_next_case_id():
    st.session_state["case_counter"] = st.session_state.get("case_counter", 0) + 1
    return f"CASE-{st.session_state['case_counter']:03d}"


def render(DATA, navigate_to):
    """Render the Case Management page."""
    _init_cases()
    cases = st.session_state["cases"]

    st.markdown(f"# \U0001f4c1 {t('case_mgmt_title')}")
    st.markdown(t("case_mgmt_subtitle"))
    st.caption(t("case_mgmt_caption"))
    st.markdown("---")

    # ── KPI Metrics ──
    total = len(cases)
    open_count = sum(1 for c in cases if c["status"] == "Open")
    in_progress = sum(1 for c in cases if c["status"] == "In Progress")
    escalated = sum(1 for c in cases if c["status"] == "Escalated")
    resolved = sum(1 for c in cases if c["status"] in ("Resolved", "Closed"))
    critical = sum(1 for c in cases if c["priority"] == "Critical")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(t("case_total"), total)
    k2.metric(t("case_open"), open_count + in_progress)
    k3.metric(t("case_escalated"), escalated)
    k4.metric(t("case_resolved"), resolved)
    k5.metric(t("case_critical"), critical)

    st.markdown("---")

    # ── Tabs: Board / Create / Detail ──
    tab_board, tab_create, tab_detail = st.tabs([
        t("case_board_tab"), t("case_create_tab"), t("case_detail_tab"),
    ])

    # ═══════════════════════════════════════════
    # TAB 1: Case Board (Kanban-style overview)
    # ═══════════════════════════════════════════
    with tab_board:
        st.markdown(f"### {t('case_board_title')}")

        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            filter_status = st.selectbox(
                t("case_filter_status"), [t("all_filter")] + STATUSES, key="case_f_status",
            )
        with f2:
            filter_priority = st.selectbox(
                t("case_filter_priority"), [t("all_filter")] + PRIORITIES, key="case_f_prio",
            )
        with f3:
            filter_assignee = st.selectbox(
                t("case_filter_assignee"),
                [t("all_filter")] + [a["name"] for a in ANALYSTS],
                key="case_f_assign",
            )

        filtered = cases
        if filter_status != t("all_filter"):
            filtered = [c for c in filtered if c["status"] == filter_status]
        if filter_priority != t("all_filter"):
            filtered = [c for c in filtered if c["priority"] == filter_priority]
        if filter_assignee != t("all_filter"):
            filtered = [c for c in filtered if c["assignee"]["name"] == filter_assignee]

        st.markdown(f"*{t('case_showing').format(shown=len(filtered), total=total)}*")

        if not filtered:
            st.info(t("case_none_found"))
        else:
            for case in filtered:
                s_color = STATUS_COLORS.get(case["status"], "#6B7280")
                p_color = PRIORITY_COLORS.get(case["priority"], "#6B7280")
                det_type = case.get("detection_type", "Manual")
                dt_color = DETECTION_TYPE_COLORS.get(det_type, "#8B5CF6")
                age = (case["updated_at"] - case["created_at"]).days
                updated_str = case["updated_at"].strftime("%Y-%m-%d %H:%M")

                node_tags = ""
                for nid in case.get("linked_nodes", [])[:3]:
                    node_tags += (
                        f'<span style="background:rgba(59,130,246,0.15); color:#3B82F6; '
                        f'padding:1px 6px; border-radius:4px; font-size:0.7rem; margin-right:4px; '
                        f'font-family:JetBrains Mono,monospace">#{nid}</span>'
                    )

                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.02); border:1px solid #1F2937; '
                    f'border-left:4px solid {s_color}; border-radius:0 8px 8px 0; '
                    f'padding:16px; margin:8px 0">'
                    # Row 1: ID + title + tags
                    f'<div style="display:flex; justify-content:space-between; align-items:center">'
                    f'<div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap">'
                    f'<span style="color:#00D4AA; font-family:JetBrains Mono,monospace; '
                    f'font-weight:700; font-size:0.85rem">{case["id"]}</span>'
                    f'<span style="color:#F9FAFB; font-weight:600; font-size:0.95rem">{case["title"]}</span>'
                    # Detection type tag
                    f'<span style="background:{dt_color}20; color:{dt_color}; padding:1px 8px; '
                    f'border-radius:4px; font-size:0.65rem; font-weight:600; letter-spacing:0.03em; '
                    f'border:1px solid {dt_color}40">{det_type}</span>'
                    f'</div>'
                    f'<div style="display:flex; align-items:center; gap:8px">'
                    f'<span style="background:{p_color}20; color:{p_color}; padding:2px 8px; '
                    f'border-radius:12px; font-size:0.7rem; font-weight:600">{case["priority"]}</span>'
                    f'<span style="background:{s_color}20; color:{s_color}; padding:2px 8px; '
                    f'border-radius:12px; font-size:0.7rem; font-weight:600">{case["status"]}</span>'
                    f'</div></div>'
                    # Row 2: description
                    f'<p style="color:#9CA3AF; font-size:0.82rem; margin:8px 0 6px 0">{case["description"][:150]}...</p>'
                    # Row 3: assignee + dates + nodes
                    f'<div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px">'
                    f'<div style="display:flex; align-items:center; gap:12px">'
                    f'<span style="font-size:1.1rem">{case["assignee"]["avatar"]}</span>'
                    f'<span style="color:#9CA3AF; font-size:0.8rem">{case["assignee"]["name"]}</span>'
                    f'<span style="color:#6B7280; font-size:0.75rem">|</span>'
                    f'<span style="color:#6B7280; font-size:0.75rem">{case["created_at"].strftime("%Y-%m-%d")}</span>'
                    f'<span style="color:#6B7280; font-size:0.75rem">| {age}d</span>'
                    f'<span style="color:#6B7280; font-size:0.75rem">|</span>'
                    f'<span style="color:#F59E0B; font-size:0.7rem; font-style:italic">'
                    f'{t("case_last_updated")} {updated_str}</span>'
                    f'</div>'
                    f'<div>{node_tags}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        # Status distribution chart
        st.markdown("---")
        st.markdown(f"### {t('case_status_overview')}")

        import plotly.graph_objects as go

        status_counts = {s: sum(1 for c in cases if c["status"] == s) for s in STATUSES}
        colors_list = [STATUS_COLORS[s] for s in STATUSES]
        fig_status = go.Figure(go.Bar(
            x=list(status_counts.keys()),
            y=list(status_counts.values()),
            marker_color=colors_list,
            text=list(status_counts.values()),
            textposition="outside",
            textfont=dict(color="#E5E7EB"),
        ))
        fig_status.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.5)",
            font=dict(color="#E5E7EB"),
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis=dict(color="#9CA3AF"),
            yaxis=dict(color="#9CA3AF", gridcolor="rgba(75,85,99,0.3)"),
        )
        st.plotly_chart(fig_status, width="stretch")

        # Priority distribution
        prio_counts = {p: sum(1 for c in cases if c["priority"] == p) for p in PRIORITIES}
        fig_prio = go.Figure(go.Pie(
            labels=list(prio_counts.keys()),
            values=list(prio_counts.values()),
            marker=dict(colors=[PRIORITY_COLORS[p] for p in PRIORITIES]),
            hole=0.5,
            textinfo="label+value",
            textfont=dict(color="#E5E7EB"),
        ))
        fig_prio.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E5E7EB"),
            margin=dict(l=20, r=20, t=10, b=10),
            title=dict(text=t("case_priority_dist"), font=dict(size=14, color="#E5E7EB")),
            legend=dict(font=dict(color="#9CA3AF")),
        )
        st.plotly_chart(fig_prio, width="stretch")

    # ═══════════════════════════════════════════
    # TAB 2: Create New Case
    # ═══════════════════════════════════════════
    with tab_create:
        st.markdown(f"### {t('case_create_title')}")

        # Handle prefill from Alert Center or Network Explorer
        prefill_node = st.session_state.pop("prefill_case_node", None)
        prefill_score = st.session_state.pop("prefill_case_score", None)
        prefill_ts = st.session_state.pop("prefill_case_ts", None)

        prefill_title = ""
        prefill_desc = ""
        prefill_nodes = ""
        if prefill_node:
            score_str = f"{prefill_score:.2%}" if prefill_score else "N/A"
            ts_str = f" at timestep {prefill_ts}" if prefill_ts else ""
            prefill_title = f"High-risk node {prefill_node}{ts_str}"
            prefill_desc = (f"Alert-generated case for node {prefill_node} "
                           f"with risk score {score_str}{ts_str}. Requires investigation.")
            prefill_nodes = str(prefill_node)

        with st.form("create_case_form"):
            title = st.text_input(t("case_title_label"), value=prefill_title,
                                  placeholder=t("case_title_placeholder"))
            description = st.text_area(t("case_desc_label"), value=prefill_desc,
                                       placeholder=t("case_desc_placeholder"), height=120)

            c1, c2, c3 = st.columns(3)
            with c1:
                priority = st.selectbox(t("case_priority_label"), PRIORITIES, index=1)
            with c2:
                assignee_name = st.selectbox(
                    t("case_assignee_label"),
                    [a["name"] for a in ANALYSTS],
                )
            with c3:
                detection_type = st.selectbox(
                    t("case_detection_type"), DETECTION_TYPES, index=0,
                )

            # Link to alert nodes
            st.markdown(f"**{t('case_link_nodes')}**")
            node_ids_input = st.text_input(
                t("case_node_ids_label"),
                value=prefill_nodes,
                placeholder=t("case_node_ids_placeholder"),
                help=t("case_node_ids_help"),
            )

            submitted = st.form_submit_button(t("case_create_btn"), type="primary")

            if submitted:
                if not title.strip():
                    st.error(t("case_title_required"))
                else:
                    assignee = next(a for a in ANALYSTS if a["name"] == assignee_name)
                    linked = []
                    if node_ids_input.strip():
                        linked = [int(x.strip()) for x in node_ids_input.split(",") if x.strip().isdigit()]

                    now = datetime.now()
                    new_case = {
                        "id": _get_next_case_id(),
                        "title": title.strip(),
                        "detection_type": detection_type,
                        "status": "Open",
                        "priority": priority,
                        "assignee": assignee,
                        "created_at": now,
                        "updated_at": now,
                        "linked_nodes": linked,
                        "description": description.strip() or f"Case created for investigation: {title}",
                        "findings": "",
                        "timeline": [
                            {"time": now, "action": f"Case created — priority: {priority}", "by": assignee["name"]},
                            {"time": now, "action": f"Assigned to {assignee['name']}", "by": "System"},
                        ],
                    }
                    st.session_state["cases"].append(new_case)
                    st.success(t("case_created_msg").format(case_id=new_case["id"]))
                    st.rerun()

        # Quick-create from alerts
        st.markdown("---")
        st.markdown(f"### {t('case_from_alerts')}")
        st.markdown(t("case_from_alerts_desc"))

        predictions = load_predictions()
        if predictions and "test_predictions" in predictions:
            high_risk = [p for p in predictions["test_predictions"] if p["risk_score"] >= 0.90][:10]
            if high_risk:
                rows = []
                for p in high_risk:
                    already_linked = any(
                        p["node_id"] in c.get("linked_nodes", []) for c in cases
                    )
                    rows.append({
                        t("node_id"): p["node_id"],
                        t("risk_score_label"): f"{p['risk_score']:.2%}",
                        t("timestep"): p["timestep"],
                        t("true_label_label"): "ILLICIT" if p["true_label"] == 1 else "LICIT",
                        t("case_linked"): "Yes" if already_linked else "No",
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                st.caption(t("case_from_alerts_caption"))
        else:
            st.info(t("case_no_predictions"))

    # ═══════════════════════════════════════════
    # TAB 3: Case Detail / Update
    # ═══════════════════════════════════════════
    with tab_detail:
        st.markdown(f"### {t('case_detail_title')}")

        if not cases:
            st.info(t("case_none_found"))
        else:
            case_options = [f"{c['id']}: {c['title']}" for c in cases]
            selected_idx = st.selectbox(
                t("case_select_label"), range(len(case_options)),
                format_func=lambda i: case_options[i],
                key="case_detail_select",
            )
            case = cases[selected_idx]

            s_color = STATUS_COLORS.get(case["status"], "#6B7280")
            p_color = PRIORITY_COLORS.get(case["priority"], "#6B7280")
            det_type = case.get("detection_type", "Manual")
            dt_color = DETECTION_TYPE_COLORS.get(det_type, "#8B5CF6")
            updated_str = case["updated_at"].strftime("%Y-%m-%d %H:%M")
            created_str = case["created_at"].strftime("%Y-%m-%d %H:%M")

            # Case header
            st.markdown(
                f'<div class="glass-card">'
                # Row 1: ID + title + detection tag + status badges
                f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px">'
                f'<div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap">'
                f'<span style="color:#00D4AA; font-family:JetBrains Mono,monospace; '
                f'font-weight:700; font-size:1.1rem">{case["id"]}</span>'
                f'<span style="color:#F9FAFB; font-weight:700; font-size:1.2rem">{case["title"]}</span>'
                f'<span style="background:{dt_color}20; color:{dt_color}; padding:2px 10px; '
                f'border-radius:4px; font-size:0.7rem; font-weight:600; '
                f'border:1px solid {dt_color}40">{det_type}</span>'
                f'</div>'
                f'<div style="display:flex; gap:8px">'
                f'<span style="background:{p_color}20; color:{p_color}; padding:4px 12px; '
                f'border-radius:12px; font-size:0.8rem; font-weight:600">{case["priority"]}</span>'
                f'<span style="background:{s_color}20; color:{s_color}; padding:4px 12px; '
                f'border-radius:12px; font-size:0.8rem; font-weight:600">{case["status"]}</span>'
                f'</div></div>'
                # Row 2: assignee
                f'<div style="margin-top:12px; display:flex; align-items:center; gap:12px">'
                f'<span style="font-size:1.3rem">{case["assignee"]["avatar"]}</span>'
                f'<div>'
                f'<span style="color:#F9FAFB; font-weight:600">{case["assignee"]["name"]}</span>'
                f'<span style="color:#6B7280; font-size:0.8rem; margin-left:8px">{case["assignee"]["role"]}</span>'
                f'</div></div>'
                # Row 3: description
                f'<p style="color:#9CA3AF; margin-top:12px">{case["description"]}</p>'
                # Row 4: timestamps
                f'<div style="margin-top:8px; display:flex; gap:24px; font-size:0.75rem">'
                f'<span style="color:#6B7280">{t("case_created_at")} '
                f'<span style="color:#9CA3AF; font-family:JetBrains Mono,monospace">{created_str}</span></span>'
                f'<span style="color:#F59E0B">{t("case_last_updated")} '
                f'<span style="font-family:JetBrains Mono,monospace">{updated_str}</span></span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Linked nodes with one-click navigation
            if case.get("linked_nodes"):
                st.markdown(f"#### {t('case_linked_nodes')}")
                for nid in case["linked_nodes"]:
                    st.markdown(
                        f'<span style="background:rgba(59,130,246,0.15); color:#3B82F6; '
                        f'padding:4px 12px; border-radius:6px; font-family:JetBrains Mono,monospace; '
                        f'margin-right:8px; display:inline-block; margin-bottom:4px">Node #{nid}</span>',
                        unsafe_allow_html=True,
                    )

                # One-click investigation buttons for linked nodes
                st.markdown(f"##### {t('case_investigate_nodes')}")
                inv_cols = st.columns(min(len(case["linked_nodes"]), 4))
                for idx, nid in enumerate(case["linked_nodes"][:4]):
                    with inv_cols[idx]:
                        st.markdown(
                            f'<div style="color:#9CA3AF; font-size:0.75rem; margin-bottom:4px; '
                            f'font-family:JetBrains Mono,monospace">Node #{nid}</div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            f"\U0001f50e {t('case_search_node')}", key=f"case_search_{case['id']}_{nid}",
                        ):
                            st.session_state["search_prefill_node"] = nid
                            navigate_to("Search")
                            st.rerun()
                        if st.button(
                            f"\U0001f578\ufe0f {t('case_view_network')}", key=f"case_net_{case['id']}_{nid}",
                        ):
                            navigate_to("Network")
                            st.rerun()
                        if st.button(
                            f"\U0001f514 {t('case_view_alerts')}", key=f"case_alert_{case['id']}_{nid}",
                        ):
                            navigate_to("Alerts")
                            st.rerun()

            # Findings
            st.markdown(f"#### {t('case_findings')}")
            if case.get("findings"):
                st.markdown(
                    f'<div class="pattern-card"><p style="color:#E5E7EB">{case["findings"]}</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info(t("case_no_findings"))

            # Case timeline
            st.markdown(f"#### {t('case_timeline')}")
            for entry in case.get("timeline", []):
                time_str = entry["time"].strftime("%Y-%m-%d %H:%M")
                st.markdown(
                    f'<div style="border-left:2px solid #1F2937; padding:8px 0 8px 16px; margin-left:8px">'
                    f'<div style="display:flex; align-items:center; gap:8px">'
                    f'<span style="width:8px; height:8px; background:#00D4AA; border-radius:50%; '
                    f'display:inline-block; margin-left:-21px"></span>'
                    f'<span style="color:#6B7280; font-family:JetBrains Mono,monospace; font-size:0.75rem">'
                    f'{time_str}</span>'
                    f'<span style="color:#9CA3AF; font-size:0.75rem">by {entry["by"]}</span>'
                    f'</div>'
                    f'<p style="color:#E5E7EB; font-size:0.85rem; margin:4px 0 0 0">{entry["action"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Actions
            st.markdown("---")
            st.markdown(f"#### {t('case_actions')}")

            a1, a2 = st.columns(2)

            with a1:
                new_status = st.selectbox(
                    t("case_update_status"), STATUSES,
                    index=STATUSES.index(case["status"]),
                    key=f"case_status_{case['id']}",
                )
                if st.button(t("case_update_status_btn"), key=f"btn_status_{case['id']}"):
                    if new_status != case["status"]:
                        now = datetime.now()
                        case["status"] = new_status
                        case["updated_at"] = now
                        case["timeline"].append({
                            "time": now,
                            "action": f"Status changed to {new_status}",
                            "by": case["assignee"]["name"],
                        })
                        st.success(t("case_status_updated").format(status=new_status))
                        st.rerun()

            with a2:
                new_assignee = st.selectbox(
                    t("case_reassign"), [a["name"] for a in ANALYSTS],
                    index=next(i for i, a in enumerate(ANALYSTS) if a["name"] == case["assignee"]["name"]),
                    key=f"case_assign_{case['id']}",
                )
                if st.button(t("case_reassign_btn"), key=f"btn_assign_{case['id']}"):
                    if new_assignee != case["assignee"]["name"]:
                        now = datetime.now()
                        old_name = case["assignee"]["name"]
                        case["assignee"] = next(a for a in ANALYSTS if a["name"] == new_assignee)
                        case["updated_at"] = now
                        case["timeline"].append({
                            "time": now,
                            "action": f"Reassigned from {old_name} to {new_assignee}",
                            "by": old_name,
                        })
                        st.success(t("case_reassigned").format(name=new_assignee))
                        st.rerun()

            # Add finding
            st.markdown(f"#### {t('case_add_finding')}")
            new_finding = st.text_area(
                t("case_finding_label"), placeholder=t("case_finding_placeholder"),
                key=f"finding_{case['id']}", height=80,
            )
            if st.button(t("case_add_finding_btn"), key=f"btn_finding_{case['id']}"):
                if new_finding.strip():
                    now = datetime.now()
                    if case.get("findings"):
                        case["findings"] += f"\n\n[{now.strftime('%Y-%m-%d %H:%M')}] {new_finding.strip()}"
                    else:
                        case["findings"] = f"[{now.strftime('%Y-%m-%d %H:%M')}] {new_finding.strip()}"
                    case["updated_at"] = now
                    case["timeline"].append({
                        "time": now,
                        "action": f"Added finding: {new_finding.strip()[:80]}...",
                        "by": case["assignee"]["name"],
                    })
                    st.success(t("case_finding_added"))
                    st.rerun()

    # ── Navigation ──
    st.markdown("---")
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button(t("case_nav_alerts"), key="case_to_alerts"):
            navigate_to("Alerts"); st.rerun()
    with nav2:
        if st.button(t("case_nav_forensics"), key="case_to_forensics"):
            navigate_to("Forensics"); st.rerun()
    with nav3:
        if st.button(t("case_nav_activity"), key="case_to_activity"):
            navigate_to("Activity"); st.rerun()
