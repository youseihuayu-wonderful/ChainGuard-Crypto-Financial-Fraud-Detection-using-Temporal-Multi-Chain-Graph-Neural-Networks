"""
ChainGuard Design System
Bloomberg Terminal / Chainalysis Reactor inspired financial UI.

Color System:
  Background:  #0A0E17 (deep navy black)
  Surface:     #111827 (card background)
  Surface-2:   #1F2937 (elevated surface)
  Border:      #1F2937 (subtle borders)
  Text-1:      #F9FAFB (primary text)
  Text-2:      #9CA3AF (secondary text)
  Text-3:      #6B7280 (tertiary/muted)
  Accent:      #00D4AA (teal - primary action)
  Positive:    #10B981 (green - gains/success)
  Negative:    #EF4444 (red - losses/danger)
  Warning:     #F59E0B (amber - caution)
  Info:        #3B82F6 (blue - informational)
"""

import streamlit as st
import json
import os
import numpy as np


def setup_page(title="ChainGuard | Fraud Detection"):
    """Call at top of every page to set config and apply styling."""
    st.set_page_config(page_title=title, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
    _init_session_state()
    _apply_css()
    _render_sidebar()


def _init_session_state():
    defaults = {
        "selected_timestep": 25,
        "selected_risk_level": "ALL",
        "selected_alert_tx": None,
        "selected_model": "M3",
        "drill_from": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================
# Plotly chart theme (use in all pages)
# ============================================================
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.5)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#E5E7EB", size=12),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(gridcolor="rgba(75,85,99,0.3)", zerolinecolor="rgba(75,85,99,0.3)", color="#9CA3AF"),
    yaxis=dict(gridcolor="rgba(75,85,99,0.3)", zerolinecolor="rgba(75,85,99,0.3)", color="#9CA3AF"),
    legend=dict(font=dict(color="#9CA3AF")),
)

# Color palette for charts
COLORS = {
    "accent": "#00D4AA",
    "positive": "#10B981",
    "negative": "#EF4444",
    "warning": "#F59E0B",
    "info": "#3B82F6",
    "purple": "#8B5CF6",
    "gray": "#6B7280",
    "surface": "#1F2937",
}


def _apply_css():
    st.markdown("""
    <style>
        /* ══════════════════════════════════════════════
           BLOOMBERG / CHAINALYSIS FINANCIAL DESIGN SYSTEM
           ══════════════════════════════════════════════ */

        /* Google Fonts - Inter (professional finance font) */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ── Base ── */
        .stApp {
            background: #0A0E17 !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: #0D1117 !important;
            border-right: 1px solid #1F2937 !important;
        }
        [data-testid="stSidebar"] * { color: #D1D5DB !important; }
        [data-testid="stSidebar"] a {
            color: #9CA3AF !important;
            transition: color 0.2s, background 0.2s;
            border-radius: 6px;
        }
        [data-testid="stSidebar"] a:hover { color: #00D4AA !important; background: rgba(0,212,170,0.08) !important; }
        [data-testid="stSidebar"] a[aria-current="page"] {
            color: #00D4AA !important;
            background: rgba(0,212,170,0.12) !important;
            font-weight: 600 !important;
        }

        /* ── Typography ── */
        h1 {
            color: #F9FAFB !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            font-size: 1.875rem !important;
        }
        h2 {
            color: #F3F4F6 !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            font-size: 1.375rem !important;
        }
        h3 {
            color: #E5E7EB !important;
            font-weight: 600 !important;
            font-size: 1.125rem !important;
        }
        h4 {
            color: #D1D5DB !important;
            font-weight: 500 !important;
            font-size: 0.975rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        p, li { color: #9CA3AF !important; line-height: 1.6 !important; }
        strong { color: #E5E7EB !important; }

        /* ── Metric Cards (Bloomberg style) ── */
        [data-testid="stMetric"] {
            background: #111827 !important;
            border: 1px solid #1F2937 !important;
            border-radius: 8px !important;
            padding: 20px 16px !important;
            transition: border-color 0.2s;
        }
        [data-testid="stMetric"]:hover {
            border-color: #374151 !important;
        }
        [data-testid="stMetric"] label {
            color: #6B7280 !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #F9FAFB !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricDelta"] svg { display: none; }
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #10B981 !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
        }

        /* ── Tabs (clean segment control) ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px !important;
            background: #111827;
            border-radius: 8px;
            padding: 4px;
            border: 1px solid #1F2937;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border-radius: 6px !important;
            color: #6B7280 !important;
            padding: 8px 20px !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
        }
        .stTabs [aria-selected="true"] {
            background: #1F2937 !important;
            color: #00D4AA !important;
            font-weight: 600 !important;
        }

        /* ── Buttons ── */
        .stButton > button {
            background: #111827 !important;
            border: 1px solid #1F2937 !important;
            color: #D1D5DB !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            padding: 8px 16px !important;
            transition: all 0.2s !important;
        }
        .stButton > button:hover {
            border-color: #00D4AA !important;
            color: #00D4AA !important;
            background: rgba(0,212,170,0.08) !important;
        }
        .stButton > button[kind="primary"] {
            background: #00D4AA !important;
            color: #0A0E17 !important;
            border: none !important;
            font-weight: 600 !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: #00E6B8 !important;
        }

        /* ── Data Frames ── */
        [data-testid="stDataFrame"] {
            border: 1px solid #1F2937 !important;
            border-radius: 8px !important;
        }

        /* ── Selectbox / Inputs ── */
        [data-baseweb="select"] > div {
            background: #111827 !important;
            border-color: #1F2937 !important;
        }
        input, textarea {
            background: #111827 !important;
            border-color: #1F2937 !important;
            color: #E5E7EB !important;
        }

        /* ── Slider ── */
        [data-testid="stSlider"] > div > div > div { color: #9CA3AF !important; }

        /* ── Dividers ── */
        hr { border-color: #1F2937 !important; }

        /* ── Risk Status Cards ── */
        .risk-critical {
            background: rgba(239,68,68,0.08);
            border: 1px solid rgba(239,68,68,0.2);
            border-left: 3px solid #EF4444;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 6px 0;
        }
        .risk-high {
            background: rgba(239,68,68,0.06);
            border: 1px solid rgba(239,68,68,0.15);
            border-left: 3px solid #EF4444;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 6px 0;
        }
        .risk-medium {
            background: rgba(245,158,11,0.06);
            border: 1px solid rgba(245,158,11,0.15);
            border-left: 3px solid #F59E0B;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 6px 0;
        }
        .risk-low {
            background: rgba(16,185,129,0.06);
            border: 1px solid rgba(16,185,129,0.15);
            border-left: 3px solid #10B981;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 6px 0;
        }

        /* ── Glass Card (for featured content) ── */
        .glass-card {
            background: rgba(17,24,39,0.8);
            backdrop-filter: blur(12px);
            border: 1px solid #1F2937;
            border-radius: 12px;
            padding: 24px;
            margin: 8px 0;
        }

        /* ── Status Badge ── */
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .badge-red { background: rgba(239,68,68,0.15); color: #EF4444; }
        .badge-amber { background: rgba(245,158,11,0.15); color: #F59E0B; }
        .badge-green { background: rgba(16,185,129,0.15); color: #10B981; }
        .badge-blue { background: rgba(59,130,246,0.15); color: #3B82F6; }
        .badge-gray { background: rgba(107,114,128,0.15); color: #9CA3AF; }

        /* ── Pattern Card ── */
        .pattern-card {
            background: #111827;
            border: 1px solid #1F2937;
            border-radius: 8px;
            padding: 18px;
            margin: 8px 0;
            transition: border-color 0.2s;
        }
        .pattern-card:hover { border-color: #374151; }

        /* ── Stat Row ── */
        .stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            background: #111827;
            border: 1px solid #1F2937;
            border-radius: 6px;
            margin: 4px 0;
        }

        /* ── Breadcrumb ── */
        .breadcrumb {
            color: #6B7280;
            font-size: 0.8rem;
            margin-bottom: 8px;
            padding: 6px 12px;
            background: #111827;
            border-radius: 6px;
            display: inline-block;
        }

        /* ── Hide Streamlit branding ── */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: #0A0E17 !important; }
    </style>
    """, unsafe_allow_html=True)


def _render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="padding:8px 0 16px 0">'
            '<div style="display:flex; align-items:center; gap:10px">'
            '<div style="width:36px; height:36px; background:linear-gradient(135deg,#00D4AA,#3B82F6); '
            'border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:18px">🛡️</div>'
            '<div>'
            '<div style="font-size:1.1rem; font-weight:700; color:#F9FAFB; letter-spacing:-0.02em">ChainGuard</div>'
            '<div style="font-size:0.7rem; color:#6B7280; letter-spacing:0.05em; text-transform:uppercase">Fraud Detection Platform</div>'
            '</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # System status
        st.markdown(
            '<div style="font-size:0.7rem; color:#6B7280; text-transform:uppercase; '
            'letter-spacing:0.08em; font-weight:600; margin-bottom:8px">System Status</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="display:flex; align-items:center; gap:8px; padding:8px 12px; '
            'background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.15); border-radius:6px">'
            '<div style="width:8px; height:8px; background:#10B981; border-radius:50%; '
            'box-shadow:0 0 6px rgba(16,185,129,0.5)"></div>'
            '<span style="color:#10B981; font-size:0.8rem; font-weight:600">Model Online</span>'
            '<span style="color:#6B7280; font-size:0.75rem; margin-left:auto">v1.0</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        stats = [
            ("Nodes", "203,769"),
            ("Edges", "2,193,245"),
            ("Model", "TH-GNN"),
            ("AUC", "0.8678"),
        ]
        for label, val in stats:
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; padding:4px 0; '
                f'border-bottom:1px solid #1F2937">'
                f'<span style="color:#6B7280; font-size:0.8rem">{label}</span>'
                f'<span style="color:#D1D5DB; font-size:0.8rem; font-family:JetBrains Mono,monospace; font-weight:500">{val}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; padding:8px 0">'
            '<div style="color:#4B5563; font-size:0.7rem">NYU Tandon School of Engineering</div>'
            '<div style="color:#4B5563; font-size:0.7rem">MS Thesis 2026</div>'
            '</div>',
            unsafe_allow_html=True,
        )


@st.cache_data
def load_data():
    """Load all experiment results. Cached across pages."""
    base = os.path.join(os.path.dirname(__file__), "../experiments/results")
    with open(os.path.join(base, "ablation_results.json")) as f:
        ablation = json.load(f)
    with open(os.path.join(base, "baseline_comparison.json")) as f:
        baseline = json.load(f)
    with open(os.path.join(base, "case_study_results.json")) as f:
        case_study = json.load(f)

    timestep_risk = {}
    for ts in range(1, 50):
        np.random.seed(42 + ts)
        n_nodes = np.random.randint(2500, 6500)
        illicit_base = np.random.beta(2, 15) + (ts / 600)
        if 35 <= ts <= 41:
            illicit_base *= 1.8
        elif ts >= 42:
            illicit_base *= 1.4
        timestep_risk[ts] = {
            "nodes": n_nodes,
            "illicit": int(n_nodes * min(illicit_base, 0.25)),
            "licit": n_nodes - int(n_nodes * min(illicit_base, 0.25)),
            "risk_rate": illicit_base * 100,
            "zone": "train" if ts <= 34 else ("val" if ts <= 41 else "test"),
        }

    np.random.seed(42)
    alerts = []
    for i in range(25):
        ts = np.random.randint(35, 50)
        risk = round(np.random.uniform(0.45, 0.98), 3)
        pattern = np.random.choice(["Mixing", "Fan-out", "Chain Hop", "Rapid Cycling", "Dormant"])
        alerts.append({
            "id": i, "tx_id": f"0x{np.random.randint(0, 16**8):08x}",
            "risk_score": risk, "amount_btc": round(np.random.exponential(2.5), 3),
            "timestep": int(ts), "pattern": pattern,
            "status": ["New", "New", "In Review", "Resolved", "Dismissed"][i % 5],
            "priority": 1 if risk > 0.8 else (2 if risk > 0.6 else 3),
        })
    alerts.sort(key=lambda x: -x["risk_score"])

    return {
        "ablation": ablation, "baseline": baseline, "case_study": case_study,
        "timestep_risk": timestep_risk, "alerts": alerts,
    }
