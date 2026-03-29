"""
Shared utilities for all ChainGuard dashboard pages.
Each page imports this to get data, CSS, and sidebar.
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
    """Initialize all shared session state keys with defaults."""
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


def _apply_css():
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #16213e 0%, #0f3460 100%); }
        [data-testid="stSidebar"] * { color: #e0e0e0; }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px; padding: 16px;
        }
        [data-testid="stMetric"] label { color: #8892b0 !important; font-size: 0.85rem; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ccd6f6 !important; }
        [data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #64ffda !important; }
        h1, h2, h3 { color: #ccd6f6 !important; }
        p, li, span { color: #8892b0; }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.05); border-radius: 8px; color: #8892b0; padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] { background: rgba(100,255,218,0.1) !important; color: #64ffda !important; }
        .risk-high { background:rgba(255,82,82,0.1); border-left:4px solid #ff5252; padding:12px 16px; border-radius:4px; margin:8px 0; }
        .risk-medium { background:rgba(255,152,0,0.1); border-left:4px solid #ff9800; padding:12px 16px; border-radius:4px; margin:8px 0; }
        .risk-low { background:rgba(100,255,218,0.1); border-left:4px solid #64ffda; padding:12px 16px; border-radius:4px; margin:8px 0; }
        hr { border-color: rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)


def _render_sidebar():
    with st.sidebar:
        st.markdown("# 🛡️ ChainGuard")
        st.markdown("**Cross-Chain Fraud Detection**")
        st.markdown("---")
        st.markdown("#### System Status")
        st.markdown('<div class="risk-low">🟢 Model Online</div>', unsafe_allow_html=True)
        st.markdown("**Nodes:** 203,769 | **Edges:** 2.19M")
        st.markdown("---")
        st.markdown(
            "<small style='color:#4a5568'>NYU Tandon School of Engineering<br>"
            "MS Thesis 2026 | © ChainGuard</small>",
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

    # Pre-compute timestep risk data
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

    # Generate alert queue
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
