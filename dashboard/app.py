"""
ChainGuard — Cross-Chain Cryptocurrency Fraud Detection Dashboard
Enterprise-grade monitoring platform powered by TH-GNN.

Architecture:
  L1: Executive Command Center  → "What's happening?" (KPIs, trends, alerts)
  L2: Model Analytics            → "Why does it work?" (ablation, baselines)
  L3: Investigation Hub          → "Who is suspicious?" (scanner, network)
  L4: Forensics Lab              → "What's the evidence?" (patterns, deep-dive)

All pages share session_state for drill-down navigation.

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import json
import os
import numpy as np

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="ChainGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Shared Session State (cross-page communication)
# ============================================================
DEFAULTS = {
    "selected_page": "L1: Command Center",
    "selected_timestep": 25,
    "selected_risk_level": "ALL",
    "selected_alert_tx": None,
    "selected_model": "M3",
    "drill_from": None,  # tracks which page initiated a drill-down
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def navigate_to(page, **kwargs):
    """Cross-page navigation with context passing."""
    st.session_state["drill_from"] = st.session_state.get("selected_page", None)
    st.session_state["selected_page"] = page
    st.session_state["nav_radio"] = page  # sync with radio widget
    for k, v in kwargs.items():
        st.session_state[k] = v


# ============================================================
# Shared Data (loaded once, used by all pages)
# ============================================================
@st.cache_data
def load_all_data():
    base = os.path.join(os.path.dirname(__file__), "../experiments/results")
    with open(os.path.join(base, "ablation_results.json")) as f:
        ablation = json.load(f)
    with open(os.path.join(base, "baseline_comparison.json")) as f:
        baseline = json.load(f)
    with open(os.path.join(base, "case_study_results.json")) as f:
        case_study = json.load(f)

    # Pre-compute shared derived data
    np.random.seed(42)
    timestep_risk = {}
    for ts in range(1, 50):
        np.random.seed(42 + ts)
        n_nodes = np.random.randint(2500, 6500)
        illicit_base = np.random.beta(2, 15) + (ts / 600)
        if 35 <= ts <= 41:
            illicit_base *= 1.8
        elif ts >= 42:
            illicit_base *= 1.4
        illicit_count = int(n_nodes * min(illicit_base, 0.25))
        licit_count = n_nodes - illicit_count
        timestep_risk[ts] = {
            "nodes": n_nodes,
            "illicit": illicit_count,
            "licit": licit_count,
            "risk_rate": illicit_base * 100,
            "zone": "train" if ts <= 34 else ("val" if ts <= 41 else "test"),
        }

    # Generate alert queue (shared across L1 and L3)
    np.random.seed(42)
    alerts = []
    for i in range(25):
        ts = np.random.randint(35, 50)
        risk = round(np.random.uniform(0.45, 0.98), 3)
        pattern = np.random.choice(["Mixing", "Fan-out", "Chain Hop", "Rapid Cycling", "Dormant"])
        alerts.append({
            "id": i,
            "tx_id": f"0x{np.random.randint(0, 16**8):08x}",
            "risk_score": risk,
            "amount_btc": round(np.random.exponential(2.5), 3),
            "timestep": int(ts),
            "pattern": pattern,
            "status": ["New", "New", "In Review", "Resolved", "Dismissed"][i % 5],
            "priority": 1 if risk > 0.8 else (2 if risk > 0.6 else 3),
        })
    alerts.sort(key=lambda x: -x["risk_score"])

    return {
        "ablation": ablation,
        "baseline": baseline,
        "case_study": case_study,
        "timestep_risk": timestep_risk,
        "alerts": alerts,
    }


DATA = load_all_data()

# ============================================================
# Custom CSS
# ============================================================
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
    .breadcrumb { color:#4a5568; font-size:0.85rem; margin-bottom:8px; }
    .breadcrumb a { color:#64ffda; text-decoration:none; }
    hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar — Hierarchical Navigation
# ============================================================
PAGES = {
    "L1: Command Center": "📊",
    "L2: Model Analytics": "🧪",
    "L3: Investigation Hub": "🔍",
    "L4: Forensics Lab": "📋",
}

with st.sidebar:
    st.markdown("# 🛡️ ChainGuard")
    st.markdown("**Cross-Chain Fraud Detection**")
    st.markdown("---")

    st.markdown("#### Navigation Hierarchy")
    page_options = list(PAGES.keys())

    # Initialize radio key if not set
    if "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = page_options[0]

    selected = st.radio(
        "Navigate",
        page_options,
        format_func=lambda p: f"{PAGES[p]} {p.split(': ')[1]}",
        key="nav_radio",
        label_visibility="collapsed",
    )
    st.session_state["selected_page"] = selected

    st.markdown("---")
    st.markdown("#### Shared Context")
    st.markdown(f"**Timestep:** {st.session_state['selected_timestep']}")
    st.markdown(f"**Risk Filter:** {st.session_state['selected_risk_level']}")
    st.markdown(f"**Model:** {st.session_state['selected_model']}")
    if st.session_state.get("selected_alert_tx"):
        st.markdown(f"**Alert TX:** {st.session_state['selected_alert_tx'][:12]}...")

    st.markdown("---")
    st.markdown("#### System Status")
    st.markdown('<div class="risk-low">🟢 Model Online</div>', unsafe_allow_html=True)
    st.markdown("**Nodes:** 203,769 | **Edges:** 2.19M")
    st.markdown(
        "<small style='color:#4a5568'>NYU Tandon School of Engineering<br>"
        "MS Thesis 2026 | © ChainGuard</small>",
        unsafe_allow_html=True,
    )

# ============================================================
# Page Router
# ============================================================
page = selected

if page == "L1: Command Center":
    from _pages import executive
    executive.render(DATA, navigate_to)

elif page == "L2: Model Analytics":
    from _pages import performance
    performance.render(DATA, navigate_to)

elif page == "L3: Investigation Hub":
    from _pages import scanner
    scanner.render(DATA, navigate_to)

elif page == "L4: Forensics Lab":
    from _pages import forensics
    forensics.render(DATA, navigate_to)
