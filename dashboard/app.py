"""
ChainGuard - Cross-Chain Cryptocurrency Fraud Detection Dashboard
Enterprise-grade monitoring and analytics platform powered by TH-GNN.

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import json
import os

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
# Custom CSS for Enterprise Look
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: #e0e0e0;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetric"] label {
        color: #8892b0 !important;
        font-size: 0.85rem;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ccd6f6 !important;
        font-size: 2rem;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #64ffda !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ccd6f6 !important;
    }
    p, li, span {
        color: #8892b0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        color: #8892b0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(100, 255, 218, 0.1) !important;
        color: #64ffda !important;
    }

    /* Custom alert box */
    .risk-high {
        background: rgba(255, 82, 82, 0.1);
        border-left: 4px solid #ff5252;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .risk-medium {
        background: rgba(255, 152, 0, 0.1);
        border-left: 4px solid #ff9800;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .risk-low {
        background: rgba(100, 255, 218, 0.1);
        border-left: 4px solid #64ffda;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1);
    }

    /* Data frames */
    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data Loading
# ============================================================
@st.cache_data
def load_results():
    base = os.path.join(os.path.dirname(__file__), "../experiments/results")
    with open(os.path.join(base, "ablation_results.json")) as f:
        ablation = json.load(f)
    with open(os.path.join(base, "baseline_comparison.json")) as f:
        baseline = json.load(f)
    with open(os.path.join(base, "case_study_results.json")) as f:
        case_study = json.load(f)
    return ablation, baseline, case_study


ablation, baseline, case_study = load_results()


# ============================================================
# Sidebar Navigation
# ============================================================
with st.sidebar:
    st.markdown("# 🛡️ ChainGuard")
    st.markdown("**Cross-Chain Fraud Detection**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "🧪 Model Performance",
            "🔍 Transaction Scanner",
            "🕸️ Network Explorer",
            "📋 Case Study & Forensics",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### System Status")
    st.markdown('<div class="risk-low">🟢 Model Online</div>', unsafe_allow_html=True)
    st.markdown(f"**Model:** TH-GNN v1.0")
    st.markdown(f"**Dataset:** Elliptic Bitcoin")
    st.markdown(f"**Nodes:** 203,769")
    st.markdown(f"**Edges:** 2,193,245")
    st.markdown("---")
    st.markdown(
        "<small style='color:#4a5568'>NYU Tandon School of Engineering<br>"
        "MS Thesis Research Project<br>"
        "© 2026 ChainGuard</small>",
        unsafe_allow_html=True,
    )


# ============================================================
# Page Router
# ============================================================
if page == "📊 Executive Dashboard":
    from pages import executive
    executive.render(ablation, baseline, case_study)

elif page == "🧪 Model Performance":
    from pages import performance
    performance.render(ablation, baseline)

elif page == "🔍 Transaction Scanner":
    from pages import scanner
    scanner.render()

elif page == "🕸️ Network Explorer":
    from pages import network
    network.render()

elif page == "📋 Case Study & Forensics":
    from pages import forensics
    forensics.render(case_study, ablation)
