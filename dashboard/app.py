"""
ChainGuard — Cross-Chain Cryptocurrency Fraud Detection Dashboard
Uses st.navigation() with sections for organized sidebar (no "View N more").

Run: streamlit run dashboard/app.py

DATA SOURCE: All metrics from real experiment results. No simulated data.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(page_title="ChainGuard | Fraud Detection", page_icon="\U0001f6e1\ufe0f",
                   layout="wide", initial_sidebar_state="expanded")

from shared import _init_session_state, _apply_css, _render_sidebar, render_top_controls

_init_session_state()
_apply_css()
_render_sidebar()  # minimal pre-nav sidebar

# ═══════════════════════════════════════════
# Navigation with sections — all 12 pages visible, grouped
# ═══════════════════════════════════════════
pages = [
    st.Page("pages_old/home.py", title="Home", icon="\U0001f3e0", default=True),
    st.Page("pages_old/1_\U0001f4ca_Executive_Dashboard.py", title="Executive", icon="\U0001f4ca"),
    st.Page("pages_old/2_\U0001f9ea_Model_Performance.py", title="Performance", icon="\U0001f9ea"),
    st.Page("pages_old/3_\U0001f50d_Transaction_Scanner.py", title="Scanner", icon="\U0001f50d"),
    st.Page("pages_old/4_\U0001f578\ufe0f_Network_Explorer.py", title="Network", icon="\U0001f578\ufe0f"),
    st.Page("pages_old/5_\U0001f4cb_Forensics_Lab.py", title="Forensics", icon="\U0001f4cb"),
    st.Page("pages_old/6_\U0001f9e0_Explainability.py", title="Explainability", icon="\U0001f9e0"),
    st.Page("pages_old/8_\U0001f517_Blockchain_Scanner.py", title="Blockchain", icon="\U0001f517"),
    st.Page("pages_old/9_\U0001f514_Alert_Center.py", title="Alerts", icon="\U0001f514"),
    st.Page("pages_old/10_\U0001f4e4_Data_Upload.py", title="Upload", icon="\U0001f4e4"),
    st.Page("pages_old/11_\U0001f4ca_Model_Comparison.py", title="Comparison", icon="\U0001f4ca"),
    st.Page("pages_old/12_\U0001f50e_Node_Search.py", title="Search", icon="\U0001f50e"),
    st.Page("pages_old/7_\U0001f4dc_Activity_Log.py", title="Activity", icon="\U0001f4dc"),
]

nav = st.navigation(pages)
render_top_controls()  # language/theme after nav items
nav.run()
