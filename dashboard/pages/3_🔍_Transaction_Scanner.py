"""Transaction Scanner — Standalone Streamlit page."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import setup_page, load_data
from _pages.scanner import render

setup_page("ChainGuard | Transaction Scanner")
DATA = load_data()

def _nav(page, **kwargs):
    """Navigation stub — cross-page links use st.switch_page in multipage mode."""
    import streamlit as st
    page_map = {
        "Executive": "pages/1_📊_Executive_Dashboard.py",
        "Performance": "pages/2_🧪_Model_Performance.py",
        "Scanner": "pages/3_🔍_Transaction_Scanner.py",
        "Network": "pages/4_🕸️_Network_Explorer.py",
        "Forensics": "pages/5_📋_Forensics_Lab.py",
    }
    for k, v in kwargs.items():
        st.session_state[k] = v
    if page in page_map:
        st.switch_page(page_map[page])

render(DATA, _nav)
