TITLE = "ChainGuard | GNN Explainability"
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import setup_page, load_data
import streamlit as st

PAGE_MAP = {
    "Executive": "pages/1_📊_Executive_Dashboard.py",
    "Performance": "pages/2_🧪_Model_Performance.py",
    "Scanner": "pages/3_🔍_Transaction_Scanner.py",
    "Network": "pages/4_🕸️_Network_Explorer.py",
    "Forensics": "pages/5_📋_Forensics_Lab.py",
    "Explainability": "pages/6_🧠_Explainability.py",
    "Activity": "pages/7_📜_Activity_Log.py",
    "Blockchain": "pages/8_🔗_Blockchain_Scanner.py",
    "Alerts": "pages/9_🔔_Alert_Center.py",
    "Upload": "pages/10_📤_Data_Upload.py",
    "Comparison": "pages/11_📊_Model_Comparison.py",
    "Search": "pages/12_🔎_Node_Search.py",
}

def nav(page, **kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v
    if page in PAGE_MAP:
        st.switch_page(PAGE_MAP[page])

setup_page(TITLE)
DATA = load_data()

from _lib.explainability import render
render(DATA, nav)
