TITLE = "ChainGuard | Executive Dashboard"
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared import setup_page, load_data
import streamlit as st

PAGE_MAP = {
    "Executive": "pages_old/1_📊_Executive_Dashboard.py",
    "Performance": "pages_old/2_🧪_Model_Performance.py",
    "Scanner": "pages_old/3_🔍_Transaction_Scanner.py",
    "Network": "pages_old/4_🕸️_Network_Explorer.py",
    "Forensics": "pages_old/5_📋_Forensics_Lab.py",
    "Explainability": "pages_old/6_🧠_Explainability.py",
    "Activity": "pages_old/7_📜_Activity_Log.py",
    "Blockchain": "pages_old/8_🔗_Blockchain_Scanner.py",
    "Alerts": "pages_old/9_🔔_Alert_Center.py",
    "Upload": "pages_old/10_📤_Data_Upload.py",
    "Comparison": "pages_old/11_📊_Model_Comparison.py",
    "Search": "pages_old/12_🔎_Node_Search.py",
}

def nav(page, **kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v
    if page in PAGE_MAP:
        st.switch_page(PAGE_MAP[page])

# setup handled by app.py
DATA = load_data()

# Import and run the render function
from _lib.executive import render
render(DATA, nav)
