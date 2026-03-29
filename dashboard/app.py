"""
ChainGuard — Cross-Chain Cryptocurrency Fraud Detection Dashboard
Home page. Navigate to specific pages via the sidebar.

Run: streamlit run dashboard/app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from shared import setup_page, load_data

import streamlit as st

setup_page()
DATA = load_data()
cs = DATA["case_study"]

st.markdown("# 🛡️ ChainGuard")
st.markdown("### Cross-Chain Cryptocurrency Fraud Detection Platform")
st.markdown("Powered by **Temporal Heterogeneous Graph Neural Network (TH-GNN)**")
st.markdown("---")

# KPI overview
k1, k2, k3, k4, k5 = st.columns(5)
total_detected = cs["both_detect"] + cs["m3_only"]
k1.metric("AUC-ROC", "0.8678", "+12.3% vs GCN")
k2.metric("Detected", f"{total_detected}/{cs['total_illicit_test']}", f"{total_detected/cs['total_illicit_test']:.0%}")
k3.metric("Precision", "71.68%", "+43.9%")
k4.metric("FP Rate", "0.93%", "-2.1%", delta_color="inverse")
k5.metric("Savings", "$12.8M/yr", "annual projection")

st.markdown("---")

st.markdown("### 📂 Navigate the Platform")
st.markdown("Use the **sidebar** to access each module:")
st.markdown("")

modules = [
    ("📊 Executive", "TRM Labs style", "CEO/CRO — KPIs, risk trends, alerts, fund flow, ROI"),
    ("🧪 Performance", "Internal ML Platform", "Data Scientists — Ablation study, baseline comparison, ROI analysis"),
    ("🔍 Scanner", "Elliptic Navigator style", "Operations — Transaction risk scoring, behavior flags"),
    ("🕸️ Network", "Chainalysis Reactor style", "Investigators — Graph topology, fraud cluster visualization"),
    ("📋 Forensics", "Compliance Reports", "AML/Audit — Detection evidence, fraud patterns, research conclusions"),
]

for icon_name, ref, desc in modules:
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.03); border-radius:8px; padding:16px; margin:8px 0; "
        f"border-left:4px solid #64ffda'>"
        f"<h4 style='margin:0; color:#ccd6f6'>{icon_name}</h4>"
        f"<small style='color:#64ffda'>{ref}</small><br>"
        f"<span style='color:#8892b0'>{desc}</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

st.markdown("### 🔗 Page Connections")
st.markdown("""
All pages are interconnected — data and context flows between them:

- **Executive** → Scanner (investigate TX), Network (explore graph), Performance (why), Forensics (evidence)
- **Scanner** → Network (view neighbors), Forensics (submit evidence)
- **Network** → Scanner (scan node), Forensics (submit findings)
- **Performance** → Scanner (try model), Forensics (see evidence)
- **Forensics** → Executive (back to overview), Performance (model details)
""")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; padding:20px'>"
    "<p style='color:#8892b0'>NYU Tandon School of Engineering · MS Thesis 2026</p>"
    "<p style='color:#64ffda; font-size:1.2rem'>AUC-ROC: 0.8678 | #1 across all baselines</p>"
    "</div>",
    unsafe_allow_html=True,
)
