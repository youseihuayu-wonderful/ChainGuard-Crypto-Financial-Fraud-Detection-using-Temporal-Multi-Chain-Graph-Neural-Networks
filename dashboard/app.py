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

# Hero
st.markdown(
    '<div style="padding:24px 0 8px 0">'
    '<div style="display:flex; align-items:center; gap:14px; margin-bottom:8px">'
    '<div style="width:48px; height:48px; background:linear-gradient(135deg,#00D4AA,#3B82F6); '
    'border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:24px">🛡️</div>'
    '<div>'
    '<h1 style="margin:0; font-size:2rem; color:#F9FAFB">ChainGuard</h1>'
    '</div></div>'
    '<p style="color:#9CA3AF; font-size:1rem; margin:4px 0 0 0">'
    'Cross-Chain Cryptocurrency Fraud Detection Platform<br>'
    'Powered by <span style="color:#00D4AA; font-weight:600">Temporal Heterogeneous Graph Neural Network</span></p>'
    '</div>',
    unsafe_allow_html=True,
)

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

# Module navigation
st.markdown(
    '<h4 style="color:#6B7280; text-transform:uppercase; letter-spacing:0.08em; font-size:0.75rem; '
    'margin-bottom:12px">Platform Modules</h4>',
    unsafe_allow_html=True,
)

modules = [
    ("📊", "Executive Dashboard", "TRM Labs", "CEO/CRO — KPIs, risk trends, alerts, fund flow, ROI", "#00D4AA"),
    ("🧪", "Model Performance", "ML Platform", "Data Scientists — Ablation study, baseline comparison, ROI", "#3B82F6"),
    ("🔍", "Transaction Scanner", "Elliptic Navigator", "Operations — Real-time transaction risk scoring", "#F59E0B"),
    ("🕸️", "Network Explorer", "Chainalysis Reactor", "Investigators — Graph topology, fraud clusters", "#8B5CF6"),
    ("📋", "Forensics Lab", "Compliance", "AML/Audit — Detection evidence, fraud patterns", "#EF4444"),
]

for icon, name, ref, desc, color in modules:
    st.markdown(
        f'<div class="pattern-card" style="border-left:3px solid {color}">'
        f'<div style="display:flex; align-items:center; gap:12px">'
        f'<span style="font-size:1.5rem">{icon}</span>'
        f'<div>'
        f'<div style="color:#F9FAFB; font-weight:600; font-size:0.95rem">{name}</div>'
        f'<span class="badge" style="background:{color}20; color:{color}">{ref}</span>'
        f'</div></div>'
        f'<p style="color:#9CA3AF; margin:8px 0 0 0; font-size:0.85rem">{desc}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# Connection map
st.markdown(
    '<h4 style="color:#6B7280; text-transform:uppercase; letter-spacing:0.08em; font-size:0.75rem; '
    'margin-bottom:12px">Data Flow</h4>',
    unsafe_allow_html=True,
)
st.markdown("""
All pages are interconnected — data and context flows between them:

- **Executive** → Scanner *(investigate TX)*, Network *(explore graph)*, Performance *(why)*, Forensics *(evidence)*
- **Scanner** → Network *(view neighbors)*, Forensics *(submit evidence)*
- **Network** → Scanner *(scan node)*, Forensics *(submit findings)*
- **Performance** → Scanner *(try model)*, Forensics *(see evidence)*
- **Forensics** → Executive *(back to overview)*, Performance *(model details)*
""")

st.markdown("---")
st.markdown(
    '<div style="text-align:center; padding:24px 0">'
    '<p style="color:#6B7280; font-size:0.8rem">NYU Tandon School of Engineering | MS Thesis 2026</p>'
    '<p style="color:#00D4AA; font-size:1.1rem; font-weight:600; font-family:JetBrains Mono,monospace">'
    'AUC-ROC: 0.8678 | Rank #1 across all baselines</p>'
    '</div>',
    unsafe_allow_html=True,
)
