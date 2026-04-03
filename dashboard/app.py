"""
ChainGuard — Cross-Chain Cryptocurrency Fraud Detection Dashboard
Home page. Navigate to specific pages via the sidebar.

Run: streamlit run dashboard/app.py

DATA SOURCE: All metrics from real experiment results. No simulated data.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from shared import setup_page, load_data

import streamlit as st
from _lib.i18n import t

setup_page()
DATA = load_data()
cs = DATA["case_study"]
abl = DATA["ablation"]

# Best model metrics (from experiment data)
best = abl["M3"]  # R-GCN + Heterogeneous Edges — highest AUC
gcn = abl["M1"]   # GCN baseline
auc_delta = (best["auc_roc"] - gcn["auc_roc"]) / gcn["auc_roc"]
prec_delta = (best["precision"] - gcn["precision"]) / gcn["precision"]
fdr = 1 - best["precision"]
gcn_fdr = 1 - gcn["precision"]

# Hero
st.markdown(
    '<div style="padding:24px 0 8px 0">'
    '<div style="display:flex; align-items:center; gap:14px; margin-bottom:8px">'
    '<div style="width:48px; height:48px; background:linear-gradient(135deg,#00D4AA,#3B82F6); '
    'border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:24px">🛡️</div>'
    '<div>'
    '<h1 style="margin:0; font-size:2rem; color:#F9FAFB">ChainGuard</h1>'
    '</div></div>'
    f'<p style="color:#9CA3AF; font-size:1rem; margin:4px 0 0 0">'
    f'{t("home_subtitle")}<br>'
    f'{t("home_powered_by")} <span style="color:#00D4AA; font-weight:600">{t("home_thgnn")}</span></p>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# KPI overview — all values derived from experiment data
k1, k2, k3, k4 = st.columns(4)
total_detected = cs["both_detect"] + cs["m3_only"]
k1.metric(t("auc_roc"), f"{best['auc_roc']:.4f}", f"+{auc_delta:.1%} vs GCN")
k2.metric(t("detected"), f"{total_detected}/{cs['total_illicit_test']}", f"{total_detected/cs['total_illicit_test']:.0%}")
k3.metric(t("precision"), f"{best['precision']:.2%}", f"+{prec_delta:.1%} vs GCN")
k4.metric(t("false_alarm"), f"{fdr:.1%}", f"-{gcn_fdr - fdr:.1%} vs GCN", delta_color="inverse")

st.markdown("---")

# Module navigation
st.markdown(
    f'<h4 style="color:#6B7280; text-transform:uppercase; letter-spacing:0.08em; font-size:0.75rem; '
    f'margin-bottom:12px">{t("platform_modules")}</h4>',
    unsafe_allow_html=True,
)

modules = [
    ("📊", "Executive Dashboard", "TRM Labs", "CEO/CRO — KPIs, real dataset statistics, detection analysis", "#00D4AA"),
    ("🧪", "Model Performance", "ML Platform", "Data Scientists — Ablation study, baseline comparison", "#3B82F6"),
    ("🔍", "Transaction Scanner", "Elliptic Navigator", "Operations — Rule-based transaction risk scoring", "#F59E0B"),
    ("🕸️", "Network Explorer", "Chainalysis Reactor", "Investigators — Real Elliptic graph topology", "#8B5CF6"),
    ("📋", "Forensics Lab", "Compliance", "AML/Audit — Detection evidence, research findings", "#EF4444"),
    ("🧠", "GNN Explainability", "XAI", "Research — Real feature importance, node explanations, training curves", "#8B5CF6"),
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
    f'<h4 style="color:#6B7280; text-transform:uppercase; letter-spacing:0.08em; font-size:0.75rem; '
    f'margin-bottom:12px">{t("data_flow")}</h4>',
    unsafe_allow_html=True,
)
st.markdown(f"""{t("data_flow_desc")}

- **Executive** → Scanner *(investigate)*, Network *(explore graph)*, Performance *(why)*, Forensics *(evidence)*
- **Scanner** → Network *(view neighbors)*, Forensics *(submit evidence)*
- **Network** → Scanner *(scan node)*, Forensics *(submit findings)*
- **Performance** → Scanner *(try model)*, Forensics *(see evidence)*
- **Forensics** → Executive *(back to overview)*, Performance *(model details)*
""")

st.markdown("---")
bl_res = DATA["baseline"]["results"]
rank = next(i for i, (k, _) in enumerate(sorted(bl_res.items(), key=lambda x: -x[1]["auc_roc"]), 1) if k == "thgnn_m3_ours")
st.markdown(
    f'<div style="text-align:center; padding:24px 0">'
    f'<p style="color:#6B7280; font-size:0.8rem">NYU Tandon School of Engineering | MS Thesis 2026</p>'
    f'<p style="color:#00D4AA; font-size:1.1rem; font-weight:600; font-family:JetBrains Mono,monospace">'
    f'AUC-ROC: {best["auc_roc"]:.4f} | Rank #{rank} across {len(bl_res)} baselines</p>'
    f'</div>',
    unsafe_allow_html=True,
)
