"""Home page for ChainGuard dashboard."""
import streamlit as st
from _lib.i18n import t


def main(DATA):
    cs = DATA["case_study"]
    abl = DATA["ablation"]

    best = abl["M3"]
    gcn = abl["M1"]
    auc_delta = (best["auc_roc"] - gcn["auc_roc"]) / gcn["auc_roc"]
    prec_delta = (best["precision"] - gcn["precision"]) / gcn["precision"]
    fdr = 1 - best["precision"]
    gcn_fdr = 1 - gcn["precision"]

    st.markdown(
        '<div style="padding:24px 0 8px 0">'
        '<div style="display:flex; align-items:center; gap:14px; margin-bottom:8px">'
        '<div style="width:48px; height:48px; background:linear-gradient(135deg,#00D4AA,#3B82F6); '
        'border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:24px">\U0001f6e1\ufe0f</div>'
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

    k1, k2, k3, k4 = st.columns(4)
    total_detected = cs["both_detect"] + cs["m3_only"]
    k1.metric(t("auc_roc"), f"{best['auc_roc']:.4f}", f"+{auc_delta:.1%} vs GCN")
    k2.metric(t("detected"), f"{total_detected}/{cs['total_illicit_test']}", f"{total_detected/cs['total_illicit_test']:.0%}")
    k3.metric(t("precision"), f"{best['precision']:.2%}", f"+{prec_delta:.1%} vs GCN")
    k4.metric(t("false_alarm"), f"{fdr:.1%}", f"-{gcn_fdr - fdr:.1%} vs GCN", delta_color="inverse")

    st.markdown("---")

    st.markdown(
        f'<h4 style="color:#6B7280; text-transform:uppercase; letter-spacing:0.08em; font-size:0.75rem; '
        f'margin-bottom:12px">{t("platform_modules")}</h4>',
        unsafe_allow_html=True,
    )

    modules = [
        ("\U0001f4ca", t("mod_executive"), t("mod_executive_ref"), t("mod_executive_desc"), "#00D4AA"),
        ("\U0001f9ea", t("mod_performance"), t("mod_performance_ref"), t("mod_performance_desc"), "#3B82F6"),
        ("\U0001f50d", t("mod_scanner"), t("mod_scanner_ref"), t("mod_scanner_desc"), "#F59E0B"),
        ("\U0001f578\ufe0f", t("mod_network"), t("mod_network_ref"), t("mod_network_desc"), "#8B5CF6"),
        ("\U0001f4cb", t("mod_forensics"), t("mod_forensics_ref"), t("mod_forensics_desc"), "#EF4444"),
        ("\U0001f9e0", t("mod_explainability"), t("mod_explainability_ref"), t("mod_explainability_desc"), "#8B5CF6"),
        ("\U0001f517", t("mod_blockchain"), t("mod_blockchain_ref"), t("mod_blockchain_desc"), "#3B82F6"),
        ("\U0001f514", t("mod_alerts"), t("mod_alerts_ref"), t("mod_alerts_desc"), "#EF4444"),
        ("\U0001f4e4", t("mod_upload"), t("mod_upload_ref"), t("mod_upload_desc"), "#F59E0B"),
        ("\U0001f4ca", t("mod_comparison"), t("mod_comparison_ref"), t("mod_comparison_desc"), "#00D4AA"),
        ("\U0001f50e", t("mod_search"), t("mod_search_ref"), t("mod_search_desc"), "#8B5CF6"),
        ("\U0001f4c1", t("mod_cases"), t("mod_cases_ref"), t("mod_cases_desc"), "#F59E0B"),
        ("\U0001f4dc", t("mod_activity"), t("mod_activity_ref"), t("mod_activity_desc"), "#10B981"),
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

    bl_res = DATA["baseline"]["results"]
    rank = next(i for i, (k, _) in enumerate(sorted(bl_res.items(), key=lambda x: -x[1]["auc_roc"]), 1) if k == "thgnn_m3_ours")
    st.markdown(
        f'<div style="text-align:center; padding:24px 0">'
        f'<p style="color:#6B7280; font-size:0.8rem">{t("nyu_footer")}</p>'
        f'<p style="color:#00D4AA; font-size:1.1rem; font-weight:600; font-family:JetBrains Mono,monospace">'
        f'AUC-ROC: {best["auc_roc"]:.4f} | {t("rank_across_baselines").format(rank=rank, n=len(bl_res))}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
