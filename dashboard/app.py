"""
ChainGuard — Cross-Chain Cryptocurrency Fraud Detection Dashboard
Uses st.navigation() for translated sidebar labels.

Run: streamlit run dashboard/app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(page_title="ChainGuard | Fraud Detection", page_icon="\U0001f6e1\ufe0f",
                   layout="wide", initial_sidebar_state="expanded")

from shared import _init_session_state, _apply_css, _render_sidebar, load_data
from _lib.i18n import t

_init_session_state()
_apply_css()
_render_sidebar()

DATA = load_data()


def _nav(page, **kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.session_state["drill_from"] = page


# ═══════════════════════════════════════════
# Page functions with unique names
# ═══════════════════════════════════════════

def page_home():
    from pages_old.home import main as _run
    _run(DATA)


def page_executive():
    from _lib.executive import render
    render(DATA, _nav)


def page_performance():
    from _lib.performance import render
    render(DATA, _nav)


def page_scanner():
    from _lib.scanner import render
    render(DATA, _nav)


def page_network():
    from _lib.network import render
    render(DATA, _nav)


def page_forensics():
    from _lib.forensics import render
    render(DATA, _nav)


def page_explainability():
    from _lib.explainability import render
    render(DATA, _nav)


def page_blockchain():
    from _lib.blockchain import render
    render(DATA, _nav)


def page_alerts():
    from _lib.alerts import render
    render(DATA, _nav)


def page_upload():
    from _lib.data_upload import render
    render(DATA, _nav)


def page_comparison():
    from _lib.comparison import render
    render(DATA, _nav)


def page_search():
    from _lib.search import render
    render(DATA, _nav)


def page_activity():
    from _lib.activity import render
    render(DATA, _nav)


def page_cases():
    from _lib.case_management import render
    render(DATA, _nav)


# ═══════════════════════════════════════════
# Navigation with translated labels
# ═══════════════════════════════════════════
pages = [
    st.Page(page_home, title=t("home_title"), icon="\U0001f3e0", default=True),
    st.Page(page_executive, title=t("mod_executive"), icon="\U0001f4ca"),
    st.Page(page_performance, title=t("mod_performance"), icon="\U0001f9ea"),
    st.Page(page_scanner, title=t("mod_scanner"), icon="\U0001f50d"),
    st.Page(page_network, title=t("mod_network"), icon="\U0001f578\ufe0f"),
    st.Page(page_forensics, title=t("mod_forensics"), icon="\U0001f4cb"),
    st.Page(page_explainability, title=t("mod_explainability"), icon="\U0001f9e0"),
    st.Page(page_blockchain, title=t("mod_blockchain"), icon="\U0001f517"),
    st.Page(page_alerts, title=t("mod_alerts"), icon="\U0001f514"),
    st.Page(page_upload, title=t("mod_upload"), icon="\U0001f4e4"),
    st.Page(page_comparison, title=t("mod_comparison"), icon="\U0001f4ca"),
    st.Page(page_search, title=t("mod_search"), icon="\U0001f50e"),
    st.Page(page_cases, title=t("mod_cases"), icon="\U0001f4c1"),
    st.Page(page_activity, title=t("mod_activity"), icon="\U0001f4dc"),
]

nav = st.navigation(pages)
nav.run()
