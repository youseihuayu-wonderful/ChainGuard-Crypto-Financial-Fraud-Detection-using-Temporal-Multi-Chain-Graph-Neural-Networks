"""
ChainGuard Design System
Bloomberg Terminal / Chainalysis Reactor inspired financial UI.

Color System:
  Background:  #0A0E17 (deep navy black)
  Surface:     #111827 (card background)
  Surface-2:   #1F2937 (elevated surface)
  Border:      #1F2937 (subtle borders)
  Text-1:      #F9FAFB (primary text)
  Text-2:      #9CA3AF (secondary text)
  Text-3:      #6B7280 (tertiary/muted)
  Accent:      #00D4AA (teal - primary action)
  Positive:    #10B981 (green - gains/success)
  Negative:    #EF4444 (red - losses/danger)
  Warning:     #F59E0B (amber - caution)
  Info:        #3B82F6 (blue - informational)
"""

import streamlit as st
import json
import os
import numpy as np


def setup_page(title="ChainGuard | Fraud Detection"):
    """Call at top of every page to set config and apply styling."""
    st.set_page_config(page_title=title, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
    _init_session_state()
    _apply_css()
    _render_sidebar()


def _init_session_state():
    defaults = {
        "selected_timestep": 25,
        "selected_risk_level": "ALL",
        "selected_alert_tx": None,
        "selected_model": "M3",
        "drill_from": None,
        "lang": "en",
        "theme": "dark",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================
# Plotly chart theme (use in all pages)
# ============================================================
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.5)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#E5E7EB", size=12),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(gridcolor="rgba(75,85,99,0.3)", zerolinecolor="rgba(75,85,99,0.3)", color="#9CA3AF"),
    yaxis=dict(gridcolor="rgba(75,85,99,0.3)", zerolinecolor="rgba(75,85,99,0.3)", color="#9CA3AF"),
    legend=dict(font=dict(color="#9CA3AF")),
)

# Color palette for charts
COLORS = {
    "accent": "#00D4AA",
    "positive": "#10B981",
    "negative": "#EF4444",
    "warning": "#F59E0B",
    "info": "#3B82F6",
    "purple": "#8B5CF6",
    "gray": "#6B7280",
    "surface": "#1F2937",
}


def _apply_css():
    theme = st.session_state.get("theme", "dark")
    is_light = theme == "light"

    # Theme-dependent color variables
    if is_light:
        bg = "#FFFFFF"
        surface = "#F3F4F6"
        surface2 = "#E5E7EB"
        border = "#E5E7EB"
        sidebar_bg = "#F9FAFB"
        text1 = "#111827"
        text2 = "#6B7280"
        text3 = "#9CA3AF"
        text_strong = "#111827"
        accent = "#0D9488"
        accent_hover = "#0F766E"
        accent_bg = "rgba(13,148,136,0.08)"
        accent_bg2 = "rgba(13,148,136,0.12)"
        header_bg = "#FFFFFF"
        metric_label = "#6B7280"
        metric_value = "#111827"
        metric_delta = "#059669"
        tab_bg = "#F3F4F6"
        tab_active_bg = "#FFFFFF"
        btn_bg = "#F3F4F6"
        btn_text = "#374151"
        btn_primary_text = "#FFFFFF"
        input_bg = "#FFFFFF"
        grid_color = "rgba(209,213,219,0.5)"
        glass_bg = "rgba(243,244,246,0.9)"
        stat_row_bg = "#F9FAFB"
        breadcrumb_bg = "#F3F4F6"
    else:
        bg = "#0A0E17"
        surface = "#111827"
        surface2 = "#1F2937"
        border = "#1F2937"
        sidebar_bg = "#0D1117"
        text1 = "#F9FAFB"
        text2 = "#9CA3AF"
        text3 = "#6B7280"
        text_strong = "#E5E7EB"
        accent = "#00D4AA"
        accent_hover = "#00E6B8"
        accent_bg = "rgba(0,212,170,0.08)"
        accent_bg2 = "rgba(0,212,170,0.12)"
        header_bg = "#0A0E17"
        metric_label = "#6B7280"
        metric_value = "#F9FAFB"
        metric_delta = "#10B981"
        tab_bg = "#111827"
        tab_active_bg = "#1F2937"
        btn_bg = "#111827"
        btn_text = "#D1D5DB"
        btn_primary_text = "#0A0E17"
        input_bg = "#111827"
        grid_color = "rgba(75,85,99,0.3)"
        glass_bg = "rgba(17,24,39,0.8)"
        stat_row_bg = "#111827"
        breadcrumb_bg = "#111827"

    st.markdown(f"""
    <style>
        /* ══════════════════════════════════════════════
           BLOOMBERG / CHAINALYSIS FINANCIAL DESIGN SYSTEM
           Theme: {theme}
           ══════════════════════════════════════════════ */

        /* Google Fonts - Inter (professional finance font) + Material Symbols for sidebar icons */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Material+Symbols+Rounded&display=swap');

        /* ── Base ── */
        .stApp {{
            background: {bg} !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}
        * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
            border-right: 1px solid {border} !important;
        }}
        [data-testid="stSidebar"] * {{ color: {text2} !important; }}
        [data-testid="stSidebar"] a {{
            color: {text2} !important;
            transition: color 0.2s, background 0.2s;
            border-radius: 6px;
        }}
        [data-testid="stSidebar"] a:hover {{ color: {accent} !important; background: {accent_bg} !important; }}
        [data-testid="stSidebar"] a[aria-current="page"] {{
            color: {accent} !important;
            background: {accent_bg2} !important;
            font-weight: 600 !important;
        }}

        /* ── Typography ── */
        h1 {{
            color: {text1} !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            font-size: 1.875rem !important;
        }}
        h2 {{
            color: {text1} !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            font-size: 1.375rem !important;
        }}
        h3 {{
            color: {text_strong} !important;
            font-weight: 600 !important;
            font-size: 1.125rem !important;
        }}
        h4 {{
            color: {text2} !important;
            font-weight: 500 !important;
            font-size: 0.975rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        p, li {{ color: {text2} !important; line-height: 1.6 !important; }}
        strong {{ color: {text_strong} !important; }}

        /* ── Metric Cards (Bloomberg style) ── */
        [data-testid="stMetric"] {{
            background: {surface} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            padding: 20px 16px !important;
            transition: border-color 0.2s;
        }}
        [data-testid="stMetric"]:hover {{
            border-color: {surface2} !important;
        }}
        [data-testid="stMetric"] label {{
            color: {metric_label} !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}
        [data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {metric_value} !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            font-family: 'JetBrains Mono', monospace !important;
        }}
        [data-testid="stMetric"] [data-testid="stMetricDelta"] svg {{ display: none; }}
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {{
            color: {metric_delta} !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
        }}

        /* ── Tabs (clean segment control) ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px !important;
            background: {tab_bg};
            border-radius: 8px;
            padding: 4px;
            border: 1px solid {border};
        }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent !important;
            border-radius: 6px !important;
            color: {text3} !important;
            padding: 8px 20px !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: {tab_active_bg} !important;
            color: {accent} !important;
            font-weight: 600 !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            background: {btn_bg} !important;
            border: 1px solid {border} !important;
            color: {btn_text} !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            padding: 8px 16px !important;
            transition: all 0.2s !important;
        }}
        .stButton > button:hover {{
            border-color: {accent} !important;
            color: {accent} !important;
            background: {accent_bg} !important;
        }}
        .stButton > button[kind="primary"] {{
            background: {accent} !important;
            color: {btn_primary_text} !important;
            border: none !important;
            font-weight: 600 !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: {accent_hover} !important;
        }}

        /* ── Data Frames ── */
        [data-testid="stDataFrame"] {{
            border: 1px solid {border} !important;
            border-radius: 8px !important;
        }}

        /* ── Selectbox / Inputs ── */
        [data-baseweb="select"] > div {{
            background: {input_bg} !important;
            border-color: {border} !important;
        }}
        input, textarea {{
            background: {input_bg} !important;
            border-color: {border} !important;
            color: {text_strong} !important;
        }}

        /* ── Slider ── */
        [data-testid="stSlider"] > div > div > div {{ color: {text2} !important; }}

        /* ── Dividers ── */
        hr {{ border-color: {border} !important; }}

        /* ── Risk Status Cards ── */
        .risk-critical {{
            background: rgba(239,68,68,0.08);
            border: 1px solid rgba(239,68,68,0.2);
            border-left: 3px solid #EF4444;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 6px 0;
        }}
        .risk-high {{
            background: rgba(239,68,68,0.06);
            border: 1px solid rgba(239,68,68,0.15);
            border-left: 3px solid #EF4444;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 6px 0;
        }}
        .risk-medium {{
            background: rgba(245,158,11,0.06);
            border: 1px solid rgba(245,158,11,0.15);
            border-left: 3px solid #F59E0B;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 6px 0;
        }}
        .risk-low {{
            background: rgba(16,185,129,0.06);
            border: 1px solid rgba(16,185,129,0.15);
            border-left: 3px solid #10B981;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 6px 0;
        }}

        /* ── Glass Card (for featured content) ── */
        .glass-card {{
            background: {glass_bg};
            backdrop-filter: blur(12px);
            border: 1px solid {border};
            border-radius: 12px;
            padding: 24px;
            margin: 8px 0;
        }}

        /* ── Status Badge ── */
        .badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .badge-red {{ background: rgba(239,68,68,0.15); color: #EF4444; }}
        .badge-amber {{ background: rgba(245,158,11,0.15); color: #F59E0B; }}
        .badge-green {{ background: rgba(16,185,129,0.15); color: #10B981; }}
        .badge-blue {{ background: rgba(59,130,246,0.15); color: #3B82F6; }}
        .badge-gray {{ background: rgba(107,114,128,0.15); color: #9CA3AF; }}

        /* ── Pattern Card ── */
        .pattern-card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 18px;
            margin: 8px 0;
            transition: border-color 0.2s;
        }}
        .pattern-card:hover {{ border-color: {surface2}; }}

        /* ── Stat Row ── */
        .stat-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            background: {stat_row_bg};
            border: 1px solid {border};
            border-radius: 6px;
            margin: 4px 0;
        }}

        /* ── Breadcrumb ── */
        .breadcrumb {{
            color: {text3};
            font-size: 0.8rem;
            margin-bottom: 8px;
            padding: 6px 12px;
            background: {breadcrumb_bg};
            border-radius: 6px;
            display: inline-block;
        }}

        /* ── Show all sidebar pages (no "View N more" truncation) ── */
        [data-testid="stSidebarNav"] > ul {{
            max-height: none !important;
            overflow: visible !important;
        }}
        [data-testid="stSidebarNav"] details {{
            display: none !important;
        }}
        [data-testid="stSidebarNav"] > ul > li {{
            display: list-item !important;
        }}

        /* ── Sidebar collapse button icon fix ── */
        [data-testid="stSidebarCollapseButton"] button {{
            font-family: 'Material Symbols Rounded' !important;
            overflow: hidden !important;
            width: 32px !important;
            height: 32px !important;
        }}
        [data-testid="stSidebarCollapseButton"] button span {{
            font-family: 'Material Symbols Rounded' !important;
        }}

        /* ── Hide Streamlit branding ── */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{ background: {header_bg} !important; }}
    </style>
    """, unsafe_allow_html=True)


def _render_sidebar():
    from _lib.i18n import t
    with st.sidebar:
        st.markdown(
            '<div style="padding:8px 0 16px 0">'
            '<div style="display:flex; align-items:center; gap:10px">'
            '<div style="width:36px; height:36px; background:linear-gradient(135deg,#00D4AA,#3B82F6); '
            'border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:18px">\U0001f6e1\ufe0f</div>'
            '<div>'
            '<div style="font-size:1.1rem; font-weight:700; color:#F9FAFB; letter-spacing:-0.02em">ChainGuard</div>'
            f'<div style="font-size:0.7rem; color:#6B7280; letter-spacing:0.05em; text-transform:uppercase">{t("platform_subtitle")}</div>'
            '</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Language dropdown (20+ languages)
        from _lib.i18n import SUPPORTED_LANGUAGES
        current_lang = st.session_state.get("lang", "en")
        lang_codes = list(SUPPORTED_LANGUAGES.keys())
        lang_labels = list(SUPPORTED_LANGUAGES.values())
        current_idx = lang_codes.index(current_lang) if current_lang in lang_codes else 0
        selected_lang = st.selectbox(
            "\U0001f310 " + t("language_label"),
            options=lang_codes,
            format_func=lambda x: SUPPORTED_LANGUAGES.get(x, x),
            index=current_idx,
            key="lang_select",
        )
        if selected_lang != st.session_state.get("lang"):
            st.session_state["lang"] = selected_lang
            st.rerun()

        # Theme toggle
        theme_options = {"dark": t("dark"), "light": t("light")}
        current_theme = st.session_state.get("theme", "dark")
        selected_theme = st.radio(
            t("theme"),
            options=list(theme_options.keys()),
            format_func=lambda x: theme_options[x],
            index=0 if current_theme == "dark" else 1,
            key="theme_toggle",
            horizontal=True,
        )
        if selected_theme != st.session_state.get("theme"):
            st.session_state["theme"] = selected_theme
            st.rerun()


def render_sidebar_bottom():
    """No sidebar bottom widgets — all space reserved for navigation."""
    pass


def render_top_controls():
    """Render language/theme controls in main content area (top-right)."""
    from _lib.i18n import t
    with st.sidebar:
        st.markdown("---")
        lang_options = {"en": "EN", "zh": "\u4e2d\u6587"}
        current_lang = st.session_state.get("lang", "en")
        selected_lang = st.radio(
            t("language_label"),
            options=list(lang_options.keys()),
            format_func=lambda x: lang_options[x],
            index=0 if current_lang == "en" else 1,
            key="lang_toggle",
            horizontal=True,
        )
        if selected_lang != st.session_state.get("lang"):
            st.session_state["lang"] = selected_lang
            st.rerun()

        theme_options = {"dark": "\u263e", "light": "\u2600\ufe0f"}
        current_theme = st.session_state.get("theme", "dark")
        selected_theme = st.radio(
            t("theme"),
            options=list(theme_options.keys()),
            format_func=lambda x: theme_options[x],
            index=0 if current_theme == "dark" else 1,
            key="theme_toggle",
            horizontal=True,
        )
        if selected_theme != st.session_state.get("theme"):
            st.session_state["theme"] = selected_theme
            st.rerun()


@st.cache_data
def load_data():
    """Load all experiment results and real dataset statistics. Cached across pages.

    ALL data is from real experiments or the actual Elliptic dataset.
    No simulated or mock data.
    """
    base = os.path.join(os.path.dirname(__file__), "../experiments/results")

    try:
        with open(os.path.join(base, "ablation_results.json")) as f:
            ablation = json.load(f)
        with open(os.path.join(base, "baseline_comparison.json")) as f:
            baseline = json.load(f)
        with open(os.path.join(base, "case_study_results.json")) as f:
            case_study = json.load(f)
    except FileNotFoundError as e:
        st.error(f"Experiment data not found: {e}. Run experiments first.")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Corrupt experiment data: {e}")
        st.stop()

    # ── Real timestep statistics from Elliptic dataset ──
    try:
        with open(os.path.join(base, "timestep_stats.json")) as f:
            ts_raw = json.load(f)
        timestep_risk = {int(k): v for k, v in ts_raw.items()}
    except FileNotFoundError:
        st.error("timestep_stats.json not found. Run data preprocessing first.")
        st.stop()

    # ── Real illicit transactions from test set ──
    try:
        with open(os.path.join(base, "real_test_illicit.json")) as f:
            real_illicit = json.load(f)
    except FileNotFoundError:
        real_illicit = []

    # ── Real graph data per timestep ──
    try:
        with open(os.path.join(base, "graph_data.json")) as f:
            graph_data = json.load(f)
    except FileNotFoundError:
        graph_data = {}

    return {
        "ablation": ablation, "baseline": baseline, "case_study": case_study,
        "timestep_risk": timestep_risk, "real_test_illicit": real_illicit,
        "graph_data": graph_data,
    }
