# app.py
import logging
from pathlib import Path
import os
import streamlit as st
import locale
locale.setlocale(locale.LC_TIME, "en_GB.UTF-8")
from utils.branding import APP_NAME, APP_TAGLINE, APP_ICON



# HIDE DEFAULT STREAMLIT PAGES NAVIGATION
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
# Hide the "Pages" sidebar menu that Streamlit adds automatically
hide_default_format = """
    <style>
    section[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

from utils.style_utils import apply_base_style

# Setup logging
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / 'app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

apply_base_style()

from utils.branding import APP_NAME, APP_LOGO

st.sidebar.image(APP_LOGO, use_container_width=True
)

# --- PAGE REGISTRY ---

# Detect if app is running on Render
ON_RENDER = os.environ.get("RENDER") is not None

if ON_RENDER:
    # Hide internal pages on the live site
    PAGES = {
        "Home": ("pages_disabled.home_page", "render_home_page"),
        "Dashboard": ("pages_disabled.dashboard_page", "render_dashboard_page"),
        "Charts & Stats": ("pages_disabled.charts_page", "render_charts_page"),
        "Matchday Hub": ("pages_disabled.matchday_hub_page", "render_matchday_hub_page"),
        "Season Review": ("pages_disabled.season_review_page", "render_season_review_page"),
        "Info": ("pages_disabled.info_page", "render_info_page"),
    }
    # --- MIGRATE OLD PAGE NAMES ---
    if st.session_state.get("page") == "Charts":
        st.session_state["page"] = "Charts & Stats"

else:
    # Show all pages locally for full control
    PAGES = {
        "Home": ("pages_disabled.home_page", "render_home_page"),
        "Dashboard": ("pages_disabled.dashboard_page", "render_dashboard_page"),
        "Charts & Stats": ("pages_disabled.charts_page", "render_charts_page"),
        "Matchday Hub": ("pages_disabled.matchday_hub_page", "render_matchday_hub_page"),
        "Season Review": ("pages_disabled.season_review_page", "render_season_review_page"),
        "Matches Management": ("pages_disabled.matches_page", "render_matches_page"),
        "Player Management": ("pages_disabled.player_management_page", "render_player_management_page"),
        "Info": ("pages_disabled.info_page", "render_info_page"),
    }
    # --- MIGRATE OLD PAGE NAMES ---
    if st.session_state.get("page") == "Charts":
        st.session_state["page"] = "Charts & Stats"


# --- MAIN ROUTER ---

# --- APPLY NAVIGATION OVERRIDES (from Home buttons etc.) ---
if "_nav_target" in st.session_state:
    target = st.session_state.pop("_nav_target")
    if target in PAGES:
        st.session_state["page"] = target
    else:
        st.session_state["page"] = "Home"

# 1) Initialise the session page (default to Home)
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# 2) Sidebar dropdown bound directly to session state
st.sidebar.selectbox(
    "",
    list(PAGES.keys()),
    key="page",
    label_visibility="collapsed",
)

# 3) Route to selected page
choice = st.session_state["page"]
module_path, func_name = PAGES[choice]
mod = __import__(module_path, fromlist=[func_name])
getattr(mod, func_name)()
