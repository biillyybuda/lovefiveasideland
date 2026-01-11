# app.py
import logging
from pathlib import Path
import os
import streamlit as st
import locale

def set_time_locale():
    for loc in ("en_GB.UTF-8", "en_GB.utf8", "en_GB", "C.UTF-8", "C"):
        try:
            locale.setlocale(locale.LC_TIME, loc)
            return loc
        except locale.Error:
            continue
    return None

_active_locale = set_time_locale()

from utils.branding import APP_NAME, APP_TAGLINE, APP_ICON
from utils.style_utils import apply_base_style

# Auth / League helpers
from utils.auth_utils import is_authed, login_ui, logout_ui
from utils.league_utils import league_selector_ui, accept_invite_flow

# -----------------------------
# Page config / style
# -----------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_default_format = """
    <style>
    section[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

# Setup logging
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / 'app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

apply_base_style()

from utils.branding import APP_LOGO
st.sidebar.image(APP_LOGO, use_container_width=True)

# -----------------------------
# AUTH GATE (runs before pages)
# -----------------------------
# If user is not logged in, show login and stop.
if not is_authed():
    st.title("Love Five")
    st.caption(APP_TAGLINE)
    login_ui()
    st.stop()

# Logged in → show logout in sidebar
st.sidebar.markdown("---")
logout_ui()
st.sidebar.markdown("---")

# -----------------------------
# INVITE FLOW (optional)
# -----------------------------
params = st.query_params
invite_token = params.get("invite")
if invite_token:
    # If they opened an invite link, handle it here before league selection
    accept_invite_flow(invite_token)
    st.stop()

# -----------------------------
# LEAGUE SELECTION GATE
# -----------------------------
if not st.session_state.get("league_id"):
    st.title("Love Five")
    st.caption("Select the league you want to view/manage.")
    ok = league_selector_ui()
    if not ok:
        st.stop()
    # if ok + auto-selected, continue

# At this point we are logged in and have a league selected
st.sidebar.success(f"League: {st.session_state.get('league_name', st.session_state['league_id'])}")

# -----------------------------
# PAGE REGISTRY (unchanged)
# -----------------------------
ON_RENDER = os.environ.get("RENDER") is not None

if ON_RENDER:
    PAGES = {
        "Home": ("pages_disabled.home_page", "render_home_page"),
        "Dashboard": ("pages_disabled.dashboard_page", "render_dashboard_page"),
        "Charts & Stats": ("pages_disabled.charts_page", "render_charts_page"),
        "Matchday Hub": ("pages_disabled.matchday_hub_page", "render_matchday_hub_page"),
        "Season Review": ("pages_disabled.season_review_page", "render_season_review_page"),
        "Info": ("pages_disabled.info_page", "render_info_page"),
    }
    if st.session_state.get("page") == "Charts":
        st.session_state["page"] = "Charts & Stats"
else:
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
    if st.session_state.get("page") == "Charts":
        st.session_state["page"] = "Charts & Stats"

# -----------------------------
# NAV OVERRIDES (unchanged)
# -----------------------------
if "_nav_target" in st.session_state:
    target = st.session_state.pop("_nav_target")
    st.session_state["page"] = target if target in PAGES else "Home"

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Only show page dropdown AFTER auth+league selection
st.sidebar.selectbox(
    "",
    list(PAGES.keys()),
    key="page",
    label_visibility="collapsed",
)

# -----------------------------
# ROUTER (unchanged)
# -----------------------------
choice = st.session_state["page"]
module_path, func_name = PAGES[choice]
mod = __import__(module_path, fromlist=[func_name])
getattr(mod, func_name)()
