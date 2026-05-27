# app.py
import logging
from pathlib import Path
import os
import streamlit as st
import locale
import importlib

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

    # ✅ IMPORTANT: rerun so we don't render the next page under the selector UI
    st.rerun()

# At this point we are logged in and have a league selected

# Warm the two core DB caches once per session/league.
# This makes the first real page load do the DB work, then page switching is faster.
try:
    _preload_key = f"preloaded_core_{st.session_state.get('league_id')}"
    if not st.session_state.get(_preload_key):
        from utils.db_utils import load_players_df, load_matches_df
        load_players_df()
        load_matches_df()
        st.session_state[_preload_key] = True
except Exception:
    # Never block the app shell just because a preload failed; the page itself
    # will show the real error if the DB is unavailable.
    pass

st.sidebar.success(f"League: {st.session_state.get('league_name', st.session_state['league_id'])}")
st.sidebar.markdown("---")

# Profile Settings (everyone)
if st.sidebar.button("👤 Profile Settings", use_container_width=True):
    st.session_state["_nav_target"] = "Profile Settings"
    st.rerun()

# Join / Invite (everyone)
if st.sidebar.button("🔗 Join / Invite", use_container_width=True):
    st.session_state["_nav_target"] = "Join / Invite"
    st.rerun()

# League Admin (admins only)
role = (st.session_state.get("league_role") or "").lower()
if role in ("admin", "owner"):
    if st.sidebar.button("⚙️ League Admin", use_container_width=True):
        st.session_state["_nav_target"] = "League Admin"
        st.rerun()

st.sidebar.markdown("---")


# -----------------------------
# PLAYER LINK GATE (Step B)
# -----------------------------
role = (st.session_state.get("league_role") or "").lower()

# Only import/run this gate for non-admins. It may query profile/player-link data,
# so avoid doing that work for admin/owner page switching.
if role not in ("admin", "owner"):
    from utils.player_link_utils import ensure_player_linked_ui
    ensure_player_linked_ui()



# -----------------------------
# PAGE REGISTRY
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
        "Info": ("pages_disabled.info_page", "render_info_page"),
    }
    if st.session_state.get("page") == "Charts":
        st.session_state["page"] = "Charts & Stats"

# Hidden pages: available via buttons/_nav_target, but not shown in dropdown
HIDDEN_PAGES = {
    "League Admin": ("pages_disabled.league_admin_page", "render_league_admin_page"),
    "Profile Settings": ("pages_disabled.profile_settings_page", "render_profile_settings_page"),
    "Join / Invite": ("pages_disabled.join_invite_page", "render_join_invite_page"),
    "Matches Management": ("pages_disabled.matches_page", "render_matches_page"),
    "Player Management": ("pages_disabled.player_management_page", "render_player_management_page"),}

# -----------------------------
# NAV OVERRIDES
# -----------------------------
if "_nav_target" in st.session_state:
    target = st.session_state.pop("_nav_target")
    if target in PAGES or target in HIDDEN_PAGES:
        st.session_state["page"] = target
    else:
        st.session_state["page"] = "Home"

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Only show page dropdown AFTER auth+league selection
page_options = list(PAGES.keys())

# ✅ If we are currently on a hidden page, include it so Streamlit doesn't bounce us back
current_page = st.session_state.get("page")
if current_page in HIDDEN_PAGES and current_page not in page_options:
    page_options = [current_page] + page_options

st.sidebar.selectbox(
    "",
    page_options,
    key="page",
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

# Manual cache refresh (useful after edits or if Render/Supabase feels stale)
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    try:
        from utils.cache_utils import invalidate_app_caches
        invalidate_app_caches()
    except Exception:
        st.cache_data.clear()
    st.sidebar.success("Data refreshed")
    st.rerun()

st.sidebar.markdown("---")

# Logout (bottom-ish)
logout_ui()
st.sidebar.markdown("---")


# -----------------------------
# ROUTER
# -----------------------------
choice = st.session_state["page"]
registry = PAGES if choice in PAGES else HIDDEN_PAGES
module_path, func_name = registry[choice]
mod = importlib.import_module(module_path)
getattr(mod, func_name)()
