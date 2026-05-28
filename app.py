# app.py
import logging
from pathlib import Path
import os
import streamlit as st
import locale
import importlib
from utils.perf_utils import render_perf_sidebar, reset_perf_trace, timed

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
from utils.league_utils import (
    accept_invite_flow,
    change_league_sidebar_ui,
    enter_demo_league_viewer,
    is_demo_league_selected,
    league_selector_ui,
)

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
reset_perf_trace()

from utils.branding import APP_LOGO
st.sidebar.image(APP_LOGO, use_container_width=True)

params = st.query_params
demo_requested = str(params.get("demo") or "").strip().lower() in ("1", "true", "yes")
if demo_requested and not is_demo_league_selected():
    enter_demo_league_viewer()

# -----------------------------
# AUTH GATE (runs before pages)
# -----------------------------
# If user is not logged in, show login and stop.
if not is_demo_league_selected() and not is_authed():
    st.title("Love Five")
    st.caption(APP_TAGLINE)
    login_ui()
    st.stop()

# -----------------------------
# INVITE FLOW (optional)
# -----------------------------
invite_token = params.get("invite")
if invite_token and not is_demo_league_selected():
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

# -----------------------------
# PLAYER LINK GATE (Step B)
# -----------------------------
role = (st.session_state.get("league_role") or "").lower()

# Only import/run this gate for non-admins. It may query profile/player-link data,
# so avoid doing that work for admin/owner page switching.
if role not in ("admin", "owner") and not is_demo_league_selected():
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

PAGE_SLUGS = {
    "Home": "home",
    "Dashboard": "dashboard",
    "Charts & Stats": "charts",
    "Matchday Hub": "matchday",
    "Season Review": "season-review",
    "Info": "info",
    "League Admin": "league-admin",
    "Profile Settings": "profile",
    "Join / Invite": "join",
    "Matches Management": "add-result",
    "Player Management": "players",
}
SLUG_PAGES = {slug: page for page, slug in PAGE_SLUGS.items()}


def _page_from_url():
    slug = st.query_params.get("page")
    if isinstance(slug, list):
        slug = slug[0] if slug else None
    return SLUG_PAGES.get(str(slug or "").strip().lower())


def _sync_page_url(page_name: str) -> None:
    slug = PAGE_SLUGS.get(page_name)
    if slug:
        st.query_params["page"] = slug


def _go_to_page(page_name: str) -> None:
    st.session_state["page"] = page_name
    _sync_page_url(page_name)


# -----------------------------
# NAV OVERRIDES
# -----------------------------
if "_nav_target" in st.session_state:
    target = st.session_state.pop("_nav_target")
    if target in PAGES or target in HIDDEN_PAGES:
        _go_to_page(target)
    else:
        _go_to_page("Home")

if "page" not in st.session_state:
    st.session_state["page"] = _page_from_url() or "Home"

if st.session_state["page"] not in PAGES and st.session_state["page"] not in HIDDEN_PAGES:
    st.session_state["page"] = "Home"

if (
    is_demo_league_selected()
    and not st.session_state.get("sb_session")
    and st.session_state["page"] in HIDDEN_PAGES
):
    st.session_state["page"] = "Home"

# Sidebar fallback list. Main navigation is handled by grouped buttons below.
page_options = list(PAGES.keys())

# ✅ If we are currently on a hidden page, include it so Streamlit doesn't bounce us back
current_page = st.session_state.get("page")
if current_page in HIDDEN_PAGES and current_page not in page_options:
    page_options = [current_page] + page_options

league_name = st.session_state.get("league_name", st.session_state["league_id"])
st.sidebar.markdown(f"**League**  \n{league_name}")
if is_demo_league_selected():
    st.sidebar.caption("Demo view only")
change_league_sidebar_ui()
st.sidebar.markdown("---")

def _sidebar_nav_button(label: str, target: str, key: str):
    active = current_page == target
    shown = f"> {label}" if active else label
    if st.sidebar.button(shown, use_container_width=True, key=key, disabled=active):
        _go_to_page(target)
        st.rerun()


st.sidebar.caption("Main")
_sidebar_nav_button("Home", "Home", "nav_home")
_sidebar_nav_button("Matchday Hub", "Matchday Hub", "nav_matchday")
_sidebar_nav_button("Dashboard", "Dashboard", "nav_dashboard")
_sidebar_nav_button("Charts & Stats", "Charts & Stats", "nav_charts")
_sidebar_nav_button("Season Review", "Season Review", "nav_season")

st.sidebar.markdown("---")
st.sidebar.caption("League")
if not (is_demo_league_selected() and not st.session_state.get("sb_session")):
    _sidebar_nav_button("Join / Invite", "Join / Invite", "nav_join")
    _sidebar_nav_button("Profile Settings", "Profile Settings", "nav_profile")
_sidebar_nav_button("Info", "Info", "nav_info")

role = (st.session_state.get("league_role") or "").lower()
if role in ("admin", "owner"):
    st.sidebar.markdown("---")
    st.sidebar.caption("Admin")
    _sidebar_nav_button("Add Result", "Matches Management", "nav_add_result")
    _sidebar_nav_button("League Admin", "League Admin", "nav_league_admin")
    _sidebar_nav_button("Player Management", "Player Management", "nav_players")

with st.sidebar.expander("All pages", expanded=False):
    st.selectbox(
        "",
        page_options,
        key="page",
        label_visibility="collapsed",
        on_change=lambda: _sync_page_url(st.session_state["page"]),
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
_sync_page_url(choice)
registry = PAGES if choice in PAGES else HIDDEN_PAGES
module_path, func_name = registry[choice]
mod = importlib.import_module(module_path)
with timed(f"Render {choice}", log_over=1.5):
    getattr(mod, func_name)()
render_perf_sidebar()
