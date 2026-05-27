import streamlit as st
from utils.branding import APP_LOGO
import base64
from pathlib import Path

def render_home_page():
    # League context
    league_name = st.session_state.get("league_name") or "Your League"

    # ---------------------------------
    # Header: league context (top)
    # ---------------------------------
    st.markdown(
        f"""
        <style>
        .home-actions {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-top: 8px;
        }}
        .home-action {{
            display: block;
            text-align: center;
            padding: 13px 10px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.04);
            font-weight: 800;
            color: #e6eef6;
            min-height: 48px;
        }}
        @media (max-width: 760px){{
            .home-actions {{grid-template-columns: 1fr; gap: 8px;}}
            .home-league-title {{font-size: 1.45rem !important;}}
            .home-logo {{width: 210px !important; max-width: 78% !important;}}
        }}
        </style>
        <div style="text-align:center; margin-top:6px;">
            <div class="home-league-title" style="font-size:28px; font-weight:700; line-height:1.1;">
                {league_name}
            </div>
            <div style="font-size:13px; color:#9aa0a6; margin-top:4px;">
                Track your 5-a-side league like a pro
            </div>
        </div>
        <div style="height:14px;"></div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------
    # Smaller logo (optional, now secondary)
    # ---------------------------------
    logo_path = Path(APP_LOGO)
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    logo_ext = logo_path.suffix.lower().replace(".", "") or "png"

    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin-top:4px; margin-bottom:10px;">
            <img class="home-logo" src="data:image/{logo_ext};base64,{logo_b64}" style="width:280px; max-width:70%; opacity:0.95;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------
    # Primary actions
    # ---------------------------------
    role = (st.session_state.get("league_role") or "").lower()
    IS_ADMIN = role in ("admin", "owner")

    st.markdown(
        """
        <div style="text-align:center; font-size:12px; color:#9aa0a6; margin-bottom:8px;">
            Quick Navigation
        </div>
        """,
        unsafe_allow_html=True,
    )

    if IS_ADMIN and st.button("Add Result", use_container_width=True):
        st.session_state["_nav_target"] = "Matches Management"
        st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state["_nav_target"] = "Dashboard"
            st.rerun()
    with c2:
        if st.button("⚽ Matchday Hub", use_container_width=True):
            st.session_state["_nav_target"] = "Matchday Hub"
            st.rerun()
    with c3:
        if st.button("📈 Charts & Stats", use_container_width=True):
            st.session_state["_nav_target"] = "Charts & Stats"
            st.rerun()
    with c4:
        if st.button("🗂 Season Review", use_container_width=True):
            st.session_state["_nav_target"] = "Season Review"
            st.rerun()

    # ---------------------------------
    # Secondary: Admin tools (still here for now)
    # ---------------------------------
    c5, c6 = st.columns(2)
    with c5:
        if st.button("Join / Invite", use_container_width=True):
            st.session_state["_nav_target"] = "Join / Invite"
            st.rerun()
    with c6:
        if st.button("Profile Settings", use_container_width=True):
            st.session_state["_nav_target"] = "Profile Settings"
            st.rerun()

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='opacity:0.18;'>", unsafe_allow_html=True)

    if IS_ADMIN:
        st.markdown(
            """
            <div style="text-align:center; font-size:12px; color:#9aa0a6; margin-top:8px; margin-bottom:8px;">
                Admin
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("League Admin", use_container_width=True):
            st.session_state["_nav_target"] = "League Admin"
            st.rerun()
        if st.button("Player Management", use_container_width=True):
            st.session_state["_nav_target"] = "Player Management"
            st.rerun()
