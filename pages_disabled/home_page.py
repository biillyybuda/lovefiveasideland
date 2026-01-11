import streamlit as st
from utils.branding import APP_NAME, APP_LOGO

import base64
from pathlib import Path



def render_home_page():
    # ----------------------------
    # Hero (logo + tagline)
    # ----------------------------
    logo_path = Path(APP_LOGO)
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    logo_ext = logo_path.suffix.lower().replace(".", "") or "png"

    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; margin-top:10px;">
            <img src="data:image/{logo_ext};base64,{logo_b64}" style="width:420px; max-width:80%; margin-bottom:8px;" />
            <div style="font-size:14px; color:#9aa0a6; text-align:center; margin-top:-4px;">
                Track your 5-a-side league like a pro
            </div>
        </div>
        <div style="height:18px;"></div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------
    # Main actions (primary nav)
    # ----------------------------
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
        if st.button("📈 Charts", use_container_width=True):
            st.session_state["_nav_target"] = "Charts & Stats"
            st.rerun()

    with c4:
        if st.button("🗂 Season Review", use_container_width=True):
            st.session_state["_nav_target"] = "Season Review"
            st.rerun()

    # Admin row (kept, but visually secondary)
    IS_ADMIN = True  # Phase 1 placeholder
    if IS_ADMIN:
        st.markdown(
            """
            <div style="text-align:center; font-size:13px; color:#9aa0a6; margin-top:14px; margin-bottom:6px;">
                Admin Tools
            </div>
            """,
            unsafe_allow_html=True,
        )
        a1, a2 = st.columns(2)
        with a1:
            if st.button("🧾 Matches Management", use_container_width=True):
                st.session_state["_nav_target"] = "Matches Management"
                st.rerun()
        with a2:
            if st.button("👤 Player Management", use_container_width=True):
                st.session_state["_nav_target"] = "Player Management"
                st.rerun()


