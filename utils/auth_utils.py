# utils/auth_utils.py
import os
import streamlit as st
from supabase import create_client


def get_supabase():
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY env vars.")
    return create_client(url, key)


def is_authed() -> bool:
    return bool(st.session_state.get("sb_session"))


def sb_client_authed():
    """
    Supabase client with PostgREST auth header set for the logged-in user.
    Useful for league_members / league_invites tables (and later RLS).
    """
    sb = get_supabase()
    sess = st.session_state.get("sb_session")
    if sess and sess.get("access_token"):
        sb.postgrest.auth(sess["access_token"])
    return sb


def login_ui():
    st.subheader("🔐 Login")

    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_pw")

    c1, c2 = st.columns(2)
    sb = get_supabase()

    with c1:
        if st.button("Login", use_container_width=True):
            try:
                res = sb.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.sb_session = {
                    "access_token": res.session.access_token,
                    "refresh_token": res.session.refresh_token,
                    "user_id": res.user.id,
                    "email": res.user.email,
                }
                st.success("Logged in.")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    with c2:
        if st.button("Create account", use_container_width=True):
            try:
                sb.auth.sign_up({"email": email, "password": password})
                st.info("Account created. If email confirmation is enabled, check your inbox.")
            except Exception as e:
                st.error(f"Sign-up failed: {e}")


def logout_ui():
    if st.button("Logout", use_container_width=True):
        st.session_state.pop("sb_session", None)
        st.session_state.pop("league_id", None)
        st.session_state.pop("league_name", None)
        st.session_state.pop("league_role", None)
        st.cache_data.clear()
        st.rerun()
