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
    # Keep your existing contract: authed if sb_session dict exists
    return bool(st.session_state.get("sb_session"))


def sb_client_authed():
    """
    Supabase client with PostgREST auth header set for the logged-in user.
    Useful for league_members / league_invites tables (and later RLS).
    """
    sb = get_supabase()
    sess = st.session_state.get("sb_session") or {}
    token = sess.get("access_token")
    if token:
        sb.postgrest.auth(token)
    return sb


def login_ui():
    sb = get_supabase()

    # Web-style mode toggle
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"  # "login" or "signup"

    mode = st.session_state["auth_mode"]

    st.subheader("🔐 Login" if mode == "login" else "🆕 Create account")

    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_pw")

    confirm = None
    if mode == "signup":
        confirm = st.text_input("Confirm password", type="password", key="signup_confirm")

    st.markdown("")

    submit_label = "Login" if mode == "login" else "Create account"
    submitted = st.button(submit_label, use_container_width=True, key="auth_submit")

    if mode == "login":
        st.caption("Don’t have an account?")
        if st.button("Create one", key="switch_to_signup"):
            st.session_state["auth_mode"] = "signup"
            st.rerun()
    else:
        st.caption("Already have an account?")
        if st.button("Login instead", key="switch_to_login"):
            st.session_state["auth_mode"] = "login"
            st.rerun()

    if not submitted:
        return

    # Validate only on submit (stops the “random backend error” UX)
    email_clean = (email or "").strip()
    if not email_clean:
        st.error("Please enter your email.")
        return
    if not password:
        st.error("Please enter your password.")
        return

    if mode == "signup":
        if not confirm:
            st.error("Please confirm your password.")
            return
        if password != confirm:
            st.error("Passwords don’t match.")
            return
        if len(password) < 6:
            st.error("Password must be at least 6 characters.")
            return

        try:
            sb.auth.sign_up({"email": email_clean, "password": password})
            st.success("Account created.")
            # Optional: flip back to login mode after signup
            st.session_state["auth_mode"] = "login"
        except Exception as e:
            st.error(f"Sign-up failed: {e}")
        return

    # Login
    try:
        res = sb.auth.sign_in_with_password({"email": email_clean, "password": password})

        # Keep your exact session dict format used elsewhere
        st.session_state.sb_session = {
            "access_token": res.session.access_token,  # type: ignore
            "refresh_token": res.session.refresh_token,  # type: ignore
            "user_id": res.user.id,  # type: ignore
            "email": res.user.email,  # type: ignore
        }

        st.success("Logged in.")
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")


def logout_ui():
    if st.sidebar.button("Logout", use_container_width=True, key="logout_btn"):
        st.session_state.pop("sb_session", None)
        st.session_state.pop("league_id", None)
        st.session_state.pop("league_name", None)
        st.session_state.pop("league_role", None)

        # Keep your current behaviour
        st.cache_data.clear()
        st.rerun()
