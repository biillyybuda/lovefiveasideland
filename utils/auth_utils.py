# utils/auth_utils.py
# Persistent Supabase login for Streamlit using client-side cookies.
# Uses streamlit-cookies-controller (more reliable on cloud than streamlit-cookies-manager).
# Ref: https://discuss.streamlit.io/t/new-component-streamlit-cookies-controller/64251

import os
import json
import base64
import hmac
import hashlib
from typing import Optional, Dict, Any

import streamlit as st
from utils.cache_utils import invalidate_app_caches
from supabase import create_client

from streamlit_cookies_controller import CookieController, RemoveEmptyElementContainer


COOKIE_KEY = "lovefive_sb_session"
COOKIE_MAX_AGE = 30 * 24 * 60 * 60  # 30 days


# -----------------------------
# Supabase client
# -----------------------------
@st.cache_resource
def get_supabase():
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY env vars.")
    return create_client(url, key)


# -----------------------------
# Cookie controller (per-session)
# -----------------------------
def _get_cookie_controller() -> CookieController:
    # Components should not be created in st.cache_*, keep it in session_state.
    if "_lf_cookie_controller" not in st.session_state:
        # Prevent empty iframe flicker (recommended by the component author)
        RemoveEmptyElementContainer()
        st.session_state["_lf_cookie_controller"] = CookieController(key="lovefive_cookies")
    return st.session_state["_lf_cookie_controller"]

def _cookie_set(controller: CookieController, key: str, value: str) -> None:
    """
    CookieController versions differ on supported kwargs.
    Prefer max_age; if unsupported, fall back to plain set (session cookie).
    """
    try:
        controller.set(key, value, max_age=COOKIE_MAX_AGE)
    except TypeError:
        controller.set(key, value)
    except Exception:
        controller.set(key, value)



# -----------------------------
# Signed (tamper-evident) payload helpers
# -----------------------------
def _cookie_secret() -> bytes:
    return (os.getenv("COOKIE_SECRET") or "dev-secret-change-me").encode("utf-8")


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def _sign(data: bytes) -> str:
    sig = hmac.new(_cookie_secret(), data, hashlib.sha256).digest()
    return _b64url_encode(sig)


def _pack(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    body = _b64url_encode(raw)
    sig = _sign(raw)
    return f"{body}.{sig}"


def _unpack(token: str) -> Optional[Dict[str, Any]]:
    try:
        body, sig = token.split(".", 1)
        raw = _b64url_decode(body)
        expected = _sign(raw)
        if not hmac.compare_digest(sig, expected):
            return None
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


# -----------------------------
# Session restore
# -----------------------------
def _restore_session_from_cookie() -> bool:
    """If session_state has no sb_session, try to restore from cookie."""
    if st.session_state.get("sb_session"):
        return True

    controller = _get_cookie_controller()
    controller.getAll()


    # Force the cookie component to render (no rerun loop).
    # On some hosts the first get() can be None; try twice within same run.
    token = controller.get(COOKIE_KEY)
    if not token:
        # Touch the component once to sync and try again
        try:
            controller.getAll()
        except Exception:
            pass
        token = controller.get(COOKIE_KEY)

    if not token:
        return False

    payload = _unpack(str(token))
    if not payload:
        # Invalid cookie -> clear and force login
        try:
            controller.remove(COOKIE_KEY)
        except Exception:
            pass
        return False

    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        return False

    sb = get_supabase()

    try:
        res = sb.auth.refresh_session(refresh_token=str(refresh_token))

        sess = {
            "access_token": res.session.access_token,  # type: ignore
            "refresh_token": res.session.refresh_token,  # type: ignore
            "user_id": res.user.id,  # type: ignore
            "email": res.user.email,  # type: ignore
        }
        st.session_state["sb_session"] = sess

        # Persist rotated refresh token
        _cookie_set(
            controller,
            COOKIE_KEY,
            _pack({"refresh_token": sess["refresh_token"], "email": sess.get("email")}),
        )
        return True

    except Exception:
        try:
            controller.remove(COOKIE_KEY)
        except Exception:
            pass
        return False


# -----------------------------
# Public helpers used by app.py
# -----------------------------
def is_authed() -> bool:
    return _restore_session_from_cookie()


def sb_client_authed():
    """Supabase client with PostgREST auth header set for the logged-in user."""
    sb = get_supabase()
    _restore_session_from_cookie()

    sess = st.session_state.get("sb_session") or {}
    token = sess.get("access_token")
    if token:
        sb.postgrest.auth(token)
    return sb


def login_ui():
    sb = get_supabase()
    controller = _get_cookie_controller()
    controller.getAll()

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
            st.session_state["auth_mode"] = "login"
        except Exception as e:
            st.error(f"Sign-up failed: {e}")
        return

    # Login
    try:
        res = sb.auth.sign_in_with_password({"email": email_clean, "password": password})

        sess = {
            "access_token": res.session.access_token,  # type: ignore
            "refresh_token": res.session.refresh_token,  # type: ignore
            "user_id": res.user.id,  # type: ignore
            "email": res.user.email,  # type: ignore
        }

        st.session_state["sb_session"] = sess

        # Persist refresh_token only (access tokens expire)
        _cookie_set(
            controller,
            COOKIE_KEY,
            _pack({"refresh_token": sess["refresh_token"], "email": sess.get("email")}),
        )

        st.success("Logged in.")
        st.rerun()

    except Exception as e:
        st.error(f"Login failed: {e}")


def logout_ui():
    controller = _get_cookie_controller()

    if st.sidebar.button("Logout", use_container_width=True, key="logout_btn"):
        st.session_state.pop("sb_session", None)
        st.session_state.pop("league_id", None)
        st.session_state.pop("league_name", None)
        st.session_state.pop("league_role", None)

        # Clear cookie
        try:
            controller.remove(COOKIE_KEY)
        except Exception:
            pass

        invalidate_app_caches()
        st.rerun()
