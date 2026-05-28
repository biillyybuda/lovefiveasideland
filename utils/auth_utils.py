# utils/auth_utils.py
# Persistent Supabase login for Streamlit using client-side cookies.
# Uses streamlit-cookies-controller (more reliable on cloud than streamlit-cookies-manager).
# Ref: https://discuss.streamlit.io/t/new-component-streamlit-cookies-controller/64251

import os
import json
import base64
import hmac
import hashlib
import time
from typing import Optional, Dict, Any

import streamlit as st
from utils.cache_utils import invalidate_app_caches
from supabase import create_client

from streamlit_cookies_controller import CookieController, RemoveEmptyElementContainer


COOKIE_KEY = "lovefive_sb_session"
LEAGUE_COOKIE_KEY = "lovefive_selected_league"
COOKIE_MAX_AGE = 30 * 24 * 60 * 60  # 30 days
AUTH_REFRESH_GRACE_SECONDS = 50 * 60  # only refresh Supabase session about once per hour
AUTH_COOKIE_PROBE_SECONDS = 2.5



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
        controller.set(key, value, max_age=COOKIE_MAX_AGE, path="/", same_site="lax")
    except TypeError:
        controller.set(key, value)
    except Exception:
        controller.set(key, value)


def _cookie_remove(controller: CookieController, key: str) -> None:
    """Clear a cookie in both the browser and the component's server cache."""
    for same_site in ("lax", "strict", None):
        try:
            controller.remove(key, path="/", same_site=same_site)
        except Exception:
            pass

    try:
        controller.set(key, "", max_age=0, path="/", same_site="lax")
    except TypeError:
        try:
            controller.set(key, "")
        except Exception:
            pass
    except Exception:
        pass


def _clear_auth_state() -> None:
    for key in (
        "sb_session",
        "league_id",
        "league_name",
        "league_role",
        "_lf_auth_checked",
        "_lf_auth_last_refresh",
        "_lf_cookie_probe_started",
        "_lf_league_cookie_checked",
        "_lf_skip_league_restore_once",
        "_lf_pending_logout",
    ):
        st.session_state.pop(key, None)



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


def _auth_cookie_payload(sess: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "refresh_token": sess.get("refresh_token"),
        "email": sess.get("email"),
    }


# -----------------------------
# Session restore
# -----------------------------
def _restore_session_from_cookie(force: bool = False) -> bool:
    """Restore auth from cookie only when needed.

    Performance note:
    Streamlit reruns the whole app on every click. Refreshing Supabase auth and
    syncing the cookie component on every rerun makes the entire app feel slow.
    This function now short-circuits when a valid session is already in
    session_state, and only refreshes from the cookie periodically or when forced.
    """
    now = time.time()

    if st.session_state.get("_lf_pending_logout"):
        return False

    sess = st.session_state.get("sb_session")
    last_refresh = float(st.session_state.get("_lf_auth_last_refresh", 0) or 0)

    # Fast path: already logged in during this Streamlit session.
    if sess and not force and (now - last_refresh) < AUTH_REFRESH_GRACE_SECONDS:
        return True

    # If we have already checked this run/session and found no cookie/session,
    # do not keep re-rendering the cookie iframe on every rerun.
    if not force and st.session_state.get("_lf_auth_checked") and not sess:
        return False

    controller = _get_cookie_controller()

    # Cookie component sync can be slow; do the minimum possible.
    try:
        controller.refresh()
        token = controller.get(COOKIE_KEY)
    except Exception:
        token = None

    if not token:
        try:
            controller.getAll()
            token = controller.get(COOKIE_KEY)
        except Exception:
            token = None

    if not token:
        # On a hard browser refresh the cookie iframe can need one short rerun
        # before its client-side cookie cache is available. Without this grace
        # pass the app may show the login form even though the refresh cookie is
        # still valid in the browser.
        first_probe = float(st.session_state.get("_lf_cookie_probe_started", 0) or 0)
        if not force and (not first_probe or (now - first_probe) < AUTH_COOKIE_PROBE_SECONDS):
            if not first_probe:
                st.session_state["_lf_cookie_probe_started"] = now
            st.info("Checking your saved login...")
            time.sleep(0.2)
            st.stop()
        st.session_state.pop("_lf_cookie_probe_started", None)
        st.session_state["_lf_auth_checked"] = True
        return False

    payload = _unpack(str(token))
    if not payload:
        _cookie_remove(controller, COOKIE_KEY)
        st.session_state["_lf_auth_checked"] = True
        return False

    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        st.session_state["_lf_auth_checked"] = True
        return False

    sb = get_supabase()

    try:
        res = sb.auth.refresh_session(refresh_token=str(refresh_token))

        sess = {
            "access_token": res.session.access_token,  # type: ignore
            "refresh_token": res.session.refresh_token,  # type: ignore
            "user_id": res.user.id,  # type: ignore
            "email": res.user.email,  # type: ignore
            "app_metadata": getattr(res.user, "app_metadata", {}) or {},  # type: ignore
        }
        st.session_state["sb_session"] = sess
        st.session_state.pop("_lf_pending_logout", None)
        st.session_state["_lf_auth_checked"] = True
        st.session_state["_lf_auth_last_refresh"] = now
        st.session_state.pop("_lf_cookie_probe_started", None)

        # Persist rotated refresh token, but do not force extra cookie reads.
        _cookie_set(
            controller,
            COOKIE_KEY,
            _pack(_auth_cookie_payload(sess)),
        )
        return True

    except Exception:
        _cookie_remove(controller, COOKIE_KEY)
        st.session_state.pop("sb_session", None)
        st.session_state.pop("_lf_cookie_probe_started", None)
        st.session_state["_lf_auth_checked"] = True
        return False


# -----------------------------
# Public helpers used by app.py
# -----------------------------
def is_authed() -> bool:
    # Hot path for normal page switching/clicking.
    if st.session_state.get("_lf_pending_logout"):
        return False
    if st.session_state.get("sb_session"):
        return True
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


def save_selected_league(league_id: int, league_name: str, league_role: Optional[str]) -> None:
    sess = st.session_state.get("sb_session") or {}
    user_id = sess.get("user_id")
    if not user_id:
        return

    last_league = {
        "user_id": user_id,
        "league_id": int(league_id),
        "league_name": league_name,
        "league_role": league_role,
    }
    controller = _get_cookie_controller()
    st.session_state.pop("_lf_skip_league_restore_once", None)
    st.session_state.pop("_lf_league_cookie_checked", None)
    _cookie_set(
        controller,
        LEAGUE_COOKIE_KEY,
        _pack(last_league),
    )


def forget_selected_league() -> None:
    controller = _get_cookie_controller()
    _cookie_remove(controller, LEAGUE_COOKIE_KEY)
    st.session_state["_lf_skip_league_restore_once"] = True
    st.session_state["_lf_league_cookie_checked"] = True


def is_superuser() -> bool:
    sess = st.session_state.get("sb_session") or {}
    metadata = sess.get("app_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    role = str(metadata.get("role") or metadata.get("app_role") or "").lower()
    roles = metadata.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    roles = [str(r).lower() for r in roles if r is not None]

    return bool(
        metadata.get("superuser")
        or metadata.get("is_superuser")
        or metadata.get("is_admin")
        or role in ("superuser", "admin")
        or "superuser" in roles
    )


def restore_selected_league(leagues: list[dict]) -> bool:
    if st.session_state.get("league_id"):
        return True

    if st.session_state.pop("_lf_skip_league_restore_once", None):
        return False

    sess = st.session_state.get("sb_session") or {}
    user_id = sess.get("user_id")
    if not user_id:
        return False

    controller = _get_cookie_controller()
    try:
        controller.refresh()
        token = controller.get(LEAGUE_COOKIE_KEY)
    except Exception:
        token = None

    if not token:
        try:
            controller.getAll()
            token = controller.get(LEAGUE_COOKIE_KEY)
        except Exception:
            token = None

    if not token:
        # On a hard refresh Streamlit can rerun before the cookie component has
        # copied browser cookies back into Python. Wait once for league restore;
        # if there is no cookie, the normal selector appears on the next run.
        if not st.session_state.get("_lf_league_cookie_checked"):
            st.session_state["_lf_league_cookie_checked"] = True
            st.info("Restoring your league...")
            st.stop()
        return False

    payload = _unpack(str(token)) if token else None

    if not payload or payload.get("user_id") != user_id:
        return False

    league_id = payload.get("league_id")
    if league_id is None:
        _cookie_remove(controller, LEAGUE_COOKIE_KEY)
        return False

    selected = next((l for l in leagues if int(l.get("id")) == int(league_id)), None)
    if not selected:
        _cookie_remove(controller, LEAGUE_COOKIE_KEY)
        return False

    st.session_state.league_id = int(selected["id"])  # type: ignore
    st.session_state.league_name = selected["name"]  # type: ignore
    st.session_state.league_role = selected.get("role")  # type: ignore
    st.session_state.pop("_lf_league_cookie_checked", None)
    return True


def _legacy_login_ui():
    sb = get_supabase()
    controller = _get_cookie_controller()

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
        st.caption("Don't have an account?")
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
            "app_metadata": getattr(res.user, "app_metadata", {}) or {},  # type: ignore
        }

        st.session_state["sb_session"] = sess
        st.session_state.pop("_lf_pending_logout", None)

        # Persist refresh_token only (access tokens expire)
        _cookie_set(
            controller,
            COOKIE_KEY,
            _pack(_auth_cookie_payload(sess)),
        )

        st.success("Logged in.")
        st.rerun()

    except Exception as e:
        st.error(f"Login failed: {e}")


def login_ui():
    """Render a compact auth card and persist a refresh-token cookie on login."""
    sb = get_supabase()
    controller = _get_cookie_controller()

    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    mode = st.session_state["auth_mode"]
    st.markdown(
        """
        <style>
        .lf-auth-wrap {
            max-width: 460px;
            margin: 18px auto 0 auto;
            padding: 22px 22px 18px 22px;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            background: rgba(255,255,255,0.035);
            box-shadow: 0 18px 55px rgba(0,0,0,0.22);
        }
        .lf-auth-title {
            font-size: 1.35rem;
            font-weight: 900;
            margin-bottom: 4px;
            text-align: center;
        }
        .lf-auth-sub {
            color: #aab3bd;
            text-align: center;
            margin-bottom: 18px;
            font-size: 0.95rem;
        }
        .lf-auth-note {
            color: #9aa2aa;
            text-align: center;
            font-size: 0.85rem;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title = "Sign in to Love Five" if mode == "login" else "Create your Love Five account"
    subtitle = "Your login is kept for 30 days on this device." if mode == "login" else "Use the same email you want linked to your league."

    _left, mid, _right = st.columns([1, 1.15, 1])
    with mid:
        st.markdown(
            f"""
            <div class="lf-auth-wrap">
                <div class="lf-auth-title">{title}</div>
                <div class="lf-auth-sub">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("auth_form", clear_on_submit=False):
            email = st.text_input("Email", key="login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="login_pw", placeholder="Your password")
            confirm = None
            if mode == "signup":
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm", placeholder="Repeat password")

            submit_label = "Sign in" if mode == "login" else "Create account"
            submitted = st.form_submit_button(submit_label, use_container_width=True)

        if mode == "login":
            st.markdown("<div class='lf-auth-note'>No account yet?</div>", unsafe_allow_html=True)
            if st.button("Create account", key="switch_to_signup", use_container_width=True):
                st.session_state["auth_mode"] = "signup"
                st.rerun()
        else:
            st.markdown("<div class='lf-auth-note'>Already have an account?</div>", unsafe_allow_html=True)
            if st.button("Sign in instead", key="switch_to_login", use_container_width=True):
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
            st.error("Passwords do not match.")
            return
        if len(password) < 6:
            st.error("Password must be at least 6 characters.")
            return

        try:
            sb.auth.sign_up({"email": email_clean, "password": password})
            st.success("Account created. You can sign in now.")
            st.session_state["auth_mode"] = "login"
            st.rerun()
        except Exception as e:
            st.error(f"Sign-up failed: {e}")
        return

    try:
        res = sb.auth.sign_in_with_password({"email": email_clean, "password": password})
        sess = {
            "access_token": res.session.access_token,  # type: ignore
            "refresh_token": res.session.refresh_token,  # type: ignore
            "user_id": res.user.id,  # type: ignore
            "email": res.user.email,  # type: ignore
            "app_metadata": getattr(res.user, "app_metadata", {}) or {},  # type: ignore
        }
        st.session_state["sb_session"] = sess
        st.session_state.pop("_lf_pending_logout", None)
        st.session_state["_lf_auth_checked"] = True
        st.session_state["_lf_auth_last_refresh"] = time.time()
        st.session_state.pop("_lf_cookie_probe_started", None)

        _cookie_set(
            controller,
            COOKIE_KEY,
            _pack(_auth_cookie_payload(sess)),
        )

        st.success("Signed in.")
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")


def logout_ui():
    controller = _get_cookie_controller()

    if st.sidebar.button("Logout", use_container_width=True, key="logout_btn"):
        sess = st.session_state.get("sb_session") or {}
        try:
            if sess.get("access_token") and sess.get("refresh_token"):
                sb = get_supabase()
                sb.auth.set_session(sess["access_token"], sess["refresh_token"])
                sb.auth.sign_out()
        except Exception:
            pass

        _cookie_remove(controller, COOKIE_KEY)
        _cookie_remove(controller, LEAGUE_COOKIE_KEY)
        _clear_auth_state()
        st.session_state["_lf_pending_logout"] = True

        invalidate_app_caches()
        st.rerun()
