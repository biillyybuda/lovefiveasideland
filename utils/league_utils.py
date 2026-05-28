# utils/league_utils.py
import streamlit as st
from utils.cache_utils import invalidate_app_caches
from utils.auth_utils import (
    forget_selected_league,
    is_superuser,
    restore_selected_league,
    save_selected_league,
    sb_client_authed,
)
from utils.db_utils import get_conn

def update_my_display_name(new_display_name: str):
    sb = st.session_state.get("sb_session") or {}

    # Support both session shapes:
    # 1) sb_session["user_id"]  (your current app)
    # 2) sb_session["user"]["id"] (common supabase client shape)
    user_id = sb.get("user_id") or (sb.get("user") or {}).get("id")

    if not user_id:
        raise RuntimeError("Not logged in.")

    clean = (new_display_name or "").strip()
    if not clean:
        raise ValueError("Display name cannot be empty.")

    conn = get_conn()
    cur = conn.cursor()

    # Update profile display name
    cur.execute(
        "update public.profiles set display_name = %s where id = %s",
        (clean, user_id),
    )

    # Update linked player record for CURRENT league (safer than global)
    league_id = st.session_state.get("league_id")
    if league_id is not None:
        cur.execute(
            """
            update public.players
            set display_name = %s
            where user_id = %s and league_id = %s
            """,
            (clean, user_id, int(league_id)),
        )
    else:
        # Fallback: if no league selected, still try global update
        cur.execute(
            "update public.players set display_name = %s where user_id = %s",
            (clean, user_id),
        )

    conn.commit()
    conn.close()


def get_league_join_code(league_id: int) -> str:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "select join_code from public.leagues where id = %s",
        (league_id,)
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else ""


def join_league_by_code(code: str):
    code = (code or "").strip().upper()

    sb = st.session_state.get("sb_session") or {}
    user_id = sb.get("user_id")

    if not user_id:
        return None

    conn = get_conn()
    cur = conn.cursor()

    # Find league
    cur.execute(
        "select id, name from public.leagues where join_code = %s",
        (code,)
    )
    league = cur.fetchone()
    if not league:
        conn.close()
        return None

    league_id, league_name = league

    # Insert membership
    cur.execute(
        """
        insert into public.league_members (league_id, user_id, role, status)
        values (%s, %s, %s, %s)
        on conflict (league_id, user_id) do nothing
        """,
        (league_id, user_id, "member", "active"),
    )

    conn.commit()
    conn.close()

    return {
        "league_id": league_id,
        "league_name": league_name,
        "role": "member",
    }


def update_league_name(league_id: str, new_name: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE leagues SET name = %s WHERE id = %s", (new_name, league_id))
    conn.commit()
    conn.close()


def load_my_leagues():
    sb = sb_client_authed()

    mem = (
        sb.table("league_members")
        .select("league_id,role,status")
        .eq("status", "active")
        .execute()
    )
    mem_rows = mem.data or []
    league_ids = [r["league_id"] for r in mem_rows] # type: ignore

    if not league_ids:
        st.session_state["_lf_my_leagues"] = []
        return []

    leagues = (
        sb.table("leagues")
        .select("id,name")
        .in_("id", league_ids)
        .execute()
    )
    league_rows = leagues.data or []

    role_map = {r["league_id"]: r.get("role") for r in mem_rows} # type: ignore
    for r in league_rows:
        r["role"] = role_map.get(r["id"]) # type: ignore

    # Stable ordering
    league_rows.sort(key=lambda x: (x.get("name") or "").lower()) # type: ignore
    st.session_state["_lf_my_leagues"] = league_rows
    return league_rows


def get_my_leagues_for_session():
    leagues = st.session_state.get("_lf_my_leagues")
    if isinstance(leagues, list):
        return leagues
    return load_my_leagues()


def change_league_sidebar_ui() -> None:
    leagues = get_my_leagues_for_session()
    can_change = len(leagues) > 1 or is_superuser()
    if not can_change:
        return

    if st.sidebar.button("Change League", use_container_width=True, key="change_league_btn"):
        forget_selected_league()
        for key in ("league_id", "league_name", "league_role"):
            st.session_state.pop(key, None)
        st.session_state["page"] = "Home"
        invalidate_app_caches()
        st.rerun()


def _legacy_league_selector_ui():
    st.subheader("🏟️ Select League")

    leagues = load_my_leagues()
    if not leagues:
        st.warning("No leagues linked to your account yet.")

        st.markdown("### 🔑 Join a league with a code")
        code = st.text_input(
            "League code",
            placeholder="Paste the league code here",
            key="join_league_code",
        )

        if st.button("Join league", use_container_width=True):
            if not (code or "").strip():
                st.error("Please enter a league code.")
                return False

            result = join_league_by_code(code)
            if not result:
                st.error("Invalid league code.")
                return False

            # Success → store league in session
            invalidate_app_caches()
            st.session_state.league_id = result["league_id"]
            st.session_state.league_name = result["league_name"]
            st.session_state.league_role = result["role"]
            save_selected_league(result["league_id"], result["league_name"], result["role"])
            st.rerun()

        st.info("Ask your league admin for a code or invite link.")
        return False

    # If only one league, auto-select
    if len(leagues) == 1 and not st.session_state.get("league_id"):
        only = leagues[0]
        invalidate_app_caches()
        st.session_state.league_id = int(only["id"]) # type: ignore
        st.session_state.league_name = only["name"] # type: ignore
        st.session_state.league_role = only.get("role") # type: ignore
        save_selected_league(int(only["id"]), only["name"], only.get("role")) # type: ignore
        return True

    labels = [f"{l['name']} ({l.get('role', 'member')})" for l in leagues] # type: ignore
    choice = st.selectbox("League", labels)

    if st.button("Enter league", use_container_width=True):
        idx = labels.index(choice)
        selected = leagues[idx]
        invalidate_app_caches()
        st.session_state.league_id = int(selected["id"]) # type: ignore
        st.session_state.league_name = selected["name"] # type: ignore
        st.session_state.league_role = selected.get("role") # type: ignore
        save_selected_league(int(selected["id"]), selected["name"], selected.get("role")) # type: ignore
        st.rerun()

    return bool(st.session_state.get("league_id"))


def _enter_league(selected: dict):
    invalidate_app_caches()
    league_id = int(selected["id"]) # type: ignore
    league_name = selected["name"] # type: ignore
    league_role = selected.get("role") # type: ignore
    st.session_state.league_id = league_id
    st.session_state.league_name = league_name
    st.session_state.league_role = league_role
    save_selected_league(league_id, league_name, league_role)


def _join_code_form(key_prefix: str) -> bool:
    with st.form(f"{key_prefix}_join_form", clear_on_submit=False):
        code = st.text_input(
            "League code",
            placeholder="Paste the code from your organiser",
            key=f"{key_prefix}_join_code",
        )
        submitted = st.form_submit_button("Join league", use_container_width=True)

    if not submitted:
        return False

    if not (code or "").strip():
        st.error("Enter the league code first.")
        return False

    result = join_league_by_code(code)
    if not result:
        st.error("That code does not match an active league.")
        return False

    invalidate_app_caches()
    st.session_state.league_id = result["league_id"]
    st.session_state.league_name = result["league_name"]
    st.session_state.league_role = result["role"]
    save_selected_league(result["league_id"], result["league_name"], result["role"])
    st.success(f"Joined {result['league_name']}.")
    st.rerun()
    return True


def league_selector_ui():
    st.markdown(
        """
        <style>
        .lf-setup-card {
            max-width: 560px;
            margin: 18px auto 10px auto;
            padding: 20px 22px;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            background: rgba(255,255,255,0.035);
        }
        .lf-setup-title {
            font-size: 1.35rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 4px;
        }
        .lf-setup-sub {
            color: #aab3bd;
            text-align: center;
            font-size: 0.95rem;
        }
        </style>
        <div class="lf-setup-card">
            <div class="lf-setup-title">Choose your league</div>
            <div class="lf-setup-sub">Pick an existing league or join with a code from your organiser.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    leagues = load_my_leagues()
    if not leagues:
        st.info("No leagues are linked to your account yet.")
        _join_code_form("first_league")
        st.caption("Ask your league organiser for a code or invite link.")
        return False

    if restore_selected_league(leagues):
        return True

    if len(leagues) == 1 and not st.session_state.get("league_id"):
        _enter_league(leagues[0])
        return True

    labels = [f"{l['name']} - {l.get('role', 'member')}" for l in leagues] # type: ignore
    with st.form("select_league_form"):
        choice = st.selectbox("League", labels, label_visibility="collapsed")
        submitted = st.form_submit_button("Enter league", use_container_width=True)

    if submitted:
        idx = labels.index(choice)
        _enter_league(leagues[idx])
        st.rerun()

    with st.expander("Join another league with a code", expanded=False):
        _join_code_form("extra_league")

    return bool(st.session_state.get("league_id"))


def accept_invite_flow(invite_token: str):
    """
    Runs in app.py if URL contains ?invite=TOKEN
    """
    sb = sb_client_authed()
    user_id = st.session_state["sb_session"]["user_id"]

    inv = (
        sb.table("league_invites")
        .select("*")
        .eq("token", invite_token)
        .maybe_single()
        .execute()
    )
    invite = inv.data # type: ignore

    st.subheader("🎟️ Accept Invite")

    if not invite:
        st.error("Invite not found or invalid.")
        return

    if invite.get("used_at"): # type: ignore
        st.warning("Invite already used.")
        return

    if st.button("✅ Accept invite", use_container_width=True):
        # Join league
        sb.table("league_members").upsert(
            {
                "league_id": invite["league_id"], # type: ignore
                "user_id": user_id,
                "role": invite["role"], # type: ignore
                "status": "active",
            },
            on_conflict="league_id,user_id",
        ).execute()

        # Mark used
        sb.table("league_invites").update(
            {"used_by": user_id, "used_at": "now()"}
        ).eq("id", invite["id"]).execute() # type: ignore

        st.success("Invite accepted! Now select your league.")
        # Remove invite from URL
        st.query_params.clear()
        st.rerun()
