# utils/league_utils.py
import streamlit as st
from utils.auth_utils import sb_client_authed


def load_my_leagues():
    sb = sb_client_authed()

    mem = (
        sb.table("league_members")
        .select("league_id,role,status")
        .eq("status", "active")
        .execute()
    )
    mem_rows = mem.data or []
    league_ids = [r["league_id"] for r in mem_rows]

    if not league_ids:
        return []

    leagues = (
        sb.table("leagues")
        .select("id,name")
        .in_("id", league_ids)
        .execute()
    )
    league_rows = leagues.data or []

    role_map = {r["league_id"]: r.get("role") for r in mem_rows}
    for r in league_rows:
        r["role"] = role_map.get(r["id"])

    # Stable ordering
    league_rows.sort(key=lambda x: (x.get("name") or "").lower())
    return league_rows


def league_selector_ui():
    st.subheader("🏟️ Select League")

    leagues = load_my_leagues()
    if not leagues:
        st.warning("No leagues linked to your account yet.")
        st.info("Ask an admin for an invite link, or create a league (admin-only page later).")
        return False

    # If only one league, auto-select
    if len(leagues) == 1 and not st.session_state.get("league_id"):
        only = leagues[0]
        st.cache_data.clear()
        st.session_state.league_id = int(only["id"])
        st.session_state.league_name = only["name"]
        st.session_state.league_role = only.get("role")
        return True

    labels = [f"{l['name']} ({l.get('role', 'member')})" for l in leagues]
    choice = st.selectbox("League", labels)

    if st.button("Enter league", use_container_width=True):
        idx = labels.index(choice)
        selected = leagues[idx]
        st.cache_data.clear()
        st.session_state.league_id = int(selected["id"])
        st.session_state.league_name = selected["name"]
        st.session_state.league_role = selected.get("role")
        st.rerun()

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
    invite = inv.data

    st.subheader("🎟️ Accept Invite")

    if not invite:
        st.error("Invite not found or invalid.")
        return

    if invite.get("used_at"):
        st.warning("Invite already used.")
        return

    if st.button("✅ Accept invite", use_container_width=True):
        # Join league
        sb.table("league_members").upsert(
            {
                "league_id": invite["league_id"],
                "user_id": user_id,
                "role": invite["role"],
                "status": "active",
            },
            on_conflict="league_id,user_id",
        ).execute()

        # Mark used
        sb.table("league_invites").update(
            {"used_by": user_id, "used_at": "now()"}
        ).eq("id", invite["id"]).execute()

        st.success("Invite accepted! Now select your league.")
        # Remove invite from URL
        st.query_params.clear()
        st.rerun()
