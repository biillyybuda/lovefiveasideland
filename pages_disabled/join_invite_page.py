import streamlit as st
from utils.league_utils import get_league_join_code, join_league_by_code


def _legacy_render_join_invite_page():
    st.title("🔗 Join / Invite")

    # --- Share ---
    st.subheader("Invite your mates")

    league_id = st.session_state.get("league_id")
    if league_id:
        code = get_league_join_code(league_id)
        st.code(code, language="text")
        st.caption(
            "They create an account, enter this code, set a display name — and they’re in."
        )
    else:
        st.info("Select a league to see its invite code.")

    st.markdown("---")

    # --- Join ---
    st.subheader("Join a league")

    code_in = st.text_input("League code").strip()

    if st.button("✅ Join league", use_container_width=True):
        result = join_league_by_code(code_in)
        if not result:
            st.error("Invalid league code.")
            return

        st.session_state["league_id"] = result["league_id"]
        st.session_state["league_name"] = result["league_name"]
        st.session_state["league_role"] = result["role"]

        st.success(f"Joined {result['league_name']}")
        st.rerun()


def render_join_invite_page():
    st.title("Join / Invite")

    role = (st.session_state.get("league_role") or "").lower()
    is_admin = role in ("admin", "owner")

    st.subheader("Invite players")
    league_id = st.session_state.get("league_id")
    if not is_admin:
        st.info("Ask a league admin for the invite code if someone needs adding.")
    elif league_id:
        code = get_league_join_code(league_id)
        league_name = st.session_state.get("league_name") or "your league"
        st.caption(f"Share this code with players you want to add to {league_name}.")
        st.code(code, language="text")
        st.text_area(
            "Share message",
            value=f"Join our Love Five league with this code: {code}",
            height=90,
            key="invite_share_message",
        )
    else:
        st.info("Select a league to see its invite code.")

    st.markdown("---")
    st.subheader("Join another league")

    with st.form("join_league_page_form", clear_on_submit=False):
        code_in = st.text_input("League code", placeholder="Paste league code").strip()
        submitted = st.form_submit_button("Join league", use_container_width=True)

    if not submitted:
        return

    if not code_in:
        st.error("Enter the league code first.")
        return

    result = join_league_by_code(code_in)
    if not result:
        st.error("That code does not match an active league.")
        return

    st.session_state["league_id"] = result["league_id"]
    st.session_state["league_name"] = result["league_name"]
    st.session_state["league_role"] = result["role"]

    st.success(f"Joined {result['league_name']}.")
    st.rerun()
