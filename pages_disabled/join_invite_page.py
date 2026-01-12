import streamlit as st
from utils.league_utils import get_league_join_code, join_league_by_code


def render_join_invite_page():
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
