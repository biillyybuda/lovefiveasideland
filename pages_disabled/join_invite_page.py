import streamlit as st
from utils.cache_utils import invalidate_app_caches
from utils.auth_utils import save_selected_league
from utils.league_utils import (
    create_league_for_current_user,
    get_league_join_code,
    join_demo_league,
    join_league_by_code,
)


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
    demo_tab, join_tab, create_tab = st.tabs(["Demo league", "Join league", "Create league"])

    with demo_tab:
        st.subheader("Demo league")
        if st.button("Try the demo league", use_container_width=True, key="join_page_demo_league"):
            result = join_demo_league()
            if not result:
                st.error("The demo league is not available right now.")
                return

            invalidate_app_caches()
            st.session_state["league_id"] = result["league_id"]
            st.session_state["league_name"] = result["league_name"]
            st.session_state["league_role"] = result["role"]
            st.session_state["_lf_demo_viewer"] = True
            st.query_params["demo"] = "1"
            st.rerun()

    with join_tab:
        st.subheader("Join another league")

        with st.form("join_league_page_form", clear_on_submit=False):
            code_in = st.text_input("League code", placeholder="Paste league code").strip()
            submitted = st.form_submit_button("Join league", use_container_width=True)

        if submitted:
            if not code_in:
                st.error("Enter the league code first.")
                return

            result = join_league_by_code(code_in)
            if not result:
                st.error("That code does not match an active league.")
                return

            invalidate_app_caches()
            st.session_state["league_id"] = result["league_id"]
            st.session_state["league_name"] = result["league_name"]
            st.session_state["league_role"] = result["role"]
            st.session_state.pop("_lf_demo_viewer", None)
            st.query_params.pop("demo", None)
            save_selected_league(result["league_id"], result["league_name"], result["role"])

            st.success(f"Joined {result['league_name']}.")
            st.rerun()

    with create_tab:
        st.subheader("Create a league")

        with st.form("create_league_page_form", clear_on_submit=False):
            league_name = st.text_input(
                "League name",
                placeholder="e.g. Thursday Night Fives",
                key="join_page_create_league_name",
            )
            create_submitted = st.form_submit_button("Create league", use_container_width=True)

        if not create_submitted:
            return

        try:
            created = create_league_for_current_user(league_name)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Could not create league: {exc}")
            return

        invalidate_app_caches()
        st.session_state["league_id"] = created["league_id"]
        st.session_state["league_name"] = created["league_name"]
        st.session_state["league_role"] = created["role"]
        st.session_state.pop("_lf_demo_viewer", None)
        st.query_params.pop("demo", None)
        save_selected_league(created["league_id"], created["league_name"], created["role"])

        st.success(f"Created {created['league_name']}.")
        st.rerun()
