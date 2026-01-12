import streamlit as st
from utils.league_utils import update_my_display_name
from utils.db_utils import get_conn


def _get_my_profile():
    sb = st.session_state.get("sb_session") or {}
    user_id = sb.get("user_id") or (sb.get("user") or {}).get("id")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "select email, display_name from public.profiles where id = %s",
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()

    email = row[0] if row else None
    display_name = row[1] if row else ""

    return {
        "user_id": user_id,
        "email": email,
        "display_name": display_name or "",
    }


def render_profile_settings_page():
    st.title("👤 Profile Settings")

    prof = _get_my_profile()

    st.caption(f"Signed in as: {prof['email'] or 'Unknown'}")

    new_name = st.text_input(
        "Display name",
        value=prof["display_name"],
        max_chars=40,
        help="This is what everyone sees across the app (tables, charts, match sheets).",
    )

    if st.button("💾 Save display name", use_container_width=True):
        clean = new_name.strip()
        if len(clean) < 2:
            st.warning("Display name must be at least 2 characters.")
            return

        update_my_display_name(clean)

        # update session cache if you store it
        st.session_state["my_display_name"] = clean

        st.success("Display name updated.")
        st.rerun()
