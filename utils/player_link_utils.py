from utils.db_utils import get_conn
import streamlit as st

def fetch_players_with_links(league_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        select
            p.id,
            p.display_name,
            p.user_id,
            pr.email,
            pr.display_name as profile_display_name
        from public.players p
        left join public.profiles pr
            on pr.id = p.user_id
        where p.league_id = %s
        order by p.display_name asc
        """,
        (league_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def fetch_league_members(league_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        select
            lm.user_id,
            pr.email,
            pr.display_name,
            lm.role,
            lm.status
        from public.league_members lm
        join public.profiles pr
            on pr.id = lm.user_id
        where lm.league_id = %s
          and lm.status = 'active'
        order by pr.email asc
        """,
        (league_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def admin_unlink_player(league_id: int, player_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        update public.players
        set user_id = null
        where id = %s and league_id = %s
        """,
        (player_id, league_id),
    )
    conn.commit()
    conn.close()

def admin_assign_player(league_id: int, player_id: int, user_id: str):
    conn = get_conn()
    cur = conn.cursor()

    # 1) Clear this user from any other player in the league (enforces 1:1)
    cur.execute(
        """
        update public.players
        set user_id = null
        where league_id = %s and user_id = %s
        """,
        (league_id, user_id),
    )

    # 2) Assign to target player
    cur.execute(
        """
        update public.players
        set user_id = %s
        where id = %s and league_id = %s
        """,
        (user_id, player_id, league_id),
    )

    conn.commit()
    conn.close()



def _get_user_id():
    sb = st.session_state.get("sb_session") or {}
    return sb.get("user_id") or (sb.get("user") or {}).get("id")


def _is_linked(league_id: int, user_id: str) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "select 1 from public.players where league_id = %s and user_id = %s limit 1",
        (league_id, user_id),
    )
    ok = cur.fetchone() is not None
    conn.close()
    return ok


def _fetch_unlinked_players(league_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        select id, display_name
        from public.players
        where league_id = %s
          and (user_id is null)
        order by display_name asc
        """,
        (league_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def link_user_to_player(league_id: int, user_id: str, player_id: int):
    conn = get_conn()
    cur = conn.cursor()
    # Only allow linking if currently unlinked
    cur.execute(
        """
        update public.players
        set user_id = %s
        where id = %s and league_id = %s and user_id is null
        """,
        (user_id, player_id, league_id),
    )
    conn.commit()
    conn.close()


def _legacy_ensure_player_linked_ui():
    league_id = st.session_state.get("league_id")
    user_id = _get_user_id()

    if not league_id or not user_id:
        return

    if _is_linked(int(league_id), str(user_id)):
        return  # all good

    st.title("👋 Quick setup")
    st.subheader("Who are you in this league?")
    st.caption("Pick your player profile so your stats and display name sync correctly.")

    players = _fetch_unlinked_players(int(league_id))
    if not players:
        st.warning("No unlinked players available. Ask an admin to add you, or fix links.")
        st.stop()

    labels = [p[1] for p in players]
    ids = [p[0] for p in players]

    choice = st.selectbox("Select your player", labels)
    idx = labels.index(choice)
    player_id = ids[idx]

    if st.button("✅ Link my account", use_container_width=True):
        link_user_to_player(int(league_id), str(user_id), int(player_id))
        st.success("Linked! You’re ready.")
        st.rerun()

    st.stop()


def ensure_player_linked_ui():
    league_id = st.session_state.get("league_id")
    user_id = _get_user_id()

    if not league_id or not user_id:
        return

    if _is_linked(int(league_id), str(user_id)):
        return

    league_name = st.session_state.get("league_name") or "this league"
    st.markdown(
        """
        <style>
        .lf-link-card {
            max-width: 620px;
            margin: 18px auto 12px auto;
            padding: 22px;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            background: rgba(255,255,255,0.035);
        }
        .lf-link-title {
            font-size: 1.35rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 6px;
        }
        .lf-link-sub {
            color: #aab3bd;
            text-align: center;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="lf-link-card">
            <div class="lf-link-title">Link your player profile</div>
            <div class="lf-link-sub">Choose who you are in {league_name}. This keeps your stats, display name and account connected.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    players = _fetch_unlinked_players(int(league_id))
    if not players:
        st.warning("There are no unlinked player profiles available.")
        st.info("Ask a league admin to add your player or unlink an old profile, then refresh this page.")
        st.stop()

    labels = [p[1] for p in players]
    ids = [p[0] for p in players]

    with st.form("link_player_form"):
        choice = st.selectbox("Your player profile", labels)
        submitted = st.form_submit_button("Link my account", use_container_width=True)

    if submitted:
        idx = labels.index(choice)
        link_user_to_player(int(league_id), str(user_id), int(ids[idx]))
        st.success("Linked. You are ready to go.")
        st.rerun()

    st.caption("Not listed? An admin can create or free up your player profile from Player Management.")
    st.stop()
