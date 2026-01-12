import streamlit as st
from utils.db_utils import get_conn

from utils.db_utils import get_conn

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


def ensure_player_linked_ui():
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
