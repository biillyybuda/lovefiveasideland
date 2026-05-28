import streamlit as st
from utils.branding import APP_LOGO
import base64
from pathlib import Path
from utils.db_utils import get_conn, get_current_league_id, load_matches_df, load_players_df
from utils.league_utils import is_demo_league_selected


@st.cache_data(ttl=300, show_spinner=False)
def _active_member_count(league_id: int) -> int:
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            select count(*)
            from public.league_members
            where league_id = %s
              and status = 'active'
            """,
            (int(league_id),),
        )
        row = cur.fetchone()
        return int(row[0] or 0) if row else 0
    except Exception:
        return 0
    finally:
        conn.close()


def _setup_step(done: bool, title: str, detail: str) -> str:
    marker = "✓" if done else "○"
    cls = "done" if done else "todo"
    return f"""
    <div class="setup-step {cls}">
        <div class="setup-marker">{marker}</div>
        <div>
            <div class="setup-title">{title}</div>
            <div class="setup-detail">{detail}</div>
        </div>
    </div>
    """


def _render_setup_checklist() -> None:
    if is_demo_league_selected():
        return

    role = (st.session_state.get("league_role") or "").lower()
    if role not in ("admin", "owner"):
        return

    try:
        league_id = get_current_league_id()
        players_df = load_players_df()
        matches_df = load_matches_df()
    except Exception:
        return

    player_count = int(len(players_df)) if players_df is not None else 0
    processed_matches = 0
    if matches_df is not None and not matches_df.empty:
        processed_matches = int(matches_df["processed"].fillna(0).astype(int).sum()) if "processed" in matches_df.columns else len(matches_df)

    member_count = _active_member_count(league_id)

    enough_players = player_count >= 10
    invite_shared = member_count >= 2
    first_result = processed_matches >= 1
    ready_to_explore = enough_players and first_result

    if enough_players and invite_shared and first_result:
        return

    st.markdown(
        f"""
        <style>
        .setup-wrap {{
            margin: 8px 0 18px 0;
            padding: 16px;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px;
            background: rgba(255,255,255,0.035);
        }}
        .setup-head {{
            display:flex;
            justify-content:space-between;
            gap:12px;
            align-items:flex-start;
            margin-bottom:12px;
        }}
        .setup-kicker {{
            color:#9aa0a6;
            font-size:12px;
            font-weight:800;
            text-transform:uppercase;
            letter-spacing:0.04em;
        }}
        .setup-heading {{
            font-size:20px;
            font-weight:900;
            margin-top:2px;
        }}
        .setup-progress {{
            color:#dbeafe;
            font-weight:900;
            white-space:nowrap;
            padding-top:2px;
        }}
        .setup-grid {{
            display:grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap:10px;
        }}
        .setup-step {{
            display:flex;
            gap:10px;
            min-height:92px;
            padding:12px;
            border-radius:10px;
            border:1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.025);
        }}
        .setup-step.done {{
            border-color:rgba(34,197,94,0.28);
            background:rgba(34,197,94,0.06);
        }}
        .setup-marker {{
            width:24px;
            height:24px;
            border-radius:999px;
            display:flex;
            align-items:center;
            justify-content:center;
            flex:0 0 24px;
            font-weight:900;
            color:#bbf7d0;
            background:rgba(34,197,94,0.12);
            border:1px solid rgba(34,197,94,0.32);
        }}
        .setup-step.todo .setup-marker {{
            color:#bfdbfe;
            background:rgba(59,130,246,0.10);
            border-color:rgba(59,130,246,0.28);
        }}
        .setup-title {{
            font-weight:900;
            font-size:14px;
            margin-bottom:4px;
        }}
        .setup-detail {{
            color:#aab3bd;
            font-size:12px;
            line-height:1.35;
        }}
        @media (max-width: 900px){{
            .setup-grid {{grid-template-columns:1fr 1fr;}}
        }}
        @media (max-width: 620px){{
            .setup-head {{display:block;}}
            .setup-grid {{grid-template-columns:1fr;}}
        }}
        </style>
        <div class="setup-wrap">
            <div class="setup-head">
                <div>
                    <div class="setup-kicker">League setup</div>
                    <div class="setup-heading">Get your league ready</div>
                </div>
                <div class="setup-progress">{sum([enough_players, invite_shared, first_result, ready_to_explore])}/4 done</div>
            </div>
            <div class="setup-grid">
                {_setup_step(enough_players, "Add players", f"{player_count}/10 players added")}
                {_setup_step(invite_shared, "Invite your group", f"{member_count} account(s) linked to this league")}
                {_setup_step(first_result, "Add first result", f"{processed_matches} processed match(es)")}
                {_setup_step(ready_to_explore, "Review the stats", "Dashboard and charts come alive after results")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Add players", use_container_width=True, key="setup_add_players"):
            st.session_state["_nav_target"] = "Player Management"
            st.rerun()
    with b2:
        if st.button("Invite people", use_container_width=True, key="setup_invite"):
            st.session_state["_nav_target"] = "Join / Invite"
            st.rerun()
    with b3:
        if st.button("Add result", use_container_width=True, key="setup_add_result", disabled=player_count < 2):
            st.session_state["_nav_target"] = "Matches Management"
            st.rerun()
    with b4:
        if st.button("View dashboard", use_container_width=True, key="setup_dashboard", disabled=processed_matches < 1):
            st.session_state["_nav_target"] = "Dashboard"
            st.rerun()

def render_home_page():
    # League context
    league_name = st.session_state.get("league_name") or "Your League"

    # ---------------------------------
    # Header: league context (top)
    # ---------------------------------
    st.markdown(
        f"""
        <style>
        .home-actions {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-top: 8px;
        }}
        .home-action {{
            display: block;
            text-align: center;
            padding: 13px 10px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.04);
            font-weight: 800;
            color: #e6eef6;
            min-height: 48px;
        }}
        @media (max-width: 760px){{
            .home-actions {{grid-template-columns: 1fr; gap: 8px;}}
            .home-league-title {{font-size: 1.45rem !important;}}
            .home-logo {{width: 210px !important; max-width: 78% !important;}}
        }}
        </style>
        <div style="text-align:center; margin-top:6px;">
            <div class="home-league-title" style="font-size:28px; font-weight:700; line-height:1.1;">
                {league_name}
            </div>
            <div style="font-size:13px; color:#9aa0a6; margin-top:4px;">
                Track your 5-a-side league like a pro
            </div>
        </div>
        <div style="height:14px;"></div>
        """,
        unsafe_allow_html=True,
    )

    _render_setup_checklist()

    # ---------------------------------
    # Smaller logo (optional, now secondary)
    # ---------------------------------
    logo_path = Path(APP_LOGO)
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    logo_ext = logo_path.suffix.lower().replace(".", "") or "png"

    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin-top:4px; margin-bottom:10px;">
            <img class="home-logo" src="data:image/{logo_ext};base64,{logo_b64}" style="width:280px; max-width:70%; opacity:0.95;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------
    # Primary actions
    # ---------------------------------
    role = (st.session_state.get("league_role") or "").lower()
    IS_ADMIN = role in ("admin", "owner")

    st.markdown(
        """
        <div style="text-align:center; font-size:12px; color:#9aa0a6; margin-bottom:8px;">
            Quick Navigation
        </div>
        """,
        unsafe_allow_html=True,
    )

    if IS_ADMIN and st.button("Add Result", use_container_width=True):
        st.session_state["_nav_target"] = "Matches Management"
        st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state["_nav_target"] = "Dashboard"
            st.rerun()
    with c2:
        if st.button("⚽ Matchday Hub", use_container_width=True):
            st.session_state["_nav_target"] = "Matchday Hub"
            st.rerun()
    with c3:
        if st.button("📈 Charts & Stats", use_container_width=True):
            st.session_state["_nav_target"] = "Charts & Stats"
            st.rerun()
    with c4:
        if st.button("🗂 Season Review", use_container_width=True):
            st.session_state["_nav_target"] = "Season Review"
            st.rerun()

    # ---------------------------------
    # Secondary: Admin tools (still here for now)
    # ---------------------------------
    c5, c6 = st.columns(2)
    with c5:
        if st.button("Join / Invite", use_container_width=True):
            st.session_state["_nav_target"] = "Join / Invite"
            st.rerun()
    with c6:
        if st.button("Profile Settings", use_container_width=True):
            st.session_state["_nav_target"] = "Profile Settings"
            st.rerun()

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='opacity:0.18;'>", unsafe_allow_html=True)

    if IS_ADMIN:
        st.markdown(
            """
            <div style="text-align:center; font-size:12px; color:#9aa0a6; margin-top:8px; margin-bottom:8px;">
                Admin
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("League Admin", use_container_width=True):
            st.session_state["_nav_target"] = "League Admin"
            st.rerun()
        if st.button("Player Management", use_container_width=True):
            st.session_state["_nav_target"] = "Player Management"
            st.rerun()
