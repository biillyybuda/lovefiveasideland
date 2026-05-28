import streamlit as st
import pandas as pd
from utils.db_utils import load_players_df, load_matches_df, get_conn, STARTING_MMR
from utils.calc_utils import compute_streaks_from_matches
from utils.export_utils import df_to_png
from collections import defaultdict
from utils.ui_components import page_header
from utils.names import display_name_map_from_players_df, player_display_name




def compute_result_from_score(score: str):
    s = str(score or "").strip().replace("–", "-").replace("—", "-")
    if "-" not in s:
        return None
    try:
        a, b = s.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
    except Exception:
        return None
    if a > b: return "A"
    if b > a: return "B"
    return "DRAW"

@st.cache_data(ttl=300)
def _season_baseline_map(player_ids: tuple[int, ...], season_start: str):
    if not player_ids:
        return {}

    conn = get_conn()
    try:
        df = pd.read_sql(
            """
            SELECT DISTINCT ON (player_id)
                   player_id, mmr_after
            FROM public.mmr_history
            WHERE player_id = ANY(%s)
              AND date < %s
            ORDER BY player_id, date DESC, id DESC
            """,
            conn,
            params=(list(player_ids), season_start),
        )
    finally:
        conn.close()

    return {
        int(row["player_id"]): float(row["mmr_after"])
        for _, row in df.iterrows()
        if pd.notna(row.get("mmr_after"))
    }


@st.cache_data(ttl=300)
def _season_mmr_map(player_rows: tuple[tuple[int, float], ...], season_start: str):
    player_ids = tuple(int(pid) for pid, _ in player_rows)
    baselines = _season_baseline_map(player_ids, season_start)
    out = {}
    for pid, current_mmr in player_rows:
        baseline = baselines.get(int(pid), float(STARTING_MMR))
        out[int(pid)] = float(STARTING_MMR) + (float(current_mmr) - baseline)
    return out


def render_dashboard_page():
    st.set_page_config(page_title="Dashboard | Love Five-A-Side", layout="wide")
    page_header(
        "Dashboard",
        "Your season overview and key stats at a glance",
        center=True,
        divider=True,
    )

    dfp = load_players_df()

    name_map = display_name_map_from_players_df(dfp)
    alias_map = {}
    for base, dispn in name_map.items():
        alias_map[base.lower()] = dispn
        alias_map[dispn.lower()] = dispn

    def disp(n: str) -> str:
        s = str(n or "").strip()
        if not s:
            return s
        return alias_map.get(s.lower(), player_display_name(s))

    dfm = load_matches_df()
    dfm['processed'] = dfm['processed'].astype(int)

    # -----------------------------
    # 🗓️ Season / Year Filter
    # -----------------------------
    dfm["date_dt"] = pd.to_datetime(dfm["date"], errors="coerce")
    years = sorted([int(y) for y in dfm["date_dt"].dropna().dt.year.unique().tolist()]) # type: ignore

    # fallback if DB has no dates parsed yet
    if not years:
        years = [pd.Timestamp.today().year]

    season_mode = st.selectbox(
        "Season View",
        options=["Combined (rolling)", "Single Year (season reset)"],
        index=1,
        help="Combined uses true rolling MMR. Single Year resets display MMR to 1000 at Jan 1 of the selected year."
    )

    selected_year = None
    if season_mode == "Single Year (season reset)":
        selected_year = st.selectbox(
            "Select Year",
            options=years,
            index=len(years) - 1
        )

    def _in_selected_year(d: pd.Timestamp) -> bool:
        if selected_year is None or pd.isna(d):
            return False
        return int(d.year) == int(selected_year)

    # Filter matches used by the whole page
    dfm_proc = dfm[dfm["processed"] == 1].copy()
    if season_mode == "Single Year (season reset)":
        dfm_proc = dfm_proc[dfm_proc["date_dt"].apply(_in_selected_year)].copy()

    # season_start string used for display transforms
    season_start = f"{selected_year}-01-01" if selected_year is not None else None


    # Total players = unique players who actually played (in processed matches, for the selected period)
    def _parse_team(s): # type: ignore
        return [p.strip() for p in str(s or "").split(",") if p.strip()]

    players_played = set()
    for _, m in dfm_proc.iterrows():
        players_played.update(_parse_team(m.get("team_a", "")))
        players_played.update(_parse_team(m.get("team_b", "")))

    total_players = len(players_played)
    total_matches = len(dfm_proc)


    # --- Avg Win % (season-aware): average of player win% ---
    def _parse_team(s):
        return [p.strip() for p in str(s or "").split(",") if p.strip()]

    matches_ct = defaultdict(int)
    wins_ct    = defaultdict(int)
    draws_ct   = defaultdict(int)

    for _, m in dfm_proc.iterrows():
        res = compute_result_from_score(m.get("score", "")) or str(m.get("result", "") or "").strip().upper()
        ta = _parse_team(m.get("team_a", ""))
        tb = _parse_team(m.get("team_b", ""))

        for p in ta + tb:
            matches_ct[p] += 1

        if res == "A":
            for p in ta:
                wins_ct[p] += 1
        elif res == "B":
            for p in tb:
                wins_ct[p] += 1
        elif res == "DRAW":
            for p in ta + tb:
                draws_ct[p] += 1

    # average ONLY players who played
    player_pcts = []
    for p, mp in matches_ct.items():
        if mp > 0:
            w = wins_ct.get(p, 0)
            d = draws_ct.get(p, 0)
            player_pcts.append(((w + 0.5 * d) / mp) * 100)

    avg_win = round(sum(player_pcts) / len(player_pcts), 1) if player_pcts else 0.0



    if season_mode == "Single Year (season reset)" and season_start:
        player_rows = tuple((int(pid), float(mmr)) for pid, mmr in zip(dfp["id"].tolist(), dfp["mmr"].tolist()))
        mmr_map = _season_mmr_map(player_rows, season_start)
        dfp["_display_mmr"] = dfp["id"].map(mmr_map).astype(float)
    else:
        dfp["_display_mmr"] = dfp["mmr"].astype(float)

    # Top player should be among players who actually played in the selected period
    dfp_rank = dfp[dfp["name"].astype(str).str.strip().str.lower().isin(
        {str(p).strip().lower() for p in players_played}
    )].copy()

    top_player = dfp_rank.loc[dfp_rank["_display_mmr"].idxmax(), "name"] if not dfp_rank.empty else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧑‍🤝‍🧑 Total Players", total_players)
    col2.metric("⚔️ Matches Played", total_matches)
    col3.metric("📈 Avg Player Win %", f"{avg_win}%")
    col4.metric("🥇 Top Player", disp(top_player) if top_player != "N/A" else "N/A") # type: ignore
    st.markdown("---")

    # --- Most recent match ---
    st.markdown("### 🕒 Most Recent Match")
    recent = dfm_proc.sort_values('date_dt', ascending=False).head(1)
    if recent.empty:
        st.info("No processed matches yet.")
    else:
        r = recent.iloc[0]
        st.markdown(f"**Date:** {r.get('date','')}  |  **Score:** {r.get('score','')}")
        team_a = [p.strip() for p in str(r['team_a']).split(',') if p.strip()]
        team_b = [p.strip() for p in str(r['team_b']).split(',') if p.strip()]
        st.markdown(f"**Team A:** {', '.join(disp(p) for p in team_a)}")
        st.markdown(f"**Team B:** {', '.join(disp(p) for p in team_b)}")

        conn = get_conn()
        try:
            mh = pd.read_sql(
                """
                SELECT mh.*, p.display_name AS name
                FROM mmr_history mh
                JOIN players p ON mh.player_id=p.id
                WHERE mh.match_id = %s
                ORDER BY mh.id
                """,
                conn,
                params=(int(r["id"]),),
            )

            if not mh.empty:
                # --- Display MMR: rolling vs season reset
                if season_mode == "Single Year (season reset)" and season_start:
                    player_ids = tuple(int(pid) for pid in mh["player_id"].dropna().unique().tolist())
                    baselines = _season_baseline_map(player_ids, season_start)

                    def season_display_mmr(row, col):
                        baseline = baselines.get(int(row["player_id"]), float(STARTING_MMR))
                        return float(STARTING_MMR) + (float(row[col]) - baseline)

                    mh["MMR Before"] = mh.apply(lambda r2: season_display_mmr(r2, "mmr_before"), axis=1)
                    mh["MMR After"] = mh.apply(lambda r2: season_display_mmr(r2, "mmr_after"), axis=1)
                else:
                    mh["MMR Before"] = mh["mmr_before"].astype(float)
                    mh["MMR After"] = mh["mmr_after"].astype(float)

                mh["Delta"] = mh["MMR After"] - mh["MMR Before"]

                display = mh[["name", "MMR Before", "MMR After", "Delta"]].rename(columns={"name": "Name"})
                display["Delta"] = display["Delta"].apply(
                    lambda v: f"🟢 +{int(v)}" if v > 0 else (f"🔴 {int(v)}" if v < 0 else "⚪ 0")
                )

                st.dataframe(display.set_index("Name"))

                st.download_button(
                    "📤 Export Match MMR Changes as PNG",
                    data=df_to_png(display, title="Most Recent Match — MMR Changes"),
                    file_name="recent_match_mmr.png",
                    mime="image/png",
                )
        finally:
            conn.close()




    st.markdown("---")

    # -----------------------------
    # 🔍 Filters
    # -----------------------------
    st.markdown("### 🔍 Filters")

    min_games = st.slider(
        "Minimum Matches Played",
        min_value=0,
        max_value=20,
        value=3,
        step=1,
        help="Only show players with this many or more recorded matches"
    )


    # --- Player Performance Overview (computed from matches) ---
    st.markdown("### 📊 Player Performance Overview")

    dfm_proc = dfm_proc.copy()
    dfm_proc["date"] = pd.to_datetime(dfm_proc["date"], errors="coerce")

    def parse_team(s):
        return [p.strip() for p in str(s or "").split(",") if p.strip()]

    def outcome(row):
        score = str(row.get('score', '') or '')
        res   = str(row.get('result', '') or '').strip().upper()
        if '-' in score:
            try:
                a, b = score.split('-', 1)
                a, b = int(a.strip()), int(b.strip())
                if a > b: return 'A'
                if b > a: return 'B'
                return 'DRAW'
            except Exception:
                pass
        if res in ('A','B','DRAW'):
            return res
        return 'UNKNOWN'

    matches_ct = defaultdict(int)
    wins_ct    = defaultdict(int)
    losses_ct  = defaultdict(int)
    draws_ct   = defaultdict(int)

    for _, m in dfm_proc.iterrows():
        ta = parse_team(m.get('team_a'))
        tb = parse_team(m.get('team_b'))
        res = outcome(m)

        for p in ta + tb:
            matches_ct[p] += 1

        if res == 'A':
            for p in ta: wins_ct[p] += 1
            for p in tb: losses_ct[p] += 1
        elif res == 'B':
            for p in tb: wins_ct[p] += 1
            for p in ta: losses_ct[p] += 1
        elif res == 'DRAW':
            for p in ta + tb: draws_ct[p] += 1

    # compute streaks
    cur_win, cur_lose, max_win, max_lose = compute_streaks_from_matches(dfm_proc)

    total_processed_matches = len(dfm_proc)
    def attendance_pct(name):
        mp = matches_ct.get(name, 0)
        return round((mp / total_processed_matches * 100), 1) if total_processed_matches > 0 else 0.0

    players_df = dfp
    if players_df.empty:
        st.info("No players found yet. Add players in Player Management to get started.")
        return

    all_players = players_df["name"].tolist()
    rows = []
    for name in all_players:
        mp = matches_ct.get(name, 0)
        w  = wins_ct.get(name, 0)
        l  = losses_ct.get(name, 0)
        d  = draws_ct.get(name, 0)
        win_pct = round((w / mp) * 100, 1) if mp > 0 else 0.0
        rows.append({
            'Name': disp(name),
            'Matches': mp,
            'Wins': w,
            'Losses': l,
            'Draws': d,
            'Attendance %': attendance_pct(name),
            'Win %': win_pct,
            'Winning Streak': cur_win.get(name, 0),
            'Losing Streak': cur_lose.get(name, 0),
        })

    df_view = pd.DataFrame(
        rows,
        columns=[
            "Name","Matches","Wins","Losses","Draws",
            "Attendance %","Win %","Winning Streak","Losing Streak"
        ]
    )
    df_view = df_view[df_view["Matches"] >= min_games]

    # 🔹 Add MMR column from players table
    df_mmr = dfp[["id", "name", "display_name", "mmr"]].copy()
    if season_mode == "Single Year (season reset)" and season_start:
        player_rows = tuple((int(pid), float(mmr)) for pid, mmr in zip(df_mmr["id"].tolist(), df_mmr["mmr"].tolist()))
        mmr_map = _season_mmr_map(player_rows, season_start)
        df_mmr["MMR"] = df_mmr["id"].map(mmr_map).astype(float).round(1)
    else:
        df_mmr["MMR"] = df_mmr["mmr"].astype(float).round(1)

    df_mmr["Name"] = df_mmr.apply(
        lambda r: player_display_name(str(r.get("name") or ""), r.get("display_name")),
        axis=1,
    )
    df_mmr = df_mmr[["Name", "MMR"]]
    df_view = pd.merge(df_view, df_mmr, on="Name", how="left")

    # 🔹 Keep existing sort order (Win % then Matches)
    df_view = df_view.sort_values(["Win %", "Matches"], ascending=[False, False]).reset_index(drop=True)
    st.data_editor(
        df_view,
        use_container_width=True,
        hide_index=True,
        height=(len(df_view) * 35) + 80  # auto-adjust height to row count
    )


    if df_view.empty:
        st.info("No players meet the current filter yet (try lowering Minimum Matches Played).")
    else:
        st.download_button(
            "📥 Export Player Stats as PNG",
            data=df_to_png(df_view, title="Player Performance Overview"),
            file_name="player_stats.png",
            mime="image/png",
        )
