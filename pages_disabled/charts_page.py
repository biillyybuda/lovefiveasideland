import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict
from utils.mmr_utils import get_season_mmr, get_current_season_start


from utils.db_utils import get_conn
from utils.export_utils import df_to_png, fig_to_png_bytes  # kept for compatibility (may be used elsewhere)
from utils.stats_shared import (
    get_chemistry_df,
    get_intensity_df,
    get_pair_chemistry,
    get_pair_intensity,
)
from utils.names import display_name as _fallback_display_name
from utils.ui_components import page_header


# ----------------------------
# Helpers
# ----------------------------
def _split_team(val: str):
    return [p.strip() for p in str(val or "").split(",") if str(p).strip()]


def get_name_map(conn) -> dict:
    """
    Map DB key -> UI display name.
    DB key = players.name
    UI name = players.display_name
    """
    df = pd.read_sql("SELECT name, display_name FROM players", conn)
    df["name"] = df["name"].fillna("").astype(str)
    df["display_name"] = df["display_name"].fillna("").astype(str)

    name_map = {}
    for _, r in df.iterrows():
        key = r["name"].strip()
        ui = r["display_name"].strip()
        if not ui:
            ui = _fallback_display_name(key)
        name_map[key] = ui
    return name_map


def to_display(key: str, name_map: dict) -> str:
    if key is None:
        return ""
    k = str(key).strip()
    return name_map.get(k, _fallback_display_name(k))


def get_season_filter_ui(matches_df: pd.DataFrame, suffix=""):
    """
    Returns (season_mode, selected_year, season_start, matches_filtered)
    """
    df = matches_df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    years = sorted([int(y) for y in df["date_dt"].dropna().dt.year.unique().tolist()]) # type: ignore
    if not years:
        years = [pd.Timestamp.today().year]

    season_mode = st.selectbox(
        "Season View",
        options=["Rolling (all years)", "Single Year (season reset)"],
        index=0,
        key=f"charts_season_mode_{suffix}",
    )

    selected_year = None
    if season_mode == "Single Year (season reset)":
        selected_year = st.selectbox(
            "Select Year",
            options=years,
            index=len(years) - 1,
            key=f"charts_selected_year_{suffix}",
        )

    if season_mode == "Single Year (season reset)":
        df_filt = df[df["date_dt"].dt.year == int(selected_year)].copy() # type: ignore
        season_start = f"{int(selected_year)}-01-01" # type: ignore
    else:
        df_filt = df.copy()
        season_start = None

    return season_mode, selected_year, season_start, df_filt



# ----------------------------
# 📈 MMR Progression Over Time (moved from Dashboard)
# ----------------------------
def render_mmr_progression_over_time(suffix="", season_mode=None, selected_year=None, season_start=None, matches=None):
    st.subheader("📈 MMR Progression Over Time")

    conn = get_conn()
    try:
        # Pull full MMR history (processed matches only)
        df_mmr = pd.read_sql(
            """
            SELECT
                m.id AS match_id,
                m.date AS match_date,
                mh.player_id,
                p.name AS player_key,
                p.display_name AS player_display,
                mh.mmr_after
            FROM mmr_history mh
            JOIN matches m ON mh.match_id = m.id
            JOIN players p ON mh.player_id = p.id
            WHERE m.processed = 1
            ORDER BY m.date ASC, mh.id ASC
            """,
            conn,
        )

        if df_mmr.empty:
            st.info("No MMR history yet.")
            return

        # Apply the same season / match filters used by the page
        if matches is not None and not matches.empty and "id" in matches.columns:
            allowed_ids = set(matches["id"].astype(int).tolist())
            df_mmr = df_mmr[df_mmr["match_id"].astype(int).isin(allowed_ids)].copy()

        df_mmr["date_dt"] = pd.to_datetime(df_mmr["match_date"], errors="coerce")

        # Remove weird microsecond ticks (normalize to day)
        df_mmr["date_dt"] = df_mmr["date_dt"].dt.normalize() # type: ignore

        name_map = get_name_map(conn)
        df_mmr["player_label"] = df_mmr["player_key"].apply(lambda k: to_display(k, name_map))

        # Season-reset display MMR
        show_season = (season_mode == "Single Year (season reset)" and season_start)
        if show_season:
            df_mmr["mmr_plot"] = df_mmr.apply(
                lambda r: float(get_season_mmr(conn, int(r["player_id"]), season_start, float(r["mmr_after"]))), # type: ignore
                axis=1,
            )
            ycol = "mmr_plot"
            ylab = "Season MMR"
        else:
            df_mmr["mmr_plot"] = df_mmr["mmr_after"].astype(float)
            ycol = "mmr_plot"
            ylab = "Rolling MMR"

        options = sorted(df_mmr["player_label"].dropna().unique().tolist())

        player_choice = st.multiselect(
            "Select Player(s)",
            options=options,
            default=[],
            key=f"mmr_progress_players_{suffix}",
        )

        if not player_choice:
            st.info("Choose one or more players to plot their MMR progression.")
            return

        df_filtered = df_mmr[df_mmr["player_label"].isin(player_choice)].copy()

        fig = px.line(
            df_filtered,
            x="date_dt",
            y=ycol,
            color="player_label",
            markers=True,
            title="MMR Over Time",
            labels={
                "date_dt": "Match Date",
                ycol: ylab,
                "player_label": "Player",
            },
        )
        st.plotly_chart(fig, use_container_width=True, key=f"mmr_progress_fig_{suffix}")
    finally:
        conn.close()

# ----------------------------
# 🌍 GLOBAL OVERVIEW
# ----------------------------
def render_global_overview(suffix="", season_mode=None, selected_year=None, season_start=None, matches=None):
    st.markdown("## 🌍 Global Overview")

    conn = get_conn()
    try:
        matches = matches.copy() if matches is not None else pd.read_sql("SELECT * FROM matches WHERE processed=1 ORDER BY date ASC", conn)
        players = pd.read_sql("SELECT * FROM players ORDER BY name", conn)

        if matches.empty:
            st.info("No processed matches yet.")
            return

        name_map = get_name_map(conn)
        plist = players["name"].tolist()

        # ----------------------------
        # Minimum games filter (Global Overview)
        # ----------------------------
        min_games = st.slider(
            "Minimum games played",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            help="Hide players with too few games in the selected season view.",
            key=f"global_min_games_{suffix}",
        )

        # Count appearances in the filtered matches (season-aware)
        games_played = defaultdict(int)
        for _, m in matches.iterrows():
            for nm in set(_split_team(m.get("team_a", "")) + _split_team(m.get("team_b", ""))):
                if nm:
                    games_played[nm] += 1

        # Only keep players meeting the threshold
        eligible_players = sorted([p for p, gp in games_played.items() if int(gp) >= int(min_games)])


        # 🥇 Win % by Player
        st.subheader("🥇 Win % by Player")

        # ✅ Only include players who meet the minimum games threshold
        plist = eligible_players

        win_rows = []
        for nm in plist:
            total = 0
            wins = 0
            for _, m in matches.iterrows():
                ta = _split_team(m.get("team_a", ""))
                tb = _split_team(m.get("team_b", ""))
                res = (m.get("result") or "").upper()
                if nm in ta or nm in tb:
                    total += 1
                    if (nm in ta and res == "A") or (nm in tb and res == "B"):
                        wins += 1
            wp = round((wins / total * 100), 1) if total > 0 else 0.0
            win_rows.append({"Name": to_display(nm, name_map), "Win %": wp})

        df_win = pd.DataFrame(win_rows).sort_values("Win %", ascending=False)
        fig_win = px.bar(df_win, x="Name", y="Win %", title="Win % by Player", text="Win %")
        fig_win.update_traces(textposition="outside")
        fig_win.update_yaxes(range=[0, df_win["Win %"].max() * 1.15 if not df_win.empty else 100])
        fig_win.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig_win, use_container_width=True, key=f"fig_win_global_{suffix}")

        # 📅 Attendance (%)
        st.subheader("📅 Attendance (All Players)")
        total_matches = len(matches)
        att = defaultdict(int)
        for _, m in matches.iterrows():
            allp = _split_team(m.get("team_a", "")) + _split_team(m.get("team_b", ""))
            for nm in allp:
                att[nm] += 1

        df_att = pd.DataFrame(
            [{"Player": k, "Attendance %": round((v / total_matches) * 100, 1)} for k, v in att.items()]
        ).sort_values("Attendance %", ascending=False)

        # ✅ Apply minimum games filter
        df_att = df_att[df_att["Player"].isin(eligible_players)].copy()

        df_att["Player"] = df_att["Player"].apply(lambda k: to_display(k, name_map))

        fig_att = px.bar(df_att, x="Player", y="Attendance %", title="Attendance % (All Players)", text="Attendance %")
        fig_att.update_traces(textposition="outside")
        fig_att.update_yaxes(range=[0, df_att["Attendance %"].max() * 1.15 if not df_att.empty else 100])
        fig_att.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig_att, use_container_width=True, key=f"fig_att_global_{suffix}")

        # 🤝 Top 10 Duos (Chemistry) & ⚔️ Top 10 Rivalries (Intensity)
        st.subheader("🤝 Top Duos & ⚔️ Rivalries")

        # These two DataFrames come from stats_shared (so formula changes live there)
        chem_df = (
            get_chemistry_df(conn, matches_df=matches)
            .sort_values(by="chemistry", ascending=False)
            .head(10)
        )

        intensity_df = (
            get_intensity_df(conn, matches_df=matches)
            .sort_values(by="intensity", ascending=False)
            .head(10)
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Top 10 Duos by Chemistry**")
            if chem_df is None or chem_df.empty:
                st.info("No chemistry data yet.")
            else:
                chem_display = (
                    chem_df.rename(
                        columns={
                            "player_a": "Player A",
                            "player_b": "Player B",
                            "matches": "Matches",
                            "wins": "Wins",
                            "win_pct": "Win %",
                            "chemistry": "Chemistry",
                        }
                    )[["Player A", "Player B", "Matches", "Win %", "Chemistry"]]
                    .reset_index(drop=True)
                )
                chem_display["Player A"] = chem_display["Player A"].apply(lambda k: to_display(k, name_map))
                chem_display["Player B"] = chem_display["Player B"].apply(lambda k: to_display(k, name_map))
                st.dataframe(chem_display, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**Top 10 Rivalries by Intensity**")
            if intensity_df is None or intensity_df.empty:
                st.info("No rivalry data yet.")
            else:
                full_rivalries = []

                # IMPORTANT:
                # - get_pair_intensity() is the single source of truth for the pair stats
                # - we pass df=intensity_df to avoid recomputing intensity_df inside stats_shared
                # - we pass matches_df=matches so season filtering matches the page
                for _, row in intensity_df.iterrows():
                    a_key = row.get("player_a", "")
                    b_key = row.get("player_b", "")

                    pair_stats = get_pair_intensity(
                        a_key,
                        b_key,
                        conn=conn,
                        df=intensity_df,
                        matches_df=matches,
                    ) or {}

                    full_rivalries.append(
                        {
                            "Player A": to_display(a_key, name_map),
                            "Player B": to_display(b_key, name_map),
                            "Matches": int(pair_stats.get("matches") or 0),
                            "Player A Wins": int(pair_stats.get("wins_a") or 0),
                            "Player B Wins": int(pair_stats.get("wins_b") or 0),
                            "Draws": int(pair_stats.get("draws") or 0),
                            "Avg Goal Diff": round(float(pair_stats.get("avg_goal_diff") or 0.0), 2),
                            "Intensity": round(float(pair_stats.get("intensity") or 0.0), 3),
                        }
                    )

                rival_display = pd.DataFrame(full_rivalries)[
                    ["Player A", "Player B", "Matches", "Player A Wins", "Player B Wins", "Draws", "Avg Goal Diff", "Intensity"]
                ].reset_index(drop=True)

                st.dataframe(rival_display, use_container_width=True, hide_index=True)

        st.divider()
        render_mmr_progression_over_time(suffix=suffix, season_mode=season_mode, selected_year=selected_year, season_start=season_start, matches=matches)

    finally:
        conn.close()


# ----------------------------
# 🎯 PLAYER INSIGHTS (NO VIDEO STATS)
# ----------------------------
def render_player_insights(suffix="", season_mode=None, selected_year=None, season_start=None, matches=None):
    st.markdown("## 🎯 Player Insights")

    conn = get_conn()
    try:
        players = pd.read_sql("SELECT id, name FROM players ORDER BY name", conn)
        matches = matches.copy() if matches is not None else pd.read_sql("SELECT * FROM matches WHERE processed=1", conn)
        mmr_hist = pd.read_sql(
            "SELECT mh.*, p.name FROM mmr_history mh JOIN players p ON mh.player_id=p.id ORDER BY mh.date ASC",
            conn,
        )

        mmr_hist["date_dt"] = pd.to_datetime(mmr_hist["date"], errors="coerce")
        if season_mode == "Single Year (season reset)" and selected_year is not None:
            mmr_hist = mmr_hist[mmr_hist["date_dt"].dt.year == int(selected_year)].copy() # type: ignore


        if matches.empty or players.empty:
            st.info("No processed matches / players found yet.")
            return

        name_map = get_name_map(conn)
        plist = players["name"].tolist()

        sel_player = st.selectbox(
            "Select a player to view detailed stats:",
            ["— Select —"] + plist,
            key=f"player_selectbox_{suffix}",
            format_func=lambda k: "— Select —" if k == "— Select —" else to_display(k, name_map),
        )

        if sel_player == "— Select —" or not sel_player:
            sel_key = str(sel_player).strip().lower()
            st.info("Select a player to view personal charts and tables.")
            return

        # Matches containing this player (safe contains)
        player_matches = matches[
            matches["team_a"].fillna("").astype(str).str.contains(sel_player, regex=False)
            | matches["team_b"].fillna("").astype(str).str.contains(sel_player, regex=False)
        ].copy()

        if player_matches.empty:
            st.info("No match data found for this player.")
            return

        total_matches = len(player_matches)

        # Win % (match results only)
        win_count = 0
        for _, m in player_matches.iterrows():
            ta = _split_team(m.get("team_a", ""))
            tb = _split_team(m.get("team_b", ""))
            res = str(m.get("result", "")).strip().upper()
            if (sel_player in ta and res == "A") or (sel_player in tb and res == "B"):
                win_count += 1

        win_pct = (win_count / total_matches * 100) if total_matches else 0.0

        # Player MMR history
        df_p = mmr_hist[mmr_hist["name"] == sel_player].copy().sort_values("date", ascending=True)

        # --------------------------
        # 📊 Performance Overview (MMR + Results only)
        # --------------------------
        st.subheader("📊 Performance Overview")

        current_mmr = None
        start_mmr = None
        net_mmr_change = 0.0
        avg_mmr_delta = 0.0

        if not df_p.empty:
            if season_mode == "Single Year (season reset)" and season_start:
                use_season_start = season_start
            else:
                use_season_start = None  # rolling
            pid = int(players[players["name"] == sel_player].iloc[0]["id"])

            if use_season_start:
                start_mmr = get_season_mmr(conn, pid, use_season_start, float(df_p.iloc[0]["mmr_before"]))
                current_mmr = get_season_mmr(conn, pid, use_season_start, float(df_p.iloc[-1]["mmr_after"]))
            else:
                start_mmr = float(df_p.iloc[0]["mmr_before"])
                current_mmr = float(df_p.iloc[-1]["mmr_after"])
            net_mmr_change = current_mmr - start_mmr

            df_p["mmr_delta"] = df_p["mmr_after"] - df_p["mmr_before"]
            avg_mmr_delta = float(df_p["mmr_delta"].mean()) if len(df_p) else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Matches Played", int(total_matches))
        c2.metric("Win %", f"{win_pct:.1f}%")
        if current_mmr is None:
            c3.metric("Current MMR", "—")
        else:
            c3.metric("Current MMR", f"{current_mmr:.0f}", f"{net_mmr_change:+.0f} vs start")

        c1, c2, c3 = st.columns(3)
        c1.metric("Wins", int(win_count))
        c2.metric("Losses/Draws", int(total_matches - win_count))
        c3.metric("Avg MMR Δ / Match", f"{avg_mmr_delta:+.2f}" if not df_p.empty else "—")

        st.caption("📌 This section uses match results + MMR history only (no video-tagged stats).")

        # --------------------------
        # 📈 MMR Over Time
        # --------------------------
        st.subheader("📈 MMR Over Time")
        if use_season_start:
            df_p["mmr_plot"] = df_p["mmr_after"].apply(lambda v: get_season_mmr(conn, pid, use_season_start, float(v)))
            ycol = "mmr_plot"
            ylab = "Season MMR"
        else:
            df_p["mmr_plot"] = df_p["mmr_after"].astype(float)
            ycol = "mmr_plot"
            ylab = "Rolling MMR"

        fig = px.line(
            df_p,
            x="date",
            y=ycol,
            markers=True,
            title=f"MMR Trend for {to_display(sel_player, name_map)}",
            labels={ycol: ylab},
        )
        st.plotly_chart(fig, use_container_width=True, key=f"player_mmr_chart_{sel_player}_{suffix}")

        # --------------------------
        # 🔥 Recent Form (Last 5 Matches) — outcomes + MMR only
        # --------------------------
        st.subheader("🔥 Recent Form (Last 5 Matches)")
        recent_hist = df_p.sort_values("date", ascending=False).head(5).copy()

        if recent_hist.empty:
            st.info("No recent matches found.")
        else:
            recent_rows = []
            form_icons = []

            for _, r in recent_hist.iterrows():
                mid = r["match_id"]
                m = matches[matches["id"] == mid]
                if m.empty:
                    continue

                mrow = m.iloc[0]
                ta = _split_team(mrow.get("team_a", ""))
                tb = _split_team(mrow.get("team_b", ""))
                res = str(mrow.get("result", "")).strip().upper()

                if (sel_player in ta and res == "A") or (sel_player in tb and res == "B"):
                    outcome = "Win"
                    icon = "🟩"
                elif res == "DRAW":
                    outcome = "Draw"
                    icon = "⬜"
                else:
                    outcome = "Loss"
                    icon = "🟥"

                mmr_delta = float(r["mmr_after"] - r["mmr_before"])
                # teammates / opponents
                my_team = ta if sel_player in ta else tb
                opp_team = tb if sel_player in ta else ta

                teammates = [p for p in my_team if p != sel_player]
                opponents = [p for p in opp_team if p]

                teammates_display = ", ".join(to_display(p, name_map) for p in teammates)
                opponents_display = ", ".join(to_display(p, name_map) for p in opponents)

                # score
                score_txt = str(mrow.get("score", "") or "").strip()

                recent_rows.append(
                    {
                        "Date": mrow.get("date", ""),
                        "Outcome": outcome,
                        "Score": score_txt,
                        "MMR Δ": round(mmr_delta, 1),
                        "Teammates": teammates_display,
                        "Opponents": opponents_display,
                    }
                )
                form_icons.append(icon)

            if recent_rows:
                st.markdown(f"**Recent Form:** {''.join(form_icons)}")

                df_recent = pd.DataFrame(recent_rows)
                emoji_map = {"Win": "🟩 Win", "Loss": "🟥 Loss", "Draw": "⬜ Draw"}
                df_recent["Outcome"] = df_recent["Outcome"].map(emoji_map)

                st.dataframe(
                    df_recent.sort_values("Date", ascending=False)
                    .set_index("Date")[["Outcome", "Score", "MMR Δ", "Teammates", "Opponents"]],
                    use_container_width=True,
                )
            else:
                st.info("Not enough matches to calculate form.")

        # -------------------------------------------------
        # 🤝 Best Teammates (Chemistry) — display names
        # -------------------------------------------------
        st.markdown("### 🤝 Best Teammates (Chemistry)")
        chem_df = get_chemistry_df(conn, matches_df=matches)

        player_chem = chem_df[
            (chem_df["player_a"] == sel_player) | (chem_df["player_b"] == sel_player)
        ].copy() if not chem_df.empty else pd.DataFrame()

        if not player_chem.empty:
            player_chem["Teammate"] = player_chem.apply(
                lambda r: r["player_b"] if r["player_a"] == sel_player else r["player_a"],
                axis=1,
            )
            player_chem["Teammate"] = player_chem["Teammate"].apply(lambda k: to_display(k, name_map))

            st.dataframe(
                player_chem[["Teammate", "matches", "wins", "win_pct", "chemistry"]]
                .rename(
                    columns={
                        "matches": "Games",
                        "wins": "Wins",
                        "win_pct": "Win %",
                        "chemistry": "Chemistry",
                    }
                )
                .sort_values("Chemistry", ascending=False)
                .head(10)
                .set_index("Teammate"),
                use_container_width=True,
            )
        else:
            st.info("No chemistry data yet.")

        # -------------------------------------------------
        # ⚔️ Toughest Opponents (Intensity) — display names
        # -------------------------------------------------
        st.markdown("### ⚔️ Toughest Opponents (Intensity)")

        # 🔑 normalised key for safe comparison
        sel_key = str(sel_player).strip().lower()

        # ✅ season-filtered intensity table
        int_df = get_intensity_df(conn, matches_df=matches)

        player_int = (
            int_df[
                (int_df["player_a"].astype(str).str.strip().str.lower() == sel_key)
                | (int_df["player_b"].astype(str).str.strip().str.lower() == sel_key)
            ].copy()
            if int_df is not None and not int_df.empty
            else pd.DataFrame()
        )

        if not player_int.empty:
            rows = []
            for _, r in player_int.iterrows():
                opponent = r["player_b"] if str(r["player_a"]).strip().lower() == sel_key else r["player_a"]
                stats = get_pair_intensity(sel_player, opponent, conn=conn, df=int_df, matches_df=matches)
                games = stats.get("matches", 0)
                if games == 0:
                    continue

                wins = stats.get("wins_a", 0)     # wins for sel_player (orientation assumed by helper)
                losses = stats.get("wins_b", 0)   # wins for opponent
                win_pct_vs = round((wins / games * 100), 1) if games else 0.0

                rows.append(
                    {
                        "Opponent": to_display(opponent, name_map),
                        "Games": games,
                        "Wins": wins,
                        "Losses": losses,
                        "W%": win_pct_vs,
                        "Intensity": round(stats.get("intensity", 0.0), 3),
                    }
                )

            df_display = pd.DataFrame(rows).sort_values("Intensity", ascending=False)
            st.dataframe(
                df_display[["Opponent", "Games", "Wins", "Losses", "W%", "Intensity"]].set_index("Opponent"),
                use_container_width=True,
            )
        else:
            st.info("No rivalry data yet.")
    finally:
        conn.close()


# ----------------------------
# ⚔️ Head-to-Head & 🤝 Duo Chemistry (Final Styled) — rewritten
# ----------------------------
def render_head_to_head_section(season_mode=None, selected_year=None, season_start=None, matches=None):
    st.markdown("## ⚔️ Head-to-Head & 🤝 Duo Chemistry")

    conn = get_conn()
    try:
        # --- Load matches (use passed filter if provided) ---
        if matches is not None:
            matches_df = matches.copy()
        else:
            matches_df = pd.read_sql("SELECT * FROM matches WHERE processed=1", conn)

        # Safety: ensure dataframe exists
        if matches_df is None:
            matches_df = pd.DataFrame()

        # --- Players / name map ---
        players = pd.read_sql("SELECT id, name FROM players ORDER BY name", conn)
        name_map = get_name_map(conn)
        all_players = sorted(players["name"].dropna().astype(str).tolist())
        select_options = ["— Select —"] + all_players

        # --- UI Selectors (same UI) ---
        c1, c2 = st.columns(2)
        with c1:
            player_a = st.selectbox(
                "Player A",
                select_options,
                index=0,
                key="h2h_player_a",
                format_func=lambda k: "— Select —" if k == "— Select —" else to_display(k, name_map),
            )
        with c2:
            player_b = st.selectbox(
                "Player B",
                select_options,
                index=0,
                key="h2h_player_b",
                format_func=lambda k: "— Select —" if k == "— Select —" else to_display(k, name_map),
            )

        if player_a == "— Select —" or player_b == "— Select —" or player_a == player_b:
            st.info("Select two players to view their head-to-head stats.")
            return

        # --- Robust normalisation helpers ---
        def _norm(s: str) -> str:
            # Lowercase, trim, remove invisible chars, collapse whitespace
            if s is None:
                return ""
            s = str(s)
            s = s.replace("\u00A0", " ")   # non-breaking space
            s = s.replace("\u200B", "")    # zero-width space
            s = s.replace("\u200C", "")
            s = s.replace("\u200D", "")
            s = s.strip().lower()
            # collapse internal whitespace
            s = " ".join(s.split())
            return s

        def _split_team(val):
            """
            Handles:
            - "['SAM K', 'BILLY']"
            - "sam k, billy"
            - None / empty
            """
            if val is None:
                return []

            s = str(val).strip().lower()

            # Remove list-like wrappers
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]

            # Remove quotes
            s = s.replace("'", "").replace('"', "")

            # Normalise separators
            for sep in [";", "|", "/"]:
                s = s.replace(sep, ",")

            return [p.strip() for p in s.split(",") if p.strip()]

        def _score_to_ints(sc):
            try:
                if isinstance(sc, str) and "-" in sc:
                    a, b = sc.split("-", 1)
                    return int(a.strip()), int(b.strip())
            except Exception:
                pass
            return None, None

        player_a_key = _norm(player_a)
        player_b_key = _norm(player_b)

        # If matches_df empty, still render UI but show zeros
        if matches_df.empty:
            pair_chem = {"matches": 0, "wins": 0, "chemistry": 0.0}
            pair_int = {"matches": 0, "wins_a": 0, "wins_b": 0, "draws": 0, "avg_goal_diff": 0.0, "intensity": 0.0}
            chem_df_local = pd.DataFrame({"chemistry": []})
            int_df_local = pd.DataFrame({"intensity": []})
        else:
            # --- Build chemistry / intensity tables from the SAME matches_df (season-aware) ---
            chem_df_local = get_chemistry_df(conn, matches_df=matches_df)
            int_df_local = get_intensity_df(conn, matches_df=matches_df)

            # --- Counts from matches_df (this is the source of truth for filters) ---
            together = 0
            wins_together = 0

            faced = 0
            wins_a = 0
            wins_b = 0
            draws = 0
            gd_list = []

            # Ensure required columns exist (avoid silent failures)
            # Expected: team_a, team_b, result, score
            for _, m in matches_df.iterrows():
                ta = _split_team(m.get("team_a", ""))
                tb = _split_team(m.get("team_b", ""))
                res = _norm(m.get("result", "")).upper()  # "A", "B", "DRAW"
                # Normalize "draw" variants
                if res == "D":
                    res = "DRAW"

                # together
                if player_a_key in ta and player_b_key in ta:
                    together += 1
                    if res == "A":
                        wins_together += 1
                elif player_a_key in tb and player_b_key in tb:
                    together += 1
                    if res == "B":
                        wins_together += 1

                # head-to-head
                a_vs_b = (player_a_key in ta and player_b_key in tb)
                b_vs_a = (player_a_key in tb and player_b_key in ta)
                if a_vs_b or b_vs_a:
                    faced += 1

                    a_sc, b_sc = _score_to_ints(m.get("score", ""))
                    if a_sc is not None and b_sc is not None:
                        gd_list.append(abs(a_sc - b_sc))

                    if res == "DRAW":
                        draws += 1
                    else:
                        # Determine which side player_a was on and whether that side won
                        player_a_on_a = player_a_key in ta
                        if (player_a_on_a and res == "A") or ((not player_a_on_a) and res == "B"):
                            wins_a += 1
                        else:
                            wins_b += 1

            pair_chem = {"matches": together, "wins": wins_together, "chemistry": 0.0}
            pair_int = {
                "matches": faced,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "draws": draws,
                "avg_goal_diff": (sum(gd_list) / len(gd_list)) if gd_list else 0.0,
                "intensity": 0.0,
            }

            # --- Pull chemistry/intensity SCORE from tables (optional) ---
            # Chemistry
            try:
                chem_score = get_pair_chemistry(player_a_key, player_b_key, conn, df=chem_df_local)
                if chem_score.get("matches", 0) == 0:
                    chem_score = get_pair_chemistry(player_b_key, player_a_key, conn, df=chem_df_local)
                pair_chem["chemistry"] = float(chem_score.get("chemistry", 0.0))
            except Exception:
                pair_chem["chemistry"] = 0.0

            # Intensity
            try:
                int_score = get_pair_intensity(player_a_key, player_b_key, conn, df=int_df_local)
                if int_score.get("matches", 0) == 0:
                    alt = get_pair_intensity(player_b_key, player_a_key, conn, df=int_df_local)
                    if alt.get("matches", 0) > 0:
                        int_score = {"intensity": alt.get("intensity", 0.0)}
                pair_int["intensity"] = float(int_score.get("intensity", 0.0))
            except Exception:
                pair_int["intensity"] = 0.0

        # --- Display names (same UI) ---
        a_disp = to_display(player_a, name_map)
        b_disp = to_display(player_b, name_map)

        # ------------------------------
        # ⚔️ Rivalry Section
        # ------------------------------
        st.markdown(f"### ⚔️ Head-to-Head: {a_disp} vs {b_disp}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Matches Played", pair_int.get("matches", 0))
        with c2:
            st.metric(
                f"{a_disp} Wins",
                pair_int.get("wins_a", 0),
                f"{(pair_int.get('wins_a', 0) / pair_int['matches'] * 100):.1f}%" if pair_int.get("matches") else None,
            )
        with c3:
            st.metric(
                f"{b_disp} Wins",
                pair_int.get("wins_b", 0),
                f"{(pair_int.get('wins_b', 0) / pair_int['matches'] * 100):.1f}%" if pair_int.get("matches") else None,
            )

        draws_v = pair_int.get("draws", 0)
        avg_gd = pair_int.get("avg_goal_diff", 0.0)
        int_val = pair_int.get("intensity", None)
        int_display = "—" if int_val is None else f"{float(int_val):.2f}"

        st.markdown(
            f"Draws: {draws_v}  |  Avg Goal Diff: {avg_gd:.2f}  |  Intensity Score: {int_display}"
        )

        # --- Rivalry Badge ---
        all_int = []
        if int_df_local is not None and not int_df_local.empty and "intensity" in int_df_local.columns:
            all_int = int_df_local["intensity"].dropna().tolist()

        int_val = pair_int.get("intensity", None)

        if int_val is None or not all_int:
            percentile_r = 0.0
        else:
            # ensure list is numeric
            all_int_num = [float(x) for x in all_int if x is not None]
            rank_r = sum(s < float(int_val) for s in all_int_num)
            percentile_r = (rank_r / len(all_int_num) * 100) if all_int_num else 0.0

        def rivalry_label(p):
            if p >= 90:
                return ("🟩", "Legendary Rivalry")
            elif p >= 70:
                return ("🟦", "Fierce Rivalry")
            elif p >= 40:
                return ("🟨", "Developing Rivalry")
            elif p >= 10:
                return ("🟧", "Minor Rivalry")
            else:
                return ("🟥", "Cold Rivalry")

        color, label = rivalry_label(percentile_r)
        st.markdown(
            f"""
        <div style="text-align:center;margin-top:10px;margin-bottom:10px;">
            <span style="font-size:20px;">{color} <b>{label}</b></span><br>
            <span style="font-size:15px;color:gray;">{percentile_r:.1f}ᵗʰ percentile among {len(all_int)} rivalries</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # ------------------------------
        # 🤝 Partnership Section
        # ------------------------------
        st.markdown(f"### 🤝 Partnership: {a_disp} & {b_disp}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Matches Together", pair_chem.get("matches", 0))
        with c2:
            live_win_pct = (pair_chem.get("wins", 0) / pair_chem["matches"] * 100) if pair_chem.get("matches") else 0
            st.metric("Wins Together", pair_chem.get("wins", 0), f"{live_win_pct:.1f}%")
        with c3:
            chem_val = pair_chem.get("chemistry", None)
            chem_display = "—" if chem_val is None else f"{chem_val:.2f}"
            st.metric("Chemistry Score", chem_display)


        losses = pair_chem.get("matches", 0) - pair_chem.get("wins", 0)
        draws_p = 0
        st.markdown(f"Draws: {draws_p}  |  Losses: {losses}")

        # --- Partnership Badge ---
        all_chem = []
        if chem_df_local is not None and not chem_df_local.empty and "chemistry" in chem_df_local.columns:
            all_chem = chem_df_local["chemistry"].dropna().tolist()

        rank_d = sum(s < pair_chem.get("chemistry", 0.0) for s in all_chem)
        percentile_d = (rank_d / len(all_chem) * 100) if all_chem else 0.0

        def partnership_label(p):
            if p >= 90:
                return ("🟩", "Elite Partnership")
            elif p >= 70:
                return ("🟦", "Strong Partnership")
            elif p >= 40:
                return ("🟨", "Developing Partnership")
            elif p >= 10:
                return ("🟧", "Needs Work")
            else:
                return ("🟥", "Poor Connection")

        color, label = partnership_label(percentile_d)
        st.markdown(
            f"""
        <div style="text-align:center;margin-top:10px;">
            <span style="font-size:20px;">{color} <b>{label}</b></span><br>
            <span style="font-size:15px;color:gray;">{percentile_d:.1f}ᵗʰ percentile among {len(all_chem)} duos</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.divider()

    finally:
        conn.close()


# ----------------------------
# 📊 PAGE COMPOSER
# ----------------------------
def render_charts_page():
    # Match Dashboard header styling / spacing
    st.set_page_config(page_title="Charts & Stats | Love Five-A-Side", layout="wide")
    page_header(
        "Charts & Stats",
        "Explore season trends, player insights and duo chemistry",
        center=True,
        divider=True,
    )

    conn = get_conn()
    try:
        matches_all = pd.read_sql("SELECT * FROM matches WHERE processed=1 ORDER BY date ASC", conn)
    finally:
        conn.close()

    if matches_all.empty:
        st.info("No processed matches yet.")
        return

    season_mode, selected_year, season_start, matches_filtered = get_season_filter_ui(matches_all, suffix="top")

    st.divider()

    with st.expander("🌍 Global Overview", expanded=False):
        render_global_overview("_exp", season_mode, selected_year, season_start, matches_filtered)

    with st.expander("🎯 Player Insights", expanded=False):
        render_player_insights("_exp", season_mode, selected_year, season_start, matches_filtered)

    with st.expander("⚔️ Head-to-Head & 🤝 Duo Chemistry", expanded=False):
        render_head_to_head_section(season_mode, selected_year, season_start, matches_filtered)
