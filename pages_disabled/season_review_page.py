import streamlit as st
import pandas as pd
import datetime
from pathlib import Path
from utils.ui_components import page_header
from utils.names import player_display_name

from utils.db_utils import get_conn, get_current_league_id, backup_db_manual, STARTING_MMR
from utils.relationships_utils import calculate_chemistry_for_all_duos, calculate_rivalry_intensity


def compute_result_from_score(score: str):
    s = str(score or "").strip().replace("–", "-").replace("—", "-")
    if "-" not in s:
        return None
    try:
        a, b = s.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
    except Exception:
        return None
    if a > b:
        return "A"
    if b > a:
        return "B"
    return "DRAW"


def quarter_date_range(year, quarter):
    """Return (start_date, end_date) for a given quarter or 'Full'."""
    if str(quarter).lower() == "full":
        return f"{year}-01-01", f"{year}-12-31"
    q = str(quarter).strip().upper()
    if q == "Q1":
        return f"{year}-01-01", f"{year}-03-31"
    if q == "Q2":
        return f"{year}-04-01", f"{year}-06-30"
    if q == "Q3":
        return f"{year}-07-01", f"{year}-09-30"
    if q == "Q4":
        return f"{year}-10-01", f"{year}-12-31"
    # fallback
    return f"{year}-01-01", f"{year}-12-31"




def _build_display_name_map(players_df: pd.DataFrame) -> dict[str, str]:
    """
    Map canonical player key (UPPER) -> display name.
    Falls back to `name` if display_name isn't present.
    """
    if players_df is None or players_df.empty:
        return {}
    name_col = "name" if "name" in players_df.columns else None
    if not name_col:
        return {}

    disp_col = "display_name" if "display_name" in players_df.columns else None
    out = {}
    for _, r in players_df.iterrows():
        base = str(r.get(name_col, "")).strip()
        if not base:
            continue
        disp = str(r.get(disp_col, "")).strip() if disp_col else ""
        out[base.upper()] = player_display_name(base, disp)
    return out


def _pretty_team(team_list: list[str], disp_map: dict[str, str]) -> str:
    # Sort for stable display
    pretty = [disp_map.get(str(p).upper(), str(p)) for p in (team_list or [])]
    return ", ".join(pretty)

def _available_season_years() -> list[int]:
    """
    Return a sorted list of available season years based on:
    - Archive/DB_YYYY folders
    - Current year as a fallback
    """
    years = set()
    today = datetime.datetime.today().year

    try:
        here = Path(__file__).resolve()
    except Exception:
        here = Path.cwd()

    # Walk up a few levels to find Archive/DB_YYYY folders
    for base in [here.parent, *list(here.parents)[:6]]:
        arch = base / "Archive"
        if not arch.exists():
            continue
        for p in arch.iterdir():
            if p.is_dir() and p.name.upper().startswith("DB_"):
                try:
                    y = int(p.name.split("_", 1)[1])
                    years.add(y)
                except Exception:
                    pass

    years.add(today)
    return sorted(years, reverse=True)



def render_season_review_page():
    page_header("Season Review", "Quarterly summaries and awards", center=True, divider=True)
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)

    today = datetime.datetime.today()
    available_years = _available_season_years()

    year = st.selectbox(
        "Season Year",
        available_years,
        index=0 if today in available_years else 0,
    )
    quarter = st.selectbox("Quarter", ["Full Year", "Q1", "Q2", "Q3", "Q4"], index=0)

    if quarter == "Full Year":
        start_date, end_date = f"{year}-01-01", f"{year}-12-31"
    else:
        start_date, end_date = quarter_date_range(int(year), quarter)

    league_id = get_current_league_id()
    conn = get_conn()

    players = pd.read_sql(
        "SELECT id, name, display_name, mmr FROM public.players WHERE league_id = %s",
        conn,
        params=(league_id,),
    )

    matches = pd.read_sql(
        """
        SELECT id, date, team_a, team_b, score, result
        FROM public.matches
        WHERE league_id = %s
          AND processed = 1
          AND date >= %s
          AND date <= %s
        ORDER BY date ASC, id ASC
        """,
        conn,
        params=(league_id, start_date, end_date),
    )

    conn.close()

    disp_map = _build_display_name_map(players)

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

    def filter_by_period(df, start, end, date_col):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        mask = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
        return df.loc[mask].copy()

    matches_q = filter_by_period(matches, start_date, end_date, "date")

    # --- Season summary (based on match score, NOT video moments)
    def score_to_ints(sc):
        try:
            if isinstance(sc, str) and "-" in sc:
                a, b = sc.split("-", 1)
                return int(a.strip()), int(b.strip())
        except Exception:
            pass
        return None, None

    total_games = len(matches_q)
    goals_total = []
    for _, m in matches_q.iterrows():
        a, b = score_to_ints(m.get("score", ""))
        if a is None:
            continue
        goals_total.append(a + b)
    avg_goals = round(sum(goals_total) / len(goals_total), 2) if goals_total else 0.0

    summary_df = pd.DataFrame(
        [
            {"Metric": "Season Year", "Value": year},
            {"Metric": "Quarter", "Value": quarter},
            {"Metric": "Date Range", "Value": f"{start_date} to {end_date}"},
            {"Metric": "Total Processed Matches", "Value": total_games},
            {"Metric": "Avg Goals/Game", "Value": avg_goals},
        ]
    )

    st.subheader("Season Summary")
    st.table(summary_df.set_index("Metric"))

    st.subheader("Current Top 5 MMR")
    top_current_df = (
        players[["name", "mmr"]]
        .rename(columns={"name": "Player", "mmr": "MMR"})
        .sort_values("MMR", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    top_current_df["Player"] = top_current_df["Player"].astype(str).apply(lambda x: disp_map.get(x.upper(), x))
    top_current_df.index = top_current_df.index + 1
    st.dataframe(top_current_df)

    # MMR history and relationship awards are the expensive part of this page
    # online, so the page now renders its useful summary before pulling them.
    if not st.toggle(
        "Load full awards and detail",
        value=False,
        key="season_review_load_full_detail",
        help="Loads MMR movement, rivalries, duos and the full awards tables.",
    ):
        st.info("Open the full detail when you want the heavier season awards and relationship tables.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    conn = get_conn()
    mh = pd.read_sql(
        """
        SELECT mh.match_id, mh.date, mh.mmr_before, mh.mmr_after, p.name, m.date AS match_date
        FROM public.mmr_history mh
        JOIN public.players p ON mh.player_id = p.id
        LEFT JOIN public.matches m ON mh.match_id = m.id
        WHERE mh.league_id = %s
          AND COALESCE(m.date, mh.date) >= %s
          AND COALESCE(m.date, mh.date) <= %s
        ORDER BY mh.id
        """,
        conn,
        params=(league_id, start_date, end_date),
    )
    conn.close()
    mh_date_col = "match_date" if (not mh.empty and "match_date" in mh.columns) else "date"

    # --- Top 5 Players by MMR
    st.subheader("Top 5 Players by MMR (All Time)")
    top_mmr_df = pd.DataFrame(columns=["Player", "MMR"])
    if not mh.empty:
        mh[mh_date_col] = pd.to_datetime(mh[mh_date_col], errors="coerce")
        mh_q = mh[(mh[mh_date_col] >= pd.to_datetime(start_date)) & (mh[mh_date_col] <= pd.to_datetime(end_date))].copy()
        if not mh_q.empty:
            last_by_player = mh_q.sort_values(mh_date_col).groupby("name").tail(1)
            top_mmr_df = (
                last_by_player[["name", "mmr_after"]]
                .rename(columns={"name": "Player", "mmr_after": "MMR"})
                .sort_values("MMR", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )

    if top_mmr_df.empty:
        top_mmr_df = (
            players[["name", "mmr"]]
            .rename(columns={"name": "Player", "mmr": "MMR"})
            .sort_values("MMR", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )

    top_mmr_df["Player"] = top_mmr_df["Player"].astype(str).apply(lambda x: disp_map.get(x.upper(), x))
    top_mmr_df.index = top_mmr_df.index + 1
    st.dataframe(top_mmr_df)

    # --- Top 5 Most Improved
    st.subheader("Top 5 Most Improved (MMR Δ in period)")
    most_imp_df = pd.DataFrame(columns=["Player", "Δ MMR"])
    if not mh.empty:
        mh[mh_date_col] = pd.to_datetime(mh[mh_date_col], errors="coerce")
        mh_q = mh[(mh[mh_date_col] >= pd.to_datetime(start_date)) & (mh[mh_date_col] <= pd.to_datetime(end_date))].copy()
        if not mh_q.empty:
            by_name = mh_q.sort_values(mh_date_col).groupby("name")
            first = by_name.head(1).set_index("name")["mmr_before"]
            last = by_name.tail(1).set_index("name")["mmr_after"]
            delta = (last - first).dropna().sort_values(ascending=False)
            most_imp_df = pd.DataFrame({"Player": delta.index, "Δ MMR": delta.values}).head(5).reset_index(drop=True)

    if most_imp_df.empty:
        with st.expander("Why am I seeing no MMR changes?"):
            if mh.empty:
                st.write("mmr_history table is empty in the selected database.")
            else:
                # Show quick diagnostics about available dates
                if mh_date_col in mh.columns:
                    dt = pd.to_datetime(mh[mh_date_col], errors="coerce")
                    st.write(f"Using `{mh_date_col}` to filter mmr_history.")
                    st.write({
                        "rows_in_mmr_history": int(len(mh)),
                        "rows_with_valid_dates": int(dt.notna().sum()),
                        "min_date": None if dt.dropna().empty else str(dt.dropna().min()),
                        "max_date": None if dt.dropna().empty else str(dt.dropna().max()),
                        "selected_start": str(start_date),
                        "selected_end": str(end_date),
                    })
                else:
                    st.write(f"No `{mh_date_col}` column available in mmr_history.")
        st.info("No MMR changes in this period.")
    else:
        most_imp_df["Player"] = most_imp_df["Player"].astype(str).apply(lambda x: disp_map.get(x.upper(), x))
        most_imp_df.index = most_imp_df.index + 1
        st.dataframe(most_imp_df)

    # --- Heaviest Defeats
    st.subheader("Heaviest Defeats")
    hd_rows = []
    for _, m in matches_q.iterrows():
        a, b = score_to_ints(m.get("score", ""))
        if a is None:
            continue
        gd = abs(a - b)

        team_a_raw = [p.strip().upper() for p in str(m.get("team_a", "") or "").split(",") if p.strip()]
        team_b_raw = [p.strip().upper() for p in str(m.get("team_b", "") or "").split(",") if p.strip()]

        hd_rows.append(
            {
                "Date": str(m.get("date", "")),
                "Team A": _pretty_team(team_a_raw, disp_map),
                "Team B": _pretty_team(team_b_raw, disp_map),
                "Score": m.get("score", ""),
                "Goal Diff": gd,
            }
        )
    df_hd = pd.DataFrame(hd_rows).sort_values("Goal Diff", ascending=False).head(5).reset_index(drop=True)
    df_hd.index = df_hd.index + 1
    st.dataframe(df_hd)

    # --- Normalize score dashes and whitespace
    matches_q["score"] = (
        matches_q["score"].astype(str)
        .str.replace("–", "-", regex=False)
        .str.replace("—", "-", regex=False)
        .str.strip()
    )

    # --- Canonical result from score
    matches_q["result"] = matches_q["score"].apply(compute_result_from_score)

    # --- Parse teams into normalized name lists (UPPER canonical keys)
    def normalize_name_list(s):
        return [p.strip().upper() for p in str(s or "").split(",") if p.strip()]

    matches_q["team_a"] = matches_q["team_a"].apply(normalize_name_list)
    matches_q["team_b"] = matches_q["team_b"].apply(normalize_name_list)

    # --- Drop any rows with missing team names (safety)
    matches_q = matches_q[matches_q["team_a"].notna() & matches_q["team_b"].notna()].copy()

    # --- Rivalries by Intensity (depth-weighted)
    st.subheader("Top 5 Rivalries by Intensity")

    df_rivals = calculate_rivalry_intensity(matches_q)
    if df_rivals is None or df_rivals.empty:
        st.info("No rivalries available for this period.")
    else:
        df_rivals = df_rivals.head(5).reset_index(drop=True)
        df_rivals.index = df_rivals.index + 1
        for col in df_rivals.columns:
            if "player" in col.lower() or "a" == col.lower() or "b" == col.lower():
                df_rivals[col] = df_rivals[col].astype(str).apply(lambda x: disp_map.get(x.upper(), x))
        st.dataframe(df_rivals)

    # --- Duos by Chemistry (depth-weighted)
    st.subheader("Top 10 Duos by Chemistry")

    df_chem = calculate_chemistry_for_all_duos(matches_q)
    if df_chem is None or df_chem.empty:
        st.info("No duo data available for this period.")
    else:
        df_chem = df_chem.head(10).reset_index(drop=True)
        df_chem.index = df_chem.index + 1
        for col in df_chem.columns:
            if "player" in col.lower() or "a" == col.lower() or "b" == col.lower():
                df_chem[col] = df_chem[col].astype(str).apply(lambda x: disp_map.get(x.upper(), x))
        st.dataframe(df_chem)

    # =========================================================
    # 🥀 Season to Forget (Minimum 10 Games) — NO video stats
    # =========================================================
    st.subheader("Period to Forget")
    st.caption("Thanks for turning up. It didn’t go your way — but things can only get better.")

    def _score_to_ints(sc):
        try:
            if isinstance(sc, str) and "-" in sc:
                a, b = sc.split("-", 1)
                return int(a.strip()), int(b.strip())
        except Exception:
            pass
        return None, None

    # base per-player match stats from matches_q (results + scored/conceded from match score only)
    stats = {}  # keyed by UPPER player name

    for _, m in matches_q.iterrows():
        team_a = m["team_a"]
        team_b = m["team_b"]
        res = m.get("result", "")
        a_goals, b_goals = _score_to_ints(m.get("score", ""))

        if res == "A":
            a_out, b_out = "W", "L"
        elif res == "B":
            a_out, b_out = "L", "W"
        else:
            a_out, b_out = "D", "D"

        for p in team_a:
            stats.setdefault(p, {"Games": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0})
            stats[p]["Games"] += 1
            stats[p][a_out] += 1
            if a_goals is not None:
                stats[p]["GF"] += a_goals
                stats[p]["GA"] += b_goals

        for p in team_b:
            stats.setdefault(p, {"Games": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0})
            stats[p]["Games"] += 1
            stats[p][b_out] += 1
            if a_goals is not None:
                stats[p]["GF"] += b_goals
                stats[p]["GA"] += a_goals

    for p, s in stats.items():
        s["GD"] = s["GF"] - s["GA"]
        s["Points"] = (3 * s["W"]) + (1 * s["D"])
        s["Win%"] = (s["W"] / s["Games"]) if s["Games"] else 0.0

    # MMR start/end/delta from mmr_history in this period (canonical UPPER)
    mh_tmp = mh.copy()
    mh_tmp[mh_date_col] = pd.to_datetime(mh_tmp[mh_date_col], errors="coerce")
    mh_tmp["name"] = mh_tmp["name"].astype(str).str.upper()

    mh_q2 = mh_tmp[
        (mh_tmp[mh_date_col] >= pd.to_datetime(start_date)) &
        (mh_tmp[mh_date_col] <= pd.to_datetime(end_date))
    ].copy()

    mmr_start = {}
    mmr_end = {}
    mmr_delta = {}

    if not mh_q2.empty:
        by_name = mh_q2.sort_values(mh_date_col).groupby("name")
        start_series = by_name.head(1).set_index("name")["mmr_before"]
        end_series = by_name.tail(1).set_index("name")["mmr_after"]

        mmr_start = start_series.dropna().to_dict()
        mmr_end = end_series.dropna().to_dict()

        for n in set(list(mmr_start.keys()) + list(mmr_end.keys())):
            mmr_delta[n] = float(mmr_end.get(n, mmr_start.get(n, STARTING_MMR))) - float(
                mmr_start.get(n, STARTING_MMR)
            )

    # Build table (min 10 games)
    rows = []
    for player, s in stats.items():
        games = s["Games"]
        if games < 10:
            continue

        ms = mmr_start.get(player, None)
        me = mmr_end.get(player, None)
        dmmr = mmr_delta.get(player, 0.0)

        # Harsh score: lower = worse season
        # Uses only non-video data: points, goal difference, and MMR delta (small)
        forget_score = (
            (s["Points"] * 1.0)
            + (s["GD"] * 0.35)
            + (dmmr * 0.15)
        )

        rows.append({
            "Player": disp_map.get(player.upper(), player),
            "Games": games,
            "W": s["W"],
            "D": s["D"],
            "L": s["L"],
            "Points": s["Points"],
            "Win%": round(s["Win%"] * 100, 1),
            "GF": s["GF"],
            "GA": s["GA"],
            "GD": s["GD"],
            "MMR Start": (round(ms, 1) if ms is not None else None),
            "MMR End": (round(me, 1) if me is not None else None),
            "MMR Δ": round(dmmr, 1),
            "Season to Forget Score": round(forget_score, 2),
        })

    df_forget = pd.DataFrame(rows)

    if df_forget.empty:
        st.info("No players met the 10-game minimum for this award.")
    else:
        df_forget = df_forget.sort_values("Season to Forget Score", ascending=True).reset_index(drop=True)
        df_forget.index = df_forget.index + 1
        st.dataframe(df_forget.head(3))

        with st.expander("See full eligible table (10+ games)"):
            st.dataframe(df_forget)

    st.markdown("</div>", unsafe_allow_html=True)


# CLI entrypoint
if __name__ == "__main__":
    import sys

    args = set(a.lower() for a in sys.argv[1:])
    if args:
        print("v33 maintenance mode — args:", args)
        backup = backup_db_manual()
        if backup:
            print(f"Database backed up to: {backup}")
        sys.exit(0)
