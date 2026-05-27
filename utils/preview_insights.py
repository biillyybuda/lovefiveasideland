import pandas as pd
import streamlit as st
import numpy as np
import re
from utils.relationships_utils import calculate_chemistry_for_all_duos, calculate_rivalry_intensity
from utils.team_ai_engine import clean_name


# -------------------------------
# Chemistry helper (same formula used on Player Relationships page)
# -------------------------------
def _chemistry_score(matches_together, win_pct, closeness):
    # closeness = average margin inverted (closer games = higher score)
    return (matches_together * win_pct * closeness) / 100


# -------------------------------
# Normalisation helpers
# -------------------------------
def _norm_list(players):
    return [clean_name(p) for p in (players or []) if str(p).strip()]

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column matching any candidate (case/space/underscore insensitive)."""
    if df is None or df.empty:
        return None
    norm_map = {re.sub(r"[\s_]+", "", str(c).strip().lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[\s_]+", "", str(cand).strip().lower())
        if key in norm_map:
            return norm_map[key]
    return None

def _norm_col(df: pd.DataFrame, candidates: list[str], out_col: str):
    """Create out_col as clean_name() of the first matching candidate column."""
    if df is None or df.empty:
        return df
    col = _find_col(df, candidates)
    if not col:
        return df
    df[out_col] = df[col].astype(str).apply(clean_name)
    return df


# -------------------------------
# 1️⃣  Key Matchups  (now using relationships_utils)
# -------------------------------
def _get_key_matchups(team_a, team_b, conn=None, matches_df=None):
    """Generate rivalry insight table using shared depth-weighted logic."""
    team_a_n = _norm_list(team_a)
    team_b_n = _norm_list(team_b)
    if matches_df is None:
        matches_df = _processed_matches_for_current_league()

    df_rivals_all = calculate_rivalry_intensity(matches_df)

    # Normalise player keys for matching
    df_rivals_all = _norm_col(df_rivals_all, ["Player A","player a","player_a","A","player1","player 1","p1"], "_A")
    df_rivals_all = _norm_col(df_rivals_all, ["Player B","player b","player_b","B","player2","player 2","p2"], "_B")

    # Defensive: if None or empty, return blank DataFrame
    if df_rivals_all is None or df_rivals_all.empty:
        return pd.DataFrame(columns=["Player A", "Player B", "Matches", "Intensity"])

    # Ensure normalised columns exist
    if "_A" not in df_rivals_all.columns or "_B" not in df_rivals_all.columns:
        df_rivals_all = _norm_col(df_rivals_all, ["Player A","player a","player_a","A","player1","player 1","p1"], "_A")
        df_rivals_all = _norm_col(df_rivals_all, ["Player B","player b","player_b","B","player2","player 2","p2"], "_B")

    # Only include cross-team rivalries between the two squads
    df_rivals = df_rivals_all[
        (df_rivals_all["_A"].isin(team_a_n) & df_rivals_all["_B"].isin(team_b_n))
        | (df_rivals_all["_A"].isin(team_b_n) & df_rivals_all["_B"].isin(team_a_n))
    ].copy()

    if df_rivals.empty:
        return pd.DataFrame(columns=["Player A", "Player B", "Matches", "Intensity"])

    return (
        df_rivals.sort_values("Intensity", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


# -------------------------------
# 2️⃣  Best Teammates  (now using relationships_utils)
# -------------------------------
def _get_best_teammates(team_a, team_b, conn=None, matches_df=None):
    """Generate teammate chemistry tables using shared depth-weighted logic."""
    team_a_n = _norm_list(team_a)
    team_b_n = _norm_list(team_b)
    if matches_df is None:
        matches_df = _processed_matches_for_current_league()
    df_chem_all = calculate_chemistry_for_all_duos(matches_df)

    # Normalise player keys for matching
    df_chem_all = _norm_col(df_chem_all, ["Player A","player a","player_a","A","player1","player 1","p1"], "_A")
    df_chem_all = _norm_col(df_chem_all, ["Player B","player b","player_b","B","player2","player 2","p2"], "_B")

    # --- Combine reversed duplicates (e.g. Billy–Jacob & Jacob–Billy) ---
    df_chem_all["pair_key"] = df_chem_all.apply(
        lambda x: "-".join(sorted([x["_A"], x["_B"]])), axis=1
    )
    df_chem_all = (
        df_chem_all.groupby("pair_key", as_index=False)
        .agg({
            "Player A": "first",
            "Player B": "first",
            "Matches": "sum",
            "Wins": "sum",
            "Win %": "mean",
            "Chemistry": "mean"
        })
        .drop(columns=["pair_key"])
    )

    # Ensure normalised columns exist
    if "_A" not in df_chem_all.columns or "_B" not in df_chem_all.columns:
        df_chem_all = _norm_col(df_chem_all, ["Player A","player a","player_a","A","player1","player 1","p1"], "_A")
        df_chem_all = _norm_col(df_chem_all, ["Player B","player b","player_b","B","player2","player 2","p2"], "_B")

    # --- Filter chemistry for teammates on each team ---
    df_a = df_chem_all[
        df_chem_all["_A"].isin(team_a_n) & df_chem_all["_B"].isin(team_a_n)
    ].sort_values("Chemistry", ascending=False)

    df_b = df_chem_all[
        df_chem_all["_A"].isin(team_b_n) & df_chem_all["_B"].isin(team_b_n)
    ].sort_values("Chemistry", ascending=False)

    # Limit to top 10 pairs
    return (
        df_a.head(10).reset_index(drop=True),
        df_b.head(10).reset_index(drop=True)
    )

# -------------------------------
# 3️⃣  Scores to Settle (Recent Meetings)
# -------------------------------
def _get_recent_meetings(team_a, team_b, conn=None, matches_df=None):
    all_matches = matches_df.copy() if matches_df is not None else _processed_matches_for_current_league().copy()
    if "date" in all_matches.columns:
        all_matches = all_matches.sort_values("date", ascending=False)

    def split_names(s):
        return [p.strip() for p in str(s or "").split(",") if p.strip()]

    overlapping_matches = []

    for _, r in all_matches.iterrows():
        ta = split_names(r["team_a"])
        tb = split_names(r["team_b"])
        res = str(r.get("result", "")).upper()

        # count overlap both ways (A vs B or reversed)
        overlap_a = len(set(ta).intersection(team_a))
        overlap_b = len(set(tb).intersection(team_b))
        overlap_rev_a = len(set(ta).intersection(team_b))
        overlap_rev_b = len(set(tb).intersection(team_a))

        if (overlap_a >= 3 and overlap_b >= 3) or (overlap_rev_a >= 3 and overlap_rev_b >= 3):
            overlapping_matches.append(r)

    # --- no overlaps large enough ---
    if not overlapping_matches:
        return "– 3+ players on each team haven’t played each other before."

    # --- summarise those overlaps ---
    df = pd.DataFrame(overlapping_matches)
    total_meetings = len(df)
    team_a_wins = 0
    team_b_wins = 0
    margins = []

    for _, r in df.iterrows():
        res = str(r.get("result", "")).upper()
        if res == "A":
            team_a_wins += 1
        elif res == "B":
            team_b_wins += 1

        if isinstance(r.get("score"), str) and "-" in r["score"]:
            try:
                a_score, b_score = [int(x) for x in r["score"].split("-")]
                margins.append(abs(a_score - b_score))
            except Exception:
                pass

    last_date = df["date"].max()
    avg_margin = np.mean(margins) if margins else None

    # build friendly text summary
    text = (
        f"These two squads (3+ overlapping players each) have faced each other **{total_meetings}** times.\n\n"
        f"Team A has won **{team_a_wins}**, Team B has won **{team_b_wins}**."
    )
    if avg_margin is not None:
        text += f"\n\nAverage margin of victory: **{avg_margin:.1f} goals**."

    text += f"\n\nMost recent meeting: **{last_date}**."
    return text

# -------------------------------
# 4️⃣  Form & Streaks (WITH DRAWS)
# -------------------------------
def _get_form_streaks(players, conn=None, recent_n=10, matches_df=None):
    matches = matches_df.copy() if matches_df is not None else _processed_matches_for_current_league().copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.sort_values("date", ascending=True).dropna(subset=["date"])

    def split_names(s):
        return [p.strip() for p in str(s or "").split(",") if p.strip()]
    
    def _parse_score(score):
        if not isinstance(score, str) or "-" not in score:
            return None, None
        try:
            a, b = score.split("-")
            return int(a.strip()), int(b.strip())
        except Exception:
            return None, None

    form_data = []

    for p in players:
        p_key = clean_name(p)
        results = []

        for _, r in matches.iterrows():
            ta = split_names(r["team_a"])
            tb = split_names(r["team_b"])

            res = str(r.get("result", "")).strip().upper()


            gA, gB = _parse_score(r.get("score"))
            if gA is None or gB is None:
                continue

            if p_key in [clean_name(x) for x in ta]:
                if gA > gB:
                    results.append("W")
                elif gA < gB:
                    results.append("L")
                else:
                    results.append("D")

            elif p_key in [clean_name(x) for x in tb]:
                if gB > gA:
                    results.append("W")
                elif gB < gA:
                    results.append("L")
                else:
                    results.append("D")

        recent_results = results[-recent_n:]

        if not recent_results:
            form = "No recent data"
            wins = losses = draws = 0
        else:
            wins = recent_results.count("W")
            losses = recent_results.count("L")
            draws = recent_results.count("D")

            # ---- STREAK LOGIC (draw breaks streak) ----
            last = recent_results[-1]

            if last == "D":
                form = "No streak"  # draw breaks streaks
            else:
                streak = 1
                for rr in reversed(recent_results[:-1]):
                    if rr != last:
                        break
                    streak += 1

                if last == "W":
                    form = f"Won {streak} straight"
                else:  # last == "L"
                    form = f"Lost {streak} straight"

        form_data.append({
            "Player": p,
            "Recent Matches": len(recent_results),
            "Wins (last 10)": wins,
            "Losses (last 10)": losses,
            "Draws (last 10)": draws,
            "Current Form": form
        })

    df = (
        pd.DataFrame(form_data)
        .sort_values(["Wins (last 10)", "Draws (last 10)"], ascending=False)
        .reset_index(drop=True)
    )
    return df

# -------------------------------
# 5️⃣  Match Outcome Prediction (MMR + Chemistry + Historical)
# -------------------------------
from utils.db_utils import get_conn as open_db, get_current_league_id
from utils.calc_utils import expected_score
from utils.stats_shared import get_chemistry_df, get_intensity_df


@st.cache_data(ttl=300, show_spinner=False)
def _processed_matches_cached(league_id: int) -> pd.DataFrame:
    """One cached read for match-preview insight calculations."""
    conn = open_db()
    try:
        return pd.read_sql(
            """
            SELECT date, team_a, team_b, result, score
            FROM matches
            WHERE processed=1 AND league_id=%s;
            """,
            conn,
            params=(int(league_id),),
        )
    finally:
        conn.close()


def _processed_matches_for_current_league() -> pd.DataFrame:
    return _processed_matches_cached(get_current_league_id())

def predict_match_outcome(team_a, team_b, conn):
    """
    Blend MMR, chemistry and historical matchups to generate a match prediction.
    Returns a dict with probA, probB and text.
    """
    league_id = get_current_league_id()
    players_df = pd.read_sql(
        "SELECT name, mmr FROM players WHERE league_id=%s",
        conn,
        params=(int(league_id),),
    )
    matches = _processed_matches_cached(league_id)

    # --- Base MMR averages
    avgA = players_df[players_df["name"].isin(team_a)]["mmr"].mean()
    avgB = players_df[players_df["name"].isin(team_b)]["mmr"].mean()

    # --- Chemistry weighting
    chem_df = get_chemistry_df(matches_df=matches)

    def avg_team_chem(team):
        # Normalize columns to handle naming differences
        cols = [c.lower().strip() for c in chem_df.columns]
        if "player a" in cols and "player b" in cols:
            a_col = chem_df.columns[cols.index("player a")]
            b_col = chem_df.columns[cols.index("player b")]
        elif "player1" in cols and "player2" in cols:
            a_col = chem_df.columns[cols.index("player1")]
            b_col = chem_df.columns[cols.index("player2")]
        else:
            # No recognizable player columns → no chemistry data
            return 0

        subset = chem_df[chem_df[a_col].isin(team) & chem_df[b_col].isin(team)]
        return (
            subset["Chemistry"].mean() / 100
            if not subset.empty and "Chemistry" in chem_df.columns
            else 0
        )

    chemA, chemB = avg_team_chem(team_a), avg_team_chem(team_b)

    effA = avgA * (1 + chemA)
    effB = avgB * (1 + chemB)

    # --- Expected outcome from MMR (Elo-style)
    probA = expected_score(effA, effB)
    probB = 1 - probA

    # --- Historical adjustment (past meetings between these players)
    relevant = matches[
        matches["team_a"].apply(lambda x: any(p in str(x) for p in team_a)) &
        matches["team_b"].apply(lambda x: any(p in str(x) for p in team_b))
    ]
    if not relevant.empty:
        wins_A = sum(relevant["result"] == "A")
        wins_B = sum(relevant["result"] == "B")
        total = wins_A + wins_B
        if total > 0:
            hist_adv = (wins_A / total) - 0.5
            probA += hist_adv * 0.1  # ±5% swing
            probA = min(max(probA, 0.05), 0.95)
            probB = 1 - probA

    # --- Convert to readable prediction
    diff = abs(effA - effB)
    if 0.48 <= probA <= 0.52:
        text = "Prediction: **Too close to call 🤝**"
    elif probA > 0.52:
        margin = "by 1 goal" if diff < 75 else "by 2+ goals"
        conf = int((probA - 0.5) * 200)
        text = f"Prediction: **Team A** to win {margin} ({conf}% confidence)"
    else:
        margin = "by 1 goal" if diff < 75 else "by 2+ goals"
        conf = int((probB - 0.5) * 200)
        text = f"Prediction: **Team B** to win {margin} ({conf}% confidence)"

    return {"text": text, "probA": probA, "probB": probB}







# -------------------------------
# Main generator
# -------------------------------
def generate_preview_insights(team_a, team_b, conn):
    team_a = [p.strip() for p in team_a if p.strip()]
    team_b = [p.strip() for p in team_b if p.strip()]
    all_players = team_a + team_b

    matches_df = _processed_matches_for_current_league()

    insights = {
        "key_matchups": _get_key_matchups(team_a, team_b, conn, matches_df=matches_df),
        "best_teammates": _get_best_teammates(team_a, team_b, conn, matches_df=matches_df),
        "recent_meetings": _get_recent_meetings(team_a, team_b, conn, matches_df=matches_df),
        "form_streaks": _get_form_streaks(all_players, conn, matches_df=matches_df),
    }
    return insights
