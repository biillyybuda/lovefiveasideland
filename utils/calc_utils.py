"""
calc_utils.py — FairMMR v2 Integrated
-------------------------------------
Replaces dynamic K with FairMMR v2 adjustments:
  • Team imbalance protection (underdog boost, favourite dampening)
  • Volatility damping based on league mean
Fully compatible with original app logic and database.
"""

import math
import pandas as pd
import streamlit as st
from utils.db_utils import K_DEFAULT, DRAW_VALUE, STARTING_MMR, get_conn, get_current_league_id
from psycopg2.extras import RealDictCursor

def get_mmr(player_name: str, conn=None):
    """
    Get a player's current MMR from the database.
    Returns 1000 if the player doesn't exist or has no recorded MMR yet.
    """
    if conn is None:
        conn = get_conn()
    try:
        league_id = get_current_league_id()
    except Exception:
        return 1000.0

    df = pd.read_sql(
        "SELECT mmr FROM players WHERE name=%s AND league_id=%s",
        conn,
        params=(player_name, league_id),
    )
    if df.empty:
        return 1000.0
    return float(df["mmr"].iloc[0])



def expected_score(mmr_a, mmr_b):
    """Standard Elo expected score."""
    return 1 / (1 + 10 ** ((mmr_b - mmr_a) / 400.0))

import math
import numpy as np
import pandas as pd
from utils.db_utils import get_conn

def expected_score_calibrated(r_a: float, r_b: float, scale: float = 200.0) -> float:
    # logistic win prob based on rating diff / scale
    x = (r_a - r_b) / float(scale)
    return 1.0 / (1.0 + math.exp(-x))

def calibrate_winprob_scale(default_scale: float = 200.0, league_id: int | None = None) -> float:
    """
    Learn the best scale so predicted probs match real match outcomes.
    Uses matches.team_a_avg/team_b_avg saved at processing time.
    """
    if league_id is None:
        try:
            league_id = get_current_league_id()
        except Exception:
            return float(default_scale)

    conn = get_conn()
    df = pd.read_sql("""
        SELECT team_a_avg, team_b_avg, result
        FROM matches
        WHERE processed = 1
          AND team_a_avg IS NOT NULL
          AND team_b_avg IS NOT NULL
          AND result IN ('A','B','Draw')
          AND league_id = %s
    """, conn, params=(int(league_id),))
    conn.close()

    if df.empty or len(df) < 12:
        return float(default_scale)

    y = df["result"].map({"A": 1.0, "Draw": 0.5, "B": 0.0}).to_numpy(dtype=float)
    d = (df["team_a_avg"] - df["team_b_avg"]).to_numpy(dtype=float)

    candidates = np.linspace(80.0, 500.0, 85)
    eps = 1e-9
    best_s, best_loss = float(default_scale), float("inf")

    for s in candidates:
        p = 1.0 / (1.0 + np.exp(-(d / s)))
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        if loss < best_loss:
            best_loss, best_s = loss, float(s)

    return best_s




def imbalance_factor(player_team_avg: float, opp_team_avg: float) -> float:
    """Boosts underdogs, reduces favourites (up to +50% / -40%)."""
    gap = abs(player_team_avg - opp_team_avg)
    boost = min(0.5, gap / 300.0)  # +50% max
    cut = min(0.4, gap / 300.0)    # -40% max
    if player_team_avg < opp_team_avg:
        return 1.0 + boost
    else:
        return 1.0 - cut


def volatility_factor(player_mmr: float, league_mean: float) -> float:
    """Reduces volatility for high-rated players."""
    v = 1.0 - 0.5 * ((player_mmr - league_mean) / 1000.0)
    return float(max(0.5, min(1.0, v)))



def process_unprocessed_matches(k_factor=K_DEFAULT, draw_value=DRAW_VALUE):
    """Process all unprocessed matches and update player MMRs using FairMMR v2."""

    st.write("🔁 Processing all unprocessed matches from clean slate...")

    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    league_id = get_current_league_id()
    cur.execute(
        'SELECT * FROM matches WHERE processed=0 AND league_id=%s ORDER BY date ASC',
        (league_id,),
    )
    rows = cur.fetchall()
    if not rows:
        conn.close()
        return 0

    def split_team(s):
        return [p.strip() for p in str(s or '').split(',') if p and p.strip()]

    def load_players_fresh():
        """Reload the latest player MMRs from the DB after each match update."""
        df_players = pd.read_sql('SELECT * FROM players WHERE league_id=%s', conn, params=(league_id,))
        return {r['name']: r for _, r in df_players.iterrows()}
    import re

    def _norm(name: str) -> str:
        name = str(name or "").strip()
        name = re.sub(r"\s+", " ", name)
        return name.lower()

    def _build_name_lookup(name_to_row: dict) -> dict:
        """
        Map normalized name -> canonical players.name
        """
        return {_norm(k): k for k in name_to_row.keys()}

    def _ensure_player_exists(canonical_name: str):
        cur.execute("SELECT 1 FROM players WHERE name=%s AND league_id=%s", (canonical_name, league_id))
        exists = cur.fetchone()
        if not exists:
            cur.execute(
                """
                INSERT INTO players (league_id, name, mmr, matches_played, wins, losses, draws, win_streak, lose_streak, last_match_date)
                VALUES (%s, %s, %s, 0, 0, 0, 0, 0, 0, NULL)
                """,
                (league_id, canonical_name, float(STARTING_MMR)),
            )
            conn.commit()

    processed_count = 0
    for row in rows:
        mid = row["id"]
        mid = int(mid)
        mdate = row.get("date")
        team_a = row.get("team_a")
        team_b = row.get("team_b")
        score = row.get("score")
        result = row.get("result")
        team_a_avg = row.get("team_a_avg")
        team_b_avg = row.get("team_b_avg")
        processed = row.get("processed")
        # 🧹 Remove any existing MMR history for this match before recalculating
        cur.execute("DELETE FROM mmr_history WHERE match_id=%s AND league_id=%s", (mid, league_id))
        conn.commit()

        # 🧩 Ensure team scores are available for goal difference checks
        def _score_to_ints(s):
            try:
                parts = [int(x) for x in str(s).replace("-", ":").split(":")]
                return parts[0], parts[1]
            except Exception:
                return (0, 0)

        a_sc, b_sc = _score_to_ints(score)
        if a_sc is None or b_sc is None:
            a_sc = b_sc = 0


        # 🔁 Load latest player MMRs before each match
        name_to_row = load_players_fresh()
        name_lookup = _build_name_lookup(name_to_row)

        def get_mmr(name):
            """Safely get MMR for a player or default to STARTING_MMR."""
            if name in name_to_row:
                return float(name_to_row[name]['mmr'])
            return float(STARTING_MMR)

        # --- Canonicalise team names ---
        ta_raw = split_team(team_a)
        tb_raw = split_team(team_b)

        ta = []
        for raw in ta_raw:
            key = _norm(raw)
            canon = name_lookup.get(key, key)
            _ensure_player_exists(canon)
            ta.append(canon)

        tb = []
        for raw in tb_raw:
            key = _norm(raw)
            canon = name_lookup.get(key, key)
            _ensure_player_exists(canon)
            tb.append(canon)

        if not ta or not tb:
            continue

        # Reload players & lookup in case new ones were inserted
        name_to_row = load_players_fresh()
        name_lookup = _build_name_lookup(name_to_row)

        # OPTIONAL BUT STRONGLY RECOMMENDED: rewrite canonical names into matches table
        cur.execute(
            "UPDATE matches SET team_a=%s, team_b=%s WHERE id=%s AND league_id=%s",
            (", ".join(ta), ", ".join(tb), mid, league_id),
        )
        conn.commit()

        a_avg = sum(get_mmr(n) for n in ta) / len(ta)
        b_avg = sum(get_mmr(n) for n in tb) / len(tb)

        # Compute league mean dynamically
        df_all = pd.read_sql('SELECT mmr FROM players WHERE league_id=%s', conn, params=(league_id,))
        league_mean = float(df_all['mmr'].mean()) if not df_all.empty else STARTING_MMR

        # Determine match result
        if str(result).upper() == 'A':
            score_a, score_b = (1.0, 0.0)
        elif str(result).upper() == 'B':
            score_a, score_b = (0.0, 1.0)
        else:
            score_a = score_b = float(draw_value)

        histories = []

        # --- Update Team A players
        for n in ta:
            before = get_mmr(n)
            exp = expected_score(before, b_avg)
            imb = imbalance_factor(a_avg, b_avg)
            vol = volatility_factor(before, league_mean)
            k_final = k_factor * imb * vol

            st.write(f"[DEBUG] Processing MMR: match_id={mid}, player_name={n}")


            # --- result-only MMR update (no goals/assists/video stats)
            base_delta = k_final * (score_a - exp)
            after = before + base_delta

            st.write(
                f"[DEBUG] {n}: baseΔ={base_delta:.2f}, "
                f"MMR {before:.2f} → {after:.2f}"
            )

            cur.execute(
                'UPDATE players SET mmr=%s WHERE name=%s AND league_id=%s',
                (after, n, league_id)
            )
            histories.append((n, before, after))



        # --- Update Team B players
        for n in tb:
            before = get_mmr(n)
            exp = expected_score(before, a_avg)
            imb = imbalance_factor(b_avg, a_avg)
            vol = volatility_factor(before, league_mean)
            k_final = k_factor * imb * vol

            st.write(f"[DEBUG] Processing MMR: match_id={mid}, player_name={n}")


            # --- result-only MMR update (no goals/assists/video stats)
            base_delta = k_final * (score_b - exp)
            after = before + base_delta

            st.write(
                f"[DEBUG] {n}: baseΔ={base_delta:.2f}, "
                f"MMR {before:.2f} → {after:.2f}"
            )

            cur.execute(
                'UPDATE players SET mmr=%s WHERE name=%s AND league_id=%s',
                (after, n, league_id)
            )
            histories.append((n, before, after))




        # --- Update player stats
        for n in set(ta + tb):
            cur.execute('SELECT * FROM players WHERE name=%s AND league_id=%s', (n, league_id))
            r = cur.fetchone()
            if not r:
                continue
            mp = (r['matches_played'] or 0) + 1
            w = r['wins'] or 0
            l = r['losses'] or 0
            d = r['draws'] or 0
            ws = r['win_streak'] or 0
            ls = r['lose_streak'] or 0

            if n in ta:
                if str(result).upper() == 'A':
                    w += 1; ws += 1; ls = 0
                elif str(result).upper() == 'B':
                    l += 1; ls += 1; ws = 0
                else:
                    d += 1; ws = 0; ls = 0
            elif str(result).upper() == 'B':
                w += 1; ws += 1; ls = 0
            elif str(result).upper() == 'A':
                l += 1; ls += 1; ws = 0
            else:
                d += 1; ws = 0; ls = 0

            cur.execute(
                'UPDATE players SET matches_played=%s, wins=%s, losses=%s, draws=%s, win_streak=%s, lose_streak=%s, last_match_date=%s WHERE name=%s AND league_id=%s',
                (mp, w, l, d, ws, ls, mdate, n, league_id)
            )

        # --- Record match MMR history (prevent duplicates)
        for n, before, after in histories:
            cur.execute('SELECT id FROM players WHERE name=%s AND league_id=%s', (n, league_id))
            prow = cur.fetchone()
            pid = int(prow["id"]) if prow and prow.get("id") is not None else None

            # remove any existing duplicate entries first
            cur.execute(
                "DELETE FROM mmr_history WHERE player_id=%s AND match_id=%s AND league_id=%s",
                (pid, mid, league_id)
            )

            # insert fresh history record
            cur.execute(
                'INSERT INTO mmr_history (league_id, player_id, match_id, date, mmr_before, mmr_after) VALUES (%s,%s,%s,%s,%s,%s)',
                (league_id, pid, mid, mdate, before, after)
            )

        # --- Mark match as processed
        cur.execute('UPDATE matches SET processed=1, team_a_avg=%s, team_b_avg=%s WHERE id=%s AND league_id=%s',
                    (a_avg, b_avg, mid, league_id))
        conn.commit()
        processed_count += 1

    conn.close()
    return processed_count

def reset_and_reprocess_season():
    """
    Full rebuild:
      - Reset all players to STARTING_MMR
      - Reset W/L/D + streaks + matches_played
      - Clear mmr_history
      - Mark all matches unprocessed
      - Re-run processing from earliest date
    """
    conn = get_conn()
    cur = conn.cursor()
    league_id = get_current_league_id()

    # Reset player MMR + season stats
    cur.execute("""
        UPDATE players
        SET mmr = %s,
            matches_played = 0,
            wins = 0,
            losses = 0,
            draws = 0,
            win_streak = 0,
            lose_streak = 0,
            last_match_date = NULL
        WHERE league_id = %s
    """, (STARTING_MMR, league_id))

    # Clear MMR history
    cur.execute("DELETE FROM mmr_history WHERE league_id = %s", (league_id,))

    # Unprocess all matches + clear stored avgs (optional but nice)
    cur.execute("""
        UPDATE matches
        SET processed = 0,
            team_a_avg = NULL,
            team_b_avg = NULL
        WHERE league_id = %s
    """, (league_id,))

    conn.commit()
    conn.close()

    # Now process from scratch
    return process_unprocessed_matches()




def compute_streaks_from_matches(matches_df):
    import pandas as _pd
    if matches_df is None or getattr(matches_df, 'empty', True):
        return ({}, {}, {}, {})
    matches = matches_df.copy()
    if 'processed' in matches.columns:
        matches = matches[matches['processed'] == 1]
    if 'date' in matches.columns:
        matches['date'] = _pd.to_datetime(matches['date'], errors='coerce')
        matches = matches.sort_values('date')

    def _split_team(s):
        if s is None:
            return []
        txt = str(s).strip()

        # handle python-list string like "['a', 'b']"
        if txt.startswith("[") and txt.endswith("]"):
            txt = txt[1:-1].replace("'", "").replace('"', "")

        return [p.strip() for p in txt.split(",") if p and p.strip()]
    all_players = set()
    for _, m in matches.iterrows():
        all_players.update(_split_team(m.get('team_a', '')))
        all_players.update(_split_team(m.get('team_b', '')))
    cur_win = {p: 0 for p in all_players}
    cur_lose = {p: 0 for p in all_players}
    max_win = {p: 0 for p in all_players}
    max_lose = {p: 0 for p in all_players}
    for _, m in matches.iterrows():
        ta = _split_team(m.get('team_a', ''))
        tb = _split_team(m.get('team_b', ''))
        result = str(m.get('result', '')).strip()
        if result == 'A':
            outcomes = {p: 'W' if p in ta else 'L' for p in ta + tb}
        elif result == 'B':
            outcomes = {p: 'W' if p in tb else 'L' for p in ta + tb}
        else:
            outcomes = {p: 'D' for p in ta + tb}
        for p in ta + tb:
            o = outcomes.get(p, 'D')
            if o == 'W':
                cur_win[p] = cur_win.get(p, 0) + 1
                cur_lose[p] = 0
                if cur_win[p] > max_win.get(p, 0):
                    max_win[p] = cur_win[p]
            elif o == 'L':
                cur_lose[p] = cur_lose.get(p, 0) + 1
                cur_win[p] = 0
                if cur_lose[p] > max_lose.get(p, 0):
                    max_lose[p] = cur_lose[p]
            else:
                cur_win[p] = 0
                cur_lose[p] = 0
    return (cur_win, cur_lose, max_win, max_lose)
