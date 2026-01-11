import streamlit as st
import pandas as pd
from collections import defaultdict
from utils.db_utils import load_players_df, load_matches_df, get_conn
from utils.calc_utils import compute_streaks_from_matches
from utils.export_utils import df_to_png
from utils.mmr_utils import get_season_mmr, get_current_season_start


def render_performance_page():
    st.markdown("<h2>📈 Performance</h2>", unsafe_allow_html=True)
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)

    dfp = load_players_df()
    dfm = load_matches_df()
    dfm['processed'] = dfm['processed'].astype(int)
    dfm_proc = dfm[dfm['processed'] == 1].copy()
    dfm_proc['date'] = pd.to_datetime(dfm_proc['date'], errors='coerce')

    # --- Helpers ---
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

    # --- Count matches, wins, losses, draws ---
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

    # --- Streaks ---
    cur_win, cur_lose, max_win, max_lose = compute_streaks_from_matches(dfm_proc)

    # --- Last 5 MMR & delta ---
    conn = get_conn()
    rows = []
    for _, r in dfp.iterrows():
        pid = r['id']
        name = r['name']
        ph = pd.read_sql(
            'SELECT * FROM mmr_history WHERE player_id = %s ORDER BY id DESC LIMIT 5',
            conn,
            params=(pid,)
        )

        season_start = get_current_season_start()

        if ph.empty:
            avg_mmr = get_season_mmr(conn, pid, season_start, r.get("mmr", 1000))
            delta = 0
        else:
            # season-adjust each mmr_after in the last 5
            ph["season_after"] = ph["mmr_after"].apply(lambda v: get_season_mmr(conn, pid, season_start, float(v)))
            ph["season_before"] = ph["mmr_before"].apply(lambda v: get_season_mmr(conn, pid, season_start, float(v)))

            avg_mmr = ph["season_after"].mean()
            delta = ph["season_after"].iloc[0] - ph["season_before"].iloc[-1]

        mp = matches_ct.get(name, 0)
        w = wins_ct.get(name, 0)
        l = losses_ct.get(name, 0)
        d = draws_ct.get(name, 0)
        win_pct = round((w / mp) * 100, 1) if mp > 0 else 0.0

        rows.append({
            'Name': name,
            'Matches': mp,
            'Wins': w,
            'Losses': l,
            'Draws': d,
            'Avg MMR (Last 5)': round(avg_mmr, 1),
            'MMR Δ': round(delta, 0),
            'Win %': win_pct,
            'Winning Streak': cur_win.get(name, 0),
            'Losing Streak': cur_lose.get(name, 0),
        })
    conn.close()

    perf = pd.DataFrame(rows).sort_values(['Win %', 'Matches'], ascending=[False, False]).reset_index(drop=True)

    st.dataframe(perf, use_container_width=True)
    st.download_button(
        "📥 Export Performance Stats as PNG",
        data=df_to_png(perf, title="Player Performance Overview"),
        file_name="performance.png",
        mime="image/png"
    )

    st.markdown("</div>", unsafe_allow_html=True)
