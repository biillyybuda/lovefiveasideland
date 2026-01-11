import streamlit as st
import pandas as pd
import itertools
import math
from utils.calc_utils import get_conn
from utils.relationships_utils import _score_to_ints

st.set_page_config(page_title="🔺 Trio Insights", layout="wide")

def render_trio_insights_page():
    st.title("🔺 Trio Insights — Experimental View")

    conn = get_conn()
    matches_df = pd.read_sql("SELECT * FROM matches WHERE processed=1;", conn)

    if matches_df.empty:
        st.warning("No processed matches found.")
        return

    # --- Normalize team strings into lists ---
    def _safe_list(x):
        if isinstance(x, list):
            return x
        s = str(x or "").strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        return [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]

    matches_df["team_a"] = matches_df["team_a"].apply(_safe_list)
    matches_df["team_b"] = matches_df["team_b"].apply(_safe_list)

    # --- Collect trio + duo stats ---
    trio_map, duo_map = {}, {}
    for _, m in matches_df.iterrows():
        ta, tb = m["team_a"], m["team_b"]
        res = (m.get("result") or "").upper()
        a_sc, b_sc = _score_to_ints(m.get("score", ""))
        if a_sc is None or b_sc is None:
            a_sc = b_sc = 0
        gd = abs(a_sc - b_sc)

        for team, label in [(ta, "A"), (tb, "B")]:
            # trios
            for trio in itertools.combinations(sorted(team), 3):
                e = trio_map.setdefault(trio, {"matches": 0, "wins": 0, "gds": []})
                e["matches"] += 1
                if res == label:
                    e["wins"] += 1
                e["gds"].append(gd)
            # duos (for later synergy)
            for duo in itertools.combinations(sorted(team), 2):
                e = duo_map.setdefault(duo, {"matches": 0, "wins": 0})
                e["matches"] += 1
                if res == label:
                    e["wins"] += 1

    # --- Build trio table ---
    rows = []
    for trio, d in trio_map.items():
        games = d["matches"]
        wins = d["wins"]
        win_pct = wins / games if games else 0
        avg_gd = sum(d["gds"]) / len(d["gds"]) if d["gds"] else 0

        depth_weight = 0.5 + 0.5 * min(1, math.log10(games + 1) / math.log10(10))
        closeness = max(0.35, 1 - (avg_gd / 8))
        strength = games * win_pct * closeness * depth_weight

        # --- Synergy vs. average of the 3 duos ---
        duos = list(itertools.combinations(trio, 2))
        duo_win_pcts = []
        for duo in duos:
            duo = tuple(sorted(duo))
            if duo in duo_map and duo_map[duo]["matches"] > 0:
                duo_win_pcts.append(duo_map[duo]["wins"] / duo_map[duo]["matches"])
        avg_duo_win = sum(duo_win_pcts) / len(duo_win_pcts) if duo_win_pcts else 0
        synergy = round((win_pct - avg_duo_win) * 100, 1)  # percentage-point diff

        rows.append({
            "Trio": " / ".join(trio),
            "Matches": games,
            "Wins": wins,
            "Win %": round(win_pct * 100, 1),
            "Avg GD": round(avg_gd, 2),
            "Strength": round(strength, 3),
            "Synergy Δ (pp)": synergy
        })

    df = pd.DataFrame(rows).sort_values("Strength", ascending=False).reset_index(drop=True)

    # --- Display ---
    st.subheader("🏆 Top 20 Trios Overall")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("🔍 Explore by Player")
    players = sorted({p for trio in trio_map.keys() for p in trio})
    sel_player = st.selectbox("Select a player", players)
    player_trios = df[df["Trio"].str.contains(sel_player, case=False)]
    st.dataframe(player_trios.head(20), use_container_width=True)

    st.caption("Formula: Matches × Win% × (1 − AvgGD / 8) × DepthWeight  |  Synergy Δ = Trio Win% − Avg(Duo Win %)")

if __name__ == "__main__":
    render_trio_insights_page()
