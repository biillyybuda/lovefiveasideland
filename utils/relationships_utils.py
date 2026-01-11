"""
relationships_utils.py
----------------------
Central source for Chemistry & Rivalry calculations.

• Chemistry now uses reliability + depth weighting.
• Rivalry intensity includes depth weighting & balance.
• Ensures consistent scoring across Charts, This Week & Season Review.
"""

import math
import pandas as pd

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def _split_team(team_str):
    """Safely handle both strings and lists for team data."""
    if isinstance(team_str, list):
        return [p.strip() for p in team_str if isinstance(p, str) and p.strip()]
    return [p.strip() for p in str(team_str or "").split(",") if p.strip()]

def _score_to_ints(score_str: str):
    """Convert '4-3' to (4,3) safely."""
    try:
        if isinstance(score_str, str) and "-" in score_str:
            a, b = score_str.split("-", 1)
            return int(a.strip()), int(b.strip())
    except Exception:
        pass
    return None, None

# --------------------------------------------------
# 🤝 Chemistry
# --------------------------------------------------
def calculate_chemistry_for_all_duos(matches_df: pd.DataFrame):
    """
    Compute chemistry for all teammate duos with depth & reliability weighting.
    Returns DataFrame of results.
    """
    if matches_df is None or matches_df.empty:
        return pd.DataFrame()

    # --- Auto-fix for stringified lists like "['Billy', 'Gav']"
    def _safe_list_convert(x):
        if isinstance(x, list):
            return x
        x = str(x).strip()
        if x.startswith("[") and x.endswith("]"):
            x = x[1:-1]
        return [p.strip().strip("'").strip('"') for p in x.split(",") if p.strip()]

    if "team_a" in matches_df.columns:
        matches_df["team_a"] = matches_df["team_a"].apply(_safe_list_convert)
    if "team_b" in matches_df.columns:
        matches_df["team_b"] = matches_df["team_b"].apply(_safe_list_convert)

    duo_map = {}
    for _, m in matches_df.iterrows():
        # ✅ Make sure we only split strings — not already parsed lists
        ta = m.get("team_a", [])
        tb = m.get("team_b", [])

        if not isinstance(ta, list):
            ta = _split_team(ta)
        if not isinstance(tb, list):
            tb = _split_team(tb)

        res = (m.get("result") or "").upper()
        a_sc, b_sc = _score_to_ints(m.get("score", ""))

        if a_sc is None or b_sc is None:
            a_sc = b_sc = 0
        gd = abs(a_sc - b_sc)

        # Record every teammate pair
        for team in (ta, tb):
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    duo = tuple(sorted((team[i], team[j])))
                    entry = duo_map.setdefault(duo, {"matches": 0, "wins": 0, "gds": []})
                    entry["matches"] += 1
                    won = (res == "A" and team is ta) or (res == "B" and team is tb)
                    if won:
                        entry["wins"] += 1
                    entry["gds"].append(gd)

    rows = []
    import re

    def _clean_name(name):
        if not isinstance(name, str):
            return name
        name = re.sub(r"^[\[\]'\"\s]+|[\[\]'\"\s]+$", "", name)
        return name.strip()

    # ✅ append rows inside the loop
    for (a, b), d in duo_map.items():
        games = d["matches"]
        wins = d["wins"]
        win_pct = wins / games if games else 0.0
        avg_gd = (sum(d["gds"]) / len(d["gds"])) if d["gds"] else 0.0

        # --- Depth weighting + closeness adjustment ---
        depth_weight = 0.5 + 0.5 * min(1.0, math.log10(games + 1) / math.log10(10))
        closeness = max(0.35, 1 - (avg_gd / 8))   # “Balanced” profile
        base_chem = games * win_pct * closeness
        chemistry = base_chem * depth_weight

        if wins == 0:
            chemistry *= 0.5

        rows.append({
            "Player A": _clean_name(a),
            "Player B": _clean_name(b),
            "Matches": games,
            "Wins": wins,
            "Win %": round((wins / games) * 100, 1),
            "Chemistry": round(float(chemistry), 3),
        })

    df = pd.DataFrame(rows).sort_values("Chemistry", ascending=False).reset_index(drop=True)

    # 🧹 Prettify player names for display (capitalize but keep uppercase internally)
    df["Player A"] = df["Player A"].apply(lambda x: x.title() if isinstance(x, str) else x)
    df["Player B"] = df["Player B"].apply(lambda x: x.title() if isinstance(x, str) else x)

    return df










# --------------------------------------------------
# ⚔️ Rivalry Intensity
# --------------------------------------------------
def calculate_rivalry_intensity(matches_df: pd.DataFrame):
    """
    Compute rivalry intensity between all opposing player pairs with depth weighting.
    Fully merges A–B and B–A so each rivalry is unique and assigns wins to the
    correct player after sorting the key.
    """
    if matches_df is None or matches_df.empty:
        return pd.DataFrame()

    # --- Normalize scores to handle fancy dashes and spaces
    if "score" in matches_df.columns:
        matches_df["score"] = (
            matches_df["score"]
            .astype(str)
            .str.replace("–", "-", regex=False)
            .str.replace("—", "-", regex=False)
            .str.replace("−", "-", regex=False)
            .str.strip()
        )

    # --- Ensure team_a and team_b are clean uppercase lists
    def _normalize_team_list(x):
        if isinstance(x, list):
            return [str(p).strip().upper() for p in x if str(p).strip()]
        return [str(p).strip().upper() for p in str(x).split(",") if str(p).strip()]

    for col in ["team_a", "team_b"]:
        if col in matches_df.columns:
            matches_df[col] = matches_df[col].apply(_normalize_team_list)

    pair_map = {}

    for _, m in matches_df.iterrows():
        ta = m.get("team_a", [])
        tb = m.get("team_b", [])
        a_sc, b_sc = _score_to_ints(m.get("score", ""))
        if a_sc is None or b_sc is None:
            continue

        gd = abs(a_sc - b_sc)

        # Every cross-team pair (use sorted key; assign wins relative to sorted names)
        for a in ta:
            for b in tb:
                p1, p2 = tuple(sorted((a, b)))  # sorted names for storage/output
                key = (p1, p2)

                entry = pair_map.setdefault(
                    key, {"matches": 0, "wins_p1": 0, "wins_p2": 0, "gds": []}
                )
                entry["matches"] += 1

                if a_sc > b_sc:
                    # Team A won; increment whoever (p1/p2) is in Team A
                    if p1 in ta:
                        entry["wins_p1"] += 1
                    if p2 in ta:
                        entry["wins_p2"] += 1
                elif b_sc > a_sc:
                    # Team B won
                    if p1 in tb:
                        entry["wins_p1"] += 1
                    if p2 in tb:
                        entry["wins_p2"] += 1

                entry["gds"].append(gd)

    # Build rows using p1/p2 consistently
    rows = []
    for (p1, p2), d in pair_map.items():
        games = d["matches"]
        if games <= 0:
            continue

        w1 = d["wins_p1"]
        w2 = d["wins_p2"]
        wpa = w1 / games
        wpb = w2 / games
        diff = abs(wpa - wpb)
        avg_gd = sum(d["gds"]) / len(d["gds"]) if d["gds"] else 0.0

        base_intensity = games * (1 - diff) * (1 - min(avg_gd, 5) / 5)
        depth_weight = min(1.0, math.log10(games + 1) / math.log10(10))
        intensity = base_intensity * depth_weight

        if games <= 2:
            intensity *= 0.6

        rows.append({
            "Player A": p1,
            "Player B": p2,
            "Matches": games,
            "Wins A": w1,
            "Wins B": w2,
            "Intensity": round(float(intensity), 3),
        })

    df = pd.DataFrame(rows).sort_values("Intensity", ascending=False).reset_index(drop=True)

    # 🧹 Prettify player names for display (capitalize but keep internal logic uppercase)
    df["Player A"] = df["Player A"].apply(lambda x: x.title() if isinstance(x, str) else x)
    df["Player B"] = df["Player B"].apply(lambda x: x.title() if isinstance(x, str) else x)

    return df

    print(f"DEBUG: Rivalry pairs created = {len(rows)}")

    df = pd.DataFrame(rows).sort_values("Intensity", ascending=False).reset_index(drop=True)
    return df
