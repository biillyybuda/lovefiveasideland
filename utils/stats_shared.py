import pandas as pd
from utils.calc_utils import get_conn
from utils.relationships_utils import (
    calculate_chemistry_for_all_duos,
    calculate_rivalry_intensity,
)

# --------------------------------------------------
# ✅ Chemistry & Intensity (Shared)
# --------------------------------------------------
def get_chemistry_df(conn=None, matches_df=None):
    """
    Compute chemistry scores for all teammate duos using the shared relationships_utils logic.
    If matches_df is provided, it is used (season-aware).
    """
    if conn is None:
        conn = get_conn()

    # ✅ Respect passed-in matches_df
    if matches_df is None:
        matches_df = pd.read_sql("SELECT * FROM matches WHERE processed=1;", conn)

    if matches_df is None or matches_df.empty:
        return pd.DataFrame(columns=["player_a", "player_b", "matches", "wins", "win_pct", "chemistry"])

    df = calculate_chemistry_for_all_duos(matches_df)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df = df.rename(columns={
        "win_%": "win_pct",
        "chemistry_score": "chemistry",
        "score": "chemistry"
    })

    # ✅ Ensure required columns exist
    for col in ["player_a", "player_b", "matches", "wins", "win_pct", "chemistry"]:
        if col not in df.columns:
            df[col] = 0

    # ✅ Normalise names so joins/comparisons are stable
    df["player_a"] = df["player_a"].fillna("").astype(str).str.strip().str.lower()
    df["player_b"] = df["player_b"].fillna("").astype(str).str.strip().str.lower()

    # ✅ Combine reversed duplicates
    df["pair_key"] = df.apply(lambda r: tuple(sorted([r["player_a"], r["player_b"]])), axis=1)
    df = (
        df.groupby("pair_key")
        .agg({
            "player_a": "first",
            "player_b": "last",
            "matches": "sum",
            "wins": "sum",
            "win_pct": "mean",
            "chemistry": "mean"
        })
        .reset_index(drop=True)
    )

    # 🚫 Remove self-pairs
    df = df[df["player_a"] != df["player_b"]].reset_index(drop=True)

    return df


def get_intensity_df(conn=None, matches_df=None):
    """
    Compute rivalry intensity scores for all opposing duos using the shared relationships_utils logic.
    Standardizes column names for consistency and merges duplicate (A,B)/(B,A) entries.
    """
    if conn is None:
        conn = get_conn()

    if matches_df is None:
        matches_df = pd.read_sql("SELECT * FROM matches WHERE processed=1;", conn)

    if matches_df is None or matches_df.empty:
        return pd.DataFrame(columns=["player_a", "player_b", "matches", "intensity"])

    df = calculate_rivalry_intensity(matches_df)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df = df.rename(columns={
        "intensity_score": "intensity",
        "rivalry_score": "intensity",
        "score": "intensity",
        "win_%": "win_pct",
    })

    # ✅ Ensure required columns exist
    for col in ["player_a", "player_b", "matches", "intensity"]:
        if col not in df.columns:
            df[col] = 0

    # ✅ Normalise names so lookups are stable (matches charts_page normalisation)
    df["player_a"] = df["player_a"].fillna("").astype(str).str.strip().str.lower()
    df["player_b"] = df["player_b"].fillna("").astype(str).str.strip().str.lower()

    # ✅ Combine reversed duplicates
    df["pair_key"] = df.apply(lambda r: tuple(sorted([r["player_a"], r["player_b"]])), axis=1)
    df = (
        df.groupby("pair_key")
        .agg({
            "player_a": "first",
            "player_b": "last",
            "matches": "sum",
            "intensity": "mean"
        })
        .reset_index(drop=True)
    )

    # 🚫 Remove self-pairs
    df = df[df["player_a"] != df["player_b"]].reset_index(drop=True)

    return df

# --------------------------------------------------
# ✅ Pair-level lookups
# --------------------------------------------------
def get_pair_chemistry(player_a, player_b, conn=None, df=None):
    """
    Returns chemistry for a duo.
    If no games together in the provided df, chemistry is None (not 0.0).
    """
    if df is None:
        df = get_chemistry_df(conn)

    a = str(player_a).strip().lower()
    b = str(player_b).strip().lower()

    pair_row = df[
        ((df["player_a"] == a) & (df["player_b"] == b)) |
        ((df["player_a"] == b) & (df["player_b"] == a))
    ]

    if pair_row.empty:
        return {"chemistry": None, "matches": 0, "wins": 0, "win_pct": 0.0}

    r = pair_row.iloc[0]
    matches = int(r.get("matches", 0))

    return {
        "chemistry": float(r["chemistry"]) if matches > 0 else None,
        "matches": matches,
        "wins": int(r.get("wins", 0)),
        "win_pct": float(r.get("win_pct", 0.0)),
    }


# --- add these tiny helpers near the top of the file if not present ---
def _split_team(val):
    """
    Robust split:
    - "Kelso, Ob"
    - "Kelso; Ob"
    - "Kelso | Ob"
    - "Kelso / Ob"
    - "['Kelso', 'Ob']"
    """
    if val is None:
        return []

    s = str(val).strip()

    # handle list-string like "['a', 'b']"
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].replace("'", "").replace('"', "")

    # normalise separators
    for sep in [";", "|", "/"]:
        s = s.replace(sep, ",")

    return [p.strip() for p in s.split(",") if p.strip()]

def _score_to_ints(sc):
    try:
        s = str(sc or "").strip()
        if "-" in s:
            a, b = s.split("-", 1)
            return int(a.strip()), int(b.strip())
        if ":" in s:
            a, b = s.split(":", 1)
            return int(a.strip()), int(b.strip())
    except Exception:
        pass
    return None, None


def get_pair_intensity(player_a, player_b, conn=None, df=None, matches_df=None):
    if conn is None:
        conn = get_conn()

    if matches_df is None:
        matches_df = pd.read_sql("SELECT * FROM matches WHERE processed=1;", conn)

    # normalise input keys
    a = str(player_a).strip().lower()
    b = str(player_b).strip().lower()

    if df is None:
        df = get_intensity_df(conn, matches_df=matches_df)

    row = df[
        ((df["player_a"] == a) & (df["player_b"] == b)) |
        ((df["player_a"] == b) & (df["player_b"] == a))
    ]
    intensity_val = float(row.iloc[0]["intensity"]) if (not row.empty and row.iloc[0]["intensity"] is not None) else 0.0

    wins_a = wins_b = draws = 0
    gd_list = []
    meetings = 0

    for _, m in matches_df.iterrows():
        ta = [p.strip().lower() for p in _split_team(m.get("team_a", ""))]
        tb = [p.strip().lower() for p in _split_team(m.get("team_b", ""))]
        res = str(m.get("result", "")).strip().upper()
        if res == "D":
            res = "DRAW"


        a_vs_b = (a in ta and b in tb) or (a in tb and b in ta)
        if not a_vs_b:
            continue

        meetings += 1

        a_sc, b_sc = _score_to_ints(m.get("score", ""))
        if a_sc is None or b_sc is None:
            a_sc = b_sc = 0
        gd_list.append(abs(a_sc - b_sc))

        if res == "DRAW" or a_sc == b_sc:
            draws += 1
        else:
            if a in ta and res == "A": wins_a += 1
            elif a in tb and res == "B": wins_a += 1
            elif b in ta and res == "A": wins_b += 1
            elif b in tb and res == "B": wins_b += 1

    avg_gd = (sum(gd_list) / len(gd_list)) if gd_list else 0.0

    return {
        "intensity": float(intensity_val or 0.0),   # never None
        "matches": meetings,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "avg_goal_diff": avg_gd,
    }
