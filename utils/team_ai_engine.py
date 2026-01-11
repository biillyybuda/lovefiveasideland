# utils/team_ai_engine.py
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd

from utils.db_utils import STARTING_MMR, get_conn
from utils.relationships_utils import calculate_chemistry_for_all_duos


def clean_name(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"\s+", " ", x, flags=re.UNICODE)
    return x.strip().lower()


def _split_team(val: str) -> list[str]:
    raw = str(val or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    parts = raw.split(",")
    cleaned: list[str] = []
    for p in parts:
        name = p.strip().strip("'").strip('"')
        if name:
            cleaned.append(name)
    return cleaned


_ENGINE_CACHE: Dict[str, Any] | None = None


def _load_db_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load matches and players from DB (BASIC MODE: no highlight_moments)."""
    conn = get_conn()
    matches = pd.read_sql(
        "SELECT * FROM matches WHERE result IN ('A','B','Draw');",
        conn
    )

    try:
        players = pd.read_sql("SELECT * FROM players;", conn)
    except Exception:
        players = pd.DataFrame(columns=["id", "name", "mmr", "fitness"])

    conn.close()
    return matches, players


def _avg(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mu = _avg(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return float(var ** 0.5)


def _fitness_adjust(label: str) -> float:
    # keep it mild - fitness is optional metadata, not video-derived
    if label == "High":
        return 10.0
    if label == "Low":
        return -10.0
    return 0.0  # Medium/unknown


def _build_engine_state() -> Dict[str, Any]:
    matches, players = _load_db_tables()

    # --- Base maps (MMR, fitness) ---
    mmr_map: Dict[str, float] = {}
    fitness_map: Dict[str, str] = {}

    for _, r in players.iterrows():
        raw_name = str(r.get("name", "")).strip()
        name = clean_name(raw_name)
        if not name:
            continue

        try:
            mmr_val = float(r.get("mmr", STARTING_MMR) or STARTING_MMR)
        except Exception:
            mmr_val = float(STARTING_MMR)

        mmr_map[name] = mmr_val
        fitness_map[name] = str(r.get("fitness", "Medium") or "Medium")

    # --- Total matches per player + form (last 8) from match results only ---
    total_matches: Dict[str, int] = defaultdict(int)
    result_rows = []

    for _, m in matches.iterrows():
        ta_raw = _split_team(m.get("team_a", ""))
        tb_raw = _split_team(m.get("team_b", ""))
        ta = [clean_name(p) for p in ta_raw if clean_name(p)]
        tb = [clean_name(p) for p in tb_raw if clean_name(p)]

        res = str(m.get("result", "")).upper()
        date = m.get("date")

        for p in ta + tb:
            total_matches[p] += 1

        if res == "A":
            winners, losers = set(ta), set(tb)
        elif res == "B":
            winners, losers = set(tb), set(ta)
        else:
            winners, losers = set(), set()

        for p in ta + tb:
            if p in winners:
                outcome = "W"
            elif p in losers:
                outcome = "L"
            else:
                outcome = "D"
            result_rows.append({"player_name": p, "date": date, "outcome": outcome})

    form_index: Dict[str, float] = {}
    if result_rows:
        rf = pd.DataFrame(result_rows)
        rf["date"] = pd.to_datetime(rf["date"], errors="coerce")
        rf = rf.sort_values("date")
        score_map = {"W": 1.0, "D": 0.5, "L": 0.0}

        for name, g in rf.groupby("player_name"):
            tail = g.tail(8)
            vals = tail["outcome"].map(score_map)
            form_index[name] = float(vals.mean()) if len(vals) > 0 else 0.5

    # defaults
    for name in mmr_map.keys():
        form_index.setdefault(name, 0.5)

    # --- Chemistry (match-based only) ---
    chem_df = calculate_chemistry_for_all_duos(matches).rename(
        columns={"Player A": "player_a", "Player B": "player_b", "Chemistry": "chemistry_score"}
    )
    base_chemistry: Dict[tuple[str, str], float] = {}
    if not chem_df.empty:
        for _, r in chem_df.iterrows():
            a = clean_name(str(r["player_a"]))
            b = clean_name(str(r["player_b"]))
            try:
                val = float(r["chemistry_score"])
            except Exception:
                val = 0.0
            if a and b:
                base_chemistry[(a, b)] = val
                base_chemistry[(b, a)] = val

    return {
        "matches": matches,
        "players": players,
        "mmr_map": mmr_map,
        "fitness_map": fitness_map,
        "form_index": form_index,
        "total_matches": dict(total_matches),
        "base_chemistry": base_chemistry,
    }


def get_engine_state(force_reload: bool = True) -> Dict[str, Any]:
    """Basic engine state (MMR + fitness + form + chemistry)."""
    global _ENGINE_CACHE
    if _ENGINE_CACHE is None or force_reload:
        _ENGINE_CACHE = _build_engine_state()
    return _ENGINE_CACHE


def _effective_mmr(name: str, state: Dict[str, Any]) -> float:
    """Effective rating used for balancing (BASIC MODE)."""
    mmr_map = state["mmr_map"]
    fitness_map = state["fitness_map"]
    form_index = state["form_index"]
    total_matches = state.get("total_matches", {})

    base = float(mmr_map.get(name, float(STARTING_MMR)))
    fit_adj = _fitness_adjust(fitness_map.get(name, "Medium"))

    # --- Form (W/D/L last 8) ---
    form_val = float(form_index.get(name, 0.5))

    # Shrink form impact for low sample sizes so new players don't swing teams
    m = int(total_matches.get(name, 0) or 0)
    # scale ramps from 0.25 -> 1.0 between 0 and 12 matches (tweakable)
    scale = 0.25 + 0.75 * min(1.0, m / 12.0)

    form_adj = (form_val - 0.5) * 40.0 * scale

    return base + fit_adj + form_adj


def _team_chemistry(team: List[str], state: Dict[str, Any]) -> float:
    """Sum of pair chemistry across team (BASIC MODE)."""
    base = state["base_chemistry"]
    total = 0.0
    for i in range(len(team)):
        for j in range(i + 1, len(team)):
            a, b = team[i], team[j]
            total += float(base.get((a, b), 0.0))
    return total


def evaluate_teams(team_a: List[str], team_b: List[str]) -> tuple[float, Dict[str, Any]]:
    """
    Returns (fairness_score, breakdown).
    Lower fairness_score = more balanced.
    """
    state = get_engine_state(force_reload=False)

    # Effective ratings (includes fitness + form)
    eff_a = [_effective_mmr(p, state) for p in team_a]
    eff_b = [_effective_mmr(p, state) for p in team_b]

    mmr_a = _avg(eff_a)
    mmr_b = _avg(eff_b)
    mmr_diff = abs(mmr_a - mmr_b)

    # Spread (prevents "carry + passengers" vs "flat team" being scored as equal)
    spread_a = _std(eff_a)
    spread_b = _std(eff_b)
    spread_diff = abs(spread_a - spread_b)

    # Chemistry (match-based)
    chem_a = _team_chemistry(team_a, state)
    chem_b = _team_chemistry(team_b, state)
    chem_diff = abs(chem_a - chem_b)

    # Combine weights
    W_MMR = 1.0
    W_SPREAD = 0.25   # fairness feel: balance team shape (tweakable)
    W_CHEM = 0.15     # keep chemistry as tie-breaker

    fairness_score = (W_MMR * mmr_diff) + (W_SPREAD * spread_diff) + (W_CHEM * chem_diff)

    breakdown = {
        "mmr_avg_a": mmr_a,
        "mmr_avg_b": mmr_b,
        "mmr_diff": mmr_diff,
        "spread_a": spread_a,
        "spread_b": spread_b,
        "spread_diff": spread_diff,
        "chem_a": chem_a,
        "chem_b": chem_b,
        "chem_diff": chem_diff,
        "fairness_score": fairness_score,
        "weights": {"W_MMR": W_MMR, "W_SPREAD": W_SPREAD, "W_CHEM": W_CHEM},
        "mode": "BASIC (MMR + fitness + form + chemistry + spread)",
    }
    return fairness_score, breakdown
