# utils/team_ai_engine.py
import re
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional


import pandas as pd

from utils.db_utils import STARTING_MMR, get_conn
from utils.relationships_utils import calculate_chemistry_for_all_duos


def clean_name(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"\s+", " ", x, flags=re.UNICODE)
    return x.strip().lower()


def _split_team(val: str) -> List[str]:
    raw = str(val or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    parts = raw.split(",")
    cleaned: List[str] = []
    for p in parts:
        name = p.strip().strip("'").strip('"')
        if name:
            cleaned.append(name)
    return cleaned


def _parse_score(score_str: Any) -> tuple[Optional[int], Optional[int]]:
    """Parse '6-4' / '6 4' etc. Returns (a,b) or (None,None)."""
    try:
        parts = str(score_str).replace("-", " ").replace("–", " ").split()
        nums = [int(x) for x in parts if str(x).lstrip("-").isdigit()]
        if len(nums) >= 2:
            return nums[0], nums[1]
    except Exception:
        pass
    return None, None



# -----------------------------
# Historic stats/style layer
# -----------------------------
# These tables are treated as a fixed historical benchmark. They do NOT replace
# MMR/results. They only describe player style: scorer, creator, link player,
# save/defensive impact, and proven creator -> finisher combinations.
GOAL_TAG_WEIGHTS: Dict[str, float] = {
    "🏆 Game-Winning Goal": 2.00,
    "🤝 Game-Saving Goal": 1.90,
    "🔥 Equaliser": 1.70,
    "🔺 Go-Ahead Goal": 1.55,
    "🔵 Pulls One Back": 1.25,
    "🟢 Doubling the Lead": 1.10,
    "💪 Extending Lead": 0.95,
    "💥 Rout": 0.55,
}

SAVE_TAG_WEIGHTS: Dict[str, float] = {
    "🧤 Unbelievable Save": 1.75,
    "🚨 Crucial Save": 1.60,
    "🔥 Big Save": 1.40,
    "💪 Strong Save": 1.20,
    "✅ Routine Save": 0.65,
}

SPECIAL_TAG_BONUS: Dict[str, float] = {
    "⚽ Worldie Goal": 0.35,
    "🎯 Brilliant Assist": 0.35,
    "🧤 Unbelievable Save": 0.35,
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def _clean_tag(x: Any) -> str:
    s = str(x or "").strip()
    if s.lower() in ("nan", "none", "nat"):
        return ""
    return s


def _tag_weight(tag: Any, default: float = 1.0) -> float:
    t = _clean_tag(tag)
    if not t:
        return default
    if t in GOAL_TAG_WEIGHTS:
        return float(GOAL_TAG_WEIGHTS[t])
    if t in SAVE_TAG_WEIGHTS:
        return float(SAVE_TAG_WEIGHTS[t])
    return default


def _special_bonus(tag: Any) -> float:
    t = _clean_tag(tag)
    if not t:
        return 0.0
    return float(SPECIAL_TAG_BONUS.get(t, 0.0))


def _normalize_scores(raw: Dict[str, float]) -> Dict[str, float]:
    """Scale a player score map to 0..1 without letting one outlier dominate."""
    vals = [float(v) for v in raw.values() if float(v) > 0]
    if not vals:
        return {}
    vals_sorted = sorted(vals)
    # Use an 85th-ish percentile cap so the best players are near 1.0 but not
    # wildly ahead just because of a long old video-stat sample.
    idx = int(round((len(vals_sorted) - 1) * 0.85))
    cap = max(vals_sorted[idx], max(vals_sorted) * 0.35, 1.0)
    return {k: max(0.0, min(1.0, float(v) / cap)) for k, v in raw.items()}


def _current_league_id_or_none() -> Optional[int]:
    try:
        from utils.db_utils import get_current_league_id
        return int(get_current_league_id())
    except Exception:
        return None


def _read_optional_table(conn, table_name: str, columns: str = "*") -> pd.DataFrame:
    """Best-effort table reader. If a legacy/dev DB lacks the table, return empty."""
    league_id = _current_league_id_or_none()
    try:
        if league_id is not None:
            return pd.read_sql(
                f"SELECT {columns} FROM public.{table_name} WHERE league_id = %s",
                conn,
                params=(league_id,),
            )
        return pd.read_sql(f"SELECT {columns} FROM public.{table_name}", conn)
    except Exception:
        try:
            return pd.read_sql(f"SELECT {columns} FROM {table_name}", conn)
        except Exception:
            return pd.DataFrame()


def _build_historic_style_state(match_stats: pd.DataFrame, highlights: pd.DataFrame) -> Dict[str, Any]:
    """Build fixed historical style profiles from old goals/assists/highlights.

    Output maps are intentionally small 0..1-ish signals used as team-shape
    nudges. They are not player-strength ratings and should not replace MMR.
    """
    raw_goals: Dict[str, float] = defaultdict(float)
    raw_assists: Dict[str, float] = defaultdict(float)
    raw_weighted_goals: Dict[str, float] = defaultdict(float)
    raw_weighted_assists: Dict[str, float] = defaultdict(float)
    raw_save_impact: Dict[str, float] = defaultdict(float)
    raw_involvement: Dict[str, float] = defaultdict(float)
    raw_clutch: Dict[str, float] = defaultdict(float)
    stat_matches: Dict[str, set] = defaultdict(set)
    creator_links_raw: Dict[Tuple[str, str], float] = defaultdict(float)
    creator_link_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    # Basic totals from match_stats: stable, simple style baseline.
    if match_stats is not None and not match_stats.empty:
        for _, r in match_stats.iterrows():
            p = clean_name(str(r.get("player_name", "")))
            if not p:
                continue
            g = _safe_float(r.get("goals", 0.0))
            a = _safe_float(r.get("assists", 0.0))
            raw_goals[p] += g
            raw_assists[p] += a
            raw_involvement[p] += g + a
            mid = r.get("match_id", None)
            if mid is not None:
                try:
                    stat_matches[p].add(int(mid))
                except Exception:
                    pass

    # Highlight moments: weighted importance, creator-finisher links, saves.
    if highlights is not None and not highlights.empty:
        h = highlights.copy()
        for _, r in h.iterrows():
            actor = clean_name(str(r.get("player_name", "")))
            if not actor:
                continue
            typ = str(r.get("type", "") or "").strip().lower()
            special = r.get("special_tag", "")

            if typ == "goal":
                w = _tag_weight(r.get("goal_tag", ""), default=1.0) + _special_bonus(special)
                raw_weighted_goals[actor] += w
                if w >= 1.55:
                    raw_clutch[actor] += w

            elif typ == "assist":
                w = _tag_weight(r.get("assist_tag", ""), default=1.0) + _special_bonus(special)
                raw_weighted_assists[actor] += w
                if w >= 1.55:
                    raw_clutch[actor] += w * 0.8

                # Link assist row to the goal row at the same match/timestamp.
                scorer = ""
                try:
                    same = h[
                        (h.get("match_id") == r.get("match_id"))
                        & (h.get("timestamp_sec") == r.get("timestamp_sec"))
                        & (h.get("type").astype(str).str.lower() == "goal")
                    ]
                    if not same.empty:
                        scorer = clean_name(str(same.iloc[0].get("player_name", "")))
                except Exception:
                    scorer = ""

                # Fallback: parse labels like "Billy Assist (Jos goal)".
                if not scorer:
                    lab = str(r.get("label", "") or "")
                    m = re.search(r"\((.*?)\s+goal\)", lab, flags=re.IGNORECASE)
                    if m:
                        scorer = clean_name(m.group(1))

                if scorer and scorer != actor:
                    creator_links_raw[(actor, scorer)] += w
                    creator_link_counts[(actor, scorer)] += 1

            elif typ == "save":
                w = _tag_weight(r.get("save_tag", ""), default=1.0) + _special_bonus(special)
                w += min(0.75, max(0.0, _safe_float(r.get("save_importance", 0.0)) * 0.15))
                raw_save_impact[actor] += w

    # Blend simple totals and tagged/weighted moments. Match_stats is a stable
    # baseline; highlights add importance and role nuance.
    all_players = set(raw_goals) | set(raw_assists) | set(raw_weighted_goals) | set(raw_weighted_assists) | set(raw_save_impact)
    finish_raw: Dict[str, float] = {}
    create_raw: Dict[str, float] = {}
    impact_raw: Dict[str, float] = {}
    save_raw: Dict[str, float] = {}
    clutch_raw: Dict[str, float] = {}

    for p in all_players:
        wg = raw_weighted_goals.get(p, 0.0)
        wa = raw_weighted_assists.get(p, 0.0)
        g = raw_goals.get(p, 0.0)
        a = raw_assists.get(p, 0.0)
        sv = raw_save_impact.get(p, 0.0)
        finish_raw[p] = (g * 0.55) + (wg * 0.45)
        create_raw[p] = (a * 0.55) + (wa * 0.45)
        save_raw[p] = sv
        clutch_raw[p] = raw_clutch.get(p, 0.0)
        impact_raw[p] = finish_raw[p] + (create_raw[p] * 0.85) + (sv * 0.65) + (clutch_raw[p] * 0.35)

    finishing = _normalize_scores(finish_raw)
    creation = _normalize_scores(create_raw)
    impact = _normalize_scores(impact_raw)
    save_impact = _normalize_scores(save_raw)
    clutch = _normalize_scores(clutch_raw)

    profiles: Dict[str, Dict[str, float]] = {}
    for p in all_players:
        f = float(finishing.get(p, 0.0))
        c = float(creation.get(p, 0.0))
        s = float(save_impact.get(p, 0.0))
        i = float(impact.get(p, 0.0))
        total = finish_raw.get(p, 0.0) + create_raw.get(p, 0.0)
        finisher_ratio = finish_raw.get(p, 0.0) / total if total > 0 else 0.5
        creator_ratio = create_raw.get(p, 0.0) / total if total > 0 else 0.5
        profiles[p] = {
            "finishing": f,
            "creation": c,
            "impact": i,
            "save_impact": s,
            "clutch": float(clutch.get(p, 0.0)),
            "finisher_ratio": float(finisher_ratio),
            "creator_ratio": float(creator_ratio),
            "historic_goals": float(raw_goals.get(p, 0.0)),
            "historic_assists": float(raw_assists.get(p, 0.0)),
            "weighted_goals": float(raw_weighted_goals.get(p, 0.0)),
            "weighted_assists": float(raw_weighted_assists.get(p, 0.0)),
            "historic_matches": float(len(stat_matches.get(p, set()))),
        }

    # Scale creator links to small useful bonuses. Direction matters: creator -> scorer.
    link_norm = _normalize_scores(creator_links_raw)
    creator_links: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, raw_val in creator_links_raw.items():
        creator_links[key] = {
            "strength": float(link_norm.get(key, 0.0)),
            "weighted_value": float(raw_val),
            "count": int(creator_link_counts.get(key, 0)),
        }

    return {
        "style_profiles": profiles,
        "creator_links": creator_links,
        "finisher_score": {p: profiles[p]["finishing"] for p in profiles},
        "creator_score": {p: profiles[p]["creation"] for p in profiles},
        "impact_index": {p: profiles[p]["impact"] for p in profiles},
        "save_impact": {p: profiles[p]["save_impact"] for p in profiles},
        "historic_style_enabled": bool(profiles),
        "historic_style_debug": {
            "players_with_profiles": len(profiles),
            "creator_links": len(creator_links),
            "match_stats_rows": int(len(match_stats)) if match_stats is not None else 0,
            "highlight_rows": int(len(highlights)) if highlights is not None else 0,
        },
    }


_ENGINE_CACHE: Dict[str, Any] | None = None


def _load_db_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load matches, players and optional historic stat tables from DB.

    match_stats/highlight_moments are fixed historical benchmarks. If they do
    not exist in a dev DB, the engine falls back to the old results-only model.
    """
    conn = get_conn()
    league_id = _current_league_id_or_none()

    try:
        if league_id is not None:
            matches = pd.read_sql(
                """
                SELECT id, date, team_a, team_b, result, score
                FROM public.matches
                WHERE league_id = %s
                  AND result IN ('A','B','Draw','D','DRAW')
                """,
                conn,
                params=(league_id,),
            )
        else:
            matches = pd.read_sql(
                """
                SELECT id, date, team_a, team_b, result, score
                FROM matches
                WHERE result IN ('A','B','Draw','D','DRAW')
                """,
                conn,
            )
    except Exception:
        matches = pd.DataFrame(columns=["id", "date", "team_a", "team_b", "result", "score"])

    try:
        if league_id is not None:
            players = pd.read_sql(
                "SELECT id, name, mmr, fitness FROM public.players WHERE league_id = %s;",
                conn,
                params=(league_id,),
            )
        else:
            players = pd.read_sql("SELECT id, name, mmr, fitness FROM players;", conn)
    except Exception:
        players = pd.DataFrame(columns=["id", "name", "mmr", "fitness"])

    match_stats = _read_optional_table(conn, "match_stats")
    highlights = _read_optional_table(conn, "highlight_moments")

    conn.close()
    return matches, players, match_stats, highlights


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
    matches, players, match_stats, highlights = _load_db_tables()

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

    # --- Player win% (results-only) for "bad pair" logic ---
    player_win_rate: Dict[str, float] = {}
    if result_rows:
        pr = pd.DataFrame(result_rows)
        score_map = {"W": 1.0, "D": 0.5, "L": 0.0}
        pr["score"] = pr["outcome"].map(score_map)
        for name, g in pr.groupby("player_name"):
            try:
                player_win_rate[name] = float(g["score"].mean())
            except Exception:
                player_win_rate[name] = 0.5

    for name in mmr_map.keys():
        player_win_rate.setdefault(name, 0.5)

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
    base_chemistry: Dict[Tuple[str, str], float] = {}
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

    # --- Duo outcomes (results-only) for "known bad pairing" penalties ---
    # Track: games, score_mean (W=1,D=0.5,L=0), and sample size.
    duo_games: Dict[Tuple[str, str], int] = defaultdict(int)
    duo_score_sum: Dict[Tuple[str, str], float] = defaultdict(float)

    score_map = {"W": 1.0, "D": 0.5, "L": 0.0}
    for _, m in matches.iterrows():
        ta = [clean_name(p) for p in _split_team(m.get("team_a", "")) if clean_name(p)]
        tb = [clean_name(p) for p in _split_team(m.get("team_b", "")) if clean_name(p)]
        res = str(m.get("result", "")).upper()

        if res == "A":
            out_a, out_b = "W", "L"
        elif res == "B":
            out_a, out_b = "L", "W"
        else:
            out_a, out_b = "D", "D"

        s_a = score_map.get(out_a, 0.5)
        s_b = score_map.get(out_b, 0.5)

        # All pairs within each team share the same match outcome
        for team, sc in ((ta, s_a), (tb, s_b)):
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    a, b = team[i], team[j]
                    if not a or not b:
                        continue
                    key1 = (a, b)
                    key2 = (b, a)
                    duo_games[key1] += 1
                    duo_games[key2] += 1
                    duo_score_sum[key1] += sc
                    duo_score_sum[key2] += sc

    # Build a "bad pair" penalty map: positive = worse (we'll add to fairness)
    bad_pair_penalty: Dict[Tuple[str, str], float] = {}

    # Tunables (kept conservative so we don't overfit)
    MIN_GAMES_BAD = 10
    ABS_BAD_WINRATE = 0.38         # stricter: avoid labelling normal variance as a toxic duo
    EXPECTED_DROP = 0.20           # must be meaningfully worse than the players' normal win rates
    MAX_PEN = 12.0                 # softer cap for a 50-100 match dataset

    for (a, b), g in duo_games.items():
        if g < MIN_GAMES_BAD:
            continue
        sc = float(duo_score_sum.get((a, b), 0.0))
        winrate = sc / float(g) if g else 0.5

        exp = 0.5 * (float(player_win_rate.get(a, 0.5)) + float(player_win_rate.get(b, 0.5)))
        drop = exp - winrate

        # only flag if it is *actually* poor and meaningfully below expectation
        if winrate <= ABS_BAD_WINRATE and drop >= EXPECTED_DROP:
            # stronger penalty as sample grows, but saturates
            sample_factor = min(1.0, g / 18.0)  # needs a proper sample before it bites
            pen = min(MAX_PEN, 4.0 + (drop * 35.0)) * sample_factor
            bad_pair_penalty[(a, b)] = float(pen)

    # --- Trio outcomes (results-only) for trio synergy (triangle chemistry) ---
    # Track: games, score_mean (W=1,D=0.5,L=0), and sample size.
    trio_games: Dict[Tuple[str, str, str], int] = defaultdict(int)
    trio_score_sum: Dict[Tuple[str, str, str], float] = defaultdict(float)

    for _, m in matches.iterrows():
        ta = [clean_name(p) for p in _split_team(m.get("team_a", "")) if clean_name(p)]
        tb = [clean_name(p) for p in _split_team(m.get("team_b", "")) if clean_name(p)]
        res = str(m.get("result", "")).upper()

        if res == "A":
            out_a, out_b = "W", "L"
        elif res == "B":
            out_a, out_b = "L", "W"
        else:
            out_a, out_b = "D", "D"

        s_a = score_map.get(out_a, 0.5)
        s_b = score_map.get(out_b, 0.5)

        for team, sc in ((ta, s_a), (tb, s_b)):
            if len(team) < 3:
                continue
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    for k in range(j + 1, len(team)):
                        a, b, c = team[i], team[j], team[k]
                        if not a or not b or not c:
                            continue
                        key = tuple(sorted((a, b, c)))
                        trio_games[key] += 1
                        trio_score_sum[key] += sc

    # Build trio synergy map: positive = better than expected, negative = worse than expected.
    trio_synergy: Dict[Tuple[str, str, str], float] = {}

    # Tunables (conservative, avoids overfitting)
    MIN_GAMES_TRIO = 8
    MAX_TRIO = 18.0  # softer cap; trios are noisy in a small/medium dataset

    for key, g in trio_games.items():
        if g < MIN_GAMES_TRIO:
            continue
        sc = float(trio_score_sum.get(key, 0.0))
        winrate = sc / float(g) if g else 0.5

        a, b, c = key
        exp = (float(player_win_rate.get(a, 0.5)) + float(player_win_rate.get(b, 0.5)) + float(player_win_rate.get(c, 0.5))) / 3.0
        delta = winrate - exp

        # sample grows confidence but saturates (0..1)
        sample_factor = min(1.0, g / 16.0)
        val = delta * 35.0 * sample_factor

        # tiny deltas are noise
        if abs(val) < 3.0:
            continue

        trio_synergy[key] = float(max(-MAX_TRIO, min(MAX_TRIO, val)))


    # --- Historic style layer (fixed old goals/assists/highlights benchmark) ---
    style_state = _build_historic_style_state(match_stats, highlights)

    state = {
        "matches": matches,
        "players": players,
        "mmr_map": mmr_map,
        "fitness_map": fitness_map,
        "form_index": form_index,
        "total_matches": dict(total_matches),
        "player_win_rate": player_win_rate,
        "base_chemistry": base_chemistry,
        "bad_pair_penalty": bad_pair_penalty,
        "trio_synergy": trio_synergy,
    }
    state.update(style_state)
    return state


def get_engine_state(force_reload: bool = False) -> Dict[str, Any]:
    """Engine state (MMR + fitness + form + chemistry + duo penalties).

    Default is cached. Use force_reload=True only immediately after DB writes.
    This avoids rebuilding all match/chemistry history on every Streamlit rerun.
    """
    global _ENGINE_CACHE
    if _ENGINE_CACHE is None or force_reload:
        _ENGINE_CACHE = _build_engine_state()
    return _ENGINE_CACHE


def _effective_mmr(name: str, state: Dict[str, Any]) -> float:
    """Effective rating used for balancing (results-only)."""
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


def _pair_list(team: List[str]) -> list[Tuple[str, str]]:
    out: list[Tuple[str, str]] = []
    for i in range(len(team)):
        for j in range(i + 1, len(team)):
            out.append((team[i], team[j]))
    return out


def _team_chemistry(team: List[str], state: Dict[str, Any]) -> tuple[float, list[float]]:
    """Sum of pair chemistry across team and also return the pair values."""
    base = state["base_chemistry"]
    vals: list[float] = []
    total = 0.0
    for a, b in _pair_list(team):
        v = float(base.get((a, b), 0.0))
        vals.append(v)
        total += v
    return total, vals


def _team_badpair_badness(team: List[str], state: Dict[str, Any]) -> float:
    """Sum of 'bad pairing' penalties within a team (higher = worse)."""
    bad_map = state.get("bad_pair_penalty", {}) or {}
    bad = 0.0
    for a, b in _pair_list(team):
        bad += float(bad_map.get((a, b), 0.0))
    return bad


def _trio_list(team: List[str]) -> list[Tuple[str, str, str]]:
    out: list[Tuple[str, str, str]] = []
    for i in range(len(team)):
        for j in range(i + 1, len(team)):
            for k in range(j + 1, len(team)):
                out.append((team[i], team[j], team[k]))
    return out


def _team_trio_synergy(team: List[str], state: Dict[str, Any]) -> tuple[float, list[float]]:
    """Sum of trio synergy across team and also return the trio values.

    Trio synergy is results-only: how well a trio performs together vs what you'd expect
    from the players' baseline win rates (shrunken for small sample sizes).
    """
    trio_map = state.get("trio_synergy", {}) or {}
    vals: list[float] = []
    total = 0.0
    for a, b, c in _trio_list(team):
        key = tuple(sorted((a, b, c)))
        v = float(trio_map.get(key, 0.0))
        vals.append(v)
        total += v
    return total, vals



def _team_style_profile(team: List[str], state: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate fixed historic style signals for a team.

    Values are averages of 0..1 player style scores, so they remain comparable
    whether the team is 4v4 captain mode or standard 5v5.
    """
    profiles = state.get("style_profiles", {}) or {}
    if not team:
        return {
            "finishing": 0.0,
            "creation": 0.0,
            "impact": 0.0,
            "save_impact": 0.0,
            "clutch": 0.0,
            "known_players": 0,
            "creator_count": 0,
            "finisher_count": 0,
        }

    vals = {"finishing": [], "creation": [], "impact": [], "save_impact": [], "clutch": []}
    creator_count = 0
    finisher_count = 0
    known = 0
    for p in team:
        prof = profiles.get(clean_name(p), {})
        if prof:
            known += 1
        f = float(prof.get("finishing", 0.0) or 0.0)
        c = float(prof.get("creation", 0.0) or 0.0)
        vals["finishing"].append(f)
        vals["creation"].append(c)
        vals["impact"].append(float(prof.get("impact", 0.0) or 0.0))
        vals["save_impact"].append(float(prof.get("save_impact", 0.0) or 0.0))
        vals["clutch"].append(float(prof.get("clutch", 0.0) or 0.0))
        if c >= 0.45:
            creator_count += 1
        if f >= 0.45:
            finisher_count += 1

    def _mean_key(k: str) -> float:
        return _avg(vals[k]) if vals[k] else 0.0

    return {
        "finishing": _mean_key("finishing"),
        "creation": _mean_key("creation"),
        "impact": _mean_key("impact"),
        "save_impact": _mean_key("save_impact"),
        "clutch": _mean_key("clutch"),
        "known_players": int(known),
        "creator_count": int(creator_count),
        "finisher_count": int(finisher_count),
    }


def _team_creator_link_bonus(team: List[str], state: Dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """Bonus for proven historic creator -> scorer links kept on the same team."""
    links = state.get("creator_links", {}) or {}
    team_set = {clean_name(p) for p in team if clean_name(p)}
    if not links or not team_set:
        return 0.0, []

    total = 0.0
    used: list[dict[str, Any]] = []
    for (creator, scorer), data in links.items():
        if creator in team_set and scorer in team_set:
            strength = float((data or {}).get("strength", 0.0) or 0.0)
            if strength <= 0:
                continue
            # Square-root softens one monster link while keeping it meaningful.
            contribution = math.sqrt(strength)
            total += contribution
            used.append({
                "creator": creator,
                "scorer": scorer,
                "strength": strength,
                "count": int((data or {}).get("count", 0) or 0),
                "weighted_value": float((data or {}).get("weighted_value", 0.0) or 0.0),
            })

    used = sorted(used, key=lambda x: (x["strength"], x["count"]), reverse=True)[:5]
    # Cap to prevent old highlight stats overpowering MMR.
    return float(min(2.0, total)), used


def _historic_style_components(team_a: List[str], team_b: List[str], state: Dict[str, Any]) -> Dict[str, Any]:
    """Return style penalties/bonuses for the fixed historic stats layer."""
    if not state.get("historic_style_enabled", False):
        return {
            "style_enabled": False,
            "style_penalty": 0.0,
            "style_link_bonus": 0.0,
            "style_net": 0.0,
        }

    sa = _team_style_profile(team_a, state)
    sb = _team_style_profile(team_b, state)
    link_a, links_a = _team_creator_link_bonus(team_a, state)
    link_b, links_b = _team_creator_link_bonus(team_b, state)

    finishing_diff = abs(float(sa["finishing"]) - float(sb["finishing"]))
    creation_diff = abs(float(sa["creation"]) - float(sb["creation"]))
    impact_diff = abs(float(sa["impact"]) - float(sb["impact"]))
    save_diff = abs(float(sa["save_impact"]) - float(sb["save_impact"]))
    clutch_diff = abs(float(sa["clutch"]) - float(sb["clutch"]))

    # Shortages matter only when the entire team lacks that role. This helps the
    # selector avoid "five finishers vs five creators" without forcing every
    # team to look identical.
    creator_shortage = 0.0
    finisher_shortage = 0.0
    if sa["creator_count"] == 0 or sb["creator_count"] == 0:
        creator_shortage += 0.75
    if sa["finisher_count"] == 0 or sb["finisher_count"] == 0:
        finisher_shortage += 0.75

    # Keep deliberately light: max contribution is only a few score points.
    style_penalty = (
        (finishing_diff * 1.65)
        + (creation_diff * 1.65)
        + (impact_diff * 1.10)
        + (save_diff * 0.65)
        + (clutch_diff * 0.60)
        + creator_shortage
        + finisher_shortage
    )
    style_penalty = float(min(4.0, style_penalty))

    # Reward proven creator -> scorer links, but only mildly and only as a tie-breaker.
    style_link_bonus = float(min(1.25, (link_a + link_b) * 0.30))
    style_net = float(style_penalty - style_link_bonus)

    return {
        "style_enabled": True,
        "style_a": sa,
        "style_b": sb,
        "style_finishing_a": float(sa["finishing"]),
        "style_finishing_b": float(sb["finishing"]),
        "style_creation_a": float(sa["creation"]),
        "style_creation_b": float(sb["creation"]),
        "style_impact_a": float(sa["impact"]),
        "style_impact_b": float(sb["impact"]),
        "style_save_a": float(sa["save_impact"]),
        "style_save_b": float(sb["save_impact"]),
        "style_clutch_a": float(sa["clutch"]),
        "style_clutch_b": float(sb["clutch"]),
        "style_finishing_diff": float(finishing_diff),
        "style_creation_diff": float(creation_diff),
        "style_impact_diff": float(impact_diff),
        "style_save_diff": float(save_diff),
        "style_clutch_diff": float(clutch_diff),
        "style_creator_shortage": float(creator_shortage),
        "style_finisher_shortage": float(finisher_shortage),
        "style_link_bonus": float(style_link_bonus),
        "style_link_bonus_a": float(link_a),
        "style_link_bonus_b": float(link_b),
        "style_links_a": links_a,
        "style_links_b": links_b,
        "style_penalty": float(style_penalty),
        "style_net": float(style_net),
    }


def _similarity_penalty(team_a: List[str], team_b: List[str], state: Dict[str, Any]) -> tuple[float, dict]:
    """Penalise repeating a very similar historic matchup that was one-sided.

    This is *results-only* and deliberately conservative.
    """
    matches = state.get("matches")
    if matches is None or not isinstance(matches, pd.DataFrame) or matches.empty:
        return 0.0, {"similarity": 0.0}

    A_now = set(team_a)
    B_now = set(team_b)

    best_sim = 0.0
    best_row = None
    best_swapped = False

    # 5v5 => max overlap is 10 (5 per side)
    for _, r in matches.iterrows():
        ta = [clean_name(p) for p in _split_team(r.get("team_a", "")) if clean_name(p)]
        tb = [clean_name(p) for p in _split_team(r.get("team_b", "")) if clean_name(p)]
        if len(ta) < 3 or len(tb) < 3:
            continue

        oa1 = len(A_now & set(ta))
        ob1 = len(B_now & set(tb))
        s1 = oa1 + ob1

        oa2 = len(A_now & set(tb))
        ob2 = len(B_now & set(ta))
        s2 = oa2 + ob2

        if s2 > s1:
            s = s2
            swapped = True
        else:
            s = s1
            swapped = False

        sim = s / 10.0
        if sim > best_sim:
            best_sim = sim
            best_row = r
            best_swapped = swapped

    if best_row is None:
        return 0.0, {"similarity": 0.0}

    # Only kick in when VERY similar
    if best_sim < 0.9:
        return 0.0, {"similarity": float(best_sim)}

    # Determine whether that match was one-sided
    res = str(best_row.get("result", "")).upper()
    gA, gB = _parse_score(best_row.get("score", ""))

    goal_diff = None
    if gA is not None and gB is not None:
        goal_diff = abs(int(gA) - int(gB))

    one_sided = False
    if goal_diff is not None:
        one_sided = goal_diff >= 5
    else:
        # If we don't have scorelines, be conservative: only treat as one-sided if not a draw
        one_sided = res in ("A", "B")

    if not one_sided:
        return 0.0, {"similarity": float(best_sim), "one_sided": False}

    # Penalty increases with similarity and goal diff (if available)
    diff_factor = 1.0
    if goal_diff is not None:
        diff_factor = min(1.6, 1.0 + (goal_diff - 3) * 0.15)

    pen = 8.0 * best_sim * diff_factor

    return float(pen), {
        "similarity": float(best_sim),
        "one_sided": True,
        "goal_diff": goal_diff,
        "swapped": bool(best_swapped),
        "match_id": best_row.get("id", None),
        "date": best_row.get("date", None),
        "score": best_row.get("score", None),
        "result": res,
    }


def _evaluate_with_state(team_a: List[str], team_b: List[str], state: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    """
    Core fairness evaluation using a provided engine state.
    Returns (fairness_score, breakdown). Lower = more balanced.
    """
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
    chem_a, chem_vals_a = _team_chemistry(team_a, state)
    chem_b, chem_vals_b = _team_chemistry(team_b, state)
    chem_diff = abs(chem_a - chem_b)

    # Chemistry density (prevents Frankenstein teams)
    pairs = max(1, (len(team_a) * (len(team_a) - 1)) // 2)  # 5v5 => 10
    dens_a = chem_a / float(pairs)
    dens_b = chem_b / float(pairs)
    dens_diff = abs(dens_a - dens_b)

    # Chemistry concentration (one super-duo + 3 dead links feels bad)
    def _top_share(vals: list[float]) -> float:
        if not vals:
            return 0.0
        total = sum(abs(v) for v in vals)
        if total <= 1e-9:
            return 0.0
        return max(abs(v) for v in vals) / total

    top_share_a = _top_share(chem_vals_a)
    top_share_b = _top_share(chem_vals_b)
    top_share_diff = abs(top_share_a - top_share_b)
    # Also penalise very high concentration on either team (not just imbalance)
    top_share_high = max(0.0, max(top_share_a, top_share_b) - 0.55)  # only when it's really a "duo carry"

    # Trio synergy (triangle chemistry; results-only)
    trio_a, trio_vals_a = _team_trio_synergy(team_a, state)
    trio_b, trio_vals_b = _team_trio_synergy(team_b, state)
    trio_diff = abs(trio_a - trio_b)

    # Trio density (stops 1 "magic triangle" doing all the work)
    trios = max(1, (len(team_a) * (len(team_a) - 1) * (len(team_a) - 2)) // 6)  # 5 => 10
    trio_dens_a = trio_a / float(trios)
    trio_dens_b = trio_b / float(trios)
    trio_dens_diff = abs(trio_dens_a - trio_dens_b)

    # Trio concentration
    top_trio_share_a = _top_share(trio_vals_a)
    top_trio_share_b = _top_share(trio_vals_b)
    top_trio_share_diff = abs(top_trio_share_a - top_trio_share_b)
    top_trio_share_high = max(0.0, max(top_trio_share_a, top_trio_share_b) - 0.55)

    # Bad pair penalties (known poor duos)
    bad_a = _team_badpair_badness(team_a, state)
    bad_b = _team_badpair_badness(team_b, state)

    # Raw bad-pair values can be noisy in a small/medium league. Keep them visible
    # in the breakdown, but soften their impact so one historic weak duo does not
    # ruin an otherwise well-balanced MMR/team-shape split.
    bad_total_raw = bad_a + bad_b
    bad_diff_raw = abs(bad_a - bad_b)

    # Human-style forgiveness: if the ratings are basically level, trust MMR more.
    bad_total = bad_total_raw
    bad_diff = bad_diff_raw
    if mmr_diff < 10.0:
        bad_total *= 0.60
        bad_diff *= 0.60
    elif mmr_diff < 20.0:
        bad_total *= 0.80
        bad_diff *= 0.80

    # Hard cap the contribution from bad-pair history.
    bad_total = min(bad_total, 4.0)
    bad_diff = min(bad_diff, 4.0)

    # Similarity penalty (avoid repeating very similar one-sided historical matchups).
    # In normal generation this is calculated live from cached match history.
    # In historical calibration states we pass similarity_penalty=0.0 because the
    # rolling calibration does not include full historical matchup memory.
    if "matches" in state:
        sim_pen, sim_debug = _similarity_penalty(team_a, team_b, state)
    else:
        sim_pen = float(state.get("similarity_penalty", 0.0) or 0.0)
        sim_debug = {}

    # Historic goals/assists/highlights layer. This is a fixed benchmark only:
    # it nudges team style balance and rewards proven creator -> scorer links.
    style = _historic_style_components(team_a, team_b, state)
    style_net = float(style.get("style_net", 0.0) or 0.0)

    # --- Weights (keep aligned with existing evaluate_teams) ---
    W_MMR = 1.00
    W_SPREAD = 0.70
    # Chemistry should influence the teams, not overpower a strong MMR split.
    W_CHEM = 0.05
    W_DENS = 0.12
    W_TOP_SHARE = 8.0
    W_TOP_HIGH = 10.0

    W_TRIO = 0.035
    W_TRIO_DENS = 0.08
    W_TRIO_TOP = 3.0
    W_TRIO_TOP_HIGH = 4.0

    # Bad-pair history is useful, but noisy at your current sample size.
    W_BAD_TOTAL = 0.02
    W_BAD_DIFF = 0.05

    W_SIM = 1.0

    # Final fairness score
    score = (
        (mmr_diff * W_MMR)
        + (spread_diff * W_SPREAD)
        + (chem_diff * W_CHEM)
        + (dens_diff * W_DENS)
        + (top_share_diff * W_TOP_SHARE)
        + (top_share_high * W_TOP_HIGH)
        + (trio_diff * W_TRIO)
        + (trio_dens_diff * W_TRIO_DENS)
        + (top_trio_share_diff * W_TRIO_TOP)
        + (top_trio_share_high * W_TRIO_TOP_HIGH)
        + (bad_total * W_BAD_TOTAL)
        + (bad_diff * W_BAD_DIFF)
        + (sim_pen * W_SIM)
        + style_net
    )

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
        "chem_density_a": dens_a,
        "chem_density_b": dens_b,
        "chem_density_diff": dens_diff,
        "chem_top_share_a": top_share_a,
        "chem_top_share_b": top_share_b,
        "chem_top_share_diff": top_share_diff,
        "chem_top_share_high": top_share_high,
        "trio_a": trio_a,
        "trio_b": trio_b,
        "trio_diff": trio_diff,
        "trio_density_a": trio_dens_a,
        "trio_density_b": trio_dens_b,
        "trio_density_diff": trio_dens_diff,
        "trio_top_share_a": top_trio_share_a,
        "trio_top_share_b": top_trio_share_b,
        "trio_top_share_diff": top_trio_share_diff,
        "trio_top_share_high": top_trio_share_high,
        "badpair_a": bad_a,
        "badpair_b": bad_b,
        "badpair_total": bad_total,
        "badpair_diff": bad_diff,
        "badpair_total_raw": bad_total_raw,
        "badpair_diff_raw": bad_diff_raw,
        "similarity_penalty": sim_pen,
        "similarity_debug": sim_debug,
    }
    breakdown.update(style)
    return float(score), breakdown


def evaluate_teams(team_a: List[str], team_b: List[str]) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (fairness_score, breakdown).
    Lower fairness_score = more balanced / more "playable".
    """
    state = get_engine_state(force_reload=False)
    return _evaluate_with_state(team_a, team_b, state)



# -----------------------------
# V2 goal-first team model
# -----------------------------

def _weighted_mean(rows: list[tuple[float, float]]) -> float | None:
    total_w = sum(float(w) for _, w in rows)
    if total_w <= 1e-9:
        return None
    return float(sum(float(v) * float(w) for v, w in rows) / total_w)


def _historical_margin_model(team_a: List[str], team_b: List[str], state: Dict[str, Any]) -> Dict[str, Any]:
    """Predict margin/close chance from similar historical matchups.

    V2 deliberately uses history as a predictor rather than a collection of hard
    penalties. For each past match, it checks how closely the current Team A/B
    shape matches either historical orientation, weights that match by overlap,
    and averages the real goal difference from those closest examples.
    """
    matches = state.get("matches")
    if matches is None or not isinstance(matches, pd.DataFrame) or matches.empty:
        return {"hist_margin": None, "hist_close_pct": None, "hist_n": 0, "hist_range": None, "hist_confidence": 0.0, "hist_avg_overlap": 0.0}

    A_now = {clean_name(p) for p in team_a if clean_name(p)}
    B_now = {clean_name(p) for p in team_b if clean_name(p)}
    if not A_now or not B_now:
        return {"hist_margin": None, "hist_close_pct": None, "hist_n": 0, "hist_range": None, "hist_confidence": 0.0, "hist_avg_overlap": 0.0}

    rows: list[dict[str, Any]] = []
    df = matches.copy()
    if "date" in df.columns:
        df["__dt"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["__dt"] = pd.NaT

    # Recency rank gives a small preference to newer games without ignoring older ones.
    df = df.sort_values("__dt", na_position="first").reset_index(drop=True)
    denom = max(1, len(df) - 1)

    for idx, r in df.iterrows():
        ta = {clean_name(p) for p in _split_team(r.get("team_a", "")) if clean_name(p)}
        tb = {clean_name(p) for p in _split_team(r.get("team_b", "")) if clean_name(p)}
        if len(ta) < 3 or len(tb) < 3:
            continue

        gA, gB = _parse_score(r.get("score", ""))
        if gA is None or gB is None:
            continue
        gd = abs(int(gA) - int(gB))

        # Same orientation and swapped orientation.
        oa1, ob1 = len(A_now & ta), len(B_now & tb)
        oa2, ob2 = len(A_now & tb), len(B_now & ta)
        s1, s2 = oa1 + ob1, oa2 + ob2
        if s2 > s1:
            oa, ob, overlap = oa2, ob2, s2
        else:
            oa, ob, overlap = oa1, ob1, s1

        min_side = min(oa, ob)
        if overlap < 5 or min_side < 2:
            continue

        overlap_ratio = overlap / 10.0
        side_balance = min_side / 5.0
        recency = 0.75 + 0.25 * (idx / denom)

        # High overlap matters a lot more than loose overlap. Side balance stops
        # one side matching well and the other side being random.
        weight = (overlap_ratio ** 3.0) * (0.60 + 0.40 * side_balance) * recency
        if weight <= 0:
            continue

        rows.append({
            "goal_diff": float(gd),
            "is_close": 1.0 if gd <= 2 else 0.0,
            "weight": float(weight),
            "overlap": int(overlap),
            "min_side": int(min_side),
        })

    if not rows:
        return {"hist_margin": None, "hist_close_pct": None, "hist_n": 0, "hist_range": None, "hist_confidence": 0.0, "hist_avg_overlap": 0.0}

    # Keep nearest / strongest examples only. This avoids distant examples washing out the signal.
    rows = sorted(rows, key=lambda x: (x["weight"], x["overlap"], x["min_side"]), reverse=True)[:30]
    margin = _weighted_mean([(r["goal_diff"], r["weight"]) for r in rows])
    close = _weighted_mean([(r["is_close"], r["weight"]) for r in rows])
    overlaps = [float(r["overlap"]) for r in rows]
    total_w = sum(float(r["weight"]) for r in rows)
    avg_overlap = _weighted_mean([(float(r["overlap"]), r["weight"]) for r in rows]) or 0.0

    # Confidence is intentionally conservative for a small league dataset.
    confidence = min(1.0, (len(rows) / 18.0) * (total_w / 6.0) * (avg_overlap / 8.0))

    if confidence >= 0.70:
        conf_label = "high"
    elif confidence >= 0.40:
        conf_label = "medium"
    else:
        conf_label = "low"

    return {
        "hist_margin": float(margin) if margin is not None else None,
        "hist_close_pct": float(close * 100.0) if close is not None else None,
        "hist_n": int(len(rows)),
        "hist_range": f"overlap {int(min(overlaps))}–{int(max(overlaps))}/10" if overlaps else None,
        "hist_confidence": float(confidence),
        "hist_confidence_label": conf_label,
        "hist_avg_overlap": float(avg_overlap),
    }


def evaluate_teams_v2(team_a: List[str], team_b: List[str]) -> Tuple[float, Dict[str, Any]]:
    """V2 goal-first team evaluation.

    Lower returned score = better expected game.
    The score is built around predicted goal margin first, then uses MMR/team
    shape/chemistry only as nudges. This is better suited to a 50-100 match
    five-a-side dataset than lots of hard duo/trio penalties.
    """
    state = get_engine_state(force_reload=False)

    # Reuse the existing tuned breakdown so the UI still has all familiar fields.
    old_score, breakdown = _evaluate_with_state(team_a, team_b, state)

    mmr_diff = float(breakdown.get("mmr_diff", 0.0) or 0.0)
    spread_diff = float(breakdown.get("spread_diff", 0.0) or 0.0)
    chem_diff = float(breakdown.get("chem_diff", 0.0) or 0.0)
    bad_total = float(breakdown.get("badpair_total", 0.0) or 0.0)
    sim_pen = float(breakdown.get("similarity_penalty", 0.0) or 0.0)
    style_net = float(breakdown.get("style_net", 0.0) or 0.0)

    hist = _historical_margin_model(team_a, team_b, state)
    hist_margin = hist.get("hist_margin")
    hist_close_pct = hist.get("hist_close_pct")
    hist_conf = float(hist.get("hist_confidence", 0.0) or 0.0)

    # Fallback model when historical overlap is weak. This stays simple and
    # readable: MMR diff is primary, shape/chemistry are nudges.
    fallback_margin = 1.15 + (mmr_diff / 18.0) + (spread_diff / 70.0) + (chem_diff / 45.0) + (bad_total / 18.0) + (sim_pen / 10.0) + (style_net / 5.0)
    fallback_margin = max(0.8, min(8.5, float(fallback_margin)))
    fallback_close = max(8.0, min(94.0, 94.0 - (fallback_margin * 11.0) - (mmr_diff * 0.20)))

    if hist_margin is not None:
        predicted_margin = (hist_conf * float(hist_margin)) + ((1.0 - hist_conf) * fallback_margin)
    else:
        predicted_margin = fallback_margin

    if hist_close_pct is not None:
        close_pct = (hist_conf * float(hist_close_pct)) + ((1.0 - hist_conf) * fallback_close)
    else:
        close_pct = fallback_close

    predicted_margin = max(0.6, min(9.0, float(predicted_margin)))
    close_pct = max(5.0, min(95.0, float(close_pct)))

    # Quality is user-facing. It rewards close chance and punishes high margins.
    quality = close_pct - max(0.0, predicted_margin - 2.0) * 7.5 - max(0.0, mmr_diff - 25.0) * 0.25 - max(0.0, style_net) * 1.2
    quality = max(0.0, min(100.0, float(quality)))

    # Ranking score: margin is king. MMR/spread/chemistry are tie-break nudges.
    v2_score = (
        predicted_margin
        + (mmr_diff / 140.0)
        + (spread_diff / 260.0)
        + (chem_diff / 180.0)
        + (bad_total / 80.0)
        + (sim_pen / 40.0)
        + (style_net / 12.0)
    )

    breakdown.update({
        "fairness_score": float(old_score),
        "v2_score": float(v2_score),
        "v2_predicted_margin": float(predicted_margin),
        "v2_close_pct": float(close_pct),
        "v2_quality": float(quality),
        "v2_method": "goal_first",
        "v2_hist_margin": hist_margin,
        "v2_hist_close_pct": hist_close_pct,
        "v2_hist_n": int(hist.get("hist_n", 0) or 0),
        "v2_hist_range": hist.get("hist_range"),
        "v2_hist_confidence": hist_conf,
        "v2_hist_confidence_label": hist.get("hist_confidence_label", "low"),
        "v2_hist_avg_overlap": float(hist.get("hist_avg_overlap", 0.0) or 0.0),
        "v2_fallback_margin": float(fallback_margin),
        "v2_fallback_close_pct": float(fallback_close),
    })

    return float(v2_score), breakdown

# -----------------------------
# True pre-match fairness calibration (no leakage)
# -----------------------------

def _parse_score_for_gd(score_txt: str) -> Tuple[Optional[int], Optional[int]]:
    s = (score_txt or "").strip()
    if not s:
        return None, None
    # Accept "7-5" or "7 – 5"
    s = s.replace("–", "-").replace("—", "-")
    m = re.search(r"(\d+)\s*-\s*(\d+)", s)
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None, None


def build_true_fairness_calibration(
    close_goal_diff: int = 2,
    bucket_size: float = 5.0,
) -> pd.DataFrame:
    """
    Returns a calibration dataframe with columns:
      - fairness_pre
      - goal_diff
      - is_close  (goal_diff <= close_goal_diff)
      - date
    computed using ONLY information available before each match.

    Notes:
    - Uses mmr_history.mmr_before when available for pre-match ratings.
    - Rebuilds form + duo + trio stats in a rolling manner (no future leakage).
    """
    conn = get_conn()
    try:
        matches = pd.read_sql(
            "SELECT id, date, team_a, team_b, result, score FROM matches WHERE result IN ('A','B','Draw','D');",
            conn,
        )
    except Exception:
        conn.close()
        return pd.DataFrame(columns=["fairness_pre", "goal_diff", "is_close", "date"])

    try:
        players = pd.read_sql("SELECT id, name, fitness FROM players;", conn)
    except Exception:
        players = pd.DataFrame(columns=["name", "fitness"])

    # mmr_history: allow either (player_id->players) or player_name directly, best-effort
    mmr_hist = None
    try:
        mmr_hist = pd.read_sql(
            "SELECT match_id, player_id, mmr_before FROM mmr_history;",
            conn,
        )
    except Exception:
        try:
            mmr_hist = pd.read_sql(
                "SELECT match_id, player_name, mmr_before FROM mmr_history;",
                conn,
            )
        except Exception:
            mmr_hist = None

    conn.close()

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.sort_values("date", na_position="last")

    # Fitness map (static)
    fitness_map: dict[str, str] = {}
    for _, r in players.iterrows():
        n = clean_name(str(r.get("name", "")))
        if n:
            fitness_map[n] = str(r.get("fitness", "Medium") or "Medium")

    # Helper: mmr_before lookup
    mmr_before_lookup: dict[tuple[int, str], float] = {}
    if mmr_hist is not None and not mmr_hist.empty:
        if "player_id" in mmr_hist.columns:
            # Need map player_id->name (if possible)
            # Best-effort: assume players table has "id" as well, but if not, skip.
            if "id" in players.columns:
                id2name = {int(r["id"]): clean_name(str(r["name"])) for _, r in players.iterrows() if str(r.get("name","")).strip()}
                for _, r in mmr_hist.iterrows():
                    try:
                        mid = int(r["match_id"])
                        pid = int(r["player_id"])
                        nm = id2name.get(pid, "")
                        if not nm:
                            continue
                        mmr_before_lookup[(mid, nm)] = float(r["mmr_before"])
                    except Exception:
                        continue
        elif "player_name" in mmr_hist.columns:
            for _, r in mmr_hist.iterrows():
                try:
                    mid = int(r["match_id"])
                    nm = clean_name(str(r["player_name"]))
                    if not nm:
                        continue
                    mmr_before_lookup[(mid, nm)] = float(r["mmr_before"])
                except Exception:
                    continue

    # Rolling stats
    from collections import defaultdict, deque
    total_matches = defaultdict(int)
    # form deque of last 8 outcomes (W=1, D=0.5, L=0)
    form_deques: dict[str, deque] = defaultdict(lambda: deque(maxlen=8))
    # player win-rate rolling sums
    player_score_sum = defaultdict(float)

    # duo/trio rolling (within-team outcome scores)
    duo_games = defaultdict(int)
    duo_score_sum = defaultdict(float)

    trio_games = defaultdict(int)
    trio_score_sum = defaultdict(float)

    def _team_pairs(team: List[str]):
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                yield (team[i], team[j])

    def _team_trios(team: List[str]):
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                for k in range(j + 1, len(team)):
                    yield (team[i], team[j], team[k])

    def _rolling_form_index(name: str) -> float:
        dq = form_deques.get(name)
        if not dq:
            return 0.5
        return float(sum(dq) / len(dq)) if len(dq) else 0.5

    def _rolling_win_rate(name: str) -> float:
        m = total_matches.get(name, 0)
        if not m:
            return 0.5
        return float(player_score_sum.get(name, 0.0) / float(m))

    def _build_state_for_match(match_id: int, team_a: List[str], team_b: List[str]) -> dict:
        # mmr_map from mmr_before where available; fallback to STARTING_MMR
        mmr_map = {}
        for p in set(team_a + team_b):
            mmr_map[p] = float(mmr_before_lookup.get((match_id, p), STARTING_MMR))

        # form_index
        form_index = {p: _rolling_form_index(p) for p in mmr_map.keys()}

        # base_chemistry from rolling duo outcomes (scaled)
        base_chemistry = {}
        for (a, b), g in duo_games.items():
            if g <= 0:
                continue
            # mean outcome 0..1 centered at 0.5
            mu = float(duo_score_sum.get((a, b), 0.0)) / float(g)
            centered = mu - 0.5
            # scale with sample, saturating
            sample_factor = min(1.0, g / 12.0)
            val = centered * 45.0 * sample_factor  # 60 chosen to give chemistry a meaningful range
            base_chemistry[(a, b)] = val

        # trio synergy from rolling trio outcomes (scaled)
        trio_synergy = {}
        for (a, b, c), g in trio_games.items():
            if g <= 0:
                continue
            mu = float(trio_score_sum.get((a, b, c), 0.0)) / float(g)
            centered = mu - 0.5
            sample_factor = min(1.0, g / 14.0)
            val = centered * 25.0 * sample_factor
            trio_synergy[(a, b, c)] = val

        # bad pair penalty based on rolling expectation
        bad_pair_penalty = {}
        MIN_GAMES_BAD = 10
        ABS_BAD_WINRATE = 0.38
        EXPECTED_DROP = 0.20
        MAX_PEN = 12.0

        for (a, b), g in duo_games.items():
            if g < MIN_GAMES_BAD:
                continue
            mu = float(duo_score_sum.get((a, b), 0.0)) / float(g)
            winrate = mu
            exp = 0.5 * (_rolling_win_rate(a) + _rolling_win_rate(b))
            drop = exp - winrate
            if winrate <= ABS_BAD_WINRATE and drop >= EXPECTED_DROP:
                sample_factor = min(1.0, g / 18.0)
                pen = min(MAX_PEN, 4.0 + (drop * 35.0)) * sample_factor
                bad_pair_penalty[(a, b)] = float(pen)

        return {
            "mmr_map": mmr_map,
            "fitness_map": fitness_map,
            "form_index": form_index,
            "total_matches": dict(total_matches),
            "base_chemistry": base_chemistry,
            "bad_pair_penalty": bad_pair_penalty,
            "trio_synergy": trio_synergy,
            # similarity_penalty not available historically; keep 0
            "similarity_penalty": 0.0,
        }

    rows = []
    for _, mrow in matches.iterrows():
        try:
            match_id = int(mrow["id"])
        except Exception:
            continue

        ta = [clean_name(p) for p in _split_team(mrow.get("team_a", "")) if clean_name(p)]
        tb = [clean_name(p) for p in _split_team(mrow.get("team_b", "")) if clean_name(p)]
        if not ta or not tb:
            continue

        state_pre = _build_state_for_match(match_id, ta, tb)
        fairness_pre, _ = _evaluate_with_state(ta, tb, state_pre)

        gA, gB = _parse_score_for_gd(str(mrow.get("score", "") or ""))
        gd = abs(gA - gB) if (gA is not None and gB is not None) else None

        rows.append(
            {
                "date": mrow.get("date"),
                "fairness_pre": float(fairness_pre),
                "goal_diff": gd,
                "is_close": (gd is not None and gd <= int(close_goal_diff)),
            }
        )

        # ---- update rolling stats using this match outcome ----
        res = str(mrow.get("result", "") or "").upper()
        if res == "DRAW":
            res = "D"
        if res == "A":
            out_a, out_b = 1.0, 0.0
        elif res == "B":
            out_a, out_b = 0.0, 1.0
        else:
            out_a, out_b = 0.5, 0.5

        for p in ta:
            total_matches[p] += 1
            player_score_sum[p] += out_a
            form_deques[p].append(out_a)
        for p in tb:
            total_matches[p] += 1
            player_score_sum[p] += out_b
            form_deques[p].append(out_b)

        # duo updates
        for team, sc in ((ta, out_a), (tb, out_b)):
            for a, b in _team_pairs(team):
                duo_games[(a, b)] += 1
                duo_games[(b, a)] += 1
                duo_score_sum[(a, b)] += sc
                duo_score_sum[(b, a)] += sc

        # trio updates
        for team, sc in ((ta, out_a), (tb, out_b)):
            for a, b, c in _team_trios(team):
                key = tuple(sorted((a, b, c)))
                trio_games[key] += 1
                trio_score_sum[key] += sc

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Remove rows without goal diff for calibration stats
    df = df.dropna(subset=["goal_diff"])
    return df


def calibration_lookup(
    fairness_score: float,
    calib_df: pd.DataFrame,
    bucket_size: float = 5.0,
    close_goal_diff: int = 2,
    nearest_n: int = 30,
) -> dict:
    """
    Interpret a generated matchup by comparing it to the closest historical
    pre-match fairness scores.

    This replaces the old fixed-bucket method. With a small/medium league data
    set, fixed buckets can make many different matchups show the same sample
    size and same numbers. Nearest-neighbour calibration gives each matchup its
    own local historical comparison group.
    """
    if calib_df is None or calib_df.empty:
        return {
            "quality": None,
            "close_pct": None,
            "typical_margin": None,
            "n": 0,
            "bucket": None,
            "range": None,
            "method": "none",
        }

    try:
        x = float(fairness_score)
    except Exception:
        return {
            "quality": None,
            "close_pct": None,
            "typical_margin": None,
            "n": 0,
            "bucket": None,
            "range": None,
            "method": "none",
        }

    d = calib_df.copy()
    if "fairness_pre" not in d.columns or "goal_diff" not in d.columns:
        return {
            "quality": None,
            "close_pct": None,
            "typical_margin": None,
            "n": 0,
            "bucket": None,
            "range": None,
            "method": "none",
        }

    d["fairness_pre"] = pd.to_numeric(d["fairness_pre"], errors="coerce")
    d["goal_diff"] = pd.to_numeric(d["goal_diff"], errors="coerce")
    d = d.dropna(subset=["fairness_pre", "goal_diff"])
    if d.empty:
        return {
            "quality": None,
            "close_pct": None,
            "typical_margin": None,
            "n": 0,
            "bucket": None,
            "range": None,
            "method": "none",
        }

    k = int(max(8, min(int(nearest_n or 30), len(d))))
    d["distance"] = (d["fairness_pre"] - x).abs()
    nearest = d.sort_values(["distance", "date"], ascending=[True, False], na_position="last").head(k)

    n = int(len(nearest))
    if n <= 0:
        return {
            "quality": None,
            "close_pct": None,
            "typical_margin": None,
            "n": 0,
            "bucket": None,
            "range": None,
            "method": "none",
        }

    if "is_close" not in nearest.columns:
        nearest["is_close"] = nearest["goal_diff"] <= int(close_goal_diff)

    close_pct = float(nearest["is_close"].astype(bool).mean() * 100.0)
    typical_margin = float(nearest["goal_diff"].mean())

    fair_min = float(nearest["fairness_pre"].min())
    fair_max = float(nearest["fairness_pre"].max())
    fair_span = fair_max - fair_min

    # Quality is mostly the historic close-game rate, with a small penalty for
    # higher average margins and a small uncertainty penalty if the nearest
    # comparison range is very wide.
    quality = close_pct
    quality -= max(0.0, (typical_margin - close_goal_diff) * 8.0)
    quality -= max(0.0, (fair_span - 25.0) * 0.25)
    quality = max(0.0, min(100.0, float(quality)))

    range_txt = f"{fair_min:.1f}–{fair_max:.1f}"

    if fair_span <= 12:
        confidence = "high"
    elif fair_span <= 25:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "quality": quality,
        "close_pct": close_pct,
        "typical_margin": typical_margin,
        "n": n,
        "bucket": f"range {range_txt}",
        "range": range_txt,
        "range_span": fair_span,
        "confidence": confidence,
        "method": "nearest",
    }
