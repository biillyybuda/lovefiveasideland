# utils/team_ai_engine.py
import re
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


_ENGINE_CACHE: Dict[str, Any] | None = None


def _load_db_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load matches and players from DB (BASIC MODE: results-only)."""
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

    # --- Duo outcomes (results-only) for "known bad pairing" penalties ---
    # Track: games, score_mean (W=1,D=0.5,L=0), and sample size.
    duo_games: Dict[tuple[str, str], int] = defaultdict(int)
    duo_score_sum: Dict[tuple[str, str], float] = defaultdict(float)

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
    bad_pair_penalty: Dict[tuple[str, str], float] = {}

    # Tunables (kept conservative so we don't overfit)
    MIN_GAMES_BAD = 5
    ABS_BAD_WINRATE = 0.42         # below this is suspicious in your league
    EXPECTED_DROP = 0.15           # how far below expected (by players) counts as "bad"
    MAX_PEN = 28.0                 # cap per pair

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
            sample_factor = min(1.0, g / 12.0)  # 0..1
            pen = min(MAX_PEN, 10.0 + (drop * 70.0)) * sample_factor
            bad_pair_penalty[(a, b)] = float(pen)

    # --- Trio outcomes (results-only) for trio synergy (triangle chemistry) ---
    # Track: games, score_mean (W=1,D=0.5,L=0), and sample size.
    trio_games: Dict[tuple[str, str, str], int] = defaultdict(int)
    trio_score_sum: Dict[tuple[str, str, str], float] = defaultdict(float)

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
    trio_synergy: Dict[tuple[str, str, str], float] = {}

    # Tunables (conservative, avoids overfitting)
    MIN_GAMES_TRIO = 4
    MAX_TRIO = 36.0  # cap per trio

    for key, g in trio_games.items():
        if g < MIN_GAMES_TRIO:
            continue
        sc = float(trio_score_sum.get(key, 0.0))
        winrate = sc / float(g) if g else 0.5

        a, b, c = key
        exp = (float(player_win_rate.get(a, 0.5)) + float(player_win_rate.get(b, 0.5)) + float(player_win_rate.get(c, 0.5))) / 3.0
        delta = winrate - exp

        # sample grows confidence but saturates (0..1)
        sample_factor = min(1.0, g / 10.0)
        val = delta * 70.0 * sample_factor

        # tiny deltas are noise
        if abs(val) < 2.0:
            continue

        trio_synergy[key] = float(max(-MAX_TRIO, min(MAX_TRIO, val)))


    return {
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


def get_engine_state(force_reload: bool = True) -> Dict[str, Any]:
    """Engine state (MMR + fitness + form + chemistry + duo penalties)."""
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


def _pair_list(team: List[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
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


def _trio_list(team: List[str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
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
    if best_sim < 0.8:
        return 0.0, {"similarity": float(best_sim)}

    # Determine whether that match was one-sided
    res = str(best_row.get("result", "")).upper()
    gA, gB = _parse_score(best_row.get("score", ""))

    goal_diff = None
    if gA is not None and gB is not None:
        goal_diff = abs(int(gA) - int(gB))

    one_sided = False
    if goal_diff is not None:
        one_sided = goal_diff >= 4
    else:
        # If we don't have scorelines, be conservative: only treat as one-sided if not a draw
        one_sided = res in ("A", "B")

    if not one_sided:
        return 0.0, {"similarity": float(best_sim), "one_sided": False}

    # Penalty increases with similarity and goal diff (if available)
    diff_factor = 1.0
    if goal_diff is not None:
        diff_factor = min(1.6, 1.0 + (goal_diff - 3) * 0.15)

    pen = 18.0 * best_sim * diff_factor

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


def evaluate_teams(team_a: List[str], team_b: List[str]) -> tuple[float, Dict[str, Any]]:
    """
    Returns (fairness_score, breakdown).
    Lower fairness_score = more balanced / more "playable".
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
    trios = max(1, (len(team_a) * (len(team_a) - 1) * (len(team_a) - 2)) // 6)  # 5v5 => 10
    trio_dens_a = trio_a / float(trios)
    trio_dens_b = trio_b / float(trios)
    trio_dens_diff = abs(trio_dens_a - trio_dens_b)

    # Penalise very negative total trio synergy (means the triangle historically underperforms)
    trio_neg_total = max(0.0, -trio_a) + max(0.0, -trio_b)

# Known bad pairing penalty (results-only)
    bad_a = _team_badpair_badness(team_a, state)
    bad_b = _team_badpair_badness(team_b, state)
    bad_total = bad_a + bad_b
    bad_diff = abs(bad_a - bad_b)

    # Similar-matchup memory penalty (results-only)
    sim_pen, sim_dbg = _similarity_penalty(team_a, team_b, state)

    # Combine weights
    # Base: MMR is still main anchor
    W_MMR = 1.0
    W_SPREAD = 0.25

    # Chemistry now matters *a lot more*, especially when MMR is already close
    if mmr_diff < 20.0:
        W_CHEM = 0.40
        W_DENS = 0.30
        W_TOPSHARE = 0.10
        W_TRIO = 0.28
        W_TRIO_DENS = 0.18
        W_TRIO_NEG = 0.14
    else:
        W_CHEM = 0.22
        W_DENS = 0.18
        W_TOPSHARE = 0.08
        W_TRIO = 0.16
        W_TRIO_DENS = 0.10
        W_TRIO_NEG = 0.10

    # Bad pairings and repeating bad old matchups: trust builders
    W_BAD_TOTAL = 0.22     # avoid bad pairs full stop
    W_BAD_DIFF = 0.08      # also don't dump them all on one team
    W_SIM = 0.18

    fairness_score = (
        (W_MMR * mmr_diff)
        + (W_SPREAD * spread_diff)
        + (W_CHEM * chem_diff)
        + (W_DENS * dens_diff)
        + (W_TOPSHARE * (top_share_diff * 25.0 + top_share_high * 25.0))
        + (W_TRIO * trio_diff)
        + (W_TRIO_DENS * (trio_dens_diff * 25.0))
        + (W_TRIO_NEG * trio_neg_total)
        + (W_BAD_TOTAL * bad_total)
        + (W_BAD_DIFF * bad_diff)
        + (W_SIM * sim_pen)
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
        "trio_a": trio_a,
        "trio_b": trio_b,
        "trio_diff": trio_diff,
        "trio_density_a": trio_dens_a,
        "trio_density_b": trio_dens_b,
        "trio_density_diff": trio_dens_diff,
        "trio_negative_total": trio_neg_total,
        "badpair_a": bad_a,
        "badpair_b": bad_b,
        "badpair_total": bad_total,
        "badpair_diff": bad_diff,
        "similarity_penalty": sim_pen,
        "similarity_debug": sim_dbg,
        "fairness_score": fairness_score,
        "weights": {
            "W_MMR": W_MMR,
            "W_SPREAD": W_SPREAD,
            "W_CHEM": W_CHEM,
            "W_DENS": W_DENS,
            "W_TOPSHARE": W_TOPSHARE,
            "W_TRIO": W_TRIO,
            "W_TRIO_DENS": W_TRIO_DENS,
            "W_TRIO_NEG": W_TRIO_NEG,
            "W_BAD_TOTAL": W_BAD_TOTAL,
            "W_BAD_DIFF": W_BAD_DIFF,
            "W_SIM": W_SIM,
        },
        "mode": "ENHANCED (MMR + fitness + form + spread + duo chemistry + trio synergy + bad-pair + matchup memory)",
    }
    return fairness_score, breakdown
