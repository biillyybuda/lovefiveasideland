import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from scipy.stats import poisson

# We reuse your ladder snapping + odds conversion by importing from team_generator_page via safe copies here.
# Keep this file self-contained so it can be imported anywhere.

_BOOKIE_LADDER_FRAC: list[str] = [
    "1/100", "1/50", "1/33", "1/25", "1/20", "1/16", "1/14", "1/12", "1/11", "1/10",
    "1/9", "1/8", "1/7", "1/6", "1/5", "2/9", "1/4", "2/7", "1/3", "4/11", "2/5",
    "4/9", "1/2", "8/15", "4/7", "8/13", "4/6", "4/5", "5/6", "10/11",
    "Evens", "11/10", "6/5", "5/4", "11/8", "6/4", "13/8", "7/4", "15/8", "2/1",
    "9/4", "5/2", "11/4", "3/1", "10/3", "7/2", "4/1", "9/2", "5/1", "6/1", "7/1",
    "8/1", "9/1", "10/1", "12/1", "14/1", "16/1", "20/1", "25/1", "33/1", "40/1", "50/1",
]

def _frac_to_decimal(frac: str) -> float:
    s = str(frac).strip().lower()
    if s in ("evens", "even", "evs", "ev"):
        return 2.0
    if "/" in s:
        a, b = s.split("/", 1)
        return 1.0 + (float(a) / float(b))
    return float("nan")

_BOOKIE_LADDER_DEC: list[tuple[str, float]] = [(f, _frac_to_decimal(f)) for f in _BOOKIE_LADDER_FRAC]

def snap_decimal_to_bookie_ladder(decimal_odds: float) -> tuple[str, float]:
    try:
        d = float(decimal_odds)
        if not math.isfinite(d) or d <= 1.0:
            return "—", float("nan")
        target = math.log(d)
        best = None
        best_dist = float("inf")
        for label, dec in _BOOKIE_LADDER_DEC:
            if not math.isfinite(dec) or dec <= 1.0:
                continue
            dist = abs(math.log(dec) - target)
            if dist < best_dist:
                best_dist = dist
                best = (label, dec)
        return best if best else ("—", float("nan"))
    except Exception:
        return "—", float("nan")

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def _apply_overround(probs: Dict[str, float], overround: float = 1.06) -> Dict[str, float]:
    # Normalize then apply margin WITHOUT renormalizing (so implied probs sum > 1)
    keys = list(probs.keys())
    p = np.array([_clamp(probs[k], 1e-9, 1.0) for k in keys], dtype=float)
    p = p / p.sum()
    p = p * float(overround)
    return {k: float(v) for k, v in zip(keys, p)}

def _probs_to_prices(probs: Dict[str, float], overround: float = 1.06) -> Dict[str, Dict[str, float | str]]:
    adj = _apply_overround(probs, overround=overround)
    out: Dict[str, Dict[str, float | str]] = {}
    for k, p in adj.items():
        p = _clamp(p, 1e-9, 0.999999)
        dec = 1.0 / p
        lab, dec_s = snap_decimal_to_bookie_ladder(dec)
        out[k] = {"p": p, "dec": float(dec_s), "label": lab}
    return out

def _score_matrix(lam_a: float, lam_b: float, max_goals: int = 15) -> np.ndarray:
    lam_a = max(0.1, float(lam_a))
    lam_b = max(0.1, float(lam_b))
    pa = poisson.pmf(np.arange(0, max_goals + 1), lam_a)
    pb = poisson.pmf(np.arange(0, max_goals + 1), lam_b)
    return np.outer(pa, pb)  # shape (A goals, B goals)

def _prob_1x2(M: np.ndarray) -> Dict[str, float]:
    max_g = M.shape[0] - 1
    p1 = pX = p2 = 0.0
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = float(M[i, j])
            if i > j:
                p1 += p
            elif i == j:
                pX += p
            else:
                p2 += p
    return {"1": p1, "X": pX, "2": p2}


def _prob_win_margin(M: np.ndarray) -> Dict[str, float]:
    # M[i,j] = P(A=i,B=j)
    max_g = M.shape[0] - 1
    probs = {
        "A 1–2": 0.0,
        "A 3+": 0.0,
        "Draw": 0.0,
        "B 1–2": 0.0,
        "B 3+": 0.0,
    }
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = float(M[i, j])
            d = i - j
            if d == 0:
                probs["Draw"] += p
            elif d in (1, 2):
                probs["A 1–2"] += p
            elif d >= 3:
                probs["A 3+"] += p
            elif d in (-1, -2):
                probs["B 1–2"] += p
            else:  # d <= -3
                probs["B 3+"] += p
    return probs

def _prob_totals_ou(M: np.ndarray, line: float) -> Dict[str, float]:
    # Over line means total >= ceil(line+0.5). With line=9.5 => total>=10
    thresh = int(math.floor(line + 0.5)) + 1  # 9.5 -> 10
    max_g = M.shape[0] - 1
    over = 0.0
    under = 0.0
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = float(M[i, j])
            if (i + j) >= thresh:
                over += p
            else:
                under += p
    return {f"Over {line}": over, f"Under {line}": under}

def _best_totals_line(M: np.ndarray, lo: float = 3.5, hi: float = 17.5, step: float = 1.0) -> float:
    """
    Choose the totals line where Over/Under is closest to 50/50 (i.e., bookie's "main line").
    Lines are x.5 (e.g. 7.5, 8.5, 9.5).
    """
    best_line = 9.5
    best_diff = 999.0
    x = lo
    while x <= hi + 1e-9:
        # enforce .5 lines
        line = round(x * 2) / 2.0
        if abs(line - round(line)) < 1e-9:
            line += 0.5

        probs = _prob_totals_ou(M, line)
        p_over = probs.get(f"Over {line}", 0.0)
        diff = abs(p_over - 0.5)
        if diff < best_diff:
            best_diff = diff
            best_line = line
        x += step
    return best_line

import pandas as pd
from utils.team_ai_engine import get_engine_state  # uses DB matches + players

def _parse_score_from_row(r: pd.Series) -> tuple[int | None, int | None]:
    # Try common DB fields first
    for a_key, b_key in [
        ("score_a", "score_b"),
        ("goals_a", "goals_b"),
        ("team_a_score", "team_b_score"),
        ("team_a_goals", "team_b_goals"),
    ]:
        if a_key in r and b_key in r:
            try:
                ga = int(r.get(a_key)) # type: ignore
                gb = int(r.get(b_key)) # type: ignore
                return ga, gb
            except Exception:
                pass

    # Fallback: scoreline string like "6-8" or "6–8"
    for k in ["scoreline", "score", "final_score", "result_score"]:
        if k in r and r.get(k):
            s = str(r.get(k))
            s = s.replace("–", "-")
            if "-" in s:
                try:
                    a, b = s.split("-", 1)
                    return int(a.strip()), int(b.strip())
                except Exception:
                    pass
    return None, None


def _split_team(val: str) -> list[str]:
    raw = str(val or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    parts = raw.split(",")
    cleaned: list[str] = []
    for p in parts:
        name = p.strip().strip("'").strip('"')
        if name:
            cleaned.append(name.strip().lower())
    return cleaned


def estimate_expected_goals_from_history(
    team_a_now: list[str],
    team_b_now: list[str],
    *,
    k_shrink: float = 8.0,
) -> tuple[float, float, dict]:
    """
    Returns (lam_a, lam_b, debug)
    Uses match scores to learn:
      - league avg goals for/against per team
      - per-player attack & defence factors (Bayesian shrinkage)
      - lineup attack/defence (average of player factors)
    """
    state = get_engine_state(force_reload=False)
    matches: pd.DataFrame = state["matches"].copy()

    if matches.empty:
        # fallback neutral
        return 3.0, 3.0, {"mode": "no_matches"}

    # Parse scores + teams
    rows = []
    for _, r in matches.iterrows():
        ta = _split_team(r.get("team_a", ""))
        tb = _split_team(r.get("team_b", ""))
        ga, gb = _parse_score_from_row(r)

        if ga is None or gb is None:
            continue

        # store lowercase names for matching
        ta = [x.strip().lower() for x in ta if x]
        tb = [x.strip().lower() for x in tb if x]

        rows.append((ta, tb, ga, gb))

    if not rows:
        return 3.0, 3.0, {"mode": "no_scored_rows"}

    # League baselines (per team per match)
    league_gf = float(np.mean([ga for _, _, ga, _ in rows] + [gb for _, _, _, gb in rows]))
    league_ga = league_gf  # symmetric in aggregate

    # Per-player aggregates
    # attack: goals_for when player played
    # defence: goals_against when player played
    gf = {}
    ga = {}
    n = {}

    def add_player(p, goals_for, goals_against):
        gf[p] = gf.get(p, 0.0) + goals_for
        ga[p] = ga.get(p, 0.0) + goals_against
        n[p] = n.get(p, 0) + 1

    for ta, tb, goals_a, goals_b in rows:
        for p in ta:
            add_player(p, goals_a, goals_b)
        for p in tb:
            add_player(p, goals_b, goals_a)

    def player_attack(p: str) -> float:
        # (gf + k*league)/(n+k) divided by league => factor around 1
        pp = p.lower()
        m = n.get(pp, 0)
        return (((gf.get(pp, 0.0) + k_shrink * league_gf) / (m + k_shrink)) / league_gf)

    def player_defence(p: str) -> float:
        # concede factor: >1 concedes more
        pp = p.lower()
        m = n.get(pp, 0)
        return (((ga.get(pp, 0.0) + k_shrink * league_ga) / (m + k_shrink)) / league_ga)

    # Team factors (average)
    def team_attack(team: list[str]) -> float:
        vals = [player_attack(p) for p in team]
        return float(np.mean(vals)) if vals else 1.0

    def team_def(team: list[str]) -> float:
        vals = [player_defence(p) for p in team]
        return float(np.mean(vals)) if vals else 1.0

    A = [x.strip().lower() for x in team_a_now]
    B = [x.strip().lower() for x in team_b_now]

    attA = team_attack(A)
    defA = team_def(A)
    attB = team_attack(B)
    defB = team_def(B)

    lam_a = max(0.3, league_gf * attA * defB)
    lam_b = max(0.3, league_gf * attB * defA)

    debug = {
        "mode": "history",
        "league_gf_per_team": league_gf,
        "attA": attA, "defA": defA,
        "attB": attB, "defB": defB,
        "lam_a": lam_a, "lam_b": lam_b,
        "player_samples_min": min([n.get(p.lower(), 0) for p in (A + B)] or [0]),
    }
    return lam_a, lam_b, debug


def blended_expected_goals(
    team_a_now: list[str],
    team_b_now: list[str],
    mmr_lam_a: float,
    mmr_lam_b: float,
) -> tuple[float, float, dict]:
    """
    Blend MMR-based lambdas with history-based lambdas.
    As data grows, history dominates.
    """
    hist_a, hist_b, dbg = estimate_expected_goals_from_history(team_a_now, team_b_now)

    # Use sample size to decide blend weight
    # If few games, lean on MMR; if many, lean on history
    # We approximate using min matches among involved players
    m_min = int(dbg.get("player_samples_min", 0) or 0)
    w_hist = min(0.85, max(0.15, m_min / 12.0))  # ramps up to 0.85 around 12 matches
    w_mmr = 1.0 - w_hist

    lam_a = w_mmr * float(mmr_lam_a) + w_hist * float(hist_a)
    lam_b = w_mmr * float(mmr_lam_b) + w_hist * float(hist_b)

    dbg["mode"] = "blend"
    dbg["w_hist"] = w_hist
    dbg["w_mmr"] = w_mmr
    dbg["mmr_lam_a"] = float(mmr_lam_a)
    dbg["mmr_lam_b"] = float(mmr_lam_b)
    dbg["blend_lam_a"] = float(lam_a)
    dbg["blend_lam_b"] = float(lam_b)
    return lam_a, lam_b, dbg



def build_markets(
    exp_goals_a: float,
    exp_goals_b: float,
    *,
    overround: float = 1.06,
    max_goals: int = 15,
    total_lines: List[float] | None = None,
    include_alt_lines: bool = True,
) -> Dict[str, Dict]:
    """
    Returns:
      {
        "winning_margin": {"prices": {...}},
        "total_goals": {"main_line": 8.5, "lines": {8.5: {...}, 10.5: {...}}}
      }

    If total_lines is None:
      - Picks the "main" totals line near evens
      - Optionally includes +/- 2 goals as alternates
    """
    M = _score_matrix(exp_goals_a, exp_goals_b, max_goals=max_goals)
    mx_probs = _prob_1x2(M)
    mx_prices = _probs_to_prices(mx_probs, overround=overround)


    wm_probs = _prob_win_margin(M)
    wm_prices = _probs_to_prices(wm_probs, overround=overround)

    if total_lines is None:
        main = _best_totals_line(M, lo=3.5, hi=17.5, step=1.0)
        lines = [main]
        if include_alt_lines:
            lines = [max(0.5, main - 2.0), main, main + 2.0]
        total_lines = sorted(list({round(l * 2) / 2.0 for l in lines}))
    else:
        main = total_lines[0] if total_lines else 9.5

    totals = {}
    for ln in total_lines:
        ou_probs = _prob_totals_ou(M, ln)
        totals[ln] = _probs_to_prices(ou_probs, overround=overround)

    return {
        "match_odds": {"prices": mx_prices},
        "winning_margin": {"prices": wm_prices},
        "total_goals": {"main_line": float(main), "lines": totals},
    }
