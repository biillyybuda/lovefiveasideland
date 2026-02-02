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

def _prob_1x2_smart(
    M: np.ndarray,
    team_a_now: list[str],
    team_b_now: list[str],
) -> tuple[Dict[str, float], dict]:
    """
    Start from Poisson 1X2, then apply a small head-to-head adjustment (4v4+ similar meetings),
    shrunk for subs and adjusted for chemistry/quality.
    """
    base = _prob_1x2(M)  # {"1","X","2"}
    delta, dbg = _head_to_head_adjustment(team_a_now, team_b_now)

    p1 = float(base["1"])
    pX = float(base["X"])
    p2 = float(base["2"])

    # Apply delta to the win probs, keep draw fixed, then renormalize
    p1 = _clamp(p1 + delta, 1e-6, 0.999)
    p2 = _clamp(p2 - delta, 1e-6, 0.999)

    s = p1 + pX + p2
    if s <= 0:
        return base, dbg

    out = {"1": p1 / s, "X": pX / s, "2": p2 / s}
    dbg["base"] = base
    dbg["after"] = out
    return out, dbg


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
from utils.team_ai_engine import clean_name

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

def _team_key_set(team: list[str]) -> set[str]:
    return {clean_name(p) for p in (team or []) if str(p).strip()}

def _overlap_count(team_now: list[str], team_hist: list[str]) -> int:
    return len(_team_key_set(team_now) & _team_key_set(team_hist))

def _best_orientation(
    team_a_now: list[str],
    team_b_now: list[str],
    ta_hist: list[str],
    tb_hist: list[str],
) -> tuple[str, int, int, list[str], list[str]]:
    """
    Returns (orientation, overlapA, overlapB, histA_aligned, histB_aligned)
    orientation: "same" or "swapped"
    """
    oa1 = _overlap_count(team_a_now, ta_hist)
    ob1 = _overlap_count(team_b_now, tb_hist)
    score1 = oa1 + ob1

    oa2 = _overlap_count(team_a_now, tb_hist)
    ob2 = _overlap_count(team_b_now, ta_hist)
    score2 = oa2 + ob2

    if score2 > score1:
        return "swapped", oa2, ob2, tb_hist, ta_hist
    return "same", oa1, ob1, ta_hist, tb_hist

def _avg_chem_with_team(p: str, team: list[str], base_chem: dict) -> float:
    """
    Average chemistry between p and everyone in team (excluding p).
    base_chem keys can be (a,b) tuples or "a|b" strings (your engine supports both).
    """
    pk = clean_name(p)
    vals = []
    for t in team:
        tk = clean_name(t)
        if tk == pk:
            continue

        v = 0.0
        if (pk, tk) in base_chem:
            v = float(base_chem.get((pk, tk)) or 0.0)
        elif (tk, pk) in base_chem:
            v = float(base_chem.get((tk, pk)) or 0.0)
        else:
            s1 = f"{pk}|{tk}"
            s2 = f"{tk}|{pk}"
            if s1 in base_chem:
                v = float(base_chem.get(s1) or 0.0)
            elif s2 in base_chem:
                v = float(base_chem.get(s2) or 0.0)

        vals.append(v)

    return float(np.mean(vals)) if vals else 0.0

def _head_to_head_adjustment(
    team_a_now: list[str],
    team_b_now: list[str],
    *,
    min_overlap: int = 4,
    top_n: int = 3,
    gd_cap: int = 6,
    max_shift: float = 0.08,
) -> tuple[float, dict]:
    """
    Returns (delta_to_teamA_win_prob, debug)
    Positive delta => nudges towards Team A.

    Uses similar historical meetings with >= min_overlap on BOTH sides (4v4+),
    allowing swapped orientation. Dominance is capped by gd_cap and shrunk by:
      - how many players changed (subs)
      - MMR quality difference between sub-ins vs sub-outs
      - chemistry fit of sub-ins vs sub-outs
    """
    state = get_engine_state(force_reload=False)
    matches: pd.DataFrame = state.get("matches", pd.DataFrame()).copy()
    players: pd.DataFrame = state.get("players", pd.DataFrame()).copy()
    base_chem: dict = state.get("base_chemistry", {}) or {}

    if matches is None or matches.empty:
        return 0.0, {"mode": "no_matches"}

    # Most recent match should not be used to price "right now" (leakage).
    # But it SHOULD be used a month later. We’ll do this by only dropping if it's the very latest row.
    # (If date exists, drop the newest; else drop last row.)
    try:
        if "date" in matches.columns:
            dt = pd.to_datetime(matches["date"], errors="coerce")
            if dt.notna().any():
                matches = matches.drop(index=dt.idxmax())
        else:
            matches = matches.iloc[:-1]
    except Exception:
        pass

    # MMR lookup (for sub quality)
    mmr_map = {}
    try:
        if not players.empty and "name" in players.columns and "mmr" in players.columns:
            for _, r in players.iterrows():
                mmr_map[clean_name(str(r.get("name") or ""))] = float(r.get("mmr") or 0.0)
    except Exception:
        mmr_map = {}

    def mmr_of(name: str) -> float:
        return float(mmr_map.get(clean_name(name), 0.0))

    # scan for similar meetings
    candidates = []
    for _, r in matches.iterrows():
        ta_hist = _split_team(r.get("team_a", ""))
        tb_hist = _split_team(r.get("team_b", ""))
        if not ta_hist or not tb_hist:
            continue

        ga, gb = _parse_score_from_row(r)
        if ga is None or gb is None:
            continue

        orient, oa, ob, histA, histB = _best_orientation(team_a_now, team_b_now, ta_hist, tb_hist)

        if oa < min_overlap or ob < min_overlap:
            continue

        # Align score to orientation
        # If swapped, then "Team A now" corresponds to historical team_b,
        # so goals for "aligned A" are gb not ga.
        if orient == "same":
            gA, gB = int(ga), int(gb)
        else:
            gA, gB = int(gb), int(ga)

        # dominance signal (capped GD)
        gd = gA - gB
        gd = max(-gd_cap, min(gd_cap, gd))

        # subs compared to this historical aligned matchup
        nowA = _team_key_set(team_a_now); nowB = _team_key_set(team_b_now)
        histA_set = _team_key_set(histA);  histB_set = _team_key_set(histB)

        sub_in_A = [p for p in team_a_now if clean_name(p) not in histA_set]
        sub_out_A = [p for p in histA if clean_name(p) not in nowA]
        sub_in_B = [p for p in team_b_now if clean_name(p) not in histB_set]
        sub_out_B = [p for p in histB if clean_name(p) not in nowB]

        # shrink for number of swaps (each swap reduces transferability a bit)
        swaps = max(len(sub_in_A), len(sub_out_A)) + max(len(sub_in_B), len(sub_out_B))
        w_swaps = 0.92 ** swaps  # 1 swap ≈ 0.92, 2 swaps ≈ 0.85, etc.

        # adjust for quality of replacements (MMR in - MMR out)
        def rep_delta(sub_in, sub_out):
            if not sub_in and not sub_out:
                return 0.0
            mmr_in = float(np.mean([mmr_of(x) for x in sub_in])) if sub_in else 0.0
            mmr_out = float(np.mean([mmr_of(x) for x in sub_out])) if sub_out else 0.0
            return mmr_in - mmr_out

        # Positive means Team A got stronger vs that historical match
        repA = rep_delta(sub_in_A, sub_out_A)
        repB = rep_delta(sub_in_B, sub_out_B)
        # Convert to a mild multiplier (don’t let MMR dominate)
        w_rep = _clamp(1.0 + (repA - repB) / 1200.0, 0.75, 1.25)

        # adjust for chemistry fit of incoming vs outgoing
        def chem_delta(sub_in, sub_out, team_now):
            if not sub_in and not sub_out:
                return 0.0
            ci = float(np.mean([_avg_chem_with_team(x, team_now, base_chem) for x in sub_in])) if sub_in else 0.0
            co = float(np.mean([_avg_chem_with_team(x, team_now, base_chem) for x in sub_out])) if sub_out else 0.0
            return ci - co

        chemA = chem_delta(sub_in_A, sub_out_A, team_a_now)
        chemB = chem_delta(sub_in_B, sub_out_B, team_b_now)
        # Mild multiplier
        w_chem = _clamp(1.0 + (chemA - chemB) / 8.0, 0.80, 1.20)

        # Recency weight (if date exists)
        w_time = 1.0
        try:
            val = r.get("date", None)
            if val is not None and str(val).strip() != "":
                d = pd.to_datetime(str(val), errors="coerce")
                if pd.notna(d):
                    days = (pd.Timestamp.utcnow() - d).days
                    w_time = float(np.exp(-days / 30.0))
        except Exception:
            w_time = 1.0

        w = float(w_swaps * w_rep * w_chem * w_time)

        candidates.append(
            {
                "w": w,
                "gd": gd,
                "orient": orient,
                "oa": oa,
                "ob": ob,
                "date": r.get("date", None),
            }
        )

    if not candidates:
        return 0.0, {"mode": "no_similar_meetings"}

    # take top_n by weight
    candidates.sort(key=lambda x: x["w"], reverse=True)
    top = candidates[: max(1, int(top_n))]

    # convert GD into small probability shifts and average them
    # 1 goal of capped GD => 1% shift (tunable)
    shifts = [0.01 * float(c["gd"]) for c in top]
    weights = [float(c["w"]) for c in top]
    delta = float(np.average(shifts, weights=weights)) if sum(weights) > 0 else float(np.mean(shifts))

    # cap the final shift (max ±8%)
    delta = _clamp(delta, -max_shift, max_shift)

    dbg = {
        "mode": "h2h",
        "delta": delta,
        "used": top,
    }
    return delta, dbg


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
    team_a_now: list[str] | None = None,
    team_b_now: list[str] | None = None,
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
    team_a_now = team_a_now or []
    team_b_now = team_b_now or []

    mx_probs, mx_dbg = _prob_1x2_smart(M, team_a_now=team_a_now, team_b_now=team_b_now)
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
        "match_odds": {"prices": mx_prices, "debug": mx_dbg},
        "winning_margin": {"prices": wm_prices},
        "total_goals": {"main_line": float(main), "lines": totals},
    }
