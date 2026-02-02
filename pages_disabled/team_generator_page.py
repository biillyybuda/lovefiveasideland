
import streamlit as st
import pandas as pd
import numpy as np
import itertools
import hashlib
import json
from fractions import Fraction
import math
import textwrap
from utils.betting_markets import build_markets, blended_expected_goals



from utils.db_utils import load_players_df, get_conn as open_db
from utils.team_ai_engine import evaluate_teams, get_engine_state, clean_name
from utils.calc_utils import calibrate_winprob_scale, expected_score_calibrated
from utils.preview_insights import generate_preview_insights
from utils.ui_components import page_header

@st.cache_data(ttl=300)
def _players_table_cached():
    conn = open_db()
    try:
        return pd.read_sql("SELECT * FROM players", conn)
    finally:
        conn.close()

def _name_ui(name: str, players_df: pd.DataFrame) -> str:
    """Return DB display_name if present, else fall back to _display_name()."""
    try:
        row = players_df[players_df["name"] == name]
        if not row.empty and "display_name" in row.columns:
            dn = str(row.iloc[0].get("display_name") or "").strip()
            if dn:
                return dn
    except Exception:
        pass
    return _display_name(name)
    



def _round_numeric(df: pd.DataFrame, dp: int = 1) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(dp)
    return out

def _format_1dp(styler, df: pd.DataFrame):
    """Force 1dp display for float columns in a styled dataframe."""
    try:
        float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
        if float_cols:
            return styler.format({c: "{:.1f}" for c in float_cols})
    except Exception:
        pass
    return styler



# ----------------------------
# Win-prob calibration (cached)
# ----------------------------
@st.cache_data(ttl=3600)
def _prob_scale():
    return calibrate_winprob_scale(default_scale=200.0)

_SCALE = _prob_scale()

def _display_name(nm: str) -> str:
    """Pretty display version of a stored player name (keeps short tokens uppercased)."""
    s = str(nm or '').strip()
    if not s:
        return ''
    parts = [p for p in s.replace('_', ' ').split() if p]
    out = []
    for p in parts:
        if len(p) <= 2:
            out.append(p.upper())
        elif len(p) == 3 and p.isupper():
            out.append(p)
        else:
            out.append(p[0].upper() + p[1:])
    return ' '.join(out)


def _df_with_display_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with common player-name columns converted to display names.

    Also removes internal helper columns like _A/_B used for filtering.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out_df = df.copy()

    # Drop internal columns (created for matching/filtering)
    for internal in ["_A", "_B"]:
        if internal in out_df.columns:
            out_df = out_df.drop(columns=[internal])

    for col in list(out_df.columns):
        c = str(col).strip().lower()
        if c in ("player", "name", "player a", "player b", "player_a", "player_b", "a", "b"):
            try:
                out_df[col] = out_df[col].astype(str).apply(_display_name)
            except Exception:
                pass
    return out_df

def expected_goals(avg_mmr_a, avg_mmr_b):
    base_goals = 6.8  # league-wide average
    mmr_diff = avg_mmr_a - avg_mmr_b

    team_a_goals = base_goals/2 + (mmr_diff / 100)
    team_b_goals = base_goals/2 - (mmr_diff / 100)

    return max(1.0, team_a_goals), max(1.0, team_b_goals)

def prob_to_odds(p):
    if p <= 0:
        return "—"
    odds = 1 / p
    return f"{odds:.2f}".replace(".00", "")
def decimal_to_fraction(decimal):
    from fractions import Fraction
    f = Fraction(decimal - 1).limit_denominator(20)
    return f"{f.numerator}/{f.denominator}"



def _parse_team_list(val):
    if isinstance(val, list):
        return [str(p).strip() for p in val if str(p).strip()]
    if isinstance(val, str):
        return [p.strip() for p in val.split(",") if p.strip()]
    return []

def _parse_score(score_str):
    try:
        parts = str(score_str).replace("-", " ").split()
        nums = [int(x) for x in parts if x.isdigit()]
        if len(nums) == 2:
            return nums[0], nums[1]
    except Exception:
        pass
    return None, None

def _team_key_set(team: list[str]) -> set[str]:
    return {clean_name(p) for p in (team or []) if str(p).strip()}

def _overlap_count(team_now: list[str], team_hist: list[str]) -> int:
    return len(_team_key_set(team_now) & _team_key_set(team_hist))

def _diff_players(team_now: list[str], team_hist: list[str]) -> tuple[list[str], list[str]]:
    """
    Returns (subbed_in_now, missing_from_now) compared to a historical team.
    - subbed_in_now: players in current team but not in the historical team
    - missing_from_now: players in historical team but not in the current team
    """
    now_set = _team_key_set(team_now)
    hist_set = _team_key_set(team_hist)

    subbed_in = [p for p in team_now if clean_name(p) not in hist_set]
    missing = [p for p in team_hist if clean_name(p) not in now_set]
    return subbed_in, missing

def _best_similar_meetings(
    team_a_now: list[str],
    team_b_now: list[str],
    matches_eng: pd.DataFrame,
    top_n: int = 1,
    min_side_overlap: int = 3,
) -> list[dict]:
    """
    Find best historical matches similar to current (Team A vs Team B), allowing side swaps.

    Scoring:
      - overlapA + overlapB (same orientation)
      - overlapA_swapped + overlapB_swapped (swapped orientation)
    Tie-breakers:
      - prefer higher "min(overlapA, overlapB)" (i.e., both sides similar)
      - prefer most recent (by date) if available
    """
    if matches_eng is None or not isinstance(matches_eng, pd.DataFrame) or matches_eng.empty:
        return []

    df = matches_eng.copy()

    # Ensure date sortable
    if "date" in df.columns:
        df["__dt"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["__dt"] = pd.NaT

    rows = []
    for _, r in df.iterrows():
        ta_hist = _parse_team_list(r.get("team_a", ""))
        tb_hist = _parse_team_list(r.get("team_b", ""))

        if not ta_hist or not tb_hist:
            continue

        # Orientation 1: (A_now ~ A_hist) and (B_now ~ B_hist)
        oa1 = _overlap_count(team_a_now, ta_hist)
        ob1 = _overlap_count(team_b_now, tb_hist)
        score1 = oa1 + ob1

        # Orientation 2 (swapped): (A_now ~ B_hist) and (B_now ~ A_hist)
        oa2 = _overlap_count(team_a_now, tb_hist)
        ob2 = _overlap_count(team_b_now, ta_hist)
        score2 = oa2 + ob2

        if score2 > score1:
            orientation = "swapped"
            oa, ob, score = oa2, ob2, score2
            hist_A = tb_hist
            hist_B = ta_hist
        else:
            orientation = "same"
            oa, ob, score = oa1, ob1, score1
            hist_A = ta_hist
            hist_B = tb_hist

        # Require at least "min_any_side_overlap" on at least one side (or exact-ish overall)
        # STRICT: must be at least 3-overlap on BOTH sides (3v3 and up)
        if oa < min_side_overlap or ob < min_side_overlap:
            continue

        gA, gB = _parse_score(r.get("score", ""))
        res = str(r.get("result") or "").upper().strip()

        rows.append(
            {
                "score": int(score),
                "overlap_a": int(oa),
                "overlap_b": int(ob),
                "min_overlap": int(min(oa, ob)),
                "orientation": orientation,
                "hist_team_a": hist_A,
                "hist_team_b": hist_B,
                "raw_team_a": ta_hist,
                "raw_team_b": tb_hist,
                "match_id": r.get("id", None),
                "date": r.get("date", None),
                "__dt": r.get("__dt", pd.NaT),
                "scoreline": r.get("score", ""),
                "result": res,
                "venue": r.get("venue", ""),
            }
        )

    if not rows:
        return []

    out = pd.DataFrame(rows)

    # Sort: best similarity first, then "both sides similar", then most recent
    out = out.sort_values(
        by=["score", "min_overlap", "__dt"],
        ascending=[False, False, False],
        na_position="last",
    )

    # Take top_n, but ensure we include exact matches first if any
    return out.head(top_n).to_dict(orient="records")

def _render_previous_meetings_block(team_a: list[str], team_b: list[str], matches_eng: pd.DataFrame, players_df: pd.DataFrame):
    meetings = _best_similar_meetings(team_a, team_b, matches_eng, top_n=1, min_side_overlap=3)
    if not meetings:
        return  # silent if requirements not met

    m = meetings[0]
    histA = m["hist_team_a"]
    histB = m["hist_team_b"]

    sub_in_A, missing_A = _diff_players(team_a, histA)
    sub_in_B, missing_B = _diff_players(team_b, histB)

    def dn(x): 
        return _name_ui(x, players_df)

    # Pair OUT -> IN (same count typically)
    def pair_swaps(missing, subbed_in):
        k = max(len(missing), len(subbed_in))
        pairs = []
        for i in range(k):
            outp = dn(missing[i]) if i < len(missing) else "—"
            inp = dn(subbed_in[i]) if i < len(subbed_in) else "—"
            pairs.append((outp, inp))
        return pairs

    swaps_A = pair_swaps(missing_A, sub_in_A)
    swaps_B = pair_swaps(missing_B, sub_in_B)

    date_txt = str(m.get("date") or "").strip()
    score_txt = str(m.get("scoreline") or "").strip()
    meta_txt = " · ".join([t for t in [date_txt] if t])

    # Try to split score for big display
    gA, gB = _parse_score(score_txt)
    if gA is None or gB is None:
        big_score = score_txt if score_txt else "—"
    else:
        big_score = f"{gA}–{gB}"

    # Build pills for lineups, highlighting players NOT involved today (red)
    todayA = {clean_name(x) for x in team_a}
    todayB = {clean_name(x) for x in team_b}

    def lineup_pills(hist_team, today_set):
        pills = []
        for p in hist_team:
            cls = "out" if clean_name(p) not in today_set else "active"
            pills.append(f"<span class='pm-pill {cls}'>{dn(p)}</span>")
        return "".join(pills)

    # Sub rows as OUT (red) -> IN (green)
    def subs_rows(pairs):
        if not pairs or all(a == "—" and b == "—" for a, b in pairs):
            return "<div class='pm-meta'>No changes vs today</div>"

        rows = []
        for outp, inp in pairs:
            rows.append(
                "<div class='pm-subrow'>"
                f"<span class='pm-pill out'>{outp}</span>"
                "<span class='pm-arrow'>→</span>"
                f"<span class='pm-pill in'>{inp}</span>"
                "</div>"
            )
        return "<div class='pm-subs'>" + "".join(rows) + "</div>"

    html = f"""
<div class="pm-top pm-scorebar">
  <div class="pm-meta" style="text-align:center;width:100%;">{meta_txt}</div>

  <div class="pm-scoreline">
    <span class="pm-scoreteam pm-a">Team A</span>
    <span class="pm-score">{big_score}</span>
    <span class="pm-scoreteam pm-b">Team B</span>
  </div>
</div>

  <div class="pm-grid">
    <div class="pm-team a">
        <div class="pm-team-h">
            <span>Team A Lineup</span>
        </div>

      <div class="pm-line">{lineup_pills(histA, todayA)}</div>

      <div class="pm-subtitle">Changes vs today</div>
      {subs_rows(swaps_A)}
    </div>

    <div class="pm-team b">
        <div class="pm-team-h">
            <span>Team B Lineup</span>
        </div>

      <div class="pm-line">{lineup_pills(histB, todayB)}</div>

      <div class="pm-subtitle">Changes vs today</div>
      {subs_rows(swaps_B)}
    </div>
  </div>
</div>
"""
    st.html(textwrap.dedent(html))


def _ensure_session_defaults():
    if "tg_top_matchups" not in st.session_state:
        st.session_state.tg_top_matchups = None
    if "team_a" not in st.session_state:
        st.session_state.team_a = []
    if "team_b" not in st.session_state:
        st.session_state.team_b = []
    if "mdk_expanded" not in st.session_state:
        st.session_state.mdk_expanded = False
    if "selected_matchup" not in st.session_state:
        st.session_state.selected_matchup = None
    if "tg_last_config" not in st.session_state:
        st.session_state.tg_last_config = None
    if "tg_has_generated" not in st.session_state:
        st.session_state.tg_has_generated = False
    if "tg_selected_players" not in st.session_state:
        st.session_state.tg_selected_players = []


def _ensure_color_settings():
    """Defaults for team colours. A future Settings page can overwrite these in st.session_state."""
    st.session_state.setdefault("teamA_label", "Blue/White")
    st.session_state.setdefault("teamB_label", "Red/Black")

    # Foreground (text) colours
    st.session_state.setdefault("teamA_fg", "#3b82f6")
    st.session_state.setdefault("teamB_fg", "#ef4444")

    # Background tints used for gradients/cards (rgba strings)
    st.session_state.setdefault("teamA_bg", "rgba(59,130,246,0.18)")
    st.session_state.setdefault("teamB_bg", "rgba(239,68,68,0.18)")

def _spacer(h: int = 16):
    st.markdown(f"<div style='height:{int(h)}px'></div>", unsafe_allow_html=True)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def _draw_rate_from_matches(matches_eng: pd.DataFrame) -> float:
    """Historic draw rate based on score (results-only)."""
    if matches_eng is None or not isinstance(matches_eng, pd.DataFrame) or matches_eng.empty:
        return 0.10
    g = matches_eng.copy()
    gA, gB = [], []
    for _, r in g.iterrows():
        a, b = _parse_score(r.get("score", ""))
        if a is None or b is None:
            continue
        gA.append(a); gB.append(b)
    if not gA:
        return 0.10
    draws = sum(1 for a,b in zip(gA,gB) if a==b)
    return draws / len(gA)

def _compute_1x2(probA_expected: float, matches_eng: pd.DataFrame):
    """Convert expected-score probability (win + 0.5 draw) into 1X2 probabilities.

    We estimate draw probability from historic draws, then allocate remaining mass to A/B.
    """
    pE = _clamp(probA_expected, 0.01, 0.99)
    base_draw = _draw_rate_from_matches(matches_eng)
    closeness = 1.0 - abs(pE - 0.5) * 2.0  # 0..1 (1 = very close)
    draw_p = _clamp(base_draw * (0.6 + 0.8 * closeness), 0.04, 0.28)

    # expected score = P(A win) + 0.5*P(draw)
    pA_win = max(0.0, pE - 0.5 * draw_p)
    pB_win = max(0.0, 1.0 - draw_p - pA_win)

    # normalize (just in case of rounding/clamping)
    s = pA_win + draw_p + pB_win
    if s <= 0:
        return 0.45, 0.10, 0.45
    return pA_win / s, draw_p / s, pB_win / s

def _decimal_odds(p: float) -> float:
    p = _clamp(p, 1e-6, 1.0)
    return 1.0 / p

def _book_odds(pA: float, pD: float, pB: float, overround: float = 1.06):
    """Apply a small bookmaker margin (overround) and return decimal odds."""
    pA = _clamp(pA, 1e-6, 1.0)
    pD = _clamp(pD, 1e-6, 1.0)
    pB = _clamp(pB, 1e-6, 1.0)
    s = pA + pD + pB
    pA, pD, pB = pA/s, pD/s, pB/s
    # scale probs up to create margin
    pA_m = _clamp(pA * overround, 1e-6, 0.999999)
    pD_m = _clamp(pD * overround, 1e-6, 0.999999)
    pB_m = _clamp(pB * overround, 1e-6, 0.999999)
    # renormalize to 1 (keeps relative)
    s2 = pA_m + pD_m + pB_m
    pA_m, pD_m, pB_m = pA_m/s2, pD_m/s2, pB_m/s2
    return _decimal_odds(pA_m), _decimal_odds(pD_m), _decimal_odds(pB_m)


# ----------------------------
# Bookmaker-style odds display
# ----------------------------
# UK bookies tend to quote from a relatively small "price ladder" of common
# fractions (rather than any arbitrary fraction). We emulate that by snapping
# computed odds to the nearest standard price.

_BOOKIE_LADDER_FRAC: list[str] = [
    # odds-on (short)
    "1/100", "1/50", "1/33", "1/25", "1/20", "1/16", "1/14", "1/12", "1/11", "1/10",
    "1/9", "1/8", "1/7", "1/6", "1/5", "2/9", "1/4", "2/7", "1/3", "4/11", "2/5",
    "4/9", "1/2", "8/15", "4/7", "8/13", "4/6", "4/5", "5/6", "10/11",
    # even & odds-against
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
        num = float(a)
        den = float(b)
        return 1.0 + (num / den)
    # fallback (shouldn't happen)
    return float("nan")


_BOOKIE_LADDER_DEC: list[tuple[str, float]] = [
    (f, _frac_to_decimal(f)) for f in _BOOKIE_LADDER_FRAC
]


def _snap_decimal_to_bookie_ladder(decimal_odds: float) -> tuple[str, float]:
    """Snap a decimal price to the nearest bookie ladder price.

    We match in log-space so short and long prices round sensibly.
    Returns (fraction_label, snapped_decimal).
    """
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


def _to_fractional_odds(decimal_odds: float, max_den: int = 20) -> str:
    """Convert decimal odds to a *bookie-like* fractional price.

    - Prefer snapping to a standard UK ladder (readability)
    - Fall back to a best-effort reduced fraction if needed
    """
    label, dec = _snap_decimal_to_bookie_ladder(decimal_odds)
    if label != "—":
        return "Evens" if label.lower() in ("evens", "even", "evs", "ev") else label

    # fallback: generic fraction
    try:
        d = float(decimal_odds)
        if not math.isfinite(d) or d <= 1.0:
            return "—"
        frac = Fraction(d - 1.0).limit_denominator(max_den)
        if frac.numerator == frac.denominator:
            return "Evens"
        return f"{frac.numerator}/{frac.denominator}"
    except Exception:
        return "—"


@st.cache_data(ttl=3600)
def _league_winprob_params():
    """Calibrate expected-score mapping using your league's historical match data.

    We try to build training examples from:
    - matches: team_a, team_b, result, id
    - mmr_history: player_id, match_id, mmr_before (pre-match rating)
    If those tables aren't available, we fall back to a reasonable default.
    Returns (bias, scale) for: pE = sigmoid(bias + diff/scale)
    where diff = avg_mmr_A - avg_mmr_B and pE ≈ P(A win) + 0.5*P(draw).
    """
    # Sensible defaults (keeps behaviour close to current)
    best_bias, best_scale = 0.0, float(_SCALE if _SCALE else 200.0)

    try:
        conn = open_db()
        try:
            matches = pd.read_sql("SELECT id, team_a, team_b, result FROM matches", conn)
        except Exception:
            conn.close()
            return best_bias, best_scale

        if matches is None or matches.empty:
            conn.close()
            return best_bias, best_scale

        # Try to grab mmr_history joined to players so we can map names -> mmr_before
        try:
            mh = pd.read_sql(
                """
                SELECT mh.match_id, p.name AS name, mh.mmr_before
                FROM mmr_history mh
                JOIN players p ON p.id = mh.player_id
                """,
                conn,
            )
        except Exception:
            conn.close()
            return best_bias, best_scale
        conn.close()

        if mh is None or mh.empty:
            return best_bias, best_scale

        mh["name"] = mh["name"].astype(str)
        mh["key"] = mh["name"].apply(clean_name)

        # Build one row per match: avg pre-match mmr A/B and outcome y in [0,1]
        rows = []
        for _, r in matches.iterrows():
            mid = r.get("id")
            ta = [clean_name(p) for p in _parse_team_list(r.get("team_a", ""))]
            tb = [clean_name(p) for p in _parse_team_list(r.get("team_b", ""))]
            res = str(r.get("result") or "").upper().strip()
            if not ta or not tb or res not in ("A", "B", "D", "DRAW", "X"):
                continue

            mhm = mh[mh["match_id"] == mid]
            if mhm.empty:
                continue

            a_mmrs = mhm[mhm["key"].isin(ta)]["mmr_before"].dropna().astype(float).tolist()
            b_mmrs = mhm[mhm["key"].isin(tb)]["mmr_before"].dropna().astype(float).tolist()
            if len(a_mmrs) < 3 or len(b_mmrs) < 3:
                continue

            diff = float(np.mean(a_mmrs) - np.mean(b_mmrs))
            if res == "A":
                y = 1.0
            elif res == "B":
                y = 0.0
            else:
                y = 0.5
            rows.append((diff, y))

        if len(rows) < 25:
            return best_bias, best_scale

        diffs = np.array([d for d, _ in rows], dtype=float)
        ys = np.array([y for _, y in rows], dtype=float)

        scales = np.arange(80.0, 520.0, 10.0)
        biases = np.arange(-0.8, 0.81, 0.05)

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        best_loss = float("inf")
        for sc in scales:
            z_base = diffs / sc
            for b in biases:
                p = sigmoid(b + z_base)
                p = np.clip(p, 1e-4, 1 - 1e-4)
                loss = float(np.mean((p - ys) ** 2))
                if loss < best_loss:
                    best_loss = loss
                    best_bias, best_scale = float(b), float(sc)

        return best_bias, best_scale

    except Exception:
        return best_bias, best_scale


_BIAS, _LEAGUE_SCALE = _league_winprob_params()


def _expected_score_league_calibrated(a_mmr: float, b_mmr: float) -> float:
    """League-calibrated expected score (win + 0.5 draw)."""
    try:
        diff = float(a_mmr) - float(b_mmr)
        z = _BIAS + (diff / float(_LEAGUE_SCALE if _LEAGUE_SCALE else 200.0))
        p = 1.0 / (1.0 + math.exp(-z))
        return _clamp(p, 0.01, 0.99)
    except Exception:
        return _clamp(expected_score_calibrated(a_mmr, b_mmr, scale=_SCALE), 0.01, 0.99)


def _style_team_columns(df: pd.DataFrame, team_a: list[str], team_b: list[str], teamA_fg: str, teamB_fg: str):
    """Return a Styler that colours Player A/B cells based on which team they belong to."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    team_a_set = {clean_name(x) for x in team_a}
    team_b_set = {clean_name(x) for x in team_b}

    def _color(val: str):
        k = clean_name(val)
        if k in team_a_set:
            return f"color:{teamA_fg}; font-weight:700;"
        if k in team_b_set:
            return f"color:{teamB_fg}; font-weight:700;"
        return ""

    styled = df.style
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ("player a", "player b", "player_a", "player_b", "player", "name", "a", "b"):
            styled = styled.applymap(_color, subset=[col]) # type: ignore
    return styled




def _has_selected_matchup() -> bool:
    a = _parse_team_list(st.session_state.get("team_a", []))
    b = _parse_team_list(st.session_state.get("team_b", []))
    return len(a) == 5 and len(b) == 5 and (not (set(a) & set(b)))

def _recent_form_boxes_html(player_name: str, matches_eng: pd.DataFrame) -> str:
    """
    Results-only: last 5 outcomes based on score.
    """
    if matches_eng is None or matches_eng.empty:
        return ""
    key = clean_name(player_name)

    df = matches_eng.copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce") # type: ignore
    df = df.sort_values("date")

    results = []
    for _, row in df.iterrows():
        ta = [clean_name(p) for p in _parse_team_list(row.get("team_a", ""))]
        tb = [clean_name(p) for p in _parse_team_list(row.get("team_b", ""))]
        if key not in ta and key not in tb:
            continue

        gA, gB = _parse_score(row.get("score", ""))
        if gA is None or gB is None:
            continue

        if key in ta:
            outcome = "W" if gA > gB else ("L" if gA < gB else "D")
        else:
            outcome = "W" if gB > gA else ("L" if gB < gA else "D")

        results.append(outcome)

    last5 = results[-5:]
    if not last5:
        return ""

    boxes = []
    for r in last5:
        color = {"W": "#22c55e", "D": "#6b7280", "L": "#ef4444"}[r]
        boxes.append(
            f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;background:{color};margin-right:4px;'></span>"
        )
    return f"<div style='display:flex;align-items:center;margin-top:2px;'>{''.join(boxes)}</div>"

def _pstats_from_players_df(players_df: pd.DataFrame, name: str) -> dict:
    row = players_df[players_df["name"] == name]
    if row.empty:
        return {"matches_played": 0, "wins": 0, "win_pct": 0.0, "win_streak": 0, "lose_streak": 0, "mmr": None}
    r = row.iloc[0]
    mp = int(r.get("matches_played", 0) or 0)
    w = int(r.get("wins", 0) or 0)
    wp = round((w / mp * 100), 1) if mp > 0 else 0.0
    mmr = r.get("mmr", None) if "mmr" in row.columns else None
    return {
        "matches_played": mp,
        "wins": w,
        "win_pct": wp,
        "win_streak": int(r.get("win_streak", 0) or 0),
        "lose_streak": int(r.get("lose_streak", 0) or 0),
        "mmr": mmr,
    }

def _best_teammate_by_chemistry(name: str, own_team: list[str], base_chemistry: dict) -> str | None:
    """
    Uses engine base_chemistry (results-only) if present.
    Expects keys like (a_clean, b_clean) or "a_clean|b_clean".
    We'll handle both.
    """
    me = clean_name(name)
    best = None
    best_val = -1e9

    def chem_lookup(a, b):
        # tuple key
        tkey = (a, b)
        if tkey in base_chemistry:
            return float(base_chemistry.get(tkey) or 0.0)
        tkey = (b, a)
        if tkey in base_chemistry:
            return float(base_chemistry.get(tkey) or 0.0)
        # string key
        skey1 = f"{a}|{b}"
        skey2 = f"{b}|{a}"
        if skey1 in base_chemistry:
            return float(base_chemistry.get(skey1) or 0.0)
        if skey2 in base_chemistry:
            return float(base_chemistry.get(skey2) or 0.0)
        return 0.0

    for tm in own_team:
        if tm == name:
            continue
        val = chem_lookup(me, clean_name(tm))
        if val > best_val:
            best_val = val
            best = tm
    return best

def _weekly_rivals_by_similarity(team_a, team_b, finisher_score, creator_score, impact_index):
    # results-only "style similarity" rivals
    pairs = []
    for a in team_a:
        ak = clean_name(a)
        for b in team_b:
            bk = clean_name(b)
            fin_a = float(finisher_score.get(ak, 0.0) or 0.0)
            cre_a = float(creator_score.get(ak, 0.0) or 0.0)
            imp_a = float(impact_index.get(ak, 0.0) or 0.0)

            fin_b = float(finisher_score.get(bk, 0.0) or 0.0)
            cre_b = float(creator_score.get(bk, 0.0) or 0.0)
            imp_b = float(impact_index.get(bk, 0.0) or 0.0)

            dist = ((fin_a - fin_b)**2 + (cre_a - cre_b)**2 + (imp_a - imp_b)**2)**0.5
            pairs.append((dist, a, b))
    pairs.sort(key=lambda x: x[0])

    used_a, used_b = set(), set()
    mapping = {}
    for _, a, b in pairs:
        if a in used_a or b in used_b:
            continue
        mapping[a] = b
        mapping[b] = a
        used_a.add(a)
        used_b.add(b)
    return mapping

def _stars_from_rank(rank_pos: int | None) -> int:
    if rank_pos is None:
        return 3
    if rank_pos <= 5:
        return 5
    if rank_pos <= 10:
        return 4
    if rank_pos <= 15:
        return 3
    if rank_pos <= 20:
        return 2
    return 1

def _rank_dict(score_dict: dict) -> dict:
    items = [(k, v) for k, v in score_dict.items() if isinstance(v, (int, float, np.floating))]
    items.sort(key=lambda x: x[1], reverse=True)
    return {k: i + 1 for i, (k, _) in enumerate(items)}

def _star_bar(stars: int) -> str:
    stars = max(1, min(5, int(stars)))
    return "★" * stars + "☆" * (5 - stars)

def _render_match_preview(team_a: list[str], team_b: list[str], teamA_label: str | None = None, teamB_label: str | None = None, teamA_fg: str | None = None, teamB_fg: str | None = None, teamA_bg: str | None = None, teamB_bg: str | None = None):
    # --- Styles (simple + stable) ---
    st.markdown(
        """
        <style>
        .mdk-section {background:#141414;border-radius:16px;padding:18px 20px;margin-top:12px;box-shadow:0 0 12px rgba(255,255,255,0.05);}
        .mdk-teambox {border-radius:14px;padding:12px 14px;margin-top:10px;border:1px solid rgba(255,255,255,0.08);}
        .nameA {color:var(--a1); font-weight:800;}
        .nameB {color:var(--b1); font-weight:800;}
        .oddsbox {display:flex;gap:10px;margin-top:10px;}
        .odd {flex:1;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:10px 12px;text-align:center;}
        .odd .lbl {font-size:0.85rem;color:#bdbdbd;margin-bottom:2px;}
        .odd .val {font-size:1.2rem;font-weight:900;}
        .odd .pct {font-size:0.85rem;color:#cfcfcf;opacity:0.9;}
        .muted {color:#bdbdbd;font-size:0.95rem;}
        .pm-card{
        border-radius:16px;
        padding:16px 18px;
        border:1px solid rgba(255,255,255,0.10);
        background:linear-gradient(180deg, rgba(255,255,255,0.05), rgba(0,0,0,0.0));
        box-shadow:0 0 16px rgba(0,0,0,0.55);
        margin-top:10px;
        }
        .pm-top{
        display:flex;
        align-items:center;
        margin-bottom:6px;
        }
        .pm-scorebar{
        flex-direction:column;
        justify-content:center;
        gap:6px;
        margin-bottom:12px;
        }

        .pm-scoreline{
        display:flex;
        justify-content:center;
        align-items:baseline;
        gap:18px;
        }

        .pm-scoreteam{
        font-size:3rem;          /* match the score size */
        font-weight:1000;
        letter-spacing:0.5px;
        text-shadow:0 3px 20px rgba(0,0,0,0.65);
        opacity:0.95;
        white-space:nowrap;
        }

        .pm-scoreteam.pm-a{ color:#93c5fd; }
        .pm-scoreteam.pm-b{ color:#fca5a5; }

        .pm-title{
        font-weight:900;
        font-size:1.25rem;
        display:flex;
        align-items:center;
        gap:10px;
        }
        .pm-meta{
        color:#cfcfcf;
        opacity:0.9;
        font-size:0.95rem;
        }
        .pm-score{
        font-size:3rem;
        font-weight:1000;
        letter-spacing:1px;
        color:#ffffff;
        text-shadow:0 3px 20px rgba(0,0,0,0.65);
        }
        .pm-grid{
        display:grid;
        grid-template-columns: 1fr 1fr;
        gap:14px;
        }
        .pm-team{
        border-radius:14px;
        padding:12px 14px;
        border:1px solid rgba(255,255,255,0.10);
        background:rgba(255,255,255,0.03);
        }
        .pm-team.a{
        background:linear-gradient(135deg, rgba(59,130,246,0.14), rgba(255,255,255,0.02));
        border:1px solid rgba(59,130,246,0.22);
        }
        .pm-team.b{
        background:linear-gradient(225deg, rgba(239,68,68,0.14), rgba(255,255,255,0.02));
        border:1px solid rgba(239,68,68,0.22);
        }
        .pm-team-h{
        display:flex;
        justify-content:space-between;
        align-items:center;
        }
        .pm-badge{
        font-size:0.85rem;
        font-weight:900;
        padding:4px 10px;
        border-radius:999px;
        border:1px solid rgba(255,255,255,0.12);
        background:rgba(255,255,255,0.04);
        color:#eaeaea;
        }

        .pm-team-h{
        font-weight:900;
        font-size:1.05rem;
        margin-bottom:8px;
        }
        .pm-line{
        display:flex;
        flex-wrap:wrap;
        gap:8px;
        }
        .pm-pill{
        padding:6px 10px;
        border-radius:999px;
        font-weight:800;
        font-size:0.98rem;
        border:1px solid rgba(255,255,255,0.10);
        background:rgba(255,255,255,0.04);
        }
        .pm-pill.out{
        background:rgba(239,68,68,0.22);
        border:1px solid rgba(239,68,68,0.55);
        color:#ffe4e6;
        box-shadow:0 0 10px rgba(239,68,68,0.15);
        }
        .pm-pill.active{
        background:rgba(255,255,255,0.06);
        border:1px solid rgba(255,255,255,0.14);
        }
        .pm-pill.in{
        background:rgba(34,197,94,0.14);
        border:1px solid rgba(34,197,94,0.35);
        color:#bbf7d0;
        }
        .pm-subtitle{
        margin-top:10px;
        color:#bdbdbd;
        font-size:0.95rem;
        font-weight:800;
        opacity:0.95;
        }
        .pm-subs{
        margin-top:6px;
        display:flex;
        flex-direction:column;
        gap:8px;
        }
        .pm-subrow{
        display:flex;
        align-items:center;
        gap:10px;
        font-weight:900;
        font-size:1.05rem;
        }
        .pm-arrow{
        opacity:0.9;
        font-size:1.1rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    players_df = _players_table_cached()

    # Engine state (results-only friendly)
    eng = get_engine_state()
    matches_eng = eng.get("matches")
    impact_index = eng.get("impact_index", {}) or {}
    finisher_score = eng.get("finisher_score", {}) or {}
    creator_score = eng.get("creator_score", {}) or {}
    base_chemistry = eng.get("base_chemistry", {}) or {}


    # Team colours (defaults come from session_state; future Settings page can override them)
    teamA_label = teamA_label or st.session_state.get("teamA_label", "Blue/White")
    teamB_label = teamB_label or st.session_state.get("teamB_label", "Red/Black")
    teamA_fg = teamA_fg or st.session_state.get("teamA_fg", "#3b82f6")
    teamB_fg = teamB_fg or st.session_state.get("teamB_fg", "#ef4444")
    teamA_bg = teamA_bg or st.session_state.get("teamA_bg", "rgba(59,130,246,0.18)")
    teamB_bg = teamB_bg or st.session_state.get("teamB_bg", "rgba(239,68,68,0.18)")

    # --- Ratings / win chance / betting-style odds ---
    def _mean_mmr(team):
        mmrs = []
        for nm in team:
            try:
                row = players_df[players_df["name"] == nm]
                if not row.empty and "mmr" in row.columns:
                    v = row.iloc[0].get("mmr", None)
                    if v is not None and str(v) != "nan":
                        mmrs.append(float(v))
            except Exception:
                pass
        return float(np.mean(mmrs)) if mmrs else 1000.0

    a_mmr = _mean_mmr(team_a)
    b_mmr = _mean_mmr(team_b)

    mmr_lam_a, mmr_lam_b = expected_goals(a_mmr, b_mmr)

    # Blend in historical scoring (uses DB match scores)
    lam_a, lam_b, lam_dbg = blended_expected_goals(team_a, team_b, mmr_lam_a, mmr_lam_b)

    markets = build_markets(
        lam_a,
        lam_b,
        team_a_now=team_a,
        team_b_now=team_b,
        overround=1.06,
        max_goals=15,
        total_lines=None,
        include_alt_lines=True,
    )


    # --- Top odds should match Betting Markets engine (markets["match_odds"]) ---
    mx = markets.get("match_odds", {}).get("prices", {})

    # Fallback (if match_odds missing for any reason): keep the old logic
    if not mx:
        pE = _expected_score_league_calibrated(a_mmr, b_mmr)
        pA_win, pDraw, pB_win = _compute_1x2(pE, matches_eng)  # type: ignore
        oddA, oddX, oddB = _book_odds(pA_win, pDraw, pB_win, overround=1.06)
        labA, oddA_s = _snap_decimal_to_bookie_ladder(oddA)
        labX, oddX_s = _snap_decimal_to_bookie_ladder(oddX)
        labB, oddB_s = _snap_decimal_to_bookie_ladder(oddB)
        try:
            impA = 1.0 / float(oddA_s)
            impX = 1.0 / float(oddX_s)
            impB = 1.0 / float(oddB_s)
            impsum = impA + impX + impB
            pA_disp = impA / impsum
            pX_disp = impX / impsum
            pB_disp = impB / impsum
        except Exception:
            pA_disp, pX_disp, pB_disp = pA_win, pDraw, pB_win
    else:
        # Use the SAME labels + probs as the betting markets engine
        labA = mx["1"]["label"]
        labX = mx["X"]["label"]
        labB = mx["2"]["label"]

        pA_disp = float(mx["1"]["p"])
        pX_disp = float(mx["X"]["p"])
        pB_disp = float(mx["2"]["p"])

    header_html = f"""
    <div style="border-radius:16px;padding:14px 16px;margin-top:8px;
                background:linear-gradient(90deg, {teamA_bg} 0%, {teamB_bg} 100%);
                border:1px solid rgba(255,255,255,0.10);text-align:center;">
        <div style="font-size:1.25rem;font-weight:900;">
            <span style="color:{teamA_fg};">🔵 Team A — {teamA_label}</span>
            &nbsp;&nbsp;vs&nbsp;&nbsp;
            <span style="color:{teamB_fg};">🔴 Team B — {teamB_label}</span>
        </div>
        <div class="oddsbox">
            <div class="odd">
                <div class="lbl">1 (Team A)</div>
                <div class="val">{labA}</div>
                <div class="pct">{pA_disp*100:.1f}%</div>
            </div>
            <div class="odd">
                <div class="lbl">X (Draw)</div>
                <div class="val">{labX}</div>
                <div class="pct">{pX_disp*100:.1f}%</div>
            </div>
            <div class="odd">
                <div class="lbl">2 (Team B)</div>
                <div class="val">{labB}</div>
                <div class="pct">{pB_disp*100:.1f}%</div>
            </div>
        </div>
        <div style="margin-top:8px;color:#e7e7e7;opacity:0.9;font-size:0.92rem;">
            Avg MMR: <b>{a_mmr:.0f}</b> vs <b>{b_mmr:.0f}</b>
        </div>
    </div>
    """



    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### 🔵 Team A — {teamA_label}  ·  Avg MMR {a_mmr:.0f}")
        st.markdown(
            f"""
            <div class="mdk-teambox"
                 style="background:linear-gradient(135deg, {teamA_bg} 0%, rgba(0,0,0,0) 70%);">
                <div style="font-weight:800;color:{teamA_fg};margin-bottom:6px;">
                    Lineup
                </div>
                <div style="line-height:1.7;">
                    {'<br>'.join([f"<span class='nameA' style='--a1:{teamA_fg};--b1:{teamB_fg};'>{_name_ui(p, players_df)}</span>" for p in team_a])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(f"#### 🔴 Team B — {teamB_label}  ·  Avg MMR {b_mmr:.0f}")
        st.markdown(
            f"""
            <div class="mdk-teambox"
                 style="text-align:right;background:linear-gradient(225deg, {teamB_bg} 0%, rgba(0,0,0,0) 70%);">
                <div style="font-weight:800;color:{teamB_fg};margin-bottom:6px;">
                    Lineup
                </div>
                <div style="line-height:1.7;">
                    {'<br>'.join([f"<span class='nameB' style='--a1:{teamA_fg};--b1:{teamB_fg};'>{_name_ui(p, players_df)}</span>" for p in team_b])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _spacer(14)

    st.markdown(header_html, unsafe_allow_html=True)

    _spacer(14)


    # Insights tables (already results-only)
    conn = open_db()
    insights = generate_preview_insights(team_a, team_b, conn)
    conn.close()

    # -----------------------------
    # 📊 Betting Markets (fun only)
    # -----------------------------
    with st.expander("📊 Betting Markets", expanded=False):

        # Helper to render odds tiles in the same style as your 1X2 row
        def _render_tiles(items):
            cols = st.columns(len(items))
            for i, it in enumerate(items):
                with cols[i]:
                    pct_html = (
                        f"<div class='pct'>{it['pct']}</div>"
                        if "pct" in it and it["pct"] not in (None, "")
                        else ""
                    )

                    st.markdown(
                        f"""
                        <div class="odd">
                            <div class="lbl">{it['label']}</div>
                            <div class="val">{it['price']}</div>
                            {pct_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Winning Margin
        with st.expander("Winning Margin", expanded=False):
            wm = markets["winning_margin"]["prices"]
            order = ["A 3+", "A 1–2", "Draw", "B 1–2", "B 3+"]
            items = []
            for k in order:
                d = wm[k]
                items.append(
                    {
                        "label": k,
                        "price": d["label"],
                        "pct": f"{float(d['p']) * 100:.1f}%",
                    }
                )
            _render_tiles(items)

        # Total Goals O/U
        with st.expander("Total Goals", expanded=False):
            main_ln = float(markets["total_goals"].get("main_line", 0.0) or 0.0)

            for ln, prices in markets["total_goals"]["lines"].items():
                # A clean row title like bet365
                tag = " (Main)" if abs(float(ln) - main_ln) < 1e-9 else ""
                st.markdown(f"<div class='bm-rowtitle'>Total {ln}{tag}</div>", unsafe_allow_html=True)

                keys = [f"Over {ln}", f"Under {ln}"]
                items = []
                for k in keys:
                    d = prices[k]
                    items.append({"label": k, "price": d["label"]})

                _render_tiles(items)
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


    with st.expander("🔮 Match Insights", expanded=False):

        st.subheader("🔥 Key Matchups")
        km = insights.get("key_matchups")
        if km is not None:
            km_d = _df_with_display_names(km)
            km_s = _style_team_columns(km_d, team_a, team_b, teamA_fg, teamB_fg) # type: ignore
            km_s = _format_1dp(km_s, km_d)
            st.dataframe(km_s, use_container_width=True)
        else:
            st.info("Key matchups not available.")

        st.subheader("🤝 Best Teammates")
        bt = insights.get("best_teammates", [])
        if isinstance(bt, (list, tuple)) and len(bt) == 2:
            st.markdown(f"**Team A:** {', '.join([_display_name(p) for p in team_a])}")
            btA_d = _df_with_display_names(bt[0])
            btA_s = _style_team_columns(btA_d, team_a, [], teamA_fg, teamA_fg) # type: ignore
            btA_s = _format_1dp(btA_s, btA_d)
            st.dataframe(btA_s, use_container_width=True)
            st.markdown(f"**Team B:** {', '.join([_display_name(p) for p in team_b])}")
            btB_d = _df_with_display_names(bt[1])
            btB_s = _style_team_columns(btB_d, team_b, [], teamB_fg, teamB_fg) # type: ignore
            btB_s = _format_1dp(btB_s, btB_d)
            st.dataframe(btB_s, use_container_width=True)
        else:
            st.info("Best teammate tables not available.")

        st.subheader("📈 Form & Streaks")
        fs = insights.get("form_streaks")

        if fs is not None:
            fs_d = _round_numeric(_df_with_display_names(fs), 1)
            st.dataframe(
                _style_team_columns(fs_d, team_a, team_b, teamA_fg, teamB_fg), # type: ignore
                use_container_width=True
            )
        else:
            st.info("Form & streaks not available.")
# Player cards (results-only: MMR/win%/streak/form boxes + best teammate/rival)
        st.markdown("<h3 style='text-align:center; margin-bottom: 6px;'>Past Matchups</h3>", unsafe_allow_html=True)
        _render_previous_meetings_block(team_a, team_b, matches_eng, players_df) # type: ignore




    with st.expander("🌟 Matchday Player Cards", expanded=False):
        rivals = _weekly_rivals_by_similarity(team_a, team_b, finisher_score, creator_score, impact_index)

        def card(name: str, own_team: list[str], opp_team: list[str], fg: str, bg: str) -> str:
            key = clean_name(name)
            ps = _pstats_from_players_df(players_df, name)

            best_tm = _best_teammate_by_chemistry(name, own_team, base_chemistry)
            this_rival = rivals.get(name)

            form_boxes = ""
            if matches_eng is not None and isinstance(matches_eng, pd.DataFrame) and (not matches_eng.empty):
                form_boxes = _recent_form_boxes_html(name, matches_eng)

            mmr_txt = ""
            if ps.get("mmr") is not None and str(ps.get("mmr")) != "nan":
                try:
                    mmr_txt = f"{float(ps['mmr']):.0f}"
                except Exception:
                    mmr_txt = str(ps.get("mmr"))

            streak_txt = ""
            if ps["win_streak"] >= 2:
                streak_txt = f"🟢 W{ps['win_streak']}"
            elif ps["lose_streak"] >= 2:
                streak_txt = f"🔴 L{ps['lose_streak']}"

            return f"""
    <div style="background:linear-gradient(135deg, {bg} 0%, rgba(0,0,0,0) 70%);border:1px solid rgba(255,255,255,0.10);padding:14px 16px;border-radius:16px;margin-bottom:12px;box-shadow:0 0 12px rgba(0,0,0,0.65);">
      <div style="font-size:1.05rem;font-weight:900;margin-bottom:6px;color:{fg};">{_name_ui(name, players_df)}</div>
      <div style="color:#d9d9d9;font-size:0.92rem;line-height:1.55;">
        <div><span style="color:#9ae6ff;">MMR:</span> <b>{mmr_txt if mmr_txt else "—"}</b> <span style="opacity:0.85;">{streak_txt}</span></div>
        <div><span style="color:#c6afff;">Career:</span> {ps["matches_played"]} matches · {ps["win_pct"]:.1f}% wins</div>
        <div style="margin-top:4px;"><span style="color:#aaaaaa;">Recent form:</span> {form_boxes}</div>
    <div style="height:1px;background:rgba(255,255,255,0.10);margin:8px 0;"></div>


        <div><span style="color:#22c55e;">Best teammate:</span> {_name_ui(best_tm, players_df) if best_tm else "—"}</div>
        <div><span style="color:#f97316;">This week's rival:</span> {_name_ui(this_rival, players_df) if this_rival else "—"}</div>
      </div>
    </div>
    """

        cA, cB = st.columns(2)
        with cA:
            for p in team_a:
                st.html(card(p, own_team=team_a, opp_team=team_b, fg=teamA_fg, bg=teamA_bg)) # type: ignore
        with cB:
            for p in team_b:
                st.html(card(p, own_team=team_b, opp_team=team_a, fg=teamB_fg, bg=teamB_bg)) # type: ignore
def render_team_generator_page(show_header: bool = True):
    _ensure_session_defaults()
    _ensure_color_settings()

    if show_header:
        page_header("Matchday Hub", "Generate balanced teams and preview the matchup", center=True, divider=True)

    
    # If a matchup is selected, switch the page into "Matchday Hub" mode
    if st.session_state.get("selected_matchup"):
        sel_payload = st.session_state.selected_matchup
        team_a = sel_payload.get("team_a", [])
        team_b = sel_payload.get("team_b", [])

        c_back, _ = st.columns([1,1])
        with c_back:
            if st.button("⬅️ Change matchup"):
                st.session_state.selected_matchup = None
                st.session_state.mdk_expanded = False
                st.rerun()

        _spacer(10)

        # Show the Matchday Card content (no duplicate title / no dropdown)
        st.session_state.team_a = team_a
        st.session_state.team_b = team_b
        st.session_state.mdk_expanded = True

        # Render preview directly (no Matchday Card expander)
        if not _has_selected_matchup():
            st.info("No matchup selected.")
        else:
            _render_match_preview(
                team_a,
                team_b,
                teamA_label=st.session_state.get('teamA_label'),
                teamB_label=st.session_state.get('teamB_label'),
                teamA_fg=st.session_state.get('teamA_fg'),
                teamB_fg=st.session_state.get('teamB_fg'),
                teamA_bg=st.session_state.get('teamA_bg'),
                teamB_bg=st.session_state.get('teamB_bg'),
            )

        return

    # Load players (ACTIVE ONLY)
    players_df = load_players_df()

    # Defensive: ensure archive column exists
    if "is_active" not in players_df.columns:
        players_df["is_active"] = 1

    # 🔒 Filter out archived players
    players_df = players_df[players_df["is_active"].fillna(1).astype(int) == 1]

    if "strengths" not in players_df.columns:
        players_df["strengths"] = ""
    if "fitness" not in players_df.columns:
        players_df["fitness"] = "Medium"

    names = players_df["name"].tolist()
    if len(names) < 10:
        st.warning("Need at least 10 players.")
        return

    # Captain Mode
    captain_mode = st.toggle("Enable Captain Mode (exclude captains from balance)")

    captainA, captainB = None, None
    if captain_mode:
        c1, c2 = st.columns(2)
        with c1:
            captainA = st.selectbox("Captain A", [""] + names, index=0, format_func=lambda n: _name_ui(n, players_df))
        with c2:
            remaining = [n for n in names if n != captainA]
            captainB = st.selectbox("Captain B", [""] + remaining, index=0, format_func=lambda n: _name_ui(n, players_df))

    # Player selection (blank by default feels intentional)
    sel = st.multiselect(
        "Select players for balance",
        names,
        default=st.session_state.tg_selected_players,
        format_func=_display_name,
    )

    if captain_mode and captainA and captainB:
        sel = [p for p in sel if p not in [captainA, captainB]]
        st.info(f"Balancing {len(sel)} players (excluding captains {captainA} & {captainB}).")

    if (captain_mode and len(sel) != 8) or (not captain_mode and len(sel) != 10):
        st.warning("Please select correct number of players.")
        return

    # Locked players
    c1, c2 = st.columns(2)
    with c1:
        locks_A = st.multiselect("🔒 Lock → Team A", options=sel, format_func=_display_name)
    with c2:
        locks_B = st.multiselect("🔒 Lock → Team B", options=[p for p in sel if p not in locks_A], format_func=_display_name)

    if set(locks_A) & set(locks_B):
        st.error("A player cannot be locked to both teams.")
        return

    # If selection inputs changed since last generation, clear previously generated options
    current_config = {
        "captain_mode": bool(captain_mode),
        "captainA": captainA or "",
        "captainB": captainB or "",
        "selected_players": list(sel),
        "locks_A": list(locks_A),
        "locks_B": list(locks_B),
    }
    current_hash = hashlib.md5(json.dumps(current_config, sort_keys=True).encode("utf-8")).hexdigest()
    if st.session_state.get("tg_last_config") != current_hash:
        st.session_state.tg_last_config = current_hash
        st.session_state.tg_top_matchups = None
        st.session_state.tg_has_generated = False

    # ----- Generate & persist results so clicking "Use this matchup" doesn't wipe them -----
    if st.button("Generate Balanced Teams"):
        st.session_state.tg_has_generated = True
        team_size = 4 if captain_mode else 5
        remain = [p for p in sel if p not in locks_A + locks_B]

        scored = []
        for comb in itertools.combinations(remain, team_size - len(locks_A)):
            A = list(locks_A) + list(comb)
            B = list(locks_B) + [p for p in remain if p not in comb]
            if len(A) == team_size and len(B) == team_size:
                # add captains back in for evaluation/display
                eval_A = [captainA] + A if (captain_mode and captainA) else A[:]
                eval_B = [captainB] + B if (captain_mode and captainB) else B[:]

                # engine keys are lowercase
                eval_A_norm = [p.lower() for p in eval_A]
                eval_B_norm = [p.lower() for p in eval_B]

                fairness_score, breakdown = evaluate_teams(eval_A_norm, eval_B_norm)
                scored.append((float(fairness_score), A, B, breakdown))

        # Remove mirrored duplicates
        unique = {}
        for score_val, A, B, breakdown in scored:
            key = frozenset([frozenset(A), frozenset(B)])
            if key not in unique or score_val < unique[key][0]:
                unique[key] = (score_val, A, B, breakdown)

        top = sorted(list(unique.values()), key=lambda x: x[0])[:3]

        st.session_state.tg_top_matchups = {
            "captain_mode": bool(captain_mode),
            "captainA": captainA,
            "captainB": captainB,
            "items": [
                {"score": score, "A": A, "B": B, "breakdown": breakdown}
                for (score, A, B, breakdown) in top
            ],
        }

    # ----- Render persisted results (if any) -----
    payload = st.session_state.get("tg_top_matchups")
    if (not st.session_state.get("tg_has_generated")) or (not payload) or (not payload.get("items")):
        return

    captain_mode = payload.get("captain_mode", False)
    captainA = payload.get("captainA")
    captainB = payload.get("captainB")


    def palette(label):
        if label == "Blue/White":
            return {"fg": "#3b82f6", "bg": "rgba(59,130,246,0.18)"}
        return {"fg": "#ef4444", "bg": "rgba(239,68,68,0.18)"}

    def choose_colors(i: int):
        if i % 2 == 1:
            return ("🔵⚪", "Blue/White"), ("🔴⚫", "Red/Black")
        return ("🔴⚫", "Red/Black"), ("🔵⚪", "Blue/White")

    for i, item in enumerate(payload["items"], 1):
        score = float(item["score"])
        A = list(item["A"])
        B = list(item["B"])
        breakdown = dict(item["breakdown"])

        disp_A = ([captainA] + A) if (captain_mode and captainA) else A[:]
        disp_B = ([captainB] + B) if (captain_mode and captainB) else B[:]

        a_eff = float(breakdown.get("mmr_avg_a", 1000))
        b_eff = float(breakdown.get("mmr_avg_b", 1000))
        probA = _expected_score_league_calibrated(a_eff, b_eff)

        (teamA_icon, teamA_color), (teamB_icon, teamB_color) = choose_colors(i)
        palA = palette(teamA_color)
        palB = palette(teamB_color)

        st.markdown(
            f"""
            <div style="
                text-align:center;
                border: 3px solid #444;
                border-radius: 12px;
                padding: 10px;
                margin: 10px 0 20px 0;
                font-size: 1.3em;
                font-weight: 700;
                background: linear-gradient(90deg, {palA['bg']} 0%, {palB['bg']} 100%);
            ">
                {teamA_icon} <span style="color:{palA['fg']};">Team A: {teamA_color}</span>
                &nbsp;&nbsp;vs&nbsp;&nbsp;
                {teamB_icon} <span style="color:{palB['fg']};">Team B: {teamB_color}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clean matchup card (no repeated sub-headers)
        boxA = f"""
        <div style='background:{palA["bg"]};border:1px solid {palA["fg"]}40;border-radius:12px;padding:10px 12px;'>
        <div style="font-weight:800;margin-bottom:6px;">Avg MMR {a_eff:.0f}</div>
        <div style="line-height:1.7;">
            {'<br>'.join(_name_ui(p, players_df) for p in disp_A)}
        </div>
        </div>
        """

        boxB = f"""
        <div style='background:{palB["bg"]};border:1px solid {palB["fg"]}40;border-radius:12px;padding:10px 12px;text-align:right;'>
        <div style="font-weight:800;margin-bottom:6px;">Avg MMR {b_eff:.0f}</div>
        <div style="line-height:1.7;">
            {'<br>'.join(_name_ui(p, players_df) for p in disp_B)}
        </div>
        </div>
        """
        cL, cR = st.columns(2)
        with cL:
            st.markdown(boxA, unsafe_allow_html=True)
        with cR:
            st.markdown(boxB, unsafe_allow_html=True)

        st.caption(f"Fairness Score: {score:.2f} (lower = more balanced)")

        use_key = f"use_matchup_{i}"
        if st.button("✅ Use this matchup for Matchday Card", key=use_key):
            st.session_state.team_a = disp_A
            st.session_state.team_b = disp_B
            st.session_state.selected_matchup = {"team_a": disp_A, "team_b": disp_B, "picked_index": i}
            st.session_state.mdk_expanded = True
            st.rerun()
        with st.expander("🔍 Fairness Breakdown"):
            st.write("**MMR & Form (effective rating)**")
            st.markdown(
                f"- Team A effective rating: `{float(breakdown.get('mmr_avg_a', 0)):.1f}`  \n"
                f"- Team B effective rating: `{float(breakdown.get('mmr_avg_b', 0)):.1f}`  \n"
                f"- Difference: `{float(breakdown.get('mmr_diff', 0)):.2f}`"
            )

            st.write("**Spread (team shape)**")
            st.markdown(
                f"- Spread A / B: `{float(breakdown.get('spread_a', 0)):.2f}` vs `{float(breakdown.get('spread_b', 0)):.2f}`  \n"
                f"- Difference: `{float(breakdown.get('spread_diff', 0)):.2f}`"
            )

            st.write("**Chemistry**")
            st.markdown(
                f"- Chemistry A / B: `{float(breakdown.get('chem_a', 0)):.2f}` vs `{float(breakdown.get('chem_b', 0)):.2f}`  \n"
                f"- Difference: `{float(breakdown.get('chem_diff', 0)):.2f}`"
            )


            st.write("**Chemistry density & links**")
            st.markdown(
                f"- Density A / B: `{float(breakdown.get('chem_density_a', 0)):.2f}` vs `{float(breakdown.get('chem_density_b', 0)):.2f}`  \n"
                f"- Density diff: `{float(breakdown.get('chem_density_diff', 0)):.2f}`  \n"
                f"- Top-link share A / B: `{float(breakdown.get('chem_top_share_a', 0)):.2f}` vs `{float(breakdown.get('chem_top_share_b', 0)):.2f}`"
            )

            
            st.write("**Trio synergy (triangle chemistry)**")
            st.markdown(
                f"- Trio synergy A / B: `{float(breakdown.get('trio_a', 0)):.2f}` vs `{float(breakdown.get('trio_b', 0)):.2f}`  \n"
                f"- Difference: `{float(breakdown.get('trio_diff', 0)):.2f}`  \n"
                f"- Trio density A / B: `{float(breakdown.get('trio_density_a', 0)):.2f}` vs `{float(breakdown.get('trio_density_b', 0)):.2f}`  \n"
                f"- Negative trio total (penalty): `{float(breakdown.get('trio_negative_total', 0)):.2f}`"
            )

            st.write("**Bad pairings (avoid these links)**")
            st.markdown(
                f"- Bad-pair score A / B: `{float(breakdown.get('badpair_a', 0)):.2f}` vs `{float(breakdown.get('badpair_b', 0)):.2f}`  \n"
                f"- Total badness: `{float(breakdown.get('badpair_total', 0)):.2f}`"
            )

            st.write("**Matchup memory**")
            sim = breakdown.get("similarity_debug", {}) or {}
            st.markdown(
                f"- Similarity penalty: `{float(breakdown.get('similarity_penalty', 0)):.2f}`  \n"
                f"- Closest historic similarity: `{float(sim.get('similarity', 0)):.2f}`"
                + (f"  \n- Historic score: `{sim.get('score')}`" if sim.get('score') else "")
                + (f"  \n- Historic date: `{sim.get('date')}`" if sim.get('date') else "")
            )
            st.write("**Combined Score**")
            st.markdown(f"- Total: **`{float(breakdown.get('fairness_score', score)):.2f}`**")

        st.divider()


    with st.expander("ℹ️ How this works"):
        st.write(
            """
- **MMR** is the main balancing factor (result-based rating).
- **Form** adjusts effective rating using recent results (score-based).
- **Fitness** adds a small adjustment if you maintain that field.
- **Spread** checks the *shape* of each team (prevents one carry + passengers vs a flat team).
- **Chemistry** is based on how well players have historically performed together (results-only).
            """.strip()
        )