import re
from typing import List, Set, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils.db_utils import load_players_df
from utils.team_ai_engine import get_engine_state, clean_name, evaluate_teams
from utils.ui_components import page_header


# ----------------------------
# Helpers: display names
# ----------------------------
def _display_name(nm: str) -> str:
    s = str(nm or "").strip()
    if not s:
        return ""
    parts = [p for p in s.replace("_", " ").split() if p]
    out = []
    for p in parts:
        if len(p) <= 2:
            out.append(p.upper())
        elif len(p) == 3 and p.isupper():
            out.append(p)
        else:
            out.append(p[0].upper() + p[1:])
    return " ".join(out)


def _name_ui(name: str, players_df: pd.DataFrame) -> str:
    try:
        row = players_df[players_df["name"] == name]
        if not row.empty and "display_name" in row.columns:
            dn = str(row.iloc[0].get("display_name") or "").strip()
            if dn:
                return dn
    except Exception:
        pass
    return _display_name(name)


def _matches_played(players_df: pd.DataFrame, nm: str) -> int:
    try:
        row = players_df.loc[players_df["name"] == nm]
        if row.empty:
            return 0
        return int(row.iloc[0].get("matches_played") or 0)
    except Exception:
        return 0


def _auto_core_and_ringers(players_df: pd.DataFrame, selected: List[str]) -> Tuple[List[str], List[str]]:
    """Auto-detect ringers from today's 10.

    We look for a clear separation in appearances inside the selected group:
    - compute median appearances of the selected 10
    - define ringer_cut = max(3, floor(0.6 * median))
    - a player is a ringer if appearances <= ringer_cut AND appearances < median
    """
    mps = [_matches_played(players_df, p) for p in selected]
    if not mps:
        return selected, []

    med = float(pd.Series(mps).median())
    if med < 8:
        # not enough history separation to reliably label ringers
        return selected, []

    ringer_cut = max(3, int(med * 0.6))
    ringers = [
        p
        for p in selected
        if (_matches_played(players_df, p) < med and _matches_played(players_df, p) <= ringer_cut)
    ]
    core = [p for p in selected if p not in ringers]
    return core, ringers


# ----------------------------
# Parsing & scoring
# ----------------------------
def _split_team_str(val) -> List[str]:
    raw = str(val or "").strip()
    if not raw or raw.lower() == "nan":
        return []
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    parts = re.split(r"[,\n\r\|/;]+", raw)
    out: List[str] = []
    for p in parts:
        nm = str(p).strip().strip("'").strip('"').strip()
        if nm:
            out.append(nm)
    return out


def _parse_goal_diff(score_txt: str):
    s = (score_txt or "").strip()
    if not s:
        return None
    s = s.replace("–", "-").replace("—", "-")
    m = re.search(r"(\d+)\s*-\s*(\d+)", s)
    if not m:
        return None
    try:
        return abs(int(m.group(1)) - int(m.group(2)))
    except Exception:
        return None


def _safe_dt(x):
    try:
        return pd.to_datetime(x, errors="coerce", utc=True)
    except Exception:
        try:
            dt = pd.to_datetime(x, errors="coerce")
            if pd.isna(dt):
                return pd.NaT
            if getattr(dt, "tzinfo", None) is None:
                return dt.tz_localize("UTC")
            return dt.tz_convert("UTC")
        except Exception:
            return pd.NaT


def _matches_from_engine() -> pd.DataFrame:
    try:
        eng = get_engine_state(force_reload=False)
    except TypeError:
        eng = get_engine_state()
    matches = eng.get("matches") if isinstance(eng, dict) else None
    if matches is None:
        return pd.DataFrame()
    try:
        return matches.copy()
    except Exception:
        return matches


def _score_match(
    overlap_core: int,
    overlap_total: int,
    goal_diff: int | None,
    dt: pd.Timestamp | None,
    core_present: bool,
):
    s = 0.0
    s += overlap_core * 100.0 if core_present else overlap_total * 100.0
    s += overlap_total * 10.0

    if goal_diff is not None:
        s += max(0.0, 25.0 - (float(goal_diff) * 6.0))

    if dt is not None and pd.notna(dt):
        now_utc = pd.Timestamp.now(tz="UTC")
        dt_utc = dt.tz_convert("UTC") if getattr(dt, "tzinfo", None) is not None else dt.tz_localize("UTC")
        days = (now_utc - dt_utc).days
        s += max(0.0, 20.0 - min(20.0, float(days) / 30.0))

    return s


# ----------------------------
# UI: match card (self-contained HTML+CSS)
# ----------------------------
def _pill(name_clean: str, today_set: Set[str], name_lookup: dict) -> str:
    """Render a player pill.

    - Players in today's 10 -> normal pill
    - Players NOT in today's 10 (i.e. historical-only) -> red highlight
    """
    cls = "active" if name_clean in today_set else "out"
    label = name_lookup.get(name_clean, name_clean)
    return f"<span class='pm-pill {cls}'>{label}</span>"


def _render_match_card(m: dict, today_set: Set[str], name_lookup: dict, idx: int, core_count: int):
    date_txt = str(m.get("date") or "").strip()
    venue = str(m.get("venue") or "").strip()
    meta = " · ".join([t for t in [date_txt, venue] if t])

    scoreline = str(m.get("scoreline") or "").strip() or "—"

    overlap_note = (
        f"core overlap {int(m.get('overlap_core', 0))}/{core_count}"
        if core_count
        else f"overlap {int(m.get('overlap_total', 0))}/10"
    )

    ta = m.get("team_a") or []
    tb = m.get("team_b") or []
    missing_today = m.get("missing_today") or []


    # Changes vs today (team-generator style)
    # OUT: players in this historical match but not in today's 10
    # IN : players in today's 10 but not in this historical match (missing_today)
    hist_only = [p for p in (ta + tb) if p not in today_set]
    swap_html = ""
    if hist_only or missing_today:
        out_pills = "".join(f"<span class='pm-pill out'>{name_lookup.get(p,p)}</span>" for p in hist_only)
        in_pills = "".join(f"<span class='pm-pill in'>{name_lookup.get(p,p)}</span>" for p in missing_today)
        swap_html = (
            "<div class='pm-subtitle'>Changes vs today</div>"
            "<div class='pm-swaprow'>"
            + (out_pills if out_pills else "<span class='pm-meta'>No OUT</span>")
            + "<span class='pm-arrow'>→</span>"
            + (in_pills if in_pills else "<span class='pm-meta'>No IN</span>")
            + "</div>"
        )



    # Suggested tweak (uses AI fairness model)
    tweak_html = ""
    try:
        current_fair, _ = evaluate_teams(ta, tb)
        best = {"fair": current_fair, "kind": None, "a": None, "b": None, "out": None, "inn": None, "team": None}

        # Case 1: 10/10 overlap -> try best single swap between teams
        if not missing_today:
            for a in ta:
                for b in tb:
                    ta2 = [b if x == a else x for x in ta]
                    tb2 = [a if x == b else x for x in tb]
                    f2, _ = evaluate_teams(ta2, tb2)
                    if f2 < best["fair"]:
                        best.update({"fair": f2, "kind": "swap", "a": a, "b": b})

        # Case 2: Not full overlap -> try best replacement of historical-only with missing-today player
        hist_only = [p for p in (ta + tb) if p not in today_set]
        if missing_today and hist_only:
            for p_in in missing_today:
                for p_out in hist_only:
                    if p_out in ta:
                        ta2 = [p_in if x == p_out else x for x in ta]
                        tb2 = tb[:]
                        team_side = "A"
                    else:
                        tb2 = [p_in if x == p_out else x for x in tb]
                        ta2 = ta[:]
                        team_side = "B"
                    f2, _ = evaluate_teams(ta2, tb2)
                    if f2 < best["fair"]:
                        best.update({"fair": f2, "kind": "replace", "out": p_out, "inn": p_in, "team": team_side})

        # Only show if it improves things meaningfully
        if best.get('kind') is not None and best.get('fair') is not None and best['fair'] < current_fair:
            if best["kind"] == "swap":
                a_lab = name_lookup.get(best["a"], best["a"])
                b_lab = name_lookup.get(best["b"], best["b"])
                tweak_html = (
                    "<div class='pm-subtitle'>Suggested tweak</div>"
                    "<div class='pm-swaprow'>"
                    f"<span class='pm-pill out'>{a_lab}</span>"
                    "<span class='pm-arrow'>↔</span>"
                    f"<span class='pm-pill out'>{b_lab}</span>"
                    f"<span class='pm-meta'> </span>"
                    "</div>"
                )
            elif best["kind"] == "replace":
                out_lab = name_lookup.get(best["out"], best["out"])
                in_lab = name_lookup.get(best["inn"], best["inn"])
                team_lab = "Team A" if best.get("team") == "A" else "Team B"
                tweak_html = (
                    "<div class='pm-subtitle'>Suggested tweak</div>"
                    "<div class='pm-swaprow'>"
                    f"<span class='pm-meta'>{team_lab}:</span>"
                    f"<span class='pm-pill out'>{out_lab}</span>"
                    "<span class='pm-arrow'>→</span>"
                    f"<span class='pm-pill in'>{in_lab}</span>"
                    f"<span class='pm-meta'> </span>"
                    "</div>"
                )
    except Exception:
        tweak_html = ""
    css = """
<style>
html,body{margin:0;padding:0;background:transparent;}
.pm-card{
  border-radius:16px;
  padding:16px 18px;
  border:1px solid rgba(255,255,255,0.10);
  background:linear-gradient(180deg, rgba(255,255,255,0.05), rgba(0,0,0,0.0));
  box-shadow:0 0 16px rgba(0,0,0,0.55);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  color: rgba(255,255,255,0.92);
}
.pm-meta{
  color:#cfcfcf;
  opacity:0.9;
  font-size:0.95rem;
  text-align:center;
}
.pm-scoreline{
  display:flex;
  justify-content:center;
  align-items:baseline;
  gap:18px;
  margin-top:6px;
  margin-bottom:12px;
}
.pm-scoreteam{
  font-size:2.2rem;
  font-weight:1000;
  letter-spacing:0.5px;
  text-shadow:0 3px 20px rgba(0,0,0,0.65);
  opacity:0.95;
  white-space:nowrap;
}
.pm-scoreteam.pm-a{ color:#93c5fd; }
.pm-scoreteam.pm-b{ color:#fca5a5; }
.pm-score{
  font-size:2.2rem;
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
.pm-pill.active{
  background:rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.14);
}
.pm-pill.out{
  background:rgba(239,68,68,0.22);
  border:1px solid rgba(239,68,68,0.55);
  color:#ffe4e6;
  box-shadow:0 0 10px rgba(239,68,68,0.15);
}
.pm-pill.in{
  background:rgba(34,197,94,0.14);
  border:1px solid rgba(34,197,94,0.35);
  color:#bbf7d0;
}
.pm-subtitle{margin-top:10px;color:#bdbdbd;font-size:0.95rem;font-weight:800;opacity:0.95;}
.pm-swaprow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:6px;}
.pm-arrow{opacity:0.9;font-weight:900;margin:0 2px;}
.pm-mini{
  text-align:center;
  color:#cfcfcf;
  opacity:0.85;
  font-size:0.9rem;
  margin-top:2px;
}
</style>
"""

    html = f"""{css}
<div class="pm-card">
  <div class="pm-meta">#{idx} · {overlap_note}</div>
  <div class="pm-mini">{meta}</div>

  <div class="pm-scoreline">
    <span class="pm-scoreteam pm-a">Team A</span>
    <span class="pm-score">{scoreline}</span>
    <span class="pm-scoreteam pm-b">Team B</span>
  </div>

  <div class="pm-grid">
    <div class="pm-team a">
      <div class="pm-team-h">Team A Lineup</div>
      <div class="pm-line">{''.join(_pill(p, today_set, name_lookup) for p in ta)}</div>
    </div>

    <div class="pm-team b">
      <div class="pm-team-h">Team B Lineup</div>
      <div class="pm-line">{''.join(_pill(p, today_set, name_lookup) for p in tb)}</div>
    </div>
  </div>

  {swap_html}
  {tweak_html}
</div>
"""
    components.html(html, height=380, scrolling=False)


# ----------------------------
# Page
# ----------------------------
def render_matchday_memory_page(show_header: bool = True):
    if show_header:
        page_header(
            "Matchday Memory",
            "Pick today's 10 and we'll find the closest historical games.",
            center=True,
            divider=True,
        )

    players_df = load_players_df()
    if players_df is None or players_df.empty or "name" not in players_df.columns:
        st.warning("No players found.")
        return

    if "is_active" in players_df.columns:
        try:
            players_df = players_df[players_df["is_active"].fillna(1).astype(int) == 1]
        except Exception:
            pass

    names = players_df["name"].dropna().astype(str).tolist()

    sel = st.multiselect(
        "Select today's 10 players",
        options=names,
        default=[],
        format_func=lambda n: _name_ui(n, players_df),
    )

    if len(sel) != 10:
        st.warning("Select exactly 10 players.")
        return
    # Auto ringers/core (internal only)
    core, ringers = _auto_core_and_ringers(players_df, sel)
    today_set: Set[str] = {clean_name(p) for p in sel if clean_name(p)}
    core_set: Set[str] = set()  # not used for overlap (ringers included)

    # Lookup for UI names keyed by clean_name
    name_lookup = {clean_name(n): _name_ui(n, players_df) for n in names if clean_name(n)}

    matches = _matches_from_engine()
    if matches.empty:
        st.warning("No match history available.")
        return

    for col in ["team_a", "team_b", "score", "result", "venue", "date", "id"]:
        if col not in matches.columns:
            matches[col] = None
    # Overlap requirement: require at least 7/10 overlap
        core_required = 5

    candidates = []
    for _, r in matches.iterrows():
        ta_raw = _split_team_str(r.get("team_a"))
        tb_raw = _split_team_str(r.get("team_b"))

        ta = [clean_name(p) for p in ta_raw if clean_name(p)]
        tb = [clean_name(p) for p in tb_raw if clean_name(p)]
        if not ta or not tb:
            continue

        hist_all = set(ta) | set(tb)
        overlap_total = len(hist_all & today_set)
        overlap_core = overlap_total  # ringers included; core not used for overlap
        if overlap_total < 7:
            continue

        missing_today = sorted(list(today_set - hist_all))

        score_txt = str(r.get("score") or "")
        gd = _parse_goal_diff(score_txt)
        dt = _safe_dt(r.get("date"))

        score = _score_match(
            overlap_core=overlap_core,
            overlap_total=overlap_total,
            goal_diff=gd,
            dt=dt,
            core_present=bool(core_set),
        )

        candidates.append(
            {
                "score": float(score),
                "overlap_total": int(overlap_total),
                "overlap_core": int(overlap_core),
                "goal_diff": gd,
                "date": r.get("date"),
                "__dt": dt,
                "team_a": ta,
                "team_b": tb,
                "missing_today": missing_today,
                "scoreline": score_txt,
                "venue": r.get("venue"),
                "match_id": r.get("id"),
            }
        )

    if not candidates:
        st.info("No close historical references found for this group.")
        return

    df = pd.DataFrame(candidates)
    # Sorting: overlap first, then closest scoreline (lower goal diff), then recency
    df["_gd_sort"] = df["goal_diff"].fillna(999).astype(float)
    df = df.sort_values(["overlap_core", "overlap_total", "_gd_sort", "__dt"], ascending=[False, False, True, False], na_position="last")

    top_n = st.slider("How many reference games to show", 1, 10, 6)
    top = df.head(int(top_n)).to_dict(orient="records")

    st.subheader("Historical Games")

    for i, m in enumerate(top, start=1):
        _render_match_card(
            m,
            today_set=today_set,
            name_lookup=name_lookup,
            idx=i,
            core_count=len(core_set),
        )

