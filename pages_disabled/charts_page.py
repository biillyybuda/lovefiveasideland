import streamlit as st
import pandas as pd
import plotly.express as px
import textwrap
from io import StringIO
from collections import defaultdict
from utils.mmr_utils import get_season_mmr, get_current_season_start


from utils.db_utils import get_conn, get_current_league_id, sql_df, load_matches_df
from utils.export_utils import df_to_png, fig_to_png_bytes  # kept for compatibility (may be used elsewhere)
from utils.stats_shared import (
    get_chemistry_df,
    get_intensity_df,
    get_pair_chemistry,
    get_pair_intensity,
)
from utils.names import display_name as _fallback_display_name
from utils.ui_components import page_header


# ----------------------------
# Helpers
# ----------------------------
def _split_team(val: str):
    return [p.strip() for p in str(val or "").split(",") if str(p).strip()]


def to_display(key: str, name_map: dict) -> str:
    if key is None:
        return ""
    k = str(key).strip()
    return name_map.get(k, _fallback_display_name(k))


# ----------------------------
# Team History Directory (matches where selected players were together)
# ----------------------------
def _norm_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00A0", " ").replace("\u200B", "")
    s = " ".join(s.strip().lower().split())
    return s

def _team_has_all(team_list: list[str], wanted: set[str]) -> bool:
    team_set = {_norm_key(p) for p in (team_list or []) if str(p).strip()}
    return wanted.issubset(team_set)

def _parse_scoreline(score_txt: str):
    try:
        s = str(score_txt or "").strip()
        if "-" in s:
            a, b = s.split("-", 1)
            return int(a.strip()), int(b.strip())
    except Exception:
        pass
    return None, None

def _outcome_for_selected_group(mrow, wanted_set: set[str], same_team_only: bool):
    """
    Returns: ("W"/"D"/"L"/None, side_str)
    side_str is "A" or "B" when selected group is clearly on one side, else None.
    """
    ta = _split_team(mrow.get("team_a", ""))
    tb = _split_team(mrow.get("team_b", ""))

    ta_set = {_norm_key(p) for p in ta}
    tb_set = {_norm_key(p) for p in tb}

    in_a = len(ta_set.intersection(wanted_set))
    in_b = len(tb_set.intersection(wanted_set))

    # Decide which side is "the selected group" sits on
    side = None
    if same_team_only:
        # group must be mainly on one side (your filter already enforces this)
        side = "A" if in_a >= in_b else "B"
    else:
        # could be split across both teams; outcome doesn't make sense
        return None, None

    res = str(mrow.get("result", "") or "").strip().upper()
    if res == "D":
        res = "DRAW"

    if res in ("DRAW", ""):
        return ("D" if res == "DRAW" else None), side

    if side == "A":
        return ("W" if res == "A" else "L"), side
    else:
        return ("W" if res == "B" else "L"), side


def _render_team_history_match_card(
    mrow,
    highlight_a: set[str],
    highlight_b: set[str],
    name_map: dict,
    compare_a: set[str] | None = None,
    compare_b: set[str] | None = None,
    changes_label: str = "Changes vs selected lineup",
):
    # Build pills for both teams; highlight selected players
    ta = _split_team(mrow.get("team_a", ""))
    tb = _split_team(mrow.get("team_b", ""))

    def pill(p, highlight_set):
        key = _norm_key(p)
        cls = "pm-pill active" if key in (highlight_set or set()) else "pm-pill"
        return f"<span class='{cls}'>{to_display(p, name_map)}</span>"

    pills_a = "".join([pill(p, highlight_a) for p in ta])
    pills_b = "".join([pill(p, highlight_b) for p in tb])

    # --- Changes vs selected lineup (optional) ---
    def _changes_block(team_list: list[str], compare_set: set[str] | None):
        if not compare_set:
            return ""
        team_set = {_norm_key(p) for p in team_list if str(p).strip()}

        outs = sorted(list(team_set - compare_set))
        ins = sorted(list(compare_set - team_set))

        if not outs and not ins:
            return f"<div class='pm-changes'><div class='pm-changes-h'>{changes_label}</div><div class='pm-changes-none'>No changes</div></div>"

        # Pair off outs -> ins like substitutions
        pairs = []
        for i in range(min(len(outs), len(ins))):
            o = outs[i]
            n = ins[i]
            pairs.append(
                f"<div class='pm-swap'><span class='pm-pill pm-out'>{to_display(o, name_map)}</span>"
                f"<span class='pm-arrow'>→</span>"
                f"<span class='pm-pill pm-in'>{to_display(n, name_map)}</span></div>"
            )

        extra_out = outs[len(pairs):]
        extra_in = ins[len(pairs):]

        extras_html = ""
        if extra_out:
            extras_html += "<div class='pm-extras'><span class='pm-extras-h'>Also out:</span> " + ", ".join(to_display(x, name_map) for x in extra_out) + "</div>"
        if extra_in:
            extras_html += "<div class='pm-extras'><span class='pm-extras-h'>Also in:</span> " + ", ".join(to_display(x, name_map) for x in extra_in) + "</div>"

        swaps_html = "".join(pairs) if pairs else ""
        return f"""
        <div class='pm-changes'>
          <div class='pm-changes-h'>{changes_label}</div>
          {swaps_html if swaps_html else ""}
          {extras_html if extras_html else ""}
        </div>
        """

    changes_a_html = _changes_block(ta, compare_a)
    changes_b_html = _changes_block(tb, compare_b)

    date_txt = str(mrow.get("date", "") or "").strip()
    score_txt = str(mrow.get("score", "") or "").strip()
    gA, gB = _parse_scoreline(score_txt)
    big_score = f"{gA}–{gB}" if (gA is not None and gB is not None) else (score_txt if score_txt else "—")

    html = f'''
<div class="pm-card">
  <div class="pm-top pm-scorebar">
    <div class="pm-meta" style="text-align:center;width:100%;">{date_txt}</div>
    <div class="pm-scoreline">
      <span class="pm-scoreteam pm-a">Team A</span>
      <span class="pm-score">{big_score}</span>
      <span class="pm-scoreteam pm-b">Team B</span>
    </div>
  </div>

  <div class="pm-grid">
    <div class="pm-team a">
      <div class="pm-team-h"><span>Team A Lineup</span></div>
      <div class="pm-line">{pills_a}</div>
      {changes_a_html}
    </div>

    <div class="pm-team b">
      <div class="pm-team-h"><span>Team B Lineup</span></div>
      <div class="pm-line">{pills_b}</div>
      {changes_b_html}
    </div>
  </div>
</div>
'''
    st.html(textwrap.dedent(html))

def render_group_vs_group(matches_df: pd.DataFrame, all_players: list[str], name_map: dict, key_prefix: str):
    """Matchup History: pick Group A + Group B and return historical matches where they faced each other.
    Cards are normalised so Group A is always shown on the left (Team A panel).
    """
    c1, c2 = st.columns(2)
    with c1:
        group_a = st.multiselect(
            "Group A",
            options=all_players,
            default=[],
            key=f"{key_prefix}_a",
            format_func=lambda k: to_display(k, name_map),
        )
    with c2:
        group_b = st.multiselect(
            "Group B",
            options=all_players,
            default=[],
            key=f"{key_prefix}_b",
            format_func=lambda k: to_display(k, name_map),
        )

    if len(group_a) == 0 or len(group_b) == 0:
        st.info("Pick at least 1 player in each group.")
        return

    A = {_norm_key(p) for p in group_a}
    B = {_norm_key(p) for p in group_b}

    # Minimum overlap sliders (how many from each group must appear on their side)
    s1, s2 = st.columns(2)
    with s1:
        if len(group_a) == 1:
            min_a = 1
            st.caption("Min from Group A on their side: **1**")
        else:
            min_a = st.slider(
                "Min from Group A on their side",
                min_value=1,
                max_value=min(5, len(group_a)),
                value=min(2, len(group_a)),
                step=1,
                key=f"{key_prefix}_min_a",
            )
    with s2:
        if len(group_b) == 1:
            min_b = 1
            st.caption("Min from Group B on their side: **1**")
        else:
            min_b = st.slider(
                "Min from Group B on their side",
                min_value=1,
                max_value=min(5, len(group_b)),
                value=min(2, len(group_b)),
                step=1,
                key=f"{key_prefix}_min_b",
            )

    def _counts(team_list: list[str], group_set: set[str]) -> int:
        team_set = {_norm_key(p) for p in (team_list or []) if str(p).strip()}
        return len(team_set.intersection(group_set))

    # Sorting newest-first
    df = matches_df.copy()
    df["__dt"] = pd.to_datetime(df.get("date", None), errors="coerce")
    df = df.sort_values("__dt", ascending=False, na_position="last")

    # Filter matches
    hits = []
    a_wins = b_wins = draws = 0

    for _, r in df.iterrows():
        ta = _split_team(r.get("team_a", ""))
        tb = _split_team(r.get("team_b", ""))

        a_in_a = _counts(ta, A)
        a_in_b = _counts(tb, A)
        b_in_a = _counts(ta, B)
        b_in_b = _counts(tb, B)

        # Must have the groups on opposite teams, meeting overlap thresholds
        ok_ab = (a_in_a >= int(min_a) and b_in_b >= int(min_b))
        ok_ba = (a_in_b >= int(min_a) and b_in_a >= int(min_b))
        if not (ok_ab or ok_ba):
            continue

        # Decide which side Group A was on in this match (for record + card normalisation)
        group_a_side = "A" if ok_ab else "B"

        # Record
        res = str(r.get("result", "") or "").strip().upper()
        if res == "D":
            res = "DRAW"

        if res == "DRAW":
            draws += 1
        elif (group_a_side == "A" and res == "A") or (group_a_side == "B" and res == "B"):
            a_wins += 1
        else:
            b_wins += 1

        # Store row + which side Group A was on
        rr = r.copy()
        rr["__group_a_side"] = group_a_side
        hits.append(rr)

    if not hits:
        st.info("No historical matches found where Group A faced Group B (in this season view).")
        return

    out = pd.DataFrame(hits)

    # Summary
    total = len(out)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", total)
    c2.metric("Group A Wins", a_wins)
    c3.metric("Draws", draws)
    c4.metric("Group B Wins", b_wins)

    # Render cards with Group A always on the left
    for _, r in out.iterrows():
        group_a_side = str(r.get("__group_a_side", "A"))

        # If Group A was on Team B in the stored data, swap teams (and swap scoreline)
        if group_a_side == "B":
            # swap team strings
            r2 = r.copy()
            r2["team_a"], r2["team_b"] = r.get("team_b", ""), r.get("team_a", "")

            # swap scoreline if parsable
            gA, gB = _parse_scoreline(str(r.get("score", "") or ""))
            if gA is not None and gB is not None:
                r2["score"] = f"{gB}-{gA}"
            _render_team_history_match_card(r2, A, B, name_map, compare_a=A, compare_b=B, changes_label="Changes vs selected lineup")
        else:
            _render_team_history_match_card(r, A, B, name_map, compare_a=A, compare_b=B, changes_label="Changes vs selected lineup")


def render_team_history_directory(matches_df: pd.DataFrame, all_players: list[str], name_map: dict, key_prefix: str = "thd"):
    """Teammate History: pick players and list matches where they were together on the SAME team."""
    # Lightweight CSS (matched to Team Generator "Past Matchups")
    st.markdown(
        '''
        <style>
        .pm-card{
          border-radius:16px;
          padding:16px 18px;
          border:1px solid rgba(255,255,255,0.10);
          background:linear-gradient(180deg, rgba(255,255,255,0.05), rgba(0,0,0,0.0));
          box-shadow:0 0 16px rgba(0,0,0,0.55);
          margin-top:10px;
        }
        .pm-top{display:flex;align-items:center;margin-bottom:6px;}
        .pm-scorebar{flex-direction:column;justify-content:center;gap:6px;margin-bottom:12px;}
        .pm-scoreline{display:flex;justify-content:center;align-items:baseline;gap:18px;}
        .pm-scoreteam{
          font-size:2.2rem;font-weight:1000;letter-spacing:0.5px;
          text-shadow:0 3px 20px rgba(0,0,0,0.65);opacity:0.95;white-space:nowrap;
        }
        .pm-scoreteam.pm-a{ color:#93c5fd; }
        .pm-scoreteam.pm-b{ color:#fca5a5; }
        .pm-meta{color:#cfcfcf;opacity:0.9;font-size:0.95rem;}
        .pm-score{font-size:2.2rem;font-weight:1000;letter-spacing:1px;color:#ffffff;text-shadow:0 3px 20px rgba(0,0,0,0.65);}
        .pm-grid{display:grid;grid-template-columns: 1fr 1fr;gap:14px;}
        .pm-team{border-radius:14px;padding:12px 14px;border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);}
        .pm-team.a{background:linear-gradient(135deg, rgba(59,130,246,0.14), rgba(255,255,255,0.02));border:1px solid rgba(59,130,246,0.22);}
        .pm-team.b{background:linear-gradient(225deg, rgba(239,68,68,0.14), rgba(255,255,255,0.02));border:1px solid rgba(239,68,68,0.22);}
        .pm-team-h{font-weight:900;font-size:1.05rem;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;}
        .pm-line{display:flex;flex-wrap:wrap;gap:8px;}
        .pm-pill{
          padding:6px 10px;border-radius:999px;font-weight:800;font-size:0.98rem;
          border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.04);
        }
        /* Selected players */
        .pm-pill.active{
          background:rgba(34,197,94,0.14);
          border:1px solid rgba(34,197,94,0.40);
          color:#bbf7d0;
          box-shadow:0 0 10px rgba(34,197,94,0.10);
        }
        .pm-changes{margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08);}
        .pm-changes-h{font-weight:900;font-size:0.9rem;color:rgba(255,255,255,0.75);margin-bottom:6px;}
        .pm-changes-none{color:rgba(255,255,255,0.55);font-size:0.9rem;}
        .pm-swap{display:flex;align-items:center;gap:10px;margin:6px 0;flex-wrap:wrap;}
        .pm-arrow{opacity:0.7;font-weight:900;}
        .pm-pill.pm-out{border-color:rgba(239,68,68,0.35);background:rgba(239,68,68,0.08);color:#fecaca;}
        .pm-pill.pm-in{border-color:rgba(34,197,94,0.35);background:rgba(34,197,94,0.10);color:#bbf7d0;}
        .pm-extras{margin-top:6px;color:rgba(255,255,255,0.65);font-size:0.9rem;line-height:1.25;}
        .pm-extras-h{font-weight:900;color:rgba(255,255,255,0.75);}
        </style>
        ''',
        unsafe_allow_html=True,
    )

    default = st.session_state.get(f"{key_prefix}_default", []) or []

    sel_players = st.multiselect(
        "Players",
        options=all_players,
        default=default,
        key=f"{key_prefix}_players",
        format_func=lambda k: to_display(k, name_map),
    )

    if len(sel_players) < 2:
        st.info("Select at least 2 players to search.")
        return

    wanted = {_norm_key(p) for p in sel_players}

    # For 2 players: fixed = 2 (no slider)
    if len(sel_players) == 2:
        min_together = 2
        st.caption("Minimum selected players together: **2**")
    else:
        min_together = st.slider(
            "Minimum selected players together",
            min_value=2,
            max_value=min(5, len(sel_players)),
            value=min(3, len(sel_players)),
            step=1,
            key=f"{key_prefix}_min_together",
            help="Return matches where at least this many of the selected players appeared together on the SAME team.",
        )

    def overlap_count(team_list: list[str]) -> int:
        team_set = {_norm_key(p) for p in (team_list or []) if str(p).strip()}
        return len(team_set.intersection(wanted))

    df = matches_df.copy()
    df["__dt"] = pd.to_datetime(df.get("date", None), errors="coerce")
    df = df.sort_values("__dt", ascending=False, na_position="last")

    hits = []
    for _, r in df.iterrows():
        ta = _split_team(r.get("team_a", ""))
        tb = _split_team(r.get("team_b", ""))

        a_n = overlap_count(ta)
        b_n = overlap_count(tb)

        # ALWAYS same-team logic for Teammate History
        if max(a_n, b_n) < int(min_together):
            continue

        hits.append(r)

    if not hits:
        st.info("No matches found for that combination (in this season view).")
        return

    out = pd.DataFrame(hits)

    # -------------------------
    # Auto summary (W/D/L + win % + avg goal diff)
    # -------------------------
    w = d = l = 0
    goal_diffs = []
    scorelines = []

    for _, r0 in out.iterrows():
        # same_team_only=True because this directory is always same-team
        outcome, _ = _outcome_for_selected_group(r0, wanted, same_team_only=True)
        if outcome == "W":
            w += 1
        elif outcome == "D":
            d += 1
        elif outcome == "L":
            l += 1

        gA, gB = _parse_scoreline(str(r0.get("score", "") or ""))
        if gA is not None and gB is not None:
            goal_diffs.append(abs(gA - gB))
            scorelines.append((abs(gA - gB), f"{gA}–{gB}", str(r0.get("date", "") or "")))

    total = len(out)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", total)
    c2.metric("W-D-L", f"{w}-{d}-{l}")
    win_pct = (w / total * 100) if total else 0.0
    c3.metric("Win %", f"{win_pct:.1f}%")
    if goal_diffs:
        c4.metric("Avg Goal Diff", f"{(sum(goal_diffs)/len(goal_diffs)):.2f}")
    else:
        c4.metric("Avg Goal Diff", "—")

    st.markdown(f"**Found:** {len(out)} match(es)")

    # Render (always expanded; no collapse toggle)
    def _highlight_sets_for_row(row):
        ta = _split_team(row.get("team_a", ""))
        tb = _split_team(row.get("team_b", ""))
        a_n = overlap_count(ta)
        b_n = overlap_count(tb)
        # Highlight only the selected players that are on the SAME team that triggered the match
        if a_n >= int(min_together) and a_n >= b_n:
            ha = {_norm_key(p) for p in ta}.intersection(wanted)
            hb = set()
        elif b_n >= int(min_together):
            ha = set()
            hb = {_norm_key(p) for p in tb}.intersection(wanted)
        else:
            ha = set()
            hb = set()
        return ha, hb

    for _, r in out.iterrows():
        ha, hb = _highlight_sets_for_row(r)
        _render_team_history_match_card(r, ha, hb, name_map)


def get_season_filter_ui(matches_df: pd.DataFrame, suffix=""):
    """
    Returns (season_mode, selected_year, season_start, matches_filtered)
    """
    df = matches_df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    years = sorted([int(y) for y in df["date_dt"].dropna().dt.year.unique().tolist()]) # type: ignore
    if not years:
        years = [pd.Timestamp.today().year]

    season_mode = st.selectbox(
        "Season View",
        options=["Rolling (all years)", "Single Year (season reset)"],
        index=0,
        key=f"charts_season_mode_{suffix}",
    )

    selected_year = None
    if season_mode == "Single Year (season reset)":
        selected_year = st.selectbox(
            "Select Year",
            options=years,
            index=len(years) - 1,
            key=f"charts_selected_year_{suffix}",
        )

    if season_mode == "Single Year (season reset)":
        df_filt = df[df["date_dt"].dt.year == int(selected_year)].copy() # type: ignore
        season_start = f"{int(selected_year)}-01-01" # type: ignore
    else:
        df_filt = df.copy()
        season_start = None

    return season_mode, selected_year, season_start, df_filt

@st.cache_data(ttl=300, show_spinner=False)
def get_name_map_cached(league_id: int) -> dict:
    df = sql_df(
        "SELECT name, display_name FROM public.players WHERE league_id=%s",
        (league_id,),
    )
    df["name"] = df["name"].fillna("").astype(str)
    df["display_name"] = df["display_name"].fillna("").astype(str)

    name_map = {}
    for _, r in df.iterrows():
        key = r["name"].strip()
        ui = r["display_name"].strip() or _fallback_display_name(key)
        name_map[key] = ui
    return name_map

@st.cache_data(ttl=300, show_spinner=False)
def load_charts_summary_cached(league_id: int):
    match_summary = sql_df(
        """
        SELECT
            COUNT(*)::int AS processed_matches,
            MIN(date) AS first_match,
            MAX(date) AS last_match
        FROM public.matches
        WHERE processed = 1 AND league_id = %s
        """,
        (league_id,),
    )
    player_summary = sql_df(
        "SELECT COUNT(*)::int AS players FROM public.players WHERE league_id = %s",
        (league_id,),
    )
    years = sql_df(
        """
        SELECT DISTINCT EXTRACT(YEAR FROM date::date)::int AS year
        FROM public.matches
        WHERE processed = 1 AND league_id = %s AND date IS NOT NULL
        ORDER BY year DESC
        """,
        (league_id,),
    )
    recent = sql_df(
        """
        SELECT date, team_a, team_b, score, result
        FROM public.matches
        WHERE processed = 1 AND league_id = %s
        ORDER BY date DESC, id DESC
        LIMIT 5
        """,
        (league_id,),
    )
    return match_summary, player_summary, years, recent


@st.cache_data(ttl=300, show_spinner=False)
def load_charts_processed_matches_cached(league_id: int) -> pd.DataFrame:
    # Keep the heavy chart payload narrow: these views only need match identity,
    # dates, teams and results, not every column in the matches table.
    return sql_df(
        """
        SELECT id, date, team_a, team_b, score, result, processed, team_a_avg, team_b_avg
        FROM public.matches
        WHERE processed = 1 AND league_id = %s
        ORDER BY date ASC, id ASC
        """,
        (league_id,),
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_player_mmr_history_cached(league_id: int, player_name: str) -> pd.DataFrame:
    return sql_df(
        """
        SELECT
            mh.match_id,
            mh.date,
            mh.mmr_before,
            mh.mmr_after,
            p.name
        FROM public.mmr_history mh
        JOIN public.players p ON mh.player_id = p.id
        WHERE mh.league_id = %s
          AND p.name = %s
        ORDER BY mh.date ASC, mh.id ASC
        """,
        (league_id, player_name),
    )


def render_charts_fast_summary(league_id: int):
    st.markdown("## Summary")
    match_summary, player_summary, years, recent = load_charts_summary_cached(league_id)

    match_row = match_summary.iloc[0].to_dict() if not match_summary.empty else {}
    player_row = player_summary.iloc[0].to_dict() if not player_summary.empty else {}
    year_list = [str(int(y)) for y in years["year"].dropna().tolist()] if not years.empty else []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed Matches", int(match_row.get("processed_matches") or 0))
    c2.metric("Players", int(player_row.get("players") or 0))
    c3.metric("First Match", str(match_row.get("first_match") or "-")[:10])
    c4.metric("Latest Match", str(match_row.get("last_match") or "-")[:10])

    if year_list:
        st.caption("Available seasons: " + ", ".join(year_list))

    st.markdown("### Latest Results")
    if recent.empty:
        st.info("No processed matches yet.")
    else:
        name_map = get_name_map_cached(league_id)
        recent_view = recent.copy()
        recent_view["team_a"] = recent_view["team_a"].apply(
            lambda s: ", ".join(to_display(p, name_map) for p in _split_team(s))
        )
        recent_view["team_b"] = recent_view["team_b"].apply(
            lambda s: ", ".join(to_display(p, name_map) for p in _split_team(s))
        )
        st.dataframe(recent_view, use_container_width=True, hide_index=True)

    st.info("Choose a detailed view above when you want the heavier charts and tables.")

@st.cache_data(ttl=300, show_spinner=False)
def load_mmr_history_cached(league_id: int) -> pd.DataFrame:
    return sql_df(
        """
        SELECT
            m.id AS match_id,
            m.date AS match_date,
            mh.player_id,
            p.name AS player_key,
            p.display_name AS player_display,
            mh.mmr_before,
            mh.mmr_after
        FROM public.mmr_history mh
        JOIN public.matches m ON mh.match_id = m.id
        JOIN public.players p ON mh.player_id = p.id
        WHERE m.processed = 1 AND m.league_id = %s
        ORDER BY m.date ASC, mh.id ASC
        """,
        (league_id,),
    )

@st.cache_data(ttl=300, show_spinner=False)
def season_baseline_map(df_json: str) -> dict[int, float]:
    df = pd.read_json(StringIO(df_json))

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").fillna(0).astype(int)

    firsts = (
        df.sort_values(["player_id", "date_dt"])
          .groupby("player_id", as_index=True)
          .first()
    )

    mmr_before_s = pd.to_numeric(firsts["mmr_before"], errors="coerce").fillna(0.0).astype(float)
    return {int(pid): float(mmr_before_s.loc[pid]) for pid in mmr_before_s.index}

@st.cache_data(ttl=300, show_spinner=False)
def load_mmr_history_full_cached(league_id: int) -> pd.DataFrame:
    return sql_df(
        """
        SELECT
            mh.*,
            p.name
        FROM public.mmr_history mh
        JOIN public.players p ON mh.player_id = p.id
        JOIN public.matches m ON mh.match_id = m.id
        WHERE m.league_id = %s
        ORDER BY mh.date ASC
        """,
        (league_id,),
    )

@st.cache_data(ttl=300, show_spinner=False)
def compute_global_win_att(matches_json: str, min_games: int):
    matches = pd.read_json(StringIO(matches_json))

    games_played = defaultdict(int)
    wins = defaultdict(int)
    appearances = defaultdict(int)

    for _, m in matches.iterrows():
        ta = _split_team(m.get("team_a", ""))
        tb = _split_team(m.get("team_b", ""))
        res = (m.get("result") or "").upper()

        allp = set(ta + tb)
        for p in allp:
            if p:
                games_played[p] += 1

        for p in ta:
            if p:
                appearances[p] += 1
        for p in tb:
            if p:
                appearances[p] += 1

        if res == "A":
            for p in ta:
                if p:
                    wins[p] += 1
        elif res == "B":
            for p in tb:
                if p:
                    wins[p] += 1

    eligible = sorted([p for p, gp in games_played.items() if int(gp) >= int(min_games)])

    # IMPORTANT: always return DataFrames with consistent columns
    win_rows = []
    for p in eligible:
        gp = int(games_played.get(p, 0))
        w = int(wins.get(p, 0))
        wp = round((w / gp * 100), 1) if gp else 0.0
        win_rows.append({"Player": p, "Win %": wp})

    att_rows = []
    total_matches = len(matches)
    for p in eligible:
        ap = int(appearances.get(p, 0))
        attp = round((ap / total_matches * 100), 1) if total_matches else 0.0
        att_rows.append({"Player": p, "Attendance %": attp})

    df_win = pd.DataFrame(win_rows, columns=["Player", "Win %"])
    df_att = pd.DataFrame(att_rows, columns=["Player", "Attendance %"])

    return eligible, df_win, df_att







# ----------------------------
# 📈 MMR Progression Over Time (moved from Dashboard)
# ----------------------------
def render_mmr_progression_over_time(suffix="", season_mode=None, selected_year=None, season_start=None, matches=None):
    st.subheader("📈 MMR Progression Over Time")

    # Pull full MMR history (processed matches only)
    league_id = get_current_league_id()
    df_mmr = load_mmr_history_cached(league_id)

    if df_mmr.empty:
        st.info("No MMR history yet.")
        return

    # Apply match filter if provided
    if matches is not None and not matches.empty and "id" in matches.columns:
        allowed_ids = set(matches["id"].astype(int).tolist())
        df_mmr = df_mmr[df_mmr["match_id"].astype(int).isin(allowed_ids)].copy()

    df_mmr["date_dt"] = pd.to_datetime(df_mmr["match_date"], errors="coerce")
    df_mmr["date_dt"] = df_mmr["date_dt"].dt.normalize()  # type: ignore

    name_map = get_name_map_cached(get_current_league_id())
    df_mmr["player_label"] = df_mmr["player_key"].apply(lambda k: to_display(k, name_map))

    # Season-reset display MMR
    show_season = (season_mode == "Single Year (season reset)" and season_start)
    if show_season:
        payload = df_mmr[["player_id", "mmr_before", "date_dt"]].to_json(date_format="iso")
        base = season_baseline_map(payload)

        df_mmr["player_id_int"] = pd.to_numeric(df_mmr["player_id"], errors="coerce").fillna(0).astype(int)
        df_mmr["base_before"] = df_mmr["player_id_int"].map(base).fillna(df_mmr["mmr_before"].astype(float))

        df_mmr["mmr_plot"] = 1000.0 + (df_mmr["mmr_after"].astype(float) - df_mmr["base_before"])
        ycol = "mmr_plot"
        ylab = "Season MMR (reset to 1000)"
    else:
        df_mmr["mmr_plot"] = df_mmr["mmr_after"].astype(float)
        ycol = "mmr_plot"
        ylab = "Rolling MMR"

    options = sorted(df_mmr["player_label"].dropna().unique().tolist())

    player_choice = st.multiselect(
        "Select Player(s)",
        options=options,
        default=[],
        key=f"mmr_progress_players_{suffix}",
    )

    if not player_choice:
        st.info("Choose one or more players to plot their MMR progression.")
        return

    df_filtered = df_mmr[df_mmr["player_label"].isin(player_choice)].copy()

    fig = px.line(
        df_filtered,
        x="date_dt",
        y=ycol,
        color="player_label",
        markers=True,
        title="MMR Over Time",
        labels={
            "date_dt": "Match Date",
            ycol: ylab,
            "player_label": "Player",
        },
    )
    st.plotly_chart(fig, use_container_width=True, key=f"mmr_progress_fig_{suffix}")

# ----------------------------
# 🌍 GLOBAL OVERVIEW
# ----------------------------
def render_global_overview(suffix="", season_mode=None, selected_year=None, season_start=None, matches=None):
    st.markdown("## 🌍 Global Overview")

    conn = get_conn()
    try:
        matches = matches.copy() if matches is not None else pd.read_sql(
            """
            SELECT id, date, team_a, team_b, score, result, processed, team_a_avg, team_b_avg
            FROM public.matches
            WHERE processed=1 AND league_id=%s
            ORDER BY date ASC, id ASC
            """,
            conn,
            params=(get_current_league_id(),),
        )
        players = sql_df(
            "SELECT name FROM public.players WHERE league_id=%s ORDER BY name",
            (get_current_league_id(),),
        )

        if matches.empty:
            st.info("No processed matches yet.")
            return

        name_map = get_name_map_cached(get_current_league_id())
        plist = players["name"].tolist()

        # ----------------------------
        # Minimum games filter (Global Overview)
        # ----------------------------
        min_games = st.slider(
            "Minimum games played",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            help="Hide players with too few games in the selected season view.",
            key=f"global_min_games_{suffix}",
        )
        payload = matches.to_json(date_format="iso")
        eligible_players, df_win_raw, df_att_raw = compute_global_win_att(payload, int(min_games))


        # 🥇 Win % by Player
        st.subheader("🥇 Win % by Player")

        df_win = df_win_raw.copy()
        df_win["Name"] = df_win["Player"].apply(lambda k: to_display(k, name_map))
        df_win = df_win[["Name", "Win %"]]

        if df_win.empty:
            st.info("No eligible players for Win % (try lowering the minimum games filter).")
            df_win = pd.DataFrame(columns=["Name", "Win %"])
        else:
            df_win = df_win.sort_values("Win %", ascending=False)
        fig_win = px.bar(df_win, x="Name", y="Win %", title="Win % by Player", text="Win %")
        fig_win.update_traces(textposition="outside")
        fig_win.update_yaxes(range=[0, df_win["Win %"].max() * 1.15 if not df_win.empty else 100])
        fig_win.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig_win, use_container_width=True, key=f"fig_win_global_{suffix}")

        # 📅 Attendance (%)
        st.subheader("📅 Attendance (All Players)")
        total_matches = len(matches)
        df_att = df_att_raw.copy()
        df_att["Player"] = df_att["Player"].apply(lambda k: to_display(k, name_map))

        if df_att.empty:
            st.info("No eligible players for Attendance (try lowering the minimum games filter).")
        else:
            df_att = df_att.sort_values("Attendance %", ascending=False)

        fig_att = px.bar(df_att, x="Player", y="Attendance %", title="Attendance % (All Players)", text="Attendance %")
        fig_att.update_traces(textposition="outside")
        fig_att.update_yaxes(range=[0, df_att["Attendance %"].max() * 1.15 if not df_att.empty else 100])
        fig_att.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig_att, use_container_width=True, key=f"fig_att_global_{suffix}")

        if not st.toggle(
            "Load duo and rivalry tables",
            value=False,
            key=f"global_load_relationship_tables_{suffix}",
            help="Keeps the first chart load quick online. Turn on when you want the heavier pair analysis.",
        ):
            st.info("Duo chemistry and rivalry tables are ready to load when needed.")
            return

        # 🤝 Top 10 Duos (Chemistry) & ⚔️ Top 10 Rivalries (Intensity)
        st.subheader("🤝 Top Duos & ⚔️ Rivalries")

        # These two DataFrames come from stats_shared (so formula changes live there)
        chem_df = (
            get_chemistry_df(conn, matches_df=matches)
            .sort_values(by="chemistry", ascending=False)
            .head(10)
        )

        intensity_df = (
            get_intensity_df(conn, matches_df=matches)
            .sort_values(by="intensity", ascending=False)
            .head(10)
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Top 10 Duos by Chemistry**")
            if chem_df is None or chem_df.empty:
                st.info("No chemistry data yet.")
            else:
                chem_display = (
                    chem_df.rename(
                        columns={
                            "player_a": "Player A",
                            "player_b": "Player B",
                            "matches": "Matches",
                            "wins": "Wins",
                            "win_pct": "Win %",
                            "chemistry": "Chemistry",
                        }
                    )[["Player A", "Player B", "Matches", "Win %", "Chemistry"]]
                    .reset_index(drop=True)
                )
                chem_display["Player A"] = chem_display["Player A"].apply(lambda k: to_display(k, name_map))
                chem_display["Player B"] = chem_display["Player B"].apply(lambda k: to_display(k, name_map))
                st.dataframe(chem_display, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**Top 10 Rivalries by Intensity**")
            if intensity_df is None or intensity_df.empty:
                st.info("No rivalry data yet.")
            else:
                full_rivalries = []

                # IMPORTANT:
                # - get_pair_intensity() is the single source of truth for the pair stats
                # - we pass df=intensity_df to avoid recomputing intensity_df inside stats_shared
                # - we pass matches_df=matches so season filtering matches the page
                for _, row in intensity_df.iterrows():
                    a_key = row.get("player_a", "")
                    b_key = row.get("player_b", "")

                    pair_stats = get_pair_intensity(
                        a_key,
                        b_key,
                        conn=conn,
                        df=intensity_df,
                        matches_df=matches,
                    ) or {}

                    full_rivalries.append(
                        {
                            "Player A": to_display(a_key, name_map),
                            "Player B": to_display(b_key, name_map),
                            "Matches": int(pair_stats.get("matches") or 0),
                            "Player A Wins": int(pair_stats.get("wins_a") or 0),
                            "Player B Wins": int(pair_stats.get("wins_b") or 0),
                            "Draws": int(pair_stats.get("draws") or 0),
                            "Avg Goal Diff": round(float(pair_stats.get("avg_goal_diff") or 0.0), 2),
                            "Intensity": round(float(pair_stats.get("intensity") or 0.0), 3),
                        }
                    )

                rival_display = pd.DataFrame(full_rivalries)[
                    ["Player A", "Player B", "Matches", "Player A Wins", "Player B Wins", "Draws", "Avg Goal Diff", "Intensity"]
                ].reset_index(drop=True)

                st.dataframe(rival_display, use_container_width=True, hide_index=True)


    finally:
        conn.close()


# ----------------------------
# 🎯 PLAYER INSIGHTS (NO VIDEO STATS)
# ----------------------------
def render_player_insights(suffix="", season_mode=None, selected_year=None, season_start=None, matches=None):
    st.markdown("## 🎯 Player Insights")

    conn = get_conn()
    try:
        players = pd.read_sql("SELECT id, name FROM players WHERE league_id = %s ORDER BY name", conn, params=(get_current_league_id(),))
        matches = matches.copy() if matches is not None else load_matches_df().query("processed == 1").copy()
        league_id = get_current_league_id()

        if matches.empty or players.empty:
            st.info("No processed matches / players found yet.")
            return

        name_map = get_name_map_cached(get_current_league_id())
        plist = players["name"].tolist()

        sel_player = st.selectbox(
            "Select a player to view detailed stats:",
            ["— Select —"] + plist,
            key=f"player_selectbox_{suffix}",
            format_func=lambda k: "— Select —" if k == "— Select —" else to_display(k, name_map),
        )

        if sel_player == "— Select —" or not sel_player:
            sel_key = str(sel_player).strip().lower()
            st.info("Select a player to view personal charts and tables.")
            return

        # Matches containing this player (safe contains)
        player_matches = matches[
            matches["team_a"].fillna("").astype(str).str.contains(sel_player, regex=False)
            | matches["team_b"].fillna("").astype(str).str.contains(sel_player, regex=False)
        ].copy()

        if player_matches.empty:
            st.info("No match data found for this player.")
            return

        total_matches = len(player_matches)

        # Win % (match results only)
        win_count = 0
        for _, m in player_matches.iterrows():
            ta = _split_team(m.get("team_a", ""))
            tb = _split_team(m.get("team_b", ""))
            res = str(m.get("result", "")).strip().upper()
            if (sel_player in ta and res == "A") or (sel_player in tb and res == "B"):
                win_count += 1

        win_pct = (win_count / total_matches * 100) if total_matches else 0.0

        # Player MMR history
        df_p = load_player_mmr_history_cached(league_id, sel_player)
        df_p["date_dt"] = pd.to_datetime(df_p["date"], errors="coerce")
        if season_mode == "Single Year (season reset)" and selected_year is not None:
            df_p = df_p[df_p["date_dt"].dt.year == int(selected_year)].copy() # type: ignore
        df_p = df_p.sort_values("date", ascending=True)

        # --------------------------
        # 📊 Performance Overview (MMR + Results only)
        # --------------------------
        st.subheader("📊 Performance Overview")

        current_mmr = None
        start_mmr = None
        net_mmr_change = 0.0
        avg_mmr_delta = 0.0
        use_season_start = None

        if not df_p.empty:
            if season_mode == "Single Year (season reset)" and season_start:
                use_season_start = season_start
            else:
                use_season_start = None  # rolling
            pid = int(players[players["name"] == sel_player].iloc[0]["id"])

            if use_season_start:
                # Season reset to 1000: baseline is the FIRST mmr_before in the season
                base_before = float(df_p.iloc[0]["mmr_before"])
                start_mmr = 1000.0
                current_mmr = 1000.0 + (float(df_p.iloc[-1]["mmr_after"]) - base_before)
            else:
                start_mmr = float(df_p.iloc[0]["mmr_before"])
                current_mmr = float(df_p.iloc[-1]["mmr_after"])

            net_mmr_change = current_mmr - start_mmr

            df_p["mmr_delta"] = df_p["mmr_after"] - df_p["mmr_before"]
            avg_mmr_delta = float(df_p["mmr_delta"].mean()) if len(df_p) else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Matches Played", int(total_matches))
        c2.metric("Win %", f"{win_pct:.1f}%")
        if current_mmr is None:
            c3.metric("Current MMR", "—")
        else:
            c3.metric("Current MMR", f"{current_mmr:.0f}", f"{net_mmr_change:+.0f} vs start")

        # --- Calculate draws & losses ---
        draw_count = 0
        loss_count = 0

        for _, m in player_matches.iterrows():
            ta = _split_team(m.get("team_a", ""))
            tb = _split_team(m.get("team_b", ""))
            res = str(m.get("result", "")).strip().upper()

            if res == "DRAW":
                draw_count += 1
            elif (sel_player in ta and res == "A") or (sel_player in tb and res == "B"):
                pass  # already counted as win
            else:
                loss_count += 1

        # --- Display ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Wins", int(win_count))
        c2.metric("Draws", int(draw_count))
        c3.metric("Losses", int(loss_count))
        c4.metric("Avg MMR Δ / Match", f"{avg_mmr_delta:+.2f}" if not df_p.empty else "—")
        # --------------------------
        # 📈 MMR Over Time
        # --------------------------
        st.subheader("📈 MMR Over Time")
        if use_season_start:
            base_before = float(df_p.iloc[0]["mmr_before"])
            df_p["mmr_plot"] = 1000.0 + (df_p["mmr_after"].astype(float) - base_before)
            ycol = "mmr_plot"
            ylab = "Season MMR (reset to 1000)"
        else:
            df_p["mmr_plot"] = df_p["mmr_after"].astype(float)
            ycol = "mmr_plot"
            ylab = "Rolling MMR"

        fig = px.line(
            df_p,
            x="date",
            y=ycol,
            markers=True,
            title=f"MMR Trend for {to_display(sel_player, name_map)}",
            labels={ycol: ylab},
        )
        st.plotly_chart(fig, use_container_width=True, key=f"player_mmr_chart_{sel_player}_{suffix}")

        # --------------------------
        # 🔥 Recent Form (Last 5 Matches) — outcomes + MMR only
        # --------------------------
        st.subheader("🔥 Recent Form (Last 5 Matches)")
        recent_hist = df_p.sort_values("date", ascending=False).head(5).copy()

        if recent_hist.empty:
            st.info("No recent matches found.")
        else:
            recent_rows = []
            form_icons = []

            for _, r in recent_hist.iterrows():
                mid = r["match_id"]
                m = matches[matches["id"] == mid]
                if m.empty:
                    continue

                mrow = m.iloc[0]
                ta = _split_team(mrow.get("team_a", ""))
                tb = _split_team(mrow.get("team_b", ""))
                res = str(mrow.get("result", "")).strip().upper()

                if (sel_player in ta and res == "A") or (sel_player in tb and res == "B"):
                    outcome = "Win"
                    icon = "🟩"
                elif res == "DRAW":
                    outcome = "Draw"
                    icon = "⬜"
                else:
                    outcome = "Loss"
                    icon = "🟥"

                mmr_delta = float(r["mmr_after"] - r["mmr_before"])
                # teammates / opponents
                my_team = ta if sel_player in ta else tb
                opp_team = tb if sel_player in ta else ta

                teammates = [p for p in my_team if p != sel_player]
                opponents = [p for p in opp_team if p]

                teammates_display = ", ".join(to_display(p, name_map) for p in teammates)
                opponents_display = ", ".join(to_display(p, name_map) for p in opponents)

                # score
                score_txt = str(mrow.get("score", "") or "").strip()

                recent_rows.append(
                    {
                        "Date": mrow.get("date", ""),
                        "Outcome": outcome,
                        "Score": score_txt,
                        "MMR Δ": round(mmr_delta, 1),
                        "Teammates": teammates_display,
                        "Opponents": opponents_display,
                    }
                )
                form_icons.append(icon)

            if recent_rows:
                st.markdown(f"**Recent Form:** {''.join(form_icons)}")

                df_recent = pd.DataFrame(recent_rows)
                emoji_map = {"Win": "🟩 Win", "Loss": "🟥 Loss", "Draw": "⬜ Draw"}
                df_recent["Outcome"] = df_recent["Outcome"].map(emoji_map)

                st.dataframe(
                    df_recent.sort_values("Date", ascending=False)
                    .set_index("Date")[["Outcome", "Score", "MMR Δ", "Teammates", "Opponents"]],
                    use_container_width=True,
                )
            else:
                st.info("Not enough matches to calculate form.")

        # --------------------------
        # All Games Involving This Player
        # --------------------------
        st.subheader("All Games Involving This Player")

        all_game_rows = []
        for _, r in df_p.sort_values("date", ascending=False).iterrows():
            mid = r.get("match_id")
            m = matches[matches["id"] == mid]
            if m.empty:
                continue

            mrow = m.iloc[0]
            ta = _split_team(mrow.get("team_a", ""))
            tb = _split_team(mrow.get("team_b", ""))
            res = str(mrow.get("result", "")).strip().upper()
            if res == "D":
                res = "DRAW"

            on_team_a = sel_player in ta
            on_team_b = sel_player in tb
            if not on_team_a and not on_team_b:
                continue

            if res == "DRAW":
                outcome = "Draw"
            elif (on_team_a and res == "A") or (on_team_b and res == "B"):
                outcome = "Win"
            else:
                outcome = "Loss"

            my_team = ta if on_team_a else tb
            opp_team = tb if on_team_a else ta

            all_game_rows.append(
                {
                    "Date": mrow.get("date", ""),
                    "Outcome": outcome,
                    "Score": str(mrow.get("score", "") or "").strip(),
                    "Team": "Team A" if on_team_a else "Team B",
                    "Teammates": ", ".join(to_display(p, name_map) for p in my_team if p != sel_player),
                    "Opponents": ", ".join(to_display(p, name_map) for p in opp_team),
                    "MMR Before": round(float(r.get("mmr_before", 0.0)), 1),
                    "MMR After": round(float(r.get("mmr_after", 0.0)), 1),
                    "MMR Delta": round(float(r.get("mmr_after", 0.0)) - float(r.get("mmr_before", 0.0)), 1),
                }
            )

        if not all_game_rows:
            st.info("No full match log found for this player.")
        else:
            outcome_filter = st.radio(
                "Filter games",
                options=["All", "Wins", "Draws", "Losses"],
                index=0,
                horizontal=True,
                key=f"player_all_games_filter_{suffix}",
            )

            df_all_games = pd.DataFrame(all_game_rows)
            if outcome_filter == "Wins":
                df_all_games = df_all_games[df_all_games["Outcome"] == "Win"]
            elif outcome_filter == "Draws":
                df_all_games = df_all_games[df_all_games["Outcome"] == "Draw"]
            elif outcome_filter == "Losses":
                df_all_games = df_all_games[df_all_games["Outcome"] == "Loss"]

            st.dataframe(
                df_all_games,
                use_container_width=True,
                hide_index=True,
                height=min(620, 72 + (len(df_all_games) * 35)),
            )

        # -------------------------------------------------
        # 🤝 Best Teammates (Chemistry) — display names
        # -------------------------------------------------
        st.markdown("### 🤝 Best Teammates (Chemistry)")
        chem_df = get_chemistry_df(conn, matches_df=matches)

        player_chem = chem_df[
            (chem_df["player_a"] == sel_player) | (chem_df["player_b"] == sel_player)
        ].copy() if not chem_df.empty else pd.DataFrame()

        if not player_chem.empty:
            player_chem["Teammate"] = player_chem.apply(
                lambda r: r["player_b"] if r["player_a"] == sel_player else r["player_a"],
                axis=1,
            )
            player_chem["Teammate"] = player_chem["Teammate"].apply(lambda k: to_display(k, name_map))

            st.dataframe(
                player_chem[["Teammate", "matches", "wins", "win_pct", "chemistry"]]
                .rename(
                    columns={
                        "matches": "Games",
                        "wins": "Wins",
                        "win_pct": "Win %",
                        "chemistry": "Chemistry",
                    }
                )
                .sort_values("Chemistry", ascending=False)
                .head(10)
                .set_index("Teammate"),
                use_container_width=True,
            )
        else:
            st.info("No chemistry data yet.")

        # -------------------------------------------------
        # ⚔️ Toughest Opponents (Intensity) — display names
        # -------------------------------------------------
        st.markdown("### ⚔️ Toughest Opponents (Intensity)")

        # 🔑 normalised key for safe comparison
        sel_key = str(sel_player).strip().lower()

        # ✅ season-filtered intensity table
        int_df = get_intensity_df(conn, matches_df=matches)

        player_int = (
            int_df[
                (int_df["player_a"].astype(str).str.strip().str.lower() == sel_key)
                | (int_df["player_b"].astype(str).str.strip().str.lower() == sel_key)
            ].copy()
            if int_df is not None and not int_df.empty
            else pd.DataFrame()
        )

        if not player_int.empty:
            rows = []
            for _, r in player_int.iterrows():
                opponent = r["player_b"] if str(r["player_a"]).strip().lower() == sel_key else r["player_a"]
                stats = get_pair_intensity(sel_player, opponent, conn=conn, df=int_df, matches_df=matches)
                games = stats.get("matches", 0)
                if games == 0:
                    continue

                wins = stats.get("wins_a", 0)     # wins for sel_player (orientation assumed by helper)
                losses = stats.get("wins_b", 0)   # wins for opponent
                win_pct_vs = round((wins / games * 100), 1) if games else 0.0

                rows.append(
                    {
                        "Opponent": to_display(opponent, name_map),
                        "Games": games,
                        "Wins": wins,
                        "Losses": losses,
                        "W%": win_pct_vs,
                        "Intensity": round(stats.get("intensity", 0.0), 3),
                    }
                )

            df_display = pd.DataFrame(rows).sort_values("Intensity", ascending=False)
            st.dataframe(
                df_display[["Opponent", "Games", "Wins", "Losses", "W%", "Intensity"]].set_index("Opponent"),
                use_container_width=True,
            )
        else:
            st.info("No rivalry data yet.")
    finally:
        conn.close()


# ----------------------------
# ⚔️ Head-to-Head & 🤝 Duo Chemistry (Final Styled) — rewritten
# ----------------------------
def render_head_to_head_section(season_mode=None, selected_year=None, season_start=None, matches=None):
    st.markdown("## ⚔️ Head-to-Head & 🤝 Duo Chemistry")

    conn = get_conn()
    try:
        # --- Load matches (use passed filter if provided) ---
        if matches is not None:
            matches_df = matches.copy()
        else:
            matches_df = load_matches_df().query("processed == 1").copy()

        # Safety: ensure dataframe exists
        if matches_df is None:
            matches_df = pd.DataFrame()

        # --- Players / name map ---
        players = pd.read_sql("SELECT id, name FROM players WHERE league_id = %s ORDER BY name", conn, params=(get_current_league_id(),))
        name_map = get_name_map_cached(get_current_league_id())
        all_players = sorted(players["name"].dropna().astype(str).tolist())
        select_options = ["— Select —"] + all_players

        # --- UI Selectors (same UI) ---
        c1, c2 = st.columns(2)
        with c1:
            player_a = st.selectbox(
                "Player A",
                select_options,
                index=0,
                key="h2h_player_a",
                format_func=lambda k: "— Select —" if k == "— Select —" else to_display(k, name_map),
            )
        with c2:
            player_b = st.selectbox(
                "Player B",
                select_options,
                index=0,
                key="h2h_player_b",
                format_func=lambda k: "— Select —" if k == "— Select —" else to_display(k, name_map),
            )

        if player_a == "— Select —" or player_b == "— Select —" or player_a == player_b:
            st.info("Select two players to view their head-to-head stats.")
            return

        # --- Robust normalisation helpers ---
        def _norm(s: str) -> str:
            # Lowercase, trim, remove invisible chars, collapse whitespace
            if s is None:
                return ""
            s = str(s)
            s = s.replace("\u00A0", " ")   # non-breaking space
            s = s.replace("\u200B", "")    # zero-width space
            s = s.replace("\u200C", "")
            s = s.replace("\u200D", "")
            s = s.strip().lower()
            # collapse internal whitespace
            s = " ".join(s.split())
            return s

        def _split_team(val):
            """
            Handles:
            - "['SAM K', 'BILLY']"
            - "sam k, billy"
            - None / empty
            """
            if val is None:
                return []

            s = str(val).strip().lower()

            # Remove list-like wrappers
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]

            # Remove quotes
            s = s.replace("'", "").replace('"', "")

            # Normalise separators
            for sep in [";", "|", "/"]:
                s = s.replace(sep, ",")

            return [p.strip() for p in s.split(",") if p.strip()]

        def _score_to_ints(sc):
            try:
                if isinstance(sc, str) and "-" in sc:
                    a, b = sc.split("-", 1)
                    return int(a.strip()), int(b.strip())
            except Exception:
                pass
            return None, None

        player_a_key = _norm(player_a)
        player_b_key = _norm(player_b)

        # If matches_df empty, still render UI but show zeros
        if matches_df.empty:
            pair_chem = {"matches": 0, "wins": 0, "chemistry": 0.0}
            pair_int = {"matches": 0, "wins_a": 0, "wins_b": 0, "draws": 0, "avg_goal_diff": 0.0, "intensity": 0.0}
            chem_df_local = pd.DataFrame({"chemistry": []})
            int_df_local = pd.DataFrame({"intensity": []})
        else:
            # --- Build chemistry / intensity tables from the SAME matches_df (season-aware) ---
            chem_df_local = get_chemistry_df(conn, matches_df=matches_df)
            int_df_local = get_intensity_df(conn, matches_df=matches_df)

            # --- Counts from matches_df (this is the source of truth for filters) ---
            together = 0
            wins_together = 0

            faced = 0
            wins_a = 0
            wins_b = 0
            draws = 0
            gd_list = []

            # Ensure required columns exist (avoid silent failures)
            # Expected: team_a, team_b, result, score
            for _, m in matches_df.iterrows():
                ta = _split_team(m.get("team_a", ""))
                tb = _split_team(m.get("team_b", ""))
                res = _norm(m.get("result", "")).upper()  # "A", "B", "DRAW"
                # Normalize "draw" variants
                if res == "D":
                    res = "DRAW"

                # together
                if player_a_key in ta and player_b_key in ta:
                    together += 1
                    if res == "A":
                        wins_together += 1
                elif player_a_key in tb and player_b_key in tb:
                    together += 1
                    if res == "B":
                        wins_together += 1

                # head-to-head
                a_vs_b = (player_a_key in ta and player_b_key in tb)
                b_vs_a = (player_a_key in tb and player_b_key in ta)
                if a_vs_b or b_vs_a:
                    faced += 1

                    a_sc, b_sc = _score_to_ints(m.get("score", ""))
                    if a_sc is not None and b_sc is not None:
                        gd_list.append(abs(a_sc - b_sc))

                    if res == "DRAW":
                        draws += 1
                    else:
                        # Determine which side player_a was on and whether that side won
                        player_a_on_a = player_a_key in ta
                        if (player_a_on_a and res == "A") or ((not player_a_on_a) and res == "B"):
                            wins_a += 1
                        else:
                            wins_b += 1

            pair_chem = {"matches": together, "wins": wins_together, "chemistry": 0.0}
            pair_int = {
                "matches": faced,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "draws": draws,
                "avg_goal_diff": (sum(gd_list) / len(gd_list)) if gd_list else 0.0,
                "intensity": 0.0,
            }

            # --- Pull chemistry/intensity SCORE from tables (optional) ---
            # Chemistry
            try:
                chem_score = get_pair_chemistry(player_a_key, player_b_key, conn, df=chem_df_local)
                if chem_score.get("matches", 0) == 0:
                    chem_score = get_pair_chemistry(player_b_key, player_a_key, conn, df=chem_df_local)
                pair_chem["chemistry"] = float(chem_score.get("chemistry", 0.0))
            except Exception:
                pair_chem["chemistry"] = 0.0

            # Intensity
            try:
                int_score = get_pair_intensity(player_a_key, player_b_key, conn, df=int_df_local)
                if int_score.get("matches", 0) == 0:
                    alt = get_pair_intensity(player_b_key, player_a_key, conn, df=int_df_local)
                    if alt.get("matches", 0) > 0:
                        int_score = {"intensity": alt.get("intensity", 0.0)}
                pair_int["intensity"] = float(int_score.get("intensity", 0.0))
            except Exception:
                pair_int["intensity"] = 0.0

        # --- Display names (same UI) ---
        a_disp = to_display(player_a, name_map)
        b_disp = to_display(player_b, name_map)

        # ------------------------------
        # ⚔️ Rivalry Section
        # ------------------------------
        st.markdown(f"### ⚔️ Head-to-Head: {a_disp} vs {b_disp}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Matches Played", pair_int.get("matches", 0))
        with c2:
            st.metric(
                f"{a_disp} Wins",
                pair_int.get("wins_a", 0),
                f"{(pair_int.get('wins_a', 0) / pair_int['matches'] * 100):.1f}%" if pair_int.get("matches") else None,
            )
        with c3:
            st.metric(
                f"{b_disp} Wins",
                pair_int.get("wins_b", 0),
                f"{(pair_int.get('wins_b', 0) / pair_int['matches'] * 100):.1f}%" if pair_int.get("matches") else None,
            )

        draws_v = pair_int.get("draws", 0)
        avg_gd = pair_int.get("avg_goal_diff", 0.0)
        int_val = pair_int.get("intensity", None)
        int_display = "—" if int_val is None else f"{float(int_val):.2f}"

        st.markdown(
            f"Draws: {draws_v}  |  Avg Goal Diff: {avg_gd:.2f}  |  Intensity Score: {int_display}"
        )

        # --- Rivalry Badge ---
        all_int = []
        if int_df_local is not None and not int_df_local.empty and "intensity" in int_df_local.columns:
            all_int = int_df_local["intensity"].dropna().tolist()

        int_val = pair_int.get("intensity", None)

        if int_val is None or not all_int:
            percentile_r = 0.0
        else:
            # ensure list is numeric
            all_int_num = [float(x) for x in all_int if x is not None]
            rank_r = sum(s < float(int_val) for s in all_int_num)
            percentile_r = (rank_r / len(all_int_num) * 100) if all_int_num else 0.0

        def rivalry_label(p):
            if p >= 90:
                return ("🟩", "Legendary Rivalry")
            elif p >= 70:
                return ("🟦", "Fierce Rivalry")
            elif p >= 40:
                return ("🟨", "Developing Rivalry")
            elif p >= 10:
                return ("🟧", "Minor Rivalry")
            else:
                return ("🟥", "Cold Rivalry")

        color, label = rivalry_label(percentile_r)
        st.markdown(
            f"""
        <div style="text-align:center;margin-top:10px;margin-bottom:10px;">
            <span style="font-size:20px;">{color} <b>{label}</b></span><br>
            <span style="font-size:15px;color:gray;">{percentile_r:.1f}ᵗʰ percentile among {len(all_int)} rivalries</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # ------------------------------
        # 🤝 Partnership Section
        # ------------------------------
        st.markdown(f"### 🤝 Partnership: {a_disp} & {b_disp}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Matches Together", pair_chem.get("matches", 0))
        with c2:
            live_win_pct = (pair_chem.get("wins", 0) / pair_chem["matches"] * 100) if pair_chem.get("matches") else 0
            st.metric("Wins Together", pair_chem.get("wins", 0), f"{live_win_pct:.1f}%")
        with c3:
            chem_val = pair_chem.get("chemistry", None)
            chem_display = "—" if chem_val is None else f"{chem_val:.2f}"
            st.metric("Chemistry Score", chem_display)


        losses = pair_chem.get("matches", 0) - pair_chem.get("wins", 0)
        draws_p = 0
        st.markdown(f"Draws: {draws_p}  |  Losses: {losses}")

        # --- Partnership Badge ---
        all_chem = []
        if chem_df_local is not None and not chem_df_local.empty and "chemistry" in chem_df_local.columns:
            all_chem = chem_df_local["chemistry"].dropna().tolist()

        rank_d = sum(s < pair_chem.get("chemistry", 0.0) for s in all_chem)
        percentile_d = (rank_d / len(all_chem) * 100) if all_chem else 0.0

        def partnership_label(p):
            if p >= 90:
                return ("🟩", "Elite Partnership")
            elif p >= 70:
                return ("🟦", "Strong Partnership")
            elif p >= 40:
                return ("🟨", "Developing Partnership")
            elif p >= 10:
                return ("🟧", "Needs Work")
            else:
                return ("🟥", "Poor Connection")

        color, label = partnership_label(percentile_d)
        st.markdown(
            f"""
        <div style="text-align:center;margin-top:10px;">
            <span style="font-size:20px;">{color} <b>{label}</b></span><br>
            <span style="font-size:15px;color:gray;">{percentile_d:.1f}ᵗʰ percentile among {len(all_chem)} duos</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.divider()

    finally:
        conn.close()


# ----------------------------
# 📊 PAGE COMPOSER
# ----------------------------
def render_charts_page():
    # Match Dashboard header styling / spacing
    st.set_page_config(page_title="Charts & Stats | Love Five-A-Side", layout="wide")
    page_header(
        "Charts & Stats",
        "Explore season trends, player insights and duo chemistry",
        center=True,
        divider=True,
    )

    league_id = get_current_league_id()
    section = st.radio(
        "View",
        [
            "Summary",
            "Global Overview",
            "Player Insights",
            "Head-to-Head & Duo Chemistry",
            "Teammate History",
            "Matchup History",
        ],
        horizontal=True,
        key="charts_active_section",
    )

    st.divider()

    if section == "Summary":
        render_charts_fast_summary(league_id)
        return

    matches_all = load_charts_processed_matches_cached(league_id)

    if matches_all.empty:
        st.info("No processed matches yet.")
        return

    season_mode, selected_year, season_start, matches_filtered = get_season_filter_ui(matches_all, suffix="top")

    st.divider()

    if section == "Global Overview":
        render_global_overview("_exp", season_mode, selected_year, season_start, matches_filtered)
        return

    if section == "Player Insights":
        render_player_insights("_exp", season_mode, selected_year, season_start, matches_filtered)
        return

    if section == "Head-to-Head & Duo Chemistry":
        render_head_to_head_section(season_mode, selected_year, season_start, matches_filtered)
        return
    # ------------------------------
    # 📚 Teammate History (Top-level dropdown)
    # ------------------------------
    if section == "Teammate History":
        name_map = get_name_map_cached(get_current_league_id())

        players_df = sql_df(
            "SELECT name FROM public.players WHERE league_id=%s ORDER BY name",
            (league_id,),
        )
        all_players = sorted(players_df["name"].dropna().astype(str).tolist())

        render_team_history_directory(
            matches_filtered,
            all_players,
            name_map,
            key_prefix="thd_main",
        )
        return


    # ------------------------------
    # 🆚 Matchup History (Team A vs Team B)
    # ------------------------------
    if section == "Matchup History":
        name_map = get_name_map_cached(get_current_league_id())

        players_df = sql_df(
            "SELECT name FROM public.players WHERE league_id=%s ORDER BY name",
            (league_id,),
        )
        all_players = sorted(players_df["name"].dropna().astype(str).tolist())

        render_group_vs_group(
            matches_filtered,
            all_players,
            name_map,
            key_prefix="gvg_main",
        )
        return
