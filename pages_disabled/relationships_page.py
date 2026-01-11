import streamlit as st
import pandas as pd
from utils.db_utils import load_players_df, load_matches_df, get_conn, backup_db_manual, STARTING_MMR
from utils.calc_utils import expected_score, process_unprocessed_matches, compute_streaks_from_matches
from utils.export_utils import df_to_png, fig_to_png_bytes


def render_relationships_page():
    st.markdown("<h2>Player Relationships & Rivalries</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Filter by players to view personalized data.</div>", unsafe_allow_html=True)

    players = load_players_df()['name'].tolist()
    selected_players = st.multiselect("Filter by Player(s)", players, default=[])

    matches = load_matches_df()
    from collections import defaultdict
    duo_counts = defaultdict(int)
    head2head = {}
    goal_diffs = defaultdict(list)

    for _, m in matches.iterrows():
        ta = [p.strip() for p in str(m.get('team_a', '')).split(',') if p.strip()]
        tb = [p.strip() for p in str(m.get('team_b', '')).split(',') if p.strip()]
        score = m.get('score', '') or ''
        a_sc = b_sc = 0
        try:
            if isinstance(score, str) and '-' in score:
                a_sc, b_sc = [int(x.strip()) for x in score.split('-', 1)]
        except:
            a_sc = b_sc = 0

        for team in (ta, tb):
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    key = tuple(sorted((team[i], team[j])))
                    duo_counts[key] += 1

        for a in ta:
            for b in tb:
                key = tuple(sorted((a, b)))
                if key not in head2head:
                    head2head[key] = {'matches': 0, 'wins_a': 0, 'wins_b': 0}
                head2head[key]['matches'] += 1
                if a_sc > b_sc:
                    head2head[key]['wins_a'] += 1
                elif b_sc > a_sc:
                    head2head[key]['wins_b'] += 1
                goal_diffs[key].append(abs(a_sc - b_sc))

    duo_rows = []
    for (a, b), cnt in duo_counts.items():
        if selected_players and not (a in selected_players or b in selected_players):
            continue
        duo_rows.append({'Player A': a, 'Player B': b, 'Matches Together': cnt})
    df_duos = pd.DataFrame(duo_rows).sort_values('Matches Together', ascending=False)

    pair_wins = {}
    for (a, b), cnt in duo_counts.items():
        pair_wins[(a, b)] = {'matches': 0, 'wins': 0}
    for _, m in matches.iterrows():
        ta = [p.strip() for p in str(m.get('team_a', '')).split(',') if p.strip()]
        tb = [p.strip() for p in str(m.get('team_b', '')).split(',') if p.strip()]
        score = m.get('score', '') or ''
        a_sc = b_sc = 0
        try:
            if isinstance(score, str) and '-' in score:
                a_sc, b_sc = [int(x.strip()) for x in score.split('-', 1)]
        except:
            a_sc = b_sc = 0
        for (a, b) in list(pair_wins.keys()):
            if a in ta and b in ta:
                pair_wins[(a, b)]['matches'] += 1
                if a_sc > b_sc:
                    pair_wins[(a, b)]['wins'] += 1
            if a in tb and b in tb:
                pair_wins[(a, b)]['matches'] += 1
                if b_sc > a_sc:
                    pair_wins[(a, b)]['wins'] += 1

    teammates_rows = []
    for (a, b), vals in pair_wins.items():
        if selected_players and not (a in selected_players or b in selected_players):
            continue
        mct = vals['matches']
        wins = vals['wins']
        winpct = round((wins / mct * 100), 1) if mct > 0 else 0.0
        teammates_rows.append({'Player A': a, 'Player B': b, 'Matches Together': mct, 'Win % Together': winpct})
    df_teammates = pd.DataFrame(teammates_rows).sort_values('Win % Together', ascending=False).reset_index(drop=True)

    st.subheader("Teammate Pairs (filtered by minimum matches)")
    min_games = st.slider('Minimum matches together', min_value=1, max_value=20, value=5)
    if df_teammates.empty:
        st.info("No teammate pairs to show.")
    else:
        df_filtered = df_teammates[df_teammates['Matches Together'] >= min_games].sort_values(
            'Win % Together', ascending=False).reset_index(drop=True)
        if df_filtered.empty:
            st.info(f"No pairs with at least {min_games} matches together.")
        else:
            df_filtered['Rank'] = df_filtered.index + 1
            st.dataframe(df_filtered)
            st.download_button(
                "Export Teammates (PNG)",
                data=df_to_png(df_filtered, title=f"Teammates (min {min_games} games)"),
                file_name="teammates.png",
                mime="image/png"
            )

    riv_rows = []
    for (a, b), stats in head2head.items():
        if selected_players and not (a in selected_players or b in selected_players):
            continue
        matches_played = stats['matches']
        wins_a = stats.get('wins_a', 0)
        wins_b = stats.get('wins_b', 0)
        win_pct_a = wins_a / matches_played if matches_played > 0 else 0
        win_pct_b = wins_b / matches_played if matches_played > 0 else 0
        win_pct_diff = abs(win_pct_a - win_pct_b)
        avg_gd = 0
        gk = (a, b) if (a, b) in goal_diffs else ((b, a) if (b, a) in goal_diffs else None)
        if gk:
            vals = goal_diffs.get(gk, [])
            avg_gd = sum(vals) / len(vals) if vals else 0
        rif = matches_played * (1 - win_pct_diff) * (1 - min(avg_gd, 5) / 5) if matches_played > 0 else 0
        riv_rows.append({
            'Player A': a, 'Player B': b, 'Matches': matches_played,
            'Wins A': wins_a, 'Wins B': wins_b, 'Avg Goal Diff': round(avg_gd, 2),
            'Intensity': round(rif, 3)
        })

    df_rivals = pd.DataFrame(riv_rows)
    if df_rivals.empty:
        st.info("No rivalries to show.")
    else:
        df_rivals = df_rivals.sort_values('Intensity', ascending=False).reset_index(drop=True)
        df_rivals['Rank'] = df_rivals.index + 1
        df_rivals['Medal'] = df_rivals['Rank'].apply(lambda i: '🥇' if i == 1 else ('🥈' if i == 2 else ('🥉' if i == 3 else '')))
        st.subheader("Rivalries (by Intensity)")

        import plotly.express as px
        st.dataframe(df_rivals[['Medal', 'Rank', 'Player A', 'Player B', 'Matches', 'Wins A', 'Wins B', 'Avg Goal Diff', 'Intensity']])
        st.download_button(
            "Export Rivalries (PNG)",
            data=df_to_png(df_rivals.head(200), title="Rivalries"),
            file_name="rivalries.png",
            mime="image/png"
        )

    st.subheader("Chemistry & Intensity — What Do They Mean?")
    info_text = """Chemistry: how well two teammates perform together.
- More games together increases confidence.
- Higher win % together boosts chemistry.
- Closer games (smaller goal gaps) improve chemistry.

Intensity: strength of a rivalry between two players.
- More head-to-head matches increase intensity.
- More balanced wins (closer to 50/50) increase intensity.
- Closer games (smaller goal gaps) increase intensity."""

    if st.button("Export Formula Info Card (PNG)"):
        from io import BytesIO
        from PIL import Image, ImageDraw, ImageFont
        import textwrap

        W, H = 1100, 620
        img = Image.new("RGBA", (W, H), (11, 13, 16, 255))
        draw = ImageDraw.Draw(img)
        try:
            font_title = ImageFont.truetype("arial.ttf", 30)
            font_body = ImageFont.truetype("arial.ttf", 20)
        except:
            font_title = ImageFont.load_default()
            font_body = ImageFont.load_default()
        y = 50
        for line in textwrap.wrap(info_text, width=85):
            draw.text((40, y), line, font=font_body, fill=(230, 238, 246))
            y += 28
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Formula Info Card", data=buf, file_name="chemistry_intensity_info.png", mime="image/png")
