# app_v4_liveupdate_stable2.py
# Final stable build: single-connection processing with retry logic, WAL mode, charts fix,
# dropdowns for Matches & This Week, Swap Teams, caching preserved, auto-refresh, attendance %.
import streamlit as st
import pandas as pd
import sqlite3, shutil, json, os, itertools, math, time
from pathlib import Path
from datetime import datetime
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False
import plotly.express as px
import plotly.graph_objects as go

# ---------- Config & Paths ----------
BASE_DIR = Path('.')
DB_PATH = BASE_DIR / 'mmr_league.db'  # user's DB file (unchanged)
BACKUP_DIR = BASE_DIR / 'backups'
BACKUP_DIR.mkdir(exist_ok=True)
CONFIG_PATH = BASE_DIR / 'config.json'
if CONFIG_PATH.exists():
    cfg = json.loads(CONFIG_PATH.read_text())
else:
    cfg = {"starting_mmr":1000, "k_factor":30, "draw_value":0.5}
K_DEFAULT = cfg.get("k_factor", 30)
DRAW_VALUE = cfg.get("draw_value", 0.5)
STARTING_MMR = cfg.get("starting_mmr", 1000)

st.set_page_config(page_title="5-a-side MMR App v4 (stable2)", layout="wide", initial_sidebar_state="expanded")

# ---------- Dark theme CSS ----------
STYLES = """
<style>
:root{
  --bg:#0b0d10;
  --card:#0f1114;
  --muted:#8b8f95;
  --accent-blue:#2E86AB;
  --accent-red:#D64545;
  --pl-purple:#37003C;
}
html, body, [data-testid="stAppViewContainer"]{background:var(--bg); color:#e6eef6;}
.stCard {background: var(--card); border-radius:12px; padding:14px; box-shadow: 0 4px 18px rgba(0,0,0,0.6);}
.small-muted{color:var(--muted); font-size:0.9em;}
.badge {display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; font-size:0.9em;}
.badge.blue{background: linear-gradient(90deg,var(--accent-blue), #6fb3d2); color:white;}
.badge.red{background: linear-gradient(90deg,var(--accent-red), #f28b8b); color:white;}
.leader-card{background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.04)); padding:12px; border-radius:10px;}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ---------- DB helpers ----------
def _enable_wal():
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH, timeout=5)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

_enable_wal()

def get_conn():
    # fresh connection each time; timeout increased so retries wait a little
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    return conn

def ensure_db():
    if not DB_PATH.exists():
        conn = get_conn(); cur = conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            mmr REAL,
            matches_played INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0,
            win_streak INTEGER DEFAULT 0,
            last_match_date TEXT
        );
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, team_a TEXT, team_b TEXT, score TEXT, result TEXT,
            team_a_avg REAL, team_b_avg REAL, processed INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS mmr_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER, match_id INTEGER, date TEXT, mmr_before REAL, mmr_after REAL
        );
        """)
        conn.commit(); conn.close()
        _enable_wal()

def backup_db():
    if DB_PATH.exists():
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        dest = BACKUP_DIR / f'mmr_league_backup_{ts}.db'
        shutil.copy2(DB_PATH, dest)
        return dest
    return None

# ---------- Cached loaders ----------
@st.cache_data
def load_players_df():
    ensure_db()
    conn = get_conn(); df = pd.read_sql('SELECT * FROM players ORDER BY name COLLATE NOCASE', conn); conn.close()
    return df

@st.cache_data
def load_matches_df():
    ensure_db()
    conn = get_conn(); df = pd.read_sql('SELECT * FROM matches ORDER BY date DESC', conn); conn.close()
    return df

# ---------- Utils ----------
def clear_and_rerun_with_message(delay=0.35):
    st.info("Refreshing data 🔄 Please wait...", icon="🔄")
    time.sleep(delay)
    st.cache_data.clear()
    st.rerun()

def compute_attendance(processed_only=True):
    conn = get_conn()
    players = pd.read_sql('SELECT id, name FROM players', conn)
    if processed_only:
        matches = pd.read_sql('SELECT id, team_a, team_b FROM matches WHERE processed=1', conn)
    else:
        matches = pd.read_sql('SELECT id, team_a, team_b FROM matches', conn)
    conn.close()
    total_matches = len(matches)
    rows = []
    for _, p in players.iterrows():
        name = p['name']
        count = 0
        for _, m in matches.iterrows():
            ta = set([x.strip() for x in str(m['team_a']).split(',') if x.strip()])
            tb = set([x.strip() for x in str(m['team_b']).split(',') if x.strip()])
            if name in ta or name in tb:
                count += 1
        pct = (count / total_matches * 100) if total_matches > 0 else 0.0
        rows.append({'name': name, 'matches': count, 'attendance_pct': round(pct,1)})
    return pd.DataFrame(rows).sort_values('attendance_pct', ascending=False).reset_index(drop=True), total_matches

# ---------- Core MMR math ----------
def team_average_mmr(team_players):
    if not team_players:
        return STARTING_MMR
    conn = get_conn(); cur = conn.cursor()
    if isinstance(team_players, str):
        names = [n.strip() for n in team_players.split(',') if n.strip()]
    else:
        names = [n.strip() for n in team_players if str(n).strip()]
    vals = []
    for nm in names:
        cur.execute('SELECT mmr FROM players WHERE name = ?', (nm,))
        r = cur.fetchone()
        vals.append(r[0] if r else STARTING_MMR)
    conn.close()
    return sum(vals)/len(vals) if vals else STARTING_MMR

def expected_score(mmr_a, mmr_b):
    return 1 / (1 + 10 ** ((mmr_b - mmr_a) / 400.0))

def goal_diff_scale(gd):
    if gd >= 5: return 2.0
    if gd == 4: return 1.75
    if gd == 3: return 1.5
    if gd == 2: return 1.25
    if gd == 1: return 1.0
    return 0.5

# ---------- Team balancing ----------
def generate_balanced_teams(list_of_10):
    best = None
    for comb in itertools.combinations(list_of_10, 5):
        teamA = list(comb)
        teamB = [p for p in list_of_10 if p not in teamA]
        a = team_average_mmr(teamA)
        b = team_average_mmr(teamB)
        diff = abs(a - b)
        if best is None or diff < best[0]:
            best = (diff, teamA, teamB, a, b)
    _, teamA, teamB, aavg, bavg = best
    return teamA, teamB, aavg, bavg

# ---------- Sidebar & Navigation ----------
st.sidebar.markdown(f"<div class='small-muted'>🏟️ <strong>5-a-side MMR League</strong></div>", unsafe_allow_html=True)
page = st.sidebar.selectbox(
    "Page",
    ["Dashboard","Team Generator","Matches","Players","Player Management","Performance","Charts","Relationships","This Week"],
)

ensure_db()
backup_db()

# ---------- Helper: safe_execute with retries ----------
def safe_execute(cur, query, params=(), retries=15, delay=0.2):
    for i in range(retries):
        try:
            cur.execute(query, params)
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "database is locked" in str(e).lower():
                time.sleep(delay)
            else:
                raise
    # last attempt (will raise if fails)
    cur.execute(query, params)

# ---------- Pages ----------
if page == 'Dashboard':
    st.markdown("<h1>Dashboard ⚽</h1>", unsafe_allow_html=True)
    dfp = load_players_df()
    dfm = load_matches_df()
    attendance_df, total_matches = compute_attendance(processed_only=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Players', len(dfp), delta=None)
    c2.metric('Processed Matches', total_matches, delta=None)
    c3.metric('Average MMR', round(dfp['mmr'].mean(),1) if not dfp.empty else 0)
    c4.metric('Avg Attendance %', f"{attendance_df['attendance_pct'].mean():.1f}%" if not attendance_df.empty else "0%")
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader('Top players 🏆 (with attendance)')
    merged = dfp.merge(attendance_df[['name','attendance_pct']], left_on='name', right_on='name', how='left')
    st.dataframe(merged.sort_values('mmr', ascending=False)[['name','mmr','matches_played','attendance_pct']].head(15))
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="leader-card"><h3>Top Attendance 🏅</h3></div>', unsafe_allow_html=True)
    if not attendance_df.empty:
        st.dataframe(attendance_df.head(10))
    else:
        st.info("No attendance data yet. Add & process matches to populate attendance.")

elif page == 'Player Management':
    st.markdown('<h2>Player Management 🧍‍♂️</h2>', unsafe_allow_html=True)
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.write('Add, edit or delete players. Click **Save Data** after edits in the table.')
    df = load_players_df()
    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True, resizable=True)
        gb.configure_column('id', editable=False)
        gb.configure_selection(selection_mode='multiple', use_checkbox=True)
        grid_options = gb.build()
        grid_response = AgGrid(df, gridOptions=grid_options, allow_unsafe_jscode=True,
                               data_return_mode=DataReturnMode.FILTERED_AND_SORTED, update_mode=GridUpdateMode.MODEL_CHANGED,
                               height=350)
        updated_df = pd.DataFrame(grid_response['data'])
        selected = grid_response.get('selected_rows', [])
    else:
        st.warning('streamlit-aggrid not installed — table editing will be basic.')
        updated_df = df.copy(); selected = []
        st.dataframe(df)
    st.markdown('### Add single player')
    new_name = st.text_input('Name', key='new_name')
    new_mmr = st.number_input('Starting MMR', value=STARTING_MMR, step=1, key='new_mmr')
    if st.button('Add Player'):
        if new_name.strip() == '':
            st.warning('Enter a name')
        else:
            backup_db()
            conn = get_conn(); cur = conn.cursor()
            try:
                cur.execute('INSERT INTO players (name, mmr) VALUES (?,?)', (new_name.strip(), float(new_mmr)))
                conn.commit()
            except Exception as e:
                st.error(f"Error adding player: {e}")
            finally:
                conn.close()
            clear_and_rerun_with_message()
    st.markdown('### Bulk import (one name per line)')
    bulk = st.text_area('Paste names here', height=120)
    if st.button('Bulk Add Players'):
        lines = [l.strip() for l in bulk.splitlines() if l.strip()]
        if not lines:
            st.warning('Paste one or more player names')
        else:
            backup_db(); added = 0
            conn = get_conn(); cur = conn.cursor()
            for nm in lines:
                try:
                    cur.execute('INSERT OR IGNORE INTO players (name, mmr) VALUES (?,?)', (nm, STARTING_MMR))
                    if cur.rowcount > 0: added += 1
                except:
                    pass
            conn.commit(); conn.close()
            st.success(f'Added {added} players (duplicates ignored)')
            clear_and_rerun_with_message()
    if selected:
        st.write(f'Selected {len(selected)} row(s) ready for deletion.')
        if st.button('Delete Selected'):
            ids = [int(r['id']) for r in selected if r.get('id') not in (None,"")]
            if ids:
                backup_db()
                conn = get_conn(); cur = conn.cursor()
                for did in ids:
                    safe_execute(cur, 'DELETE FROM players WHERE id=?', (did,))
                conn.commit(); conn.close()
                st.success(f'Deleted {len(ids)} players')
                clear_and_rerun_with_message()
    if st.button('Save Data (apply edits)'):
        try:
            if 'id' in updated_df.columns and 'name' in updated_df.columns and 'mmr' in updated_df.columns:
                backup_db(); conn = get_conn(); cur = conn.cursor()
                for _, row in updated_df.iterrows():
                    rid = row.get('id', None)
                    name = str(row.get('name','')).strip()
                    mmr = float(row.get('mmr') if row.get('mmr') not in (None,"") else STARTING_MMR)
                    if rid and not pd.isna(rid):
                        safe_execute(cur, 'UPDATE players SET name=?, mmr=? WHERE id=?', (name, mmr, int(rid)))
                    else:
                        safe_execute(cur, 'INSERT OR IGNORE INTO players (name, mmr) VALUES (?,?)', (name, mmr))
                conn.commit(); conn.close()
                st.success('Player edits saved')
                clear_and_rerun_with_message()
            else:
                st.error("Table doesn't contain expected columns (id, name, mmr).")
        except Exception as e:
            st.error(f'Error saving players: {e}')
    st.markdown('</div>', unsafe_allow_html=True)

elif page == 'Team Generator':
    st.markdown('<h2>Team Generator 🤝</h2>', unsafe_allow_html=True)
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    names = load_players_df()['name'].tolist()
    if len(names) < 10:
        st.warning('Not enough players to pick 10. Add players in Player Management.')
    cols = st.columns(2); picks = []
    for i in range(10):
        col = cols[i%2]; picks.append(col.selectbox(f'Player {i+1}', names, key=f'p{i}', index=0))
    if st.button('Generate Balanced Teams'):
        teamA, teamB, aavg, bavg = generate_balanced_teams(picks)
        st.markdown(f"<div class='badge blue'>Team A</div>  <div class='badge red'>Team B</div>", unsafe_allow_html=True)
        st.write('Team A:', ', '.join(teamA), 'Avg MMR:', round(aavg,1))
        st.write('Team B:', ', '.join(teamB), 'Avg MMR:', round(bavg,1))
        probA = expected_score(aavg, bavg)
        st.write(f'Predicted win probability Team A: {probA:.2%}')
        if st.button('Create match (save draft)'):
            backup_db()
            conn = get_conn(); cur = conn.cursor()
            date = datetime.today().isoformat()
            safe_execute(cur, 'INSERT INTO matches (date, team_a, team_b, score, result, team_a_avg, team_b_avg, processed) VALUES (?,?,?,?,?,?,?,?)',
                         (date, ', '.join(teamA), ', '.join(teamB), '', '', 0, aavg, bavg))
            conn.commit(); conn.close()
            clear_and_rerun_with_message()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Matches page (dropdowns + swap) ----------------
elif page == 'Matches':
    st.markdown('<h2>Matches ⚽</h2>', unsafe_allow_html=True)
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    dfm = load_matches_df()
    st.dataframe(dfm)

    st.markdown('### Add Match (use dropdowns to avoid typos)')
    players = load_players_df()['name'].tolist()
    col1, col2 = st.columns([2,1])
    with col1:
        date = st.date_input('Date', datetime.today())
    with col2:
        st.write('')  # spacer

    if 'matches_team_a' not in st.session_state:
        st.session_state['matches_team_a'] = []
    if 'matches_team_b' not in st.session_state:
        st.session_state['matches_team_b'] = []

    ta_col, tb_col = st.columns(2)
    with ta_col:
        team_a = st.multiselect('Team A players (select exactly 5)', options=players, key='matches_team_a')
        st.markdown(f"<div class='small-muted'>Selected: {len(st.session_state.get('matches_team_a',[]))} — recommended: 5</div>", unsafe_allow_html=True)
    with tb_col:
        team_b = st.multiselect('Team B players (select exactly 5)', options=players, key='matches_team_b')
        st.markdown(f"<div class='small-muted'>Selected: {len(st.session_state.get('matches_team_b',[]))} — recommended: 5</div>", unsafe_allow_html=True)

    sw_col1, sw_col2, sw_col3 = st.columns([1,1,1])
    with sw_col1:
        if st.button('Swap Teams 🔄'):
            a = st.session_state.get('matches_team_a', []).copy()
            b = st.session_state.get('matches_team_b', []).copy()
            st.session_state['matches_team_a'] = b
            st.session_state['matches_team_b'] = a
            clear_and_rerun_with_message()
    with sw_col2:
        if st.button('Clear Selections'):
            st.session_state['matches_team_a'] = []
            st.session_state['matches_team_b'] = []
            st.experimental_rerun()

    score = st.text_input('Score (e.g. 4-2) — optional (leave blank for draft)')
    result = st.selectbox('Result', ['', 'A', 'B', 'Draw'])

    def valid_teams(a,b):
        if len(a) != 5 or len(b) != 5:
            return False, 'Each team must have exactly 5 players.'
        if set(a) & set(b):
            return False, 'A player appears in both teams — remove duplicates.'
        return True, ''

    if st.button('Save Match'):
        ok, msg = valid_teams(st.session_state.get('matches_team_a', []), st.session_state.get('matches_team_b', []))
        if not ok:
            st.error(msg)
        else:
            backup_db()
            conn = get_conn(); cur = conn.cursor()
            avgA = team_average_mmr(st.session_state.get('matches_team_a', [])); avgB = team_average_mmr(st.session_state.get('matches_team_b', []))
            safe_execute(cur, 'INSERT INTO matches (date, team_a, team_b, score, result, team_a_avg, team_b_avg, processed) VALUES (?,?,?,?,?,?,?,0)',
                         (date.isoformat(), ', '.join(st.session_state.get('matches_team_a', [])), ', '.join(st.session_state.get('matches_team_b', [])), score, result, avgA, avgB))
            conn.commit(); conn.close()
            st.success('Match saved (draft). Processing and UI will refresh.')
            clear_and_rerun_with_message()

    st.markdown('### Process unprocessed matches (will update player MMRs)')
    K_val = st.number_input('K-Factor', value=K_DEFAULT)
    if st.button('Process unprocessed matches'):
        # Single connection and cursor used for processing to avoid locks
        conn = get_conn(); cur = conn.cursor()
        cur.execute('SELECT id, date, team_a, team_b, score, result FROM matches WHERE processed = 0')
        rows = cur.fetchall()
        for row in rows:
            mid, date, team_a_s, team_b_s, score_s, result_s = row
            try:
                if isinstance(score_s, str) and '-' in score_s:
                    a_sc,b_sc = [int(x.strip()) for x in score_s.split('-',1)]
                else:
                    a_sc,b_sc = 0,0
            except:
                a_sc,b_sc = 0,0
            gd = abs(a_sc-b_sc)
            sf = goal_diff_scale(gd)
            avgA = team_average_mmr(team_a_s); avgB = team_average_mmr(team_b_s)
            expA = expected_score(avgA, avgB); expB = 1-expA
            if (result_s or '').upper() == 'A':
                valA, valB = 1.0, 0.0
            elif (result_s or '').upper() == 'B':
                valA, valB = 0.0, 1.0
            else:
                valA = valB = DRAW_VALUE

            # process both teams using the single cursor 'cur' with safe_execute
            for team, val, exp in ((team_a_s, valA, expA), (team_b_s, valB, expB)):
                names = [n.strip() for n in team.split(',') if n.strip()]
                for nm in names:
                    cur.execute('SELECT id, mmr, matches_played, wins, losses, draws, win_streak FROM players WHERE name=?', (nm,))
                    r = cur.fetchone()
                    if not r: continue
                    pid, old_mmr, mp, w, l, d, streak = r
                    new_mmr = old_mmr + (K_val * sf * (val - exp))
                    new_mmr = round(new_mmr)
                    mp = (mp or 0) + 1
                    if val == 1: w = (w or 0) + 1; streak = (streak or 0) + 1
                    elif val == 0: l = (l or 0) + 1; streak = 0
                    else: d = (d or 0) + 1; streak = 0
                    safe_execute(cur,
                        'UPDATE players SET mmr=?, matches_played=?, wins=?, losses=?, draws=?, win_streak=?, last_match_date=? WHERE id=?',
                        (new_mmr, mp, w, l, d, streak, date, pid)
                    )
                    safe_execute(cur,
                        'INSERT INTO mmr_history (player_id, match_id, date, mmr_before, mmr_after) VALUES (?,?,?,?,?)',
                        (pid, mid, date, old_mmr, new_mmr)
                    )

            safe_execute(cur, 'UPDATE matches SET processed=1, team_a_avg=?, team_b_avg=? WHERE id=?', (avgA, avgB, mid))

        conn.commit(); conn.close()
        st.success('Processed matches and updated MMRs. Refreshing UI now.')
        clear_and_rerun_with_message()

    st.markdown('</div>', unsafe_allow_html=True)

elif page == 'Players':
    st.markdown('<h2>Players</h2>', unsafe_allow_html=True)
    dfp = load_players_df()
    attendance_df, total_matches = compute_attendance(processed_only=True)
    players_view = dfp.merge(attendance_df[['name','attendance_pct']], left_on='name', right_on='name', how='left').fillna(0)
    st.dataframe(players_view[['name','mmr','matches_played','attendance_pct']].sort_values('attendance_pct', ascending=False))

elif page == 'Performance':
    st.markdown('<h2>Performance (last 5)</h2>', unsafe_allow_html=True)
    conn = get_conn(); dfp = pd.read_sql('SELECT * FROM players', conn); hist = pd.read_sql('SELECT * FROM mmr_history ORDER BY date DESC', conn); conn.close()
    rows = []
    for _, r in dfp.iterrows():
        pid = r['id']; ph = hist[hist['player_id']==pid].head(5)
        if ph.empty: avg = r['mmr']; delta = 0
        else:
            avg = ph['mmr_after'].mean()
            delta = ph['mmr_after'].iloc[0] - ph['mmr_before'].iloc[-1] if len(ph)>1 else ph['mmr_after'].iloc[0] - ph['mmr_before'].iloc[0]
        rows.append((r['name'], avg, delta, r['win_streak']))
    perf = pd.DataFrame(rows, columns=['Player','Avg MMR (Last 5)','MMR Δ','Current Streak'])
    attendance_df, total_matches = compute_attendance(processed_only=True)
    perf = perf.merge(attendance_df[['name','attendance_pct']].rename(columns={'name':'Player'}), on='Player', how='left').fillna(0)
    perf['FormRating'] = (perf['MMR Δ'] * 0.5) + (perf['Current Streak'] * 1.0)
    st.dataframe(perf.sort_values('FormRating', ascending=False))

elif page == 'Charts':
    st.markdown('<h2>Charts</h2>', unsafe_allow_html=True)
    dfp = load_players_df(); names = dfp['name'].tolist()
    sel = st.selectbox('Player', names)
    conn = get_conn()
    hist = pd.read_sql(
        'SELECT mh.*, p.name AS player_name FROM mmr_history mh JOIN players p ON mh.player_id=p.id WHERE p.name = ? ORDER BY date',
        conn, params=(sel,)
    )
    conn.close()
    hist = hist.loc[:, ~hist.columns.duplicated()]
    if hist.empty:
        st.info('No MMR history yet for this player.')
    else:
        hist['date'] = pd.to_datetime(hist['date'])
        fig = px.line(hist, x='date', y='mmr_after', title=f'MMR history - {sel}', markers=True)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Relationships Page ----------------
elif page == 'Relationships':
    st.markdown('<h2>Player Relationships & Rivalries 🤝🔥</h2>', unsafe_allow_html=True)
    MIN_MATCHES = 5
    chemistry_scale = 10.0
    rivalry_scale = 1.0

    @st.cache_data
    def load_matches_players():
        conn = get_conn()
        matches = pd.read_sql('SELECT * FROM matches ORDER BY date', conn)
        players = pd.read_sql('SELECT * FROM players ORDER BY name COLLATE NOCASE', conn)
        conn.close()
        return matches, players

    @st.cache_data
    def compute_relationships(matches_df, min_matches=5):
        from collections import defaultdict
        tm_stats = defaultdict(lambda: {'matches':0,'wins':0,'losses':0,'draws':0,'goal_diff_sum':0})
        rv_stats = defaultdict(lambda: {'matches':0,'wins':0,'losses':0,'draws':0,'goal_diff_sum':0})
        for _, m in matches_df.iterrows():
            ta = [p.strip() for p in str(m.get('team_a','')).split(',') if p.strip()]
            tb = [p.strip() for p in str(m.get('team_b','')).split(',') if p.strip()]
            score = m.get('score','') or ''
            a_score = b_score = 0
            try:
                if isinstance(score, str) and '-' in score:
                    a_score, b_score = [int(x.strip()) for x in score.split('-',1)]
            except:
                a_score, b_score = 0,0
            gd = abs(a_score - b_score)
            if a_score > b_score: resA, resB = 1, 0
            elif a_score < b_score: resA, resB = 0, 1
            else: resA = resB = 0.5
            for team, team_res in ((ta, resA),(tb, resB)):
                for i in range(len(team)):
                    for j in range(i+1, len(team)):
                        p1,p2 = sorted((team[i], team[j]))
                        key = (p1,p2)
                        tm_stats[key]['matches'] += 1
                        tm_stats[key]['goal_diff_sum'] += gd
                        if team_res == 1: tm_stats[key]['wins'] += 1
                        elif team_res == 0: tm_stats[key]['losses'] += 1
                        else: tm_stats[key]['draws'] += 1
            for a in ta:
                for b in tb:
                    key_ab = (a,b)
                    rv_stats[key_ab]['matches'] += 1
                    rv_stats[key_ab]['goal_diff_sum'] += gd
                    if resA == 1: rv_stats[key_ab]['wins'] += 1
                    elif resA == 0: rv_stats[key_ab]['losses'] += 1
                    else: rv_stats[key_ab]['draws'] += 1
        tm_rows = []
        for (p1,p2), d in tm_stats.items():
            mcount = d['matches']
            if mcount < min_matches: continue
            wins = d['wins']; losses = d['losses']; draws = d['draws']
            win_pct = wins / mcount if mcount else 0
            avg_gd = d['goal_diff_sum'] / mcount if mcount else 0
            chemistry = (win_pct * mcount * chemistry_scale)
            tm_rows.append({'player_a':p1,'player_b':p2,'matches':mcount,'wins':wins,'losses':losses,'draws':draws,'win_pct':win_pct,'avg_goal_diff':avg_gd,'chemistry':chemistry})
        rival_comb = {}
        for (a,b), d in rv_stats.items():
            key = tuple(sorted((a,b)))
            if key not in rival_comb: rival_comb[key] = {'pairs':{}, 'matches':0, 'goal_diff_sum':0}
            rival_comb[key]['pairs'][(a,b)] = d
            rival_comb[key]['matches'] += d['matches']
            rival_comb[key]['goal_diff_sum'] += d['goal_diff_sum']
        rv_rows = []
        for (p1,p2), d in rival_comb.items():
            mcount = d['matches']
            if mcount < min_matches: continue
            w1 = rv_stats.get((p1,p2), {}).get('wins',0)
            w2 = rv_stats.get((p2,p1), {}).get('wins',0)
            draws = rv_stats.get((p1,p2), {}).get('draws',0) + rv_stats.get((p2,p1), {}).get('draws',0)
            avg_gd = d['goal_diff_sum'] / mcount if mcount else 0
            win_diff = abs(w1 - w2)
            balance_factor = 1 + (win_diff / mcount)
            intensity = mcount * balance_factor * (1 + avg_gd/2.0) * rivalry_scale
            rv_rows.append({'player_a':p1,'player_b':p2,'matches':mcount,'wins_a_vs_b':w1,'wins_b_vs_a':w2,'draws':draws,'avg_goal_diff':avg_gd,'intensity':intensity})
        teammates_df = pd.DataFrame(tm_rows)
        if not teammates_df.empty:
            teammates_df = teammates_df.sort_values(['chemistry','matches'], ascending=False).reset_index(drop=True)
        else:
            teammates_df = pd.DataFrame(columns=['player_a','player_b','matches','wins','losses','draws','win_pct','avg_goal_diff','chemistry'])
        rivals_df = pd.DataFrame(rv_rows)
        if not rivals_df.empty:
            rivals_df = rivals_df.sort_values(['intensity','matches'], ascending=False).reset_index(drop=True)
        else:
            rivals_df = pd.DataFrame(columns=['player_a','player_b','matches','wins_a_vs_b','wins_b_vs_a','draws','avg_goal_diff','intensity'])
        return teammates_df, rivals_df

    matches_df, players_df = load_matches_players()
    min_matches_ui = st.sidebar.number_input('Min matches for relationships', value=MIN_MATCHES, min_value=1, max_value=50, step=1)
    compute_relationships.clear()
    teammates_df, rivals_df = compute_relationships(matches_df, min_matches=min_matches_ui)

    cols = st.columns([1,1])
    with cols[0]:
        st.markdown('<div class="leader-card"><h3>Top Duos 🤝</h3></div>', unsafe_allow_html=True)
        if not teammates_df.empty:
            df_duos = teammates_df.copy().sort_values('chemistry', ascending=False)
            df_duos['duo'] = df_duos['player_a'] + ' + ' + df_duos['player_b']
            st.dataframe(df_duos[['duo','matches','win_pct','chemistry']].head(20))
            fig = px.bar(df_duos.head(12).sort_values('chemistry'), x='chemistry', y='duo', orientation='h', hover_data=['matches','win_pct'], title='Top Duos by Chemistry', color_discrete_sequence=['#2E86AB'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No duos meet the threshold.')
    with cols[1]:
        st.markdown('<div class="leader-card"><h3>Top Rivalries 🔥</h3></div>', unsafe_allow_html=True)
        if not rivals_df.empty:
            df_r = rivals_df.copy().sort_values('intensity', ascending=False)
            df_r['rivalry'] = df_r['player_a'] + ' v ' + df_r['player_b']
            st.dataframe(df_r[['rivalry','matches','wins_a_vs_b','wins_b_vs_a','intensity']].head(20))
            fig2 = px.bar(df_r.head(12).sort_values('intensity'), x='intensity', y='rivalry', orientation='h', hover_data=['matches','avg_goal_diff'], title='Top Rivalries by Intensity', color_discrete_sequence=['#D64545'])
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info('No rivalries meet the threshold.')

    st.markdown('---')
    st.subheader('Player analysis')
    sel = st.selectbox('Select player', players_df['name'].tolist())
    if sel:
        mask = (teammates_df['player_a']==sel) | (teammates_df['player_b']==sel)
        p_tm = teammates_df[mask].copy()
        if not p_tm.empty:
            p_tm['partner'] = p_tm.apply(lambda r: r['player_b'] if r['player_a']==sel else r['player_a'], axis=1)
            best = p_tm.sort_values(['chemistry','win_pct'], ascending=False).head(1).iloc[0]
            worst = p_tm.sort_values(['chemistry','win_pct'], ascending=True).head(1).iloc[0]
            most_freq = p_tm.sort_values('matches', ascending=False).head(1).iloc[0]
            st.markdown(f"**Best teammate:** {best['partner']} — Chemistry {best['chemistry']:.1f}, Matches {best['matches']}")
            st.markdown(f"**Worst teammate:** {worst['partner']} — Chemistry {worst['chemistry']:.1f}, Matches {worst['matches']}")
            st.markdown(f"**Most frequent teammate:** {most_freq['partner']} — Matches {most_freq['matches']}")
            st.dataframe(p_tm[['partner','matches','win_pct','chemistry']].sort_values('chemistry', ascending=False).reset_index(drop=True))
            figp = px.bar(p_tm.sort_values('chemistry', ascending=False).head(10).assign(partner=lambda df: df['player_a'].where(df['player_a']!=sel, df['player_b'])), x='chemistry', y='partner', orientation='h', title=f'Top Partners for {sel}', hover_data=['matches','win_pct'])
            figp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(figp, use_container_width=True)
        else:
            st.info('No teammate data for this player under the current threshold.')
        maskr = (rivals_df['player_a']==sel) | (rivals_df['player_b']==sel)
        p_rv = rivals_df[maskr].copy()
        if not p_rv.empty:
            p_rv['opponent'] = p_rv.apply(lambda r: r['player_b'] if r['player_a']==sel else r['player_a'], axis=1)
            st.markdown('**Rivalries**')
            st.dataframe(p_rv[['opponent','matches','wins_a_vs_b','wins_b_vs_a','intensity']].reset_index(drop=True))
            figr = px.bar(p_rv.sort_values('intensity', ascending=False).head(10).assign(opponent=lambda df: df['opponent']), x='intensity', y='opponent', orientation='h', title=f'Top Rivals vs {sel}', hover_data=['matches','avg_goal_diff'])
            figr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(figr, use_container_width=True)
        else:
            st.info('No rivalry data for this player under the current threshold.')

    st.markdown('---')
    st.subheader(f'Network map (pairs with ≥ {min_matches_ui} matches)')
    if not teammates_df.empty:
        import math
        pairs = teammates_df.copy()
        nodes = sorted(list(set(pairs['player_a'].tolist() + pairs['player_b'].tolist())))
        node_idx = {n:i for i,n in enumerate(nodes)}
        N = len(nodes)
        node_x = []; node_y = []
        for i,n in enumerate(nodes):
            angle = 2*math.pi*(i)/N
            node_x.append(math.cos(angle)); node_y.append(math.sin(angle))
        edge_x = []; edge_y = []
        for _, r in pairs.iterrows():
            x0 = node_x[node_idx[r['player_a']]]; y0 = node_y[node_idx[r['player_a']]]
            x1 = node_x[node_idx[r['player_b']]]; y1 = node_y[node_idx[r['player_b']]]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='rgba(150,150,150,0.4)'), hoverinfo='none', mode='lines')
        node_trace = go.Scatter(x=node_x, y=node_y, text=nodes, mode='markers+text', textposition='bottom center', marker=dict(size=12, color='#2E86AB'))
        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.info('Not enough teammate relationships to show a network (lower threshold or add more matches).')

# end of app
