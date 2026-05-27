import os, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import streamlit as st

# -----------------------------
# Config (keep exactly like before)
# -----------------------------
BASE_DIR = Path(".")
CONFIG_PATH = BASE_DIR / "config.json"

if CONFIG_PATH.exists():
    cfg = json.loads(CONFIG_PATH.read_text())
else:
    cfg = {"starting_mmr": 1000, "k_factor": 30, "draw_value": 0.5}

K_DEFAULT = cfg.get("k_factor", 30)
DRAW_VALUE = cfg.get("draw_value", 0.5)
STARTING_MMR = cfg.get("starting_mmr", 1000)

# League scoping (your current league)
def get_current_league_id() -> int:
    league_id = st.session_state.get("league_id")
    if not league_id:
        raise RuntimeError("No league selected in session_state")
    return int(league_id)

# -----------------------------
# Postgres / Supabase connection
# -----------------------------
# This app opens DB connections from lots of pages. On Render/Supabase, creating a
# new SSL connection for every rerun is slow, so get_conn() now returns a pooled
# connection wrapper. Existing code can still call conn.close(); close() returns
# the connection to the pool instead of really closing it.

def _connect_raw():
    """Create a real psycopg2 connection. Used only as a fallback/debug path."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url, sslmode="require")

    host = os.getenv("PGHOST", "").strip()
    password = os.getenv("PGPASSWORD", "")
    if not host or not password:
        raise RuntimeError(
            "Postgres env vars not set. Set DATABASE_URL (recommended) "
            "or PGHOST/PGDATABASE/PGUSER/PGPASSWORD/PGPORT."
        )

    return psycopg2.connect(
        host=host,
        dbname=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "postgres"),
        password=password,
        port=int(os.getenv("PGPORT", "5432")),
        sslmode="require",
    )


# -----------------------------
# Connection pool (faster on cloud)
# -----------------------------
@st.cache_resource
def _get_pool():
    """Create a small Postgres connection pool per Streamlit process."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return SimpleConnectionPool(1, 8, dsn=db_url, sslmode="require")

    host = os.getenv("PGHOST", "").strip()
    password = os.getenv("PGPASSWORD", "")
    if not host or not password:
        raise RuntimeError(
            "Postgres env vars not set. Set DATABASE_URL (recommended) "
            "or PGHOST/PGDATABASE/PGUSER/PGPASSWORD/PGPORT."
        )

    return SimpleConnectionPool(
        1,
        8,
        host=host,
        dbname=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "postgres"),
        password=password,
        port=int(os.getenv("PGPORT", "5432")),
        sslmode="require",
    )


class PooledConnection:
    """Small DB-API wrapper that makes existing conn.close() code pool-safe."""

    def __init__(self, pool, conn):
        self._pool = pool
        self._conn = conn
        self._returned = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
        self.close()
        return False

    def close(self):
        """Return the underlying connection to the pool once."""
        if self._returned:
            return
        self._returned = True
        try:
            if getattr(self._conn, "closed", 1) == 0:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                self._pool.putconn(self._conn)
        except Exception:
            try:
                self._pool.putconn(self._conn, close=True)
            except Exception:
                pass


def get_conn():
    """Return a pooled Postgres connection. Call .close() when finished."""
    if str(os.getenv("LOVEFIVE_DISABLE_DB_POOL", "")).lower() in ("1", "true", "yes"):
        return _connect_raw()

    pool = _get_pool()
    return PooledConnection(pool, pool.getconn())


from contextlib import contextmanager

@contextmanager
def pooled_conn():
    conn = get_conn()
    try:
        yield conn
    finally:
        conn.close()

# -----------------------------
# Generic cached query helper
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def query_df_cached(query: str, params: tuple = ()):
    with pooled_conn() as conn:
        return pd.read_sql(query, conn, params=params)


# -----------------------------
# Data loaders (used widely)
# -----------------------------
# Important performance/correctness note:
# Streamlit cache_data is process-wide, so zero-argument cached loaders can leak
# data between leagues. Keep public functions zero-arg for compatibility, but
# put the league_id into the cached function key.

@st.cache_data(ttl=300, show_spinner=False)
def _load_players_df_cached(league_id: int):
    with pooled_conn() as conn:
        return pd.read_sql(
            """
            SELECT *
            FROM public.players
            WHERE league_id = %s
            ORDER BY name
            """,
            conn,
            params=(int(league_id),),
        )


def load_players_df():
    return _load_players_df_cached(get_current_league_id())


@st.cache_data(ttl=300, show_spinner=False)
def _load_matches_df_cached(league_id: int):
    with pooled_conn() as conn:
        return pd.read_sql(
            """
            SELECT *
            FROM public.matches
            WHERE league_id = %s
            ORDER BY date
            """,
            conn,
            params=(int(league_id),),
        )


def load_matches_df():
    return _load_matches_df_cached(get_current_league_id())


@st.cache_data(ttl=300, show_spinner=False)
def _load_active_players_light_cached(league_id: int):
    with pooled_conn() as conn:
        return pd.read_sql(
            """
            SELECT id, name, display_name, mmr, matches_played, wins,
                   win_streak, lose_streak, fitness, strengths, is_active
            FROM public.players
            WHERE league_id = %s
              AND COALESCE(is_active, 1) = 1
            ORDER BY name
            """,
            conn,
            params=(int(league_id),),
        )


def load_active_players_light_df():
    """Fast player loader for UI pickers/cards that do not need every column."""
    return _load_active_players_light_cached(get_current_league_id())


# -----------------------------
# Backups (Supabase handles backups)
# Keep function so callers don't break.
# -----------------------------
def backup_db_manual():
    # No local .db file anymore. Leave as a compatibility stub.
    return None

# -----------------------------
# Backwards-compatible SQL helpers
# -----------------------------
# Some existing pages import sql_df directly from utils.db_utils.
# Keep this API stable while still using the pooled/cached query path above.

def sql_df(query: str, params: tuple = ()):
    """Compatibility wrapper used by older pages such as charts_page.py."""
    if params is None:
        params = ()
    if not isinstance(params, tuple):
        try:
            params = tuple(params)
        except Exception:
            params = (params,)
    return query_df_cached(query, params)


def clear_db_caches():
    """Clear Streamlit data caches after writes/imports."""
    try:
        st.cache_data.clear()
    except Exception:
        pass
