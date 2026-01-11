import os, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import psycopg2

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
LEAGUE_ID = int(os.getenv("LEAGUE_ID", "1"))

# -----------------------------
# Postgres / Supabase connection
# -----------------------------
# Set these in CMD before running Streamlit:
#   set PGHOST=db.pxtiginazcpwyquvquji.supabase.co
#   set PGDATABASE=postgres
#   set PGUSER=postgres
#   set PGPASSWORD=your_password
#   set PGPORT=5432
#
# (We use separate fields so % in password is fine.)
def get_conn():
    # 1️⃣ Preferred: single connection string (Render / Supabase)
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url, sslmode="require")

    # 2️⃣ Fallback: discrete PG* vars (local / legacy)
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
# Data loaders (used widely)
# -----------------------------
def load_players_df():
    conn = get_conn()
    df = pd.read_sql(
        'SELECT * FROM public.players WHERE league_id = %s ORDER BY name',
        conn,
        params=(LEAGUE_ID,),
    )
    conn.close()
    return df

def load_matches_df():
    conn = get_conn()
    df = pd.read_sql(
        'SELECT * FROM public.matches WHERE league_id = %s ORDER BY date',
        conn,
        params=(LEAGUE_ID,),
    )
    conn.close()
    return df

# -----------------------------
# Backups (Supabase handles backups)
# Keep function so callers don't break.
# -----------------------------
def backup_db_manual():
    # No local .db file anymore. Leave as a compatibility stub.
    return None
