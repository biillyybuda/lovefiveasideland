import os
from pathlib import Path
import psycopg2

def _load_local_env():
    if os.getenv("RENDER"):
        return

    env_path = Path(".env.local")
    if not env_path.exists():
        return

    seen_keys = set()
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = raw_line.split("=", 1)
        key = key.strip()
        seen_keys.add(key)
        os.environ[key] = value

    if "DATABASE_URL" not in seen_keys:
        os.environ.pop("DATABASE_URL", None)


_load_local_env()

def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "db.pxtiginazcpwyquvquji.supabase.co"),
        dbname=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        port=int(os.getenv("PGPORT", "5432")),
        sslmode="require",
        connect_timeout=int(os.getenv("PGCONNECT_TIMEOUT", "5")),
    )
