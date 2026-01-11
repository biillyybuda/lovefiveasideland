import os
import psycopg2

def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "db.pxtiginazcpwyquvquji.supabase.co"),
        dbname=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        port=int(os.getenv("PGPORT", "5432")),
        sslmode="require",
    )
