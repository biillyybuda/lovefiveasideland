@echo off
cd /d "C:\Users\Bills PC\Documents\5 a side stats\MMRApp\LoveFiveASideApp_2026"

REM ---- Supabase / Postgres env vars ----
set PGHOST=db.pxtiginazcpwyquvquji.supabase.co
set PGDATABASE=postgres
set PGUSER=postgres
set PGPASSWORD=nb%%d-F69a@U9v5/
set PGPORT=5432
set LEAGUE_ID=1

echo Using venv312 Python:
venv312\Scripts\python.exe --version

echo Launching Streamlit from venv312...
venv312\Scripts\python.exe -m streamlit run app.py --server.address localhost --server.port 8501
start "" http://localhost:8501
pause
