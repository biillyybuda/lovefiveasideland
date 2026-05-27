# Love Five

Streamlit app for running a 5-a-side football league: player ratings, match history, charts, season review, league membership, and AI team generation.

## Local Setup

1. Create or activate the Python environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set the required environment variables. Use `.env.example` as the checklist.
4. Run the app:

```powershell
streamlit run app.py --server.address localhost --server.port 8501
```

Or use `Run MMR App 2026.bat` after your environment variables are already set.

## Required Secrets

- `DATABASE_URL`, or the individual `PGHOST`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`, `PGPORT` values
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `COOKIE_SECRET`

Do not commit real secrets. Local secrets can live in `.streamlit/secrets.toml`, local shell environment variables, or Render environment variables.

## Deployment Notes

Render should provide the same environment variables listed above. After rotating the Supabase database password, update Render before redeploying.

For local Windows runs, Supabase's direct database port `5432` may resolve to IPv6 only and time out. Use the pooler/dedicated pooler port `6543` locally if `5432` hangs.

For Render, prefer the Supabase pooler connection string if direct connections are slow or unreliable. Keep `PGCONNECT_TIMEOUT=5` so connection failures return quickly instead of freezing the app.

The app expects Supabase tables for leagues, league members, invites, profiles, players, matches, and MMR/history data.

## Security Note

An earlier local launcher stored a database password in a tracked file. Rotate that Supabase database password, then update Render and local environment variables with the new value.

## Useful Files

- `app.py` - Streamlit entry point and page router
- `pages_disabled/` - app pages loaded manually by the router
- `utils/db_utils.py` - Postgres connection pooling and cached loaders
- `utils/auth_utils.py` - Supabase login and cookie session handling
- `utils/league_utils.py` - league selection, join codes, and invites
- `utils/team_ai_engine.py` - team generation engine
- `sql/speed_indexes.sql` - optional Supabase performance indexes
