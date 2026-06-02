# Love Five Web

Next.js frontend for the public Love Five website and demo experience.

This is intended to run alongside the current Streamlit app while the web product is rebuilt gradually.

## Environment

Create `.env.local`:

```bash
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
NEXT_PUBLIC_DEMO_JOIN_CODE=DEMO2026
```

## Run

```bash
npm install
npm run dev
```

First milestone: read-only demo pages backed by the existing Supabase demo league.
