# Love Five Go-Live Runbook

## Current Launch Recommendation

Launch `lovefive-web` as the public website and read-only demo first. Keep the
existing Streamlit app online for real sign-in, league management, result entry
and player admin until those workflows are rebuilt in Next.js.

## Deploy Target

Deploy the Next.js app from:

```text
lovefive-web
```

Recommended hosting:

- Vercel: simplest fit for Next.js.
- Render: also fine if you prefer to keep hosting in one place.

## Required Production Environment Variables

Add these in the hosting provider dashboard:

```text
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
NEXT_PUBLIC_DEMO_JOIN_CODE=DEMO2026
```

Do not add local-only secrets such as database passwords to the public Next.js
site unless a server-only API route needs them. Any variable starting with
`NEXT_PUBLIC_` is visible to browsers.

## Vercel Settings

- Framework preset: `Next.js`
- Root directory: `lovefive-web`
- Install command: `npm install`
- Build command: `npm run build`
- Output directory: leave as Vercel default

Suggested domain split:

- `www.lovefive.co.uk` -> new Next.js public site
- `app.lovefive.co.uk` -> old Streamlit app while migration continues

## Render Settings

- Service type: Web Service
- Root directory: `lovefive-web`
- Runtime: Node
- Build command: `npm install && npm run build`
- Start command: `npm run start`
- Environment: add the production variables listed above

## Pre-Launch Checks

- Production build passes with `npm run build`.
- Demo league loads using the production Supabase anon key.
- `/login` clearly routes people back to the current app until web auth exists.
- Old Streamlit app still handles real accounts and admin workflows.
- Supabase database password has been rotated if any old tracked launcher or
  local file ever contained it.
- Supabase RLS hardening is completed before public signups, paid leagues or
  live multi-league admin features.

## Not Ready For Full Retirement Yet

The old app should not be retired until these are live in Next.js:

- User authentication.
- Logged-in league dashboard.
- Join/invite flow.
- Add/edit/delete result workflow.
- Process/reprocess matches.
- Player management.
- League admin role checks.
- Database-backed RLS policies for all league-owned tables.

## Rollback Plan

Keep the current Streamlit app and domain available during the soft launch. If
the new site has an issue, point the public domain back to the old app or remove
the new domain assignment while the fix is deployed.
