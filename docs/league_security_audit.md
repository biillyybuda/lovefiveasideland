# Love Five League/Security Audit

Date: 2026-05-27
Supabase project: `LoveFive` / `pxtiginazcpwyquvquji`

## Executive Summary

The app is mostly moving in the right direction for multi-league use: core
tables have `league_id`, most active queries now filter by the selected league,
and the admin-only result/player-management pages are gated.

It is not subscription-ready yet. The biggest remaining risk is the live
Supabase security layer: several league-owned tables have RLS enabled but no
policies, `anon`/`authenticated` still have broad table grants, and the app
server connects directly to Postgres, so server-side role checks remain
essential.

## Live Database Findings

### Tables

All main league-owned tables have `league_id`:

- `players`
- `matches`
- `mmr_history`
- `match_player_stats`
- `match_stats`
- `highlight_moments`
- `fantasy_points`
- `seasons`
- `league_members`
- `league_invites`

`profiles` is user-owned by `id`, not league-owned.

No null `league_id` values were found in the checked league-owned tables.

### RLS Policies

RLS is enabled, but policies are incomplete.

Existing policies found:

- `leagues`: members can select their leagues.
- `league_members`: users can select their own membership row.
- `players`: league members can select; league admins can write.
- `league_invites`: league admins can manage.
- `profiles`: users can select their own profile.

Missing policies:

- `matches`
- `mmr_history`
- `match_player_stats`
- `match_stats`
- `highlight_moments`
- `fantasy_points`
- `seasons`

Supabase security advisors also flagged these as `RLS Enabled No Policy`.

### Grants / API Exposure

Supabase advisors flagged public and signed-in GraphQL exposure for app tables.
The direct privilege check showed `anon` and `authenticated` have broad table
privileges on several public tables.

This does not automatically mean rows are readable when RLS blocks them, but it
is not the right posture before subscriptions. `anon` should not have broad
app-table privileges, and signed-in access should rely on tight RLS policies.

### Function Exposure

`public.handle_new_user()` is a `SECURITY DEFINER` function executable by
`anon` and `authenticated`. Supabase advisors flagged this. It should either
have execute revoked from public roles or be moved/hardened depending on how it
is used by auth triggers.

### Role Model Mismatch

The Python app checks for `owner` and `admin`, but the live
`league_members.role` constraint only allowed `admin` and `member`. This should
be aligned before subscriptions.

## App Code Findings

### Safe / improved areas

- Dashboard, Charts, Matchday Hub, team generation and Season Review are using
  selected league context for active flows.
- Add Result is now visible only to admins/owners and the page itself blocks
  non-admins.
- Player Management is now visible only to admins/owners and the page itself
  blocks non-admins.
- Season Review no longer uses a hardcoded league.
- Match processing and season reset are scoped by `league_id`.

### Fixes applied in this audit

- Non-admins no longer see the invite code on Join / Invite.
- `calc_utils.get_mmr()` no longer falls back to a global player lookup if no
  current league is available.
- `calc_utils.calibrate_winprob_scale()` no longer calibrates against all
  leagues if no league is available.
- `team_ai_engine` optional table readers no longer fall back to unscoped global
  table reads.
- `team_ai_engine` historical fairness loading no longer falls back to all
  matches/players/MMR history.
- Dormant `performance_page` MMR history lookup now includes `league_id`.

### Remaining code risks

- The Streamlit server uses a direct Postgres connection. If that connection is
  `postgres`, `service_role`, or table owner-like, RLS may not protect those
  server-side queries. Sensitive server-side actions must keep explicit app
  role checks.
- Some functions still rely on session role checks rather than re-checking
  league membership in the database immediately before writes.
- `Join / Invite` still lets any logged-in user join another league by code,
  which is intended, but invite-code entropy and rotation should be reviewed.
- `SELECT *` is still used in some scoped loaders. This is mainly performance,
  not cross-league leakage, but should be narrowed over time.

## Files Added

- `sql/rls_hardening.sql`: draft RLS/grants hardening SQL.

## Must Fix Before Subscriptions

1. Apply/test RLS policies for every league-owned table on a Supabase branch.
2. Revoke broad `anon` privileges from app tables.
3. Decide whether browser/API clients should use Supabase Data API at all. If
   not, disable or restrict it.
4. Harden or revoke direct execution of `public.handle_new_user()`.
5. Align DB role values with app roles: `owner`, `admin`, `member`.
6. Remove `league_id DEFAULT 1` from league-owned tables after confirming all
   insert paths explicitly set `league_id`.
7. Add database-backed role verification for destructive server actions before
   allowing paid public signup.
8. Add tests/checks for: League A member cannot read League B rows; League A
   admin cannot edit League B rows; member cannot perform admin writes.

## Recommended Next Step

Create a Supabase development branch, apply `sql/rls_hardening.sql`, then run
the app against that branch and test:

- normal login
- league selection
- member dashboard
- admin adding a result
- admin processing matches
- player management
- invite/join flow
- charts/season review

Only after that should Stripe/payment work begin.
