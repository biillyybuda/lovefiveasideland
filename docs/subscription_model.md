# Love Five Subscription Model

## Recommended Launch Plans

| Plan | Price | Best for | Limits |
| --- | --- | --- | --- |
| Free | £0 | Trying Love Five with one casual group | 1 league, 15 players, 1 admin |
| Pro League | £7.99/month or £79/year | A regular weekly five-a-side league | 1 league, 60 players, 3 admins |
| Club | £29/month or £290/year | Organisers running multiple leagues or divisions | Unlimited leagues, players and admins |

## Feature Split

Free should stay useful enough that people trust the app:
- Core ratings
- Match history
- Basic dashboard
- Invite code

Pro League is the main paid product:
- AI team generator
- Matchday Hub
- Charts and Season Review
- WhatsApp share tools
- Player profiles
- Exports and backups

Club is for organisers who outgrow one weekly group:
- Multiple leagues
- Multiple admins
- Club-wide player database
- Divisions and groups
- Priority support

## Implementation Notes

Subscriptions are league-owned, not player-owned. A league admin or owner pays
for a league, while normal players can join without payment friction.

Before payments go live:
- Confirm every table and query is scoped by `league_id`.
- Confirm role checks are enforced in both UI and page logic.
- Add Supabase RLS so the database itself blocks cross-league access.
- Connect Stripe only after the league/security audit is clean.
