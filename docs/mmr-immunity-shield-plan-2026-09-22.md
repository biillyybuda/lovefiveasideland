# MMR Immunity Shield implementation

Status (22 September 2026): released from `main`. The production database migration is installed and the updated website is live on Render. All 97 tests and the production build passed before release.

## Agreed rules

- Each Shield fixture is a standalone final. Every player on the winning side earns one Shield honour.
- A draw stays a draw. Nobody wins a Shield. There is no penalty or tiebreak input.
- League fixtures alone determine public MMR, league counters, records and official season awards.
- Shield fixtures contribute to Shield records and All competitions totals.
- Both competitions train the separate team-balancing model, including partnership evidence. Browsing filters never restrict its training history.
- Competition is chosen explicitly. A rearranged league game remains a league game regardless of weekday.
- Existing games default to League. No historical games are reclassified by date or weekday.

## One set of pages

The shared `Competition` selector offers League, MMR Immunity Shield and All competitions. League is the default. Home, Stats Centre and Season Review use shared match scoping instead of duplicate cup pages. Old demo statistics routes redirect to the existing Stats Centre with their requested view.

The selected matches drive appearances, wins/draws/losses, form, goals and assists, player styles, relationships and activity. Missing individual goals/assists stay unknown rather than becoming zero. Shield honours are derived from saved winning lineups, including archived players, so retries, edits and deletions cannot leave a separate trophy counter out of sync.

Shield-only summaries show honours and football records instead of league rating charts. Combined views label retained ratings as League MMR. Official Season Review awards and their comparison periods remain league-only. Direct match reports use the saved fixture's competition for records, stories, honours and navigation. Shared Shield reports state that MMR is protected.

Result entry and editing expose a competition choice. Changing a saved match's competition rebuilds the appropriate records atomically. The fixture choice is also carried through Matchday, editable lineups, copied text and exported lineup images.

## Calculation and storage

- `lovefive-web/lib/competition.ts`: classification, scope, Shield winner and honours.
- `lovefive-web/lib/match-loader.ts`: shared paginated reads; only a missing competition column permits legacy read fallback during rollout.
- `lovefive-web/lib/result-rebuild.ts`: validates every result, replays only League into public MMR, and replays both competitions into balancing.
- `lovefive-web/lib/season-awards.ts`: guards award eligibility and seeding against Shield input.
- `supabase/migrations/20260922074013_mmr_immunity_shield.sql`: adds constrained `matches.competition` with default `league` and the atomic `commit_competition_rating_run` function.

The new save function requires the competition policy version, ten valid participant history entries for each rated league match, no Shield rating history, and balancing records for all completed matches. It preserves existing notes, league permissions, concurrency checks and request retry handling. The old `commit_rating_run` rejects stale website saves after migration. Existing row-level access policies remain in place.

The migration itself neither recalculates ratings nor changes scores or lineups. The existing notes, activity and rating foundation migrations are not edited or reapplied. The remote function definitions were inspected read-only to check compatibility.

## Verification

`npm test` runs the existing league award golden checks, rating/model tests and new competition tests. The new database test uses an isolated PGlite PostgreSQL database, applies the actual rating foundation, notes and Shield migrations, and exercises the real functions, triggers, roles and policies.

Coverage includes winning/drawn Shield fixtures, add/edit/delete, backdating, full rebuilds, reclassification, duplicate retries, conflicting request IDs, stale saves, invalid rating history, cross-league access, members and anonymous callers. Public league ratings, counters and intermediate histories stay equal to the league-only baseline after Shield changes. Existing award scores and qualification remain identical across all 13 golden views.

Browser checks use fictional players and results: competition switching, player records, goals, winning and drawn reports, share output and mobile layout. No production result is created for testing. The temporary preview route is removed before delivery.

## Production release verification

- Published the application changes and the previously untracked migration history to `main` in commit `165775d`.
- Saved a local application-data and function-definition backup before the database update.
- Applied only `20260922074013_mmr_immunity_shield.sql`; the remote migration history is now up to date.
- Compared all 17 application tables with the backup: existing data and access policies are unchanged. All 122 existing fixtures default to League, and all 1,220 rating-history entries are preserved.
- Verified that the competition save function exists, permits authenticated callers and rejects anonymous callers. The security-advisor findings are unchanged from before the release.
- Verified the live site's League default and switched to MMR Immunity Shield, which correctly shows no fixtures or honours yet.
- Refresh existing admin pages before saving results so they load the new competition-aware writer.

## Release procedure

1. Prepare the tested website build and take the normal database backup. Confirm the current migration history still matches the checked foundation and notes functions.
2. Apply only `20260922074013_mmr_immunity_shield.sql` through the normal migration workflow and deploy the matching website in the same release window. Result saving is unavailable between the two steps: the new website requires the new function, and the migration rejects old writers.
3. Refresh existing admin sessions before entering a Shield fixture. Do not allow the first Shield result while old readers are still being served; old cached pages do not understand competition scoping.
4. Verify League as the default, empty Shield views, a real Shield result when available, protected public MMR and updated balancing records. Use an isolated staging league for any test fixture.

Do not restore the old writer while retaining Shield data. If rollback is necessary, stop result entry and use the coordinated application/database recovery procedure with the pre-release backup.
