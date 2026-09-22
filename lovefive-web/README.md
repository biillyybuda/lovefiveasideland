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

## Database setup for saving results

**11 September 2026:** the foundation migration below is now installed in the
shared LoveFive project (`pxtiginazcpwyquvquji`) as version `20260908080902`.
The API schema cache was refreshed and the installed save functions were
verified. Existing results and ratings were preserved. Do not reapply the SQL
to that project; the following instructions remain relevant for other databases.

The result-management page now uses `lib/result-rebuild.ts` to save a match and
its recalculated ratings in one database transaction. Running Next.js locally
does **not** create or migrate the Supabase database. The project is selected by
`NEXT_PUBLIC_SUPABASE_URL` in `.env.local` (restart `npm run dev` after changing it).

Before saving, the connected project needs the existing migration
[`20260908080902_mmr_v2_foundation.sql`](../supabase/migrations/20260908080902_mmr_v2_foundation.sql).
It creates `rating_runs`, `rating_adjustments`, `balance_model_states`,
`team_selection_snapshots`, `rating_workspace` and `commit_rating_run`, plus
`players.initial_mmr`, `matches.selection_source`, permissions and write guards.
`rating_runs` is the save-request/audit table; `mmr_history` is per-player rating
history and cannot replace it. Creating only an empty `rating_runs` table is
insufficient.

1. Check that `.env.local` points to the intended Supabase project. On
   11 September 2026 the local app pointed to the shared **LoveFive** project
   (`pxtiginazcpwyquvquji`); all of the migration's new objects were absent there.
2. For that shared project, take a fresh database backup and coordinate the
   updated website release using the
   [existing rollout instructions](../docs/mmr-v2-implementation-2026-09-08.md#database-storage-and-transition).
   The migration itself does not replay results. After a league's first
   successful new result save/rebuild, its older web and Streamlit result
   writers are blocked. A separate development Supabase project avoids that
   shared cutover, but must first contain the existing base schema and data.
3. In the intended project's **SQL Editor**, run the **complete existing
   migration** once, inside `BEGIN;` / `COMMIT;`, or apply it through your
   established migration workflow. This migration is not safe to reapply:
   check the schema first if any of its objects already exist.
4. After it commits, run:

   ```sql
   NOTIFY pgrst, 'reload schema';
   ```

   This [refreshes the Supabase API schema cache](https://supabase.com/docs/guides/troubleshooting/refresh-postgrest-schema).
   It cannot create a missing table or replace the migration.
5. Verify the objects, then refresh the website and retry as an active league
   owner/admin:

   ```sql
   select to_regclass('public.rating_runs') as rating_runs,
          to_regclass('public.rating_adjustments') as rating_adjustments,
          to_regclass('public.balance_model_states') as balance_model_states,
          to_regclass('public.team_selection_snapshots') as team_selection_snapshots,
          to_regprocedure('public.rating_workspace(bigint)') as rating_workspace,
          to_regprocedure('public.commit_rating_run(bigint,text,uuid,jsonb,jsonb,text)') as commit_rating_run;
   ```

All six values should be non-null. If SQL sees the objects but the API still
reports them missing, recheck the URL, reload the cache and check Data API
exposure and the migration's authenticated grants/RLS. Do not disable RLS or
put a service-role key in `NEXT_PUBLIC_` variables.

### Match review and optional notes

The shared LoveFive database also has
`supabase/migrations/20260911073619_match_review_notes.sql` installed (11 September
2026). No further SQL is needed for the current local website. For another
database, apply that complete migration once after the foundation migration,
using the existing CLI migration workflow or the Supabase SQL Editor inside a
transaction. It adds `matches.review_note` and `save_match_review_note`, and
extends the atomic result writer to save notes with new or corrected results.
It reloads the API schema cache automatically.

Notes are optional, limited to 1,500 characters and displayed as plain text under
“From the match”. Only active league owners/admins can edit them. A note-only
edit does not replay ratings; concurrent edits are checked before saving. Older
web clients that omit the field preserve an existing note when correcting a
result. The migration does not alter existing results or publish new ratings.

Match Review uses the match's year and only completed results up to its date and
ID. It calls the same award functions and uses the same active award roster as
Season Review. Archived players remain available for historical lineups,
appearance counts, relationships and the Matchday reconstruction. Award roster
membership is the established current eligibility rule; historical archive
status is not available in this schema. First recorded appearances are labelled
as such rather than assuming a real-world debut from a truncated archive.

League-provided groups live in `lib/league-groups.ts`, scoped by league ID and
stable player IDs. LoveFive's HLHS group contains Glee, Tom D, Kelso, Gav and
Sam K. **Four or more on the same team** triggers the headline and group panel.
Three do not. Reunion records cover all qualifying games in that season, while
the current four's record (including appearances with the fifth member) and the
all-five record are shown separately. A guest is named in the team label, such
as “HLHS + Jos”. Headlines respond to the actual result and prior record.

`npm test` covers report chronology, aliases, archived lineups, milestones,
partnerships, group thresholds, note handling, shared-view content and established
award rules. An optional `LOVEFIVE_REPORT_SNAPSHOT` environment variable pointing
to a private league export also runs the match-129 regression; no private match
history is committed as a fixture.

### Adding watch readings to a past match

Open a completed game from match history and choose **Add / edit player activity**
at the top of its review. An active league admin can select a recorded participant,
including an archived player, and enter distance (km), playing time (minutes),
match average/peak heart rate (bpm), and an optional watch/app name. Missing readings
remain blank; at least one numeric reading is required. Entries can be corrected
or removed from the same panel. Saved readings are visible to active league
members, with a count of how many participants have activity recorded.

`supabase/migrations/20260911084608_match_player_activity.sql` was applied to the
shared LoveFive database on 11 September 2026. No extra setup is needed there.
Other databases need that migration applied once, after the existing migrations.
It creates `match_player_activity`, its access policies and the version-checked
`save_match_player_activity` function. Storage is independent of results, public
MMR, team balancing, awards and player style profiles. Deleting a match removes
its activity entries. Records cannot be attached to a player from another league
or someone outside the recorded lineup. Anonymous/demo visitors and screenshot
share views do not expose these readings.

Values are entered manually from any device. This release does not import watch
accounts, infer missing figures or rank players by heart rate.

Optional **resting heart rate** and **personal maximum heart rate** can also be
recorded in the Heart-rate baselines section. Use values known around the selected
match date and leave unknown values blank. These are per-match snapshots, with no
automatic copying from newer games or changes to other games. The personal maximum
is a separate baseline from the peak observed during the match. Either baseline can
be saved on its own; providing all readings is not required. Baseline figures are
collected only: no VO2 max estimate, fitness score or matchmaking rule is enabled.

Apply `supabase/migrations/20260911093328_match_activity_heart_rate_baselines.sql`
after the activity migration above on other databases. It was applied to the
shared LoveFive database on 11 September 2026; no extra setup is needed there.
It adds two nullable columns and updates the
existing save function while preserving its access checks and version handling.
Older clients that omit these new fields preserve saved baselines; explicit nulls
clear them. Existing readings and ratings are preserved.

### Health & Activity in Charts & Stats

Signed-in league members have a **Health & Activity** tab with distance per
recorded game, total distance, distance per minute, top recorded speed and recording-count leaderboards.
Each ranking shows its own number of qualifying games, a minimum-games filter,
and shared ranks for values equal at the displayed precision. Fewer than three
games is labelled an early sample, not a confidence or health assessment.
The coverage panel links to recent match activity; admins enter or edit readings
on the existing match page.

**Player Insights → Health & Activity** shows distance and time totals, trends
for all eighteen stored readings plus distance per minute, heart-rate summaries and
the latest entered baselines in the selected season. It includes archived players.
Selecting a leaderboard name opens that player's health view. The Season View
filter applies to all these figures, and the match-by-match table retains missing
readings and their sources. Chart gaps represent missing readings; a single
reading is still plotted. Historical baselines are not carried across seasons.

Distance per minute is total metres divided by total minutes **only for matches
with both values**. Average distance uses only recorded distance values. Average
match HR is the unweighted mean of recorded match averages, with a count shown.
Blank readings remain unknown; recorded zero distance is valid. Roles, time in
goal and differences between devices can affect comparisons. There is no HR
leaderboard, VO2 max formula, fitness ranking or change to matchmaking/awards.

Activity is loaded independently with league-scoped pagination and the existing
member RLS policies. Read failures show an explicit retry state, not zero activity.
The public demo and share views do not load or show health readings. These charts use
the activity migrations above and the watch-stat migration below.
`/app/charts?view=health&season=2026` opens the league view;
`/app/charts?view=player&section=health&player=PLAYER_ID&season=2026` opens a player.

### Additional watch readings (Garmin and other devices)

In a previous match, choose **Player activity → Add/Edit activity → More watch readings**.
Optional fields cover average/moving/best pace, average/moving/maximum speed,
moving and elapsed time, total calories, and average/minimum/maximum breathing rate.
Use the existing total/playing time field for Garmin's Total Time. Duration fields
accept decimal minutes, `m:ss` or `h:mm:ss`; pace accepts `m:ss` per kilometre
(or numeric seconds per kilometre). Seconds are preserved. Pace and speed remain
independent watch-reported values, because each can be rounded differently.
Do not derive missing distance from the screenshot's rounded pace or speed.

The player's health view has a latest-watch-detail panel, a trend option for every
reading, and expandable watch details for each historical game. Top recorded speed
uses the maximum of recorded match peaks, with a count and minimum-games filter;
it is not a sprint test. A shared watch can cover a different player each week,
so missing readings never become zero and totals are not assumed to cover everyone.
Calories and respiration are personal recording context, with no league ranking.
For shared watches, the wearer's profile can affect calorie estimates.

Apply `supabase/migrations/20260911095758_match_activity_watch_stats.sql` after the
baseline migration above. It was applied to the shared LoveFive database on
11 September 2026. It adds twelve nullable fields and extends the existing save
function, keeping league access, participant validation, version checks and
retry handling. Older clients that omit these fields preserve saved readings;
explicit null clears an individual field. Existing rows remain unchanged apart
from the new null columns. No matchmaking or fitness-score rules are introduced.

## MMR Immunity Shield

Implemented locally on 22 September 2026; deployment and the database migration
are still pending. Choose League or MMR Immunity Shield when recording a result.
League alone changes public MMR and official season awards. Both competitions
train team balancing. A Shield win awards one honour to each winning player;
a draw stays a draw and awards nobody a Shield.

The shared League / MMR Immunity Shield / All competitions filter scopes Home,
Stats Centre and Season Review. Goals, assists, form, relationships and activity
use the selected matches. Honours derive from results, so edits and deletions
correct them automatically. Match reports, copied lineups and exported lineup
images identify the fixture competition.

Release this website together with
`supabase/migrations/20260922074013_mmr_immunity_shield.sql`. It adds the competition
field and `commit_competition_rating_run`, preserves existing games as League,
and blocks outdated writers. New website saves require that function; there is
no legacy write fallback. Reading a database without the new column remains
supported for rollout. No public rating recalculation occurs on migration.

Run `npm test` and `npm run build` before release. Competition tests include a
local PostgreSQL integration test using PGlite; they never write to Supabase.
See `../docs/mmr-immunity-shield-plan-2026-09-22.md` for verified rules and the
coordinated release procedure.
