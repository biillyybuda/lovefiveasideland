# Matchday Hub AI Team Generator Review

Date: 2026-05-27

## Executive Summary

The Matchday Hub AI generator is already much smarter than a simple MMR splitter. It checks every legal team split, keeps MMR as the main anchor, then nudges the ranking with form, team shape, chemistry, trio synergy, awkward pairings, similar historical games, predicted score margin and betting-style probabilities.

The biggest opportunity now is not adding more signals. It is making the signals more trustworthy, less duplicated, and clearer about confidence. With the current live dataset, the model has enough match history to be useful, but not enough to treat every duo, trio, role or similar matchup as hard truth.

## Live Data Shape

Current Supabase match history, sampled at review time:

- Processed matches: 62
- Date range: 2025-01-02 to 2026-05-22
- Average goal difference: 3.58
- One-goal games: 21
- Games within 2 goals: 30
- Heavy wins by 5+ goals: 18
- Average total goals: 20.03
- Players seen in processed matches: 28
- Players with 10+ games: 17
- Average games per player: 22.14
- Minimum games for a seen player: 1
- Maximum games for a seen player: 59
- Exact repeat 5v5 matchups: 0
- Strong historical similarity examples between prior games: 106 at 7+ overlapping players
- Very strong similarity examples: 30 at 8+ overlapping players
- Match stats rows: 220
- Highlight moments rows: 1010

What this means: the app has a useful league memory, but it rarely has exact repeat fixtures. The AI should lean on "similar enough" games, not pretend it has lots of perfect head-to-head evidence.

## How The Generator Works Today

The Matchday Hub page sends the AI generator through this flow:

1. The user selects 10 players, or 8 plus captains.
2. The app generates every legal split.
3. Mirrored duplicates are removed.
4. Each split is scored by `evaluate_teams_v2`.
5. `rank_generated_matchups` converts each split into a clearer 0-100 recommendation score.
6. The full explorer stores every possible split.
7. The main screen shows a diverse shortlist of five smart options.
8. Each shortlist option gets plain-English explanation, risks, suggested tweaks and WhatsApp text.
9. The chosen matchup opens the Matchday Card with odds, insights, past matchups and player cards.

This is the right general structure. The exhaustive generation step is a major strength because it means the app is choosing from all possible balanced games, not guessing randomly.

## Main Strengths

- MMR is still the anchor, which is correct.
- Form is shrunk for players with low sample sizes, so one or two results do not overrule rating.
- Bad pairings and trio synergy are capped, which avoids one old pattern taking over.
- Similar past games are weighted by overlap and side balance, which is much better than exact-match-only logic.
- The UI shows reasons and risks, not just a black-box score.
- The explorer lets you sanity-check every possible split.
- The shortlist is deliberately varied, so you do not just see five tiny versions of the same team.
- Betting markets now blend scoreline, MMR, form, chemistry, trio links, bad pairings and historical scorelines.

## Holes And Risks

### 1. Chemistry is still too close to "good players winning together"

The chemistry system rewards duos for games, wins and close results. That is useful, but it can confuse genuine chemistry with two strong players simply being on good teams.

Better approach: calculate residual chemistry. In plain English: "Did this pair do better than expected after accounting for MMR and team strength?" That would make chemistry much more realistic.

### 2. Trio synergy can be noisy

The current trio model is sample-shrunk and capped, which is good. But with only 62 processed matches and many player combinations, most trios will still be based on thin evidence.

The UI should show confidence or coverage for trio claims. For example: "strong trio signal from 10 games" versus "weak trio signal from 3 games".

### 3. Similar past games are useful, but not exact head-to-heads

There are no exact repeat matchups in the live data. The historical system is therefore working from partial overlap. That is fine, but the wording should stay careful.

Best wording: "Closest previous games" or "Similar past games", not "head-to-head record" unless the overlap is very high.

### 4. The model has several prediction sources

At the moment there are several overlapping prediction layers:

- `evaluate_teams_v2`
- `matchup_recommendation_score`
- `build_true_fairness_calibration`
- `betting_markets`
- `blend_market_probabilities`
- `preview_insights`
- relationship chemistry/rivalry helpers

They mostly point in the same direction, but they are not one shared source of truth. Over time this can create small contradictions, such as the team card liking a game while the odds imply a stronger favourite.

Long-term fix: make one central "match prediction" object and let the page, odds, explanations and WhatsApp text all read from it.

### 5. The top five are not strictly the best five

The first option is intended to be best overall, but the shortlist is built for diversity. That is good product design, because captains often want different flavours of game. But it should be labelled clearly as "Smart shortlist" rather than implying all five are strict ranks.

Suggested UI wording:

- "Best overall"
- "Closest expected game"
- "Most even ratings"
- "Highest chemistry"
- "Strong alternative"

### 6. Fitness can quietly move teams

Fitness affects effective MMR. That is useful, but if fitness is stale, the model may overrate or underrate someone without the user noticing.

Improvement: show a small note when fitness adjustments are affecting a matchup, and eventually track when fitness was last updated.

### 7. Role balance is only partly covered

The historic style layer uses old stats/highlights to infer finishing, creation, saves, impact and clutch moments. That helps, but it is not the same as explicit roles.

If keeper/defender/runner/finisher balance matters, add optional player roles and let the generator avoid teams with no defensive/keeper profile.

### 8. Spread can miss "two top-heavy teams"

The model compares spread difference between teams. That catches one team being much more top-heavy than the other. But if both teams are equally top-heavy, the spread difference can look fine while the game still feels awkward.

Improvement: track absolute spread as well as spread difference, and flag "both teams rely on one big carry" style games.

### 9. Explanations are good, but confidence should be clearer

The explanations are football-friendly, but they do not always tell the user how much evidence sits behind a claim.

Add confidence chips:

- History: low/medium/high
- Chemistry coverage: low/medium/high
- Trio coverage: low/medium/high
- Style data coverage: low/medium/high

### 10. Name-based matching will limit worldwide scaling

The AI currently normalises names and compares strings. That is okay for one league, but subscriptions and multiple leagues should eventually use player IDs throughout the model.

This matters if two players share a name, a player changes display name, or imported data has spelling differences.

## Fixes Applied During Review

Two small reliability fixes were applied:

1. Added `clear_engine_cache()` to `utils/team_ai_engine.py`.

   The app was already trying to clear the AI engine cache after results/player changes, but that function did not exist. The error was swallowed silently, so the engine could keep stale MMR/chemistry/history data until restart.

2. Made `calculate_chemistry_for_all_duos()` copy its input DataFrame before converting team columns.

   This prevents the chemistry helper from mutating match data that other charts, previews or AI functions may still be using during the same Streamlit run.

## Best Next Improvements

### Priority 1: Add model confidence to the UI

For each generated option, show a compact confidence line:

`Evidence: history medium | chemistry high | trio low | style medium`

This would make the AI feel more honest and more professional.

### Priority 2: Upgrade chemistry to expected-vs-actual chemistry

Current chemistry says: "Did this pair win together?"

Better chemistry says: "Did this pair beat expectation together?"

This would make the generator much fairer when strong players often play together.

### Priority 3: Unify prediction, odds and explanation

Build one central prediction function that returns:

- recommended score
- predicted margin
- tight-game chance
- Team A/B win/draw probabilities
- key reasons
- key risks
- confidence labels

Then the Matchday Card, betting markets, shortlist and WhatsApp text all tell the same story.

### Priority 4: Add optional football roles

Let admins tag players with simple roles:

- Keeper
- Defender
- Runner
- Creator
- Finisher
- Physical

Then the generator can avoid obviously unrealistic teams even when MMR looks balanced.

### Priority 5: Add a post-match learning view

After a result is entered, show:

- predicted margin vs actual margin
- predicted tight-game chance vs actual outcome
- whether the chosen split was one of the top options
- which signals were right or wrong

This is how the model gets better without guessing.

## Suggested Product Direction

The Matchday Hub should become less like "the app picked teams" and more like:

"Here are the best games, why they work, what could go wrong, and how confident the app is."

That is the right feeling for captains. It keeps the human in charge while making the AI genuinely useful.
