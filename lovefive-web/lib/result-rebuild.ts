import type { SupabaseClient } from "@supabase/supabase-js";
import { scoreParts, type Match, type MmrHistory, type Player } from "./demo-data";
import { calculateMatchMmrUpdates, scoreToResult, STARTING_MMR } from "./mmr-engine";
import { balanceRatings, compareMatches, matchDay, replayBalance, resolveLineup, BALANCE_CONFIG, BALANCE_VERSION } from "./balance-model";
import { COMPETITION_POLICY, isLeagueMatch, matchCompetition } from "./competition";

export const DRAFT_MATCH_ID = Number.MAX_SAFE_INTEGER;
export type RatingAdjustment = { player_id: number; delta: number; after_date: string; after_match_id: number };
export type RatingWorkspace = { token: string; players: Player[]; matches: Match[]; history: MmrHistory[]; adjustments: RatingAdjustment[] };
export type LeagueChange = { action: "rebuild" } | { action: "add"; match: Match } |
  { action: "edit"; match: Match; original: Match } | { action: "delete"; match: Match };
export type ResultRebuildSummary = { processedMatches: number; ratedMatches: number; playersUpdated: number; historyRows: number };

function ratingDatabaseError(error: { code?: string; message: string }) {
  const missingObject = ["PGRST202", "PGRST204", "PGRST205", "42P01", "42703", "42883"].includes(error.code || "");
  if (missingObject && /commit_competition_rating_run|competition/.test(error.message)) return new Error("Cup support needs the cup competition database migration. No result has been saved.");
  const ratingObject = /\b(rating_runs|rating_workspace|commit_rating_run|rating_adjustments|balance_model_states|initial_mmr|selection_source)\b/.test(error.message);
  if (missingObject && ratingObject) {
    return new Error("Result saving needs a database update. Apply supabase/migrations/20260908080902_mmr_v2_foundation.sql " +
      "to the Supabase project configured for this website, then refresh and retry. See Database setup in lovefive-web/README.md.");
  }
  return new Error(error.code === "40001" ? "The league changed while saving. Refresh and try again." : error.message);
}

export function normaliseScoreText(value: string | null | undefined) {
  const parsed = scoreParts(value);
  return parsed ? `${parsed[0]}-${parsed[1]}` : "";
}
export function resultLabelFromScore(value: string | null | undefined) {
  const parsed = scoreParts(value);
  if (!parsed) throw new Error("Enter a complete score, such as 10-8.");
  return parsed[0] === parsed[1] ? "Draw" : parsed[0] > parsed[1] ? "A" : "B";
}

export async function loadRatingWorkspace(supabase: SupabaseClient, leagueId: number): Promise<RatingWorkspace> {
  const { data, error } = await supabase.rpc("rating_workspace", { p_league_id: leagueId });
  if (error) throw new Error(`Could not load the rating workspace: ${ratingDatabaseError(error).message}`);
  if (!data?.token || !Array.isArray(data.players) || !Array.isArray(data.matches)) throw new Error("Incomplete rating workspace.");
  return data as RatingWorkspace;
}

export function applyLeagueChange(matches: Match[], change: LeagueChange): Match[] {
  if (change.action === "rebuild") return [...matches];
  if (change.action === "add") return [...matches, { ...change.match, id: DRAFT_MATCH_ID }];
  const old = matches.find(m => m.id === change.match.id);
  const expected = change.action === "edit" ? change.original : change.match;
  if (!old || ["date", "team_a", "team_b", "score", "processed"].some(k => old[k as keyof Match] !== expected[k as keyof Match]) ||
    matchCompetition(old) !== matchCompetition(expected) ||
    (old.review_note || "") !== (expected.review_note || "")) {
    throw new Error("This match has changed since you opened it. Refresh and try again.");
  }
  return change.action === "delete" ? matches.filter(m => m.id !== old.id) : matches.map(m => m.id === old.id ? { ...m, ...change.match } : m);
}

/** Pure replay: no writes until both models and every lineup have been validated. */
export function buildLeagueRatings(workspace: RatingWorkspace, matches = workspace.matches) {
  const players = workspace.players.map(p => ({ ...p, mmr: p.initial_mmr ?? STARTING_MMR,
    matches_played: 0, wins: 0, draws: 0, losses: 0, win_streak: 0, lose_streak: 0 }));
  if (players.some(p => !Number.isFinite(p.mmr))) throw new Error("Invalid starting rating.");
  const byId = new Map(players.map(p => [p.id, p]));
  const adjustments = [...workspace.adjustments].sort((a, b) => a.after_date.localeCompare(b.after_date) || a.after_match_id - b.after_match_id);
  let adjustmentIndex = 0;
  function applyAdjustments(before?: Match) {
    while (adjustmentIndex < adjustments.length) {
      const row = adjustments[adjustmentIndex];
      if (before && compareMatches({ date: row.after_date, id: row.after_match_id }, before) >= 0) break;
      const player = byId.get(row.player_id);
      if (!player || !Number.isFinite(row.delta)) throw new Error("Invalid manual rating adjustment.");
      player.mmr += row.delta;
      adjustmentIndex++;
    }
  }
  const history: Array<{ player_id: number; match_id: number; date: string | null; mmr_before: number; mmr_after: number }> = [];
  const processed = [...matches].filter(m => m.processed === 1).sort(compareMatches);
  for (const match of processed) {
    matchDay(match.date || "");
    if (!scoreParts(match.score)) throw new Error(`Match ${match.id} has an invalid score.`);
    applyAdjustments(match);
    const { a, b } = resolveLineup(match, players);
    // Shield results are validated and train balancing, but never touch public rating history or league counters.
    if (!isLeagueMatch(match)) continue;
    const result = scoreToResult(match);
    const updates = calculateMatchMmrUpdates({ teamA: a.map(id => byId.get(id)!), teamB: b.map(id => byId.get(id)!), match, players });
    for (const update of updates) {
      const p = byId.get(update.player.id)!;
      const win = result === update.team, draw = result === "DRAW", loss = !win && !draw;
      p.mmr = Math.round(update.after);
      p.matches_played++; p.wins += Number(win); p.draws += Number(draw); p.losses += Number(loss);
      p.win_streak = win ? p.win_streak + 1 : 0; p.lose_streak = loss ? p.lose_streak + 1 : 0;
      history.push({ player_id: p.id, match_id: match.id, date: match.date, mmr_before: update.before, mmr_after: p.mmr });
    }
  }
  applyAdjustments();
  const shadow = replayBalance(workspace.players, processed);
  return { competitionPolicy: COMPETITION_POLICY, players, history, shadow: { ...shadow, ratings: balanceRatings(shadow.state), config: BALANCE_CONFIG },
    summary: { processedMatches: processed.length, ratedMatches: processed.filter(isLeagueMatch).length, playersUpdated: players.length, historyRows: history.length } };
}

export async function saveLeagueChange(supabase: SupabaseClient, leagueId: number, change: LeagueChange, requestId: string): Promise<ResultRebuildSummary> {
  if (change.action !== "rebuild") change = { ...change, match: { ...change.match, competition: matchCompetition(change.match) } };
  // A response can be lost after a successful delete/edit. Check the request before
  // applying the change again locally, then let the RPC verify its exact payload.
  const prior = await supabase.from("rating_runs").select("id").eq("league_id", leagueId).eq("id", requestId).maybeSingle();
  if (prior.error) throw ratingDatabaseError(prior.error);
  if (prior.data) {
    const retry = await supabase.rpc("commit_competition_rating_run", { p_league_id: leagueId, p_expected_token: "", p_request_id: requestId,
      p_change: change, p_calculated: {}, p_model_version: BALANCE_VERSION });
    if (retry.error) throw ratingDatabaseError(retry.error);
    return retry.data as ResultRebuildSummary;
  }
  const workspace = await loadRatingWorkspace(supabase, leagueId);
  const calculated = buildLeagueRatings(workspace, applyLeagueChange(workspace.matches, change));
  const { data, error } = await supabase.rpc("commit_competition_rating_run", {
    p_league_id: leagueId, p_expected_token: workspace.token, p_request_id: requestId,
    p_change: change, p_calculated: calculated, p_model_version: BALANCE_VERSION
  });
  if (error) throw ratingDatabaseError(error);
  return data as ResultRebuildSummary;
}
export async function recalculateLeagueResults(supabase: SupabaseClient, leagueId: number): Promise<ResultRebuildSummary> {
  return saveLeagueChange(supabase, leagueId, { action: "rebuild" }, crypto.randomUUID());
}
