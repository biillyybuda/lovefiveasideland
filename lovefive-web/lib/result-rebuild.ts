import type { SupabaseClient } from "@supabase/supabase-js";
import { normalizeName, scoreParts, splitTeam, type Match, type Player } from "./demo-data";
import { calculateMatchMmrUpdates, scoreToResult, STARTING_MMR } from "./mmr-engine";

const PAGE_SIZE = 500;
const PLAYER_SELECT = "id,name,display_name,mmr,matches_played,wins,draws,losses,win_streak,lose_streak,fitness";
const MATCH_SELECT = "id,date,team_a,team_b,score,result,processed";

type RebuildPlayer = Player & {
  mmr: number;
  matches_played: number;
  wins: number;
  draws: number;
  losses: number;
  win_streak: number;
  lose_streak: number;
};

type MmrHistoryInsert = {
  league_id: number;
  player_id: number;
  match_id: number;
  date: string | null;
  mmr_before: number;
  mmr_after: number;
};

export type ResultRebuildSummary = {
  processedMatches: number;
  playersUpdated: number;
  historyRows: number;
};

export function normaliseScoreText(value: string | null | undefined) {
  const parsed = scoreParts(value);
  return parsed ? `${parsed[0]}-${parsed[1]}` : "";
}

export function resultLabelFromScore(value: string | null | undefined) {
  const result = scoreToResult({ score: value || "", result: "" });
  return result === "DRAW" ? "Draw" : result;
}

export async function recalculateLeagueResults(
  supabase: SupabaseClient,
  leagueId: number
): Promise<ResultRebuildSummary> {
  const [players, matches] = await Promise.all([
    loadAllPlayers(supabase, leagueId),
    loadAllProcessedMatches(supabase, leagueId)
  ]);

  const playerState = initialisePlayerState(players);
  const nameLookup = buildNameLookup(playerState);
  const historyRows: MmrHistoryInsert[] = [];
  const matchResultUpdates: Array<{ id: number; result: string }> = [];

  for (const match of matches) {
    if (!scoreParts(match.score)) {
      throw new Error(`${matchLabel(match)} has an invalid score.`);
    }

    const result = scoreToResult(match);
    const resultLabel = result === "DRAW" ? "Draw" : result;
    if (String(match.result || "").trim() !== resultLabel) {
      matchResultUpdates.push({ id: match.id, result: resultLabel });
    }

    const teamA = resolveTeam(match, "A", nameLookup);
    const teamB = resolveTeam(match, "B", nameLookup);
    if (!teamA.length || !teamB.length) {
      throw new Error(`${matchLabel(match)} needs players on both teams.`);
    }

    const updates = calculateMatchMmrUpdates({
      teamA,
      teamB,
      match: { ...match, result: resultLabel },
      players: Array.from(playerState.values())
    });

    for (const update of updates) {
      const player = playerState.get(update.player.id);
      if (!player) continue;

      const after = Math.round(update.after);
      const outcome = outcomeForSide(result, update.team);
      player.mmr = after;
      player.matches_played += 1;
      player.wins += outcome === "W" ? 1 : 0;
      player.draws += outcome === "D" ? 1 : 0;
      player.losses += outcome === "L" ? 1 : 0;
      player.win_streak = outcome === "W" ? player.win_streak + 1 : 0;
      player.lose_streak = outcome === "L" ? player.lose_streak + 1 : 0;

      historyRows.push({
        league_id: leagueId,
        player_id: player.id,
        match_id: match.id,
        date: match.date,
        mmr_before: Math.round(update.before),
        mmr_after: after
      });
    }
  }

  await updateStoredMatchResults(supabase, leagueId, matchResultUpdates);
  await replaceMmrHistory(supabase, leagueId, historyRows);
  await updatePlayers(supabase, leagueId, Array.from(playerState.values()));

  return {
    processedMatches: matches.length,
    playersUpdated: playerState.size,
    historyRows: historyRows.length
  };
}

async function loadAllPlayers(supabase: SupabaseClient, leagueId: number): Promise<Player[]> {
  const rows: Player[] = [];
  for (let from = 0; ; from += PAGE_SIZE) {
    const { data, error } = await supabase
      .from("players")
      .select(PLAYER_SELECT)
      .eq("league_id", leagueId)
      .order("name", { ascending: true })
      .range(from, from + PAGE_SIZE - 1);

    if (error) throw new Error(error.message);
    const chunk = (data || []) as Player[];
    rows.push(...chunk);
    if (chunk.length < PAGE_SIZE) break;
  }
  return rows;
}

async function loadAllProcessedMatches(supabase: SupabaseClient, leagueId: number): Promise<Match[]> {
  const rows: Match[] = [];
  for (let from = 0; ; from += PAGE_SIZE) {
    const { data, error } = await supabase
      .from("matches")
      .select(MATCH_SELECT)
      .eq("league_id", leagueId)
      .eq("processed", 1)
      .order("date", { ascending: true })
      .order("id", { ascending: true })
      .range(from, from + PAGE_SIZE - 1);

    if (error) throw new Error(error.message);
    const chunk = (data || []) as Match[];
    rows.push(...chunk);
    if (chunk.length < PAGE_SIZE) break;
  }
  return rows;
}

function initialisePlayerState(players: Player[]) {
  return new Map<number, RebuildPlayer>(
    players.map((player) => [
      player.id,
      {
        ...player,
        mmr: STARTING_MMR,
        matches_played: 0,
        wins: 0,
        draws: 0,
        losses: 0,
        win_streak: 0,
        lose_streak: 0
      }
    ])
  );
}

function buildNameLookup(players: Map<number, RebuildPlayer>) {
  const lookup = new Map<string, RebuildPlayer>();
  for (const player of players.values()) {
    addNameLookup(lookup, player.name, player);
    addNameLookup(lookup, player.display_name, player);
  }
  return lookup;
}

function addNameLookup(lookup: Map<string, RebuildPlayer>, value: string | null | undefined, player: RebuildPlayer) {
  const key = normalizeName(value);
  if (key && !lookup.has(key)) lookup.set(key, player);
}

function resolveTeam(match: Match, side: "A" | "B", lookup: Map<string, RebuildPlayer>) {
  const team = splitTeam(side === "A" ? match.team_a : match.team_b);
  const seen = new Set<number>();
  const players: RebuildPlayer[] = [];

  for (const rawName of team) {
    const player = lookup.get(normalizeName(rawName));
    if (!player) {
      throw new Error(`${matchLabel(match)} includes ${rawName}, who is not in this league's player list.`);
    }
    if (!seen.has(player.id)) {
      players.push(player);
      seen.add(player.id);
    }
  }

  return players;
}

function outcomeForSide(result: "A" | "B" | "DRAW", side: "A" | "B") {
  if (result === "DRAW") return "D";
  return result === side ? "W" : "L";
}

async function updateStoredMatchResults(
  supabase: SupabaseClient,
  leagueId: number,
  updates: Array<{ id: number; result: string }>
) {
  for (const update of updates) {
    const { error } = await supabase
      .from("matches")
      .update({ result: update.result })
      .eq("id", update.id)
      .eq("league_id", leagueId);
    if (error) throw new Error(error.message);
  }
}

async function replaceMmrHistory(supabase: SupabaseClient, leagueId: number, rows: MmrHistoryInsert[]) {
  const { error: deleteError } = await supabase.from("mmr_history").delete().eq("league_id", leagueId);
  if (deleteError) throw new Error(deleteError.message);

  for (let index = 0; index < rows.length; index += PAGE_SIZE) {
    const chunk = rows.slice(index, index + PAGE_SIZE);
    const { error } = await supabase.from("mmr_history").insert(chunk);
    if (error) throw new Error(error.message);
  }
}

async function updatePlayers(supabase: SupabaseClient, leagueId: number, players: RebuildPlayer[]) {
  for (const player of players) {
    const { error } = await supabase
      .from("players")
      .update({
        mmr: player.mmr,
        matches_played: player.matches_played,
        wins: player.wins,
        draws: player.draws,
        losses: player.losses,
        win_streak: player.win_streak,
        lose_streak: player.lose_streak
      })
      .eq("id", player.id)
      .eq("league_id", leagueId);

    if (error) throw new Error(error.message);
  }
}

function matchLabel(match: Match) {
  return `Match ${match.id}${match.date ? ` on ${match.date}` : ""}`;
}
