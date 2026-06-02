import { getSupabase } from "./supabase";
import { chemistryScoreFor, rivalryScoreFor } from "./relationship-scoring";

export type League = {
  id: number;
  name: string;
  join_code: string | null;
};

export type Player = {
  id: number;
  name: string;
  display_name: string | null;
  mmr: number | null;
  matches_played: number | null;
  wins: number | null;
  draws: number | null;
  losses: number | null;
  win_streak: number | null;
  lose_streak: number | null;
  fitness: string | null;
};

export type Match = {
  id: number;
  date: string | null;
  team_a: string | null;
  team_b: string | null;
  score: string | null;
  result: string | null;
  processed: number | null;
};

export type MmrHistory = {
  id: number;
  player_id: number | null;
  match_id: number | null;
  date: string | null;
  mmr_before: number | null;
  mmr_after: number | null;
};

export type PlayerSummary = Player & {
  label: string;
  form: string[];
  goalsFor: number;
  goalsAgainst: number;
  goalDiff: number;
  recentMatches: Match[];
  allMatches: Match[];
  periodMmrStart: number | null;
  periodMmrEnd: number | null;
  periodMmrChange: number | null;
};

export function displayName(player: Pick<Player, "name" | "display_name">): string {
  return player.display_name?.trim() || player.name;
}

export function normalizeName(value: string | null | undefined): string {
  return String(value || "").trim().toLowerCase();
}

export function splitTeam(value: string | null | undefined): string[] {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function scoreParts(score: string | null | undefined): [number, number] | null {
  const parts = String(score || "").split(/[-:]/).map((part) => Number(part.trim()));
  if (parts.length !== 2 || parts.some((part) => Number.isNaN(part))) {
    return null;
  }
  return [parts[0], parts[1]];
}

export function formatUkDate(value: string | null | undefined): string {
  const raw = String(value || "").trim();
  if (!raw) return "-";
  const [year, month, day] = raw.slice(0, 10).split("-");
  if (year?.length === 4 && month?.length === 2 && day?.length === 2) {
    return `${day}/${month}/${year}`;
  }
  return raw;
}

export function resultFor(match: Match, side: "A" | "B"): "W" | "D" | "L" {
  const score = scoreParts(match.score);
  if (!score || score[0] === score[1]) {
    return "D";
  }
  const aWon = score[0] > score[1];
  return (side === "A" && aWon) || (side === "B" && !aWon) ? "W" : "L";
}

export function makeNameMap(players: Player[]) {
  const map = new Map<string, string>();
  for (const player of players) {
    map.set(normalizeName(player.name), displayName(player));
    map.set(normalizeName(player.display_name), displayName(player));
  }
  return map;
}

export function formatTeam(value: string | null | undefined, nameMap: Map<string, string>): string[] {
  return splitTeam(value).map((name) => nameMap.get(normalizeName(name)) || name);
}

export async function getDemoLeague(): Promise<League> {
  const supabase = getSupabase();
  const joinCode = process.env.NEXT_PUBLIC_DEMO_JOIN_CODE || "DEMO2026";
  const { data, error } = await supabase
    .from("leagues")
    .select("id,name,join_code")
    .eq("join_code", joinCode)
    .order("id", { ascending: true })
    .limit(1);

  const league = data?.[0];

  if (error || !league) {
    throw new Error(error?.message || "Demo league not found");
  }

  return league as League;
}

export async function getDemoPlayers(leagueId: number): Promise<Player[]> {
  const supabase = getSupabase();
  const { data, error } = await supabase
    .from("players")
    .select("id,name,display_name,mmr,matches_played,wins,draws,losses,win_streak,lose_streak,fitness")
    .eq("league_id", leagueId)
    .eq("is_active", 1)
    .order("mmr", { ascending: false });

  if (error) {
    throw new Error(error.message);
  }

  return (data || []) as Player[];
}

export async function getDemoMatches(leagueId: number, limit = 12): Promise<Match[]> {
  const supabase = getSupabase();
  const { data, error } = await supabase
    .from("matches")
    .select("id,date,team_a,team_b,score,result,processed")
    .eq("league_id", leagueId)
    .eq("processed", 1)
    .order("date", { ascending: false })
    .limit(limit);

  if (error) {
    throw new Error(error.message);
  }

  return (data || []) as Match[];
}

export async function getDemoMmrHistory(leagueId: number): Promise<MmrHistory[]> {
  const supabase = getSupabase();
  const { data, error } = await supabase
    .from("mmr_history")
    .select("id,player_id,match_id,date,mmr_before,mmr_after")
    .eq("league_id", leagueId)
    .order("date", { ascending: true });

  if (error) {
    throw new Error(error.message);
  }

  return (data || []) as MmrHistory[];
}

export async function getDemoSummary() {
  const league = await getDemoLeague();
  const [players, matches, mmrHistory] = await Promise.all([
    getDemoPlayers(league.id),
    getDemoMatches(league.id, 80),
    getDemoMmrHistory(league.id)
  ]);

  const firstMatch = [...matches].sort((a, b) => String(a.date).localeCompare(String(b.date)))[0];
  const latestMatch = matches[0];
  const nameMap = makeNameMap(players);

  return {
    league,
    players,
    matches,
    mmrHistory,
    nameMap,
    firstMatch,
    latestMatch
  };
}

export function buildPlayerSummaries(players: Player[], matches: Match[]): PlayerSummary[] {
  return players.map((player) => {
    const key = normalizeName(player.name);
    let goalsFor = 0;
    let goalsAgainst = 0;
    let wins = 0;
    let draws = 0;
    let losses = 0;
    const involved: Array<{ match: Match; side: "A" | "B" }> = [];

    for (const match of matches) {
      const teamA = splitTeam(match.team_a).map(normalizeName);
      const teamB = splitTeam(match.team_b).map(normalizeName);
      const side = teamA.includes(key) ? "A" : teamB.includes(key) ? "B" : null;
      const score = scoreParts(match.score);

      if (!side || !score) {
        continue;
      }

      involved.push({ match, side });
      goalsFor += side === "A" ? score[0] : score[1];
      goalsAgainst += side === "A" ? score[1] : score[0];
      const result = resultFor(match, side);
      wins += result === "W" ? 1 : 0;
      draws += result === "D" ? 1 : 0;
      losses += result === "L" ? 1 : 0;
    }

    return {
      ...player,
      matches_played: involved.length,
      wins,
      draws,
      losses,
      label: displayName(player),
      form: involved.slice(0, 6).map(({ match, side }) => resultFor(match, side)),
      goalsFor,
      goalsAgainst,
      goalDiff: goalsFor - goalsAgainst,
      recentMatches: involved.slice(0, 10).map(({ match }) => match),
      allMatches: involved.map(({ match }) => match),
      periodMmrStart: null,
      periodMmrEnd: Number(player.mmr || 1000),
      periodMmrChange: null
    };
  });
}

export function seasonBreakdown(matches: Match[]) {
  const seasons = new Map<string, { season: string; matches: number; goals: number; draws: number }>();

  for (const match of matches) {
    const season = String(match.date || "Unknown").slice(0, 4);
    const score = scoreParts(match.score);
    const row = seasons.get(season) || { season, matches: 0, goals: 0, draws: 0 };
    row.matches += 1;
    row.goals += score ? score[0] + score[1] : 0;
    row.draws += score && score[0] === score[1] ? 1 : 0;
    seasons.set(season, row);
  }

  return [...seasons.values()].sort((a, b) => b.season.localeCompare(a.season));
}

export function duoChemistry(players: Player[], matches: Match[], mode: "team" | "opponent") {
  const nameMap = makeNameMap(players);
  const playerPointRates = buildPlayerPointRates(matches);
  const rows = new Map<string, {
    aKey: string;
    bKey: string;
    a: string;
    b: string;
    matches: number;
    wins: number;
    draws: number;
    losses: number;
    goalDiff: number;
    goalGapTotal: number;
    scoreSum: number;
    actualRate: number;
    expectedRate: number;
    residual: number;
    score: number;
    winPct: number;
    avgGoalDiff: number;
  }>();

  function touch(a: string, b: string) {
    const key = [normalizeName(a), normalizeName(b)].sort().join("|");
    const [aKey, bKey] = key.split("|");
    const names = [aKey, bKey].map((name) => nameMap.get(name) || name);
    if (!rows.has(key)) {
      rows.set(key, {
        aKey,
        bKey,
        a: names[0],
        b: names[1],
        matches: 0,
        wins: 0,
        draws: 0,
        losses: 0,
        goalDiff: 0,
        goalGapTotal: 0,
        scoreSum: 0,
        actualRate: 0.5,
        expectedRate: 0.5,
        residual: 0,
        score: 0,
        winPct: 0,
        avgGoalDiff: 0
      });
    }
    return rows.get(key)!;
  }

  for (const match of matches) {
    const score = scoreParts(match.score);
    if (!score) continue;
    const teamA = splitTeam(match.team_a);
    const teamB = splitTeam(match.team_b);
    const sides = mode === "team"
      ? [[teamA, "A" as const], [teamB, "B" as const]]
      : teamA.flatMap((a) => teamB.map((b) => [[a, b], "A" as const] as const));

    if (mode === "team") {
      for (const [team, side] of sides as Array<[string[], "A" | "B"]>) {
        for (let i = 0; i < team.length; i += 1) {
          for (let j = i + 1; j < team.length; j += 1) {
            const row = touch(team[i], team[j]);
            const res = resultFor(match, side);
            row.matches += 1;
            row.wins += res === "W" ? 1 : 0;
            row.draws += res === "D" ? 1 : 0;
            row.losses += res === "L" ? 1 : 0;
            row.scoreSum += res === "W" ? 1 : res === "D" ? 0.5 : 0;
            row.goalDiff += side === "A" ? score[0] - score[1] : score[1] - score[0];
            row.goalGapTotal += Math.abs(score[0] - score[1]);
          }
        }
      }
    } else {
      for (const a of teamA) {
        for (const b of teamB) {
          const row = touch(a, b);
          const rowAIsTeamA = normalizeName(row.a) === normalizeName(a);
          const resForRowA = score[0] === score[1]
            ? "D"
            : (score[0] > score[1]) === rowAIsTeamA ? "W" : "L";
          row.matches += 1;
          row.goalDiff += Math.abs(score[0] - score[1]);
          row.goalGapTotal += Math.abs(score[0] - score[1]);
          row.wins += resForRowA === "W" ? 1 : 0;
          row.draws += resForRowA === "D" ? 1 : 0;
          row.losses += resForRowA === "L" ? 1 : 0;
        }
      }
    }
  }

  return [...rows.values()]
    .filter((row) => row.matches > 0)
    .map((row) => {
      const avgGoalDiff = row.goalGapTotal / Math.max(row.matches, 1);
      const winPct = row.wins / Math.max(row.matches, 1);
      const actualRate = row.scoreSum / Math.max(row.matches, 1);
      const expectedRate = mode === "team"
        ? (pointRateFor(row.aKey, playerPointRates) + pointRateFor(row.bKey, playerPointRates)) / 2
        : 0.5;
      const residual = mode === "team" ? actualRate - expectedRate : 0;
      const score = mode === "team"
        ? chemistryScoreFor({ ...row, actualRate, expectedRate, goalGapTotal: row.goalGapTotal })
        : rivalryScoreFor({
            matches: row.matches,
            winsA: row.wins,
            winsB: row.losses,
            draws: row.draws,
            totalGoalGap: row.goalGapTotal
          });
      return { ...row, avgGoalDiff, winPct: winPct * 100, actualRate, expectedRate, residual, score };
    })
    .sort((a, b) => b.score - a.score || b.residual - a.residual || b.matches - a.matches)
    .slice(0, 12);
}

function buildPlayerPointRates(matches: Match[]) {
  const rows = new Map<string, number[]>();

  for (const match of matches) {
    const score = scoreParts(match.score);
    if (!score) continue;
    for (const side of ["A", "B"] as const) {
      const value = resultFor(match, side) === "W" ? 1 : resultFor(match, side) === "D" ? 0.5 : 0;
      const team = splitTeam(side === "A" ? match.team_a : match.team_b).map(normalizeName).filter(Boolean);
      for (const name of team) {
        const playerRows = rows.get(name) || [];
        playerRows.push(value);
        rows.set(name, playerRows);
      }
    }
  }

  const rates = new Map<string, number>();
  for (const [name, values] of rows) {
    rates.set(name, values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1));
  }
  return rates;
}

function pointRateFor(name: string, rates: Map<string, number>) {
  return rates.get(normalizeName(name)) ?? 0.5;
}

export function trioInsights(players: Player[], matches: Match[]) {
  const nameMap = makeNameMap(players);
  const trioRows = new Map<string, { trio: string[]; matches: number; wins: number; goalDiffs: number[] }>();
  const duoRows = new Map<string, { matches: number; wins: number }>();

  function combos(team: string[], size: number) {
    const out: string[][] = [];
    function walk(start: number, path: string[]) {
      if (path.length === size) {
        out.push([...path]);
        return;
      }
      for (let i = start; i < team.length; i += 1) {
        walk(i + 1, [...path, team[i]]);
      }
    }
    walk(0, []);
    return out;
  }

  function keyFor(names: string[]) {
    return names.map(normalizeName).sort().join("|");
  }

  for (const match of matches) {
    const score = scoreParts(match.score);
    if (!score) continue;
    const gd = Math.abs(score[0] - score[1]);
    const teams = [
      { side: "A" as const, names: splitTeam(match.team_a) },
      { side: "B" as const, names: splitTeam(match.team_b) }
    ];

    for (const team of teams) {
      const won = resultFor(match, team.side) === "W";
      for (const trio of combos(team.names, 3)) {
        const key = keyFor(trio);
        const existing = trioRows.get(key) || {
          trio: key.split("|").map((name) => nameMap.get(name) || name),
          matches: 0,
          wins: 0,
          goalDiffs: []
        };
        existing.matches += 1;
        existing.wins += won ? 1 : 0;
        existing.goalDiffs.push(gd);
        trioRows.set(key, existing);
      }
      for (const duo of combos(team.names, 2)) {
        const key = keyFor(duo);
        const existing = duoRows.get(key) || { matches: 0, wins: 0 };
        existing.matches += 1;
        existing.wins += won ? 1 : 0;
        duoRows.set(key, existing);
      }
    }
  }

  return [...trioRows.values()]
    .map((row) => {
      const winPct = row.matches ? (row.wins / row.matches) * 100 : 0;
      const avgGd = row.goalDiffs.reduce((sum, item) => sum + item, 0) / Math.max(row.goalDiffs.length, 1);
      const duoKeys = [
        [row.trio[0], row.trio[1]],
        [row.trio[0], row.trio[2]],
        [row.trio[1], row.trio[2]]
      ].map(keyFor);
      const duoWinPcts = duoKeys
        .map((key) => duoRows.get(key))
        .filter(Boolean)
        .map((duo) => (duo!.wins / Math.max(duo!.matches, 1)) * 100);
      const avgDuoWin = duoWinPcts.reduce((sum, item) => sum + item, 0) / Math.max(duoWinPcts.length, 1);
      const strength = row.matches * (winPct / 100) * Math.max(0.35, 1 - avgGd / 8);
      return {
        trio: row.trio,
        matches: row.matches,
        wins: row.wins,
        winPct,
        avgGd,
        strength,
        synergy: winPct - avgDuoWin
      };
    })
    .sort((a, b) => b.strength - a.strength)
    .slice(0, 40);
}
