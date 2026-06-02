import { type Match, type MmrHistory, type Player, type PlayerSummary } from "./demo-data";

export const STARTING_MMR = 1000;
export const DEFAULT_K_FACTOR = 30;
export const DEFAULT_DRAW_VALUE = 0.5;

export type MmrUpdate = {
  player: Player;
  team: "A" | "B";
  before: number;
  after: number;
  expected: number;
  delta: number;
  kFinal: number;
};

export type ImprovementLeader = {
  player: PlayerSummary;
  season: string;
  currentGain: number;
  previousAverageGain: number;
  improvementScore: number;
  previousPeriods: number;
  currentMatches: number;
};

export function expectedScore(mmrA: number, mmrB: number) {
  return 1 / (1 + 10 ** ((mmrB - mmrA) / 400));
}

export function expectedScoreCalibrated(ratingA: number, ratingB: number, scale = 200) {
  return 1 / (1 + Math.exp(-((ratingA - ratingB) / scale)));
}

export function imbalanceFactor(playerTeamAvg: number, oppTeamAvg: number) {
  const gap = Math.abs(playerTeamAvg - oppTeamAvg);
  const boost = Math.min(0.5, gap / 300);
  const cut = Math.min(0.4, gap / 300);
  return playerTeamAvg < oppTeamAvg ? 1 + boost : 1 - cut;
}

export function volatilityFactor(playerMmr: number, leagueMean: number) {
  const value = 1 - 0.5 * ((playerMmr - leagueMean) / 1000);
  return Math.max(0.5, Math.min(1, value));
}

export function scoreToResult(match: Pick<Match, "score" | "result">): "A" | "B" | "DRAW" {
  const parsed = String(match.score || "")
    .replace(":", "-")
    .split("-")
    .map((part) => Number(part.trim()));

  if (parsed.length === 2 && parsed.every((part) => !Number.isNaN(part))) {
    if (parsed[0] > parsed[1]) return "A";
    if (parsed[1] > parsed[0]) return "B";
    return "DRAW";
  }

  const raw = String(match.result || "").trim().toUpperCase();
  if (raw === "A") return "A";
  if (raw === "B") return "B";
  return "DRAW";
}

export function calculateMatchMmrUpdates({
  teamA,
  teamB,
  match,
  players,
  kFactor = DEFAULT_K_FACTOR,
  drawValue = DEFAULT_DRAW_VALUE
}: {
  teamA: Player[];
  teamB: Player[];
  match: Pick<Match, "score" | "result">;
  players: Player[];
  kFactor?: number;
  drawValue?: number;
}): MmrUpdate[] {
  const leagueMean = average(players.map((player) => numberOr(player.mmr, STARTING_MMR)));
  const teamAAvg = average(teamA.map((player) => numberOr(player.mmr, STARTING_MMR)));
  const teamBAvg = average(teamB.map((player) => numberOr(player.mmr, STARTING_MMR)));
  const result = scoreToResult(match);
  const scoreA = result === "A" ? 1 : result === "B" ? 0 : drawValue;
  const scoreB = result === "B" ? 1 : result === "A" ? 0 : drawValue;

  return [
    ...teamA.map((player) => calculatePlayerUpdate(player, "A", teamAAvg, teamBAvg, scoreA, leagueMean, kFactor)),
    ...teamB.map((player) => calculatePlayerUpdate(player, "B", teamBAvg, teamAAvg, scoreB, leagueMean, kFactor))
  ];
}

export function getSeasonDisplayMmr(playerId: number, seasonStart: string, rollingMmr: number, history: MmrHistory[]) {
  const baselineRow = [...history]
    .filter((row) => row.player_id === playerId && String(row.date || "") < seasonStart)
    .sort(compareMmrRows)
    .pop();
  const baseline = numberOr(baselineRow?.mmr_after, STARTING_MMR);
  return STARTING_MMR + (rollingMmr - baseline);
}

export function applyPeriodMmr(summaries: PlayerSummary[], history: MmrHistory[], season = "all"): PlayerSummary[] {
  const seasonStart = season === "all" ? null : `${season}-01-01`;
  const byPlayer = new Map<number, MmrHistory[]>();

  for (const row of history) {
    if (!row.player_id) continue;
    const existing = byPlayer.get(row.player_id) || [];
    existing.push(row);
    byPlayer.set(row.player_id, existing);
  }

  for (const rows of byPlayer.values()) {
    rows.sort(compareMmrRows);
  }

  return summaries.map((player) => {
    const rows = byPlayer.get(player.id) || [];
    if (!rows.length) {
      return {
        ...player,
        mmr: season === "all" ? player.mmr : STARTING_MMR,
        periodMmrStart: STARTING_MMR,
        periodMmrEnd: season === "all" ? numberOr(player.mmr, STARTING_MMR) : STARTING_MMR,
        periodMmrChange: 0
      };
    }

    if (season === "all") {
      const first = rows[0];
      const latest = rows[rows.length - 1];
      const start = numberOr(first.mmr_before, STARTING_MMR);
      const end = numberOr(latest.mmr_after, numberOr(player.mmr, start));
      return {
        ...player,
        mmr: numberOr(player.mmr, end),
        periodMmrStart: start,
        periodMmrEnd: end,
        periodMmrChange: Math.round(end - start)
      };
    }

    const seasonRows = rows.filter((row) => String(row.date || "").startsWith(season));
    if (!seasonRows.length) {
      return {
        ...player,
        mmr: STARTING_MMR,
        periodMmrStart: STARTING_MMR,
        periodMmrEnd: STARTING_MMR,
        periodMmrChange: 0
      };
    }

    const endRolling = numberOr(seasonRows[seasonRows.length - 1].mmr_after, STARTING_MMR);
    const seasonMmr = getSeasonDisplayMmr(player.id, seasonStart || "", endRolling, history);

    return {
      ...player,
      mmr: seasonMmr,
      periodMmrStart: STARTING_MMR,
      periodMmrEnd: seasonMmr,
      periodMmrChange: Math.round(seasonMmr - STARTING_MMR)
    };
  });
}

export function findMostImprovedPlayer(
  summaries: PlayerSummary[],
  history: MmrHistory[],
  seasons: string[],
  selectedSeason = "all"
): ImprovementLeader | null {
  return findMostImprovedPlayers(summaries, history, seasons, selectedSeason)[0] || null;
}

export function findMostImprovedPlayers(
  summaries: PlayerSummary[],
  history: MmrHistory[],
  seasons: string[],
  selectedSeason = "all"
): ImprovementLeader[] {
  const orderedSeasons = [...new Set(seasons)]
    .filter(Boolean)
    .sort((a, b) => Number(a) - Number(b));
  const currentSeason = selectedSeason === "all" ? orderedSeasons[orderedSeasons.length - 1] : selectedSeason;
  if (!currentSeason) return [];

  const previousSeasons = orderedSeasons.filter((season) => Number(season) < Number(currentSeason));
  const candidates = summaries
    .map((player) => {
      const current = seasonGain(player.id, currentSeason, history);
      if (!current || current.matches < 3) return null;

      const previousGains = previousSeasons
        .map((season) => seasonGain(player.id, season, history))
        .filter((row): row is { gain: number; matches: number } => Boolean(row && row.matches >= 3))
        .map((row) => row.gain);

      const previousAverageGain = previousGains.length ? average(previousGains) : 0;
      const historyConfidence = previousGains.length ? 1 : 0.65;
      const improvementScore = (current.gain - previousAverageGain) * historyConfidence;

      return {
        player,
        season: currentSeason,
        currentGain: Math.round(current.gain),
        previousAverageGain: Math.round(previousAverageGain),
        improvementScore: Math.round(improvementScore),
        previousPeriods: previousGains.length,
        currentMatches: current.matches
      };
    })
    .filter((row): row is ImprovementLeader => Boolean(row));

  return candidates.sort((a, b) => {
    return b.improvementScore - a.improvementScore
      || b.currentGain - a.currentGain
      || b.currentMatches - a.currentMatches;
  });
}

function calculatePlayerUpdate(
  player: Player,
  team: "A" | "B",
  playerTeamAvg: number,
  oppTeamAvg: number,
  actualScore: number,
  leagueMean: number,
  kFactor: number
): MmrUpdate {
  const before = numberOr(player.mmr, STARTING_MMR);
  const expected = expectedScore(before, oppTeamAvg);
  const kFinal = kFactor * imbalanceFactor(playerTeamAvg, oppTeamAvg) * volatilityFactor(before, leagueMean);
  const delta = kFinal * (actualScore - expected);
  return {
    player,
    team,
    before,
    after: before + delta,
    expected,
    delta,
    kFinal
  };
}

function compareMmrRows(a: MmrHistory, b: MmrHistory) {
  const dateOrder = String(a.date || "").localeCompare(String(b.date || ""));
  return dateOrder || (a.id || 0) - (b.id || 0);
}

function seasonGain(playerId: number, season: string, history: MmrHistory[]) {
  const rows = [...history].filter((row) => row.player_id === playerId).sort(compareMmrRows);
  const seasonRows = rows.filter((row) => String(row.date || "").startsWith(season));
  if (!seasonRows.length) return null;

  const baselineRow = rows.filter((row) => String(row.date || "") < `${season}-01-01`).pop();
  const baseline = numberOr(baselineRow?.mmr_after, STARTING_MMR);
  const end = numberOr(seasonRows[seasonRows.length - 1].mmr_after, baseline);

  return {
    gain: end - baseline,
    matches: seasonRows.length
  };
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function numberOr(value: number | null | undefined, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}
