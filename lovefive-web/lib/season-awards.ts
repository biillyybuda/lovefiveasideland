import { buildPlayerSummaries, normalizeName, scoreParts, splitTeam, type Match, type MmrHistory, type Player, type PlayerSummary } from "./demo-data";
import { applyPeriodMmr } from "./mmr-engine";
import { calculateAwardMmrUpdates, STARTING_MMR } from "./award-rating";

export type Quarter = "full" | "ytd" | "q1" | "q2" | "q3" | "q4";

export type ImprovementRow = {
  key: string;
  label: string;
  value: number;
  detail?: string;
};

export type AwardStat = {
  label: string;
  value: string;
  detail: string;
};

export type MvpRow = ImprovementRow & {
  player: PlayerSummary;
  stats: AwardStat[];
};

export type ImprovedRow = ImprovementRow & {
  stats: AwardStat[];
  unit: "pts" | "MMR";
};

export const quarters: Array<{ value: Quarter; label: string; start: string; end: string }> = [
  { value: "full", label: "Full year", start: "01-01", end: "12-31" },
  { value: "ytd", label: "Year to date", start: "01-01", end: "latest" },
  { value: "q1", label: "Jan-Mar", start: "01-01", end: "03-31" },
  { value: "q2", label: "Apr-Jun", start: "04-01", end: "06-30" },
  { value: "q3", label: "Jul-Sep", start: "07-01", end: "09-30" },
  { value: "q4", label: "Oct-Dec", start: "10-01", end: "12-31" }
];

export function getPeriod(season: string, quarter: Quarter, matches: Match[]) {
  if (season === "all") {
    return { start: null, end: null, label: "All seasons" };
  }
  const row = quarters.find((item) => item.value === quarter) || quarters[0];
  const end = row.value === "ytd" ? latestMatchDateForSeason(matches, season) || `${season}-12-31` : `${season}-${row.end}`;
  return {
    start: `${season}-${row.start}`,
    end,
    label: `${season} - ${row.label}`
  };
}

export function latestMatchDateForSeason(matches: Match[], season: string) {
  return matches
    .map((match) => String(match.date || "").slice(0, 10))
    .filter((date) => date.startsWith(`${season}-`))
    .sort((a, b) => b.localeCompare(a))[0] || null;
}

export function previousComparisonPeriod(
  season: string,
  quarter: Quarter,
  matches: Match[],
  currentPeriod: { start: string | null; end: string | null; label: string }
) {
  if (!/^\d{4}$/.test(season) || !currentPeriod.start || !currentPeriod.end) return null;
  if (isNamedQuarter(quarter)) return null;
  const previousSeason = String(Number(season) - 1);
  if (quarter === "ytd") {
    return {
      season: previousSeason,
      start: shiftYear(currentPeriod.start, -1),
      end: shiftYear(currentPeriod.end, -1),
      label: `${previousSeason} - same point`
    };
  }
  return {
    season: previousSeason,
    ...getPeriod(previousSeason, quarter, matches)
  };
}

export function isNamedQuarter(quarter: Quarter) {
  return quarter === "q1" || quarter === "q2" || quarter === "q3" || quarter === "q4";
}

export function shiftYear(date: string, offset: number) {
  const year = Number(date.slice(0, 4));
  if (!Number.isFinite(year)) return date;
  return `${year + offset}${date.slice(4)}`;
}

export function inPeriod(date: string | null | undefined, start: string | null, end: string | null) {
  const value = String(date || "").slice(0, 10);
  if (!value) return false;
  if (start && value < start) return false;
  if (end && value > end) return false;
  return true;
}

export function playerSummariesForPeriod(
  players: Player[],
  periodMatches: Match[],
  mmrHistory: MmrHistory[],
  start: string | null,
  end: string | null,
  mmrSeason: string
) {
  const base = buildPlayerSummaries(players, periodMatches);
  const withSeasonMmr = applyPeriodMmr(base, mmrHistory, mmrSeason);
  return withSeasonMmr.map((player) => {
    const change = periodMmrChange(mmrHistory, player.id, start, end);
    return change
      ? { ...player, periodMmrStart: change.start, periodMmrEnd: change.end, periodMmrChange: Math.round(change.end - change.start) }
      : player;
  });
}

export function periodMmrChange(history: MmrHistory[], playerId: number, start: string | null, end: string | null) {
  const rows = history
    .filter((row) => row.player_id === playerId && inPeriod(row.date, start, end))
    .sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")) || Number(a.id || 0) - Number(b.id || 0));
  if (!rows.length) return null;
  return {
    start: Number(rows[0].mmr_before || 1000),
    end: Number(rows[rows.length - 1].mmr_after || rows[0].mmr_before || 1000)
  };
}

export function mvpRace(players: PlayerSummary[], matches: Match[]): MvpRow[] {
  const matchCount = matches.length;
  const minimumMatches = minimumAwardMatches(matchCount);
  const eligible = players.filter((player) => Number(player.matches_played || 0) >= minimumMatches);
  const seasonRatings = seasonOnlyRatings(players, matches);

  return eligible
    .map((player) => {
      const seasonRating = seasonRatings.get(player.id) ?? STARTING_MMR;
      const components = awardScoreComponents(player, seasonRating, matchCount);

      return {
        key: String(player.id),
        label: player.label,
        value: components.score,
        detail: mvpDetail(player, seasonRating),
        player,
        stats: mvpStats(player, components, seasonRating)
      };
    })
    .sort((a, b) => b.value - a.value || (seasonRatings.get(b.player.id) ?? STARTING_MMR) - (seasonRatings.get(a.player.id) ?? STARTING_MMR) || Number(b.player.matches_played || 0) - Number(a.player.matches_played || 0) || recordPoints(b.player) - recordPoints(a.player));
}

export function improvedRace(
  currentPlayers: PlayerSummary[],
  currentMatches: Match[],
  previousPlayers: PlayerSummary[],
  previousMatches: Match[],
  previousPeriod: { season: string; start: string | null; end: string | null; label: string } | null,
  quarter: Quarter
): { rows: ImprovedRow[]; title: string; subtitle: string } {
  const quarterMode = isNamedQuarter(quarter);
  const fallback = {
    rows: periodMmrImprovementRows(currentPlayers, currentMatches),
    title: quarterMode ? "Quarter surge" : "Most improved",
    subtitle: quarterMode ? "Ranked by MMR gained in this quarter" : "Ranked by period MMR gain"
  };

  if (quarterMode) return fallback;
  if (!previousPeriod || !previousMatches.length) return fallback;

  const currentMinimum = minimumAwardMatches(currentMatches.length);
  const previousMinimum = minimumAwardMatches(previousMatches.length);
  const previousById = new Map(previousPlayers.map((player) => [player.id, player]));
  const currentRatings = seasonOnlyRatings(currentPlayers, currentMatches);
  const previousRatings = seasonOnlyRatings(previousPlayers, previousMatches);
  const rows = currentPlayers
    .filter((player) => Number(player.matches_played || 0) >= currentMinimum)
    .map((player): ImprovedRow | null => {
      const previous = previousById.get(player.id);
      if (!previous || Number(previous.matches_played || 0) < previousMinimum) return null;

      const currentRating = currentRatings.get(player.id) ?? STARTING_MMR;
      const previousRating = previousRatings.get(player.id) ?? STARTING_MMR;
      const currentComponents = awardScoreComponents(player, currentRating, currentMatches.length);
      const previousComponents = awardScoreComponents(previous, previousRating, previousMatches.length);
      const scoreGain = currentComponents.score - previousComponents.score;

      return {
        key: String(player.id),
        label: player.label,
        value: scoreGain,
        unit: "pts" as const,
        detail: seasonImprovementDetail(player, previous, currentComponents, previousComponents),
        stats: seasonImprovementStats(player, previous, currentComponents, previousComponents, currentRating, previousRating)
      };
    })
    .filter((row): row is ImprovedRow => Boolean(row))
    .sort((a, b) => b.value - a.value || a.label.localeCompare(b.label))
    .slice(0, 5);

  if (!rows.length) return fallback;

  return {
    rows,
    title: "Most improved",
    subtitle: `MVP-style score change vs ${previousPeriod.label}`
  };
}

export function periodMmrImprovementRows(players: PlayerSummary[], matches: Match[]): ImprovedRow[] {
  return [...players]
    .filter((player) => player.periodMmrChange !== null && Number(player.matches_played || 0) >= minimumAwardMatches(matches.length))
    .sort((a, b) => Number(b.periodMmrChange || 0) - Number(a.periodMmrChange || 0))
    .slice(0, 5)
    .map((player) => ({
      key: String(player.id),
      label: player.label,
      value: Number(player.periodMmrChange || 0),
      unit: "MMR" as const,
      detail: improvementDetail(player),
      stats: improvementStats(player)
    }));
}

export function awardScoreComponents(player: PlayerSummary, seasonRating: number, matchCount: number) {
  const played = Number(player.matches_played || 0);
  const appearances = matchCount ? played / matchCount : 0;
  const rating = ratingPoints(seasonRating);
  const games = Math.round(appearances * 20);
  return {
    rating,
    games,
    score: rating + games
  };
}

export function seasonOnlyRatings(players: PlayerSummary[], matches: Match[]) {
  const seedRatings = seededSeasonStarts(players);
  const state = new Map<number, Player>(
    players.map((player) => [
      player.id,
      {
        ...player,
        mmr: seedRatings.get(player.id) ?? STARTING_MMR
      }
    ])
  );
  const nameLookup = new Map<string, Player>();

  for (const player of state.values()) {
    addPlayerLookup(nameLookup, player.name, player);
    addPlayerLookup(nameLookup, player.display_name, player);
  }

  const orderedMatches = [...matches]
    .filter((match) => scoreParts(match.score))
    .sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")) || a.id - b.id);

  for (const match of orderedMatches) {
    const teamA = resolveSeasonTeam(match.team_a, nameLookup);
    const teamB = resolveSeasonTeam(match.team_b, nameLookup);
    if (!teamA.length || !teamB.length) continue;

    const updates = calculateAwardMmrUpdates({
      teamA,
      teamB,
      match,
      players: Array.from(state.values())
    });

    for (const update of updates) {
      const player = state.get(update.player.id);
      if (player) player.mmr = Math.round(update.after);
    }
  }

  return new Map(Array.from(state.values()).map((player) => [player.id, Number(player.mmr || STARTING_MMR)]));
}

export function seededSeasonStarts(players: PlayerSummary[]) {
  const involved = players.filter((player) => Number(player.matches_played || 0) > 0);
  const startValues = involved.map((player) => finiteOr(player.periodMmrStart, STARTING_MMR));
  const leagueAverage = startValues.reduce((sum, value) => sum + value, 0) / Math.max(startValues.length, 1);

  return new Map(
    players.map((player) => {
      const start = finiteOr(player.periodMmrStart, finiteOr(player.mmr, STARTING_MMR));
      const seeded = STARTING_MMR + (start - leagueAverage) * 0.35;
      return [player.id, Math.round(seeded)] as const;
    })
  );
}

export function addPlayerLookup(lookup: Map<string, Player>, value: string | null | undefined, player: Player) {
  const key = normalizeName(value);
  if (key && !lookup.has(key)) lookup.set(key, player);
}

export function resolveSeasonTeam(value: string | null | undefined, lookup: Map<string, Player>) {
  const seen = new Set<number>();
  const team: Player[] = [];

  for (const name of splitTeam(value)) {
    const player = lookup.get(normalizeName(name));
    if (!player || seen.has(player.id)) continue;
    team.push(player);
    seen.add(player.id);
  }

  return team;
}

export function recordPoints(player: PlayerSummary) {
  return Number(player.wins || 0) * 3 + Number(player.draws || 0);
}

export function recordLabel(player: PlayerSummary) {
  return `${Number(player.wins || 0)}-${Number(player.draws || 0)}-${Number(player.losses || 0)}`;
}

export function mvpDetail(player: PlayerSummary, seasonRating: number) {
  return `${player.matches_played} games played, ${Math.round(seasonRating)} season-only rating, ${recordLabel(player)} record`;
}

export function mvpStats(
  player: PlayerSummary,
  components: { rating: number; games: number },
  seasonRating: number
): AwardStat[] {
  return [
    { label: "Season rating", value: String(Math.round(seasonRating)), detail: `${components.rating}/80 pts` },
    { label: "Games", value: String(player.matches_played || 0), detail: `${components.games}/20 pts` },
    { label: "Rolling change", value: signed(Number(player.periodMmrChange || 0)), detail: "context" },
    { label: "Record", value: recordLabel(player), detail: "W-D-L" }
  ];
}

export function seasonImprovementStats(
  player: PlayerSummary,
  previous: PlayerSummary,
  currentComponents: { rating: number; games: number; score: number },
  previousComponents: { rating: number; games: number; score: number },
  currentRating: number,
  previousRating: number
): AwardStat[] {
  return [
    {
      label: "This season",
      value: `${currentComponents.score}/100`,
      detail: `${Math.round(currentRating)} rating, ${player.matches_played || 0} apps`
    },
    {
      label: "Last season",
      value: `${previousComponents.score}/100`,
      detail: `${Math.round(previousRating)} rating, ${previous.matches_played || 0} apps`
    },
    {
      label: "Rating swing",
      value: signed(currentComponents.rating - previousComponents.rating),
      detail: `${currentComponents.rating}/80 vs ${previousComponents.rating}/80 pts`
    },
    {
      label: "Games swing",
      value: signed(currentComponents.games - previousComponents.games),
      detail: `${currentComponents.games}/20 vs ${previousComponents.games}/20 pts`
    },
    { label: "Record", value: recordLabel(player), detail: `was ${recordLabel(previous)}` }
  ];
}

export function seasonImprovementDetail(
  player: PlayerSummary,
  previous: PlayerSummary,
  currentComponents: { rating: number; games: number; score: number },
  previousComponents: { rating: number; games: number; score: number }
) {
  return `${currentComponents.score}/100 this season vs ${previousComponents.score}/100 last season; ${recordLabel(player)} now, ${recordLabel(previous)} before`;
}

export function improvementStats(player: PlayerSummary): AwardStat[] {
  const start = player.periodMmrStart === null ? null : Math.round(Number(player.periodMmrStart || 0));
  const end = player.periodMmrEnd === null ? null : Math.round(Number(player.periodMmrEnd || 0));
  return [
    { label: "Games", value: String(player.matches_played || 0), detail: "qualified" },
    { label: "Start", value: start === null ? "-" : String(start), detail: "MMR" },
    { label: "Now", value: end === null ? "-" : String(end), detail: "MMR" },
    { label: "Gain", value: signed(Number(player.periodMmrChange || 0)), detail: "MMR" },
    { label: "Record", value: recordLabel(player), detail: "W-D-L" }
  ];
}

export function improvementDetail(player: PlayerSummary) {
  const start = player.periodMmrStart === null ? null : Math.round(Number(player.periodMmrStart || 0));
  const end = player.periodMmrEnd === null ? null : Math.round(Number(player.periodMmrEnd || 0));
  const movement = start !== null && end !== null ? `${start} to ${end} MMR` : `${signed(Number(player.periodMmrChange || 0))} MMR`;
  return `${player.matches_played} apps, ${movement}, ${recordLabel(player)}`;
}

export function minimumAwardMatches(matchCount: number) {
  if (matchCount <= 3) return 1;
  return Math.max(3, Math.ceil(matchCount * 0.25));
}

export function ratingPoints(rating: number) {
  const scaled = ((rating - 900) / 200) * 80;
  return Math.round(Math.max(0, Math.min(80, scaled)));
}

export function finiteOr(value: number | null | undefined, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function signed(value: number) {
  return `${value > 0 ? "+" : ""}${Math.round(value)}`;
}

/** Same award rules as Season Review, with an explicit historical cutoff.
 * The caller supplies the established award roster separately from historical lineups.
 */
export function seasonAwardSnapshot(players: Player[], matches: Match[], history: MmrHistory[], season: string) {
  const ids = new Set(matches.map(match => match.id));
  const scopedHistory = history.filter(row => row.match_id !== null && ids.has(row.match_id));
  const period = getPeriod(season, "full", matches);
  const currentMatches = matches.filter(match => inPeriod(match.date, period.start, period.end));
  const currentPlayers = playerSummariesForPeriod(players, currentMatches, scopedHistory, period.start, period.end, season);
  const previousPeriod = previousComparisonPeriod(season, "full", matches, period);
  const previousMatches = previousPeriod ? matches.filter(match => inPeriod(match.date, previousPeriod.start, previousPeriod.end)) : [];
  const previousPlayers = previousPeriod ? playerSummariesForPeriod(players, previousMatches, scopedHistory,
    previousPeriod.start, previousPeriod.end, previousPeriod.season) : [];
  return { mvp: mvpRace(currentPlayers, currentMatches),
    improved: improvedRace(currentPlayers, currentMatches, previousPlayers, previousMatches, previousPeriod, "full") };
}
