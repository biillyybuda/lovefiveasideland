import { scoreParts, type Match, type Player } from "./demo-data";

// Season awards are a competition rule, independent of changes to team selection
// or a future public rating model. Change this version only for a deliberate
// award-rule change, with a before/after season audit.
export const AWARD_RATING_VERSION = "season-elo-v1";
export const STARTING_MMR = 1000;

/** Frozen calculation used by the existing MVP / Most Improved races. */
export function calculateAwardMmrUpdates({ teamA, teamB, match, players }: {
  teamA: Player[]; teamB: Player[]; match: Pick<Match, "score" | "result">; players: Player[];
}) {
  const average = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
  const numberOr = (value: number | null) => Number.isFinite(Number(value)) ? Number(value) : STARTING_MMR;
  const leagueMean = average(players.map(player => numberOr(player.mmr)));
  const avgA = average(teamA.map(player => numberOr(player.mmr)));
  const avgB = average(teamB.map(player => numberOr(player.mmr)));
  const score = scoreParts(match.score);
  const result = score ? score[0] > score[1] ? "A" : score[0] < score[1] ? "B" : "DRAW" : String(match.result || "").trim().toUpperCase();
  if (!["A", "B", "DRAW"].includes(result)) throw new Error("A completed match needs a score or result.");
  const update = (player: Player, side: "A" | "B", teamAverage: number, opponentAverage: number) => {
    const before = numberOr(player.mmr);
    const expected = 1 / (1 + 10 ** ((opponentAverage - before) / 400));
    const gap = Math.abs(teamAverage - opponentAverage);
    const imbalance = teamAverage < opponentAverage ? 1 + Math.min(0.5, gap / 300) : 1 - Math.min(0.4, gap / 300);
    const volatility = Math.max(0.5, Math.min(1, 1 - 0.5 * ((before - leagueMean) / 1000)));
    const kFinal = 30 * imbalance * volatility;
    const actual = result === side ? 1 : result === "DRAW" ? 0.5 : 0;
    return { player, after: before + kFinal * (actual - expected) };
  };
  return [...teamA.map(player => update(player, "A", avgA, avgB)), ...teamB.map(player => update(player, "B", avgB, avgA))];
}
