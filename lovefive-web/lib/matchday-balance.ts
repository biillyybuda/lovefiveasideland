import type { Match, Player } from "./demo-data";
import { advanceBalance, balanceRatings, matchDay, replayBalance, selectBalanceTeams, type BalanceOption } from "./balance-model";

/** The selector is read-only. Public MMR and award history are never its output. */
export function buildMatchdayBalance(players: Player[], matches: Match[], date: string) {
  matchDay(date);
  // A past date must not borrow results (or team-variety information) from the future.
  const history = matches.filter(match => match.processed === 1 && String(match.date || "") <= date);
  const { state } = replayBalance(players, history);
  advanceBalance(state, date);
  return { state, history, ratings: balanceRatings(state) };
}

export function matchdayOptions(model: ReturnType<typeof buildMatchdayBalance>, selected: Player[], locksA: number[], locksB: number[]) {
  return selectBalanceTeams(model.state, [...selected].sort((a, b) => a.id - b.id), locksA, locksB, model.history);
}

/** Draw colours and display order after ranking, without changing any teammates. */
export function drawMatchdayOptions(options: BalanceOption[], locksA: number[], locksB: number[], seed: number): BalanceOption[] {
  return options.map(option => {
    // Key each draw by the split so rerenders and date-driven ranking changes
    // keep the same colours and order until the user explicitly reshuffles.
    const random = splitRandom(seed, option.key);
    const swap = random() < 0.5 && locksA.length === 0 && locksB.length === 0;
    const prediction = option.prediction;
    return {
      ...option,
      teamA: shuffled(swap ? option.teamB : option.teamA, random),
      teamB: shuffled(swap ? option.teamA : option.teamB, random),
      prediction: swap ? {
        ...prediction,
        meanMargin: -prediction.meanMargin || 0,
        win: prediction.loss,
        loss: prediction.win,
        expected: prediction.loss + prediction.draw / 2
      } : { ...prediction }
    };
  });
}

function shuffled<T>(items: T[], random: () => number): T[] {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function splitRandom(seed: number, key: string): () => number {
  let state = seed >>> 0;
  for (const char of key) state = Math.imul(state ^ char.charCodeAt(0), 16777619) >>> 0;
  // Mulberry32 gives a reproducible draw from the session's random seed.
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
