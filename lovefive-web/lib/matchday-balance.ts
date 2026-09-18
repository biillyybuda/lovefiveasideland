import type { Match, Player } from "./demo-data";
import { advanceBalance, balanceRatings, matchDay, replayBalance, selectBalanceTeams } from "./balance-model";

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
