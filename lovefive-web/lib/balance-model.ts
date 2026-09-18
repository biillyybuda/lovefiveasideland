import { normalizeName, scoreParts, splitTeam, type Match, type Player } from "./demo-data";

// Frozen settings from the chronological study. Changing these requires a new version.
export const BALANCE_VERSION = "margin-v1";
export const BALANCE_CONFIG = Object.freeze({ priorSd: 1, noiseSd: 4.266145801540309, dynamics: 0.04, closeGoals: 2 });
export type BalanceState = {
  version: string;
  playerIds: number[];
  mu: number[];
  covariance: number[][];
  appearances: number[];
  day: number | null;
};
export type BalancePrediction = {
  meanMargin: number;
  skillVariance: number;
  predictiveSd: number;
  win: number;
  draw: number;
  loss: number;
  expected: number;
  close: number;
  absoluteMargin: number;
};
export type BalanceRecord = { matchId: number; date: string; teamA: number[]; teamB: number[]; prediction: BalancePrediction };
export type BalanceOption = { teamA: Player[]; teamB: Player[]; prediction: BalancePrediction; roleGap: number; repeatCost: number; key: string };

export function createBalanceState(players: Player[]): BalanceState {
  const ids = players.map(p => p.id).sort((a, b) => a - b);
  if (new Set(ids).size !== ids.length) throw new Error("Duplicate player IDs.");
  return { version: BALANCE_VERSION, playerIds: ids, mu: ids.map(() => 0),
    covariance: ids.map((_, i) => ids.map((__, j) => i === j ? BALANCE_CONFIG.priorSd ** 2 : 0)),
    appearances: ids.map(() => 0), day: null };
}

export function matchDay(date: string) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error("Choose a valid match date.");
  const time = Date.parse(`${date}T00:00:00Z`);
  if (!Number.isFinite(time) || new Date(time).toISOString().slice(0, 10) !== date) throw new Error("Choose a valid match date.");
  return time / 86400000;
}

export function advanceBalance(state: BalanceState, date: string) {
  const day = matchDay(date);
  if (state.day !== null && day < state.day) throw new Error("Replay matches in date order.");
  const variance = state.day === null ? 0 : BALANCE_CONFIG.dynamics ** 2 * (day - state.day) / 7;
  state.covariance.forEach((row, i) => { row[i] += variance; });
  state.day = day;
}

function contrast(state: BalanceState, a: number[], b: number[]) {
  if (a.length !== 5 || b.length !== 5 || new Set([...a, ...b]).size !== 10) throw new Error("Choose five different players on each side.");
  if ([...a, ...b].some(id => !state.playerIds.includes(id))) throw new Error("A selected player is missing from the model.");
  return state.playerIds.map(id => a.includes(id) ? 1 : b.includes(id) ? -1 : 0);
}

// Normal CDF, maximum approximation error approximately 8e-8.
function cdf(z: number) {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const tail = Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI) * t *
    (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return z >= 0 ? 1 - tail : tail;
}
const dot = (a: number[], b: number[]) => a.reduce((sum, v, i) => sum + v * b[i], 0);

export function predictBalance(state: BalanceState, a: number[], b: number[]): BalancePrediction {
  const x = contrast(state, a, b);
  const m = dot(x, state.mu);
  const skillVariance = Math.max(0, dot(x, state.covariance.map(row => dot(row, x))));
  const s = Math.sqrt(skillVariance + BALANCE_CONFIG.noiseSd ** 2);
  const win = cdf((m - 0.5) / s);
  const loss = cdf((-m - 0.5) / s);
  const draw = Math.max(0, 1 - win - loss);
  const boundary = BALANCE_CONFIG.closeGoals + 0.5;
  return { meanMargin: m, skillVariance, predictiveSd: s, win, draw, loss, expected: win + draw / 2,
    close: cdf((boundary - m) / s) - cdf((-boundary - m) / s),
    absoluteMargin: s * Math.sqrt(2 / Math.PI) * Math.exp(-m * m / (2 * s * s)) + m * (2 * cdf(m / s) - 1) };
}

export function updateBalance(state: BalanceState, a: number[], b: number[], goalDifference: number) {
  if (!Number.isInteger(goalDifference) || Math.abs(goalDifference) > 999) throw new Error("Invalid goal difference.");
  const x = contrast(state, a, b);
  const gainNumerator = state.covariance.map(row => dot(row, x));
  const residual = goalDifference - dot(x, state.mu);
  const noise = BALANCE_CONFIG.noiseSd;
  const variance = dot(x, gainNumerator) + noise ** 2 * Math.max(1, Math.abs(residual) / (2 * noise));
  state.mu = state.mu.map((v, i) => v + gainNumerator[i] / variance * residual);
  state.covariance = state.covariance.map((row, i) => row.map((v, j) => v - gainNumerator[i] * gainNumerator[j] / variance));
  state.appearances = state.appearances.map((v, i) => v + (x[i] === 0 ? 0 : 1));
}

export function resolveLineup(match: Match, players: Player[]) {
  const lookup = new Map<string, number>();
  // Canonical names take precedence over another player's display name.
  for (const p of players) {
    const key = normalizeName(p.name);
    if (lookup.has(key)) throw new Error(`Duplicate player name: ${p.name}.`);
    lookup.set(key, p.id);
  }
  for (const p of players) {
    const key = normalizeName(p.display_name);
    if (key && !lookup.has(key)) lookup.set(key, p.id);
  }
  const resolve = (text: string | null) => splitTeam(text).map(name => {
    const id = lookup.get(normalizeName(name));
    if (id === undefined) throw new Error(`Match ${match.id}: player ${name} cannot be found.`);
    return id;
  });
  const a = resolve(match.team_a), b = resolve(match.team_b);
  if (a.length !== 5 || b.length !== 5 || new Set([...a, ...b]).size !== 10) throw new Error(`Match ${match.id} needs ten distinct players, five on each side.`);
  return { a, b };
}

export function replayBalance(players: Player[], matches: Match[]) {
  const state = createBalanceState(players);
  const records: BalanceRecord[] = [];
  for (const match of [...matches].filter(m => m.processed === 1).sort(compareMatches)) {
    const goals = scoreParts(match.score);
    if (!goals) throw new Error(`Match ${match.id} has an invalid score.`);
    const { a, b } = resolveLineup(match, players);
    advanceBalance(state, match.date || "");
    records.push({ matchId: match.id, date: match.date!, teamA: a, teamB: b, prediction: predictBalance(state, a, b) });
    updateBalance(state, a, b, goals[0] - goals[1]);
  }
  return { state, records };
}
export function compareMatches(a: Pick<Match, "id" | "date">, b: Pick<Match, "id" | "date">) {
  return String(a.date || "").localeCompare(String(b.date || "")) || a.id - b.id;
}

export function balanceRatings(state: BalanceState) {
  const n = state.mu.length;
  const mean = state.mu.reduce((a, b) => a + b, 0) / Math.max(n, 1);
  const rowSums = state.covariance.map(row => row.reduce((a, b) => a + b, 0));
  const total = rowSums.reduce((a, b) => a + b, 0);
  return state.playerIds.map((id, i) => ({ playerId: id, rating: 1000 + 100 * (state.mu[i] - mean),
    uncertainty: 100 * Math.sqrt(Math.max(0, state.covariance[i][i] - 2 * rowSums[i] / n + total / (n * n))),
    appearances: state.appearances[i] }));
}

/** Roles/variety only break near ties; they never manufacture a probability. */
export function selectBalanceTeams(state: BalanceState, selected: Player[], locksA: number[] = [], locksB: number[] = [], matches: Match[] = []): BalanceOption[] {
  if (selected.length !== 10 || new Set(selected.map(p => p.id)).size !== 10) return [];
  const ids = new Set(selected.map(p => p.id));
  if ([...locksA, ...locksB].some(id => !ids.has(id)) || locksA.some(id => locksB.includes(id)) || locksA.length > 5 || locksB.length > 5) return [];
  const options: BalanceOption[] = [];
  const seen = new Set<string>();
  const roleCount = (team: Player[], role: string) => team.filter(p => String(p.position || "").toLowerCase() === role).length;
  const recent = [...matches].filter(m => m.processed === 1).sort(compareMatches).slice(-8);
  function walk(index: number, a: Player[]) {
    if (a.length > 5 || a.length + selected.length - index < 5) return;
    if (index < selected.length) { walk(index + 1, [...a, selected[index]]); walk(index + 1, a); return; }
    const aIds = a.map(p => p.id), b = selected.filter(p => !aIds.includes(p.id)), bIds = b.map(p => p.id);
    if (locksA.some(id => !aIds.includes(id)) || locksB.some(id => !bIds.includes(id))) return;
    const key = [aIds.slice().sort((x, y) => x - y).join(","), bIds.slice().sort((x, y) => x - y).join(",")].sort().join("|");
    if (seen.has(key)) return;
    seen.add(key);
    const roleGap = ["goalkeeper", "defender", "midfielder", "forward"].reduce((sum, role) => sum + Math.abs(roleCount(a, role) - roleCount(b, role)), 0);
    const aNames = new Set(a.map(p => normalizeName(p.name))), bNames = new Set(b.map(p => normalizeName(p.name)));
    let repeatCost = 0;
    recent.forEach((m, i) => {
      const ma = splitTeam(m.team_a).map(normalizeName), mb = splitTeam(m.team_b).map(normalizeName);
      const overlap = Math.max(ma.filter(n => aNames.has(n)).length + mb.filter(n => bNames.has(n)).length,
        ma.filter(n => bNames.has(n)).length + mb.filter(n => aNames.has(n)).length);
      repeatCost += Math.max(0, overlap - 7) * (i + 1) / Math.max(recent.length, 1);
    });
    options.push({ teamA: a, teamB: b, prediction: predictBalance(state, aIds, bIds), roleGap, repeatCost, key });
  }
  walk(0, []);
  const best = Math.max(0, ...options.map(o => o.prediction.close));
  // Fixed bands avoid non-transitive pairwise "within 1%" comparators.
  return options.sort((a, b) => Math.floor((best - a.prediction.close) / 0.01) - Math.floor((best - b.prediction.close) / 0.01)
    || a.roleGap - b.roleGap || a.repeatCost - b.repeatCost || b.prediction.close - a.prediction.close || a.key.localeCompare(b.key));
}
