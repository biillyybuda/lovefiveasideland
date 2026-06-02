import {
  displayName,
  normalizeName,
  resultFor,
  scoreParts,
  splitTeam,
  type Match,
  type Player
} from "./demo-data";
import { STARTING_MMR } from "./mmr-engine";

type PairStats = {
  matches: number;
  wins: number;
  draws: number;
  losses: number;
  goalDiffs: number[];
  scoreSum: number;
};

const EXPECTED_CHEMISTRY_MIN_GAMES = 4;
const EXPECTED_CHEMISTRY_FULL_GAMES = 10;
const BAD_PAIR_MIN_GAMES = 5;
const BAD_PAIR_FULL_GAMES = 12;

type EngineState = {
  matches: Match[];
  players: Player[];
  mmrMap: Map<string, number>;
  fitnessMap: Map<string, string>;
  formIndex: Map<string, number>;
  totalMatches: Map<string, number>;
  playerWinRate: Map<string, number>;
  baseChemistry: Map<string, number>;
  badPairPenalty: Map<string, number>;
  trioSynergy: Map<string, number>;
};

export type TeamOption = {
  teamA: Player[];
  teamB: Player[];
  avgA: number;
  avgB: number;
  diff: number;
  quality: number;
  styleGap: number;
  experienceGap: number;
  chemistryGap: number;
  badPairTotal: number;
  similarityPenalty: number;
  breakdown: Record<string, number | string | boolean | null>;
  ranking: {
    recommendationScore: number;
    predictedMargin: number;
    closePct: number;
  };
};

export function buildEngineState(players: Player[], matches: Match[]): EngineState {
  const mmrMap = new Map<string, number>();
  const fitnessMap = new Map<string, string>();
  const totalMatches = new Map<string, number>();
  const playerScore = new Map<string, number[]>();
  const formRows = new Map<string, Array<{ date: string; outcome: "W" | "D" | "L" }>>();

  for (const player of players) {
    const key = normalizeName(player.name);
    if (!key) continue;
    mmrMap.set(key, numberOr(player.mmr, STARTING_MMR));
    fitnessMap.set(key, player.fitness || "Medium");
    totalMatches.set(key, 0);
  }

  for (const match of matches) {
    const teamA = splitTeam(match.team_a).map(normalizeName).filter(Boolean);
    const teamB = splitTeam(match.team_b).map(normalizeName).filter(Boolean);

    for (const name of [...teamA, ...teamB]) {
      totalMatches.set(name, (totalMatches.get(name) || 0) + 1);
    }

    for (const [team, side] of [[teamA, "A"], [teamB, "B"]] as const) {
      for (const name of team) {
        const outcome = resultFor(match, side);
        const score = outcome === "W" ? 1 : outcome === "D" ? 0.5 : 0;
        const scores = playerScore.get(name) || [];
        scores.push(score);
        playerScore.set(name, scores);
        const rows = formRows.get(name) || [];
        rows.push({ date: String(match.date || ""), outcome });
        formRows.set(name, rows);
      }
    }
  }

  const playerWinRate = new Map<string, number>();
  const formIndex = new Map<string, number>();

  for (const key of mmrMap.keys()) {
    playerWinRate.set(key, average(playerScore.get(key) || [0.5]));
    const rows = (formRows.get(key) || []).sort((a, b) => a.date.localeCompare(b.date)).slice(-8);
    formIndex.set(key, average(rows.map((row) => row.outcome === "W" ? 1 : row.outcome === "D" ? 0.5 : 0)));
  }

  const { pairRows, baseChemistry, badPairPenalty } = buildPairState(matches, playerWinRate);
  const trioSynergy = buildTrioSynergy(matches, playerWinRate);

  return {
    matches,
    players,
    mmrMap,
    fitnessMap,
    formIndex,
    totalMatches,
    playerWinRate,
    baseChemistry,
    badPairPenalty,
    trioSynergy
  };
}

export function bestBalancedSplit(selected: Player[], locksA: string[] = [], locksB: string[] = [], matches: Match[] = []) {
  const total = selected.length;
  const teamSize = Math.floor(total / 2);
  const selectedKeys = selected.map((player) => normalizeName(player.name));
  const lockedA = new Set(locksA.map(normalizeName));
  const lockedB = new Set(locksB.map(normalizeName));
  const byKey = new Map(selected.map((player) => [normalizeName(player.name), player]));
  const state = buildEngineState(selected, matches);
  const candidates: TeamOption[] = [];

  function walk(index: number, teamAKeys: string[]) {
    if (teamAKeys.length > teamSize) return;
    if (index === selectedKeys.length) {
      if (teamAKeys.length !== teamSize) return;
      const teamASet = new Set(teamAKeys);
      for (const key of lockedA) if (!teamASet.has(key)) return;
      for (const key of lockedB) if (teamASet.has(key)) return;

      const teamA = teamAKeys.map((key) => byKey.get(key)!).filter(Boolean);
      const teamB = selectedKeys.filter((key) => !teamASet.has(key)).map((key) => byKey.get(key)!).filter(Boolean);
      const evaluation = evaluateTeams(teamA.map((player) => player.name), teamB.map((player) => player.name), state);
      const ranking = matchupRecommendationScore(evaluation.score, evaluation.breakdown);

      candidates.push({
        teamA,
        teamB,
        avgA: numberOr(evaluation.breakdown.mmr_avg_a, STARTING_MMR),
        avgB: numberOr(evaluation.breakdown.mmr_avg_b, STARTING_MMR),
        diff: numberOr(evaluation.breakdown.mmr_diff, 0),
        quality: ranking.recommendationScore,
        styleGap: numberOr(evaluation.breakdown.style_net, 0),
        experienceGap: numberOr(evaluation.breakdown.experience_gap, 0),
        chemistryGap: numberOr(evaluation.breakdown.chem_diff, 0),
        badPairTotal: numberOr(evaluation.breakdown.badpair_total, 0),
        similarityPenalty: numberOr(evaluation.breakdown.similarity_penalty, 0),
        breakdown: evaluation.breakdown,
        ranking
      });
      return;
    }

    walk(index + 1, [...teamAKeys, selectedKeys[index]]);
    walk(index + 1, teamAKeys);
  }

  walk(0, []);
  return rankGeneratedMatchups(candidates).slice(0, 20);
}

export function evaluateTeams(teamA: string[], teamB: string[], state: EngineState) {
  const aKeys = teamA.map(normalizeName).filter(Boolean);
  const bKeys = teamB.map(normalizeName).filter(Boolean);
  const effA = aKeys.map((name) => effectiveMmr(name, state));
  const effB = bKeys.map((name) => effectiveMmr(name, state));
  const mmrA = average(effA);
  const mmrB = average(effB);
  const mmrDiff = Math.abs(mmrA - mmrB);
  const spreadA = standardDeviation(effA);
  const spreadB = standardDeviation(effB);
  const spreadDiff = Math.abs(spreadA - spreadB);
  const chemA = teamChemistry(aKeys, state);
  const chemB = teamChemistry(bKeys, state);
  const trioA = teamTrioSynergy(aKeys, state);
  const trioB = teamTrioSynergy(bKeys, state);
  const badA = teamBadPairBadness(aKeys, state);
  const badB = teamBadPairBadness(bKeys, state);
  const style = teamStyleComponents(aKeys, bKeys, state);
  const similarityPenalty = similarityPenaltyForTeams(aKeys, bKeys, state);
  const expA = average(aKeys.map((name) => state.totalMatches.get(name) || 0));
  const expB = average(bKeys.map((name) => state.totalMatches.get(name) || 0));
  const experienceGap = Math.abs(expA - expB);

  const badTotalRaw = badA + badB;
  const badDiffRaw = Math.abs(badA - badB);
  const badSoftener = mmrDiff < 8 ? 0.85 : mmrDiff < 16 ? 0.95 : 1;
  const badTotal = Math.min(12, badTotalRaw * badSoftener);
  const badDiff = Math.min(10, badDiffRaw * badSoftener);
  const experiencePenalty = Math.max(0, experienceGap - 1.5) * 0.35;

  const score =
    mmrDiff
    + spreadDiff * 0.7
    + Math.abs(chemA.total - chemB.total) * 0.2
    + Math.abs(chemA.density - chemB.density) * 0.35
    + Math.abs(chemA.topShare - chemB.topShare) * 3
    + Math.max(0, Math.max(chemA.topShare, chemB.topShare) - 0.6) * 4
    + Math.abs(trioA.total - trioB.total) * 0.035
    + Math.abs(trioA.density - trioB.density) * 0.08
    + Math.abs(trioA.topShare - trioB.topShare) * 3
    + Math.max(0, Math.max(trioA.topShare, trioB.topShare) - 0.55) * 4
    + badTotal * 0.35
    + badDiff * 0.25
    + similarityPenalty
    + experiencePenalty
    + style.style_net;

  const breakdown = {
    mmr_avg_a: mmrA,
    mmr_avg_b: mmrB,
    mmr_diff: mmrDiff,
    spread_a: spreadA,
    spread_b: spreadB,
    spread_diff: spreadDiff,
    chem_a: chemA.total,
    chem_b: chemB.total,
    chem_diff: Math.abs(chemA.total - chemB.total),
    chem_density_a: chemA.density,
    chem_density_b: chemB.density,
    chem_evidence_a: chemA.evidenceLinks,
    chem_evidence_b: chemB.evidenceLinks,
    trio_a: trioA.total,
    trio_b: trioB.total,
    trio_diff: Math.abs(trioA.total - trioB.total),
    badpair_a: badA,
    badpair_b: badB,
    badpair_total: badTotal,
    badpair_diff: badDiff,
    badpair_total_raw: badTotalRaw,
    badpair_diff_raw: badDiffRaw,
    similarity_penalty: similarityPenalty,
    experience_gap: experienceGap,
    experience_penalty: experiencePenalty,
    ...style
  };

  return { score, breakdown };
}

export function rankGeneratedMatchups(candidates: TeamOption[]) {
  const seen = new Set<string>();
  const unique: TeamOption[] = [];

  for (const candidate of [...candidates]
    .sort((a, b) =>
      b.ranking.recommendationScore - a.ranking.recommendationScore
      || a.ranking.predictedMargin - b.ranking.predictedMargin
      || a.diff - b.diff
    )) {
    const key = canonicalSplitKey(candidate);
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(candidate);
  }

  return unique.map((option, index) => ({ ...option, rank: index + 1 }));
}

export function findBestTweaks(teamA: Player[], teamB: Player[], matches: Match[], maxSuggestions = 3) {
  const allPlayers = [...teamA, ...teamB];
  const state = buildEngineState(allPlayers, matches);
  const current = evaluateTeams(teamA.map((player) => player.name), teamB.map((player) => player.name), state);
  const currentRank = matchupRecommendationScore(current.score, current.breakdown);
  const suggestions: Array<{
    swapA: Player;
    swapB: Player;
    recommendationGain: number;
    scoreGain: number;
    reason: string;
  }> = [];

  for (const playerA of teamA) {
    for (const playerB of teamB) {
      const newA = teamA.map((player) => player.id === playerA.id ? playerB : player);
      const newB = teamB.map((player) => player.id === playerB.id ? playerA : player);
      const next = evaluateTeams(newA.map((player) => player.name), newB.map((player) => player.name), state);
      const nextRank = matchupRecommendationScore(next.score, next.breakdown);
      const recommendationGain = nextRank.recommendationScore - currentRank.recommendationScore;
      const scoreGain = current.score - next.score;
      if (recommendationGain < 1 && scoreGain < 0.08) continue;
      suggestions.push({
        swapA: playerA,
        swapB: playerB,
        recommendationGain,
        scoreGain,
        reason: explainImprovement(current.breakdown, next.breakdown)
      });
    }
  }

  return {
    currentScore: current.score,
    currentRecommendation: currentRank.recommendationScore,
    suggestions: suggestions
      .sort((a, b) => b.recommendationGain - a.recommendationGain || b.scoreGain - a.scoreGain)
      .slice(0, maxSuggestions)
  };
}

function matchupRecommendationScore(score: number, breakdown: Record<string, number | string | boolean | null>) {
  const mmrDiff = clamp(numberOr(breakdown.mmr_diff, 0), 0, 500);
  const spreadDiff = clamp(numberOr(breakdown.spread_diff, 0), 0, 500);
  const chemDiff = clamp(numberOr(breakdown.chem_diff, 0), 0, 500);
  const badTotal = clamp(numberOr(breakdown.badpair_total, 0), 0, 100);
  const simPen = clamp(numberOr(breakdown.similarity_penalty, 0), 0, 100);
  const experienceGap = clamp(numberOr(breakdown.experience_gap, 0), 0, 50);
  const styleNet = clamp(numberOr(breakdown.style_net, 0), 0, 20);
  const fallbackMargin = 1.05 + mmrDiff / 18 + spreadDiff / 75 + chemDiff / 22 + badTotal / 8 + simPen / 8 + Math.max(0, experienceGap - 1.5) / 8 + styleNet / 5;
  const predictedMargin = clamp(fallbackMargin, 0.4, 12);
  const closePct = clamp(94 - predictedMargin * 11 - mmrDiff * 0.2, 8, 94);
  const recommendationScore = clamp(closePct - Math.max(0, predictedMargin - 2) * 7.5 - Math.max(0, mmrDiff - 25) * 0.25 - score / 45, 0, 100);

  return {
    recommendationScore,
    predictedMargin,
    closePct
  };
}

function buildPairState(matches: Match[], playerWinRate: Map<string, number>) {
  const pairRows = new Map<string, PairStats>();
  const scoreMap = { W: 1, D: 0.5, L: 0 };

  for (const match of matches) {
    const score = scoreParts(match.score);
    const teams = [
      { side: "A" as const, names: splitTeam(match.team_a).map(normalizeName).filter(Boolean) },
      { side: "B" as const, names: splitTeam(match.team_b).map(normalizeName).filter(Boolean) }
    ];

    for (const team of teams) {
      const outcome = resultFor(match, team.side);
      for (let i = 0; i < team.names.length; i += 1) {
        for (let j = i + 1; j < team.names.length; j += 1) {
          const row = touchPair(pairRows, team.names[i], team.names[j]);
          row.matches += 1;
          row.wins += outcome === "W" ? 1 : 0;
          row.draws += outcome === "D" ? 1 : 0;
          row.losses += outcome === "L" ? 1 : 0;
          row.scoreSum += scoreMap[outcome];
          row.goalDiffs.push(score ? Math.abs(score[0] - score[1]) : 0);
        }
      }
    }
  }

  const baseChemistry = new Map<string, number>();
  const badPairPenalty = new Map<string, number>();

  for (const [key, row] of pairRows) {
    const [a, b] = key.split("|");
    const actualRate = row.scoreSum / Math.max(row.matches, 1);
    const expectedRate = (numberOr(playerWinRate.get(a), 0.5) + numberOr(playerWinRate.get(b), 0.5)) / 2;
    const residual = actualRate - expectedRate;
    const avgGd = average(row.goalDiffs);
    const evidence = evidenceConfidence(row.matches, EXPECTED_CHEMISTRY_MIN_GAMES, EXPECTED_CHEMISTRY_FULL_GAMES);
    const closeness = Math.max(0.55, 1 - avgGd / 10);
    const chemistry = evidence > 0 ? clamp(residual * 22 * closeness * evidence, -8, 8) : 0;
    baseChemistry.set(key, chemistry);

    const drop = expectedRate - actualRate;
    if (row.matches >= BAD_PAIR_MIN_GAMES && drop >= 0.16) {
      const sampleFactor = evidenceConfidence(row.matches, BAD_PAIR_MIN_GAMES, BAD_PAIR_FULL_GAMES);
      const heavyLossLift = Math.max(0, avgGd - 2) * 0.65;
      badPairPenalty.set(key, clamp((drop * 30 + heavyLossLift) * sampleFactor, 0, 12));
    }
  }

  return { pairRows, baseChemistry, badPairPenalty };
}

function buildTrioSynergy(matches: Match[], playerWinRate: Map<string, number>) {
  const games = new Map<string, number>();
  const scoreSums = new Map<string, number>();

  for (const match of matches) {
    const teams = [
      { side: "A" as const, names: splitTeam(match.team_a).map(normalizeName).filter(Boolean) },
      { side: "B" as const, names: splitTeam(match.team_b).map(normalizeName).filter(Boolean) }
    ];

    for (const team of teams) {
      const outcome = resultFor(match, team.side);
      const score = outcome === "W" ? 1 : outcome === "D" ? 0.5 : 0;
      for (const key of trioKeys(team.names)) {
        games.set(key, (games.get(key) || 0) + 1);
        scoreSums.set(key, (scoreSums.get(key) || 0) + score);
      }
    }
  }

  const synergy = new Map<string, number>();
  for (const [key, count] of games) {
    if (count < 8) continue;
    const names = key.split("|");
    const winrate = numberOr(scoreSums.get(key), 0) / count;
    const expected = average(names.map((name) => numberOr(playerWinRate.get(name), 0.5)));
    const sampleFactor = Math.min(1, count / 16);
    const value = (winrate - expected) * 35 * sampleFactor;
    if (Math.abs(value) >= 3) {
      synergy.set(key, clamp(value, -18, 18));
    }
  }

  return synergy;
}

function effectiveMmr(name: string, state: EngineState) {
  const base = numberOr(state.mmrMap.get(name), STARTING_MMR);
  const fitness = fitnessAdjust(state.fitnessMap.get(name));
  const form = numberOr(state.formIndex.get(name), 0.5);
  const matches = numberOr(state.totalMatches.get(name), 0);
  const scale = 0.25 + 0.75 * Math.min(1, matches / 12);
  return base + fitness + (form - 0.5) * 40 * scale;
}

function teamChemistry(team: string[], state: EngineState) {
  const values = pairKeys(team).map((key) => numberOr(state.baseChemistry.get(key), 0));
  const total = values.reduce((sum, value) => sum + value, 0);
  return {
    total,
    density: total / Math.max(values.length, 1),
    topShare: topShare(values),
    evidenceLinks: values.filter((value) => Math.abs(value) > 0).length
  };
}

function teamTrioSynergy(team: string[], state: EngineState) {
  const values = trioKeys(team).map((key) => numberOr(state.trioSynergy.get(key), 0));
  const total = values.reduce((sum, value) => sum + value, 0);
  return {
    total,
    density: total / Math.max(values.length, 1),
    topShare: topShare(values)
  };
}

function teamBadPairBadness(team: string[], state: EngineState) {
  return pairKeys(team).reduce((sum, key) => sum + numberOr(state.badPairPenalty.get(key), 0), 0);
}

function teamStyleComponents(teamA: string[], teamB: string[], state: EngineState) {
  return {
    style_finishing_diff: 0,
    style_creation_diff: 0,
    style_defence_diff: 0,
    style_creator_shortage: 0,
    style_finisher_shortage: 0,
    style_net: 0
  };
}

function similarityPenaltyForTeams(teamA: string[], teamB: string[], state: EngineState) {
  const aNow = new Set(teamA);
  const bNow = new Set(teamB);
  let penalty = 0;

  for (const match of state.matches) {
    const score = scoreParts(match.score);
    if (!score) continue;
    const histA = new Set(splitTeam(match.team_a).map(normalizeName).filter(Boolean));
    const histB = new Set(splitTeam(match.team_b).map(normalizeName).filter(Boolean));
    const same = countOverlap(aNow, histA) + countOverlap(bNow, histB);
    const swapped = countOverlap(aNow, histB) + countOverlap(bNow, histA);
    const similarity = Math.max(same, swapped) / Math.max(teamA.length + teamB.length, 1);
    const goalDiff = Math.abs(score[0] - score[1]);
    if (similarity >= 0.75 && goalDiff >= 4) {
      const similarityLift = (similarity - 0.75) / 0.25;
      const marginLift = Math.min(2, 1 + (goalDiff - 4) * 0.18);
      penalty = Math.max(penalty, (2 + similarityLift * 6) * marginLift);
    }
  }

  return penalty;
}

function explainImprovement(before: Record<string, number | string | boolean | null>, after: Record<string, number | string | boolean | null>) {
  const reasons: string[] = [];
  for (const [label, key, threshold] of [
    ["MMR gap", "mmr_diff", 1],
    ["team-shape gap", "spread_diff", 1],
    ["chemistry gap", "chem_diff", 1.5],
    ["matchup-memory risk", "similarity_penalty", 0.5]
  ] as const) {
    if (numberOr(before[key], 0) - numberOr(after[key], 0) >= threshold) {
      reasons.push(`lowers the ${label}`);
    }
  }
  return reasons.slice(0, 3).join(", ") || "nudges the overall game balance up";
}

function pairKeys(team: string[]) {
  const keys: string[] = [];
  for (let i = 0; i < team.length; i += 1) {
    for (let j = i + 1; j < team.length; j += 1) {
      keys.push(pairKey(team[i], team[j]));
    }
  }
  return keys;
}

function trioKeys(team: string[]) {
  const keys: string[] = [];
  for (let i = 0; i < team.length; i += 1) {
    for (let j = i + 1; j < team.length; j += 1) {
      for (let k = j + 1; k < team.length; k += 1) {
        keys.push([team[i], team[j], team[k]].sort().join("|"));
      }
    }
  }
  return keys;
}

function touchPair(map: Map<string, PairStats>, a: string, b: string) {
  const key = pairKey(a, b);
  const existing = map.get(key) || { matches: 0, wins: 0, draws: 0, losses: 0, goalDiffs: [], scoreSum: 0 };
  map.set(key, existing);
  return existing;
}

function pairKey(a: string, b: string) {
  return [a, b].map(normalizeName).sort().join("|");
}

function canonicalSplitKey(option: TeamOption) {
  const a = option.teamA.map((player) => normalizeName(player.name)).sort().join(",");
  const b = option.teamB.map((player) => normalizeName(player.name)).sort().join(",");
  return [a, b].sort().join("::");
}

function evidenceConfidence(matches: number, minGames: number, fullGames: number) {
  if (matches < minGames) return 0;
  if (matches >= fullGames) return 1;
  return (matches - minGames + 1) / Math.max(fullGames - minGames + 1, 1);
}

function topShare(values: number[]) {
  const total = values.reduce((sum, value) => sum + Math.abs(value), 0);
  if (total <= 0) return 0;
  return Math.max(...values.map(Math.abs)) / total;
}

function fitnessAdjust(label: string | null | undefined) {
  const value = normalizeName(label);
  if (value === "high" || value === "excellent") return 10;
  if (value === "low") return -10;
  return 0;
}

function countOverlap(a: Set<string>, b: Set<string>) {
  let count = 0;
  for (const value of a) {
    if (b.has(value)) count += 1;
  }
  return count;
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function standardDeviation(values: number[]) {
  const avg = average(values);
  return Math.sqrt(average(values.map((value) => (value - avg) ** 2)));
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function numberOr(value: unknown, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function playerLabel(player: Player) {
  return displayName(player);
}
