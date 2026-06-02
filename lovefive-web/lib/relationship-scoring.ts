export type ChemistryScoreInput = {
  matches: number;
  wins: number;
  draws: number;
  losses: number;
  goalDiff: number;
  goalGapTotal?: number;
  scoreSum?: number;
  actualRate?: number;
  expectedRate?: number;
};

export type RivalryScoreInput = {
  matches: number;
  winsA: number;
  winsB: number;
  draws: number;
  totalGoalGap: number;
};

export const CHEMISTRY_MIN_GAMES = 4;
export const CHEMISTRY_FULL_CONFIDENCE_GAMES = 10;

export function chemistryScoreFor(row: ChemistryScoreInput) {
  const matches = Math.max(0, row.matches);
  if (!matches) return 0;

  const actualRate = finiteOr(row.actualRate, finiteOr(row.scoreSum, row.wins + row.draws * 0.5) / matches);
  const expectedRate = finiteOr(row.expectedRate, NaN);
  const avgGoalGap = finiteOr(row.goalGapTotal, Math.abs(row.goalDiff)) / matches;
  const closeness = Math.max(0.55, 1 - Math.min(avgGoalGap, 10) / 10);

  if (Number.isFinite(expectedRate)) {
    if (matches < CHEMISTRY_MIN_GAMES) return 0;
    const confidence = chemistryEvidenceConfidence(matches);
    const residual = actualRate - clamp(expectedRate, 0, 1);
    return clamp(residual * 20 * closeness * confidence, 0, 6);
  }

  if (matches < CHEMISTRY_MIN_GAMES) return 0;

  const pointsRate = (row.wins * 3 + row.draws) / (matches * 3);
  const winRate = row.wins / matches;
  const goalDiffPerGame = clamp(row.goalDiff / matches, -3, 3);
  const evidence = chemistryEvidenceConfidence(matches);
  const volume = volumeWeight(matches, 9);
  const unbeatenBonus = row.losses === 0 ? 0.35 : 0;
  const goalDiffLift = Math.max(-0.6, goalDiffPerGame * 0.35);

  return clamp((pointsRate * 3 + winRate * 1.1 + volume + unbeatenBonus + goalDiffLift) * evidence * 1.2, 0, 6);
}

export function rivalryScoreFor(row: RivalryScoreInput) {
  const matches = Math.max(0, row.matches);
  if (!matches) return 0;

  const balance = 1 - Math.abs(row.winsA - row.winsB) / matches;
  const avgGoalGap = row.totalGoalGap / matches;
  const closeness = Math.max(0, 1 - Math.min(avgGoalGap, 6) / 6);
  const evidence = evidenceWeight(matches, 2);
  const volume = volumeWeight(matches, 10);
  const drawNudge = (row.draws / matches) * 0.55;
  const splitWinsBonus = row.winsA > 0 && row.winsB > 0 ? 0.5 : 0;

  return clamp((balance * 2.2 + closeness * 1.8 + volume * 1.2 + drawNudge + splitWinsBonus) * evidence * 1.15, 0, 6);
}

export function evidenceLabel(matches: number) {
  if (matches >= CHEMISTRY_FULL_CONFIDENCE_GAMES) return "Proven";
  if (matches >= CHEMISTRY_MIN_GAMES) return "Useful";
  if (matches >= 2) return "Building";
  return "One game";
}

export function chemistryEvidenceConfidence(matches: number) {
  if (matches < CHEMISTRY_MIN_GAMES) return 0;
  if (matches >= CHEMISTRY_FULL_CONFIDENCE_GAMES) return 1;
  return (matches - CHEMISTRY_MIN_GAMES + 1) / (CHEMISTRY_FULL_CONFIDENCE_GAMES - CHEMISTRY_MIN_GAMES + 1);
}

function finiteOr(value: unknown, fallback: number) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function evidenceWeight(matches: number, shrink: number) {
  return matches / (matches + shrink);
}

function volumeWeight(matches: number, cap: number) {
  return Math.min(1, Math.log(matches + 1) / Math.log(cap + 1));
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
