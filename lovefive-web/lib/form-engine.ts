import type { PlayerSummary } from "./demo-data";

const RECENCY_WEIGHTS = [6, 5, 4, 3, 2, 1];

export function formResultValue(result: string) {
  if (result === "W") return 3;
  if (result === "D") return 1;
  return 0;
}

export function weightedFormScore(player: Pick<PlayerSummary, "form">) {
  return player.form.slice(0, RECENCY_WEIGHTS.length).reduce((sum, result, index) => {
    return sum + formResultValue(result) * RECENCY_WEIGHTS[index];
  }, 0);
}

export function currentWinStreak(player: Pick<PlayerSummary, "form">) {
  let count = 0;
  for (const result of player.form) {
    if (result !== "W") break;
    count += 1;
  }
  return count;
}

export function compareFormPlayers(a: PlayerSummary, b: PlayerSummary) {
  return weightedFormScore(b) - weightedFormScore(a)
    || currentWinStreak(b) - currentWinStreak(a)
    || b.goalDiff - a.goalDiff
    || Number(b.mmr || 0) - Number(a.mmr || 0)
    || a.label.localeCompare(b.label);
}
