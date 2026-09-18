import type { BalancePrediction } from "./balance-model";

/** Plain-language display labels; the model's probabilities remain unchanged. */
export function matchdayOutlook(prediction: BalancePrediction) {
  const difference = prediction.win - prediction.loss;
  if (Math.abs(difference) < 0.01) return "No clear favourite";
  const team = difference > 0 ? "Blue team" : "Red team";
  return Math.abs(difference) < 0.1 ? `Slight edge: ${team}` : `${team} favoured`;
}
