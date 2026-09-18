"use client";

import { useMemo, useState } from "react";
import {
  formatUkDate,
  makeNameMap,
  normalizeName,
  resultFor,
  scoreParts,
  splitTeam,
  type Match,
  type Player
} from "@/lib/demo-data";
import { bestBalancedSplit, findBestTweaks, type TeamOption } from "@/lib/team-ai-engine";
import { BalanceComparison } from "@/components/balance-comparison";
import { BalancedMatchday } from "@/components/balanced-matchday";
import { LineupEditor } from "@/components/lineup-editor";
import type { LineupSeed } from "@/lib/lineup-editor";

type MatchdayOption = TeamOption & { rank?: number };
type SmartOption = {
  option: MatchdayOption;
  label: string;
  index: number;
};
type PlayerRecord = {
  played: number;
  wins: number;
  draws: number;
  losses: number;
  goalsFor: number;
  goalsAgainst: number;
  form: string[];
  streak: string;
};
type PairRecord = {
  a: string;
  b: string;
  matches: number;
  winsA: number;
  winsB: number;
  draws: number;
  goalDiff: number;
  goalDiffTotal: number;
  scoreSum: number;
  actualRate: number;
  expectedRate: number;
  residual: number;
  score: number;
};

const EXPECTED_CHEMISTRY_MIN_GAMES = 4;
const EXPECTED_CHEMISTRY_FULL_GAMES = 10;

function playerLabel(player: Player) {
  return player.display_name || player.name;
}

function splitKey(option: MatchdayOption) {
  const a = option.teamA.map((player) => normalizeName(player.name)).sort().join(",");
  const b = option.teamB.map((player) => normalizeName(player.name)).sort().join(",");
  return [a, b].sort().join("::");
}

function breakdownNumber(option: MatchdayOption, key: string, fallback = 0) {
  const value = Number(option.breakdown[key]);
  return Number.isFinite(value) ? value : fallback;
}

function hasShortlistRisk(option: MatchdayOption) {
  const rotationPenalty = breakdownNumber(option, "rotating_keeper_penalty");
  const stylePenalty = breakdownNumber(option, "style_net");
  return (
    option.similarityPenalty >= 7
    || option.badPairTotal >= 9
    || rotationPenalty >= 9
    || stylePenalty >= 3.5
    || option.diff >= 45
    || option.ranking.predictedMargin >= 6.5
  );
}

function keeperRotationScore(option: MatchdayOption) {
  return (
    breakdownNumber(option, "rotating_keeper_penalty")
    + breakdownNumber(option, "rotation_floor_gap") / 12
    + option.ranking.predictedMargin / 8
  );
}

function profileBalanceScore(option: MatchdayOption) {
  if (!option.breakdown.style_enabled) return Number.POSITIVE_INFINITY;
  return breakdownNumber(option, "style_net") + option.ranking.predictedMargin / 10;
}

function buildSmartOptions(options: MatchdayOption[]) {
  const picked: SmartOption[] = [];
  const seen = new Set<string>();
  const maxShortlistRank = Math.max(24, Math.ceil(options.length * 0.5));
  const isShortlistSafe = (option: MatchdayOption, index: number) => index < maxShortlistRank && !hasShortlistRisk(option);

  function add(label: string, index: number, allowRisk = false) {
    const option = options[index];
    if (!option) return;
    if (!allowRisk && !isShortlistSafe(option, index)) return;
    const key = splitKey(option);
    if (seen.has(key)) return;
    picked.push({ option, label, index });
    seen.add(key);
  }

  const bestOverall = bestIndex(options, (option) => option.ranking.recommendationScore, "desc", isShortlistSafe);
  add("Best overall", bestOverall >= 0 ? bestOverall : 0, bestOverall < 0);
  add("Closest expected game", bestIndex(options, (option) => option.ranking.predictedMargin, "asc", isShortlistSafe));
  add("Best profile balance", bestIndex(
    options,
    profileBalanceScore,
    "asc",
    (option, index) => isShortlistSafe(option, index) && Boolean(option.breakdown.style_enabled)
  ));
  add("Best keeper rotation", bestIndex(options, keeperRotationScore, "asc", isShortlistSafe));
  add("Most even ratings", bestIndex(options, (option) => option.diff, "asc", isShortlistSafe));
  add("Highest chemistry balance", bestIndex(options, (option) => option.chemistryGap, "asc", isShortlistSafe));

  for (let index = 0; picked.length < 4 && index < options.length; index += 1) {
    add("Strong alternative", index);
  }

  if (!picked.length) add("Best available", 0, true);

  return picked.slice(0, 4);
}

function bestIndex(
  options: MatchdayOption[],
  getter: (option: MatchdayOption) => number,
  direction: "asc" | "desc",
  predicate: (option: MatchdayOption, index: number) => boolean = () => true
) {
  if (!options.length) return -1;
  return options.reduce((best, option, index) => {
    if (!predicate(option, index)) return best;
    if (best < 0) return index;
    const current = getter(option);
    const previous = getter(options[best]);
    return direction === "asc" ? current < previous ? index : best : current > previous ? index : best;
  }, -1);
}

function optionExplanation(option: MatchdayOption) {
  const positives: string[] = [];
  const risks: string[] = [];
  const closePct = option.ranking.closePct;
  const rotationPenalty = breakdownNumber(option, "rotating_keeper_penalty");
  const styleNet = breakdownNumber(option, "style_net");
  const hasProfiles = Boolean(option.breakdown.style_enabled);

  if (option.diff <= 10) positives.push("Ratings are extremely tight, so the game should not need one side to overperform.");
  else if (option.diff <= 25) positives.push("MMR balance is healthy enough for a competitive game.");
  else risks.push("The MMR gap is noticeable, so one team may start with a small edge.");

  if (closePct >= 70) positives.push("The balance rules give this matchup a strong closeness score.");
  else if (closePct < 45) risks.push("The closeness score is lower than ideal.");

  const saveRotationPenalty = breakdownNumber(option, "rotation_save_penalty");
  if (rotationPenalty <= 3) positives.push("Rotating-keeper burden is shared evenly when everyone takes a turn.");
  else if (rotationPenalty >= 7) risks.push("One team has a weaker floor during keeper rotations.");
  if (saveRotationPenalty >= 1.2) risks.push("Save-impact history gives one team a stronger rotating-keeper floor.");

  if (hasProfiles && styleNet <= 1.1) positives.push("Goal, assist and save profiles are balanced across the two teams.");
  else if (hasProfiles && styleNet >= 2.5) risks.push("One side has the stronger rate-based goal, assist and save profile mix.");

  if (option.chemistryGap <= 4) positives.push("Proven chemistry is spread fairly evenly once expectation is accounted for.");
  else risks.push("One side has the stronger expected-vs-actual chemistry edge.");

  if (option.experienceGap <= 2) positives.push("Experience is shared fairly evenly, so one side is not carrying all the regulars.");
  else risks.push("One team has noticeably more matchday experience.");

  if (option.similarityPenalty <= 1) positives.push("No strong warning from similar heavy past results.");
  else risks.push("Similar historic lineups produced a lopsided result before.");

  return {
    positives: positives.slice(0, 3),
    risks: risks.length ? risks.slice(0, 3) : ["No major warning signs in the current data."]
  };
}

function recommendationDetails(option: MatchdayOption, label: string, options: MatchdayOption[]) {
  const tolerance = 0.05;
  const bestScore = Math.max(...options.map((row) => row.ranking.recommendationScore));
  const bestClose = Math.max(...options.map((row) => row.ranking.closePct));
  const bestMargin = Math.min(...options.map((row) => row.ranking.predictedMargin));
  const bestMmrGap = Math.min(...options.map((row) => row.diff));
  const bestChemGap = Math.min(...options.map((row) => row.chemistryGap));
  const bestRepeatRisk = Math.min(...options.map((row) => row.similarityPenalty));
  const bestKeeperRotation = Math.min(...options.map((row) => breakdownNumber(row, "rotating_keeper_penalty")));
  const profileOptions = options.filter((row) => Boolean(row.breakdown.style_enabled));
  const bestProfileBalance = profileOptions.length ? Math.min(...profileOptions.map((row) => breakdownNumber(row, "style_net"))) : null;
  const chips: string[] = [];

  if (option.ranking.recommendationScore >= bestScore - tolerance) chips.push("Best combined score");
  if (option.ranking.closePct >= bestClose - tolerance) chips.push("Highest closeness score");
  if (option.ranking.predictedMargin <= bestMargin + tolerance) chips.push("Lowest expected margin");
  if (option.diff <= bestMmrGap + tolerance) chips.push("Most even ratings");
  if (option.chemistryGap <= bestChemGap + tolerance) chips.push("Best chemistry balance");
  if (option.similarityPenalty <= bestRepeatRisk + tolerance) chips.push("Lowest repeat risk");
  if (breakdownNumber(option, "rotating_keeper_penalty") <= bestKeeperRotation + tolerance) chips.push("Best keeper rotation");
  if (bestProfileBalance !== null && Boolean(option.breakdown.style_enabled) && breakdownNumber(option, "style_net") <= bestProfileBalance + tolerance) chips.push("Best profile balance");

  const uniqueChips = [...new Set(chips)].slice(0, 4);

  if (label === "Most even ratings") {
    return {
      chips: uniqueChips.length ? uniqueChips : ["Most even ratings", `${fmt(option.diff, 1)} MMR gap`],
      summary: "The pure ratings split: average MMR is almost level, even if the wider model prefers another matchup."
    };
  }

  if (label === "Highest chemistry balance") {
    return {
      chips: uniqueChips.length ? uniqueChips : ["Best chemistry balance", `${fmt(option.chemistryGap, 1)} chemistry gap`],
      summary: "Built to spread proven teammate links evenly so neither side gets the stronger relationship base."
    };
  }

  if (label === "Closest expected game") {
    return {
      chips: uniqueChips.length ? uniqueChips : ["Lowest expected margin", `${fmt(option.ranking.closePct)}/100 closeness score`],
      summary: "The model expects this one to stay closest on the scoreboard."
    };
  }

  if (label === "Best profile balance") {
    return {
      chips: uniqueChips.length ? uniqueChips : ["Best profile balance", `${fmt(breakdownNumber(option, "style_net"), 1)} profile mix`],
      summary: "Best spread of per-game goal, assist, save and clutch profiles across the two teams."
    };
  }

  if (label === "Best keeper rotation") {
    return {
      chips: uniqueChips.length ? uniqueChips : ["Best keeper rotation", `${fmt(breakdownNumber(option, "rotating_keeper_penalty"), 1)} rotation risk`],
      summary: "Best at sharing the rotating-keeper burden across rating floor, experience and save-impact profiles."
    };
  }

  return {
    chips: uniqueChips.length ? uniqueChips : ["Best overall model score", `${fmt(option.ranking.closePct)}/100 closeness score`],
    summary: "The best all-round blend of rating balance, projected margin, player profiles, proven chemistry, keeper rotation and repeat-risk checks."
  };
}

function buildPlayerRecords(players: Player[], matches: Match[]) {
  const records = new Map<string, PlayerRecord>();
  const sortedMatches = [...matches].sort((a, b) => String(b.date || "").localeCompare(String(a.date || "")));
  for (const player of players) {
    records.set(normalizeName(player.name), {
      played: 0,
      wins: 0,
      draws: 0,
      losses: 0,
      goalsFor: 0,
      goalsAgainst: 0,
      form: [],
      streak: ""
    });
  }

  for (const match of sortedMatches) {
    const score = scoreParts(match.score);
    if (!score) continue;

    for (const side of ["A", "B"] as const) {
      const team = splitTeam(side === "A" ? match.team_a : match.team_b).map(normalizeName);
      const result = resultFor(match, side);
      for (const key of team) {
        const row = records.get(key);
        if (!row) continue;
        row.played += 1;
        row.wins += result === "W" ? 1 : 0;
        row.draws += result === "D" ? 1 : 0;
        row.losses += result === "L" ? 1 : 0;
        row.goalsFor += side === "A" ? score[0] : score[1];
        row.goalsAgainst += side === "A" ? score[1] : score[0];
        row.form.push(result);
      }
    }
  }

  for (const row of records.values()) {
    row.form = row.form.slice(0, 5);
    const first = row.form[0];
    if (first === "W" || first === "L") {
      const count = row.form.findIndex((value) => value !== first);
      const streakCount = count === -1 ? row.form.length : count;
      row.streak = streakCount >= 2 ? `${first}${streakCount}` : "";
    }
  }

  return records;
}

function pairKey(a: string, b: string) {
  return [normalizeName(a), normalizeName(b)].sort().join("|");
}

function buildPairRecords(matches: Match[]) {
  const teammate = new Map<string, PairRecord>();
  const rivals = new Map<string, PairRecord>();
  const playerScore = new Map<string, number[]>();

  for (const match of matches) {
    const score = scoreParts(match.score);
    if (!score) continue;
    for (const side of ["A", "B"] as const) {
      const result = resultFor(match, side);
      const value = result === "W" ? 1 : result === "D" ? 0.5 : 0;
      for (const name of splitTeam(side === "A" ? match.team_a : match.team_b).map(normalizeName)) {
        const rows = playerScore.get(name) || [];
        rows.push(value);
        playerScore.set(name, rows);
      }
    }
  }

  function touch(map: Map<string, PairRecord>, a: string, b: string) {
    const key = pairKey(a, b);
    const [first, second] = key.split("|");
    const row = map.get(key) || {
      a: first,
      b: second,
      matches: 0,
      winsA: 0,
      winsB: 0,
      draws: 0,
      goalDiff: 0,
      goalDiffTotal: 0,
      scoreSum: 0,
      actualRate: 0.5,
      expectedRate: 0.5,
      residual: 0,
      score: 0
    };
    map.set(key, row);
    return row;
  }

  for (const match of matches) {
    const score = scoreParts(match.score);
    if (!score) continue;
    const teamA = splitTeam(match.team_a).map(normalizeName);
    const teamB = splitTeam(match.team_b).map(normalizeName);

    for (const side of [{ team: teamA, goalsFor: score[0], goalsAgainst: score[1] }, { team: teamB, goalsFor: score[1], goalsAgainst: score[0] }]) {
      for (let i = 0; i < side.team.length; i += 1) {
        for (let j = i + 1; j < side.team.length; j += 1) {
          const row = touch(teammate, side.team[i], side.team[j]);
          row.matches += 1;
          row.draws += side.goalsFor === side.goalsAgainst ? 1 : 0;
          row.winsA += side.goalsFor > side.goalsAgainst ? 1 : 0;
          row.winsB += side.goalsFor < side.goalsAgainst ? 1 : 0;
          row.scoreSum += side.goalsFor > side.goalsAgainst ? 1 : side.goalsFor === side.goalsAgainst ? 0.5 : 0;
          row.goalDiff += side.goalsFor - side.goalsAgainst;
          row.goalDiffTotal += Math.abs(side.goalsFor - side.goalsAgainst);
        }
      }
    }

    for (const a of teamA) {
      for (const b of teamB) {
        const row = touch(rivals, a, b);
        const [first] = pairKey(a, b).split("|");
        const aIsFirst = normalizeName(a) === first;
        row.matches += 1;
        row.draws += score[0] === score[1] ? 1 : 0;
        row.goalDiffTotal += Math.abs(score[0] - score[1]);
        if (score[0] > score[1]) {
          row.winsA += aIsFirst ? 1 : 0;
          row.winsB += aIsFirst ? 0 : 1;
          row.goalDiff += aIsFirst ? score[0] - score[1] : score[1] - score[0];
        } else if (score[1] > score[0]) {
          row.winsA += aIsFirst ? 0 : 1;
          row.winsB += aIsFirst ? 1 : 0;
          row.goalDiff += aIsFirst ? score[0] - score[1] : score[1] - score[0];
        }
        row.score = rivalryScore(row);
      }
    }
  }

  for (const row of teammate.values()) {
    row.actualRate = row.scoreSum / Math.max(row.matches, 1);
    row.expectedRate = (average(playerScore.get(row.a) || [0.5]) + average(playerScore.get(row.b) || [0.5])) / 2;
    row.residual = row.actualRate - row.expectedRate;
    row.score = chemistryScore(row);
  }

  return { teammate, rivals };
}

function depthWeight(games: number, floor = 0) {
  const weight = Math.log10(games + 1) / Math.log10(10);
  return floor + (1 - floor) * Math.min(1, weight);
}

function chemistryScore(row: PairRecord) {
  const games = row.matches;
  if (!games) return 0;
  const avgGd = row.goalDiffTotal / games;
  const evidence = evidenceConfidence(games, EXPECTED_CHEMISTRY_MIN_GAMES, EXPECTED_CHEMISTRY_FULL_GAMES);
  const closeness = Math.max(0.55, 1 - avgGd / 10);
  return evidence > 0 ? row.residual * 22 * closeness * evidence : 0;
}

function rivalryScore(row: PairRecord) {
  const games = row.matches;
  if (!games) return 0;
  const diff = Math.abs(row.winsA / games - row.winsB / games);
  const avgGd = row.goalDiffTotal / games;
  let intensity = games * (1 - diff) * (1 - Math.min(avgGd, 5) / 5) * depthWeight(games);
  if (games <= 2) intensity *= 0.6;
  return Math.max(0, intensity);
}

function bestTeammate(player: Player, team: Player[], pairRecords: Map<string, PairRecord>) {
  const key = normalizeName(player.name);
  return team
    .filter((mate) => mate.id !== player.id)
    .map((mate) => ({ mate, row: pairRecords.get(pairKey(key, mate.name)) }))
    .filter((item): item is { mate: Player; row: PairRecord } => Boolean(item.row && item.row.matches > 0))
    .sort((a, b) => b.row.score - a.row.score || b.row.matches - a.row.matches)[0];
}

function playerRival(player: Player, opponents: Player[], pairRecords: Map<string, PairRecord>) {
  const key = normalizeName(player.name);
  return opponents
    .map((opponent) => ({ opponent, row: pairRecords.get(pairKey(key, opponent.name)) }))
    .filter((item): item is { opponent: Player; row: PairRecord } => Boolean(item.row && item.row.matches > 0))
    .sort((a, b) => b.row.score - a.row.score || b.row.matches - a.row.matches)[0];
}

function crossTeamMatchups(teamA: Player[], teamB: Player[], pairRecords: Map<string, PairRecord>) {
  return teamA
    .flatMap((a) => teamB.map((b) => ({ a, b, row: pairRecords.get(pairKey(a.name, b.name)) })))
    .filter((item): item is { a: Player; b: Player; row: PairRecord } => Boolean(item.row && item.row.matches > 0))
    .sort((a, b) => b.row.score - a.row.score || b.row.matches - a.row.matches)
    .slice(0, 5);
}

function sameTeamLinks(team: Player[], pairRecords: Map<string, PairRecord>) {
  const links: Array<{ a: Player; b: Player; row: PairRecord }> = [];
  for (let i = 0; i < team.length; i += 1) {
    for (let j = i + 1; j < team.length; j += 1) {
      const row = pairRecords.get(pairKey(team[i].name, team[j].name));
      if (row && row.matches > 0) links.push({ a: team[i], b: team[j], row });
    }
  }
  return links.sort((a, b) => b.row.score - a.row.score || b.row.matches - a.row.matches).slice(0, 3);
}

function teammateLinkSummary(row: PairRecord) {
  const residual = row.residual * 100;
  if (row.matches < EXPECTED_CHEMISTRY_MIN_GAMES) return "Building";
  return residual >= 0 ? `Lift +${fmt(residual, 0)}` : `Drop ${fmt(residual, 0)}`;
}

function rivalrySummary(row: PairRecord, a: Player, b: Player) {
  const aIsFirst = normalizeName(a.name) === row.a;
  const winsForA = aIsFirst ? row.winsA : row.winsB;
  const winsForB = aIsFirst ? row.winsB : row.winsA;
  return `${winsForA}-${row.draws}-${winsForB} | intensity ${fmt(row.score, 1)}`;
}

function teamFormPoints(team: Player[], records: Map<string, PlayerRecord>) {
  return team.reduce((sum, player) => {
    const record = records.get(normalizeName(player.name));
    return sum + (record?.form || []).reduce((total, result) => total + (result === "W" ? 3 : result === "D" ? 1 : 0), 0);
  }, 0);
}

function teamChemistryScore(team: Player[], pairRecords: Map<string, PairRecord>) {
  let score = 0;
  for (let i = 0; i < team.length; i += 1) {
    for (let j = i + 1; j < team.length; j += 1) {
      const row = pairRecords.get(pairKey(team[i].name, team[j].name));
      if (row) score += row.score;
    }
  }
  return score;
}

function topRated(team: Player[]) {
  return [...team].sort((a, b) => Number(b.mmr || 0) - Number(a.mmr || 0))[0] || null;
}

function formDots(form: string[]) {
  return form.length ? form : ["N", "N", "N", "N", "N"];
}

function displayFromMap(name: string, nameMap: Map<string, string>) {
  return nameMap.get(normalizeName(name)) || name;
}

function marketForOption(option: MatchdayOption, matches: Match[]) {
  const drawRate = historicalDrawRate(matches);
  const ratingDiff = option.avgA - option.avgB;
  const expectedA = 1 / (1 + Math.pow(10, -ratingDiff / 400));
  const closeness = 1 - Math.min(1, Math.abs(expectedA - 0.5) * 2);
  const draw = clamp(drawRate * (0.7 + closeness * 0.7), 0.05, 0.24);
  const teamA = clamp(expectedA - draw / 2, 0.05, 0.9);
  const teamB = clamp(1 - draw - teamA, 0.05, 0.9);
  const total = teamA + draw + teamB;
  const probs = {
    teamA: teamA / total,
    draw: draw / total,
    teamB: teamB / total
  };
  const avgTotal = averageTotalGoals(matches);
  const favouriteA = option.avgA >= option.avgB;
  const spread = Math.max(0.5, option.ranking.predictedMargin);
  return {
    probs,
    odds: {
      teamA: decimalOdds(probs.teamA),
      draw: decimalOdds(probs.draw),
      teamB: decimalOdds(probs.teamB)
    },
    expectedScore: {
      a: Math.max(1, avgTotal / 2 + (favouriteA ? spread : -spread) / 2),
      b: Math.max(1, avgTotal / 2 + (favouriteA ? -spread : spread) / 2)
    }
  };
}

function historicalDrawRate(matches: Match[]) {
  const scored = matches.map((match) => scoreParts(match.score)).filter(Boolean) as Array<[number, number]>;
  if (!scored.length) return 0.1;
  return scored.filter(([a, b]) => a === b).length / scored.length;
}

function averageTotalGoals(matches: Match[]) {
  const scored = matches.map((match) => scoreParts(match.score)).filter(Boolean) as Array<[number, number]>;
  if (!scored.length) return 20;
  return scored.reduce((sum, [a, b]) => sum + a + b, 0) / scored.length;
}

function evidenceConfidence(matches: number, minGames: number, fullGames: number) {
  if (matches < minGames) return 0;
  if (matches >= fullGames) return 1;
  return (matches - minGames + 1) / Math.max(fullGames - minGames + 1, 1);
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function decimalOdds(probability: number) {
  return (1 / clamp(probability, 0.01, 0.99)).toFixed(2);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function similarMatchesForOption(option: MatchdayOption | null, matches: Match[]) {
  if (!option) return [];
  const currentA = new Set(option.teamA.map((player) => normalizeName(player.name)));
  const currentB = new Set(option.teamB.map((player) => normalizeName(player.name)));
  const currentAll = new Set([...currentA, ...currentB]);
  const totalPlayers = option.teamA.length + option.teamB.length;

  return matches
    .map((match) => {
      const score = scoreParts(match.score);
      const rawHistA = splitTeam(match.team_a);
      const rawHistB = splitTeam(match.team_b);
      const histA = new Set(rawHistA.map(normalizeName));
      const histB = new Set(rawHistB.map(normalizeName));
      const same = countOverlap(currentA, histA) + countOverlap(currentB, histB);
      const swapped = countOverlap(currentA, histB) + countOverlap(currentB, histA);
      const involved = new Set([...histA, ...histB]);
      const overlap = countOverlap(currentAll, involved);
      const useSwappedOrientation = swapped > same;
      const orientedA = useSwappedOrientation ? rawHistB : rawHistA;
      const orientedB = useSwappedOrientation ? rawHistA : rawHistB;
      const scoreLabel = score ? orientedScoreLabel(score, useSwappedOrientation) : match.score || "-";
      const orientedASet = new Set(orientedA.map(normalizeName));
      const orientedBSet = new Set(orientedB.map(normalizeName));
      const sameSide = Math.max(same, swapped);
      const returningA = orientedA.filter((name) => currentA.has(normalizeName(name)));
      const returningB = orientedB.filter((name) => currentB.has(normalizeName(name)));
      return {
        match,
        overlap,
        sameSide,
        swapped: useSwappedOrientation,
        margin: score ? Math.abs(score[0] - score[1]) : 0,
        goals: score ? score[0] + score[1] : 0,
        scoreLabel,
        teamA: orientedA,
        teamB: orientedB,
        labelA: useSwappedOrientation ? "Old Team B" : "Old Team A",
        labelB: useSwappedOrientation ? "Old Team A" : "Old Team B",
        returningA,
        returningB,
        missingA: orientedA.filter((name) => !currentA.has(normalizeName(name))),
        missingB: orientedB.filter((name) => !currentB.has(normalizeName(name))),
        newA: option.teamA.map((player) => player.name).filter((name) => !orientedASet.has(normalizeName(name))),
        newB: option.teamB.map((player) => player.name).filter((name) => !orientedBSet.has(normalizeName(name)))
      };
    })
    .filter((row) => row.sameSide >= Math.max(4, Math.floor(totalPlayers * 0.45)))
    .sort((a, b) => b.sameSide - a.sameSide || a.margin - b.margin || b.goals - a.goals || b.overlap - a.overlap)
    .slice(0, 3);
}

function countOverlap(a: Set<string>, b: Set<string>) {
  let count = 0;
  for (const item of a) {
    if (b.has(item)) count += 1;
  }
  return count;
}

function orientedScoreLabel(score: [number, number], swapped: boolean) {
  return swapped ? `${score[1]}-${score[0]}` : `${score[0]}-${score[1]}`;
}

function fmt(value: number, digits = 0) {
  return value.toFixed(digits);
}

function signedFmt(value: number, digits = 0) {
  const rounded = fmt(value, digits);
  return value > 0 ? `+${rounded}` : rounded;
}

function chemistryBarValue(value: number) {
  return Math.max(0, value + 8);
}

function HistoricalSideCard({
  label,
  historicTeam,
  currentTeam,
  nameMap,
  tone
}: {
  label: string;
  historicTeam: string[];
  currentTeam: string[];
  nameMap: Map<string, string>;
  tone: "a" | "b";
}) {
  const currentSet = new Set(currentTeam.map(normalizeName));
  const historicSet = new Set(historicTeam.map(normalizeName));
  const keptInThisMatchup = historicTeam.filter((name) => currentSet.has(normalizeName(name)));
  const inThisMatchup = currentTeam.filter((name) => !historicSet.has(normalizeName(name)));
  const outThisMatchup = historicTeam.filter((name) => !currentSet.has(normalizeName(name)));

  return (
    <div className={`similar-side-card ${tone}`}>
      <div className="previous-side-head">
        <strong>{label}</strong>
        <span>{keptInThisMatchup.length}/{historicTeam.length} kept</span>
      </div>
      <div className="mini-change-block">
        <div className="history-change-row previous">
          <small>Old</small>
          <div className="pill-row">
            {historicTeam.map((name) => (
              <span
                className={outThisMatchup.map(normalizeName).includes(normalizeName(name)) ? "pill out" : "pill kept"}
                key={`old-${tone}-${name}`}
              >
                {displayFromMap(name, nameMap)}
              </span>
            ))}
          </div>
        </div>
        <div className="history-change-row subs">
          <small>Subs</small>
          <div className="pill-row">
            {inThisMatchup.length ? inThisMatchup.map((name) => (
              <span className="pill in" key={`new-${tone}-${name}`}>{displayFromMap(name, nameMap)}</span>
            )) : <small>Same five</small>}
          </div>
        </div>
      </div>
    </div>
  );
}

function TeamLinkRow({
  a,
  b,
  row,
  teamLabel
}: {
  a: Player;
  b: Player;
  row: PairRecord;
  teamLabel: string;
}) {
  const names = `${playerLabel(a)} + ${playerLabel(b)}`;
  return (
    <div className="link-scouting-row">
      <strong title={names}>{names}</strong>
      <span>{row.matches} games</span>
      <span>{row.winsA}-{row.draws}-{row.winsB}</span>
      <em>{teammateLinkSummary(row)}</em>
      <small>{teamLabel}</small>
    </div>
  );
}

type ProfileKey = "finishing" | "creation" | "saveImpact" | "clutch" | "impact";

const PROFILE_SPECIALIST_FLOOR = 0.7;
const PROFILE_SECONDARY_FLOOR = 0.72;
const PROFILE_SECONDARY_MAX_GAP = 0.14;
type ProfileRoleKey = "finisher" | "creator" | "save";

function profileValue(player: Player, key: ProfileKey) {
  return Number(player.style_profile?.[key] || 0);
}

function profileRoleTraits(player: Player): Array<{ key: ProfileRoleKey; label: string; value: number }> {
  const profile = player.style_profile;
  if (!profile) return [];
  const traits = [
    { key: "finisher" as const, label: "🎯 Finisher", value: Number(profile.finishing || 0) },
    { key: "creator" as const, label: "⚙️ Creator", value: Number(profile.creation || 0) },
    { key: "save" as const, label: "🧤 Save impact", value: Number(profile.saveImpact || 0) }
  ].sort((a, b) => b.value - a.value);
  const [first, second] = traits;
  if (!first || first.value < PROFILE_SPECIALIST_FLOOR) return [];
  if (second && second.value >= PROFILE_SECONDARY_FLOOR && first.value - second.value <= PROFILE_SECONDARY_MAX_GAP) {
    return [first, second];
  }
  return [first];
}

function isProfileCreator(player: Player) {
  return profileRoleTraits(player).some((trait) => trait.key === "creator");
}

function isProfileFinisher(player: Player) {
  return profileRoleTraits(player).some((trait) => trait.key === "finisher");
}

function teamProfileSummary(team: Player[]) {
  const profiled = team.filter((player) => player.style_profile);
  return {
    count: profiled.length,
    finishing: average(profiled.map((player) => profileValue(player, "finishing"))),
    creation: average(profiled.map((player) => profileValue(player, "creation"))),
    saveImpact: average(profiled.map((player) => profileValue(player, "saveImpact"))),
    clutch: average(profiled.map((player) => profileValue(player, "clutch"))),
    impact: average(profiled.map((player) => profileValue(player, "impact"))),
    goals: profiled.reduce((sum, player) => sum + Number(player.style_profile?.historicGoals || 0), 0),
    assists: profiled.reduce((sum, player) => sum + Number(player.style_profile?.historicAssists || 0), 0),
    saves: profiled.reduce((sum, player) => sum + Number(player.style_profile?.saveMoments || 0), 0),
    goalRate: profiled.reduce((sum, player) => sum + Number(player.style_profile?.goalsPerGame || 0), 0),
    assistRate: profiled.reduce((sum, player) => sum + Number(player.style_profile?.assistsPerGame || 0), 0),
    saveRate: profiled.reduce((sum, player) => sum + Number(player.style_profile?.saveMomentsPerGame || 0), 0),
    avgGames: average(profiled.map((player) => Number(player.style_profile?.historicMatches || 0))),
    specials: profiled.reduce((sum, player) => sum + Number(player.style_profile?.specialMoments || 0), 0),
    creators: profiled.filter(isProfileCreator).length,
    finishers: profiled.filter(isProfileFinisher).length
  };
}

function playerProfileRole(player: Player) {
  const profile = player.style_profile;
  if (!profile) return "No profile";
  const bestValue = Math.max(
    Number(profile.finishing || 0),
    Number(profile.creation || 0),
    Number(profile.saveImpact || 0),
    Number(profile.clutch || 0)
  );
  if (bestValue < 0.25) return "🃏 Wildcard profile";
  const roleTraits = profileRoleTraits(player);
  if (!roleTraits.length) return "🤝 Locker Room Leader";
  return roleTraits.map((trait) => trait.label).join(" / ");
}

function profileScoreLabel(score: number) {
  if (score <= 0.75) return "Elite balance";
  if (score <= 1.6) return "Playable balance";
  if (score <= 2.6) return "Watch the roles";
  return "Profile warning";
}

function profileScoreTone(score: number) {
  if (score <= 0.75) return "good";
  if (score <= 1.6) return "steady";
  return "alert";
}

function profileMetricWidth(value: number) {
  return `${Math.max(6, Math.min(100, Number(value || 0) * 100))}%`;
}

function profilePlayerTraits(player: Player) {
  const profile = player.style_profile;
  if (!profile) return [];
  return [
    { label: "F", title: "Finishing", value: Number(profile.finishing || 0), className: "finish" },
    { label: "C", title: "Creation", value: Number(profile.creation || 0), className: "create" },
    { label: "S", title: "Save impact", value: Number(profile.saveImpact || 0), className: "save" }
  ];
}

function profileRateLine(goalsPerGame: number, assistsPerGame: number, savesPerGame: number) {
  return `${fmt(goalsPerGame, 1)}G ${fmt(assistsPerGame, 1)}A ${fmt(savesPerGame, 1)}S/game`;
}

function profileCompactRateLine(goalsPerGame: number, assistsPerGame: number, savesPerGame: number) {
  return `${fmt(goalsPerGame, 1)}G ${fmt(assistsPerGame, 1)}A ${fmt(savesPerGame, 1)}S/g`;
}

function profileSampleLine(avgGames: number) {
  return `${fmt(avgGames, 1)} avg profile games/player`;
}

function plural(count: number, singular: string) {
  return `${count} ${singular}${count === 1 ? "" : "s"}`;
}

function profileDebugReasons(
  option: MatchdayOption,
  summaryA: ReturnType<typeof teamProfileSummary>,
  summaryB: ReturnType<typeof teamProfileSummary>,
  profileRows: Array<{ label: string; a: number; b: number; gap: number; weight: number }>,
  allOptions?: MatchdayOption[]
) {
  const reasons: string[] = [];
  const styleNet = breakdownNumber(option, "style_net");
  const bestProfile = allOptions?.filter((row) => Boolean(row.breakdown.style_enabled)).sort((a, b) => breakdownNumber(a, "style_net") - breakdownNumber(b, "style_net"))[0];
  const profileGap = bestProfile ? styleNet - breakdownNumber(bestProfile, "style_net") : 0;
  const shortagePenalty = breakdownNumber(option, "style_finisher_shortage") + breakdownNumber(option, "style_creator_shortage");
  const topGapRow = [...profileRows].sort((a, b) => b.gap - a.gap)[0];

  if (styleNet <= 0.75) {
    const bestText = bestProfile ? `only ${fmt(profileGap, 1)} away from the best profile-only split` : "already inside the safest profile range";
    reasons.push(`Profile mix is clean: ${fmt(styleNet, 1)} is ${bestText}, so the profiles support this split rather than arguing for a different team.`);
  } else if (styleNet <= 1.6) {
    reasons.push(`Profile mix is playable: ${fmt(styleNet, 1)} leaves a small role-shape gap, but not enough to override MMR, margin and repeat-risk checks.`);
  } else {
    reasons.push(`Profile mix needs a look: ${fmt(styleNet, 1)} means one side owns more of the rate-based role profile than ideal.`);
  }

  const coverageText = `Specialist cover is A ${plural(summaryA.finishers, "finisher")}/${plural(summaryA.creators, "creator")} and B ${plural(summaryB.finishers, "finisher")}/${plural(summaryB.creators, "creator")}`;
  if (shortagePenalty <= 0) {
    reasons.push(`${coverageText}. No finisher-or-creator shortage penalty is being added.`);
  } else {
    reasons.push(`${coverageText}. The score adds ${fmt(shortagePenalty, 1)} because one side is light on a key specialist role.`);
  }

  if (topGapRow && topGapRow.gap >= 0.12) {
    const leader = topGapRow.a >= topGapRow.b ? "Team A" : "Team B";
    const leaderValue = topGapRow.a >= topGapRow.b ? topGapRow.a : topGapRow.b;
    const trailerValue = topGapRow.a >= topGapRow.b ? topGapRow.b : topGapRow.a;
    if (topGapRow.label === "Save impact") {
      reasons.push(`Biggest swing is save impact: ${leader} leads ${fmt(leaderValue, 2)} to ${fmt(trailerValue, 2)} on the 0-1 rate score, but its x${topGapRow.weight} weight is low because keeper rotation handles most save-risk separately.`);
    } else {
      reasons.push(`Biggest swing is ${topGapRow.label.toLowerCase()}: ${leader} leads ${fmt(leaderValue, 2)} to ${fmt(trailerValue, 2)} on the 0-1 rate score, and its x${topGapRow.weight} weight controls how much that matters.`);
    }
  } else {
    reasons.push("No single profile lane is pulling the teams badly apart; the balance is coming from the whole profile mix.");
  }

  return reasons.slice(0, 3);
}

function profileShareLabel(score: number) {
  if (score <= 0.75) return "Clean split";
  if (score <= 1.6) return "Pretty fair";
  if (score <= 2.6) return "Role lean";
  return "Check the roles";
}

function profileTeamEdge(
  summary: ReturnType<typeof teamProfileSummary>,
  opponent: ReturnType<typeof teamProfileSummary>
) {
  const edges = [
    { label: "Finishing", value: summary.finishing - opponent.finishing },
    { label: "Creation", value: summary.creation - opponent.creation },
    { label: "Save cover", value: summary.saveImpact - opponent.saveImpact }
  ].sort((a, b) => b.value - a.value);
  const best = edges[0];
  if (!best || best.value < 0.04) return { label: "No clear edge", note: "Roles are close" };
  return { label: best.label, note: `+${fmt(best.value, 2)} role edge` };
}

function profileSwingLabel(row: { label: string; a: number; b: number; gap: number; weight: number } | undefined) {
  if (!row) return { label: "No big swing", note: "Role balance is tight" };
  const leader = row.a >= row.b ? "Team A" : "Team B";
  const label = row.label === "Save impact" ? "save cover" : row.label.toLowerCase();
  const note = row.label === "Save impact"
    ? "Useful, but keeper rotation softens it"
    : `${fmt(row.gap, 2)} gap in the role mix`;
  return { label: `${leader} ${label}`, note };
}

function profileShareNudge(
  label: "Team A" | "Team B",
  summary: ReturnType<typeof teamProfileSummary>,
  opponent: ReturnType<typeof teamProfileSummary>
) {
  if (summary.saveImpact - opponent.saveImpact >= 0.08) return `${label} has stronger save cover`;
  if (summary.creators > opponent.creators) return `${label} has more creators`;
  if (summary.finishers > opponent.finishers) return `${label} has more finishers`;
  const edge = profileTeamEdge(summary, opponent);
  if (edge.label === "No clear edge") return "";
  return `${label} has stronger ${edge.label.toLowerCase()}`;
}

function profileShareRead(
  styleNet: number,
  shortagePenalty: number,
  summaryA: ReturnType<typeof teamProfileSummary>,
  summaryB: ReturnType<typeof teamProfileSummary>
) {
  const balance = styleNet <= 1.6 ? "Pretty even split." : "There is a role lean.";
  const nudges = [
    profileShareNudge("Team A", summaryA, summaryB),
    profileShareNudge("Team B", summaryB, summaryA)
  ].filter(Boolean);
  const middle = nudges.length ? nudges.join(", ") + "." : "Roles are close across both sides.";
  const flags = shortagePenalty <= 0 ? "No obvious role red flags." : "One side is light on a main finisher or creator.";
  return `${balance} ${middle} ${flags}`;
}

function ProfileDebugSection({
  option,
  allOptions
}: {
  option: MatchdayOption;
  allOptions?: MatchdayOption[];
}) {
  const [showProfileDetails, setShowProfileDetails] = useState(false);
  const summaryA = teamProfileSummary(option.teamA);
  const summaryB = teamProfileSummary(option.teamB);
  const profileRows = [
    { label: "Finishing", a: summaryA.finishing, b: summaryB.finishing, gap: breakdownNumber(option, "style_finishing_diff"), weight: 1.65 },
    { label: "Creation", a: summaryA.creation, b: summaryB.creation, gap: breakdownNumber(option, "style_creation_diff"), weight: 1.65 },
    { label: "Save impact", a: summaryA.saveImpact, b: summaryB.saveImpact, gap: breakdownNumber(option, "style_save_diff"), weight: 0.35 },
    { label: "Clutch", a: summaryA.clutch, b: summaryB.clutch, gap: breakdownNumber(option, "style_clutch_diff"), weight: 0.6 },
    { label: "Overall impact", a: summaryA.impact, b: summaryB.impact, gap: breakdownNumber(option, "style_impact_diff"), weight: 1.1 }
  ];
  const bestProfile = allOptions?.filter((row) => Boolean(row.breakdown.style_enabled)).sort((a, b) => breakdownNumber(a, "style_net") - breakdownNumber(b, "style_net"))[0] || null;
  const enabled = Boolean(option.breakdown.style_enabled);
  const loadedProfiles = summaryA.count + summaryB.count;
  const totalPlayers = option.teamA.length + option.teamB.length;
  const styleNet = breakdownNumber(option, "style_net");
  const shareProfileRows = profileRows.filter((row) => row.label !== "Clutch");
  const topGapRow = [...shareProfileRows].sort((a, b) => b.gap - a.gap)[0];
  const detailTopGapRow = [...profileRows].sort((a, b) => b.gap - a.gap)[0];
  const shortagePenalty = breakdownNumber(option, "style_finisher_shortage") + breakdownNumber(option, "style_creator_shortage");
  const bestProfileScore = bestProfile ? breakdownNumber(bestProfile, "style_net") : null;
  const profileTone = profileScoreTone(styleNet);
  const teamAEdge = profileTeamEdge(summaryA, summaryB);
  const teamBEdge = profileTeamEdge(summaryB, summaryA);
  const swing = profileSwingLabel(topGapRow);
  const teamColumns = [
    { label: "Team A", team: option.teamA, summary: summaryA, className: "a" },
    { label: "Team B", team: option.teamB, summary: summaryB, className: "b" }
  ];
  const profileExplainers = [
    {
      title: "F / C / S",
      copy: "Relative 0-1 scores from per-game finishing, creation and save-impact rates. 1.00 means best in this group."
    },
    {
      title: "Role badge",
      copy: "A badge only appears when a player clears the specialist line. Otherwise they can show as Locker Room Leader or Wildcard."
    },
    {
      title: "G/A/S rate",
      copy: "Goals, assists and tagged save moments per game, so longer histories do not automatically look stronger."
    },
    {
      title: "Sample",
      copy: "Games are used as confidence only. Small samples are softened so one hot night does not dominate the role badge."
    },
    {
      title: "Gap | weight",
      copy: "Gap is the team average difference. Weight is how much that lane affects the profile-mix score."
    }
  ];

  return (
    <section className={`profile-debug-panel ${profileTone}`}>
      <div className="profile-hero-strip">
        <div>
          <div className="profile-hero-title-row">
            <span className="stat-label">Group chat read</span>
            {enabled ? (
              <button
                aria-pressed={showProfileDetails}
                className="ghost-button profile-mode-toggle"
                onClick={() => setShowProfileDetails((value) => !value)}
                type="button"
              >
                {showProfileDetails ? "Share view" : "Deep dive"}
              </button>
            ) : null}
          </div>
          <h3>Profile vibe check</h3>
          <p>
            {enabled
              ? `${loadedProfiles}/${totalPlayers} profiles loaded. Rates are per game, not lifetime totals.`
              : "Profile data is unavailable for this split."}
          </p>
        </div>
        <div className="profile-score-lockup">
          <span>Profile mix</span>
          <strong>{enabled ? fmt(styleNet, 1) : "Off"}</strong>
          <small>{enabled ? profileScoreLabel(styleNet) : "MMR fallback"}</small>
        </div>
      </div>

      <div className="profile-share-summary">
        <div>
          <span>Verdict</span>
          <strong>{profileShareLabel(styleNet)}</strong>
          <small>{fmt(styleNet, 1)} profile mix</small>
        </div>
        <div>
          <span>Team A edge</span>
          <strong>{teamAEdge.label}</strong>
          <small>{teamAEdge.note}</small>
        </div>
        <div>
          <span>Team B edge</span>
          <strong>{teamBEdge.label}</strong>
          <small>{teamBEdge.note}</small>
        </div>
        <div>
          <span>Main swing</span>
          <strong>{swing.label}</strong>
          <small>{swing.note}</small>
        </div>
      </div>

      {enabled ? (
        <>
          <p className="profile-share-read">{profileShareRead(styleNet, shortagePenalty, summaryA, summaryB)}</p>

          {showProfileDetails ? (
            <>
              <div className="profile-debug-summary">
                <div>
                  <span>Loaded profiles</span>
                  <strong>{loadedProfiles}/{totalPlayers}</strong>
                  <small>{bestProfileScore !== null ? `${fmt(bestProfileScore, 1)} best mix available` : "No comparison split"}</small>
                </div>
                <div>
                  <span>Team A rates</span>
                  <strong>{profileRateLine(summaryA.goalRate, summaryA.assistRate, summaryA.saveRate)}</strong>
                  <small>{profileSampleLine(summaryA.avgGames)}</small>
                </div>
                <div>
                  <span>Team B rates</span>
                  <strong>{profileRateLine(summaryB.goalRate, summaryB.assistRate, summaryB.saveRate)}</strong>
                  <small>{profileSampleLine(summaryB.avgGames)}</small>
                </div>
                <div>
                  <span>Specialist coverage</span>
                  <strong>A {summaryA.finishers}F/{summaryA.creators}C | B {summaryB.finishers}F/{summaryB.creators}C</strong>
                  <small>{fmt(shortagePenalty, 1)} shortage penalty</small>
                </div>
                <div>
                  <span>Keeper save floor</span>
                  <strong>A {fmt(breakdownNumber(option, "rotation_save_floor_a"), 2)} | B {fmt(breakdownNumber(option, "rotation_save_floor_b"), 2)}</strong>
                  <small>{fmt(breakdownNumber(option, "rotation_save_penalty"), 1)} keeper-profile penalty</small>
                </div>
                <div>
                  <span>Biggest swing</span>
                  <strong>{detailTopGapRow ? detailTopGapRow.label : "-"}</strong>
                  <small>{detailTopGapRow ? `${fmt(detailTopGapRow.gap, 2)} gap at x${detailTopGapRow.weight}` : "No profile gap"}</small>
                </div>
              </div>

              <div className="profile-balance-grid">
                {profileRows.map((row) => {
                  return (
                    <div className="profile-metric-row" key={row.label}>
                      <div className="profile-metric-head">
                        <strong>{row.label}</strong>
                        <span>gap {fmt(row.gap, 2)} | x{row.weight}</span>
                      </div>
                      <div className="profile-bars">
                        <span>A {fmt(row.a, 2)}</span>
                        <div className="profile-track"><i className="a" style={{ width: profileMetricWidth(row.a) }} /></div>
                        <div className="profile-track"><i className="b" style={{ width: profileMetricWidth(row.b) }} /></div>
                        <span>B {fmt(row.b, 2)}</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="profile-debug-copy">
                <span>Profile read</span>
                {profileDebugReasons(option, summaryA, summaryB, profileRows, allOptions).map((reason) => <p key={reason}>{reason}</p>)}
              </div>

              <div className="profile-explainer-grid">
                {profileExplainers.map((item) => (
                  <div key={item.title}>
                    <strong>{item.title}</strong>
                    <span>{item.copy}</span>
                  </div>
                ))}
              </div>
            </>
          ) : null}

          <div className="profile-player-columns">
            {teamColumns.map(({ label, team, summary, className }) => (
              <div className={`profile-player-team ${className}`} key={label}>
                <div className="profile-player-team-head">
                  <span>{label}</span>
                  <strong>{showProfileDetails ? profileRateLine(summary.goalRate, summary.assistRate, summary.saveRate) : `${summary.finishers} finishers / ${summary.creators} creators`}</strong>
                </div>
                {team.map((player) => {
                  const profile = player.style_profile;
                  const traits = profilePlayerTraits(player);
                  return (
                    <div className={showProfileDetails ? "profile-player-row" : "profile-player-row compact"} key={player.id}>
                      <div className="profile-player-main">
                        <strong>{playerLabel(player)}</strong>
                        <span>{playerProfileRole(player)}</span>
                      </div>
                      {profile ? (
                        <>
                          {showProfileDetails ? (
                            <div className="profile-player-traits" aria-label={`${playerLabel(player)} profile traits`}>
                              {traits.map((trait) => (
                                <span className={trait.className} title={trait.title} key={trait.label}>
                                  {trait.label} {fmt(trait.value, 2)}
                                </span>
                              ))}
                            </div>
                          ) : null}
                          <small>{showProfileDetails ? `${profileRateLine(profile.goalsPerGame, profile.assistsPerGame, profile.saveMomentsPerGame)} | ${fmt(profile.historicMatches)} games` : profileCompactRateLine(profile.goalsPerGame, profile.assistsPerGame, profile.saveMomentsPerGame)}</small>
                        </>
                      ) : (
                        <small>No rate profile</small>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </>
      ) : (
        <p className="muted">Profiles are off for this card, so the team engine is falling back to MMR, chemistry, repeat risk and keeper rotation.</p>
      )}
    </section>
  );
}

function MatchdayAnalysisCard({
  option,
  matches,
  nameMap,
  playerRecords,
  pairRecords,
  allOptions,
  onClose
}: {
  option: MatchdayOption;
  matches: Match[];
  nameMap: Map<string, string>;
  playerRecords: Map<string, PlayerRecord>;
  pairRecords: { teammate: Map<string, PairRecord>; rivals: Map<string, PairRecord> };
  allOptions?: MatchdayOption[];
  onClose: () => void;
}) {
  const cardExplanation = optionExplanation(option);
  const similarMatches = useMemo(() => similarMatchesForOption(option, matches), [matches, option]);
  const tweaks = useMemo(() => findBestTweaks(option.teamA, option.teamB, matches, 2), [matches, option]);
  const cardMarket = useMemo(() => marketForOption(option, matches), [matches, option]);
  const keyMatchups = useMemo(
    () => crossTeamMatchups(option.teamA, option.teamB, pairRecords.rivals),
    [option, pairRecords.rivals]
  );
  const teamALinks = useMemo(
    () => sameTeamLinks(option.teamA, pairRecords.teammate),
    [option, pairRecords.teammate]
  );
  const teamBLinks = useMemo(
    () => sameTeamLinks(option.teamB, pairRecords.teammate),
    [option, pairRecords.teammate]
  );
  const formA = teamFormPoints(option.teamA, playerRecords);
  const formB = teamFormPoints(option.teamB, playerRecords);
  const chemistryA = teamChemistryScore(option.teamA, pairRecords.teammate);
  const chemistryB = teamChemistryScore(option.teamB, pairRecords.teammate);
  const anchorA = topRated(option.teamA);
  const anchorB = topRated(option.teamB);
  const bestLinkA = teamALinks[0];
  const bestLinkB = teamBLinks[0];
  const dangerA = keyMatchups[0];
  const dangerB = keyMatchups[0];
  const comparisonRows = [
    { label: "Effective MMR", a: option.avgA, b: option.avgB, left: fmt(option.avgA), right: fmt(option.avgB) },
    { label: "Recent form", a: formA, b: formB, left: `${formA} pts`, right: `${formB} pts` },
    { label: "Chemistry", a: chemistryBarValue(chemistryA), b: chemistryBarValue(chemistryB), left: signedFmt(chemistryA, 0), right: signedFmt(chemistryB, 0) },
    { label: "Anchor rating", a: Number(anchorA?.mmr || 0), b: Number(anchorB?.mmr || 0), left: anchorA ? `${playerLabel(anchorA)} ${fmt(Number(anchorA.mmr || 0))}` : "-", right: anchorB ? `${playerLabel(anchorB)} ${fmt(Number(anchorB.mmr || 0))}` : "-" }
  ];

  return (
    <section className="panel matchday-card-panel">
      <div className="panel-title-row">
        <div>
          <span className="stat-label">Selected matchup</span>
          <h2>Matchday Card</h2>
        </div>
        <button className="ghost-button" type="button" onClick={onClose}>Change matchup</button>
      </div>

      <div className="match-report-hero">
        <div className="report-team report-team-a">
          <span>Team A</span>
          <strong>{fmt(option.avgA)}</strong>
          <small>avg MMR</small>
          <div className="mini-lineup">{option.teamA.map((player) => <i key={player.id}>{playerLabel(player)}</i>)}</div>
        </div>
        <div className="report-score">
          <span>Projected score</span>
          <strong>{cardMarket ? `${fmt(cardMarket.expectedScore.a, 1)} - ${fmt(cardMarket.expectedScore.b, 1)}` : "-"}</strong>
          <small>{fmt(option.ranking.closePct)}/100 closeness score</small>
        </div>
        <div className="report-team report-team-b">
          <span>Team B</span>
          <strong>{fmt(option.avgB)}</strong>
          <small>avg MMR</small>
          <div className="mini-lineup">{option.teamB.map((player) => <i key={player.id}>{playerLabel(player)}</i>)}</div>
        </div>
      </div>

      <div className="match-report-board">
        <section className="analysis-centrepiece">
          <div className="section-subhead"><strong>Team comparison</strong><span>How the two sides stack up</span></div>
          <div className="comparison-bar-list">
            {comparisonRows.map((row) => {
              const max = Math.max(row.a, row.b, 1);
              return (
                <div className="comparison-bar-row" key={row.label}>
                  <div className="comparison-value left">{row.left}</div>
                  <div className="comparison-track">
                    <span className="bar-a" style={{ width: `${Math.max(12, (row.a / max) * 92)}%` }} />
                    <em>{row.label}</em>
                    <span className="bar-b" style={{ width: `${Math.max(12, (row.b / max) * 92)}%` }} />
                  </div>
                  <div className="comparison-value right">{row.right}</div>
                </div>
              );
            })}
          </div>
          <div className="analysis-signal-row">
            <div><span>Match potential</span><strong>{fmt(option.ranking.recommendationScore)}%</strong></div>
            <div><span>Expected margin</span><strong>{fmt(option.ranking.predictedMargin, 1)}</strong></div>
            {option.breakdown.style_enabled ? (
              <div><span>Profile mix</span><strong>{fmt(breakdownNumber(option, "style_net"), 1)}</strong></div>
            ) : (
              <div><span>Profile mix</span><strong>Off</strong></div>
            )}
            <div><span>Keeper rotation</span><strong>{fmt(breakdownNumber(option, "rotating_keeper_penalty"), 1)}</strong></div>
            <div><span>Repeat risk</span><strong>{fmt(option.similarityPenalty, 1)}</strong></div>
          </div>
        </section>

        <aside className="analyst-verdict">
          <div className="section-subhead"><strong>Analyst verdict</strong></div>
          <div className="verdict-copy">
            {cardExplanation.positives.map((reason) => <p key={reason}>{reason}</p>)}
            {cardExplanation.risks.map((risk) => <p className="warning" key={risk}>{risk}</p>)}
          </div>
          <div className="verdict-tweak">
            <span>Best tweak</span>
            {tweaks?.suggestions.length ? (
              tweaks.suggestions.slice(0, 1).map((suggestion) => (
                <strong key={`${suggestion.swapA.id}-${suggestion.swapB.id}`}>{playerLabel(suggestion.swapA)} &lt;-&gt; {playerLabel(suggestion.swapB)}</strong>
              ))
            ) : (
              <strong>No tweak needed</strong>
            )}
          </div>
        </aside>
      </div>

      <div className="team-report-grid">
        <section className="team-report-card team-report-a">
          <div className="section-subhead"><strong>Team A</strong></div>
          <div className="pill-row">{option.teamA.map((player) => <span className="pill" key={player.id}>{playerLabel(player)}</span>)}</div>
          <div className="team-report-stats">
            <div><span>Form</span><strong>{formA} pts</strong></div>
            <div><span>Best link</span><strong>{bestLinkA ? `${playerLabel(bestLinkA.a)} + ${playerLabel(bestLinkA.b)}` : "-"}</strong></div>
            <div><span>Likely rival</span><strong>{dangerA ? `${playerLabel(dangerA.a)} vs ${playerLabel(dangerA.b)}` : "-"}</strong></div>
          </div>
        </section>
        <section className="team-report-card team-report-b">
          <div className="section-subhead"><strong>Team B</strong></div>
          <div className="pill-row">{option.teamB.map((player) => <span className="pill" key={player.id}>{playerLabel(player)}</span>)}</div>
          <div className="team-report-stats">
            <div><span>Form</span><strong>{formB} pts</strong></div>
            <div><span>Best link</span><strong>{bestLinkB ? `${playerLabel(bestLinkB.a)} + ${playerLabel(bestLinkB.b)}` : "-"}</strong></div>
            <div><span>Likely rival</span><strong>{dangerB ? `${playerLabel(dangerB.b)} vs ${playerLabel(dangerB.a)}` : "-"}</strong></div>
          </div>
        </section>
      </div>

      <div className="scouting-grid">
        <section className="scouting-card">
          <div className="section-subhead"><strong>Key matchups</strong><span>Rivalries across the two teams</span></div>
          <div className="scouting-list">
            {keyMatchups.length ? keyMatchups.map(({ a, b, row }) => (
              <div key={`${a.id}-${b.id}`}>
                <strong>{playerLabel(a)} vs {playerLabel(b)}</strong>
                <span>{row.matches} meetings</span>
                <em>{rivalrySummary(row, a, b)}</em>
              </div>
            )) : <p className="muted">No strong rivalry history yet.</p>}
          </div>
        </section>
        <section className="scouting-card">
          <div className="section-subhead"><strong>Best teammate links</strong><span>Strongest same-side history</span></div>
          <div className="scouting-list link-scouting-list">
            {[...teamALinks, ...teamBLinks].slice(0, 6).map(({ a, b, row }) => (
              <TeamLinkRow
                a={a}
                b={b}
                row={row}
                teamLabel={option.teamA.some((player) => player.id === a.id) ? "Team A" : "Team B"}
                key={`${a.id}-${b.id}`}
              />
            ))}
            {!teamALinks.length && !teamBLinks.length ? <p className="muted">Not enough teammate history yet.</p> : null}
          </div>
        </section>
      </div>

      {similarMatches.length ? (
        <div className="similar-match-list">
          <div className="section-subhead">
            <strong>Closest previous games</strong>
            <span>Ranked by same-team returners and score tightness</span>
          </div>
          <div className="grid compact-grid">
            {similarMatches.map((row) => (
              <article className="previous-match-card similar-preview-card" key={row.match.id}>
                <div className="previous-match-top similar-match-header">
                  <div>
                    <span>Previous meeting</span>
                    <strong>{formatUkDate(row.match.date)}</strong>
                  </div>
                  <div>
                    <span>Score</span>
                    <em>{row.scoreLabel}</em>
                  </div>
                  <div>
                    <span>Same-team players</span>
                    <strong>{row.sameSide}</strong>
                  </div>
                  <div>
                    <span>Match shape</span>
                    <strong>{row.swapped ? "Swapped" : "Direct"}</strong>
                  </div>
                </div>
                <div className="similar-change-grid">
                  <HistoricalSideCard
                    currentTeam={option.teamA.map((player) => player.name)}
                    historicTeam={row.teamA}
                    label={`${row.labelA} -> Team A`}
                    nameMap={nameMap}
                    tone="a"
                  />
                  <HistoricalSideCard
                    currentTeam={option.teamB.map((player) => player.name)}
                    historicTeam={row.teamB}
                    label={`${row.labelB} -> Team B`}
                    nameMap={nameMap}
                    tone="b"
                  />
                </div>
              </article>
            ))}
          </div>
        </div>
      ) : null}

      <ProfileDebugSection option={option} allOptions={allOptions} />
    </section>
  );
}

export function InteractiveMatchday({
  players,
  matches,
  modelPlayers = players,
  adminLeagueId,
  leagueKey = "matchday"
}: {
  players: Player[];
  matches: Match[];
  modelPlayers?: Player[];
  adminLeagueId?: number;
  leagueKey?: string;
}) {
  const [mode, setMode] = useState<"balance" | "editor" | "generator" | "memory">("balance");
  const [lineupSeed, setLineupSeed] = useState<LineupSeed | null>(null);
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [locksA, setLocksA] = useState<string[]>([]);
  const [locksB, setLocksB] = useState<string[]>([]);
  const [optionIndex, setOptionIndex] = useState<number | null>(null);
  const [cardIndex, setCardIndex] = useState<number | null>(null);
  const [memoryCardOption, setMemoryCardOption] = useState<MatchdayOption | null>(null);
  const [showAllOptions, setShowAllOptions] = useState(false);
  const [memoryCount, setMemoryCount] = useState(6);
  const nameMap = useMemo(() => makeNameMap(players), [players]);
  const playerRecords = useMemo(() => buildPlayerRecords(players, matches), [matches, players]);
  const pairRecords = useMemo(() => buildPairRecords(matches), [matches]);

  const selectedPlayers = useMemo(
    () => players.filter((player) => selectedKeys.includes(normalizeName(player.name))),
    [players, selectedKeys]
  );

  const activeLocksA = useMemo(() => locksA.filter((key) => selectedKeys.includes(key)), [locksA, selectedKeys]);
  const activeLocksB = useMemo(() => locksB.filter((key) => selectedKeys.includes(key)), [locksB, selectedKeys]);
  const canGenerate = selectedPlayers.length === 10;
  const playerCountGap = 10 - selectedPlayers.length;
  const selectedProfileCount = selectedPlayers.filter((player) => player.style_profile).length;
  const profileCoverageLabel = canGenerate
    ? selectedProfileCount > 0
      ? `Profiles active: ${selectedProfileCount}/${selectedPlayers.length}`
      : "Profiles unavailable"
    : "";
  const options = useMemo<MatchdayOption[]>(
    () => canGenerate && (mode === "generator" || mode === "memory") ? bestBalancedSplit(selectedPlayers, activeLocksA, activeLocksB, matches) : [],
    [activeLocksA, activeLocksB, canGenerate, matches, selectedPlayers, mode]
  );
  const smartOptions = useMemo(() => buildSmartOptions(options), [options]);
  const displayedOptions = showAllOptions
    ? options.map((option, index) => ({ option, index, label: `Option ${index + 1}` }))
    : smartOptions;
  const selectedOption = optionIndex !== null ? options[optionIndex] || null : null;
  const cardOption = cardIndex !== null ? options[cardIndex] || null : null;
  const cardExplanation = cardOption ? optionExplanation(cardOption) : null;
  const similarMatches = useMemo(() => similarMatchesForOption(cardOption, matches), [cardOption, matches]);
  const tweaks = useMemo(
    () => cardOption ? findBestTweaks(cardOption.teamA, cardOption.teamB, matches, 2) : null,
    [cardOption, matches]
  );
  const cardMarket = useMemo(() => cardOption ? marketForOption(cardOption, matches) : null, [cardOption, matches]);
  const keyMatchups = useMemo(
    () => cardOption ? crossTeamMatchups(cardOption.teamA, cardOption.teamB, pairRecords.rivals) : [],
    [cardOption, pairRecords.rivals]
  );
  const teamALinks = useMemo(
    () => cardOption ? sameTeamLinks(cardOption.teamA, pairRecords.teammate) : [],
    [cardOption, pairRecords.teammate]
  );
  const teamBLinks = useMemo(
    () => cardOption ? sameTeamLinks(cardOption.teamB, pairRecords.teammate) : [],
    [cardOption, pairRecords.teammate]
  );

  const memoryMatches = useMemo(() => {
    if (!canGenerate || mode !== "memory") return [];
    const today = new Set(selectedKeys);
    return matches
      .map((match) => {
        const rawTeamA = splitTeam(match.team_a);
        const rawTeamB = splitTeam(match.team_b);
        const teamA = rawTeamA.map(normalizeName);
        const teamB = rawTeamB.map(normalizeName);
        const involved = new Set([...teamA, ...teamB]);
        const overlap = countOverlap(today, involved);
        const returningAKeys = teamA.filter((name) => today.has(name));
        const returningBKeys = teamB.filter((name) => today.has(name));
        const score = scoreParts(match.score);
        const lockedOptions = bestBalancedSplit(selectedPlayers, returningAKeys, returningBKeys, matches);
        return {
          match,
          overlap,
          totalGoals: score ? score[0] + score[1] : 0,
          teamA: rawTeamA,
          teamB: rawTeamB,
          returningA: rawTeamA.filter((name) => today.has(normalizeName(name))),
          returningB: rawTeamB.filter((name) => today.has(normalizeName(name))),
          missingA: rawTeamA.filter((name) => !today.has(normalizeName(name))),
          missingB: rawTeamB.filter((name) => !today.has(normalizeName(name))),
          option: lockedOptions[0] || null
        };
      })
      .filter((row) => row.overlap >= 4)
      .sort((a, b) => b.overlap - a.overlap || b.totalGoals - a.totalGoals)
      .slice(0, memoryCount);
  }, [canGenerate, matches, memoryCount, selectedKeys, selectedPlayers, mode]);

  function toggleSelected(key: string) {
    const isSelected = selectedKeys.includes(key);
    setSelectedKeys((current) => {
      return current.includes(key) ? current.filter((item) => item !== key) : [...current, key];
    });
    if (isSelected) {
      setLocksA((locks) => locks.filter((item) => item !== key));
      setLocksB((locks) => locks.filter((item) => item !== key));
    }
    setOptionIndex(null);
    setCardIndex(null);
    setMemoryCardOption(null);
  }

  function toggleLock(key: string, side: "A" | "B") {
    if (side === "A") {
      setLocksB((current) => current.filter((item) => item !== key));
      setLocksA((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current, key]);
    } else {
      setLocksA((current) => current.filter((item) => item !== key));
      setLocksB((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current, key]);
    }
    setOptionIndex(null);
    setCardIndex(null);
    setMemoryCardOption(null);
  }

  function clearSelection() {
    setSelectedKeys([]);
    setLocksA([]);
    setLocksB([]);
    setOptionIndex(null);
    setCardIndex(null);
    setMemoryCardOption(null);
  }

  function selectOption(index: number) {
    setOptionIndex(index);
    setCardIndex(null);
    setMemoryCardOption(null);
  }

  return (
    <div className="interactive-stack matchday-stack">
      <div className="segmented">
        <button className={mode === "balance" ? "active" : ""} onClick={() => setMode("balance")}>Team Generator</button>
        <button className={mode === "editor" ? "active" : ""} onClick={() => { setLineupSeed(null); setMode("editor"); }}>Lineup editor</button>
        <button className={mode === "generator" ? "active" : ""} onClick={() => setMode("generator")}>Legacy selector</button>
        <button className={mode === "memory" ? "active" : ""} onClick={() => setMode("memory")}>Legacy memory</button>
      </div>

      <div className="panel matchday-selection-panel">
        <div className="panel-title-row">
          <div>
            <span className="stat-label">Player pool</span>
            <h2>Today&apos;s Players</h2>
            <small className="muted">Numbers below are public MMR. The Team Generator uses its own strength and uncertainty estimates.</small>
          </div>
          <div className="selection-actions">
            <strong>{selectedPlayers.length} selected</strong>
            <button type="button" onClick={clearSelection}>Clear</button>
          </div>
        </div>
        <div className="toggle-grid">
          {players.map((player) => {
            const key = normalizeName(player.name);
            return (
              <button className={selectedKeys.includes(key) ? "toggle active" : "toggle"} key={player.id} onClick={() => toggleSelected(key)}>
                {playerLabel(player)}
                <span>{Math.round(player.mmr || 0)}</span>
              </button>
            );
          })}
        </div>
      </div>

      {mode === "balance" ? <BalancedMatchday players={modelPlayers} selected={selectedPlayers} matches={matches}
        onEditLineup={seed => { setLineupSeed(seed); setMode("editor"); }}
        locksA={selectedPlayers.filter(p => activeLocksA.includes(normalizeName(p.name))).map(p => p.id)}
        locksB={selectedPlayers.filter(p => activeLocksB.includes(normalizeName(p.name))).map(p => p.id)}
        onLock={(player, side) => toggleLock(normalizeName(player.name), side)}
        onClearLocks={() => { setLocksA([]); setLocksB([]); }}
        onReuseMemory={match => {
          const returning = (team: string | null) => selectedPlayers.filter(player => splitTeam(team).some(name => [normalizeName(player.name), normalizeName(player.display_name)].includes(normalizeName(name)))).map(player => normalizeName(player.name));
          setLocksA(returning(match.team_a)); setLocksB(returning(match.team_b));
        }} /> : mode === "editor" ? <LineupEditor key={`${leagueKey}:${lineupSeed?.revision || "draft"}`} players={players} modelPlayers={modelPlayers} matches={matches} selected={selectedPlayers} seed={lineupSeed}
          storageKey={`lovefive-lineup-v1:${leagueKey}`} onSelectionChange={ids => setSelectedKeys(players.filter(p => ids.includes(p.id)).map(p => normalizeName(p.name)))}
          sideLocks={{ blue: selectedPlayers.filter(p => activeLocksA.includes(normalizeName(p.name))).map(p => p.id), red: selectedPlayers.filter(p => activeLocksB.includes(normalizeName(p.name))).map(p => p.id) }} /> : mode === "generator" ? (
        <>
          {!selectedPlayers.length ? (
            <div className="panel empty-state">
              <h2>Select today's 10 players</h2>
              <p className="muted">The team engine unlocks once exactly 10 players have been picked.</p>
            </div>
          ) : !canGenerate ? (
            <div className="panel empty-state">
              <h2>{playerCountGap > 0 ? `Pick ${playerCountGap} more` : `Remove ${Math.abs(playerCountGap)}`}</h2>
              <p className="muted">Recommendations only appear when exactly 10 players are selected.</p>
            </div>
          ) : (
            <section className="matchday-generator-grid">
              <div className="panel recommendation-panel">
                <div className="panel-title-row">
                  <div>
                    <span className="stat-label">Smart shortlist</span>
                    <h2>Recommended Teams</h2>
                    {profileCoverageLabel ? <span className="stat-label">{profileCoverageLabel}</span> : null}
                  </div>
                  <button className="ghost-button" type="button" onClick={() => setShowAllOptions((value) => !value)}>
                    {showAllOptions ? "Show shortlist" : `View all ${options.length}`}
                  </button>
                </div>

                <div className="recommendation-list">
                  {displayedOptions.map(({ option, label, index }) => {
                    const explanation = optionExplanation(option);
                    const details = recommendationDetails(option, label, options);
                    const selected = optionIndex === index;
                    return (
                      <button
                        className={selected ? "recommendation-card selected" : "recommendation-card"}
                        key={`${label}-${index}-${splitKey(option)}`}
                        type="button"
                        onClick={() => selectOption(index)}
                      >
                        <div className="recommendation-head">
                          <span>{label}</span>
                          <strong>{fmt(option.ranking.recommendationScore)}%</strong>
                        </div>
                        <div className="lineups">
                          <div className="team-panel a">
                            <div className="stat-label">Team A - {fmt(option.avgA)} avg</div>
                            <div className="pill-row">{option.teamA.map((player) => <span className="pill" key={player.id}>{playerLabel(player)}</span>)}</div>
                          </div>
                          <div className="team-panel b">
                            <div className="stat-label">Team B - {fmt(option.avgB)} avg</div>
                            <div className="pill-row">{option.teamB.map((player) => <span className="pill" key={player.id}>{playerLabel(player)}</span>)}</div>
                          </div>
                        </div>
                        <div className="engine-breakdown">
                          <span>{fmt(option.diff, 1)} MMR gap</span>
                          <span>{fmt(option.ranking.closePct)}/100 closeness score</span>
                          <span>{fmt(option.ranking.predictedMargin, 1)} expected margin</span>
                          {option.breakdown.style_enabled ? <span>{fmt(breakdownNumber(option, "style_net"), 1)} profile mix</span> : <span>profiles off</span>}
                          <span>{fmt(breakdownNumber(option, "rotating_keeper_penalty"), 1)} keeper rotation</span>
                        </div>
                        <div className="recommendation-reasons">
                          {details.chips.map((chip) => <span key={chip}>{chip}</span>)}
                        </div>
                        <p>{details.summary || explanation.positives[0]}</p>
                      </button>
                    );
                  })}
                </div>

                {selectedOption ? (
                  <div className="selected-matchup-actions">
                    <span>{fmt(selectedOption.ranking.closePct)}/100 closeness score</span>
                    <button type="button" onClick={() => setCardIndex(optionIndex)}>Use this matchup</button>
                  </div>
                ) : (
                  <p className="muted">Choose a recommended split to open the matchday card.</p>
                )}
              </div>

              <div className="panel locks-panel">
                <h2>Locks</h2>
                <div className="lock-list">
                  {selectedPlayers.map((player) => {
                    const key = normalizeName(player.name);
                    return (
                      <div key={player.id}>
                        <strong>{playerLabel(player)}</strong>
                        <button className={activeLocksA.includes(key) ? "active" : ""} onClick={() => toggleLock(key, "A")}>A</button>
                        <button className={activeLocksB.includes(key) ? "active" : ""} onClick={() => toggleLock(key, "B")}>B</button>
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>
          )}

          {canGenerate && cardOption && cardExplanation ? (
            <section className="panel matchday-card-panel">
              <div className="panel-title-row">
                <div>
                  <span className="stat-label">Selected matchup</span>
                  <h2>Matchday Card</h2>
                </div>
                <button className="ghost-button" type="button" onClick={() => setCardIndex(null)}>Change matchup</button>
              </div>

              {(() => {
                const formA = teamFormPoints(cardOption.teamA, playerRecords);
                const formB = teamFormPoints(cardOption.teamB, playerRecords);
                const chemistryA = teamChemistryScore(cardOption.teamA, pairRecords.teammate);
                const chemistryB = teamChemistryScore(cardOption.teamB, pairRecords.teammate);
                const anchorA = topRated(cardOption.teamA);
                const anchorB = topRated(cardOption.teamB);
                const bestLinkA = teamALinks[0];
                const bestLinkB = teamBLinks[0];
                const dangerA = keyMatchups[0];
                const dangerB = keyMatchups[0];
                const comparisonRows = [
                  { label: "Effective MMR", a: cardOption.avgA, b: cardOption.avgB, left: fmt(cardOption.avgA), right: fmt(cardOption.avgB) },
                  { label: "Recent form", a: formA, b: formB, left: `${formA} pts`, right: `${formB} pts` },
                  { label: "Chemistry", a: chemistryBarValue(chemistryA), b: chemistryBarValue(chemistryB), left: signedFmt(chemistryA, 0), right: signedFmt(chemistryB, 0) },
                  { label: "Anchor rating", a: Number(anchorA?.mmr || 0), b: Number(anchorB?.mmr || 0), left: anchorA ? `${playerLabel(anchorA)} ${fmt(Number(anchorA.mmr || 0))}` : "-", right: anchorB ? `${playerLabel(anchorB)} ${fmt(Number(anchorB.mmr || 0))}` : "-" }
                ];
                return (
                  <>
                    <div className="match-report-hero">
                      <div className="report-team report-team-a">
                        <span>Team A</span>
                        <strong>{fmt(cardOption.avgA)}</strong>
                        <small>avg MMR</small>
                        <div className="mini-lineup">{cardOption.teamA.map((player) => <i key={player.id}>{playerLabel(player)}</i>)}</div>
                      </div>
                      <div className="report-score">
                        <span>Projected score</span>
                        <strong>{cardMarket ? `${fmt(cardMarket.expectedScore.a, 1)} - ${fmt(cardMarket.expectedScore.b, 1)}` : "-"}</strong>
                        <small>{fmt(cardOption.ranking.closePct)}/100 closeness score</small>
                      </div>
                      <div className="report-team report-team-b">
                        <span>Team B</span>
                        <strong>{fmt(cardOption.avgB)}</strong>
                        <small>avg MMR</small>
                        <div className="mini-lineup">{cardOption.teamB.map((player) => <i key={player.id}>{playerLabel(player)}</i>)}</div>
                      </div>
                    </div>

                    <div className="match-report-board">
                      <section className="analysis-centrepiece">
                        <div className="section-subhead"><strong>Team comparison</strong><span>How the two sides stack up</span></div>
                        <div className="comparison-bar-list">
                          {comparisonRows.map((row) => {
                            const max = Math.max(row.a, row.b, 1);
                            return (
                              <div className="comparison-bar-row" key={row.label}>
                                <div className="comparison-value left">{row.left}</div>
                                <div className="comparison-track">
                                  <span className="bar-a" style={{ width: `${Math.max(12, (row.a / max) * 92)}%` }} />
                                  <em>{row.label}</em>
                                  <span className="bar-b" style={{ width: `${Math.max(12, (row.b / max) * 92)}%` }} />
                                </div>
                                <div className="comparison-value right">{row.right}</div>
                              </div>
                            );
                          })}
                        </div>
                        <div className="analysis-signal-row">
                          <div><span>Match potential</span><strong>{fmt(cardOption.ranking.recommendationScore)}%</strong></div>
                          <div><span>Expected margin</span><strong>{fmt(cardOption.ranking.predictedMargin, 1)}</strong></div>
                          {cardOption.breakdown.style_enabled ? (
                            <div><span>Profile mix</span><strong>{fmt(breakdownNumber(cardOption, "style_net"), 1)}</strong></div>
                          ) : (
                            <div><span>Profile mix</span><strong>Off</strong></div>
                          )}
                          <div><span>Keeper rotation</span><strong>{fmt(breakdownNumber(cardOption, "rotating_keeper_penalty"), 1)}</strong></div>
                          <div><span>Repeat risk</span><strong>{fmt(cardOption.similarityPenalty, 1)}</strong></div>
                        </div>
                      </section>

                      <aside className="analyst-verdict">
                        <div className="section-subhead"><strong>Analyst verdict</strong></div>
                        <div className="verdict-copy">
                          {cardExplanation.positives.map((reason) => <p key={reason}>{reason}</p>)}
                          {cardExplanation.risks.map((risk) => <p className="warning" key={risk}>{risk}</p>)}
                        </div>
                        <div className="verdict-tweak">
                          <span>Best tweak</span>
                          {tweaks?.suggestions.length ? (
                            tweaks.suggestions.slice(0, 1).map((suggestion) => (
                              <strong key={`${suggestion.swapA.id}-${suggestion.swapB.id}`}>{playerLabel(suggestion.swapA)} &lt;-&gt; {playerLabel(suggestion.swapB)}</strong>
                            ))
                          ) : (
                            <strong>No tweak needed</strong>
                          )}
                        </div>
                      </aside>
                    </div>

                    <div className="team-report-grid">
                      <section className="team-report-card team-report-a">
                        <div className="section-subhead"><strong>Team A</strong></div>
                        <div className="pill-row">{cardOption.teamA.map((player) => <span className="pill" key={player.id}>{playerLabel(player)}</span>)}</div>
                        <div className="team-report-stats">
                          <div><span>Form</span><strong>{formA} pts</strong></div>
                          <div><span>Best link</span><strong>{bestLinkA ? `${playerLabel(bestLinkA.a)} + ${playerLabel(bestLinkA.b)}` : "-"}</strong></div>
                          <div><span>Likely rival</span><strong>{dangerA ? `${playerLabel(dangerA.a)} vs ${playerLabel(dangerA.b)}` : "-"}</strong></div>
                        </div>
                      </section>
                      <section className="team-report-card team-report-b">
                        <div className="section-subhead"><strong>Team B</strong></div>
                        <div className="pill-row">{cardOption.teamB.map((player) => <span className="pill" key={player.id}>{playerLabel(player)}</span>)}</div>
                        <div className="team-report-stats">
                          <div><span>Form</span><strong>{formB} pts</strong></div>
                          <div><span>Best link</span><strong>{bestLinkB ? `${playerLabel(bestLinkB.a)} + ${playerLabel(bestLinkB.b)}` : "-"}</strong></div>
                          <div><span>Likely rival</span><strong>{dangerB ? `${playerLabel(dangerB.b)} vs ${playerLabel(dangerB.a)}` : "-"}</strong></div>
                        </div>
                      </section>
                    </div>
                  </>
                );
              })()}

              <div className="scouting-grid">
                <section className="scouting-card">
                  <div className="section-subhead"><strong>Key matchups</strong><span>Rivalries across the two teams</span></div>
                  <div className="scouting-list">
                    {keyMatchups.length ? keyMatchups.map(({ a, b, row }) => (
                      <div key={`${a.id}-${b.id}`}>
                        <strong>{playerLabel(a)} vs {playerLabel(b)}</strong>
                        <span>{row.matches} meetings</span>
                        <em>{rivalrySummary(row, a, b)}</em>
                      </div>
                    )) : <p className="muted">No strong rivalry history yet.</p>}
                  </div>
                </section>
                <section className="scouting-card">
                  <div className="section-subhead"><strong>Best teammate links</strong><span>Strongest same-side history</span></div>
                  <div className="scouting-list link-scouting-list">
                    {[...teamALinks, ...teamBLinks].slice(0, 6).map(({ a, b, row }) => (
                      <TeamLinkRow
                        a={a}
                        b={b}
                        row={row}
                        teamLabel={cardOption.teamA.some((player) => player.id === a.id) ? "Team A" : "Team B"}
                        key={`${a.id}-${b.id}`}
                      />
                    ))}
                    {!teamALinks.length && !teamBLinks.length ? <p className="muted">Not enough teammate history yet.</p> : null}
                  </div>
                </section>
              </div>

              <div className="player-card-section">
                <div className="section-subhead"><strong>Matchday player cards</strong><span>Form, record, best link and likely rival</span></div>
                <div className="player-card-columns">
                  {[["A", cardOption.teamA, cardOption.teamB], ["B", cardOption.teamB, cardOption.teamA]].map(([side, ownTeam, opponentTeam]) => (
                    <div className="player-card-team" key={String(side)}>
                      <span className="stat-label">Team {String(side)}</span>
                      {(ownTeam as Player[]).map((player) => {
                        const record = playerRecords.get(normalizeName(player.name));
                        const teammate = bestTeammate(player, ownTeam as Player[], pairRecords.teammate);
                        const rival = playerRival(player, opponentTeam as Player[], pairRecords.rivals);
                        return (
                          <article className="matchday-player-card" key={player.id}>
                            <div>
                              <strong>{playerLabel(player)}</strong>
                              <span>{Math.round(player.mmr || 0)} MMR {record?.streak ? `| ${record.streak}` : ""}</span>
                            </div>
                            <div className="form-dots compact">
                              {formDots(record?.form || []).map((result, index) => <i className={result} key={`${player.id}-${index}`}>{result}</i>)}
                            </div>
                            <small>{record?.played || 0} matches | {record?.played ? fmt(((record.wins || 0) / record.played) * 100, 1) : "0.0"}% wins</small>
                            <small>Best link: {teammate ? playerLabel(teammate.mate) : "-"}</small>
                            <small>Likely rival: {rival ? playerLabel(rival.opponent) : "-"}</small>
                          </article>
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>

              {similarMatches.length ? (
                <div className="similar-match-list">
                  <div className="section-subhead">
                    <strong>Closest previous games</strong>
                    <span>Ranked by same-team returners and score tightness</span>
                  </div>
                  <div className="grid compact-grid">
                    {similarMatches.map((row) => (
                      <article className="previous-match-card similar-preview-card" key={row.match.id}>
                        <div className="previous-match-top similar-match-header">
                          <div>
                            <span>Previous meeting</span>
                            <strong>{formatUkDate(row.match.date)}</strong>
                          </div>
                          <div>
                            <span>Score</span>
                            <em>{row.scoreLabel}</em>
                          </div>
                          <div>
                            <span>Same-team players</span>
                            <strong>{row.sameSide}</strong>
                          </div>
                          <div>
                            <span>Match shape</span>
                            <strong>{row.swapped ? "Swapped" : "Direct"}</strong>
                          </div>
                        </div>
                        <div className="similar-change-grid">
                          <HistoricalSideCard
                            currentTeam={cardOption.teamA.map((player) => player.name)}
                            historicTeam={row.teamA}
                            label={`${row.labelA} -> Team A`}
                            nameMap={nameMap}
                            tone="a"
                          />
                          <HistoricalSideCard
                            currentTeam={cardOption.teamB.map((player) => player.name)}
                            historicTeam={row.teamB}
                            label={`${row.labelB} -> Team B`}
                            nameMap={nameMap}
                            tone="b"
                          />
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              ) : null}

              <ProfileDebugSection option={cardOption} allOptions={options} />
            </section>
          ) : null}
        </>
      ) : !canGenerate ? (
        <div className="panel empty-state">
          <h2>Select today's 10 players</h2>
          <p className="muted">Matchday Memory unlocks once the full squad is selected, then it finds old games with the closest player overlap.</p>
        </div>
      ) : (
        <section className="memory-workspace">
          <div className="panel memory-control-panel">
            <div className="panel-title-row">
              <div>
                <span className="stat-label">Historical templates</span>
                <h2>Matchday Memory</h2>
              </div>
              <strong className="muted">{selectedPlayers.length} selected</strong>
            </div>
            <div className="control-bar compact">
              <label>
                <span>Reference games</span>
                <input min={1} max={10} value={memoryCount} type="range" onChange={(event) => setMemoryCount(Number(event.target.value))} />
              </label>
              <div className="range-value">{memoryCount}</div>
            </div>
            <div className="story-list">
              <div>
                <span>How it works</span>
                <strong>Pick a memory match to reuse its shape</strong>
                <small>Players from that old game keep their old side where possible, then the engine fills any gaps.</small>
              </div>
            </div>
          </div>

          <div className="memory-match-list">
            {memoryMatches.map((row) => {
              const selected = Boolean(memoryCardOption && row.option && splitKey(memoryCardOption) === splitKey(row.option));
              return (
                <button
                  className={selected ? "previous-match-card similar-preview-card memory-preview-card selected" : "previous-match-card similar-preview-card memory-preview-card"}
                  disabled={!row.option}
                  key={row.match.id}
                  onClick={() => row.option ? setMemoryCardOption(row.option) : null}
                  type="button"
                >
                  <div className="previous-match-top similar-match-header">
                    <div>
                      <span>Previous match</span>
                      <strong>{formatUkDate(row.match.date)}</strong>
                    </div>
                    <div>
                      <span>Score</span>
                      <em>{row.match.score || "-"}</em>
                    </div>
                    <div>
                      <span>Same players</span>
                      <strong>{row.overlap}</strong>
                    </div>
                  </div>

                  <div className="similar-change-grid">
                    <HistoricalSideCard
                      currentTeam={row.option ? row.option.teamA.map((player) => player.name) : []}
                      historicTeam={row.teamA}
                      label="Old Team A"
                      nameMap={nameMap}
                      tone="a"
                    />
                    <HistoricalSideCard
                      currentTeam={row.option ? row.option.teamB.map((player) => player.name) : []}
                      historicTeam={row.teamB}
                      label="Old Team B"
                      nameMap={nameMap}
                      tone="b"
                    />
                  </div>

                  <div className="memory-card-foot">
                    <span>{row.option ? "Use this matchup" : "Select an even number of players to use this"}</span>
                  </div>
                </button>
              );
            })}
          </div>

          <div className="memory-selected-card">
            {canGenerate && memoryCardOption ? (
              <MatchdayAnalysisCard
                option={memoryCardOption}
                matches={matches}
                nameMap={nameMap}
                playerRecords={playerRecords}
                pairRecords={pairRecords}
                allOptions={options}
                onClose={() => setMemoryCardOption(null)}
              />
            ) : (
              <div className="panel empty-state">
                <h2>Select a memory match</h2>
                <p className="muted">The matchday card will appear here once you choose one of the historical templates.</p>
              </div>
            )}
          </div>
        </section>
      )}
      {adminLeagueId ? <BalanceComparison key={adminLeagueId} leagueId={adminLeagueId} selected={selectedPlayers}
        locksA={selectedPlayers.filter(p => activeLocksA.includes(normalizeName(p.name))).map(p => p.id)}
        locksB={selectedPlayers.filter(p => activeLocksB.includes(normalizeName(p.name))).map(p => p.id)} current={options[0]} /> : null}
    </div>
  );
}
