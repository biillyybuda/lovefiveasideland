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

function buildSmartOptions(options: MatchdayOption[]) {
  const picked: SmartOption[] = [];
  const seen = new Set<string>();

  function add(label: string, index: number) {
    const option = options[index];
    if (!option) return;
    const key = splitKey(option);
    if (seen.has(key)) return;
    picked.push({ option, label, index });
    seen.add(key);
  }

  add("Best overall", 0);
  add("Closest expected game", bestIndex(options, (option) => option.ranking.predictedMargin, "asc"));
  add("Most even ratings", bestIndex(options, (option) => option.diff, "asc"));
  add("Highest chemistry balance", bestIndex(options, (option) => option.chemistryGap, "asc"));

  for (let index = 0; picked.length < 3 && index < options.length; index += 1) {
    add("Strong alternative", index);
  }

  return picked.slice(0, 3);
}

function bestIndex(options: MatchdayOption[], getter: (option: MatchdayOption) => number, direction: "asc" | "desc") {
  if (!options.length) return -1;
  return options.reduce((best, option, index) => {
    const current = getter(option);
    const previous = getter(options[best]);
    return direction === "asc" ? current < previous ? index : best : current > previous ? index : best;
  }, 0);
}

function optionExplanation(option: MatchdayOption) {
  const positives: string[] = [];
  const risks: string[] = [];
  const closePct = option.ranking.closePct;

  if (option.diff <= 10) positives.push("Ratings are extremely tight, so the game should not need one side to overperform.");
  else if (option.diff <= 25) positives.push("MMR balance is healthy enough for a competitive game.");
  else risks.push("The MMR gap is noticeable, so one team may start with a small edge.");

  if (closePct >= 70) positives.push("The model expects a close game rather than a runaway result.");
  else if (closePct < 45) risks.push("The close-game chance is lower than ideal.");

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
  const chips: string[] = [];

  if (option.ranking.recommendationScore >= bestScore - tolerance) chips.push("Best combined score");
  if (option.ranking.closePct >= bestClose - tolerance) chips.push("Highest tight-game chance");
  if (option.ranking.predictedMargin <= bestMargin + tolerance) chips.push("Lowest expected margin");
  if (option.diff <= bestMmrGap + tolerance) chips.push("Most even ratings");
  if (option.chemistryGap <= bestChemGap + tolerance) chips.push("Best chemistry balance");
  if (option.similarityPenalty <= bestRepeatRisk + tolerance) chips.push("Lowest repeat risk");

  const uniqueChips = [...new Set(chips)].slice(0, 3);

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
      chips: uniqueChips.length ? uniqueChips : ["Lowest expected margin", `${fmt(option.ranking.closePct)}% tight-game chance`],
      summary: "The model expects this one to stay closest on the scoreboard."
    };
  }

  return {
    chips: uniqueChips.length ? uniqueChips : ["Best overall model score", `${fmt(option.ranking.closePct)}% tight-game chance`],
    summary: "The best all-round blend of rating balance, projected margin, proven chemistry and repeat-risk checks."
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

function MatchdayAnalysisCard({
  option,
  matches,
  nameMap,
  playerRecords,
  pairRecords,
  onClose
}: {
  option: MatchdayOption;
  matches: Match[];
  nameMap: Map<string, string>;
  playerRecords: Map<string, PlayerRecord>;
  pairRecords: { teammate: Map<string, PairRecord>; rivals: Map<string, PairRecord> };
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
          <small>{fmt(option.ranking.closePct)}% tight-game chance</small>
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
                    <em>{row.match.score || "-"}</em>
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
    </section>
  );
}

export function InteractiveMatchday({
  players,
  matches
}: {
  players: Player[];
  matches: Match[];
}) {
  const [mode, setMode] = useState<"generator" | "memory">("generator");
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
  const options = useMemo<MatchdayOption[]>(
    () => canGenerate ? bestBalancedSplit(selectedPlayers, activeLocksA, activeLocksB, matches) : [],
    [activeLocksA, activeLocksB, canGenerate, matches, selectedPlayers]
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
    if (!canGenerate) return [];
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
  }, [canGenerate, matches, memoryCount, selectedKeys, selectedPlayers]);

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
        <button className={mode === "generator" ? "active" : ""} onClick={() => setMode("generator")}>AI Team Generator</button>
        <button className={mode === "memory" ? "active" : ""} onClick={() => setMode("memory")}>Matchday Memory</button>
      </div>

      <div className="panel matchday-selection-panel">
        <div className="panel-title-row">
          <div>
            <span className="stat-label">Player pool</span>
            <h2>Today&apos;s Players</h2>
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

      {mode === "generator" ? (
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
                  </div>
                  <button className="ghost-button" type="button" onClick={() => setShowAllOptions((value) => !value)}>
                    {showAllOptions ? "Show top 3" : `View all ${options.length}`}
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
                          <span>{fmt(option.ranking.closePct)}% tight-game chance</span>
                          <span>{fmt(option.ranking.predictedMargin, 1)} expected margin</span>
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
                    <span>{fmt(selectedOption.ranking.closePct)}% tight-game chance</span>
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
                        <small>{fmt(cardOption.ranking.closePct)}% tight-game chance</small>
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
                            <em>{row.match.score || "-"}</em>
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
    </div>
  );
}
