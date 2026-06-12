import {
  displayName,
  formatTeam,
  formatUkDate,
  makeNameMap,
  normalizeName,
  resultFor,
  scoreParts,
  splitTeam,
  type Match,
  type MmrHistory,
  type Player
} from "./demo-data";
import { calculateMatchMmrUpdates, expectedScore, scoreToResult, STARTING_MMR } from "./mmr-engine";

type ReportFact = { label: string; value: string; detail: string; people?: string[] };
type ReportChange = { label: string; value: string; detail: string; people?: string[] };
type ReportNote = { title: string; body: string; people: string[] };
type ReportPlayerNote = {
  name: string;
  team: "Team A" | "Team B";
  tone: "a" | "b";
  tag: string;
  detail: string;
  delta: number | null;
};
type PlayerDetailAngle = { tag: string; detail: string; priority: number };
type PreviousMeetingSide = {
  label: string;
  tone: "a" | "b";
  historicTeam: string[];
  subs: string[];
  missing: string[];
  keptCount: number;
  totalCount: number;
};
type PreviousMeeting = {
  dateLabel: string;
  scoreLabel: string;
  resultLabel: string;
  sharedPlayers: number;
  sameSidePlayers: number;
  swapped: boolean;
  detail: string;
  sides: [PreviousMeetingSide, PreviousMeetingSide];
};

type StoryCandidate = {
  key: string;
  priority: number;
  line: string;
  factLabel?: string;
  factValue?: string;
  factDetail?: string;
  headlines?: string[];
  people?: string[];
};

export type MatchReport = {
  match: Match;
  headline: string;
  summary: string;
  resultLabel: string;
  dateLabel: string;
  scoreLabel: string;
  winnerLabel: string;
  teamA: string[];
  teamB: string[];
  rating: {
    avgA: number;
    avgB: number;
    favourite: "Team A" | "Team B" | "Level";
    expectedA: number;
    upset: boolean;
  };
  facts: ReportFact[];
  storylines: string[];
  changes: ReportChange[];
  playerNotes: ReportPlayerNote[];
  previousMeeting: PreviousMeeting | null;
  notes: ReportNote[];
};

export function buildMatchReport({
  match,
  matches,
  players,
  mmrHistory
}: {
  match: Match;
  matches: Match[];
  players: Player[];
  mmrHistory: MmrHistory[];
}): MatchReport {
  const nameMap = makeNameMap(players);
  const score = scoreParts(match.score);
  const result = scoreToResult(match);
  const teamA = formatTeam(match.team_a, nameMap);
  const teamB = formatTeam(match.team_b, nameMap);
  const teamAPlayers = playersForTeam(match.team_a, players);
  const teamBPlayers = playersForTeam(match.team_b, players);
  const avgA = average(teamAPlayers.map((player) => ratingBeforeMatch(player, match, mmrHistory)));
  const avgB = average(teamBPlayers.map((player) => ratingBeforeMatch(player, match, mmrHistory)));
  const expectedA = expectedScore(avgA || STARTING_MMR, avgB || STARTING_MMR);
  const winnerLabel = result === "DRAW" ? "Draw" : result === "A" ? "Team A" : "Team B";
  const loserLabel = result === "A" ? "Team B" : result === "B" ? "Team A" : "No loser";
  const winnerAvg = result === "A" ? avgA : result === "B" ? avgB : average([avgA, avgB]);
  const loserAvg = result === "A" ? avgB : result === "B" ? avgA : average([avgA, avgB]);
  const ratingGap = Math.round(Math.abs(avgA - avgB));
  const margin = score ? Math.abs(score[0] - score[1]) : 0;
  const totalGoals = score ? score[0] + score[1] : 0;
  const upset = result !== "DRAW" && winnerAvg + 8 < loserAvg;
  const favourite = Math.abs(avgA - avgB) < 8 ? "Level" : avgA > avgB ? "Team A" : "Team B";
  const candidates = storyCandidatesFor({
    match,
    matches,
    players,
    mmrHistory,
    nameMap,
    result,
    winnerLabel,
    loserLabel,
    avgA,
    avgB,
    upset,
    favourite,
    margin,
    totalGoals
  });
  const headline = headlineFor({ match, result, winnerLabel, margin, totalGoals, upset, candidates });
  const leadFact = leadFactFor(candidates, { result, winnerLabel, margin, totalGoals, upset });
  const changes = buildChangeItems(
    candidates,
    leadFact,
    seasonAwardChanges({ match, matches, players, mmrHistory, nameMap })
  );
  const storylines = buildStorylines({
    match,
    matches,
    result,
    winnerLabel,
    loserLabel,
    margin,
    totalGoals,
    upset,
    favourite,
    ratingGap,
    candidates,
    leadFact
  });
  const usedReportPeople = new Set([...peopleFromFact(leadFact), ...changes.flatMap((item) => item.people || [])]);
  const notes = [
    pairNote(match, matches, nameMap, result, usedReportPeople),
    rivalryNote(match, matches, nameMap, result, usedReportPeople)
  ].filter((note): note is ReportNote => Boolean(note));

  return {
    match,
    headline,
    summary: summaryFor({ match, result, winnerLabel, loserLabel, margin, totalGoals, upset, favourite, ratingGap, leadFact, changes }),
    resultLabel: result === "DRAW" ? "Draw" : `${winnerLabel} win`,
    dateLabel: formatUkDate(match.date),
    scoreLabel: match.score || "-",
    winnerLabel,
    teamA,
    teamB,
    rating: {
      avgA,
      avgB,
      favourite,
      expectedA,
      upset
    },
    facts: [
      leadFact,
      ratingFactFor({ favourite, ratingGap, winnerLabel, upset, avgA, avgB }),
      scoreProfileFact(match, matches, margin, totalGoals, leadFact.label)
    ],
    storylines,
    changes,
    playerNotes: buildPlayerNotes({ match, matches, players, mmrHistory, nameMap, result, margin, candidates }),
    previousMeeting: previousMeetingFor(match, matches, nameMap),
    notes
  };
}

function storyCandidatesFor({
  match,
  matches,
  players,
  mmrHistory,
  nameMap,
  result,
  winnerLabel,
  loserLabel,
  avgA,
  avgB,
  upset,
  favourite,
  margin,
  totalGoals
}: {
  match: Match;
  matches: Match[];
  players: Player[];
  mmrHistory: MmrHistory[];
  nameMap: Map<string, string>;
  result: "A" | "B" | "DRAW";
  winnerLabel: string;
  loserLabel: string;
  avgA: number;
  avgB: number;
  upset: boolean;
  favourite: "Team A" | "Team B" | "Level";
  margin: number;
  totalGoals: number;
}) {
  const teamA = splitTeam(match.team_a);
  const teamB = splitTeam(match.team_b);
  const winnerTeam = result === "A" ? teamA : result === "B" ? teamB : [];
  const loserTeam = result === "A" ? teamB : result === "B" ? teamA : [];
  const candidates: StoryCandidate[] = [];
  const winnerAvg = result === "A" ? avgA : avgB;
  const loserAvg = result === "A" ? avgB : avgA;
  const ratingGap = Math.round(Math.abs(avgA - avgB));

  if (result !== "DRAW" && upset) {
    const gap = Math.round(loserAvg - winnerAvg);
    candidates.push({
      key: "rating-upset",
      priority: 110 + Math.min(gap, 30),
      line: `${winnerLabel} were ${gap} MMR lower on average and still took the win.`,
      factLabel: "Rating swing",
      factValue: `${winnerLabel} beat the numbers`,
      factDetail: `${gap} MMR lower on average`,
      headlines: [
        `${winnerLabel} win against the numbers`,
        `${winnerLabel} make a mess of the ratings`,
        `${winnerLabel} turn the rating gap upside down`,
        `${winnerLabel} take the underdog route`
      ]
    });
  }

  if (result !== "DRAW" && !upset && favourite === winnerLabel && ratingGap >= 12) {
    candidates.push({
      key: "rating-edge-held",
      priority: 62 + Math.min(ratingGap, 28),
      line: `${winnerLabel} had the stronger average rating and made that edge count.`,
      factLabel: "Rating angle",
      factValue: `${winnerLabel} made it count`,
      factDetail: `${ratingGap} MMR average edge`,
      headlines: [
        `${winnerLabel} back up the rating edge`,
        `${winnerLabel} make the numbers count`,
        `${winnerLabel} turn the pre-match edge into points`
      ]
    });
  }

  if (result === "DRAW") {
    candidates.push({
      key: "draw-balance",
      priority: 85,
      line:
        favourite === "Level"
          ? "The ratings had this one close before kick-off and the scoreline followed the same script."
          : `${favourite} had the rating edge, but neither side could force a winner.`,
      factLabel: "Balance check",
      factValue: favourite === "Level" ? "Ratings were tight" : `${favourite} held`,
      factDetail: ratingGap ? `${ratingGap} MMR average gap` : "Almost nothing in it",
      headlines: [
        "Nothing splits a stubborn matchup",
        "No winner, but plenty of evidence",
        "The sides cancel each other out",
        "A draw that keeps the argument open"
      ]
    });
  }

  addScoreCandidates(candidates, match, matches, result, winnerLabel, loserLabel, margin, totalGoals);
  addPlayerRunCandidates(candidates, match, matches, players, result, winnerTeam, loserTeam);
  addRatingMovementCandidates(candidates, match, players, mmrHistory);
  addPartnershipCandidates(candidates, match, matches, nameMap, result);
  addRivalryCandidates(candidates, match, matches, nameMap, result);

  return sortStoryCandidates(candidates, match);
}

function addScoreCandidates(
  candidates: StoryCandidate[],
  match: Match,
  matches: Match[],
  result: "A" | "B" | "DRAW",
  winnerLabel: string,
  loserLabel: string,
  margin: number,
  totalGoals: number
) {
  const previousScores = matchesBefore(match, matches)
    .map((row) => scoreParts(row.score))
    .filter((score): score is [number, number] => Boolean(score));
  const previousTotals = previousScores.map((score) => score[0] + score[1]);
  const maxPreviousTotal = previousTotals.length ? Math.max(...previousTotals) : 0;
  const averagePreviousTotal = previousTotals.length ? average(previousTotals) : 0;

  if (totalGoals && previousTotals.length >= 5 && totalGoals >= maxPreviousTotal) {
    candidates.push({
      key: "archive-high-goals",
      priority: 92,
      line: `At ${totalGoals} total goals, this matched or beat every earlier scoreline in the current archive.`,
      factLabel: "Score heat",
      factValue: "Archive shootout",
      factDetail: `${totalGoals} total goals`,
      headlines: [
        `${winnerLabel} survive the wildest scoreline`,
        `${winnerLabel} come through a ${totalGoals}-goal storm`,
        `${winnerLabel} win the shootout`
      ]
    });
    return;
  }

  if (totalGoals && averagePreviousTotal && totalGoals >= averagePreviousTotal + 5) {
    candidates.push({
      key: "high-goals",
      priority: 74,
      line: `This ran hotter than the usual game: ${totalGoals} goals against an archive average of ${averagePreviousTotal.toFixed(1)}.`,
      factLabel: "Score heat",
      factValue: "Open game",
      factDetail: `${totalGoals} total goals`,
      headlines: [
        `${winnerLabel} come through an open one`,
        `${winnerLabel} win a busy night`,
        `${winnerLabel} find enough in a high-scoring game`
      ]
    });
  }

  if (result !== "DRAW" && margin === 1) {
    const resultScore = resultScoreLabel(match, result);
    candidates.push({
      key: "one-goal-game",
      priority: 106,
      line: `${winnerLabel} only had a single goal of breathing room in a ${resultScore} finish, so this one stayed alive right to the end.`,
      factLabel: "Fine margins",
      factValue: "One-goal game",
      factDetail: match.score ? `${match.score} final score` : `${winnerLabel} edged ${loserLabel}`,
      headlines: [
        `${winnerLabel} edge a tight contest`,
        `${winnerLabel} win by the finest margin`,
        `${winnerLabel} squeeze through by one`,
        `${winnerLabel} settle it by a single goal`
      ]
    });
  } else if (result !== "DRAW" && margin >= 5) {
    candidates.push({
      key: "clear-margin",
      priority: 82 + Math.min(margin, 8),
      line: `${winnerLabel} put real daylight on the scoreboard with a ${margin}-goal gap.`,
      factLabel: "Separation",
      factValue: `${margin}-goal gap`,
      factDetail: `${winnerLabel} pulled away`,
      headlines: [
        `${winnerLabel} put daylight on the scoreboard`,
        `${winnerLabel} turn it into a statement`,
        `${winnerLabel} leave no doubt late on`
      ]
    });
  }
}

function addPlayerRunCandidates(
  candidates: StoryCandidate[],
  match: Match,
  matches: Match[],
  players: Player[],
  result: "A" | "B" | "DRAW",
  winnerTeam: string[],
  loserTeam: string[]
) {
  const nonLosingTeam = result === "DRAW" ? [...splitTeam(match.team_a), ...splitTeam(match.team_b)] : winnerTeam;
  const winningRuns = winnerTeam
    .map((name) => ({ key: normalizeName(name), name: displayFromName(name, players), count: currentRun(name, "W", match, matches) }))
    .filter((row) => row.count >= 3)
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
    .slice(0, 4);

  for (const row of winningRuns) {
    candidates.push({
      key: `winning-run-${row.key}`,
      priority: 98 + row.count,
      line: `${row.name} extends a ${row.count}-game winning run.`,
      factLabel: "Streak watch",
      factValue: `${row.name} stays hot`,
      factDetail: `${row.count} wins in a row`,
      people: [row.key],
      headlines: [
        `${row.name} keeps the run alive`,
        `${row.name} keeps the form run alive`,
        `${row.name} stretches the winning run`,
        `${row.name} stays hot`
      ]
    });
  }

  const unbeatenRuns = nonLosingTeam
    .map((name) => ({
      key: normalizeName(name),
      name: displayFromName(name, players),
      count: currentRunWhere(name, match, matches, (item) => item !== "L"),
      winningCount: currentRun(name, "W", match, matches)
    }))
    .filter((row) => row.count >= 4 && row.winningCount < 3)
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
    .slice(0, 4);

  for (const row of unbeatenRuns) {
    candidates.push({
      key: `unbeaten-run-${row.key}`,
      priority: 86 + row.count,
      line: `${row.name} is now ${row.count} games unbeaten.`,
      factLabel: "Momentum watch",
      factValue: `${row.name} stays unbeaten`,
      factDetail: `${row.count} without defeat`,
      people: [row.key],
      headlines: [
        `${row.name} keeps the unbeaten run intact`,
        `${row.name} avoids the loss again`,
        `${row.name} keeps finding a way`
      ]
    });
  }

  const bounceBacks = winnerTeam
    .map((name) => ({ key: normalizeName(name), name: displayFromName(name, players), count: previousRun(name, "L", match, matches) }))
    .filter((row) => row.count >= 2)
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
    .slice(0, 4);

  for (const row of bounceBacks) {
    candidates.push({
      key: `bounce-${row.key}`,
      priority: 96 + row.count,
      line: `${row.name} snaps a ${row.count}-game losing run with this win.`,
      factLabel: "Bounce watch",
      factValue: `${row.name} gets the reset`,
      factDetail: `${row.count}-game losing run ended`,
      people: [row.key],
      headlines: [
        `${row.name} gets the bounce at last`,
        `${row.name} turns the run around`,
        `${row.name} gets back on the right side`,
        `${row.name} ends the wait`
      ]
    });
  }

  const haltedRuns = loserTeam
    .map((name) => ({ key: normalizeName(name), name: displayFromName(name, players), count: previousRun(name, "W", match, matches) }))
    .filter((row) => row.count >= 3)
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
    .slice(0, 4);

  for (const row of haltedRuns) {
    candidates.push({
      key: `halted-run-${row.key}`,
      priority: 84 + row.count,
      line: `${row.name}'s ${row.count}-game winning run comes to an end.`,
      factLabel: "Run stopped",
      factValue: `${row.name} cooled off`,
      factDetail: `${row.count}-game winning run ended`,
      people: [row.key],
      headlines: [
        `${row.name}'s run is finally stopped`,
        `${row.name} is brought back down`,
        `${row.name}'s streak ends here`
      ]
    });
  }

  const losingRuns = loserTeam
    .map((name) => ({ key: normalizeName(name), name: displayFromName(name, players), count: currentRun(name, "L", match, matches) }))
    .filter((row) => row.count >= 3)
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
    .slice(0, 3);

  for (const row of losingRuns) {
    candidates.push({
      key: `rough-run-${row.key}`,
      priority: 68 + row.count,
      line: `${row.name}'s wait for a reset continues after ${row.count} straight defeats.`,
      factLabel: "Pressure point",
      factValue: `${row.name} is under pressure`,
      factDetail: `${row.count} defeats in a row`,
      people: [row.key],
      headlines: [
        `${row.name} is left waiting for the bounce`,
        `${row.name}'s rough run grows`,
        `${row.name} still needs the turnaround`
      ]
    });
  }
}

function addRatingMovementCandidates(candidates: StoryCandidate[], match: Match, players: Player[], mmrHistory: MmrHistory[]) {
  const changes = matchMmrChanges(match, players, mmrHistory);
  const gains = changes
    .filter((row) => row.delta >= 8)
    .sort((a, b) => b.delta - a.delta || a.name.localeCompare(b.name))
    .slice(0, 4);

  gains.forEach((gain, index) => {
    candidates.push({
      key: `rating-lift-${gain.key}`,
      priority: 76 + Math.min(gain.delta, 16) - index * 3,
      line:
        index === 0
          ? `${gain.name} takes the biggest rating lift from this match at ${signed(gain.delta)} MMR.`
          : `${gain.name} also leaves with a ${signed(gain.delta)} MMR lift from the result.`,
      factLabel: "Rating lift",
      factValue: `${gain.name} ${signed(gain.delta)}`,
      factDetail: index === 0 ? "Best rating move from the game" : "Rating move from the game",
      people: [gain.key],
      headlines:
        index === 0
          ? [
              `${gain.name} gets the rating lift`,
              `${gain.name} leaves with the biggest boost`,
              `${gain.name} cashes in on the result`
            ]
          : undefined
    });
  });

  const drops = changes
    .filter((row) => row.delta <= -8)
    .sort((a, b) => a.delta - b.delta || a.name.localeCompare(b.name))
    .slice(0, 3);

  drops.forEach((drop, index) => {
    candidates.push({
      key: `rating-hit-${drop.key}`,
      priority: 72 + Math.min(Math.abs(drop.delta), 16) - index * 3,
      line:
        index === 0
          ? `${drop.name} took the biggest rating hit from this match at ${signed(drop.delta)} MMR.`
          : `${drop.name} was also pulled down by ${signed(drop.delta)} MMR after the defeat.`,
      factLabel: "Rating hit",
      factValue: `${drop.name} ${signed(drop.delta)}`,
      factDetail: index === 0 ? "Biggest rating drop from the game" : "Rating move from the game",
      people: [drop.key]
    });
  });
}

function addPartnershipCandidates(
  candidates: StoryCandidate[],
  match: Match,
  matches: Match[],
  nameMap: Map<string, string>,
  result: "A" | "B" | "DRAW"
) {
  const teams = result === "DRAW"
    ? [splitTeam(match.team_a), splitTeam(match.team_b)]
    : [splitTeam(result === "A" ? match.team_a : match.team_b)];

  for (const team of teams) {
    const best = pairRecords(team, matchesUpTo(match, matches), nameMap)
      .filter((row) => row.matches >= 4 && row.pointsRate >= 0.65)
      .sort((a, b) => b.pointsRate - a.pointsRate || b.matches - a.matches)[0];
    if (!best) continue;

    const rate = Math.round(best.pointsRate * 100);
    candidates.push({
      key: `pair-${normalizeName(best.a)}-${normalizeName(best.b)}`,
      priority: 80 + Math.min(best.matches, 8),
      line: `${best.a} + ${best.b} keep building the link: ${best.wins}-${best.draws}-${best.losses} together across ${best.matches} games.`,
      factLabel: "Partnership watch",
      factValue: `${best.a} + ${best.b}`,
      factDetail: `${rate}% points rate together`,
      people: [normalizeName(best.a), normalizeName(best.b)],
      headlines: [
        `${best.a} and ${best.b} keep clicking`,
        `${best.a} + ${best.b} keep the link strong`,
        `${best.a} and ${best.b} add another one`
      ]
    });
  }
}

function addRivalryCandidates(
  candidates: StoryCandidate[],
  match: Match,
  matches: Match[],
  nameMap: Map<string, string>,
  result: "A" | "B" | "DRAW"
) {
  const teamA = splitTeam(match.team_a);
  const teamB = splitTeam(match.team_b);
  const best = rivalryRecords(teamA, teamB, matchesUpTo(match, matches), nameMap)
    .filter((row) => row.matches >= 4)
    .sort((a, b) => b.matches - a.matches || b.balance - a.balance)[0];

  if (!best) return;

  if (result === "DRAW") {
    candidates.push({
      key: `rivalry-draw-${normalizeName(best.a)}-${normalizeName(best.b)}`,
      priority: 72 + Math.min(best.matches, 10),
      line: `${best.a} and ${best.b} split the latest round of a ${best.matches}-meeting rivalry.`,
      factLabel: "Rivalry watch",
      factValue: `${best.a} vs ${best.b}`,
      factDetail: `${best.matches} meetings now`,
      people: [normalizeName(best.a), normalizeName(best.b)],
      headlines: [
        `${best.a} and ${best.b} leave it unresolved`,
        `${best.a} vs ${best.b} stays live`,
        `${best.a} and ${best.b} split the latest round`
      ]
    });
    return;
  }

  const winner = result === "A" ? best.a : best.b;
  const loser = result === "A" ? best.b : best.a;
  const winnerWins = result === "A" ? best.aWins : best.bWins;
  const loserWins = result === "A" ? best.bWins : best.aWins;
  candidates.push({
    key: `rivalry-${normalizeName(winner)}-${normalizeName(loser)}`,
    priority: 78 + Math.min(best.matches, 10),
    line: `${winner} lands the latest blow in a rivalry with ${loser}.`,
    factLabel: "Rivalry watch",
    factValue: `${winner} over ${loser}`,
    factDetail: `${best.matches} meetings tracked`,
    people: [normalizeName(winner), normalizeName(loser)],
    headlines: [
      `${winner} takes the latest rivalry round`,
      `${winner} nudges the rivalry again`,
      `${winner} wins the latest personal battle`,
      `${winner} keeps the edge over ${loser}`
    ]
  });
}

function headlineFor({
  match,
  result,
  winnerLabel,
  margin,
  totalGoals,
  upset,
  candidates
}: {
  match: Match;
  result: "A" | "B" | "DRAW";
  winnerLabel: string;
  margin: number;
  totalGoals: number;
  upset: boolean;
  candidates: StoryCandidate[];
}) {
  const preferredHeadlineKeys =
    result === "DRAW"
      ? ["draw-balance"]
      : ["rating-upset", "archive-high-goals", "one-goal-game", "clear-margin", "high-goals", "rating-edge-held"];
  const headlineCandidate =
    preferredHeadlineKeys
      .map((key) => candidates.find((candidate) => candidate.key === key && candidate.headlines?.length))
      .find(Boolean) || candidates.find((candidate) => candidate.headlines?.length);
  if (headlineCandidate?.headlines?.length) {
    return stablePick(headlineCandidate.headlines, `${match.id}-${headlineCandidate.key}`);
  }

  if (result === "DRAW") {
    return stablePick([
      "Nothing splits the sides",
      "The argument stays open",
      "A draw with more questions than answers"
    ], match.id);
  }

  if (upset) return `${winnerLabel} win against the numbers`;
  if (totalGoals >= 20) return `${winnerLabel} come through the storm`;
  if (margin >= 5) return `${winnerLabel} put daylight on the scoreboard`;
  if (margin <= 1) return `${winnerLabel} edge a tight one`;
  return stablePick([
    `${winnerLabel} find the answer`,
    `${winnerLabel} get it done`,
    `${winnerLabel} take control of the night`,
    `${winnerLabel} bank the win`
  ], match.id);
}

function summaryFor({
  match,
  result,
  winnerLabel,
  loserLabel,
  margin,
  totalGoals,
  upset,
  favourite,
  ratingGap,
  leadFact,
  changes
}: {
  match: Match;
  result: "A" | "B" | "DRAW";
  winnerLabel: string;
  loserLabel: string;
  margin: number;
  totalGoals: number;
  upset: boolean;
  favourite: "Team A" | "Team B" | "Level";
  ratingGap: number;
  leadFact: ReportFact;
  changes: ReportChange[];
}) {
  const score = resultScoreLabel(match, result);
  const secondary = changes.find((item) => item.label !== leadFact.label || item.value !== leadFact.value);
  const secondarySentence = summaryFollowUpFor(secondary);

  if (result === "DRAW") {
    const ratingLine =
      favourite === "Level"
        ? "the ratings were already calling it tight"
        : `${favourite} had the pre-match edge`;
    return `It finished ${score}, and ${ratingLine}. No clean swing, just another result that keeps the matchup open.`;
  }

  if (upset) {
    return `${winnerLabel} took a ${score} win from the lower-rated side of the draw. The scoreline matters, but the bigger story is that ${loserLabel}'s pre-match edge did not hold.${secondarySentence}`;
  }

  if (margin === 1) {
    const ratingLine =
      favourite === "Level"
        ? `with only ${ratingGap} MMR between the teams before kick-off`
        : `with ${favourite} carrying the pre-match rating edge`;
    return `${winnerLabel} edged a ${score} win ${ratingLine}.${secondarySentence}`;
  }

  if (margin >= 5) {
    return `${winnerLabel} won ${score} and gave the result real separation. ${loserLabel} were kept at distance once the gap opened.${secondarySentence}`;
  }

  if (totalGoals >= 16) {
    return `${winnerLabel} came through a busy ${score} game with ${totalGoals} goals on the board. It reads like a night where the attack won out just enough.${secondarySentence}`;
  }

  return `${winnerLabel} banked a ${score} win and kept ${loserLabel} just far enough away. The rating movement now gives the result its aftertaste.${secondarySentence}`;
}

function summaryFollowUpFor(change: ReportChange | undefined) {
  if (!change) return "";

  if (change.label === "Streak watch") {
    const name = change.value.replace(/\s+stays hot$/i, "");
    const wins = change.detail.match(/^(\d+)\s+wins/i)?.[1];
    return wins ? ` ${name}'s winning run now stretches to ${wins}.` : ` ${change.value}.`;
  }

  if (change.label === "Momentum watch") {
    const name = change.value.replace(/\s+stays unbeaten$/i, "");
    const run = change.detail.match(/^(\d+)\s+without defeat/i)?.[1];
    return run ? ` ${name} is up to ${run} without defeat.` : ` ${change.value}.`;
  }

  if (change.label === "Rating lift") {
    return ` ${change.value} was the biggest ratings winner from the night.`;
  }

  if (change.label === "Rating hit") {
    return ` ${change.value} took the biggest ratings hit.`;
  }

  return ` ${change.value}.`;
}

function buildStorylines({
  match,
  matches,
  result,
  winnerLabel,
  loserLabel,
  margin,
  totalGoals,
  upset,
  favourite,
  ratingGap,
  candidates,
  leadFact
}: {
  match: Match;
  matches: Match[];
  result: "A" | "B" | "DRAW";
  winnerLabel: string;
  loserLabel: string;
  margin: number;
  totalGoals: number;
  upset: boolean;
  favourite: "Team A" | "Team B" | "Level";
  ratingGap: number;
  candidates: StoryCandidate[];
  leadFact: ReportFact;
}) {
  const lines: string[] = [];
  const scoreHeat = candidates.find((candidate) => candidate.key === "archive-high-goals" || candidate.key === "high-goals");
  const closeGame = candidates.find((candidate) => candidate.key === "one-goal-game");
  const clearMargin = candidates.find((candidate) => candidate.key === "clear-margin");
  const ratingUpset = candidates.find((candidate) => candidate.key === "rating-upset");
  const ratingEdge = candidates.find((candidate) => candidate.key === "rating-edge-held");
  const relationship = selectRelationshipAngle(candidates, leadFact);
  const streak = candidates.find((candidate) => candidate.factLabel === "Streak watch");
  const bounce = candidates.find((candidate) => candidate.factLabel === "Bounce watch");
  const stoppedRun = candidates.find((candidate) => candidate.factLabel === "Run stopped");
  const pressure = candidates.find((candidate) => candidate.factLabel === "Pressure point");
  const seasonAverageGoals = averageSeasonGoalsUpTo(match, matches);

  if (result === "DRAW") {
    lines.push("Neither side got the clean break, so the result keeps the matchup argument open for next time.");
  } else if (ratingUpset) {
    lines.push(`${winnerLabel} changed the tone of the matchup by making the pre-match numbers look too tidy.`);
  } else if (clearMargin) {
    lines.push(`${winnerLabel} did not just win the game; they made the last stretch feel comfortable.`);
  } else if (closeGame) {
    lines.push(`${winnerLabel} spent the night close enough to be caught, then found enough to close it out.`);
  } else if (ratingEdge) {
    lines.push(`${winnerLabel} kept control of the game state instead of letting the underdog route open up.`);
  } else {
    lines.push(`${winnerLabel} kept ${loserLabel} at arm's length without needing a dramatic swing.`);
  }

  if (scoreHeat) {
    lines.push("The scoring stayed lively enough that both teams kept leaving evidence in the result.");
  } else if (closeGame && favourite === "Level") {
    lines.push("The close finish felt earned rather than random, because the teams came in with very little between them.");
  } else if (closeGame) {
    lines.push(`${loserLabel} were close enough to keep asking the question, but never close enough to stop the result landing.`);
  } else if (totalGoals >= 16 && margin > 0) {
    lines.push(scorePaceLine(totalGoals, seasonAverageGoals));
  } else if (margin > 0 && !clearMargin && !closeGame) {
    lines.push(`${loserLabel} never quite found the spell that would turn pressure into a proper chase.`);
  } else if (result === "DRAW") {
    lines.push(scoreHeat ? "Even the busier scoreline could not separate the sides." : "The draw matters most because neither side got a clean break in the matchup.");
  }

  if (relationship) {
    lines.push(relationship.line);
  } else if (streak && pressure && !overlapsFact(streak, leadFact) && !overlapsFact(pressure, leadFact)) {
    lines.push("The form table moved in opposite directions, so the result will feel very different player to player.");
  } else if (streak && bounce && !overlapsFact(streak, leadFact) && !overlapsFact(bounce, leadFact)) {
    lines.push("The form angle is split between a run continuing and a reset finally landing.");
  } else if (streak && stoppedRun && !overlapsFact(streak, leadFact) && !overlapsFact(stoppedRun, leadFact)) {
    lines.push("One run survives the night while another gets clipped, which gives the result a bit more bite.");
  } else if (bounce && pressure && !overlapsFact(bounce, leadFact) && !overlapsFact(pressure, leadFact)) {
    lines.push("For one player this was a reset; for another it leaves the next match carrying extra weight.");
  }

  return uniqueStrings(lines).slice(0, 3);
}

function buildChangeItems(candidates: StoryCandidate[], leadFact: ReportFact, awardChanges: ReportChange[] = []): ReportChange[] {
  const consequenceLabels = new Set([
    "Streak watch",
    "Momentum watch",
    "Bounce watch",
    "Run stopped",
    "Pressure point",
    "Rating lift",
    "Rating hit",
    "Rating swing"
  ]);
  const eligible = candidates
    .filter((candidate) => candidate.factLabel && candidate.factValue && candidate.factDetail)
    .filter((candidate) => consequenceLabels.has(candidate.factLabel || ""))
    .filter((candidate) => candidate.factLabel !== leadFact.label || candidate.factValue !== leadFact.value);

  const changes = selectDiverseCandidates(eligible, 4, peopleFromFact(leadFact)).map((candidate) => ({
    label: candidate.factLabel || "Change",
    value: candidate.factValue || "Updated",
    detail: candidate.factDetail || candidate.line,
    people: candidate.people || []
  }));

  return dedupeChanges([...awardChanges, ...changes]).slice(0, 4);
}

function seasonAwardChanges({
  match,
  matches,
  players,
  mmrHistory,
  nameMap
}: {
  match: Match;
  matches: Match[];
  players: Player[];
  mmrHistory: MmrHistory[];
  nameMap: Map<string, string>;
}): ReportChange[] {
  const season = String(match.date || "").slice(0, 4);
  if (!/^\d{4}$/.test(season)) return [];

  const before = seasonAwardSnapshot({ match, matches: matchesBefore(match, matches), players, mmrHistory, nameMap, season });
  const after = seasonAwardSnapshot({ match, matches: matchesUpTo(match, matches), players, mmrHistory, nameMap, season });
  if (!after) return [];

  const changes: ReportChange[] = [];
  const afterMvp = after.mvp[0];
  const beforeMvp = before?.mvp[0] || null;
  const nextMvp = after.mvp[1] || null;

  if (afterMvp) {
    const changed = beforeMvp && beforeMvp.key !== afterMvp.key;
    changes.push({
      label: "MVP race",
      value: changed ? `${afterMvp.name} takes #1` : `${afterMvp.name} leads`,
      detail: awardRaceDetail(season, afterMvp.score, nextMvp ? `${nextMvp.name} ${nextMvp.score} pts` : null),
      people: [afterMvp.key]
    });
  }

  const afterImproved = after.improved[0];
  const beforeImproved = before?.improved[0] || null;
  const nextImproved = after.improved[1] || null;

  if (afterImproved) {
    changes.push(improvedRaceChange(match, afterImproved, beforeImproved, nextImproved));
  }

  return changes;
}

function improvedRaceChange(
  match: Match,
  leader: { key: string; name: string; change: number },
  previousLeader: { key: string; name: string; change: number } | null,
  next: { key: string; name: string; change: number } | null
): ReportChange {
  const participantKeys = new Set([...splitTeam(match.team_a), ...splitTeam(match.team_b)].map(normalizeName));
  const leaderPlayed = participantKeys.has(leader.key);
  const nextPlayed = next ? participantKeys.has(next.key) : false;
  const changed = previousLeader && previousLeader.key !== leader.key;

  if (!leaderPlayed && next && nextPlayed && next.change === leader.change) {
    return {
      label: "Most improved",
      value: `${next.name} moves level`,
      detail: `${signed(next.change)} MMR; level with ${leader.name}`,
      people: [next.key, leader.key]
    };
  }

  if (!leaderPlayed && next && nextPlayed) {
    const gap = leader.change - next.change;
    return {
      label: "Most improved",
      value: `${next.name} closes in`,
      detail: `${signed(next.change)} MMR; ${gap} behind ${leader.name}`,
      people: [next.key, leader.key]
    };
  }

  return {
    label: "Most improved",
    value: changed ? `${leader.name} takes #1` : `${leader.name} holds on`,
    detail: improvementRaceDetail(leader.change, next),
    people: [leader.key]
  };
}

function seasonAwardSnapshot({
  match,
  matches,
  players,
  mmrHistory,
  nameMap,
  season
}: {
  match: Match;
  matches: Match[];
  players: Player[];
  mmrHistory: MmrHistory[];
  nameMap: Map<string, string>;
  season: string;
}) {
  const seasonMatches = matches
    .filter((row) => String(row.date || "").startsWith(`${season}-`) && scoreParts(row.score))
    .sort(compareMatchesAsc);
  if (!seasonMatches.length) return null;

  const seasonHistory = mmrHistory.filter((row) => String(row.date || "").startsWith(`${season}-`));
  const minimumMatches = minimumAwardMatches(seasonMatches.length);
  const allPlayerRows = players.map((player) => seasonAwardPlayer(player, seasonMatches, seasonHistory, nameMap));
  const playerRows = allPlayerRows.filter((row) => row.played >= minimumMatches);
  const seasonRatings = seasonOnlyAwardRatings(allPlayerRows, seasonMatches);

  return {
    mvp: playerRows
      .map((row) => ({
        ...row,
        seasonRating: seasonRatings.get(row.id) ?? STARTING_MMR,
        score: ratingPoints(seasonRatings.get(row.id) ?? STARTING_MMR) + Math.round((row.played / Math.max(seasonMatches.length, 1)) * 20)
      }))
      .sort((a, b) => b.score - a.score || b.seasonRating - a.seasonRating || b.played - a.played || b.recordPoints - a.recordPoints),
    improved: playerRows
      .filter((row) => Number.isFinite(row.change))
      .sort((a, b) => b.change - a.change),
    match
  };
}

function seasonAwardPlayer(player: Player, matches: Match[], history: MmrHistory[], nameMap: Map<string, string>) {
  const name = displayName(player);
  const keys = [player.name, player.display_name, name].map(normalizeName).filter(Boolean);
  let wins = 0;
  let draws = 0;
  const playedMatches = matches.filter((match) => {
    const teamA = splitTeam(match.team_a).map(normalizeName);
    const teamB = splitTeam(match.team_b).map(normalizeName);
    const side = keys.some((key) => teamA.includes(key)) ? "A" : keys.some((key) => teamB.includes(key)) ? "B" : null;
    if (side) {
      const playerResult = resultFor(match, side);
      wins += playerResult === "W" ? 1 : 0;
      draws += playerResult === "D" ? 1 : 0;
    }
    return Boolean(side);
  });
  const rows = history
    .filter((row) => row.player_id === player.id && matches.some((match) => match.id === row.match_id))
    .sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")) || Number(a.id || 0) - Number(b.id || 0));
  const first = rows[0];
  const last = rows.at(-1);
  const startMmr = numberOr(first?.mmr_before, STARTING_MMR);
  const currentMmr = numberOr(last?.mmr_after, numberOr(player.mmr, STARTING_MMR));

  return {
    ...player,
    matchNames: [player.name, player.display_name, name].filter(Boolean),
    key: normalizeName(name),
    name: nameMap.get(normalizeName(name)) || name,
    played: playedMatches.length,
    startMmr,
    currentMmr,
    change: Math.round(currentMmr - startMmr),
    recordPoints: wins * 3 + draws
  };
}

function seasonOnlyAwardRatings(players: ReturnType<typeof seasonAwardPlayer>[], matches: Match[]) {
  const seedRatings = seededAwardStarts(players);
  const state = new Map<number, Player>(
    players.map((player) => [
      player.id,
      {
        ...player,
        mmr: seedRatings.get(player.id) ?? STARTING_MMR
      }
    ])
  );
  const nameLookup = new Map<string, Player>();

  for (const player of state.values()) {
    const aliases = "matchNames" in player && Array.isArray(player.matchNames) ? player.matchNames : [player.name, player.display_name];
    for (const alias of aliases) {
      addPlayerLookup(nameLookup, alias, player);
    }
  }

  const orderedMatches = [...matches]
    .filter((match) => scoreParts(match.score))
    .sort(compareMatchesAsc);

  for (const match of orderedMatches) {
    const teamA = resolveSeasonTeam(match.team_a, nameLookup);
    const teamB = resolveSeasonTeam(match.team_b, nameLookup);
    if (!teamA.length || !teamB.length) continue;

    const updates = calculateMatchMmrUpdates({
      teamA,
      teamB,
      match,
      players: Array.from(state.values())
    });

    for (const update of updates) {
      const player = state.get(update.player.id);
      if (player) player.mmr = Math.round(update.after);
    }
  }

  return new Map(Array.from(state.values()).map((player) => [player.id, Number(player.mmr || STARTING_MMR)]));
}

function seededAwardStarts(players: ReturnType<typeof seasonAwardPlayer>[]) {
  const involved = players.filter((player) => Number(player.played || 0) > 0);
  const startValues = involved.map((player) => numberOr(player.startMmr, STARTING_MMR));
  const leagueAverage = average(startValues);

  return new Map(
    players.map((player) => {
      const start = numberOr(player.startMmr, numberOr(player.mmr, STARTING_MMR));
      const seeded = STARTING_MMR + (start - leagueAverage) * 0.35;
      return [player.id, Math.round(seeded)] as const;
    })
  );
}

function addPlayerLookup(lookup: Map<string, Player>, value: string | null | undefined, player: Player) {
  const key = normalizeName(value);
  if (key && !lookup.has(key)) lookup.set(key, player);
}

function resolveSeasonTeam(value: string | null | undefined, lookup: Map<string, Player>) {
  const seen = new Set<number>();
  const team: Player[] = [];

  for (const name of splitTeam(value)) {
    const player = lookup.get(normalizeName(name));
    if (!player || seen.has(player.id)) continue;
    team.push(player);
    seen.add(player.id);
  }

  return team;
}

function awardRaceDetail(season: string, score: number, next: string | null) {
  return next ? `${season} leader, ${score} pts; next ${next}` : `${season} leader, ${score} MVP pts`;
}

function improvementRaceDetail(change: number, next: { name: string; change: number } | null) {
  if (!next) return `${signed(change)} MMR this season`;
  const gap = change - next.change;
  return gap <= 0
    ? `${signed(change)} MMR; level with ${next.name}`
    : gap <= 6
      ? `${signed(change)} MMR; ${next.name} only ${gap} back`
    : `${signed(change)} MMR; ${next.name} next at ${signed(next.change)}`;
}

function selectStorylineAngle(candidates: StoryCandidate[], leadFact: ReportFact) {
  const storylineLabels = new Set([
    "Streak watch",
    "Momentum watch",
    "Bounce watch",
    "Run stopped",
    "Pressure point",
    "Rating lift",
    "Rating hit"
  ]);
  const eligible = candidates
    .filter((candidate) => candidate.factLabel && storylineLabels.has(candidate.factLabel))
    .filter((candidate) => candidate.factLabel !== leadFact.label || candidate.factValue !== leadFact.value);

  return selectDiverseCandidates(eligible, 1, peopleFromFact(leadFact))[0] || null;
}

function selectRelationshipAngle(candidates: StoryCandidate[], leadFact: ReportFact) {
  const eligible = candidates
    .filter((candidate) => candidate.factLabel === "Partnership watch" || candidate.factLabel === "Rivalry watch")
    .filter((candidate) => candidate.factLabel !== leadFact.label || candidate.factValue !== leadFact.value);

  return selectDiverseCandidates(eligible, 1, peopleFromFact(leadFact))[0] || null;
}

function selectDiverseCandidates(candidates: StoryCandidate[], limit: number, initialPeople: string[] = []) {
  const selected: StoryCandidate[] = [];
  const remaining = [...candidates];
  const usedPeople = new Set(initialPeople);
  const usedLabels = new Set<string>();

  while (selected.length < limit && remaining.length) {
    let bestIndex = 0;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < remaining.length; index += 1) {
      const score = diversityScore(remaining[index], usedPeople, usedLabels);
      if (score > bestScore) {
        bestIndex = index;
        bestScore = score;
      }
    }

    const [best] = remaining.splice(bestIndex, 1);
    selected.push(best);
    for (const person of best.people || []) {
      usedPeople.add(person);
    }
    if (best.factLabel) usedLabels.add(best.factLabel);
  }

  return selected;
}

function diversityScore(candidate: StoryCandidate, usedPeople: Set<string>, usedLabels: Set<string>) {
  const people = candidate.people || [];
  const overlaps = people.filter((person) => usedPeople.has(person)).length;
  const freshPeople = people.filter((person) => !usedPeople.has(person)).length;
  const labelPenalty = candidate.factLabel && usedLabels.has(candidate.factLabel) ? 10 : 0;
  return candidate.priority + freshPeople * 6 - overlaps * 36 - labelPenalty;
}

function peopleFromFact(fact: Pick<ReportFact, "people">) {
  return fact.people || [];
}

function overlapsFact(candidate: StoryCandidate, fact: ReportFact) {
  const factPeople = new Set(peopleFromFact(fact));
  return Boolean(candidate.people?.some((person) => factPeople.has(person)));
}

function previousMeetingFor(match: Match, matches: Match[], nameMap: Map<string, string>): PreviousMeeting | null {
  const currentTeamA = splitTeam(match.team_a);
  const currentTeamB = splitTeam(match.team_b);
  const currentA = new Set(currentTeamA.map(normalizeName));
  const currentB = new Set(currentTeamB.map(normalizeName));
  const currentPlayers = new Set([...currentA, ...currentB]);
  const closest = matchesBefore(match, matches)
    .map((row) => {
      const previousA = splitTeam(row.team_a);
      const previousB = splitTeam(row.team_b);
      const previousASet = new Set(previousA.map(normalizeName));
      const previousBSet = new Set(previousB.map(normalizeName));
      const directSameSide = countOverlap(currentA, previousASet) + countOverlap(currentB, previousBSet);
      const swappedSameSide = countOverlap(currentA, previousBSet) + countOverlap(currentB, previousASet);
      const swapped = swappedSameSide > directSameSide;
      const sameSidePlayers = Math.max(directSameSide, swappedSameSide);
      const previousPlayers = [...previousA, ...previousB].map(normalizeName);
      const sharedPlayers = previousPlayers.filter((name) => currentPlayers.has(name)).length;
      return {
        match: row,
        sharedPlayers,
        sameSidePlayers,
        swapped,
        orientedA: swapped ? previousB : previousA,
        orientedB: swapped ? previousA : previousB
      };
    })
    .filter((row) => row.sharedPlayers >= 4 && scoreParts(row.match.score))
    .sort((a, b) => b.sameSidePlayers - a.sameSidePlayers || b.sharedPlayers - a.sharedPlayers || compareMatchesAsc(b.match, a.match))[0];

  if (!closest) return null;

  const currentResult = scoreToResult(match);
  const previousResult = scoreToResult(closest.match);
  const orientedPreviousResult = closest.swapped ? swapResult(previousResult) : previousResult;
  const orientedScore = orientedScoreLabel(closest.match.score, closest.swapped);
  const previousWinner =
    orientedPreviousResult === "DRAW" ? "a draw" : orientedPreviousResult === "A" ? "a Team A win" : "a Team B win";
  const continuity = `${closest.sharedPlayers} ${closest.sharedPlayers === 1 ? "player was" : "players were"} back, ${closest.sameSidePlayers} on the same side`;
  const direction =
    currentResult !== "DRAW" && orientedPreviousResult !== "DRAW" && currentResult !== orientedPreviousResult
      ? "this was a reversal of the last similar meeting"
      : currentResult === orientedPreviousResult && currentResult !== "DRAW"
        ? "the same side came out on top again"
        : "the comparison stays live";
  return {
    dateLabel: formatUkDate(closest.match.date),
    scoreLabel: orientedScore,
    resultLabel: orientedPreviousResult === "DRAW" ? "Draw" : orientedPreviousResult === "A" ? "Team A won" : "Team B won",
    sharedPlayers: closest.sharedPlayers,
    sameSidePlayers: closest.sameSidePlayers,
    swapped: closest.swapped,
    detail: `Closest recent comparator: ${orientedScore} on ${formatUkDate(closest.match.date)}, ${previousWinner}. With ${continuity}, ${direction}.`,
    sides: [
      previousMeetingSide({
        historicTeam: closest.orientedA,
        currentTeam: currentTeamA,
        label: closest.swapped ? "Old Team B -> Team A" : "Old Team A -> Team A",
        tone: "a",
        nameMap
      }),
      previousMeetingSide({
        historicTeam: closest.orientedB,
        currentTeam: currentTeamB,
        label: closest.swapped ? "Old Team A -> Team B" : "Old Team B -> Team B",
        tone: "b",
        nameMap
      })
    ]
  };
}

function previousMeetingSide({
  historicTeam,
  currentTeam,
  label,
  tone,
  nameMap
}: {
  historicTeam: string[];
  currentTeam: string[];
  label: string;
  tone: "a" | "b";
  nameMap: Map<string, string>;
}): PreviousMeetingSide {
  const currentSet = new Set(currentTeam.map(normalizeName));
  const historicSet = new Set(historicTeam.map(normalizeName));
  const missing = historicTeam.filter((name) => !currentSet.has(normalizeName(name)));
  const subs = currentTeam.filter((name) => !historicSet.has(normalizeName(name)));

  return {
    label,
    tone,
    historicTeam: historicTeam.map((name) => displayFromMap(name, nameMap)),
    subs: subs.map((name) => displayFromMap(name, nameMap)),
    missing: missing.map((name) => displayFromMap(name, nameMap)),
    keptCount: historicTeam.length - missing.length,
    totalCount: historicTeam.length
  };
}

function leadFactFor(
  candidates: StoryCandidate[],
  fallback: {
    result: "A" | "B" | "DRAW";
    winnerLabel: string;
    margin: number;
    totalGoals: number;
    upset: boolean;
  }
): ReportFact {
  const candidate = candidates.find((item) => item.factLabel && item.factValue && item.factDetail);
  if (candidate?.factLabel && candidate.factValue && candidate.factDetail) {
    return {
      label: candidate.factLabel,
      value: candidate.factValue,
      detail: candidate.factDetail,
      people: candidate.people || []
    };
  }

  if (fallback.result === "DRAW") {
    return {
      label: "Main angle",
      value: "Still unresolved",
      detail: fallback.totalGoals ? `${fallback.totalGoals} goals, no winner` : "No winner"
    };
  }

  return {
    label: "Main angle",
    value: fallback.upset ? `${fallback.winnerLabel} upset it` : fallback.margin >= 5 ? "Clear separation" : "Win banked",
    detail: fallback.margin ? `${fallback.margin}-goal gap` : "Score recorded"
  };
}

function ratingFactFor({
  favourite,
  ratingGap,
  winnerLabel,
  upset,
  avgA,
  avgB
}: {
  favourite: "Team A" | "Team B" | "Level";
  ratingGap: number;
  winnerLabel: string;
  upset: boolean;
  avgA: number;
  avgB: number;
}): ReportFact {
  if (favourite === "Level") {
    return {
      label: "Pre-match edge",
      value: "Almost level",
      detail: ratingGap ? `${ratingGap} MMR between teams` : `${Math.round(avgA)} vs ${Math.round(avgB)} avg MMR`
    };
  }

  if (upset) {
    return {
      label: "Pre-match edge",
      value: `${favourite} were favoured`,
      detail: `${winnerLabel} won from the lower average`
    };
  }

  return {
    label: "Pre-match edge",
    value: `${favourite} +${ratingGap}`,
    detail: `${Math.round(avgA)} vs ${Math.round(avgB)} avg MMR`
  };
}

function scoreProfileFact(match: Match, matches: Match[], margin: number, totalGoals: number, leadLabel: string): ReportFact {
  const currentSeasonMatches = seasonMatchesUpTo(match, matches);
  const currentSeasonScores = currentSeasonMatches
    .map((row) => scoreParts(row.score))
    .filter((score): score is [number, number] => Boolean(score));
  const currentSeasonTotals = currentSeasonScores.map((score) => score[0] + score[1]);
  const currentSeasonAverage = currentSeasonTotals.length ? average(currentSeasonTotals) : 0;
  const maxSeasonTotal = currentSeasonTotals.length ? Math.max(...currentSeasonTotals) : 0;
  const seasonAverageLabel = currentSeasonAverage ? `${totalGoals} goals vs ${currentSeasonAverage.toFixed(1)} season avg` : `${totalGoals} total goals`;

  if (leadLabel === "Score heat") {
    return {
      label: "Margin check",
      value: margin ? `${margin}-goal gap` : "Level",
      detail: totalGoals ? seasonAverageLabel : "Score recorded"
    };
  }

  if (leadLabel === "Fine margins" || leadLabel === "Separation") {
    if (totalGoals && currentSeasonAverage) {
      if (totalGoals <= currentSeasonAverage - 2) {
        return {
          label: "Score profile",
          value: "Tighter than usual",
          detail: seasonAverageLabel
        };
      }
      if (totalGoals >= currentSeasonAverage + 2) {
        return {
          label: "Score profile",
          value: "Busier than usual",
          detail: seasonAverageLabel
        };
      }
    }

    return {
      label: "Score profile",
      value: totalGoals ? `${totalGoals} goals` : "Recorded",
      detail: currentSeasonAverage ? seasonAverageLabel : "No season average yet"
    };
  }

  if (totalGoals && currentSeasonTotals.length >= 5 && totalGoals >= maxSeasonTotal) {
    return {
      label: "Score profile",
      value: "Season high",
      detail: seasonAverageLabel
    };
  }

  if (totalGoals && currentSeasonAverage && totalGoals >= currentSeasonAverage + 5) {
    return {
      label: "Score profile",
      value: "High tempo",
      detail: seasonAverageLabel
    };
  }

  if (margin === 1) {
    return {
      label: "Score profile",
      value: "Fine margins",
      detail: "One goal decided it"
    };
  }

  if (margin >= 5) {
    return {
      label: "Score profile",
      value: "Pulled away",
      detail: `${margin}-goal separation`
    };
  }

  return {
    label: "Score profile",
    value: margin ? `${margin}-goal gap` : "Level",
    detail: totalGoals ? seasonAverageLabel : "Score recorded"
  };
}

function averageSeasonGoalsUpTo(match: Match, matches: Match[]) {
  const totals = seasonMatchesUpTo(match, matches)
    .map((row) => scoreParts(row.score))
    .filter((score): score is [number, number] => Boolean(score))
    .map((score) => score[0] + score[1]);
  return totals.length ? average(totals) : 0;
}

function scorePaceLine(totalGoals: number, seasonAverage: number) {
  if (!seasonAverage) return `The ${totalGoals}-goal finish gave the game plenty of movement without changing the basic read.`;
  if (totalGoals >= seasonAverage + 2) {
    return "The scoring pace ran above the season norm, so the scoreboard had more heat than usual.";
  }
  if (totalGoals <= seasonAverage - 2) {
    return "The scoring pace sat below the season norm, so this was tighter than the raw score first looks.";
  }
  return "The scoring pace sat right around the season norm: busy, but not unusual for 2026.";
}

function buildPlayerNotes({
  match,
  matches,
  players,
  mmrHistory,
  nameMap,
  result,
  margin,
  candidates
}: {
  match: Match;
  matches: Match[];
  players: Player[];
  mmrHistory: MmrHistory[];
  nameMap: Map<string, string>;
  result: "A" | "B" | "DRAW";
  margin: number;
  candidates: StoryCandidate[];
}) {
  const changes = new Map(matchMmrChanges(match, players, mmrHistory).map((row) => [row.key, row.delta]));
  const angles = playerDetailAngles({ match, matches, players, mmrHistory, nameMap, candidates });
  const usedTags = new Map<string, number>();
  const teamA = splitTeam(match.team_a).map((name) => playerNoteFor({ name, side: "A", match, result, margin, matches, nameMap, changes, angles, usedTags }));
  const teamB = splitTeam(match.team_b).map((name) => playerNoteFor({ name, side: "B", match, result, margin, matches, nameMap, changes, angles, usedTags }));
  return [...teamA, ...teamB];
}

function playerDetailAngles({
  match,
  matches,
  players,
  mmrHistory,
  nameMap,
  candidates
}: {
  match: Match;
  matches: Match[];
  players: Player[];
  mmrHistory: MmrHistory[];
  nameMap: Map<string, string>;
  candidates: StoryCandidate[];
}) {
  const angles = new Map<string, PlayerDetailAngle[]>();
  addSeasonAwardPlayerAngles(angles, { match, matches, players, mmrHistory, nameMap });

  for (const candidate of candidates) {
    if (!candidate.people?.length) continue;
    const angle = candidateAngle(candidate);
    if (!angle) continue;
    for (const person of candidate.people) {
      addPlayerAngle(angles, person, angle);
    }
  }

  return angles;
}

function addSeasonAwardPlayerAngles(
  angles: Map<string, PlayerDetailAngle[]>,
  {
    match,
    matches,
    players,
    mmrHistory,
    nameMap
  }: {
    match: Match;
    matches: Match[];
    players: Player[];
    mmrHistory: MmrHistory[];
    nameMap: Map<string, string>;
  }
) {
  const season = String(match.date || "").slice(0, 4);
  if (!/^\d{4}$/.test(season)) return;
  const snapshot = seasonAwardSnapshot({ match, matches: matchesUpTo(match, matches), players, mmrHistory, nameMap, season });
  if (!snapshot) return;

  const mvp = snapshot.mvp[0];
  const mvpChaser = snapshot.mvp[1];
  if (mvp) {
    addPlayerAngle(angles, mvp.key, {
      tag: "MVP leader",
      detail: mvpChaser
        ? `Leads the ${season} MVP race on ${mvp.score} pts; ${mvpChaser.name} next on ${mvpChaser.score}.`
        : `Leads the ${season} MVP race on ${mvp.score} pts.`,
      priority: 130
    });
  }
  if (mvpChaser && mvp && mvp.score - mvpChaser.score <= 12) {
    addPlayerAngle(angles, mvpChaser.key, {
      tag: "MVP chase",
      detail: `${mvpChaser.score} MVP pts, ${mvp.score - mvpChaser.score} behind ${mvp.name}.`,
      priority: 104
    });
  }

  const improved = snapshot.improved[0];
  const improvedChaser = snapshot.improved[1];
  const participantKeys = new Set([...splitTeam(match.team_a), ...splitTeam(match.team_b)].map(normalizeName));
  if (improved) {
    const gap = improvedChaser ? improved.change - improvedChaser.change : null;
    addPlayerAngle(angles, improved.key, {
      tag: "Most improved",
      detail:
        improvedChaser && gap !== null && gap <= 0
          ? `Joint-top for ${season} improvement at ${signed(improved.change)} MMR.`
          : improvedChaser && gap !== null && gap <= 8
            ? `Top of Most Improved at ${signed(improved.change)} MMR; ${improvedChaser.name} is ${gap} back.`
            : `Top of Most Improved at ${signed(improved.change)} MMR this season.`,
      priority: 128
    });
  }
  if (improvedChaser && improved && improved.change - improvedChaser.change <= 8) {
    const chaserPlayed = participantKeys.has(improvedChaser.key);
    addPlayerAngle(angles, improvedChaser.key, {
      tag: "Improver chase",
      detail:
        chaserPlayed && improvedChaser.change === improved.change
          ? `Moved level with ${improved.name} at ${signed(improvedChaser.change)} MMR this season.`
          : `${signed(improvedChaser.change)} MMR this season, still close to ${improved.name}.`,
      priority: 102
    });
  }
}

function candidateAngle(candidate: StoryCandidate): PlayerDetailAngle | null {
  if (!candidate.factLabel || !candidate.factValue || !candidate.factDetail) return null;
  if (candidate.factLabel === "Bounce watch") {
    return { tag: "Bounce", detail: `${candidate.factValue}. ${candidate.factDetail}.`, priority: 112 };
  }
  if (candidate.factLabel === "Streak watch") {
    return { tag: "Streak", detail: `${candidate.factValue}. ${candidate.factDetail}.`, priority: 106 };
  }
  if (candidate.factLabel === "Run stopped") {
    return { tag: "Run stopped", detail: `${candidate.factValue}. ${candidate.factDetail}.`, priority: 104 };
  }
  if (candidate.factLabel === "Pressure point") {
    return { tag: "Pressure", detail: `${candidate.factValue}. ${candidate.factDetail}.`, priority: 100 };
  }
  if (candidate.factLabel === "Rivalry watch") {
    return { tag: "Rivalry", detail: candidate.line, priority: 92 };
  }
  if (candidate.factLabel === "Partnership watch") {
    return { tag: "Partnership", detail: candidate.line, priority: 88 };
  }
  return null;
}

function addPlayerAngle(angles: Map<string, PlayerDetailAngle[]>, person: string, angle: PlayerDetailAngle) {
  const key = normalizeName(person);
  const rows = angles.get(key) || [];
  rows.push(angle);
  angles.set(key, rows);
}

function playerNoteFor({
  name,
  side,
  match,
  result,
  margin,
  matches,
  nameMap,
  changes,
  angles,
  usedTags
}: {
  name: string;
  side: "A" | "B";
  match: Match;
  result: "A" | "B" | "DRAW";
  margin: number;
  matches: Match[];
  nameMap: Map<string, string>;
  changes: Map<string, number>;
  angles: Map<string, PlayerDetailAngle[]>;
  usedTags: Map<string, number>;
}): ReportPlayerNote {
  const display = displayFromMap(name, nameMap);
  const delta = changes.get(normalizeName(display)) ?? changes.get(normalizeName(name));
  const team = side === "A" ? "Team A" : "Team B";
  const won = result === side;
  const lost = result !== "DRAW" && !won;
  const run = result === "DRAW" ? currentRunWhere(name, match, matches, (item) => item !== "L") : currentRun(name, won ? "W" : "L", match, matches);
  const ratingMove = typeof delta === "number" && delta !== 0 ? `${signed(delta)} MMR` : "No MMR move";
  const availableAngles = [...(angles.get(normalizeName(display)) || []), ...(angles.get(normalizeName(name)) || [])];

  let fallback: PlayerDetailAngle;
  if (typeof delta === "number" && delta >= 14) {
    fallback = { tag: "Big lift", detail: `One of the strongest rating moves from the game. ${ratingMove}.`, priority: 62 };
  } else if (typeof delta === "number" && delta >= 8) {
    fallback = { tag: "Lift", detail: `Rating moved up with the result. ${ratingMove}.`, priority: 50 };
  } else if (typeof delta === "number" && delta <= -16) {
    fallback = { tag: "Heavy hit", detail: `One of the sharper rating drops from the game. ${ratingMove}.`, priority: 62 };
  } else if (typeof delta === "number" && delta <= -8) {
    fallback = { tag: "Hit", detail: `Rating moved against ${team}. ${ratingMove}.`, priority: 50 };
  } else if (won && run >= 3) {
    fallback = { tag: "Streak", detail: `${run} wins in a row after this result. ${ratingMove}.`, priority: 46 };
  } else if (won && margin === 1) {
    fallback = { tag: "Edged it", detail: `Finished on the right side of the tight one. ${ratingMove}.`, priority: 38 };
  } else if (won) {
    fallback = { tag: "Winner", detail: `Banked the win with ${team}. ${ratingMove}.`, priority: 20 };
  } else if (lost && margin === 1) {
    fallback = { tag: "Close hit", detail: `Lost by one, so the damage stays narrow. ${ratingMove}.`, priority: 38 };
  } else if (lost) {
    fallback = { tag: "Hit", detail: `Result moved against ${team}. ${ratingMove}.`, priority: 20 };
  } else if (run >= 4) {
    fallback = { tag: "Unbeaten", detail: `${run} without defeat after the draw. ${ratingMove}.`, priority: 42 };
  } else {
    fallback = { tag: "Draw", detail: `Shared the result for ${team}. ${ratingMove}.`, priority: 20 };
  }
  const selected = selectPlayerDetailAngle([...availableAngles, fallback], usedTags);
  usedTags.set(selected.tag, (usedTags.get(selected.tag) || 0) + 1);

  return {
    name: display,
    team,
    tone: side === "A" ? "a" : "b",
    tag: selected.tag,
    detail: selected.detail.trim(),
    delta: typeof delta === "number" ? delta : null
  };
}

function selectPlayerDetailAngle(angles: PlayerDetailAngle[], usedTags: Map<string, number>) {
  return [...angles].sort((a, b) => playerAngleScore(b, usedTags) - playerAngleScore(a, usedTags))[0];
}

function playerAngleScore(angle: PlayerDetailAngle, usedTags: Map<string, number>) {
  const used = usedTags.get(angle.tag) || 0;
  return angle.priority - used * 24;
}

function pairNote(
  match: Match,
  matches: Match[],
  nameMap: Map<string, string>,
  result: "A" | "B" | "DRAW",
  avoidPeople: Set<string>
) {
  if (result === "DRAW") return null;
  const side = result === "A" ? "A" : "B";
  const team = splitTeam(side === "A" ? match.team_a : match.team_b);
  const rows = pairRecords(team, matchesUpTo(match, matches), nameMap);
  const best = rows
    .filter((row) => row.matches >= 3)
    .sort((a, b) => relationshipVarietySort(a, b, avoidPeople, "pair"))[0];
  if (!best) return null;
  return {
    title: "Partnership note",
    body: `${best.a} + ${best.b} are now ${best.wins}-${best.draws}-${best.losses} together across ${best.matches} games.`,
    people: [normalizeName(best.a), normalizeName(best.b)]
  };
}

function rivalryNote(
  match: Match,
  matches: Match[],
  nameMap: Map<string, string>,
  result: "A" | "B" | "DRAW",
  avoidPeople: Set<string>
) {
  const teamA = splitTeam(match.team_a);
  const teamB = splitTeam(match.team_b);
  const rows = rivalryRecords(teamA, teamB, matchesUpTo(match, matches), nameMap);
  const best = rows
    .filter((row) => row.matches >= 3)
    .sort((a, b) => relationshipVarietySort(a, b, avoidPeople, "rivalry"))[0];
  if (!best) return null;

  if (result === "DRAW") {
    return {
      title: "Series status",
      body: `${best.a} ${best.aWins}, ${best.b} ${best.bWins}, with ${best.draws} draws across ${best.matches} meetings.`,
      people: [normalizeName(best.a), normalizeName(best.b)]
    };
  }

  const winner = result === "A" ? best.a : best.b;
  const loser = result === "A" ? best.b : best.a;
  const winnerWins = result === "A" ? best.aWins : best.bWins;
  const loserWins = result === "A" ? best.bWins : best.aWins;
  const record = `${winnerWins}-${best.draws}-${loserWins}`;
  const status =
    winnerWins > loserWins
      ? `${winner} now leads ${loser} ${record}`
      : winnerWins === loserWins
        ? `${winner} has pulled level with ${loser} at ${record}`
        : `${winner} still trails ${loser} ${record}, but this chips away at the gap`;

  return {
    title: "Series status",
    body: `${status} after ${best.matches} meetings.`,
    people: [normalizeName(winner), normalizeName(loser)]
  };
}

function relationshipVarietySort<
  T extends { a: string; b: string; matches: number; pointsRate?: number; balance?: number }
>(a: T, b: T, avoidPeople: Set<string>, mode: "pair" | "rivalry") {
  const aOverlap = relationshipOverlap(a, avoidPeople);
  const bOverlap = relationshipOverlap(b, avoidPeople);
  if (aOverlap !== bOverlap) return aOverlap - bOverlap;

  if (mode === "pair") {
    return numberOr(b.pointsRate, 0) - numberOr(a.pointsRate, 0) || b.matches - a.matches;
  }

  return b.matches - a.matches || numberOr(b.balance, 0) - numberOr(a.balance, 0);
}

function relationshipOverlap(row: { a: string; b: string }, avoidPeople: Set<string>) {
  return [normalizeName(row.a), normalizeName(row.b)].filter((person) => avoidPeople.has(person)).length;
}

function pairRecords(team: string[], matches: Match[], nameMap: Map<string, string>) {
  const keys = team.map(normalizeName);
  const rows = new Map<string, { a: string; b: string; matches: number; wins: number; draws: number; losses: number; points: number }>();

  for (const match of matches) {
    const teams = [
      { side: "A" as const, names: splitTeam(match.team_a) },
      { side: "B" as const, names: splitTeam(match.team_b) }
    ];
    for (const entry of teams) {
      const normalized = entry.names.map(normalizeName);
      const result = resultFor(match, entry.side);
      for (let i = 0; i < normalized.length; i += 1) {
        for (let j = i + 1; j < normalized.length; j += 1) {
          if (!keys.includes(normalized[i]) || !keys.includes(normalized[j])) continue;
          const key = [normalized[i], normalized[j]].sort().join("|");
          const [a, b] = key.split("|");
          const row = rows.get(key) || { a: nameMap.get(a) || a, b: nameMap.get(b) || b, matches: 0, wins: 0, draws: 0, losses: 0, points: 0 };
          row.matches += 1;
          row.wins += result === "W" ? 1 : 0;
          row.draws += result === "D" ? 1 : 0;
          row.losses += result === "L" ? 1 : 0;
          row.points += result === "W" ? 1 : result === "D" ? 0.5 : 0;
          rows.set(key, row);
        }
      }
    }
  }

  return [...rows.values()].map((row) => ({ ...row, pointsRate: row.points / Math.max(row.matches, 1) }));
}

function rivalryRecords(teamA: string[], teamB: string[], matches: Match[], nameMap: Map<string, string>) {
  const aKeys = new Set(teamA.map(normalizeName));
  const bKeys = new Set(teamB.map(normalizeName));
  const rows = new Map<string, { a: string; b: string; matches: number; aWins: number; bWins: number; draws: number }>();

  for (const match of matches) {
    const score = scoreParts(match.score);
    if (!score) continue;
    const histA = splitTeam(match.team_a).map(normalizeName);
    const histB = splitTeam(match.team_b).map(normalizeName);
    for (const a of aKeys) {
      for (const b of bKeys) {
        const aSide = histA.includes(a) ? "A" : histB.includes(a) ? "B" : null;
        const bSide = histA.includes(b) ? "A" : histB.includes(b) ? "B" : null;
        if (!aSide || !bSide || aSide === bSide) continue;
        const key = `${a}|${b}`;
        const row = rows.get(key) || { a: nameMap.get(a) || a, b: nameMap.get(b) || b, matches: 0, aWins: 0, bWins: 0, draws: 0 };
        row.matches += 1;
        if (score[0] === score[1]) row.draws += 1;
        else {
          const teamAWon = score[0] > score[1];
          const aWon = (aSide === "A" && teamAWon) || (aSide === "B" && !teamAWon);
          row.aWins += aWon ? 1 : 0;
          row.bWins += aWon ? 0 : 1;
        }
        rows.set(key, row);
      }
    }
  }

  return [...rows.values()].map((row) => ({ ...row, balance: 1 - Math.abs(row.aWins - row.bWins) / Math.max(row.matches, 1) }));
}

function matchMmrChanges(match: Match, players: Player[], history: MmrHistory[]) {
  const participantKeys = new Set([...splitTeam(match.team_a), ...splitTeam(match.team_b)].map(normalizeName));
  return players
    .filter((player) => participantKeys.has(normalizeName(player.name)) || participantKeys.has(normalizeName(player.display_name)))
    .map((player) => {
      const row = history.find((item) => item.match_id === match.id && item.player_id === player.id);
      const before = numberOr(row?.mmr_before, NaN);
      const after = numberOr(row?.mmr_after, NaN);
      return {
        key: normalizeName(displayName(player)),
        name: displayName(player),
        delta: Math.round(after - before)
      };
    })
    .filter((row) => Number.isFinite(row.delta));
}

function currentRun(playerName: string, result: "W" | "L", match: Match, matches: Match[]) {
  return currentRunWhere(playerName, match, matches, (item) => item === result);
}

function currentRunWhere(playerName: string, match: Match, matches: Match[], predicate: (result: "W" | "D" | "L") => boolean) {
  const rows = playerResults(playerName, matchesUpTo(match, matches)).reverse();
  let count = 0;
  for (const row of rows) {
    if (!predicate(row.result)) break;
    count += 1;
  }
  return count;
}

function previousRun(playerName: string, result: "W" | "L", match: Match, matches: Match[]) {
  const rows = playerResults(playerName, matchesBefore(match, matches)).reverse();
  let count = 0;
  for (const row of rows) {
    if (row.result !== result) break;
    count += 1;
  }
  return count;
}

function playerResults(playerName: string, matches: Match[]) {
  const key = normalizeName(playerName);
  return [...matches]
    .sort(compareMatchesAsc)
    .flatMap((match) => {
      const teamA = splitTeam(match.team_a).map(normalizeName);
      const teamB = splitTeam(match.team_b).map(normalizeName);
      if (teamA.includes(key)) return [{ match, result: resultFor(match, "A") }];
      if (teamB.includes(key)) return [{ match, result: resultFor(match, "B") }];
      return [];
    });
}

function sortStoryCandidates(candidates: StoryCandidate[], match: Match) {
  const unique = new Map<string, StoryCandidate>();
  for (const candidate of candidates) {
    if (!unique.has(candidate.line)) unique.set(candidate.line, candidate);
  }

  return [...unique.values()].sort((a, b) => {
    return b.priority - a.priority || stableHash(`${match.id}-${a.key}`) - stableHash(`${match.id}-${b.key}`);
  });
}

function uniqueStrings(values: string[]) {
  return [...new Set(values.filter(Boolean))];
}

function dedupeChanges(values: ReportChange[]) {
  const seen = new Set<string>();
  return values.filter((item) => {
    const key = `${item.label}|${item.value}|${item.detail}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function stablePick<T>(items: T[], seed: string | number | null | undefined) {
  return items[Math.abs(stableHash(String(seed || ""))) % items.length];
}

function stableHash(value: string) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  return hash;
}

function matchesUpTo(match: Match, matches: Match[]) {
  return matches.filter((row) => compareMatchesAsc(row, match) <= 0);
}

function seasonMatchesUpTo(match: Match, matches: Match[]) {
  const season = String(match.date || "").slice(0, 4);
  if (!/^\d{4}$/.test(season)) return matchesUpTo(match, matches);
  return matchesUpTo(match, matches).filter((row) => String(row.date || "").startsWith(`${season}-`));
}

function matchesBefore(match: Match, matches: Match[]) {
  return matches.filter((row) => compareMatchesAsc(row, match) < 0);
}

function playersForTeam(value: string | null | undefined, players: Player[]) {
  const keys = splitTeam(value).map(normalizeName);
  return keys
    .map((key) => players.find((player) => normalizeName(player.name) === key || normalizeName(player.display_name) === key))
    .filter((player): player is Player => Boolean(player));
}

function ratingBeforeMatch(player: Player, match: Match, history: MmrHistory[]) {
  const row = history.find((item) => item.match_id === match.id && item.player_id === player.id);
  return numberOr(row?.mmr_before, numberOr(player.mmr, STARTING_MMR));
}

function displayFromName(name: string, players: Player[]) {
  const key = normalizeName(name);
  const player = players.find((item) => normalizeName(item.name) === key || normalizeName(item.display_name) === key);
  return player ? displayName(player) : name;
}

function displayFromMap(name: string, nameMap: Map<string, string>) {
  return nameMap.get(normalizeName(name)) || name;
}

function countOverlap(a: Set<string>, b: Set<string>) {
  let count = 0;
  for (const item of a) {
    if (b.has(item)) count += 1;
  }
  return count;
}

function compareMatchesAsc(a: Match, b: Match) {
  return String(a.date || "").localeCompare(String(b.date || "")) || Number(a.id || 0) - Number(b.id || 0);
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function numberOr(value: unknown, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function swapResult(result: "A" | "B" | "DRAW") {
  return result === "A" ? "B" : result === "B" ? "A" : "DRAW";
}

function orientedScoreLabel(scoreValue: string | null | undefined, swapped: boolean) {
  const score = scoreParts(scoreValue);
  if (!score) return scoreValue || "-";
  return swapped ? `${score[1]}-${score[0]}` : `${score[0]}-${score[1]}`;
}

function resultScoreLabel(match: Match, result: "A" | "B" | "DRAW") {
  const score = scoreParts(match.score);
  if (!score) return match.score || "the recorded score";
  if (result === "B") return `${score[1]}-${score[0]}`;
  return `${score[0]}-${score[1]}`;
}

function ratingPoints(rating: number) {
  const scaled = ((rating - 900) / 200) * 80;
  return Math.round(Math.max(0, Math.min(80, scaled)));
}

function minimumAwardMatches(matchCount: number) {
  if (matchCount <= 3) return 1;
  return Math.max(3, Math.ceil(matchCount * 0.25));
}

function signed(value: number) {
  return value > 0 ? `+${value}` : String(value);
}
