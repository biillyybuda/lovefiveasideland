import { displayName, formatUkDate, resultFor, scoreParts, splitTeam, type Match, type MmrHistory, type Player } from "./demo-data";
import { compareMatches, replayBalance, type BalancePrediction } from "./balance-model";
import { seasonAwardSnapshot, signed } from "./season-awards";
import { groupsForLeague, type LeagueGroup } from "./league-groups";

type Side = "A" | "B";
type Story = { key: string; priority: number; label: string; title: string; detail: string; people: string[]; scope: string; headline?: string };
type Change = { label: string; name: string; before: string; after: string; detail: string; priority: number };
type Record = { played: number; wins: number; draws: number; losses: number };
export type GroupContext = { name: string; teamLabel: string; side: Side; members: string[]; guests: string[];
  description: string; minimumTogether: number; seasonRecord: Record; beforeRecord: Record; allFiveRecord: Record;
  currentCoreRecord: Record; coreSize: number; streak: number; outcome: "W" | "D" | "L" };

export type MatchReport = {
  match: Match; season: string; matchNumber: number; headline: string; summary: string; resultLabel: string;
  dateLabel: string; scoreLabel: string; teamA: string[]; teamB: string[]; teamALabel: string; teamBLabel: string;
  highlights: Story[]; stories: Story[]; changes: Change[];
  standings: { label: string; name: string; value: string; detail: string }[];
  playerNotes: { name: string; side: Side; appearances: number; record: string; form: string; before: number | null; after: number | null;
    delta: number | null; note: string }[];
  scoreContext: { total: number; averageBefore: number | null; priorMatches: number; rank: number; tied: boolean };
  rating: { avgA: number | null; avgB: number | null; prediction: BalancePrediction | null };
  relationships: { title: string; detail: string }[];
  groups: GroupContext[];
  previousMeeting: { id: number; date: string; score: string; shared: number; sameSide: number; sides: { label: string; kept: string[]; joined: string[]; left: string[] }[] } | null;
  note: string;
};

const key = (value: string | null | undefined) => String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
const ordinal = (n: number) => `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as {[n: number]: string})[n % 10] || "th"}`;
const rankWord = (n: number) => ["", "first", "second", "third"][n] || ordinal(n);
const recordText = (record: Record) => `${record.wins}W ${record.draws}D ${record.losses}L`;
function record(outcomes: string[]): Record {
  return { played: outcomes.length, wins: outcomes.filter(r => r === "W").length,
    draws: outcomes.filter(r => r === "D").length, losses: outcomes.filter(r => r === "L").length };
}
function streak(outcomes: string[], outcome: string) {
  let count = 0;
  for (const value of [...outcomes].reverse()) { if (value !== outcome) break; count++; }
  return count;
}
export function completedReportMatch(match: Match) {
  const date = String(match.date || "");
  const timestamp = Date.parse(`${date}T00:00:00Z`);
  return match.processed === 1 && Boolean(scoreParts(match.score)) && /^\d{4}-\d{2}-\d{2}$/.test(date) &&
    Number.isFinite(timestamp) && new Date(timestamp).toISOString().slice(0, 10) === date;
}

/** Everything in the report stops at this match (date, then ID). No current player
 * counters or current ratings are used as historical evidence. */
export function buildMatchReport({ match, matches, players, mmrHistory, awardPlayers = players.filter(p => p.is_active !== 0), leagueId = 0,
  groups = groupsForLeague(leagueId) }: {
  match: Match; matches: Match[]; players: Player[]; mmrHistory: MmrHistory[]; awardPlayers?: Player[]; leagueId?: number; groups?: LeagueGroup[];
}): MatchReport {
  if (!completedReportMatch(match)) throw new Error("A match review needs a completed result and a valid date.");
  const lookup = new Map<string, Player>();
  for (const p of players) for (const name of [p.name, p.display_name]) if (key(name)) lookup.set(key(name), p);
  const canonical = (value: string | null) => splitTeam(value).map(name => lookup.get(key(name))?.name || name).join(", ");
  const canonicalMatch = (m: Match) => ({ ...m, team_a: canonical(m.team_a), team_b: canonical(m.team_b) });
  match = canonicalMatch(match);
  const all = [...new Map([...matches, match].filter(completedReportMatch).filter(m => compareMatches(m, match) <= 0)
    .map(m => [m.id, canonicalMatch(m)])).values()].sort(compareMatches);
  const previous = all.filter(m => compareMatches(m, match) < 0);
  const season = match.date!.slice(0, 4);
  const year = all.filter(m => m.date!.startsWith(`${season}-`));
  const priorYear = year.filter(m => m.id !== match.id);
  const names = (m: Match, side: Side) => splitTeam(side === "A" ? m.team_a : m.team_b);
  const label = (name: string) => { const p = lookup.get(key(name)); return p ? displayName(p) : name; };
  const contains = (m: Match, side: Side, name: string) => names(m, side).some(n => key(n) === key(name));
  const sideOf = (m: Match, name: string): Side | null => contains(m, "A", name) ? "A" : contains(m, "B", name) ? "B" : null;
  const outcomes = (rows: Match[], name: string) => rows.flatMap(m => { const side = sideOf(m, name); return side ? [resultFor(m, side)] : []; });
  const teamA = names(match, "A").map(label), teamB = names(match, "B").map(label);
  const participants = [...names(match, "A"), ...names(match, "B")];
  const participantIds = new Set(participants.map(n => lookup.get(key(n))?.id));
  const score = scoreParts(match.score)!;
  const scoreLabel = `${score[0]}–${score[1]}`;
  const winner: Side | null = score[0] > score[1] ? "A" : score[1] > score[0] ? "B" : null;
  const candidates: Story[] = [];
  const changes: Change[] = [];
  const add = (story: Story) => candidates.push(story);
  const beforeAwards = seasonAwardSnapshot(awardPlayers, previous, mmrHistory, season);
  const afterAwards = seasonAwardSnapshot(awardPlayers, all, mmrHistory, season);
  let climbingMvp: { name: string; rank: number } | null = null;
  let winningNewcomer: string | null = null;

  for (const [index, row] of afterAwards.mvp.entries()) {
    if (!participantIds.has(row.player.id)) continue;
    const beforeIndex = beforeAwards.mvp.findIndex(r => r.key === row.key);
    const before = beforeAwards.mvp[beforeIndex];
    const rank = index + 1;
    if (before && (before.value !== row.value || beforeIndex !== index)) {
      const leaderGap = afterAwards.mvp[0].value - row.value;
      const beforeGap = beforeAwards.mvp[0].value - before.value;
      changes.push({ label: `${season} MVP race`, name: row.label, before: `${ordinal(beforeIndex + 1)} · ${before.value} pts`,
        after: `${ordinal(rank)} · ${row.value} pts`, detail: rank === 1 ? "Leads the season MVP race" : `Gap to the leader: ${beforeGap} → ${leaderGap} pts`,
        priority: (beforeIndex > index ? 90 : 40) - rank });
      if (beforeIndex > index && rank <= 3) {
        if (!climbingMvp) climbingMvp = { name: row.label, rank };
        add({ key: `mvp-${row.key}`, priority: 95 - rank, label: "MVP movement", title: `${row.label} moves into ${rankWord(rank)}`,
          detail: `${ordinal(beforeIndex + 1)} → ${ordinal(rank)} in the ${season} MVP race; ${before.value} → ${row.value} points.${rank > 1 ? ` Gap to ${afterAwards.mvp[0].label}: ${beforeGap} → ${leaderGap}.` : ""}`,
          people: [row.label], scope: season });
      }
    } else if (!before) {
      changes.push({ label: `${season} MVP race`, name: row.label, before: "Not yet qualified", after: `${ordinal(rank)} · ${row.value} pts`,
        detail: "Now meets the season appearance requirement", priority: 75 - rank });
    }
  }
  for (const [index, row] of afterAwards.improved.rows.entries()) {
    if (!participantIds.has(Number(row.key))) continue;
    const before = beforeAwards.improved.rows.find(r => r.key === row.key);
    if (!before || before.value !== row.value) changes.push({ label: afterAwards.improved.title, name: row.label,
      before: before ? `${signed(before.value)} ${before.unit}` : "Outside the top five",
      after: `${signed(row.value)} ${row.unit} · ${ordinal(index + 1)}`, detail: afterAwards.improved.subtitle, priority: 55 - index });
  }

  const playerNotes = participants.map(name => {
    const player = lookup.get(key(name)), side = sideOf(match, name)!;
    const history = mmrHistory.filter(h => h.match_id === match.id && h.player_id === player?.id);
    const ratingRow = history.length === 1 ? history[0] : undefined;
    const before = ratingRow?.mmr_before ?? null, after = ratingRow?.mmr_after ?? null;
    const priorCareer = outcomes(previous, name), priorSeason = outcomes(priorYear, name), current = resultFor(match, side);
    const seasonResults = [...priorSeason, current], seasonRecord = record(seasonResults);
    const person = label(name), total = priorCareer.length + 1;
    let note = "";
    if (!priorCareer.length) {
      note = "First appearance in the recorded league history";
      if (current === "W" && !winningNewcomer) winningNewcomer = person;
      add({ key: `first-${name}`, priority: 100, label: "First recorded appearance", title: `${person} ${current === "W" ? "starts with a win" : current === "D" ? "starts with a draw" : "makes a first appearance"}`,
        detail: `${note}${current === "W" ? `, on the winning side of a ${scoreLabel} result` : ""}.`, people: [person], scope: "Recorded career" });
    } else if (!priorSeason.length) {
      note = `First appearance of ${season}; ${total} across the recorded archive`;
      add({ key: `season-first-${name}`, priority: 49, label: "Season return", title: `${person}'s first game of ${season}`,
        detail: `${priorCareer.length} earlier recorded appearances.`, people: [person], scope: season });
    }
    if (total >= 25 && total % 25 === 0) {
      note = `${ordinal(total)} recorded career appearance; ${ordinal(seasonResults.length)} of ${season}`;
      add({ key: `career-${name}`, priority: 66, label: "Career milestone", title: `${person} reaches ${total} appearances`,
        detail: `${seasonResults.length} in ${season}. Career count covers the recorded league archive.`, people: [person], scope: "Recorded career" });
    } else if (seasonResults.length >= 10 && seasonResults.length % 10 === 0) {
      add({ key: `appearance-${name}`, priority: 47, label: "Season milestone", title: `${person}'s ${ordinal(seasonResults.length)} game of ${season}`,
        detail: `${recordText(seasonRecord)} this season.`, people: [person], scope: season });
    }
    const priorLosses = streak(priorSeason, "L"), wins = streak(seasonResults, "W"), losses = streak(seasonResults, "L");
    if (priorLosses >= 3 && current !== "L") {
      note = `Ended a ${priorLosses}-game losing run in ${season}`;
      add({ key: `run-${name}`, priority: 80, label: "Run ended", title: `${person} ends a ${priorLosses}-game losing run`,
        detail: `${current === "W" ? "A win" : "A draw"} after ${priorLosses} consecutive defeats in their ${season} appearances.`, people: [person], scope: season });
    } else if (wins >= 3 || losses >= 3) {
      const count = wins || losses;
      note ||= `${count} consecutive ${wins ? "wins" : "defeats"} in ${season}`;
      add({ key: `run-${name}`, priority: wins ? 59 : 44, label: "Season form", title: `${person}: ${count} ${wins ? "wins" : "defeats"} in a row`,
        detail: `Across their own ${season} appearances.`, people: [person], scope: season });
    }
    return { name: person, side, appearances: seasonResults.length, record: recordText(seasonRecord), form: seasonResults.slice(-5).join(" "),
      before, after, delta: before !== null && after !== null ? after - before : null, note };
  });

  // Same-side and opposite-side records are seasonal and attributed to the pair,
  // rather than implying either individual caused the team's result.
  const partnerships: { title: string; detail: string; games: number; wins: number }[] = [];
  const rivalries: { title: string; detail: string; games: number }[] = [];
  for (let i = 0; i < participants.length; i++) for (let j = i + 1; j < participants.length; j++) {
    const one = participants[i], two = participants[j], same = sideOf(match, one) === sideOf(match, two);
    const pairRows = year.filter(m => { const a = sideOf(m, one), b = sideOf(m, two); return a && b && (same ? a === b : a !== b); });
    const results = pairRows.map(m => resultFor(m, sideOf(m, one)!));
    const r = record(results), a = label(one), b = label(two);
    if (same && r.played >= 3) {
      partnerships.push({ title: `${a} + ${b}`, detail: `${recordText(r)} in ${r.played} games together in ${season}.`, games: r.played, wins: r.wins });
      if (r.played >= 4 && r.wins === 1 && results.at(-1) === "W") add({ key: `pair-${one}-${two}`, priority: 88,
        label: "Partnership breakthrough", title: `${a} + ${b}: first win of ${season}`,
        detail: `Their first win together at the ${ordinal(r.played)} attempt this season; previously ${recordText(record(results.slice(0, -1)))}.`, people: [a, b], scope: season });
    } else if (!same && r.played >= 3) rivalries.push({ title: `${a} vs ${b}`, detail: `${r.wins} wins for ${a}, ${r.draws} draws, ${r.losses} wins for ${b} in ${r.played} meetings in ${season}.`, games: r.played });
  }

  const groupContexts: GroupContext[] = groups.flatMap(group => {
    const members = new Set(group.playerIds);
    const ids = (m: Match, side: Side) => names(m, side).flatMap(n => { const p = lookup.get(key(n)); return p ? [p.id] : []; });
    const groupSide = (m: Match): Side | null => {
      const a = ids(m, "A").filter(id => members.has(id)).length, b = ids(m, "B").filter(id => members.has(id)).length;
      return a >= group.minimumTogether && a > b ? "A" : b >= group.minimumTogether && b > a ? "B" : null;
    };
    const side = groupSide(match);
    if (!side) return [];
    const core = ids(match, side).filter(id => members.has(id));
    const guests = names(match, side).filter(n => !members.has(lookup.get(key(n))?.id ?? -1)).map(label);
    const teamLabel = guests.length ? `${group.name} + ${guests.join(" & ")}` : group.name;
    const rows = year.flatMap(m => { const s = groupSide(m); return s ? [{ match: m, side: s, result: resultFor(m, s) }] : []; });
    const groupRecord = record(rows.map(r => r.result)), beforeRecord = record(rows.filter(r => r.match.id !== match.id).map(r => r.result));
    const allFiveRecord = record(rows.filter(r => group.playerIds.every(id => ids(r.match, r.side).includes(id))).map(r => r.result));
    const currentCoreRecord = record(rows.filter(r => core.every(id => ids(r.match, r.side).includes(id))).map(r => r.result));
    const outcome = resultFor(match, side), run = streak(rows.map(r => r.result), outcome);
    const previousLosses = streak(rows.slice(0, -1).map(r => r.result), "L");
    const groupHeadline = outcome === "W" ? previousLosses >= 3 ? `${group.name} break the losing spell` : `${group.name} reunite and get the win`
      : outcome === "D" ? `${group.name} reunion ends all square`
      : beforeRecord.played >= 3 && beforeRecord.losses / beforeRecord.played >= 0.6 ? [
        `${teamLabel}, same old story`, `${group.name} reunion, familiar result`, `${group.name} are back, and so are the defeats`
      ][groupRecord.played % 3]
      : `${group.name} reunite, but the win goes elsewhere`;
    add({ key: `group-${group.name}`, priority: 110, label: `${group.name} watch`, headline: groupHeadline,
      title: !groupRecord.wins ? `Winless in ${groupRecord.played} ${groupRecord.played === 1 ? "reunion" : "reunions"}`
        : !beforeRecord.wins && outcome === "W" ? `First ${group.name} win of ${season}` : `${groupRecord.wins} ${groupRecord.wins === 1 ? "win" : "wins"} from ${groupRecord.played} reunions`,
      detail: `${groupRecord.wins} ${groupRecord.wins === 1 ? "win" : "wins"}, ${groupRecord.draws} ${groupRecord.draws === 1 ? "draw" : "draws"} and ${groupRecord.losses} ${groupRecord.losses === 1 ? "defeat" : "defeats"} in ${groupRecord.played} games with ${group.minimumTogether}+ ${group.name} players on one side in ${season}.`,
      people: names(match, side).map(label), scope: season });
    return [{ name: group.name, teamLabel, side, members: names(match, side).filter(n => members.has(lookup.get(key(n))?.id ?? -1)).map(label), guests,
      description: group.description, minimumTogether: group.minimumTogether, seasonRecord: groupRecord, beforeRecord, allFiveRecord, currentCoreRecord,
      coreSize: core.length, streak: run, outcome }];
  });

  const total = score[0] + score[1], priorTotals = priorYear.map(m => scoreParts(m.score)!).map(s => s[0] + s[1]);
  const averageBefore = priorTotals.length ? priorTotals.reduce((sum, n) => sum + n, 0) / priorTotals.length : null;
  const rank = priorTotals.filter(n => n > total).length + 1, tied = priorTotals.includes(total);
  if (priorTotals.length >= 5 && rank <= 5) add({ key: "goals", priority: 70, label: "Score context",
    title: `${tied ? "Joint " : ""}${ordinal(rank)}-highest goal total of ${season}`,
    detail: `${total} goals, compared with ${averageBefore!.toFixed(1)} per game in the ${priorTotals.length} earlier season matches.`, people: [], scope: season });

  const stories = candidates.sort((a, b) => b.priority - a.priority || a.key.localeCompare(b.key));
  const highlights = stories.slice(0, 3);
  const teamALabel = groupContexts.find(g => g.side === "A")?.teamLabel || "Team A";
  const teamBLabel = groupContexts.find(g => g.side === "B")?.teamLabel || "Team B";
  const resultLabel = winner ? `${winner === "A" ? teamALabel : teamBLabel} win` : "Draw";
  const defaultHeadline = winner ? `${winner === "A" ? teamALabel : teamBLabel} win ${scoreLabel}` : `Honours even at ${scoreLabel}`;
  const headline = groupContexts.length ? stories.find(s => s.key.startsWith("group-"))!.headline! : winningNewcomer ? `${winningNewcomer} starts with a win${climbingMvp ? ` as ${climbingMvp.name} moves into ${rankWord(climbingMvp.rank)}` : ""}`
    : stories[0]?.title || defaultHeadline;
  const lead = winner ? `${winner === "A" ? teamALabel : teamBLabel} won ${scoreLabel}${groupContexts.length && (winner === "A" ? teamBLabel : teamALabel) !== `Team ${winner === "A" ? "B" : "A"}` ? ` against ${winner === "A" ? teamBLabel : teamALabel}` : ""}.` : `The match finished ${scoreLabel}.`;
  const summary = `${lead} ${stories.filter(s => !s.key.startsWith("group-")).slice(0, 4).map(s => {
    if (s.key.startsWith("first-")) return `${s.people[0]} made a first appearance in the recorded league history.`;
    if (s.key.startsWith("pair-")) return `${s.people.join(" and ")} secured their first win together in ${season}.`;
    return `${s.title.replace(" moves into ", " moved into ").replace(" ends a ", " ended a ").replace(" reaches ", " reached ")}.`;
  }).join(" ")}`;
  const avg = (side: Side) => { const notes = playerNotes.filter(n => n.side === side); return notes.length && notes.every(n => n.before !== null)
    ? notes.reduce((sum, n) => sum + n.before!, 0) / notes.length : null; };
  let prediction: BalancePrediction | null = null;
  try { prediction = replayBalance(players, all).records.find(r => r.matchId === match.id)?.prediction || null; }
  catch { /* Incomplete historical lineups cannot support a reconstructed estimate. */ }

  const previousMeeting = priorYear.map(m => {
    const a = names(match, "A").map(key), b = names(match, "B").map(key);
    const ma = names(m, "A").map(key), mb = names(m, "B").map(key);
    const overlap = (x: string[], y: string[]) => x.filter(n => y.includes(n)).length;
    const straight = overlap(a, ma) + overlap(b, mb), swapped = overlap(a, mb) + overlap(b, ma);
    return { m, swapped: swapped > straight, sameSide: Math.max(straight, swapped), shared: overlap([...a, ...b], [...ma, ...mb]) };
  }).filter(r => r.shared >= 8 && r.sameSide >= 6).sort((a, b) => b.sameSide - a.sameSide || b.shared - a.shared || compareMatches(b.m, a.m))[0];
  const standings = [];
  const mvp = afterAwards.mvp[0], improved = afterAwards.improved.rows[0];
  if (mvp) standings.push({ label: `${season} MVP leader`, name: mvp.label, value: `${mvp.value} pts`, detail: "Established season rating + attendance" });
  if (improved) standings.push({ label: `${season} ${afterAwards.improved.title.toLowerCase()}`, name: improved.label,
    value: `${signed(improved.value)} ${improved.unit}`, detail: afterAwards.improved.subtitle });
  return { match, season, matchNumber: year.length, headline, summary, resultLabel, dateLabel: formatUkDate(match.date), scoreLabel,
    teamA, teamB, teamALabel, teamBLabel, highlights, stories, changes: changes.sort((a, b) => b.priority - a.priority), standings, playerNotes,
    scoreContext: { total, averageBefore, priorMatches: priorYear.length, rank, tied }, rating: { avgA: avg("A"), avgB: avg("B"), prediction },
    groups: groupContexts, relationships: [partnerships.sort((a, b) => b.wins - a.wins || b.games - a.games)[0], rivalries.sort((a, b) => b.games - a.games)[0]].filter(Boolean),
    previousMeeting: previousMeeting ? { id: previousMeeting.m.id, date: formatUkDate(previousMeeting.m.date), score: previousMeeting.m.score || "",
      shared: previousMeeting.shared, sameSide: previousMeeting.sameSide, sides: (["A", "B"] as Side[]).map(side => {
        const historic = names(previousMeeting.m, previousMeeting.swapped ? side === "A" ? "B" : "A" : side);
        const current = names(match, side);
        return { label: `Team ${side}`, kept: current.filter(n => historic.some(h => key(n) === key(h))).map(label),
          joined: current.filter(n => !historic.some(h => key(n) === key(h))).map(label), left: historic.filter(n => !current.some(c => key(c) === key(n))).map(label) };
      }) } : null, note: match.review_note?.trim() || "" };
}
