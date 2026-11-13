import { displayName, scoreParts, type Match, type Player } from "./demo-data";
import { matchDay, resolveLineup } from "./balance-model";
import { matchCompetition, getCompetitionLabels, type Competition } from "./competition";

export type Role = "GK" | "DF" | "MF" | "ST";
export type Side = "blue" | "red";
export type FormationId = "1-1-2-1" | "1-2-1-1" | "1-1-1-2";
export type Slot = { role: Role; x: number; y: number };
export const ROLE_NAMES: Record<Role, string> = { GK: "Goalkeeper", DF: "Defender", MF: "Midfielder", ST: "Striker" };
export const SIDES: Side[] = ["blue", "red"];
export const PITCH = { width: 600, height: 1120, cardWidth: 154, cardHeight: 190 };
const slot = (role: Role, x: number, y: number): Slot => ({ role, x, y });
export const FORMATIONS: Record<FormationId, { label: string; slots: Slot[]; edges: [number, number][] }> = {
  "1-1-2-1": { label: "1–1–2–1", slots: [slot("GK", 300, 970), slot("DF", 300, 715), slot("MF", 155, 460), slot("MF", 445, 460), slot("ST", 300, 205)], edges: [[0, 1], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4]] },
  "1-2-1-1": { label: "1–2–1–1", slots: [slot("GK", 300, 970), slot("DF", 155, 715), slot("DF", 445, 715), slot("MF", 300, 460), slot("ST", 300, 205)], edges: [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4]] },
  "1-1-1-2": { label: "1–1–1–2", slots: [slot("GK", 300, 970), slot("DF", 300, 715), slot("MF", 300, 460), slot("ST", 155, 205), slot("ST", 445, 205)], edges: [[0, 1], [1, 2], [2, 3], [2, 4], [3, 4]] }
};
export type TeamLineup = { formation: FormationId; ids: number[] };
export type PositionLock = { side: Side; role: Role };
export type LineupDraft = {
  competition?: Competition;
  version: 1; date: string; blue: TeamLineup; red: TeamLineup;
  locks: Record<number, PositionLock>; preferences: Record<number, Role>;
  autoFormations: Record<Side, boolean>;
};
export type LineupSeed = { date: string; blue: number[]; red: number[]; revision: number; competition?: Competition };
export type LinkColour = "green" | "amber" | "red";
export const LINK_COLOURS: Record<LinkColour, string> = { green: "#39e765", amber: "#ffc442", red: "#ff3d59" };
export type PlayerWins = { games: number; wins: number };
export type Chemistry = { players: Map<number, PlayerWins>; pairs: Map<string, PlayerWins> };
export type PlayerLink = PlayerWins & { a: number; b: number; winRate: number | null; averageRate: number | null; difference: number | null; colour: LinkColour };
export const pairKey = (a: number, b: number) => [a, b].sort((x, y) => x - y).join(":");
export const isFormation = (value: unknown): value is FormationId => typeof value === "string" && Object.hasOwn(FORMATIONS, value);
export const isRole = (value: unknown): value is Role => ["GK", "DF", "MF", "ST"].includes(String(value));

/** Win percentage, with draws counted as non-wins in both sides of the comparison. */
export function buildLineupChemistry(players: Player[], matches: Match[], date: string): Chemistry {
  matchDay(date);
  const result: Chemistry = { players: new Map(), pairs: new Map() };
  const seen = new Set<number>();
  for (const match of matches) {
    if (match.processed !== 1 || !match.date || match.date > date || seen.has(match.id)) continue;
    const score = scoreParts(match.score);
    if (!score) throw new Error(`Match ${match.id} has an invalid score.`);
    const { a, b } = resolveLineup(match, players);
    seen.add(match.id);
    for (const [team, won] of [[a, score[0] > score[1]], [b, score[1] > score[0]]] as const) {
      for (const id of team) {
        const row = result.players.get(id) || { games: 0, wins: 0 };
        result.players.set(id, { games: row.games + 1, wins: row.wins + Number(won) });
      }
      for (let i = 0; i < team.length; i++) for (let j = i + 1; j < team.length; j++) {
        const key = pairKey(team[i], team[j]);
        const row = result.pairs.get(key) || { games: 0, wins: 0 };
        result.pairs.set(key, { games: row.games + 1, wins: row.wins + Number(won) });
      }
    }
  }
  return result;
}

export function playerLink(chemistry: Chemistry, a: number, b: number): PlayerLink {
  const pair = chemistry.pairs.get(pairKey(a, b)) || { games: 0, wins: 0 };
  const pa = chemistry.players.get(a), pb = chemistry.players.get(b);
  const winRate = pair.games ? pair.wins / pair.games : null;
  const averageRate = pa?.games && pb?.games ? (pa.wins / pa.games + pb.wins / pb.games) / 2 : null;
  const difference = winRate !== null && averageRate !== null ? winRate - averageRate : null;
  // No shared games is red by the group's requested display rule, not a fabricated 0% record.
  const colour = difference === null ? "red" : difference > 0.05 + 1e-10 ? "green" : difference < -0.05 - 1e-10 ? "red" : "amber";
  return { a, b, ...pair, winRate, averageRate, difference, colour };
}

export function lineupLinks(team: TeamLineup, chemistry: Chemistry) {
  return FORMATIONS[team.formation].edges.map(([a, b]) => ({ from: a, to: b, ...playerLink(chemistry, team.ids[a], team.ids[b]) }));
}

export function savedRole(value: string | null | undefined): Role | undefined {
  const text = String(value || "").trim().toLowerCase();
  if (/^(gk|keeper|goalkeeper|goalie)$/.test(text)) return "GK";
  if (/^(df|def|defender|defence|defense|cb|lb|rb)$/.test(text)) return "DF";
  if (/^(mf|mid|midfielder|midfield|cm|cam|cdm|lm|rm)$/.test(text)) return "MF";
  if (/^(st|striker|forward|attacker|attack|cf|lw|rw)$/.test(text)) return "ST";
  return undefined;
}

export function validateDraft(value: unknown, players: Player[]): LineupDraft {
  if (!value || typeof value !== "object") throw new Error("Choose ten players first.");
  const draft = value as LineupDraft;
  matchCompetition(draft);
  if (draft.version !== 1) throw new Error("This lineup draft is no longer supported.");
  matchDay(draft.date);
  const available = new Set(players.filter(p => p.is_active !== 0).map(p => p.id));
  const ids: number[] = [];
  for (const side of SIDES) {
    const team = draft[side];
    if (!team || !isFormation(team.formation) || !Array.isArray(team.ids) || team.ids.length !== 5) throw new Error("Each side needs five players and every position covered.");
    if (team.ids.some(id => !Number.isInteger(id) || !available.has(id))) throw new Error("A player in this lineup is no longer available. Choose your ten again.");
    ids.push(...team.ids);
  }
  if (new Set(ids).size !== 10) throw new Error("A player cannot appear twice in a lineup.");
  if (!draft.autoFormations || SIDES.some(side => typeof draft.autoFormations[side] !== "boolean")) throw new Error("Invalid formation settings.");
  if (!draft.locks || typeof draft.locks !== "object" || Array.isArray(draft.locks) || !draft.preferences || typeof draft.preferences !== "object" || Array.isArray(draft.preferences)) throw new Error("Invalid position settings.");
  for (const [key, lock] of Object.entries(draft.locks)) {
    const id = Number(key);
    if (!lock || !SIDES.includes(lock.side) || !isRole(lock.role)) throw new Error("Invalid position lock.");
    const index = draft[lock.side].ids.indexOf(id);
    if (index < 0 || FORMATIONS[draft[lock.side].formation].slots[index].role !== lock.role) throw new Error("A locked player has moved. Unlock their position first.");
  }
  for (const [id, role] of Object.entries(draft.preferences)) if (!available.has(Number(id)) || !isRole(role)) throw new Error("Invalid preferred position.");
  return draft;
}

export function swapLineupPlayers(draft: LineupDraft, from: { side: Side; index: number }, to: { side: Side; index: number }): LineupDraft {
  const a = draft[from.side]?.ids[from.index], b = draft[to.side]?.ids[to.index];
  if (!a || !b) throw new Error("Choose two player cards to swap.");
  if (a === b) return draft;
  if (draft.locks[a] || draft.locks[b]) throw new Error("Unlock both players before swapping them.");
  const next = structuredClone(draft);
  next[from.side].ids[from.index] = b;
  next[to.side].ids[to.index] = a;
  return next;
}

export function swapLineupColours(draft: LineupDraft): LineupDraft {
  const next = structuredClone(draft);
  [next.blue, next.red] = [next.red, next.blue];
  [next.autoFormations.blue, next.autoFormations.red] = [next.autoFormations.red, next.autoFormations.blue];
  for (const lock of Object.values(next.locks)) lock.side = lock.side === "blue" ? "red" : "blue";
  return next;
}

function permutations(ids: number[]): number[][] {
  return ids.length < 2 ? [ids] : ids.flatMap((id, index) => permutations(ids.filter((_, i) => i !== index)).map(rest => [id, ...rest]));
}

/** Rank only within the existing five: preferences first, then a blend of links and attacking roles. */
export function suggestTeam(draft: LineupDraft, side: Side, players: Player[], chemistry: Chemistry, formation?: FormationId): TeamLineup {
  const byId = new Map(players.map(p => [p.id, p]));
  const candidates: { team: TeamLineup; preference: number; score: number; movement: number }[] = [];
  const current = draft[side];
  for (const id of formation ? [formation] : Object.keys(FORMATIONS) as FormationId[]) {
    const layout = FORMATIONS[id];
    for (const ids of permutations([...current.ids].sort((a, b) => a - b))) {
      if (ids.some((playerId, index) => draft.locks[playerId] && (draft.locks[playerId].side !== side || draft.locks[playerId].role !== layout.slots[index].role))) continue;
      let preference = 0, roleFit = 0, movement = 0;
      ids.forEach((playerId, index) => {
        const player = byId.get(playerId)!;
        const role = layout.slots[index].role;
        const preferred = draft.preferences[playerId] || savedRole(player.position);
        if (preferred) preference += preferred === role ? 1 : 0;
        const profile = player.style_profile;
        const confidence = Math.min(1, Math.max(0, profile?.historicMatches || 0) / 8);
        const goals = Math.min(1, Math.max(0, profile?.goalsPerGame || 0) / 3);
        const assists = Math.min(1, Math.max(0, profile?.assistsPerGame || 0) / 2);
        // Missing profiles are neutral. Saves are clips, not keeper ability or minutes in goal.
        const evidence = role === "ST" ? goals : role === "MF" ? assists * 0.7 + goals * 0.3 : role === "DF" ? assists * 0.25 : 0;
        roleFit += 0.5 + confidence * (evidence - 0.5);
        if (current.ids.indexOf(playerId) !== index || current.formation !== id) movement++;
      });
      const team = { formation: id, ids };
      const links = lineupLinks(team, chemistry);
      const quality = links.reduce((sum, link) => sum + (link.colour === "green" ? 1 : link.colour === "red" ? -1 : 0), 0) / links.length;
      candidates.push({ team, preference, score: quality * 0.55 + roleFit / 5 * 0.45, movement });
    }
  }
  candidates.sort((a, b) => b.preference - a.preference || b.score - a.score || a.movement - b.movement || a.team.ids.join(",").localeCompare(b.team.ids.join(",")));
  if (!candidates.length) throw new Error("That formation cannot keep every position lock. Unlock a player or choose another shape.");
  return candidates[0].team;
}

export function initialLineup(date: string, blue: number[], red: number[], players: Player[], chemistry: Chemistry, competition: Competition = "league"): LineupDraft {
  const draft: LineupDraft = { version: 1, competition, date, blue: { formation: "1-1-2-1", ids: blue }, red: { formation: "1-1-2-1", ids: red }, locks: {}, preferences: {}, autoFormations: { blue: true, red: true } };
  validateDraft(draft, players);
  for (const side of SIDES) draft[side] = suggestTeam(draft, side, players, chemistry);
  return draft;
}

/** Keep whole graphemes and fit long names identically in the pitch and PNG. */
export function shieldName(name: string) {
  const letters = [...new Intl.Segmenter("en", { granularity: "grapheme" }).segment(name.toUpperCase())].map(s => s.segment);
  const words = name.toUpperCase().trim().split(/\s+/);
  const lines: string[] = [];
  let line = "";
  for (const word of words) {
    if (line && [...line, ...word].length > 12) { lines.push(line); line = ""; }
    if (word.length > 14) {
      if (line) { lines.push(line); line = ""; }
      const chunks = [...new Intl.Segmenter("en", { granularity: "grapheme" }).segment(word)].map(s => s.segment);
      while (chunks.length > 14) lines.push(chunks.splice(0, 14).join(""));
      line = chunks.join("");
    } else line = line ? `${line} ${word}` : word;
  }
  if (line) lines.push(line);
  return { lines: lines.length ? lines : ["PLAYER"], size: Math.min(20, 48 / Math.max(1, lines.length), letters.length > 14 ? 18 : 20) };
}

export function lineupText(draft: LineupDraft, players: Player[], cupName?: string) {
  const byId = new Map(players.map(p => [p.id, p]));
  return `LOVE FIVE · ${getCompetitionLabels(cupName)[matchCompetition(draft)]} · ${draft.date}${draft.competition === "immunity_shield" ? " · MMR protected; a draw awards no trophy" : ""}\n\n${SIDES.map(side => `${side === "blue" ? "🔵 BLUE" : "🔴 RED"} TEAM · ${FORMATIONS[draft[side].formation].label}\n${draft[side].ids.map((id, index) => `${FORMATIONS[draft[side].formation].slots[index].role}: ${displayName(byId.get(id)!)}`).join("\n")}`).join("\n\n")}`;
}
