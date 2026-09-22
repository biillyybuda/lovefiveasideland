import type { Match, Player } from "./demo-data";

export type Competition = "league" | "immunity_shield";
export type CompetitionScope = Competition | "all";
export const COMPETITION_POLICY = "league-shield-v1";
export const competitionLabels: Record<CompetitionScope, string> = {
  league: "League", immunity_shield: "MMR Immunity Shield", all: "All competitions"
};

/** Missing fields are legacy league records. Unknown values must never become rated games. */
export function matchCompetition(match: Pick<Match, "competition">): Competition {
  if (match.competition === undefined || match.competition === null) return "league";
  if (match.competition === "league" || match.competition === "immunity_shield") return match.competition;
  throw new Error("Unknown match competition. Refresh before saving.");
}
export function isLeagueMatch(match: Pick<Match, "competition">) { return matchCompetition(match) === "league"; }
export function competitionScope(value: string | null | undefined): CompetitionScope {
  return value === "all" || value === "immunity_shield" ? value : "league";
}
export function selectMatches(matches: Match[], scope: CompetitionScope = "league", season = "all") {
  return matches.filter(m => m.processed === 1 && (scope === "all" || matchCompetition(m) === scope) &&
    (season === "all" || String(m.date || "").startsWith(`${season}-`)));
}
export function shieldWinner(match: Match): "A" | "B" | null {
  if (isLeagueMatch(match) || match.processed !== 1) return null;
  const score = /^\s*(\d{1,3})\s*[-:]\s*(\d{1,3})\s*$/.exec(match.score || "");
  if (!score || Number(score[1]) === Number(score[2])) return null;
  return Number(score[1]) > Number(score[2]) ? "A" : "B";
}
export function shieldWins(player: Pick<Player, "name" | "display_name">, matches: Match[]) {
  const clean = (s: string | null) => (s || "").trim().toLowerCase().replace(/\s+/g, " ");
  const keys = new Set([clean(player.name), clean(player.display_name)].filter(Boolean));
  const won = new Set<number>();
  for (const match of matches) {
    const winner = shieldWinner(match);
    if (winner && (winner === "A" ? match.team_a : match.team_b)?.split(",").some(name => keys.has(clean(name)))) won.add(match.id);
  }
  return won.size;
}

/** Browsing scope never controls rating replay or the balancing model. */
export function competitionHref(href: string, scope: CompetitionScope) {
  const [path, hash] = href.split("#");
  const [base, query] = path.split("?");
  const params = new URLSearchParams(query);
  params.set("competition", scope);
  return `${base}?${params}${hash ? `#${hash}` : ""}`;
}
