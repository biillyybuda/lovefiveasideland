import Link from "next/link";
import { formatTeam, formatUkDate, normalizeName, type Match } from "@/lib/demo-data";

export function MatchCard({
  match,
  nameMap,
  highlightPlayer,
  reportHref
}: {
  match: Match;
  nameMap: Map<string, string>;
  highlightPlayer?: string;
  reportHref?: string;
}) {
  const teamA = formatTeam(match.team_a, nameMap);
  const teamB = formatTeam(match.team_b, nameMap);
  const [scoreA, scoreB] = String(match.score || "-").split("-");
  const highlightKey = normalizeName(highlightPlayer);

  function playerPill(player: string, side: "a" | "b") {
    const highlighted = highlightKey && normalizeName(player) === highlightKey;
    return (
      <span className={highlighted ? "pill highlighted" : "pill"} key={`${side}-${match.id}-${player}`}>
        {player}
      </span>
    );
  }

  return (
    <article className="match-card">
      <div className="match-card-top">
        <div>
          <span>Matchday</span>
          <strong>{formatUkDate(match.date)}</strong>
        </div>
        <div className="score-badge">{match.result || "Result"}</div>
      </div>
      <div className="scoreline scoreboard">
        <span className="team-a">Team A</span>
        <strong>{scoreA || "-"}</strong>
        <em>-</em>
        <strong>{scoreB || "-"}</strong>
        <span className="team-b">Team B</span>
      </div>
      <div className="lineups">
        <div className="team-panel a">
          <div className="stat-label">Team A</div>
          <div className="pill-row">
            {teamA.map((player) => playerPill(player, "a"))}
          </div>
        </div>
        <div className="team-panel b">
          <div className="stat-label">Team B</div>
          <div className="pill-row">
            {teamB.map((player) => playerPill(player, "b"))}
          </div>
        </div>
      </div>
      {reportHref ? (
        <div className="match-card-actions">
          <Link className="mini-link" href={reportHref}>Match report</Link>
        </div>
      ) : null}
    </article>
  );
}
