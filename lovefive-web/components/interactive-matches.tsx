"use client";

import { useMemo, useState } from "react";
import { MatchCard } from "@/components/match-card";
import {
  formatUkDate,
  formatTeam,
  makeNameMap,
  normalizeName,
  scoreParts,
  splitTeam,
  type Match,
  type Player
} from "@/lib/demo-data";

export function InteractiveMatches({
  matches,
  players
}: {
  matches: Match[];
  players: Player[];
}) {
  const [query, setQuery] = useState("");
  const [season, setSeason] = useState("all");
  const [player, setPlayer] = useState("all");
  const [result, setResult] = useState("all");

  const nameMap = useMemo(() => makeNameMap(players), [players]);
  const seasons = useMemo(() => ["all", ...Array.from(new Set(matches.map((match) => String(match.date || "").slice(0, 4))))], [matches]);

  const visibleMatches = useMemo(() => {
    return matches.filter((match) => {
      const teamNames = [...formatTeam(match.team_a, nameMap), ...formatTeam(match.team_b, nameMap)];
      const score = scoreParts(match.score);
      const text = `${match.date} ${formatUkDate(match.date)} ${match.score} ${teamNames.join(" ")}`.toLowerCase();
      const rawTeamA = splitTeam(match.team_a).map(normalizeName);
      const rawTeamB = splitTeam(match.team_b).map(normalizeName);
      const selectedSide = player === "all" ? null : rawTeamA.includes(player) ? "a" : rawTeamB.includes(player) ? "b" : null;
      const selectedPlayer =
        player === "all" ||
        selectedSide !== null;
      const outcome = resultKey(match, selectedSide);

      return (
        text.includes(query.trim().toLowerCase()) &&
        (season === "all" || String(match.date || "").startsWith(season)) &&
        selectedPlayer &&
        (result === "all" || result === outcome)
      );
    });
  }, [matches, nameMap, player, query, result, season]);

  return (
    <div className="interactive-stack">
      <div className="control-bar">
        <label>
          <span>Search</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Score, date or player" />
        </label>
        <label>
          <span>Season</span>
          <select value={season} onChange={(event) => setSeason(event.target.value)}>
            {seasons.map((item) => (
              <option value={item} key={item}>{item === "all" ? "All seasons" : item}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Player</span>
          <select value={player} onChange={(event) => setPlayer(event.target.value)}>
            <option value="all">All players</option>
            {players.map((item) => (
              <option value={normalizeName(item.name)} key={item.id}>
                {item.display_name || item.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Result</span>
          <select value={result} onChange={(event) => setResult(event.target.value)}>
            <option value="all">All results</option>
            <option value="win">{player === "all" ? "Team A wins" : "Player wins"}</option>
            <option value="loss">{player === "all" ? "Team B wins" : "Player losses"}</option>
            <option value="draw">Draws</option>
          </select>
        </label>
      </div>

      <div className="section-subhead">
        <strong>{visibleMatches.length}</strong>
        <span>matches found</span>
      </div>

      <div className="grid">
        {visibleMatches.map((match) => (
          <MatchCard
            highlightPlayer={player === "all" ? undefined : nameMap.get(player) || player}
            match={match}
            nameMap={nameMap}
            reportHref={`/demo/matches/${match.id}`}
            key={match.id}
          />
        ))}
      </div>
    </div>
  );
}

function resultKey(match: Match, selectedSide: "a" | "b" | null) {
  const score = scoreParts(match.score);
  const stored = String(match.result || "").trim().toLowerCase();
  const teamAWon = score ? score[0] > score[1] : stored === "a" || stored === "team a";
  const teamBWon = score ? score[1] > score[0] : stored === "b" || stored === "team b";
  const draw = score ? score[0] === score[1] : stored === "draw" || stored === "d";

  if (draw) return "draw";
  if (!selectedSide) return teamAWon ? "win" : teamBWon ? "loss" : "unknown";
  if (selectedSide === "a") return teamAWon ? "win" : teamBWon ? "loss" : "unknown";
  return teamBWon ? "win" : teamAWon ? "loss" : "unknown";
}
