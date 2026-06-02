"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import type { PlayerSummary } from "@/lib/demo-data";

type SortKey = "mmr" | "played" | "wins" | "goalDiff" | "streak";

export function InteractivePlayers({
  players,
  showFitness = false
}: {
  players: PlayerSummary[];
  showFitness?: boolean;
}) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("mmr");
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const visiblePlayers = useMemo(() => {
    return players
      .filter((player) => player.label.toLowerCase().includes(query.trim().toLowerCase()))
      .sort((a, b) => {
        if (sortKey === "played") return (b.matches_played || 0) - (a.matches_played || 0);
        if (sortKey === "wins") return (b.wins || 0) - (a.wins || 0);
        if (sortKey === "goalDiff") return b.goalDiff - a.goalDiff;
        if (sortKey === "streak") return (b.win_streak || 0) - (a.win_streak || 0);
        return (b.mmr || 0) - (a.mmr || 0);
      });
  }, [players, query, sortKey]);
  const selected = selectedId ? visiblePlayers.find((player) => player.id === selectedId) : null;

  return (
    <div className="interactive-stack">
      <div className="control-bar">
        <label>
          <span>Search</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find a player" />
        </label>
        <label>
          <span>Sort by</span>
          <select value={sortKey} onChange={(event) => setSortKey(event.target.value as SortKey)}>
            <option value="mmr">MMR</option>
            <option value="played">Matches played</option>
            <option value="wins">Wins</option>
            <option value="goalDiff">Goal difference</option>
            <option value="streak">Win streak</option>
          </select>
        </label>
      </div>

      <div className="player-board">
        {visiblePlayers.map((player, index) => (
          <button
            className="player-card"
            key={player.id}
            onClick={() => {
              if (showFitness) {
                setSelectedId(selected?.id === player.id ? null : player.id);
              } else {
                router.push(`/demo/charts?view=player&player=${player.id}#player-insights`);
              }
            }}
          >
            <span className="player-rank">{index + 1}</span>
            <span className="player-main">
              <strong>{player.label}</strong>
              <small>{player.wins || 0}-{player.draws || 0}-{player.losses || 0} - {player.matches_played || 0} played</small>
            </span>
            <span className="player-rating">{Math.round(player.mmr || 0)}</span>
            <span className="form-dots inline">
              {player.form.slice(0, 5).map((result, resultIndex) => (
                <b className={`form ${result.toLowerCase()}`} key={`${player.id}-${resultIndex}`}>
                  {result}
                </b>
              ))}
            </span>
            <span className={player.goalDiff >= 0 ? "goal-diff positive" : "goal-diff"}>
              {player.goalDiff > 0 ? "+" : ""}{player.goalDiff}
            </span>
          </button>
        ))}
      </div>

      {showFitness && selected ? (
        <aside className="player-drawer">
          <div className="section-title-row">
            <div>
              <span>Player Detail</span>
              <h2>{selected.label}</h2>
            </div>
            <button className="mini-link" onClick={() => setSelectedId(null)}>Close</button>
          </div>
          <div className="profile-grid">
            <div><span>MMR</span><strong>{Math.round(selected.mmr || 0)}</strong></div>
            <div><span>MMR change</span><strong>{formatDelta(selected.periodMmrChange)}</strong></div>
            <div><span>Record</span><strong>{selected.wins || 0}-{selected.draws || 0}-{selected.losses || 0}</strong></div>
            <div><span>Goals for</span><strong>{selected.goalsFor}</strong></div>
            <div><span>Goal diff</span><strong>{selected.goalDiff > 0 ? "+" : ""}{selected.goalDiff}</strong></div>
            <div><span>Fitness</span><strong>{selected.fitness || "Average"}</strong></div>
          </div>
        </aside>
      ) : null}
    </div>
  );
}

function formatDelta(value: number | null | undefined) {
  const safe = Math.round(Number(value || 0));
  return safe > 0 ? `+${safe}` : String(safe);
}
