"use client";

import { useEffect, useMemo, useState } from "react";
import { formatUkDate, resultFor, scoreParts, splitTeam, normalizeName, type MmrHistory, type PlayerSummary } from "@/lib/demo-data";
import { compareFormPlayers, weightedFormScore } from "@/lib/form-engine";

type View = "ratings" | "form";

export function InteractiveCharts({
  enablePreviousGames = true,
  players,
  mmrHistory,
  matchCount
}: {
  enablePreviousGames?: boolean;
  players: PlayerSummary[];
  mmrHistory: MmrHistory[];
  matchCount: number;
}) {
  const [view, setView] = useState<View>("ratings");
  const [minGames, setMinGames] = useState(0);
  const [selectedPlayerId, setSelectedPlayerId] = useState(players[0]?.id || 0);

  useEffect(() => {
    if (!enablePreviousGames && view === "form") {
      setView("ratings");
    }
  }, [enablePreviousGames, view]);

  const rows = useMemo(() => {
    return players
      .filter((player) => (player.matches_played || 0) >= minGames)
      .sort((a, b) => {
        if (view === "form") {
          return compareFormPlayers(a, b);
        }
        return (b.mmr || 0) - (a.mmr || 0);
      });
  }, [minGames, players, view]);
  const tableRows = useMemo(() => {
    return rows.map((player) => view === "form" ? previousFiveRow(player) : periodRow(player));
  }, [rows, view]);

  const leader = rows[0];
  const topRating = useMemo(() => [...players].sort((a, b) => Number(b.mmr || 0) - Number(a.mmr || 0))[0], [players]);
  const topForm = useMemo(() => [...players].sort(compareFormPlayers)[0], [players]);
  const attendanceLeader = useMemo(() => [...players].sort((a, b) => Number(b.matches_played || 0) - Number(a.matches_played || 0))[0], [players]);
  const avgMmr = useMemo(
    () => players.reduce((sum, player) => sum + Number(player.mmr || 0), 0) / Math.max(players.length, 1),
    [players]
  );
  const attendancePct = attendanceLeader ? Math.round((Number(attendanceLeader.matches_played || 0) / Math.max(matchCount, 1)) * 100) : 0;
  const selectedPlayer = players.find((player) => player.id === selectedPlayerId) || players[0];
  const selectedHistory = useMemo(
    () => mmrHistory.filter((row) => row.player_id === selectedPlayer?.id),
    [mmrHistory, selectedPlayer?.id]
  );
  const chartTitle = view === "form" ? "Previous 5 Games" : "Rating Leaders";
  const progression = useMemo(() => {
    if (!selectedHistory.length) {
      return {
        points: "",
        min: 0,
        mid: 0,
        max: 0,
        startDate: "",
        midDate: "",
        endDate: ""
      };
    }
    const baseline = Number(selectedHistory[0]?.mmr_before ?? selectedPlayer?.periodMmrStart ?? 1000);
    const displayStart = Number(selectedPlayer?.periodMmrStart ?? baseline);
    const values = selectedHistory.map((row) => displayStart + (Number(row.mmr_after || row.mmr_before || baseline) - baseline));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const paddedMin = Math.floor((min - 8) / 10) * 10;
    const paddedMax = Math.ceil((max + 8) / 10) * 10;
    const points = values
      .map((value, index) => {
        const x = selectedHistory.length === 1 ? 6 : 6 + (index / (selectedHistory.length - 1)) * 88;
        const y = 92 - ((value - paddedMin) / Math.max(paddedMax - paddedMin, 1)) * 84;
        return `${x},${y}`;
      })
      .join(" ");
    const midIndex = Math.floor((selectedHistory.length - 1) / 2);
    return {
      points,
      min: paddedMin,
      mid: Math.round((paddedMin + paddedMax) / 2),
      max: paddedMax,
      startDate: shortDate(selectedHistory[0]?.date),
      midDate: shortDate(selectedHistory[midIndex]?.date),
      endDate: shortDate(selectedHistory[selectedHistory.length - 1]?.date)
    };
  }, [selectedHistory, selectedPlayer?.periodMmrStart]);

  return (
    <div className="interactive-stack">
      <section className="stats-snapshot-grid">
        <div className="snapshot-card primary">
          <span>Rating leader</span>
          <strong>{topRating?.label || "-"}</strong>
          <small>{topRating ? `${Math.round(topRating.mmr || 0)} MMR` : "No players"}</small>
        </div>
        <div className="snapshot-card">
          <span>{enablePreviousGames ? "Previous 5 form" : "Goal diff leader"}</span>
          <strong>{enablePreviousGames ? topForm?.label || "-" : goalDiffLeader(players)?.label || "-"}</strong>
          <small>
            {enablePreviousGames
              ? topForm ? `${weightedFormScore(topForm)} form pts` : "No form"
              : goalDiffLeader(players) ? `${signed(goalDiffLeader(players)!.goalDiff)} goal diff` : "No players"}
          </small>
        </div>
        <div className="snapshot-card">
          <span>Attendance</span>
          <strong>{attendanceLeader?.label || "-"}</strong>
          <small>{attendanceLeader ? `${attendancePct}% attendance` : "No games"}</small>
        </div>
        <div className="snapshot-card">
          <span>Average MMR</span>
          <strong>{Math.round(avgMmr)}</strong>
          <small>League average rating</small>
        </div>
      </section>

      <div className="segmented">
        <button className={view === "ratings" ? "active" : ""} onClick={() => setView("ratings")}>Ratings</button>
        {enablePreviousGames ? (
          <button className={view === "form" ? "active" : ""} onClick={() => setView("form")}>Previous 5 Games</button>
        ) : null}
      </div>

      <div className="control-bar compact">
        <label>
          <span>Minimum games</span>
          <input
            max={40}
            min={0}
            onChange={(event) => setMinGames(Number(event.target.value))}
            type="range"
            value={minGames}
          />
        </label>
        <div className="range-value">{minGames}+</div>
      </div>

      <section className="two-col">
        <div className="panel">
          <div className="section-title-row">
            <div>
              <span>{leader ? `Leader: ${leader.label}` : "No leader"}</span>
              <h2>{chartTitle}</h2>
            </div>
          </div>
          <div className="chart-bars">
            {rows.slice(0, 10).map((player, index) => {
              const value =
                view === "form"
                  ? weightedFormScore(player)
                  : Math.round((player.mmr || 0) - 900);
              const max = Math.max(...rows.slice(0, 10).map((item) => {
                if (view === "form") return weightedFormScore(item);
                return Math.round((item.mmr || 0) - 900);
              }), 1);

              return (
                <div className="chart-row" key={player.id}>
                  <strong>{index + 1}. {player.label}</strong>
                  <span><i style={{ width: `${(value / max) * 100}%` }} /></span>
                  <em>{view === "ratings" ? Math.round(player.mmr || 0) : value}</em>
                </div>
              );
            })}
          </div>
        </div>

        <div className="panel">
          <h2>MMR Progression</h2>
          <div className="control-bar">
            <label>
              <span>Player</span>
              <select value={selectedPlayerId} onChange={(event) => setSelectedPlayerId(Number(event.target.value))}>
                {players.map((player) => (
                  <option value={player.id} key={player.id}>{player.label}</option>
                ))}
              </select>
            </label>
          </div>
          <div className="mmr-chart">
            <div className="mmr-y-axis" aria-hidden="true">
              <span>{progression.max || "-"}</span>
              <span>{progression.mid || "-"}</span>
              <span>{progression.min || "-"}</span>
            </div>
            <div>
              <div className="line-chart">
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${selectedPlayer?.label || "Player"} MMR chart`}>
                  <polyline points={progression.points} />
                </svg>
              </div>
              <div className="mmr-x-axis" aria-hidden="true">
                <span>{progression.startDate || "-"}</span>
                <span>{progression.midDate || "-"}</span>
                <span>{progression.endDate || "-"}</span>
              </div>
            </div>
          </div>
          <div className="section-subhead">
            <strong>{selectedPlayer ? Math.round(selectedPlayer.mmr || 0) : "-"}</strong>
            <span>{selectedPlayer?.label || "Player"} rating in this view</span>
          </div>
        </div>
      </section>

      <div className="panel">
        <h2>Table View</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Player</th>
                  <th>MMR</th>
                  <th>Matches</th>
                  <th>W-D-L</th>
                  <th>Goal Diff</th>
                </tr>
              </thead>
              <tbody>
                {tableRows.map((row, index) => (
                  <tr key={row.id}>
                    <td>{index + 1}</td>
                    <td>{row.label}</td>
                    <td>{Math.round(row.mmr || 0)}</td>
                    <td>{row.matches}</td>
                    <td>{row.wins}-{row.draws}-{row.losses}</td>
                    <td>{signed(row.goalDiff)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
      </div>
    </div>
  );
}

function shortDate(value: string | null | undefined) {
  return formatUkDate(value);
}

function previousFiveRow(player: PlayerSummary) {
  const matches = player.allMatches.slice(0, 5);
  return summarizePlayerMatches(player, matches);
}

function periodRow(player: PlayerSummary) {
  return {
    id: player.id,
    label: player.label,
    mmr: player.mmr,
    matches: player.matches_played || 0,
    wins: player.wins || 0,
    draws: player.draws || 0,
    losses: player.losses || 0,
    goalDiff: player.goalDiff
  };
}

function summarizePlayerMatches(player: PlayerSummary, matches: PlayerSummary["allMatches"]) {
  const keys = [player.name, player.display_name, player.label].map(normalizeName).filter(Boolean);
  let wins = 0;
  let draws = 0;
  let losses = 0;
  let goalDiff = 0;

  for (const match of matches) {
    const score = scoreParts(match.score);
    if (!score) continue;
    const teamA = splitTeam(match.team_a).map(normalizeName);
    const teamB = splitTeam(match.team_b).map(normalizeName);
    const side = keys.some((key) => teamA.includes(key)) ? "A" : keys.some((key) => teamB.includes(key)) ? "B" : null;
    if (!side) continue;

    const result = resultFor(match, side);
    wins += result === "W" ? 1 : 0;
    draws += result === "D" ? 1 : 0;
    losses += result === "L" ? 1 : 0;
    goalDiff += side === "A" ? score[0] - score[1] : score[1] - score[0];
  }

  return { id: player.id, label: player.label, mmr: player.mmr, matches: matches.length, wins, draws, losses, goalDiff };
}

function goalDiffLeader(players: PlayerSummary[]) {
  return [...players].sort((a, b) => b.goalDiff - a.goalDiff)[0];
}

function signed(value: number) {
  return `${value > 0 ? "+" : ""}${value}`;
}
