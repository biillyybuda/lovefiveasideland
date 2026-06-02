"use client";

import { useMemo, useState } from "react";
import { MatchCard } from "@/components/match-card";
import { makeNameMap, normalizeName, resultFor, scoreParts, splitTeam, type Match, type Player, type PlayerSummary } from "@/lib/demo-data";
import { chemistryScoreFor, evidenceLabel, rivalryScoreFor } from "@/lib/relationship-scoring";

type LinkRow = {
  key: string;
  name: string;
  matches: number;
  wins: number;
  draws: number;
  losses: number;
  goalDiff: number;
  goalGapTotal: number;
  scoreSum: number;
  actualRate: number;
  expectedRate: number;
  residual: number;
  score: number;
};

export function InteractiveInsights({
  initialPlayerId,
  players,
  rawPlayers,
  matches
}: {
  initialPlayerId?: string;
  players: PlayerSummary[];
  rawPlayers: Player[];
  matches: Match[];
}) {
  const [selectedId, setSelectedId] = useState(() => {
    const raw = Number(initialPlayerId);
    return players.some((player) => player.id === raw) ? raw : null;
  });
  const [depth, setDepth] = useState("all");
  const nameMap = useMemo(() => makeNameMap(rawPlayers), [rawPlayers]);
  const selected = selectedId ? players.find((player) => player.id === selectedId) || null : null;

  const selectedMatches = useMemo(() => {
    if (!selected) return [];
    if (depth === "5") return selected.recentMatches.slice(0, 5);
    if (depth === "10") return selected.recentMatches.slice(0, 10);
    return selected.allMatches;
  }, [depth, matches, selected]);
  const playerLinks = useMemo(
    () => selected ? buildPlayerLinks(selected, matches, nameMap) : { teammates: [], rivals: [] },
    [matches, nameMap, selected]
  );

  return (
    <div className="interactive-stack">
      <div className="control-bar">
        <label>
          <span>Player</span>
          <select
            value={selectedId ? String(selectedId) : ""}
            onChange={(event) => setSelectedId(event.target.value ? Number(event.target.value) : null)}
          >
            <option value="">Select a player</option>
            {players.map((player) => (
              <option value={player.id} key={player.id}>{player.label}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Game history</span>
          <select value={depth} onChange={(event) => setDepth(event.target.value)}>
            <option value="5">Recent 5</option>
            <option value="10">Recent 10</option>
            <option value="all">All games</option>
          </select>
        </label>
      </div>

      <section className="two-col">
        <div className="panel">
          <h2>{selected?.label || "Player Profile"}</h2>
          {selected ? (
            <>
              <div className="profile-grid">
                <div><span>MMR</span><strong>{Math.round(selected.mmr || 0)}</strong></div>
                <div><span>MMR change</span><strong>{formatDelta(selected.periodMmrChange)}</strong></div>
                <div><span>Record</span><strong>{selected.wins || 0}-{selected.draws || 0}-{selected.losses || 0}</strong></div>
                <div><span>Form</span><strong>{selected.form.join(" ") || "-"}</strong></div>
                <div><span>Goal diff</span><strong>{selected.goalDiff > 0 ? "+" : ""}{selected.goalDiff}</strong></div>
              </div>
              <div className="relationship-list insight-links">
                <h3>Best Teammates</h3>
                {playerLinks.teammates.slice(0, 4).map((row) => (
                  <div className="relationship-row compact" key={`team-${row.name}`}>
                    <strong>{row.name}</strong>
                    <span>{row.matches} games</span>
                    <span>{row.wins}-{row.draws}-{row.losses}</span>
                    <span>Chem {row.score.toFixed(1)}</span>
                    <small>{teammateEvidence(row)}</small>
                  </div>
                ))}
                <h3>Toughest Opponents</h3>
                {playerLinks.rivals.slice(0, 4).map((row) => (
                  <div className="relationship-row compact" key={`rival-${row.name}`}>
                    <strong>{row.name}</strong>
                    <span>{row.matches} meetings</span>
                    <span>{row.wins}-{row.draws}-{row.losses}</span>
                    <span>Rivalry {row.score.toFixed(1)}</span>
                    <small>{opponentEvidence(row)}</small>
                  </div>
                ))}
              </div>
              <div className="form-dots insight-form">
                {selected.form.map((result, index) => (
                  <b className={`form ${result.toLowerCase()}`} key={index}>{result}</b>
                ))}
              </div>
            </>
          ) : (
            <div className="empty-state">
              <strong>Select a player</strong>
              <p>Choose someone above to see their MMR, record, form, best teammates, toughest opponents and game history.</p>
            </div>
          )}
        </div>

        <div className="panel">
          <h2>{selected ? (depth === "all" ? "All Games" : `Recent ${depth}`) : "Game History"}</h2>
          {selected ? (
            <div className="mini-match-list">
              {selectedMatches.map((match) => (
                <MatchCard highlightPlayer={selected.label} match={match} nameMap={nameMap} key={match.id} />
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <strong>No player selected</strong>
              <p>The game list will appear here once you pick a player.</p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function buildPlayerLinks(player: PlayerSummary, matches: Match[], nameMap: Map<string, string>) {
  const keys = [player.name, player.display_name, player.label].map(normalizeName).filter(Boolean);
  const teammates = new Map<string, LinkRow>();
  const rivals = new Map<string, LinkRow>();
  const playerPointRates = buildPlayerPointRates(matches);
  const selectedPointRate = pointRateForKeys(keys, playerPointRates);

  for (const match of matches) {
    const score = scoreParts(match.score);
    const teamA = splitTeam(match.team_a);
    const teamB = splitTeam(match.team_b);
    const teamAKeys = teamA.map(normalizeName);
    const teamBKeys = teamB.map(normalizeName);
    const side = keys.some((key) => teamAKeys.includes(key)) ? "A" : keys.some((key) => teamBKeys.includes(key)) ? "B" : null;
    if (!side || !score) continue;

    const ownTeam = side === "A" ? teamA : teamB;
    const otherTeam = side === "A" ? teamB : teamA;
    const result = resultFor(match, side);
    const goalDiff = side === "A" ? score[0] - score[1] : score[1] - score[0];
    const goalGap = Math.abs(score[0] - score[1]);

    for (const name of ownTeam) {
      if (keys.includes(normalizeName(name))) continue;
      touch(teammates, name, nameMap, result, goalDiff, goalGap);
    }
    for (const name of otherTeam) {
      touch(rivals, name, nameMap, result, goalDiff, goalGap);
    }
  }

  const teammateRows = [...teammates.values()]
    .map((row) => {
      const actualRate = row.scoreSum / Math.max(row.matches, 1);
      const expectedRate = (selectedPointRate + pointRateForKey(row.key, playerPointRates)) / 2;
      const residual = actualRate - expectedRate;
      return { ...row, actualRate, expectedRate, residual, score: chemistryScore({ ...row, actualRate, expectedRate, residual }) };
    })
    .sort((a, b) => b.score - a.score || b.residual - a.residual || b.matches - a.matches);
  const rivalRows = [...rivals.values()]
    .map((row) => {
      const actualRate = row.scoreSum / Math.max(row.matches, 1);
      const expectedRate = expectedHeadToHeadRate(selectedPointRate, pointRateForKey(row.key, playerPointRates));
      const residual = actualRate - expectedRate;
      return { ...row, actualRate, expectedRate, residual, score: rivalryScore({ ...row, actualRate, expectedRate, residual }) };
    })
    .sort((a, b) => b.score - a.score || b.matches - a.matches || opponentThreatValue(b) - opponentThreatValue(a));

  return { teammates: teammateRows, rivals: rivalRows };
}

function touch(rows: Map<string, LinkRow>, name: string, nameMap: Map<string, string>, result: "W" | "D" | "L", goalDiff: number, goalGap: number) {
  const key = normalizeName(name);
  const row = rows.get(key) || {
    key,
    name: nameMap.get(key) || name,
    matches: 0,
    wins: 0,
    draws: 0,
    losses: 0,
    goalDiff: 0,
    goalGapTotal: 0,
    scoreSum: 0,
    actualRate: 0.5,
    expectedRate: 0.5,
    residual: 0,
    score: 0
  };
  row.matches += 1;
  row.wins += result === "W" ? 1 : 0;
  row.draws += result === "D" ? 1 : 0;
  row.losses += result === "L" ? 1 : 0;
  row.scoreSum += result === "W" ? 1 : result === "D" ? 0.5 : 0;
  row.goalDiff += goalDiff;
  row.goalGapTotal += goalGap;
  rows.set(key, row);
}

function chemistryScore(row: LinkRow) {
  return chemistryScoreFor(row);
}

function teammateEvidence(row: LinkRow) {
  if (row.matches < 4) return evidenceLabel(row.matches);
  return impactLabel("Lift", row.residual);
}

function opponentEvidence(row: LinkRow) {
  if (row.matches < 4) return evidenceLabel(row.matches);
  const threat = row.expectedRate - row.actualRate;
  if (threat > 0.005) return impactLabel("Threat", threat);
  if (threat < -0.005) return impactLabel("Edge", Math.abs(threat));
  return "On par";
}

function rivalryScore(row: LinkRow) {
  const base = rivalryScoreFor({
    matches: row.matches,
    winsA: row.wins,
    winsB: row.losses,
    draws: row.draws,
    totalGoalGap: row.goalGapTotal
  });
  const dangerBonus = row.matches ? (row.losses / row.matches) * Math.min(1.2, row.matches / 3) : 0;
  const threatBonus = row.matches >= 4 ? Math.max(0, row.expectedRate - row.actualRate) * 4 : 0;
  return base + dangerBonus + threatBonus;
}

function formatDelta(value: number | null | undefined) {
  const safe = Math.round(Number(value || 0));
  return safe > 0 ? `+${safe}` : String(safe);
}

function buildPlayerPointRates(matches: Match[]) {
  const rows = new Map<string, number[]>();
  for (const match of matches) {
    const score = scoreParts(match.score);
    if (!score) continue;
    for (const side of ["A", "B"] as const) {
      const result = resultFor(match, side);
      const value = result === "W" ? 1 : result === "D" ? 0.5 : 0;
      const team = splitTeam(side === "A" ? match.team_a : match.team_b).map(normalizeName).filter(Boolean);
      for (const name of team) {
        const values = rows.get(name) || [];
        values.push(value);
        rows.set(name, values);
      }
    }
  }

  const rates = new Map<string, number>();
  for (const [name, values] of rows) {
    rates.set(name, values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1));
  }
  return rates;
}

function pointRateForKey(key: string, rates: Map<string, number>) {
  return rates.get(normalizeName(key)) ?? 0.5;
}

function pointRateForKeys(keys: string[], rates: Map<string, number>) {
  for (const key of keys) {
    const value = rates.get(normalizeName(key));
    if (typeof value === "number") return value;
  }
  return 0.5;
}

function expectedHeadToHeadRate(playerRate: number, opponentRate: number) {
  return clamp(0.5 + (playerRate - opponentRate) * 0.7, 0.18, 0.82);
}

function opponentThreatValue(row: LinkRow) {
  return row.matches >= 4 ? Math.max(0, row.expectedRate - row.actualRate) : 0;
}

function impactLabel(label: "Lift" | "Threat" | "Edge", value: number) {
  return `${label} +${Math.max(0, Math.round(value * 100))}`;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
