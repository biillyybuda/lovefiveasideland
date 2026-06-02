"use client";

import { useMemo, useState } from "react";
import {
  displayName,
  formatUkDate,
  formatTeam,
  makeNameMap,
  normalizeName,
  resultFor,
  scoreParts,
  splitTeam,
  type Match,
  type Player,
  type duoChemistry
} from "@/lib/demo-data";
import { chemistryScoreFor, evidenceLabel, rivalryScoreFor } from "@/lib/relationship-scoring";

type DuoRow = ReturnType<typeof duoChemistry>[number];
type RelationshipMode = "head" | "team" | "matchup";

type PairStats = {
  a: string;
  b: string;
  aKeys: string[];
  bKeys: string[];
  together: number;
  togetherWins: number;
  togetherDraws: number;
  togetherLosses: number;
  togetherGoalDiff: number;
  togetherScoreSum: number;
  chemistryActualRate: number;
  chemistryExpectedRate: number;
  chemistryResidual: number;
  faced: number;
  aWins: number;
  bWins: number;
  facedDraws: number;
  totalGoalGap: number;
  chemistryScore: number;
  rivalryScore: number;
  latestMatch: Match | null;
  latestTogetherMatch: Match | null;
  latestFacedMatch: Match | null;
};

type TeamHistoryHit = {
  match: Match;
  side: "A" | "B";
  selectedCount: number;
  outcome: "W" | "D" | "L";
  goalGap: number;
};

type MatchupHit = {
  match: Match;
  normalizedMatch: Match;
  selectedA: number;
  selectedB: number;
  groupAResult: "W" | "D" | "L";
};

export function InteractiveChemistry({
  players,
  matches,
  initialMode = "head"
}: {
  teammateRows?: DuoRow[];
  matchupRows?: DuoRow[];
  players: Player[];
  matches: Match[];
  initialMode?: RelationshipMode;
}) {
  const nameMap = useMemo(() => makeNameMap(players), [players]);
  const playerOptions = useMemo(
    () => players.map((player) => displayName(player)).sort((a, b) => a.localeCompare(b)),
    [players]
  );
  const [mode, setMode] = useState<RelationshipMode>(initialMode);
  const [playerA, setPlayerA] = useState("");
  const [playerB, setPlayerB] = useState("");
  const [teamPlayers, setTeamPlayers] = useState<string[]>([]);
  const [minTogether, setMinTogether] = useState(2);
  const [groupA, setGroupA] = useState<string[]>([]);
  const [groupB, setGroupB] = useState<string[]>([]);
  const [minGroupA, setMinGroupA] = useState(2);
  const [minGroupB, setMinGroupB] = useState(2);

  const selectedPair = useMemo(() => {
    if (!playerA || !playerB || playerA === playerB) return null;
    const a = players.find((player) => displayName(player) === playerA) || players[0];
    const b = players.find((player) => displayName(player) === playerB) || players.find((player) => player.id !== a?.id) || players[0];
    return a && b ? buildPairStats(a, b, matches) : null;
  }, [matches, playerA, playerB, players]);

  const teamHistory = useMemo(
    () => buildTeamHistory(teamPlayers, minTogether, matches, nameMap),
    [matches, minTogether, nameMap, teamPlayers]
  );
  const matchupHistory = useMemo(
    () => buildMatchupHistory(groupA, groupB, minGroupA, minGroupB, matches, nameMap),
    [groupA, groupB, matches, minGroupA, minGroupB, nameMap]
  );

  function toggleTeamPlayer(player: string) {
    setTeamPlayers((current) =>
      current.includes(player) ? current.filter((item) => item !== player) : [...current, player]
    );
  }

  function toggleGroupPlayer(group: "A" | "B", player: string) {
    if (group === "A") {
      setGroupA((current) => current.includes(player) ? current.filter((item) => item !== player) : [...current, player]);
      setGroupB((current) => current.filter((item) => item !== player));
    } else {
      setGroupB((current) => current.includes(player) ? current.filter((item) => item !== player) : [...current, player]);
      setGroupA((current) => current.filter((item) => item !== player));
    }
  }

  return (
    <div className="interactive-stack" id="relationships">
      <div className="segmented">
        <button className={mode === "head" ? "active" : ""} onClick={() => setMode("head")}>Head-to-head</button>
        <button className={mode === "team" ? "active" : ""} onClick={() => setMode("team")}>Teammates</button>
        <button className={mode === "matchup" ? "active" : ""} onClick={() => setMode("matchup")}>Rivalries</button>
      </div>

      {mode === "head" ? (
        <HeadToHeadView
          nameMap={nameMap}
          players={playerOptions}
          playerA={playerA}
          playerB={playerB}
          selectedPair={selectedPair}
          setPlayerA={setPlayerA}
          setPlayerB={setPlayerB}
        />
      ) : null}

      {mode === "team" ? (
        <TeamHistoryView
          history={teamHistory}
          matches={matches}
          minTogether={minTogether}
          nameMap={nameMap}
          playerOptions={playerOptions}
          selectedPlayers={teamPlayers}
          setMinTogether={setMinTogether}
          togglePlayer={toggleTeamPlayer}
        />
      ) : null}

      {mode === "matchup" ? (
        <MatchupHistoryView
          groupA={groupA}
          groupB={groupB}
          history={matchupHistory}
          minGroupA={minGroupA}
          minGroupB={minGroupB}
          nameMap={nameMap}
          playerOptions={playerOptions}
          setMinGroupA={setMinGroupA}
          setMinGroupB={setMinGroupB}
          toggleGroupPlayer={toggleGroupPlayer}
        />
      ) : null}
    </div>
  );
}

function HeadToHeadView({
  players,
  playerA,
  playerB,
  nameMap,
  selectedPair,
  setPlayerA,
  setPlayerB
}: {
  players: string[];
  playerA: string;
  playerB: string;
  nameMap: Map<string, string>;
  selectedPair: PairStats | null;
  setPlayerA: (value: string) => void;
  setPlayerB: (value: string) => void;
}) {
  const latest = selectedPair?.latestMatch;
  const latestTogether = selectedPair?.latestTogetherMatch;
  const latestFaced = selectedPair?.latestFacedMatch;
  return (
    <div className="interactive-stack">
      <div className="control-bar">
        <label>
          <span>Player A</span>
          <select value={playerA} onChange={(event) => setPlayerA(event.target.value)}>
            <option value="">Select player A</option>
            {players.map((player) => (
              <option value={player} key={`a-${player}`}>{player}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Player B</span>
          <select value={playerB} onChange={(event) => setPlayerB(event.target.value)}>
            <option value="">Select player B</option>
            {players.map((player) => (
              <option value={player} key={`b-${player}`}>{player}</option>
            ))}
          </select>
        </label>
      </div>

      {selectedPair ? (
        <>
          <section className="relationship-hero-grid">
            <article className="relationship-hero-card rivalry">
              <span>Rivalry</span>
              <h2>{selectedPair.a} v {selectedPair.b}</h2>
              <strong>{selectedPair.faced}</strong>
              <small>meetings against each other</small>
            </article>
            <article className="relationship-hero-card chemistry">
              <span>Partnership</span>
              <h2>{selectedPair.a} + {selectedPair.b}</h2>
              <strong>{selectedPair.together}</strong>
              <small>games together</small>
            </article>
            <article className="relationship-hero-card">
              <span>Latest shared match</span>
              <h2>{formatUkDate(latest?.date)}</h2>
              <strong>{latest?.score || "-"}</strong>
              <small>{latest && selectedPair ? sharedMatchLabel(selectedPair, latest) : "No shared match yet"}</small>
            </article>
          </section>

          <section className="relationship-card-grid">
            <SharedMatchSummary
              match={latest || null}
              nameMap={nameMap}
              pair={selectedPair}
              title="Most recent involving both"
            />
            <SharedMatchSummary
              match={latestTogether || null}
              nameMap={nameMap}
              pair={selectedPair}
              title="Last game together"
            />
            <SharedMatchSummary
              match={latestFaced || null}
              nameMap={nameMap}
              pair={selectedPair}
              title="Last game against each other"
            />
          </section>

          <section className="two-col">
            <div className="panel">
              <h2>Head-to-head record</h2>
              <div className="profile-grid">
                <div><span>{selectedPair.a} wins</span><strong>{selectedPair.aWins}</strong></div>
                <div><span>{selectedPair.b} wins</span><strong>{selectedPair.bWins}</strong></div>
                <div><span>Draws</span><strong>{selectedPair.facedDraws}</strong></div>
                <div><span>Avg gap</span><strong>{avgGap(selectedPair)}</strong></div>
                <div><span>Rivalry score</span><strong>{selectedPair.rivalryScore.toFixed(1)}</strong></div>
              </div>
            </div>
            <div className="panel">
              <h2>Team-mate record</h2>
              <div className="profile-grid">
                <div><span>W-D-L</span><strong>{selectedPair.togetherWins}-{selectedPair.togetherDraws}-{selectedPair.togetherLosses}</strong></div>
                <div><span>Win rate</span><strong>{pct(selectedPair.togetherWins, selectedPair.together)}</strong></div>
                <div><span>Goal diff</span><strong>{signed(selectedPair.togetherGoalDiff)}</strong></div>
                <div><span>Chemistry score</span><strong>{selectedPair.chemistryScore.toFixed(1)}</strong></div>
                <div><span>Chem lift</span><strong>{chemistryExpectation(selectedPair)}</strong></div>
                <div><span>Evidence</span><strong>{evidenceLabel(selectedPair.together)}</strong></div>
              </div>
            </div>
          </section>
        </>
      ) : (
        <div className="empty-state">
          <strong>Choose two players</strong>
          <p>Pick a pair to compare their head-to-head record, partnership chemistry and latest shared match.</p>
        </div>
      )}
    </div>
  );
}

function TeamHistoryView({
  history,
  matches,
  minTogether,
  nameMap,
  playerOptions,
  selectedPlayers,
  setMinTogether,
  togglePlayer
}: {
  history: TeamHistoryHit[];
  matches: Match[];
  minTogether: number;
  nameMap: Map<string, string>;
  playerOptions: string[];
  selectedPlayers: string[];
  setMinTogether: (value: number) => void;
  togglePlayer: (player: string) => void;
}) {
  const selectedCap = Math.min(5, selectedPlayers.length);
  const summary = summarizeTeamHistory(history);

  return (
    <div className="interactive-stack">
      <section className="relationship-tool-panel">
        <div className="section-title-row">
          <div>
            <span>Teammate History</span>
            <h2>Shared appearances</h2>
          </div>
        </div>
        <ChipPicker options={playerOptions} selected={selectedPlayers} onToggle={togglePlayer} />
        {selectedPlayers.length > 2 ? (
          <div className="control-bar compact relationship-range">
            <label>
              <span>Minimum selected players together</span>
              <input
                min={2}
                max={selectedCap}
                onChange={(event) => setMinTogether(Number(event.target.value))}
                type="range"
                value={Math.min(minTogether, selectedCap)}
              />
            </label>
            <div className="range-value">{Math.min(minTogether, selectedCap)}+</div>
          </div>
        ) : (
          <p className="muted">Select at least 2 players. Two-player searches require both players together.</p>
        )}
      </section>

      {selectedPlayers.length < 2 ? (
        <div className="panel"><p className="muted">Pick two or more players to find games where they shared a side.</p></div>
      ) : (
        <>
          <RelationshipSummary
            items={[
              ["Matches", String(history.length)],
              ["W-D-L", `${summary.wins}-${summary.draws}-${summary.losses}`],
              ["Win %", pct(summary.wins, history.length)],
              ["Avg Goal Diff", summary.avgGap]
            ]}
          />
          <div className="found-line"><strong>{history.length}</strong> of {matches.length} matches found</div>
          <section className="relationship-card-grid">
            {history.map((hit) => (
              <HistoryMatchCard
                hitSide={hit.side}
                key={`team-${hit.match.id}-${hit.side}`}
                match={hit.match}
                nameMap={nameMap}
                selectedA={hit.side === "A" ? selectedPlayers : []}
                selectedB={hit.side === "B" ? selectedPlayers : []}
                subtitle={`${hit.selectedCount} matching players`}
              />
            ))}
          </section>
        </>
      )}
    </div>
  );
}

function SharedMatchSummary({
  match,
  nameMap,
  pair,
  title
}: {
  match: Match | null;
  nameMap: Map<string, string>;
  pair: PairStats;
  title: string;
}) {
  if (!match) {
    return (
      <article className="relationship-match-card compact-empty">
        <div className="relationship-match-head">
          <div>
            <span>{title}</span>
            <strong>No match found</strong>
          </div>
        </div>
      </article>
    );
  }

  const teamA = formatTeam(match.team_a, nameMap);
  const teamB = formatTeam(match.team_b, nameMap);
  const score = scoreParts(match.score);
  const highlightKeys = [...pair.aKeys, ...pair.bKeys];

  return (
    <article className="relationship-match-card shared-summary-card">
      <div className="relationship-match-head">
        <div>
          <span>{title}</span>
          <strong>{formatUkDate(match.date)}</strong>
        </div>
        <div>
          <span>Score</span>
          <strong>{score ? `${score[0]}-${score[1]}` : match.score || "-"}</strong>
        </div>
        <div>
          <span>Setup</span>
          <strong>{playersTogetherByKeys(pair.aKeys, pair.bKeys, match) ? "Together" : "Opposite"}</strong>
        </div>
      </div>
      <div className="relationship-match-body">
        <MiniTeamPanel highlightKeys={highlightKeys} label="Team A" team={teamA} tone="a" />
        <MiniTeamPanel highlightKeys={highlightKeys} label="Team B" team={teamB} tone="b" />
      </div>
    </article>
  );
}

function MiniTeamPanel({
  highlightKeys,
  label,
  team,
  tone
}: {
  highlightKeys: string[];
  label: string;
  team: string[];
  tone: "a" | "b";
}) {
  const selectedSet = toNormSet(highlightKeys);
  return (
    <div className={`relationship-team-panel ${tone}`}>
      <h3>{label}</h3>
      <div className="history-pill-row">
        {team.map((player) => (
          <b className={selectedSet.has(normalizeName(player)) ? "in" : ""} key={`${label}-${player}`}>
            {player}
          </b>
        ))}
      </div>
    </div>
  );
}

function MatchupHistoryView({
  groupA,
  groupB,
  history,
  minGroupA,
  minGroupB,
  nameMap,
  playerOptions,
  setMinGroupA,
  setMinGroupB,
  toggleGroupPlayer
}: {
  groupA: string[];
  groupB: string[];
  history: MatchupHit[];
  minGroupA: number;
  minGroupB: number;
  nameMap: Map<string, string>;
  playerOptions: string[];
  setMinGroupA: (value: number) => void;
  setMinGroupB: (value: number) => void;
  toggleGroupPlayer: (group: "A" | "B", player: string) => void;
}) {
  const summary = summarizeMatchupHistory(history);

  return (
    <div className="interactive-stack">
      <section className="relationship-tool-panel matchup">
        <div className="section-title-row">
          <div>
            <span>Matchup History</span>
            <h2>Rivalry history</h2>
          </div>
        </div>
        <div className="relationship-group-grid">
          <div>
            <h3>Group A</h3>
            <ChipPicker
              options={playerOptions}
              selected={groupA}
              onToggle={(player) => toggleGroupPlayer("A", player)}
              tone="a"
            />
            <OverlapRange groupSize={groupA.length} label="Min from Group A on their side" value={minGroupA} onChange={setMinGroupA} />
          </div>
          <div>
            <h3>Group B</h3>
            <ChipPicker
              options={playerOptions}
              selected={groupB}
              onToggle={(player) => toggleGroupPlayer("B", player)}
              tone="b"
            />
            <OverlapRange groupSize={groupB.length} label="Min from Group B on their side" value={minGroupB} onChange={setMinGroupB} />
          </div>
        </div>
      </section>

      {groupA.length === 0 || groupB.length === 0 ? (
        <div className="panel"><p className="muted">Pick at least one player in each group.</p></div>
      ) : (
        <>
          <RelationshipSummary
            items={[
              ["Matches", String(history.length)],
              ["Group A Wins", String(summary.groupAWins)],
              ["Draws", String(summary.draws)],
              ["Group B Wins", String(summary.groupBWins)]
            ]}
          />
          <section className="relationship-card-grid">
            {history.map((hit) => (
              <HistoryMatchCard
                key={`matchup-${hit.match.id}`}
                match={hit.normalizedMatch}
                nameMap={nameMap}
                selectedA={groupA}
                selectedB={groupB}
                subtitle={`${hit.selectedA + hit.selectedB} matching players`}
              />
            ))}
          </section>
        </>
      )}
    </div>
  );
}

function ChipPicker({
  options,
  selected,
  onToggle,
  tone = "neutral"
}: {
  options: string[];
  selected: string[];
  onToggle: (player: string) => void;
  tone?: "neutral" | "a" | "b";
}) {
  return (
    <div className={`relationship-chip-grid ${tone}`}>
      {options.map((option) => (
        <button
          className={selected.includes(option) ? "selected" : ""}
          key={option}
          onClick={() => onToggle(option)}
          type="button"
        >
          {option}
        </button>
      ))}
    </div>
  );
}

function OverlapRange({
  groupSize,
  label,
  value,
  onChange
}: {
  groupSize: number;
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  const max = Math.min(5, Math.max(groupSize, 1));
  if (groupSize <= 1) {
    return <p className="muted">{label}: 1</p>;
  }

  return (
    <div className="control-bar compact relationship-range">
      <label>
        <span>{label}</span>
        <input
          min={1}
          max={max}
          onChange={(event) => onChange(Number(event.target.value))}
          type="range"
          value={Math.min(value, max)}
        />
      </label>
      <div className="range-value">{Math.min(value, max)}+</div>
    </div>
  );
}

function RelationshipSummary({ items }: { items: Array<[string, string]> }) {
  return (
    <section className="relationship-summary-grid">
      {items.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </section>
  );
}

function HistoryMatchCard({
  hitSide,
  match,
  nameMap,
  selectedA,
  selectedB,
  subtitle
}: {
  hitSide?: "A" | "B";
  match: Match;
  nameMap: Map<string, string>;
  selectedA: string[];
  selectedB: string[];
  subtitle: string;
}) {
  const teamA = formatTeam(match.team_a, nameMap);
  const teamB = formatTeam(match.team_b, nameMap);
  const score = scoreParts(match.score);
  return (
    <article className="relationship-match-card">
      <div className="relationship-match-head">
        <div>
          <span>Previous meeting</span>
          <strong>{formatUkDate(match.date)}</strong>
        </div>
        <div>
          <span>Score</span>
          <strong>{score ? `${score[0]}-${score[1]}` : match.score || "-"}</strong>
        </div>
        <div>
          <span>Same players</span>
          <strong>{subtitle.split(" ")[0]}</strong>
        </div>
      </div>
      <div className="relationship-match-body">
        <HistoryTeamPanel
          label={hitSide === "A" ? "Team A overlap" : "Team A"}
          selectedPlayers={selectedA}
          team={teamA}
          tone="a"
        />
        <HistoryTeamPanel
          label={hitSide === "B" ? "Team B overlap" : "Team B"}
          selectedPlayers={selectedB}
          team={teamB}
          tone="b"
        />
      </div>
    </article>
  );
}

function HistoryTeamPanel({
  label,
  selectedPlayers,
  team,
  tone
}: {
  label: string;
  selectedPlayers: string[];
  team: string[];
  tone: "a" | "b";
}) {
  const selectedSet = toNormSet(selectedPlayers);
  const teamSet = toNormSet(team);
  const selectedInGame = selectedPlayers.filter((player) => teamSet.has(normalizeName(player)));
  const selectedOutOfGame = selectedPlayers.filter((player) => !teamSet.has(normalizeName(player)));
  const selectedOnTeam = team.filter((player) => selectedSet.has(normalizeName(player)));
  const others = team.filter((player) => !selectedSet.has(normalizeName(player)));

  return (
    <div className={`relationship-team-panel ${tone}`}>
      <h3>{label}</h3>
      <span>Full previous team</span>
      <div className="history-pill-row">
        {selectedOnTeam.map((player) => <b className="in" key={`in-${tone}-${player}`}>{player}</b>)}
        {others.map((player) => <b key={`other-${tone}-${player}`}>{player}</b>)}
      </div>
      {selectedPlayers.length ? (
        <>
          <span>Changes</span>
          <div className="history-change-row">
            <small>In</small>
            <div className="history-pill-row">
              {selectedInGame.length ? selectedInGame.map((player) => (
                <b className="in" key={`selected-${tone}-${player}`}>{player}</b>
              )) : <small>None</small>}
            </div>
          </div>
          <div className="history-change-row">
            <small>Out</small>
            <div className="history-pill-row">
              {selectedOutOfGame.length ? selectedOutOfGame.map((player) => (
                <b className="out" key={`out-${tone}-${player}`}>{player}</b>
              )) : <small>No selected players missing</small>}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}

function buildTeamHistory(selectedPlayers: string[], minTogether: number, matches: Match[], nameMap: Map<string, string>): TeamHistoryHit[] {
  if (selectedPlayers.length < 2) return [];
  const selectedSet = toNormSet(selectedPlayers);
  const needed = selectedPlayers.length === 2 ? 2 : Math.min(minTogether, selectedPlayers.length, 5);

  return [...matches]
    .sort((a, b) => String(b.date || "").localeCompare(String(a.date || "")))
    .flatMap((match) => {
      const score = scoreParts(match.score);
      if (!score) return [];
      const teamA = formatTeam(match.team_a, nameMap);
      const teamB = formatTeam(match.team_b, nameMap);
      const aCount = countSelected(teamA, selectedSet);
      const bCount = countSelected(teamB, selectedSet);
      if (Math.max(aCount, bCount) < needed) return [];
      const side = aCount >= bCount ? "A" : "B";
      return [{
        match,
        side,
        selectedCount: side === "A" ? aCount : bCount,
        outcome: resultFor(match, side),
        goalGap: Math.abs(score[0] - score[1])
      }];
    });
}

function buildMatchupHistory(
  groupA: string[],
  groupB: string[],
  minA: number,
  minB: number,
  matches: Match[],
  nameMap: Map<string, string>
): MatchupHit[] {
  if (!groupA.length || !groupB.length) return [];
  const groupASet = toNormSet(groupA);
  const groupBSet = toNormSet(groupB);
  const neededA = Math.min(Math.max(minA, 1), Math.min(5, groupA.length));
  const neededB = Math.min(Math.max(minB, 1), Math.min(5, groupB.length));

  return [...matches]
    .sort((a, b) => String(b.date || "").localeCompare(String(a.date || "")))
    .flatMap((match) => {
      const score = scoreParts(match.score);
      if (!score) return [];
      const teamA = formatTeam(match.team_a, nameMap);
      const teamB = formatTeam(match.team_b, nameMap);
      const aInA = countSelected(teamA, groupASet);
      const aInB = countSelected(teamB, groupASet);
      const bInA = countSelected(teamA, groupBSet);
      const bInB = countSelected(teamB, groupBSet);
      const okAB = aInA >= neededA && bInB >= neededB;
      const okBA = aInB >= neededA && bInA >= neededB;
      if (!okAB && !okBA) return [];

      const groupASide = okAB ? "A" : "B";
      const normalizedMatch = groupASide === "A"
        ? match
        : {
            ...match,
            team_a: match.team_b,
            team_b: match.team_a,
            score: `${score[1]}-${score[0]}`
          };
      return [{
        match,
        normalizedMatch,
        selectedA: groupASide === "A" ? aInA : aInB,
        selectedB: groupASide === "A" ? bInB : bInA,
        groupAResult: resultFor(normalizedMatch, "A")
      }];
    });
}

function summarizeTeamHistory(history: TeamHistoryHit[]) {
  const wins = history.filter((hit) => hit.outcome === "W").length;
  const draws = history.filter((hit) => hit.outcome === "D").length;
  const losses = history.filter((hit) => hit.outcome === "L").length;
  const avg = history.reduce((sum, hit) => sum + hit.goalGap, 0) / Math.max(history.length, 1);
  return { wins, draws, losses, avgGap: history.length ? avg.toFixed(2) : "-" };
}

function summarizeMatchupHistory(history: MatchupHit[]) {
  return {
    groupAWins: history.filter((hit) => hit.groupAResult === "W").length,
    draws: history.filter((hit) => hit.groupAResult === "D").length,
    groupBWins: history.filter((hit) => hit.groupAResult === "L").length
  };
}

function buildPairStats(playerA: Player, playerB: Player, matches: Match[]): PairStats {
  const a = displayName(playerA);
  const b = displayName(playerB);
  const aKeys = keysFor(playerA);
  const bKeys = keysFor(playerB);
  const playerPointRates = buildPlayerPointRates(matches);
  const stats: PairStats = {
    a,
    b,
    aKeys,
    bKeys,
    together: 0,
    togetherWins: 0,
    togetherDraws: 0,
    togetherLosses: 0,
    togetherGoalDiff: 0,
    togetherScoreSum: 0,
    chemistryActualRate: 0.5,
    chemistryExpectedRate: 0.5,
    chemistryResidual: 0,
    faced: 0,
    aWins: 0,
    bWins: 0,
    facedDraws: 0,
    totalGoalGap: 0,
    chemistryScore: 0,
    rivalryScore: 0,
    latestMatch: null,
    latestTogetherMatch: null,
    latestFacedMatch: null
  };

  for (const match of matches) {
    const score = scoreParts(match.score);
    if (!score) continue;
    const sideA = sideFor(aKeys, match);
    const sideB = sideFor(bKeys, match);
    if (!sideA || !sideB) continue;

    if (!stats.latestMatch || String(match.date || "") > String(stats.latestMatch.date || "")) {
      stats.latestMatch = match;
    }

    if (sideA === sideB) {
      const result = resultFor(match, sideA);
      const goalDiff = sideA === "A" ? score[0] - score[1] : score[1] - score[0];
      stats.together += 1;
      stats.togetherWins += result === "W" ? 1 : 0;
      stats.togetherDraws += result === "D" ? 1 : 0;
      stats.togetherLosses += result === "L" ? 1 : 0;
      stats.togetherScoreSum += result === "W" ? 1 : result === "D" ? 0.5 : 0;
      stats.togetherGoalDiff += goalDiff;
      if (!stats.latestTogetherMatch || String(match.date || "") > String(stats.latestTogetherMatch.date || "")) {
        stats.latestTogetherMatch = match;
      }
    } else {
      const result = resultFor(match, sideA);
      stats.faced += 1;
      stats.aWins += result === "W" ? 1 : 0;
      stats.bWins += result === "L" ? 1 : 0;
      stats.facedDraws += result === "D" ? 1 : 0;
      stats.totalGoalGap += Math.abs(score[0] - score[1]);
      if (!stats.latestFacedMatch || String(match.date || "") > String(stats.latestFacedMatch.date || "")) {
        stats.latestFacedMatch = match;
      }
    }
  }

  stats.chemistryActualRate = stats.togetherScoreSum / Math.max(stats.together, 1);
  stats.chemistryExpectedRate = (pointRateForKeys(aKeys, playerPointRates) + pointRateForKeys(bKeys, playerPointRates)) / 2;
  stats.chemistryResidual = stats.chemistryActualRate - stats.chemistryExpectedRate;
  stats.chemistryScore = chemistryScore(stats);
  stats.rivalryScore = rivalryScore(stats);
  return stats;
}

function keysFor(player: Player) {
  return [player.name, player.display_name, displayName(player)].map(normalizeName).filter(Boolean);
}

function sideFor(keys: string[], match: Match): "A" | "B" | null {
  const teamA = splitTeam(match.team_a).map(normalizeName);
  const teamB = splitTeam(match.team_b).map(normalizeName);
  if (keys.some((key) => teamA.includes(key))) return "A";
  if (keys.some((key) => teamB.includes(key))) return "B";
  return null;
}

function sharedMatchLabel(pair: PairStats, match: Match) {
  return playersTogetherByKeys(pair.aKeys, pair.bKeys, match)
    ? `${pair.a} and ${pair.b} were teammates`
    : `${pair.a} and ${pair.b} faced each other`;
}

function playersTogetherByKeys(aKeys: string[], bKeys: string[], match: Match) {
  const teamA = splitTeam(match.team_a).map(normalizeName);
  const teamB = splitTeam(match.team_b).map(normalizeName);
  const aInA = aKeys.some((key) => teamA.includes(normalizeName(key)));
  const bInA = bKeys.some((key) => teamA.includes(normalizeName(key)));
  const aInB = aKeys.some((key) => teamB.includes(normalizeName(key)));
  const bInB = bKeys.some((key) => teamB.includes(normalizeName(key)));
  return (aInA && bInA) || (aInB && bInB);
}

function toNormSet(values: string[]) {
  return new Set(values.map(normalizeName).filter(Boolean));
}

function countSelected(team: string[], selectedSet: Set<string>) {
  return team.filter((player) => selectedSet.has(normalizeName(player))).length;
}

function chemistryScore(row: PairStats) {
  return chemistryScoreFor({
    matches: row.together,
    wins: row.togetherWins,
    draws: row.togetherDraws,
    losses: row.togetherLosses,
    goalDiff: row.togetherGoalDiff,
    scoreSum: row.togetherScoreSum,
    actualRate: row.chemistryActualRate,
    expectedRate: row.chemistryExpectedRate
  });
}

function rivalryScore(row: PairStats) {
  return rivalryScoreFor({
    matches: row.faced,
    winsA: row.aWins,
    winsB: row.bWins,
    draws: row.facedDraws,
    totalGoalGap: row.totalGoalGap
  });
}

function avgGap(row: PairStats) {
  return (row.totalGoalGap / Math.max(row.faced, 1)).toFixed(1);
}

function pct(wins: number, matches: number) {
  if (!matches) return "0%";
  return `${Math.round((wins / matches) * 100)}%`;
}

function signed(value: number) {
  return value > 0 ? `+${value}` : String(value);
}

function chemistryExpectation(row: PairStats) {
  if (row.together < 4) return "Building";
  const residual = Math.round(row.chemistryResidual * 100);
  return residual >= 0 ? `Lift +${residual}` : `Drop ${residual}`;
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

function pointRateForKeys(keys: string[], rates: Map<string, number>) {
  for (const key of keys) {
    const value = rates.get(normalizeName(key));
    if (typeof value === "number") return value;
  }
  return 0.5;
}
