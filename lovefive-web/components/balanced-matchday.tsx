"use client";

import { useMemo, useState } from "react";
import { displayName, formatTeam, formatUkDate, makeNameMap, normalizeName, splitTeam, type Match, type Player } from "@/lib/demo-data";
import { buildMatchdayBalance, matchdayOptions } from "@/lib/matchday-balance";
import { BalanceMatchdayCard } from "@/components/balance-matchday-card";
import { matchdayOutlook } from "@/lib/matchday-outlook";
import type { LineupSeed } from "@/lib/lineup-editor";

function today() {
  const date = new Date();
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export function BalancedMatchday({ players, selected, matches, locksA, locksB, onLock, onClearLocks, onReuseMemory, onEditLineup }: {
  players: Player[]; selected: Player[]; matches: Match[]; locksA: number[]; locksB: number[];
  onLock: (player: Player, side: "A" | "B") => void;
  onClearLocks: () => void;
  onReuseMemory: (match: Match) => void;
  onEditLineup?: (seed: LineupSeed) => void;
}) {
  const [date, setDate] = useState(today);
  const [chosenKey, setChosenKey] = useState("");
  const [showAll, setShowAll] = useState(false);
  const [copyMessage, setCopyMessage] = useState("");
  const calculation = useMemo(() => {
    try { return { model: buildMatchdayBalance(players, matches, date), error: "" }; }
    catch (error) { return { model: null, error: error instanceof Error ? error.message : "Could not calculate teams." }; }
  }, [players, matches, date]);
  const model = calculation.model;
  const options = useMemo(() => model ? matchdayOptions(model, selected, locksA, locksB) : [], [model, selected, locksA, locksB]);
  const chosen = options.find(option => option.key === chosenKey) || options[0];
  const ratings = new Map(model?.ratings.map(row => [row.playerId, row]));
  const names = makeNameMap(players);
  const playerNames = new Set(selected.flatMap(player => [normalizeName(player.name), normalizeName(player.display_name)]).filter(Boolean));
  const memories = (model?.history || []).map(match => ({ match,
    overlap: [...splitTeam(match.team_a), ...splitTeam(match.team_b)].filter(name => playerNames.has(normalizeName(name))).length
  })).filter(row => row.overlap >= 4).sort((a, b) => b.overlap - a.overlap || String(b.match.date).localeCompare(String(a.match.date)) || b.match.id - a.match.id).slice(0, 6);
  const selectionKey = `${date}:${chosen?.teamA.map(player => player.id).join(",") || ""}:${chosen?.teamB.map(player => player.id).join(",") || ""}`;

  async function copyTeams() {
    if (!chosen) return;
    try {
      await navigator.clipboard.writeText(`Love Five · ${formatUkDate(date)}\n\n🔵 Blue team (Team A): ${chosen.teamA.map(displayName).join(", ")}\n\n🔴 Red team (Team B): ${chosen.teamB.map(displayName).join(", ")}\n\n${matchdayOutlook(chosen.prediction)}\nEstimated results: Blue win ${percent(chosen.prediction.win)} · Draw ${percent(chosen.prediction.draw)} · Red win ${percent(chosen.prediction.loss)}\nFinish within 2 goals: ${percent(chosen.prediction.close)} (draw, or either team wins by 1–2).`);
      setCopyMessage(selectionKey);
    } catch { setCopyMessage("Copy unavailable. You can select and copy the team names below."); }
  }

  return <div className="interactive-stack balance-default">
    <section className="panel">
      <div className="panel-title-row"><div><span className="stat-label">Team balancing</span><h2>Strength + uncertainty</h2></div>
        <label><span>Match date</span><input type="date" value={date} onChange={event => { setDate(event.target.value); setChosenKey(""); }} /></label>
      </div>
      <p className="muted">Teams are picked using what past scorelines tell us about each player, including how certain those estimates are. Public MMR and season award scores keep their existing rules.</p>
      {model ? <small className="muted">Based on {model.history.length} completed games on or before {formatUkDate(date)}. A close game means a draw or a win by one or two goals.</small> : null}
      {calculation.error ? <p role="alert">{calculation.error} Check the date or ask a league admin to correct the match history.</p> : null}
    </section>

    {selected.length !== 10 ? <section className="panel empty-state"><h2>{selected.length < 10 ? `Pick ${10 - selected.length} more players` : `Remove ${selected.length - 10} players`}</h2><p className="muted">Choose exactly ten players to see teams and their estimated chances.</p></section> : model ? <>
      <section className="matchday-generator-grid">
        <div className="panel recommendation-panel" id="balance-recommendations">
          <div className="panel-title-row"><div><span className="stat-label">{options.length} possible splits</span><h2>Recommended Teams</h2></div>
            <button className="ghost-button" type="button" onClick={() => setShowAll(value => !value)}>{showAll ? "Show shortlist" : `View all ${options.length}`}</button>
          </div>
          <p className="muted">Ranked by estimated close-game chance. Within each one-percentage-point band, known positions and recent team variety decide the order.</p>
          {!options.length ? <p role="alert">These locks cannot make two teams of five. Release a lock to continue.</p> : null}
          <div className="recommendation-list">
            {(showAll ? options : options.slice(0, 4)).map((option, index) => <button key={option.key} type="button" aria-pressed={chosen?.key === option.key}
              className={`recommendation-card${chosen?.key === option.key ? " selected" : ""}`} onClick={() => setChosenKey(option.key)}>
              <div className="recommendation-head"><span>{index === 0 ? "Recommended" : `Alternative ${index}`}</span><strong>{percent(option.prediction.close)} chance of finishing within 2 goals</strong></div>
              <div className="lineups">{(["A", "B"] as const).map(side => <div className={`team-panel ${side.toLowerCase()}`} key={side}>
                <div className="stat-label">{side === "A" ? "Blue" : "Red"} team · Team {side}</div><div className="pill-row">{option[side === "A" ? "teamA" : "teamB"].map(player => <span className="pill" key={player.id}>{displayName(player)}</span>)}</div>
              </div>)}</div>
              <div className="engine-breakdown"><span>{matchdayOutlook(option.prediction)}</span><span>Uncertainty included</span></div>
            </button>)}
          </div>
        </div>
        <aside className="panel locks-panel"><div className="panel-title-row"><h2>Side locks</h2><button className="ghost-button" type="button" onClick={onClearLocks}>Clear locks</button></div>
          <p className="muted">Keep players on a side, or lock your own teams to see their forecast. A is blue; B is red.</p>
          <div className="lock-list">{selected.map(player => <div key={player.id}><strong>{displayName(player)}</strong>
            {(["A", "B"] as const).map(side => <button key={side} type="button" aria-label={`Lock ${displayName(player)} to Team ${side}`} aria-pressed={(side === "A" ? locksA : locksB).includes(player.id)}
              className={(side === "A" ? locksA : locksB).includes(player.id) ? "active" : ""} onClick={() => onLock(player, side)}>{side}</button>)}
          </div>)}</div>
        </aside>
      </section>

      {chosen ? <BalanceMatchdayCard option={chosen} model={model} players={players} date={date}
        rank={options.indexOf(chosen) + 1} onCopy={copyTeams}
        onEditLineup={() => onEditLineup?.({ date, blue: chosen.teamA.map(p => p.id), red: chosen.teamB.map(p => p.id), revision: Date.now() })}
        copyStatus={copyMessage === selectionKey ? "Teams copied." : copyMessage.startsWith("Copy unavailable") ? copyMessage : ""} /> : null}

      <details className="panel"><summary>Matchday Memory</summary><p className="muted">Reuse an old game&apos;s shape. Returning players keep their old side, replacing your current locks; the strength and uncertainty model fills the spaces.</p>
        {memories.length ? <div className="balance-memory-list">{memories.map(({ match, overlap }) => <button type="button" key={match.id} className="recommendation-card" onClick={() => { onReuseMemory(match); setChosenKey(""); }}>
          <strong>{formatUkDate(match.date)} · {match.score} · {overlap} returning players</strong>
          <span>Old Team A: {formatTeam(match.team_a, names).join(", ")}</span><span>Old Team B: {formatTeam(match.team_b, names).join(", ")}</span><small>Reuse these sides</small>
        </button>)}</div> : <p className="muted">No previous game has at least four of these players.</p>}
      </details>
    </> : null}

    {model ? <details className="panel" open><summary>Player strength & uncertainty</summary>
      <p className="muted">Strength is a separate estimate centred on 1,000. The ± figure is one standard deviation: larger means less certainty. Evidence usually narrows it; time between games adds uncertainty without lowering strength. New players begin at the model&apos;s neutral estimate, even if an admin has set a different public MMR.</p>
      <div className="balance-table-scroll"><table><thead><tr><th>Player</th><th>Public MMR</th><th>Strength</th><th>Uncertainty</th><th>Games</th></tr></thead><tbody>
        {(selected.length ? selected : players.filter(player => player.is_active !== 0)).slice().sort((a, b) => (ratings.get(b.id)?.rating || 1000) - (ratings.get(a.id)?.rating || 1000)).map(player => {
          const row = ratings.get(player.id);
          return row ? <tr key={player.id}><td>{displayName(player)}</td><td>{Math.round(player.mmr ?? 1000)}</td><td>{Math.round(row.rating)}</td><td>±{Math.round(row.uncertainty)}</td><td>{row.appearances}</td></tr> : null;
        })}
      </tbody></table></div>
      <p className="muted">Fitness, individual goals, assists, saves and chemistry tables remain useful context for the group. They do not add separate bonuses to this selector.</p>
    </details> : null}
  </div>;
}
