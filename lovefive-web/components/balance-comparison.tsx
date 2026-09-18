"use client";

import { useMemo, useState } from "react";
import { getBrowserSupabase } from "@/lib/auth-client";
import { displayName, type Player } from "@/lib/demo-data";
import { predictBalance, BALANCE_CONFIG, BALANCE_VERSION } from "@/lib/balance-model";
import { buildMatchdayBalance, matchdayOptions } from "@/lib/matchday-balance";
import { loadBalancePreview, type BalancePreviewWorkspace } from "@/lib/balance-preview";
import type { TeamOption } from "@/lib/team-ai-engine";

export function BalanceComparison({ leagueId, selected, locksA, locksB, current }: {
  leagueId: number; selected: Player[]; locksA: number[]; locksB: number[]; current?: TeamOption;
}) {
  const [workspace, setWorkspace] = useState<BalancePreviewWorkspace | null>(null);
  const [date, setDate] = useState(() => {
    const today = new Date();
    return `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
  });
  const [chosenKey, setChosenKey] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const preview = useMemo(() => {
    if (!workspace) return null;
    try {
      const model = buildMatchdayBalance(workspace.players, workspace.matches, date);
      const { state } = model;
      const options = matchdayOptions(model, selected, locksA, locksB);
      const prior = current ? predictBalance(state, current.teamA.map(p => p.id), current.teamB.map(p => p.id)) : null;
      return { state, options, prior, ratings: model.ratings, error: "" };
    } catch (error) { return { error: error instanceof Error ? error.message : "Could not calculate comparison." }; }
  }, [workspace, selected, locksA, locksB, current, date]);
  const options = preview && "options" in preview ? preview.options : [];
  const chosen = options?.find(o => o.key === chosenKey) || options?.[0];
  async function load() {
    setBusy(true); setMessage("");
    try { setWorkspace(await loadBalancePreview(getBrowserSupabase(), leagueId)); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Could not load comparison."); }
    finally { setBusy(false); }
  }
  async function record() {
    if (!workspace?.token || !preview || !("state" in preview) || !chosen) return;
    setBusy(true); setMessage("");
    try {
      const { error } = await getBrowserSupabase().rpc("save_team_selection", {
        p_league_id: leagueId, p_expected_token: workspace.token, p_id: crypto.randomUUID(), p_match_date: date,
        p_snapshot: { version: BALANCE_VERSION, config: BALANCE_CONFIG, state: preview.state,
          selected: selected.map(p => ({ id: p.id, name: p.name, fitness: p.fitness, position: p.position || null })),
          locksA, locksB, chosen, candidates: options, legacy: current || null }
      });
      if (error) throw new Error(error.message);
      setMessage("Pre-match comparison recorded. Enter the final teams actually played when saving the result.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Could not record comparison."); }
    finally { setBusy(false); }
  }
  const names = (team: Player[]) => team.map(displayName).join(", ");
  return <details className="panel">
    <summary>Admin forecast log</summary>
    <p className="muted">Load a fresh forecast to record ahead of a game. Choose the teams and date to log below. These estimated probabilities can later be checked against actual results.</p>
    <div className="control-bar">
      <label><span>Match date</span><input type="date" value={date} onChange={event => setDate(event.target.value)} /></label>
      <button className="button" type="button" disabled={busy} onClick={load}>{workspace ? "Refresh comparison" : "Load comparison"}</button>
    </div>
    {preview?.error ? <p role="alert">{preview.error}</p> : null}
    {chosen && preview && "ratings" in preview ? <>
      <label><span>Proposed teams to record</span><select value={chosen.key} onChange={event => setChosenKey(event.target.value)}>
        {options?.slice(0, 12).map((o, i) => <option value={o.key} key={o.key}>Option {i + 1} · {(100 * o.prediction.close).toFixed(1)}% estimated close-game chance</option>)}
      </select></label>
      <div className="team-picker-grid"><div><h3>Team A</h3><p>{names(chosen.teamA)}</p></div><div><h3>Team B</h3><p>{names(chosen.teamB)}</p></div></div>
      <p>Expected advantage: {Math.abs(chosen.prediction.meanMargin).toFixed(1)} goals to Team {chosen.prediction.meanMargin >= 0 ? "A" : "B"}. Estimated close-game chance: {(100 * chosen.prediction.close).toFixed(1)}%.</p>
      {preview.prior ? <p className="muted">Using the same model, the legacy selector’s top choice has an expected gap of {Math.abs(preview.prior.meanMargin).toFixed(1)} goals and a {(100 * preview.prior.close).toFixed(1)}% close-game chance. These compare predictions, not results of games played.</p> : null}
      <p className="muted">Known positions and recent team variety help choose between options within one percentage point of predicted closeness. Current fitness remains visible for the group’s judgement; its effect is not yet fitted.</p>
      {workspace?.token ? <button className="button primary" type="button" disabled={busy} onClick={record}>Record pre-match comparison</button>
        : <p className="muted">Forecast recording is not enabled for this league yet. The Team Generator is available without it.</p>}
      <details><summary>Compare player ratings</summary><p className="muted">The strength scale is centred on 1,000. Uncertainty describes how much evidence we have; it is not a penalty or a change to public MMR.</p>
        <div style={{ overflowX: "auto" }}><table><thead><tr><th>Player</th><th>Public MMR</th><th>Strength estimate</th><th>Uncertainty</th></tr></thead><tbody>
          {preview.ratings?.slice().sort((a, b) => b.rating - a.rating).map(row => {
            const p = workspace?.players.find(p => p.id === row.playerId);
            if (!p || p.is_active === 0) return null;
            return <tr key={row.playerId}><td>{displayName(p)}</td><td>{Math.round(p.mmr ?? 1000)}</td><td>{Math.round(row.rating)}</td><td>±{Math.round(row.uncertainty)}</td></tr>;
          })}
        </tbody></table></div></details>
    </> : workspace && !preview?.error ? <p>Select ten players to compare teams.</p> : null}
    {message ? <p role="status">{message}</p> : null}
  </details>;
}
