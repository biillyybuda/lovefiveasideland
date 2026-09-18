"use client";

import { useEffect, useMemo, useState } from "react";
import { getBrowserSupabase } from "@/lib/auth-client";
import { splitTeam, type Match, type Player } from "@/lib/demo-data";
import { activityDraft, activityPlayers, baselineActivityFields, matchActivityFields, watchActivityFields, isPace, isDuration, formatActivityValue, loadMatchActivity, parseActivity, savePlayerActivity,
  type ActivityField, type ActivityPlayer, type ActivityValues, type MatchActivity } from "@/lib/match-activity";

export function MatchActivityForm({ player, original, onSave, onCancel }: {
  player: ActivityPlayer; original?: MatchActivity; onSave: (values: ActivityValues | null) => Promise<void>; onCancel: () => void;
}) {
  const [draft, setDraft] = useState(() => activityDraft(original));
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  async function save(remove = false) {
    setMessage(""); setBusy(true);
    try { await onSave(remove ? null : parseActivity(draft)); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Could not save activity."); }
    finally { setBusy(false); }
  }
  const fields = (items: readonly ActivityField[]) => <div className="activity-fields">{items.map(field => {
    const timed = isPace(field.key) || isDuration(field.key);
    return <label key={field.key}><span>{field.label}</span><input type={timed ? "text" : "number"} inputMode={timed ? "text" : field.step === "1" ? "numeric" : "decimal"}
      min={timed ? undefined : field.min} max={timed ? undefined : field.max} step={timed ? undefined : field.step} placeholder={field.placeholder}
      value={draft[field.key]} onChange={event => setDraft(previous => ({ ...previous, [field.key]: event.target.value }))}/></label>;
  })}</div>;
  return <form className="activity-form" onSubmit={event => { event.preventDefault(); void save(); }}>
    <h3>{original ? "Edit" : "Add"} activity · {player.name}</h3>
    <p className="review-footnote">All figures are optional. Leave anything you do not know blank.</p>
    <fieldset disabled={busy}>
      <h4>Match readings</h4>
      <p className="review-footnote">Enter the match portion of the recording.</p>
      {fields(matchActivityFields)}
      <details className="activity-watch-fields"><summary>More watch readings · pace, speed, timing &amp; breathing</summary>
        <p className="review-footnote">Copy the figures shown by the watch. Times accept m:ss or h:mm:ss; pace accepts m:ss per km. Total time is the activity timer; moving time and elapsed time are separate. Leave distance blank if it is not shown.</p>
        {fields(watchActivityFields)}
        <p className="review-footnote">Calories are a watch estimate. For a shared watch, use the wearer's profile where possible; a different profile can affect calorie estimates. Pace and speed are stored as shown, without filling one from the other.</p>
      </details>
      <h4>Heart-rate baselines</h4>
      <p className="review-footnote">Use readings known around the date of this game. Resting heart rate is measured at rest; personal maximum is your known overall maximum. It may be higher than this match's peak. These readings are saved with this match.</p>
      {fields(baselineActivityFields)}
      <label className="activity-source"><span>Watch, app or source (optional)</span><input maxLength={80} value={draft.source} placeholder="e.g. Apple Watch, Garmin or Strava" onChange={event => setDraft(previous => ({ ...previous, source: event.target.value }))}/></label>
      <div className="review-nav"><button className="button primary" type="submit">{busy ? "Saving…" : "Save activity"}</button>
        <button className="mini-link" type="button" onClick={onCancel}>Cancel</button>
        {original && <button className="mini-link activity-remove" type="button" onClick={() => void save(true)}>Remove activity</button>}</div>
    </fieldset>
    {message && <p role="alert" className="activity-message">{message}</p>}
  </form>;
}

export function MatchActivityPanel({ leagueId, match, players, canEdit }: {
  leagueId: number; match: Match; players: Player[]; canEdit: boolean;
}) {
  const participants = useMemo(() => activityPlayers(match, players), [match, players]);
  const totalPlayers = new Set([...splitTeam(match.team_a), ...splitTeam(match.team_b)].map(n => n.toLowerCase())).size;
  const [rows, setRows] = useState<MatchActivity[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [notice, setNotice] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [editing, setEditing] = useState<{ player: ActivityPlayer; original?: MatchActivity } | null>(null);
  const [reload, setReload] = useState(0);
  useEffect(() => {
    let cancelled = false;
    setLoading(true); setLoadError("");
    loadMatchActivity(getBrowserSupabase(), leagueId, match.id).then(data => { if (!cancelled) setRows(data); })
      .catch(error => { if (!cancelled) setLoadError(error.message || "Could not load saved activity."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [leagueId, match.id, reload]);
  const recorded = rows.filter(row => participants.some(p => p.id === row.player_id));
  const openEditor = (player: ActivityPlayer) => { setSelectedId(String(player.id)); setNotice(""); setEditing({ player, original: rows.find(row => row.player_id === player.id) }); };
  return <section id="match-activity" className="panel match-activity" aria-labelledby="match-activity-title">
    <h2 id="match-activity-title">Player activity</h2>
    <p className="review-footnote">Match activity and heart-rate baselines from any watch or app. Visible to league members. We are collecting readings; they do not affect team selection, ratings or awards.</p>
    {loading ? <p role="status">Loading saved activity…</p> : loadError ? <p role="alert">{loadError}</p> : <>
      <p className="activity-coverage">Activity recorded for <strong>{recorded.length} of {totalPlayers}</strong> players</p>
      {!recorded.length && <p className="muted">No activity has been added for this match yet.</p>}
      <div className="activity-records">{recorded.map(row => {
        const player = participants.find(p => p.id === row.player_id)!;
        return <div className="activity-record" key={row.player_id}>
          <div className="activity-record-head"><strong>{player.name} <small>· Team {player.side}</small></strong>
            {canEdit && !editing && <button className="mini-link" type="button" onClick={() => openEditor(player)}>Edit {player.name}</button>}</div>
          <dl>{matchActivityFields.map(field => <div key={field.key}><dt>{field.label.replace("minutes or m:ss", "m:ss")}</dt><dd>{row[field.key] == null ? "Not recorded" : formatActivityValue(field.key, row[field.key])}</dd></div>)}</dl>
          <p className="review-footnote">Heart-rate baselines for this match</p>
          <dl>{baselineActivityFields.map(field => <div key={field.key}><dt>{field.label}</dt><dd>{row[field.key] == null ? "Not recorded" : formatActivityValue(field.key, row[field.key])}</dd></div>)}</dl>
          {watchActivityFields.some(field => row[field.key] != null) && <details className="activity-watch-fields"><summary>Watch details</summary><dl>{watchActivityFields.filter(field => row[field.key] != null).map(field => <div key={field.key}><dt>{field.label.replace("minutes or m:ss", "m:ss")}</dt><dd>{formatActivityValue(field.key, row[field.key])}</dd></div>)}</dl></details>}
          {row.source && <p className="review-footnote">Source: {row.source}</p>}
        </div>;
      })}</div>
    </>}
    {canEdit && !loading && !loadError && !editing && <div className="activity-picker">
      <label><span>Choose a player from this match</span><select value={selectedId} onChange={event => setSelectedId(event.target.value)}>
        <option value="">Select player</option>{participants.map(player => <option key={player.id} value={player.id}>{player.name} · Team {player.side}{rows.some(r => r.player_id === player.id) ? " · Activity saved" : ""}</option>)}</select></label>
      <button className="button" type="button" disabled={!selectedId} onClick={() => { const player = participants.find(p => String(p.id) === selectedId); if (player) openEditor(player); }}>
        {rows.some(row => String(row.player_id) === selectedId) ? "Edit activity" : "Add activity"}</button>
    </div>}
    {canEdit && editing && <MatchActivityForm key={`${editing.player.id}:${editing.original?.version || "new"}`} player={editing.player} original={editing.original}
      onCancel={() => setEditing(null)} onSave={async values => {
        const saved = await savePlayerActivity(getBrowserSupabase(), leagueId, match.id, editing.player.id, editing.original, values);
        setRows(previous => [...previous.filter(row => row.player_id !== editing.player.id), ...(saved ? [saved] : [])]);
        setNotice(`${editing.player.name}'s activity ${saved ? "saved" : "removed"}.`); setEditing(null);
      }}/>}
    {!editing && <button className="mini-link activity-reload" type="button" disabled={loading} onClick={() => { setNotice(""); setReload(value => value + 1); }}>Reload saved activity</button>}
    {notice && <p role="status" className="activity-message">{notice}</p>}
  </section>;
}
