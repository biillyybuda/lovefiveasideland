"use client";

import { useMemo, useState } from "react";
import { InteractivePlayers } from "@/components/interactive-players";
import { getBrowserSupabase } from "@/lib/auth-client";
import { buildPlayerSummaries, type Match, type Player } from "@/lib/demo-data";
import { canonicalName, displayPlayerName, tidyName } from "@/lib/live-data";

export function LivePlayerManagement({
  leagueId,
  players,
  matches,
  refresh
}: {
  leagueId: number;
  players: Player[];
  matches: Match[];
  refresh: () => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [mmr, setMmr] = useState(1000);
  const [fitness, setFitness] = useState("Medium");
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editDisplayName, setEditDisplayName] = useState("");
  const [editMmr, setEditMmr] = useState(1000);
  const [editFitness, setEditFitness] = useState("Medium");
  const [message, setMessage] = useState("");
  const summaries = useMemo(() => buildPlayerSummaries(players, matches), [matches, players]);
  const selected = players.find((player) => player.id === editingId) || null;

  function startEdit(player: Player) {
    setEditingId(player.id);
    setEditDisplayName(displayPlayerName(player));
    setEditMmr(Math.round(Number(player.mmr || 1000)));
    setEditFitness(player.fitness || "Medium");
  }

  async function addPlayer(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage("");
    const cleanName = canonicalName(name);
    if (!cleanName) {
      setMessage("Enter a player name.");
      return;
    }

    const supabase = getBrowserSupabase();
    const { error } = await supabase.from("players").insert({
      league_id: leagueId,
      name: cleanName,
      display_name: displayName.trim() || tidyName(cleanName),
      mmr,
      matches_played: 0,
      wins: 0,
      draws: 0,
      losses: 0,
      fitness,
      is_active: 1
    });

    if (error) {
      setMessage(error.message);
      return;
    }

    setName("");
    setDisplayName("");
    setMmr(1000);
    setFitness("Medium");
    setMessage("Player added.");
    await refresh();
  }

  async function savePlayer(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selected) return;
    setMessage("");
    const supabase = getBrowserSupabase();
    const { error } = await supabase
      .from("players")
      .update({
        display_name: editDisplayName.trim() || tidyName(selected.name),
        mmr: editMmr,
        fitness: editFitness
      })
      .eq("id", selected.id)
      .eq("league_id", leagueId);

    if (error) {
      setMessage(error.message);
      return;
    }
    setEditingId(null);
    setMessage("Player updated.");
    await refresh();
  }

  async function archivePlayer(player: Player) {
    setMessage("");
    const supabase = getBrowserSupabase();
    const { error } = await supabase
      .from("players")
      .update({ is_active: 0, archived_at: new Date().toISOString() })
      .eq("id", player.id)
      .eq("league_id", leagueId);
    if (error) {
      setMessage(error.message);
      return;
    }
    await refresh();
  }

  return (
    <div className="interactive-stack">
      <form className="panel auth-form" onSubmit={addPlayer}>
        <h2>Add Player</h2>
        <div className="control-bar">
          <label><span>Name</span><input value={name} onChange={(event) => setName(event.target.value)} placeholder="bill smith" /></label>
          <label><span>Display name</span><input value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="Bill" /></label>
          <label><span>Starting MMR</span><input type="number" value={mmr} onChange={(event) => setMmr(Number(event.target.value))} /></label>
          <label><span>Fitness</span><select value={fitness} onChange={(event) => setFitness(event.target.value)}><option>High</option><option>Medium</option><option>Low</option></select></label>
        </div>
        <button className="button primary" type="submit">Add player</button>
      </form>

      {selected ? (
        <form className="panel auth-form" onSubmit={savePlayer}>
          <h2>Edit {displayPlayerName(selected)}</h2>
          <div className="control-bar compact">
            <label><span>Display name</span><input value={editDisplayName} onChange={(event) => setEditDisplayName(event.target.value)} /></label>
            <label><span>MMR</span><input type="number" value={editMmr} onChange={(event) => setEditMmr(Number(event.target.value))} /></label>
            <label><span>Fitness</span><select value={editFitness} onChange={(event) => setEditFitness(event.target.value)}><option>High</option><option>Medium</option><option>Low</option></select></label>
          </div>
          <div className="button-row">
            <button className="button primary" type="submit">Save changes</button>
            <button className="button" type="button" onClick={() => setEditingId(null)}>Cancel</button>
            <button className="button" type="button" onClick={() => archivePlayer(selected)}>Archive</button>
          </div>
        </form>
      ) : null}

      {message ? <p className="auth-message">{message}</p> : null}

      <div className="panel">
        <h2>Manage Players</h2>
        <div className="admin-player-list">
          {players.map((player) => (
            <button type="button" onClick={() => startEdit(player)} key={player.id}>
              <strong>{displayPlayerName(player)}</strong>
              <span>{Math.round(Number(player.mmr || 1000))} MMR</span>
            </button>
          ))}
        </div>
      </div>

      <InteractivePlayers players={summaries} showFitness />
    </div>
  );
}
