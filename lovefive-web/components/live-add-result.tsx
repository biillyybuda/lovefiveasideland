"use client";

import { useMemo, useState } from "react";
import { Pencil, RefreshCw, Save, Trash2, X } from "lucide-react";
import { MatchCard } from "@/components/match-card";
import { getBrowserSupabase } from "@/lib/auth-client";
import { makeNameMap, scoreParts, type Match, type Player } from "@/lib/demo-data";
import { displayPlayerName } from "@/lib/live-data";
import { normaliseScoreText, recalculateLeagueResults, resultLabelFromScore } from "@/lib/result-rebuild";

export function LiveAddResult({
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
  const [date, setDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [teamA, setTeamA] = useState<number[]>([]);
  const [teamB, setTeamB] = useState<number[]>([]);
  const [score, setScore] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [editingMatch, setEditingMatch] = useState<Match | null>(null);
  const [editDate, setEditDate] = useState("");
  const [editScore, setEditScore] = useState("");
  const nameMap = useMemo(() => makeNameMap(players), [players]);

  function togglePlayer(id: number, side: "A" | "B") {
    if (side === "A") {
      setTeamB((rows) => rows.filter((item) => item !== id));
      setTeamA((rows) => rows.includes(id) ? rows.filter((item) => item !== id) : [...rows, id]);
    } else {
      setTeamA((rows) => rows.filter((item) => item !== id));
      setTeamB((rows) => rows.includes(id) ? rows.filter((item) => item !== id) : [...rows, id]);
    }
  }

  async function saveMatch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage("");
    const parsed = scoreParts(score);
    const result = resultLabelFromScore(score);
    const cleanScore = normaliseScoreText(score);
    if (!teamA.length || !teamB.length) {
      setMessage("Select at least one player for each team.");
      return;
    }
    if (!parsed || !cleanScore) {
      setMessage("Enter the score like 10-8.");
      return;
    }

    setBusy(true);
    let matchSaved = false;

    try {
      const supabase = getBrowserSupabase();
      const teamAPlayers = players.filter((player) => teamA.includes(player.id));
      const teamBPlayers = players.filter((player) => teamB.includes(player.id));
      const teamAString = teamAPlayers.map((player) => player.name).join(", ");
      const teamBString = teamBPlayers.map((player) => player.name).join(", ");

      const { error: matchError } = await supabase
        .from("matches")
        .insert({
          league_id: leagueId,
          date,
          team_a: teamAString,
          team_b: teamBString,
          score: cleanScore,
          result,
          processed: 1
        });

      if (matchError) throw new Error(matchError.message);
      matchSaved = true;
      const summary = await recalculateLeagueResults(supabase, leagueId);

      setTeamA([]);
      setTeamB([]);
      setScore("");
      setMessage(`Match saved. Recalculated ${summary.processedMatches} matches.`);
      await refresh();
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Could not save match.";
      setMessage(matchSaved ? `Match saved, but stats could not be rebuilt: ${detail}` : detail);
    } finally {
      setBusy(false);
    }
  }

  function startEdit(match: Match) {
    setEditingMatch(match);
    setEditDate(String(match.date || "").slice(0, 10));
    setEditScore(normaliseScoreText(match.score) || String(match.score || ""));
    setMessage("");
  }

  function cancelEdit() {
    setEditingMatch(null);
    setEditDate("");
    setEditScore("");
  }

  async function saveEditedMatch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!editingMatch) return;
    setMessage("");
    const parsed = scoreParts(editScore);
    const cleanScore = normaliseScoreText(editScore);
    if (!editDate) {
      setMessage("Choose a match date.");
      return;
    }
    if (!parsed || !cleanScore) {
      setMessage("Enter the score like 10-8.");
      return;
    }

    setBusy(true);
    try {
      const supabase = getBrowserSupabase();
      const { error } = await supabase
        .from("matches")
        .update({
          date: editDate,
          score: cleanScore,
          result: resultLabelFromScore(cleanScore),
          processed: 1
        })
        .eq("id", editingMatch.id)
        .eq("league_id", leagueId);

      if (error) throw new Error(error.message);
      const summary = await recalculateLeagueResults(supabase, leagueId);
      cancelEdit();
      setMessage(`Result corrected. Recalculated ${summary.processedMatches} matches.`);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not update match.");
    } finally {
      setBusy(false);
    }
  }

  async function deleteMatch(matchId: number) {
    setMessage("");
    const confirmed = window.confirm("Delete this match and rebuild the league stats?");
    if (!confirmed) return;

    setBusy(true);
    try {
      const supabase = getBrowserSupabase();
      const { error } = await supabase.from("matches").delete().eq("id", matchId).eq("league_id", leagueId);
      if (error) throw new Error(error.message);
      const summary = await recalculateLeagueResults(supabase, leagueId);
      if (editingMatch?.id === matchId) cancelEdit();
      setMessage(`Match deleted. Recalculated ${summary.processedMatches} matches.`);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not delete match.");
    } finally {
      setBusy(false);
    }
  }

  async function rebuildStats() {
    setMessage("");
    setBusy(true);
    try {
      const supabase = getBrowserSupabase();
      const summary = await recalculateLeagueResults(supabase, leagueId);
      setMessage(`Recalculated ${summary.processedMatches} matches and ${summary.historyRows} rating rows.`);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not recalculate stats.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="interactive-stack">
      <form className="panel" onSubmit={saveMatch}>
        <h2>Add Result</h2>
        <div className="control-bar match-management-controls">
          <label><span>Date</span><input type="date" value={date} onChange={(event) => setDate(event.target.value)} /></label>
          <label><span>Score</span><input value={score} onChange={(event) => setScore(event.target.value)} placeholder="10-8" /></label>
          <button className="button primary" disabled={busy} type="submit"><Save />{busy ? "Saving..." : "Save result"}</button>
        </div>
        <div className="team-picker-grid">
          <div>
            <h3>Team A</h3>
            <div className="player-chip-grid">
              {players.map((player) => (
                <button className={teamA.includes(player.id) ? "active" : ""} type="button" onClick={() => togglePlayer(player.id, "A")} key={player.id}>
                  {displayPlayerName(player)}
                </button>
              ))}
            </div>
          </div>
          <div>
            <h3>Team B</h3>
            <div className="player-chip-grid">
              {players.map((player) => (
                <button className={teamB.includes(player.id) ? "active" : ""} type="button" onClick={() => togglePlayer(player.id, "B")} key={player.id}>
                  {displayPlayerName(player)}
                </button>
              ))}
            </div>
          </div>
        </div>
        {message ? <p className="auth-message">{message}</p> : null}
      </form>

      {editingMatch ? (
        <form className="panel auth-form" onSubmit={saveEditedMatch}>
          <div className="admin-section-head">
            <h2>Edit Result</h2>
            <button className="mini-link" disabled={busy} type="button" onClick={cancelEdit}><X />Cancel</button>
          </div>
          <div className="control-bar match-management-controls">
            <label><span>Date</span><input type="date" value={editDate} onChange={(event) => setEditDate(event.target.value)} /></label>
            <label><span>Score</span><input value={editScore} onChange={(event) => setEditScore(event.target.value)} placeholder="10-8" /></label>
            <button className="button primary" disabled={busy} type="submit"><Save />Save correction</button>
          </div>
        </form>
      ) : null}

      <div className="panel">
        <div className="admin-section-head">
          <h2>Recent Matches</h2>
          <button className="button" disabled={busy} type="button" onClick={rebuildStats}><RefreshCw />Recalculate stats</button>
        </div>
        <div className="grid compact-grid">
          {matches.slice(0, 12).map((match) => (
            <div className="admin-match-wrap" key={match.id}>
              <MatchCard match={match} nameMap={nameMap} />
              <div className="match-management-actions">
                <button className="button" disabled={busy} type="button" onClick={() => startEdit(match)}><Pencil />Edit score</button>
                <button className="button danger" disabled={busy} type="button" onClick={() => deleteMatch(match.id)}><Trash2 />Delete</button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
