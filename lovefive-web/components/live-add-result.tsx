"use client";

import { useMemo, useState } from "react";
import { MatchCard } from "@/components/match-card";
import { getBrowserSupabase } from "@/lib/auth-client";
import { makeNameMap, scoreParts, type Match, type Player } from "@/lib/demo-data";
import { calculateMatchMmrUpdates } from "@/lib/mmr-engine";
import { displayPlayerName } from "@/lib/live-data";

function resultFromScore(score: string) {
  const parsed = scoreParts(score);
  if (!parsed) return "";
  if (parsed[0] > parsed[1]) return "A";
  if (parsed[1] > parsed[0]) return "B";
  return "Draw";
}

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
    const result = resultFromScore(score);
    if (!teamA.length || !teamB.length) {
      setMessage("Select at least one player for each team.");
      return;
    }
    if (!parsed || !result) {
      setMessage("Enter the score like 10-8.");
      return;
    }

    setBusy(true);
    const supabase = getBrowserSupabase();
    const teamAPlayers = players.filter((player) => teamA.includes(player.id));
    const teamBPlayers = players.filter((player) => teamB.includes(player.id));
    const teamAString = teamAPlayers.map((player) => player.name).join(", ");
    const teamBString = teamBPlayers.map((player) => player.name).join(", ");

    const { data: match, error: matchError } = await supabase
      .from("matches")
      .insert({
        league_id: leagueId,
        date,
        team_a: teamAString,
        team_b: teamBString,
        score,
        result,
        processed: 1
      })
      .select("id,date,team_a,team_b,score,result,processed")
      .single();

    if (matchError || !match) {
      setBusy(false);
      setMessage(matchError?.message || "Could not save match.");
      return;
    }

    const updates = calculateMatchMmrUpdates({
      teamA: teamAPlayers,
      teamB: teamBPlayers,
      match,
      players
    });

    for (const update of updates) {
      const current = update.player;
      const sideWon = result === update.team;
      const draw = result === "Draw";
      const { error } = await supabase
        .from("players")
        .update({
          mmr: Math.round(update.after),
          matches_played: Number(current.matches_played || 0) + 1,
          wins: Number(current.wins || 0) + (sideWon ? 1 : 0),
          draws: Number(current.draws || 0) + (draw ? 1 : 0),
          losses: Number(current.losses || 0) + (!sideWon && !draw ? 1 : 0)
        })
        .eq("id", current.id)
        .eq("league_id", leagueId);
      if (error) {
        setBusy(false);
        setMessage(error.message);
        return;
      }

      const { error: historyError } = await supabase.from("mmr_history").insert({
        league_id: leagueId,
        player_id: current.id,
        match_id: match.id,
        date,
        mmr_before: Math.round(update.before),
        mmr_after: Math.round(update.after)
      });
      if (historyError) {
        setBusy(false);
        setMessage(historyError.message);
        return;
      }
    }

    setTeamA([]);
    setTeamB([]);
    setScore("");
    setBusy(false);
    setMessage("Match saved and ratings updated.");
    await refresh();
  }

  async function deleteMatch(matchId: number) {
    setMessage("");
    const supabase = getBrowserSupabase();
    const { error } = await supabase.from("matches").delete().eq("id", matchId).eq("league_id", leagueId);
    if (error) {
      setMessage(error.message);
      return;
    }
    await refresh();
  }

  return (
    <div className="interactive-stack">
      <form className="panel" onSubmit={saveMatch}>
        <h2>Add Result</h2>
        <div className="control-bar compact">
          <label><span>Date</span><input type="date" value={date} onChange={(event) => setDate(event.target.value)} /></label>
          <label><span>Score</span><input value={score} onChange={(event) => setScore(event.target.value)} placeholder="10-8" /></label>
          <button className="button primary" disabled={busy} type="submit">{busy ? "Saving..." : "Save result"}</button>
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

      <div className="panel">
        <h2>Recent Matches</h2>
        <div className="grid compact-grid">
          {matches.slice(0, 12).map((match) => (
            <div className="admin-match-wrap" key={match.id}>
              <MatchCard match={match} nameMap={nameMap} />
              <button className="button" type="button" onClick={() => deleteMatch(match.id)}>Delete</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
