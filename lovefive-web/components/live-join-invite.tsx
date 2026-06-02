"use client";

import { useState } from "react";
import { getBrowserSupabase } from "@/lib/auth-client";
import { canonicalName } from "@/lib/live-data";

export function LiveJoinInvite({
  leagueId,
  leagueName,
  joinCode
}: {
  leagueId: number;
  leagueName: string;
  joinCode: string | null;
}) {
  const [code, setCode] = useState("");
  const [leagueToCreate, setLeagueToCreate] = useState("");
  const [message, setMessage] = useState("");

  async function joinLeague(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage("");
    const supabase = getBrowserSupabase();
    const clean = code.trim().toUpperCase();
    if (!clean) {
      setMessage("Enter a league code.");
      return;
    }

    const { data: userResult } = await supabase.auth.getUser();
    const user = userResult.user;
    if (!user) {
      setMessage("Sign in again first.");
      return;
    }

    const { data: league, error: leagueError } = await supabase
      .from("leagues")
      .select("id,name")
      .eq("join_code", clean)
      .maybeSingle();

    if (leagueError || !league) {
      setMessage(leagueError?.message || "That code does not match a league.");
      return;
    }

    const { error } = await supabase.from("league_members").upsert({
      league_id: league.id,
      user_id: user.id,
      role: "member",
      status: "active"
    }, { onConflict: "league_id,user_id" });

    if (error) {
      setMessage(error.message);
      return;
    }

    window.localStorage.setItem("lovefive-selected-league", String(league.id));
    window.location.href = "/app";
  }

  async function createLeague(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage("");
    const supabase = getBrowserSupabase();
    const name = leagueToCreate.trim();
    if (name.length < 3) {
      setMessage("League name must be at least 3 characters.");
      return;
    }

    const { data: userResult } = await supabase.auth.getUser();
    const user = userResult.user;
    if (!user) {
      setMessage("Sign in again first.");
      return;
    }

    const generatedCode = canonicalName(name).replace(/[^a-z0-9]/g, "").slice(0, 4).toUpperCase() + Math.random().toString(36).slice(2, 6).toUpperCase();
    const { data: league, error: leagueError } = await supabase
      .from("leagues")
      .insert({ name, join_code: generatedCode })
      .select("id,name")
      .single();

    if (leagueError || !league) {
      setMessage(leagueError?.message || "Could not create league.");
      return;
    }

    const { error } = await supabase.from("league_members").insert({
      league_id: league.id,
      user_id: user.id,
      role: "admin",
      status: "active"
    });

    if (error) {
      setMessage(error.message);
      return;
    }

    window.localStorage.setItem("lovefive-selected-league", String(league.id));
    window.location.href = "/app";
  }

  return (
    <div className="tool-grid">
      <div className="panel">
        <h2>Invite Players</h2>
        <div className="profile-grid">
          <div><span>League</span><strong>{leagueName}</strong></div>
          <div><span>League code</span><strong>{joinCode || "-"}</strong></div>
        </div>
      </div>
      <form className="panel auth-form" onSubmit={joinLeague}>
        <h2>Join League</h2>
        <label><span>League code</span><input value={code} onChange={(event) => setCode(event.target.value)} /></label>
        <button className="button primary" type="submit">Join league</button>
      </form>
      <form className="panel auth-form" onSubmit={createLeague}>
        <h2>Create League</h2>
        <label><span>League name</span><input value={leagueToCreate} onChange={(event) => setLeagueToCreate(event.target.value)} /></label>
        <button className="button primary" type="submit">Create league</button>
      </form>
      {message ? <div className="panel auth-message">{message}</div> : null}
    </div>
  );
}
