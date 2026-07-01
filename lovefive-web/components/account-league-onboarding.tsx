"use client";

import { useState } from "react";
import { getBrowserSupabase } from "@/lib/auth-client";

type OnboardingLeagueResult = {
  league_id: number;
  name: string;
  join_code?: string;
  role?: string;
};

export function AccountLeagueOnboarding({
  email,
  compact = false
}: {
  email?: string | null;
  compact?: boolean;
}) {
  const [code, setCode] = useState("");
  const [leagueToCreate, setLeagueToCreate] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState<"join" | "create" | null>(null);

  async function joinLeague(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage("");
    const clean = code.trim().toUpperCase();
    if (!clean) {
      setMessage("Enter the league code from your organiser.");
      return;
    }

    setBusy("join");
    const supabase = getBrowserSupabase();
    const { data: userResult } = await supabase.auth.getUser();
    const user = userResult.user;
    if (!user) {
      setBusy(null);
      setMessage("Sign in again first.");
      return;
    }

    const { data, error } = await supabase.rpc("join_league_by_code", { invite_code: clean });
    const league = Array.isArray(data) ? data[0] as OnboardingLeagueResult | undefined : undefined;

    if (error || !league) {
      setBusy(null);
      setMessage(error?.message || "That code does not match a league.");
      return;
    }

    setBusy(null);
    window.localStorage.setItem("lovefive-selected-league", String(league.league_id));
    window.location.href = "/app";
  }

  async function createLeague(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage("");
    const name = leagueToCreate.trim();
    if (name.length < 3) {
      setMessage("League name must be at least 3 characters.");
      return;
    }

    setBusy("create");
    const supabase = getBrowserSupabase();
    const { data: userResult } = await supabase.auth.getUser();
    const user = userResult.user;
    if (!user) {
      setBusy(null);
      setMessage("Sign in again first.");
      return;
    }

    const { data, error } = await supabase.rpc("create_league_for_current_user", { league_name: name });
    const league = Array.isArray(data) ? data[0] as OnboardingLeagueResult | undefined : undefined;

    if (error || !league) {
      setBusy(null);
      setMessage(error?.message || "Could not create league.");
      return;
    }

    setBusy(null);
    window.localStorage.setItem("lovefive-selected-league", String(league.league_id));
    window.location.href = "/app";
  }

  return (
    <div className={compact ? "onboarding-panel compact" : "onboarding-panel"}>
      <div className="onboarding-intro">
        <div className="eyebrow">League setup</div>
        <h1>Join a league or start your own.</h1>
        <p className="lead">
          {email ? `You are signed in as ${email}. ` : ""}
          Use a league code from your organiser, or create a new league for your group.
        </p>
      </div>

      <div className="onboarding-choice-grid">
        <form className="panel auth-form onboarding-choice" onSubmit={joinLeague}>
          <div>
            <span>Join existing</span>
            <h2>I have a league code</h2>
            <p className="muted">Ask your organiser for the code, then you will be added as a member.</p>
          </div>
          <label>
            <span>League code</span>
            <input value={code} onChange={(event) => setCode(event.target.value)} placeholder="Example: LOVE5A" />
          </label>
          <button className="button primary" disabled={busy !== null} type="submit">
            {busy === "join" ? "Joining..." : "Join league"}
          </button>
        </form>

        <form className="panel auth-form onboarding-choice" onSubmit={createLeague}>
          <div>
            <span>Create new</span>
            <h2>I run the group</h2>
            <p className="muted">Create the league, then add players and share the invite code.</p>
          </div>
          <label>
            <span>League name</span>
            <input value={leagueToCreate} onChange={(event) => setLeagueToCreate(event.target.value)} placeholder="Thursday 5s" />
          </label>
          <button className="button primary" disabled={busy !== null} type="submit">
            {busy === "create" ? "Creating..." : "Create league"}
          </button>
        </form>
      </div>

      {message ? <p className="auth-message">{message}</p> : null}
    </div>
  );
}
