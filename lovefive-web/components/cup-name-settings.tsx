"use client";

import { useState, type FormEvent } from "react";
import { getBrowserSupabase } from "@/lib/auth-client";
import { MAX_CUP_NAME_LENGTH } from "@/lib/competition";
import { saveCupName } from "@/lib/cup-settings";
import { isAdminRole, type LeagueOption } from "@/lib/live-data";
import { useCupName } from "./cup-name-provider";

export function CupNameSettings({ league }: { league: LeagueOption }) {
  const { name, setName } = useCupName();
  const [draft, setDraft] = useState(name);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  if (!isAdminRole(league.role)) return null;

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy) return;
    setBusy(true); setError(""); setMessage("");
    try {
      const saved = await saveCupName(getBrowserSupabase(), league.id, draft);
      setName(saved); setDraft(saved);
      setMessage(`Cup name saved as ${saved}. Pages and newly generated graphics will use this name.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save the cup name.");
    } finally { setBusy(false); }
  }

  return <section className="panel cup-name-settings">
    <h2>Cup name</h2>
    <p className="muted">Choose the name used throughout {league.name}, including lineup and player-card images. Cup results keep their MMR immunity.</p>
    <form onSubmit={save}>
      <label htmlFor="cup-name">Cup name</label>
      <input id="cup-name" name="cup-name" required maxLength={MAX_CUP_NAME_LENGTH} value={draft}
        onChange={event => { setDraft(event.target.value); setMessage(""); setError(""); }} disabled={busy} />
      <button className="primary-button" type="submit" disabled={busy || !draft.trim()}>{busy ? "Saving…" : "Save cup name"}</button>
    </form>
    {message && <p role="status">{message}</p>}
    {error && <p role="alert">{error}</p>}
  </section>;
}
