"use client";

import { useState } from "react";
import { getBrowserSupabase } from "@/lib/auth-client";
import { MATCH_NOTE_LIMIT, saveMatchNote } from "@/lib/match-notes";

export function MatchNoteEditor({ leagueId, matchId, note, refresh }: { leagueId: number; matchId: number; note: string; refresh: () => Promise<void> }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(note);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  if (!editing) return <button className="mini-link" type="button" onClick={() => { setValue(note); setEditing(true); }}>{note ? "Edit match note" : "Add a match note"}</button>;
  return <form className="review-note-form" onSubmit={async event => {
    event.preventDefault(); setBusy(true); setMessage("");
    try { await saveMatchNote(getBrowserSupabase(), leagueId, matchId, note, value); await refresh(); setEditing(false); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Could not save the note."); }
    finally { setBusy(false); }
  }}>
    <label className="review-note-label"><span>Anything worth mentioning?</span><textarea maxLength={MATCH_NOTE_LIMIT} value={value} onChange={e => setValue(e.target.value)} placeholder="A comeback, a debut, a great save, or a moment from the group chat…"/>
      <small>Optional · {value.length}/{MATCH_NOTE_LIMIT} · Visible to everyone who can read this match.</small></label>
    <div className="review-nav"><button className="button primary" type="submit" disabled={busy}>{busy ? "Saving…" : "Save note"}</button><button className="mini-link" type="button" disabled={busy} onClick={() => { setEditing(false); setMessage(""); }}>Cancel</button></div>
    {message && <p role="alert">{message}</p>}
  </form>;
}
