"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import type { User } from "@supabase/supabase-js";
import { getBrowserSupabase } from "@/lib/auth-client";
import { INVALID_RECOVERY_MESSAGE, MIN_PASSWORD_LENGTH, RECOVERY_STORAGE_KEY, saveRecoveredPassword, verifyPasswordRecovery } from "@/lib/password-recovery";

type State = "checking" | "ready" | "invalid" | "saved";

export function ResetPasswordPanel() {
  const [state, setState] = useState<State>("checking");
  const [user, setUser] = useState<User | null>(null);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const verification = useRef<Promise<User> | null>(null);
  const saving = useRef(false);

  useEffect(() => {
    let mounted = true;
    let unsubscribe = () => {};
    try {
      // Capture before constructing the client, which consumes the URL fragment.
      const href = window.location.href;
      const supabase = getBrowserSupabase();
      verification.current ??= verifyPasswordRecovery(supabase, href, window.sessionStorage);
      verification.current.then(verified => {
        if (mounted) { setUser(verified); setState("ready"); }
      }).catch(cause => {
        if (mounted) { setMessage(cause instanceof Error ? cause.message : INVALID_RECOVERY_MESSAGE); setState("invalid"); }
      }).finally(() => {
        // Never leave recovery credentials in browser history or shareable URLs.
        if (mounted) window.history.replaceState(window.history.state, "", "/reset-password");
      });
      const { data } = supabase.auth.onAuthStateChange(event => {
        if (event === "SIGNED_OUT" && !saving.current && mounted) {
          window.sessionStorage.removeItem(RECOVERY_STORAGE_KEY);
          setUser(null); setState("invalid"); setMessage(INVALID_RECOVERY_MESSAGE);
        }
      });
      unsubscribe = () => data.subscription.unsubscribe();
    } catch {
      setState("invalid"); setMessage("Could not open password reset. Please refresh and try again.");
    }
    return () => { mounted = false; unsubscribe(); };
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (saving.current || state !== "ready" || !user) return;
    saving.current = true; setBusy(true); setMessage("");
    try {
      const supabase = getBrowserSupabase();
      await saveRecoveredPassword(supabase, password, confirm, user.id);
      setPassword(""); setConfirm("");
      window.sessionStorage.removeItem(RECOVERY_STORAGE_KEY);
      setState("saved");
      // A sign-out failure must not suggest the already-saved password failed.
      try { await supabase.auth.signOut(); } catch { /* The password was saved successfully. */ }
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : "Could not save your password. Please try again.");
    } finally { saving.current = false; setBusy(false); }
  }

  return <div className="auth-layout"><section className="auth-panel panel">
    <div className="eyebrow">Account recovery</div>
    <h1>{state === "saved" ? "Password updated" : "Set a new password"}</h1>
    {state === "checking" && <p role="status">Checking your reset link…</p>}
    {state === "ready" && <>
      <p className="lead">Choose a new password for {user?.email}.</p>
      <form className="auth-form" onSubmit={submit}>
        <label><span>New password</span><input type="password" autoComplete="new-password" minLength={MIN_PASSWORD_LENGTH} required value={password} disabled={busy} onChange={event => setPassword(event.target.value)} /></label>
        <label><span>Confirm new password</span><input type="password" autoComplete="new-password" minLength={MIN_PASSWORD_LENGTH} required value={confirm} disabled={busy} onChange={event => setConfirm(event.target.value)} /></label>
        <p className="muted">Use at least {MIN_PASSWORD_LENGTH} characters.</p>
        {message && <p className="auth-message" role="alert">{message}</p>}
        <button className="button primary" type="submit" disabled={busy}>{busy ? "Saving…" : "Save new password"}</button>
      </form>
    </>}
    {state === "invalid" && <><p className="auth-message" role="alert">{message || INVALID_RECOVERY_MESSAGE}</p><Link className="button primary" href="/login?mode=reset">Request a new reset link</Link></>}
    {state === "saved" && <p role="status">Your password has been changed. Sign in with your new password.</p>}
    {state !== "checking" && <div className="auth-actions"><Link className="button" href="/login">Back to sign in</Link></div>}
  </section></div>;
}
