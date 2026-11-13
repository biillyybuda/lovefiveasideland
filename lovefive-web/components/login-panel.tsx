"use client";

import { useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { getBrowserSupabase } from "@/lib/auth-client";
import { requestPasswordReset, safeNextPath } from "@/lib/password-recovery";

type AuthMode = "login" | "signup" | "reset";

export function LoginPanel() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialMode = searchParams.get("mode") === "reset" ? "reset" : searchParams.get("mode") === "signup" ? "signup" : "login";
  const [mode, setMode] = useState<AuthMode>(initialMode);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const submitting = useRef(false);

  const nextPath = useMemo(() => {
    return safeNextPath(searchParams.get("next"));
  }, [searchParams]);

  function changeMode(next: AuthMode) {
    setMode(next); setMessage(""); setPassword(""); setConfirm("");
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting.current) return;
    setMessage("");

    const cleanEmail = email.trim();
    if (!cleanEmail || (mode !== "reset" && !password)) {
      setMessage(mode === "reset" ? "Enter your email address." : "Enter your email and password.");
      return;
    }

    if (mode === "signup") {
      if (password.length < 6) {
        setMessage("Password must be at least 6 characters.");
        return;
      }
      if (password !== confirm) {
        setMessage("Passwords do not match.");
        return;
      }
    }

    submitting.current = true;
    setBusy(true);
    try {
      const supabase = getBrowserSupabase();
      if (mode === "reset") {
        setMessage(await requestPasswordReset(supabase, cleanEmail, window.location.origin));
        return;
      }
      const result =
        mode === "login"
          ? await supabase.auth.signInWithPassword({ email: cleanEmail, password })
          : await supabase.auth.signUp({
              email: cleanEmail,
              password,
              options: {
                emailRedirectTo: `${window.location.origin}/app`
              }
            });
      if (result.error) {
        setMessage(result.error.message);
        return;
      }

      if (mode === "signup" && !result.data.session) {
        setMessage("Account created. Check your email, then sign in.");
        setPassword(""); setConfirm(""); setMode("login");
        return;
      }

      router.push(nextPath);
      router.refresh();
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : "Could not connect. Please try again.");
    } finally { submitting.current = false; setBusy(false); }
  }

  return (
    <div className="auth-layout">
      <div className="auth-panel panel">
        <div>
          <div className="eyebrow">Account</div>
          <h1>{mode === "reset" ? "Reset your password" : mode === "login" ? "Sign in to Love Five" : "Create your Love Five account"}</h1>
          <p className="lead">
            {mode === "reset" ? "Enter the email address you registered with. We’ll send you a link to choose a new password." : mode === "login"
              ? "Open your league dashboard, manage matchday, and keep your group stats moving."
              : "Create your account first. Next you can join a league with a code or create one for your group."}
          </p>
        </div>

        <form className="auth-form" onSubmit={submit}>
          <label>
            <span>Email</span>
            <input autoComplete="email" inputMode="email" type="email" required disabled={busy} value={email} onChange={(event) => setEmail(event.target.value)} />
          </label>
          {mode !== "reset" && <label>
            <span>Password</span>
            <input
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              type="password"
              required
              disabled={busy}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>}
          {mode === "signup" ? (
            <label>
              <span>Confirm password</span>
              <input autoComplete="new-password" type="password" required disabled={busy} value={confirm} onChange={(event) => setConfirm(event.target.value)} />
            </label>
          ) : null}

          {message ? <p className="auth-message" role="status">{message}</p> : null}

          <button className="button primary" disabled={busy} type="submit">
            {busy ? "Working..." : mode === "reset" ? "Send reset link" : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>

        <div className="auth-actions">
          {mode === "login" && <button className="button" type="button" disabled={busy} onClick={() => changeMode("reset")}>Forgot password?</button>}
          <button
            className="button"
            type="button"
            disabled={busy}
            onClick={() => changeMode(mode === "login" ? "signup" : "login")}
          >
            {mode === "login" ? "Create an account" : "Sign in instead"}
          </button>
          <Link className="button" href="/demo">
            View demo
          </Link>
          <Link className="button" href="/">
            Back to home
          </Link>
        </div>
      </div>
    </div>
  );
}
