"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { getBrowserSupabase } from "@/lib/auth-client";

type AuthMode = "login" | "signup";

export function LoginPanel() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialMode = searchParams.get("mode") === "signup" ? "signup" : "login";
  const [mode, setMode] = useState<AuthMode>(initialMode);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const nextPath = useMemo(() => {
    const raw = searchParams.get("next");
    return raw && raw.startsWith("/") ? raw : "/app";
  }, [searchParams]);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage("");

    const cleanEmail = email.trim();
    if (!cleanEmail || !password) {
      setMessage("Enter your email and password.");
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

    setBusy(true);
    const supabase = getBrowserSupabase();
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
    setBusy(false);

    if (result.error) {
      setMessage(result.error.message);
      return;
    }

    if (mode === "signup" && !result.data.session) {
      setMessage("Account created. Check your email, then sign in.");
      setMode("login");
      return;
    }

    router.push(nextPath);
    router.refresh();
  }

  return (
    <div className="auth-layout">
      <div className="auth-panel panel">
        <div>
          <div className="eyebrow">Account</div>
          <h1>{mode === "login" ? "Sign in to Love Five" : "Create your Love Five account"}</h1>
          <p className="lead">
            {mode === "login"
              ? "Open your league dashboard, manage matchday, and keep your group stats moving."
              : "Create your account first. Next you can join a league with a code or create one for your group."}
          </p>
        </div>

        <form className="auth-form" onSubmit={submit}>
          <label>
            <span>Email</span>
            <input autoComplete="email" inputMode="email" type="email" value={email} onChange={(event) => setEmail(event.target.value)} />
          </label>
          <label>
            <span>Password</span>
            <input
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
          {mode === "signup" ? (
            <label>
              <span>Confirm password</span>
              <input autoComplete="new-password" type="password" value={confirm} onChange={(event) => setConfirm(event.target.value)} />
            </label>
          ) : null}

          {message ? <p className="auth-message">{message}</p> : null}

          <button className="button primary" disabled={busy} type="submit">
            {busy ? "Working..." : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>

        <div className="auth-actions">
          <button
            className="button"
            type="button"
            onClick={() => {
              setMessage("");
              setMode(mode === "login" ? "signup" : "login");
            }}
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
