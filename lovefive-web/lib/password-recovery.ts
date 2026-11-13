import type { SupabaseClient } from "@supabase/supabase-js";

export const RECOVERY_STORAGE_KEY = "lovefive-password-recovery-user";
export const RESET_EMAIL_MESSAGE = "If an account exists for that email, you’ll receive a password-reset link. Check your inbox and spam folder.";
export const INVALID_RECOVERY_MESSAGE = "This reset link is invalid or has expired. Request a new link and open the newest email.";
export const MIN_PASSWORD_LENGTH = 8;

type RecoveryStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

/** Keep login redirects within this site, including URL-parser edge cases. */
export function safeNextPath(value: string | null) {
  if (!value?.startsWith("/") || value.startsWith("//") || /[\\\u0000-\u0020]/.test(value)) return "/app";
  const base = "https://lovefive.invalid";
  try {
    const url = new URL(value, base);
    return url.origin === base ? `${url.pathname}${url.search}${url.hash}` : "/app";
  } catch { return "/app"; }
}

export async function requestPasswordReset(supabase: SupabaseClient, email: string, origin: string) {
  const cleanEmail = email.trim();
  if (!cleanEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanEmail)) throw new Error("Enter a valid email address.");
  const { error } = await supabase.auth.resetPasswordForEmail(cleanEmail, {
    redirectTo: new URL("/reset-password", origin).href
  });
  if (error) {
    if (error.status === 429) throw new Error("Too many reset requests. Wait a few minutes, then try again.");
    throw new Error("Could not send the reset email. Please try again, or contact your league admin if it keeps happening.");
  }
  return RESET_EMAIL_MESSAGE;
}

/** Supabase validates the recovery token; the tab marker only supports refreshes
 * after the SDK removes the URL fragment. It is never a substitute for auth. */
export async function verifyPasswordRecovery(supabase: SupabaseClient, href: string, storage: RecoveryStorage) {
  const url = new URL(href);
  const hash = new URLSearchParams(url.hash.slice(1));
  const hasError = [hash, url.searchParams].some(params => ["error", "error_code", "error_description"].some(key => params.has(key)));
  if (hasError) {
    storage.removeItem(RECOVERY_STORAGE_KEY);
    throw new Error(INVALID_RECOVERY_MESSAGE);
  }
  const isRecoveryLink = hash.get("type") === "recovery" && Boolean(hash.get("access_token") && hash.get("refresh_token"));
  const hasCallback = url.hash.length > 1 || ["code", "token_hash", "type"].some(key => url.searchParams.has(key));
  if (hasCallback && !isRecoveryLink) {
    storage.removeItem(RECOVERY_STORAGE_KEY);
    throw new Error(INVALID_RECOVERY_MESSAGE);
  }
  const { error: initializationError } = await supabase.auth.initialize();
  if (initializationError) {
    storage.removeItem(RECOVERY_STORAGE_KEY);
    throw new Error(INVALID_RECOVERY_MESSAGE);
  }
  const { data: sessionData, error: sessionError } = await supabase.auth.getSession();
  const session = sessionData.session;
  if (sessionError || !session || (!isRecoveryLink && storage.getItem(RECOVERY_STORAGE_KEY) !== session.user.id)) {
    storage.removeItem(RECOVERY_STORAGE_KEY);
    throw new Error(INVALID_RECOVERY_MESSAGE);
  }
  // Validate with the Auth server, not just the browser's stored session.
  const { data, error } = await supabase.auth.getUser();
  if (error || !data.user || data.user.id !== session.user.id) {
    storage.removeItem(RECOVERY_STORAGE_KEY);
    throw new Error("Could not verify this reset session. Try again or request a new link.");
  }
  storage.setItem(RECOVERY_STORAGE_KEY, data.user.id);
  return data.user;
}

export async function saveRecoveredPassword(supabase: SupabaseClient, password: string, confirm: string, userId: string) {
  if (password.length < MIN_PASSWORD_LENGTH) throw new Error(`Use at least ${MIN_PASSWORD_LENGTH} characters.`);
  if (password !== confirm) throw new Error("Passwords do not match.");
  const { data, error: sessionError } = await supabase.auth.getUser();
  if (sessionError || !data.user || data.user.id !== userId) throw new Error(INVALID_RECOVERY_MESSAGE);
  const { error } = await supabase.auth.updateUser({ password });
  if (error) throw new Error(error.message);
}
