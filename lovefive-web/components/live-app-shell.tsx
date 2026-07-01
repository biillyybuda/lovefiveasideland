"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { BrandMark } from "@/components/brand-mark";
import { getBrowserSupabase } from "@/lib/auth-client";
import { clearSelectedLeague, isAdminRole, type LeagueOption } from "@/lib/live-data";

export function LiveAppShell({
  active,
  children,
  league
}: {
  active: "overview" | "matchday" | "charts" | "season" | "info" | "admin" | "players" | "matches" | "join";
  children: React.ReactNode;
  league: LeagueOption;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const admin = isAdminRole(league.role);

  async function signOut() {
    const supabase = getBrowserSupabase();
    await supabase.auth.signOut();
    router.replace("/login");
  }

  function changeLeague() {
    clearSelectedLeague();
    router.replace("/app");
  }

  return (
    <div className="shell">
      <header className="site-header">
        <div className="site-header-inner">
          <Link className="site-brand" href="/">
            <BrandMark />
          </Link>
          <nav className="site-nav" aria-label="League pages">
            <Link className={`site-nav-link ${active === "overview" ? "active" : ""}`} href="/app">Home</Link>
            <Link className={`site-nav-link ${active === "matchday" ? "active" : ""}`} href="/app/matchday">Matchday Hub</Link>
            <Link className={`site-nav-link ${active === "charts" ? "active" : ""}`} href="/app/charts">Charts & Stats</Link>
            <Link className={`site-nav-link ${active === "season" ? "active" : ""}`} href="/app/season-review">Season Review</Link>
            <Link className={`site-nav-link ${active === "info" ? "active" : ""}`} href="/app/info">Info</Link>
            <span className="site-league-link">{league.name}</span>
          </nav>
          <div className="header-actions">
            <details className="profile-menu">
              <summary>
                <span>{league.role || "member"}</span>
                <strong>LF</strong>
              </summary>
              <div className="profile-panel">
                <div>
                  <span>{league.role || "member"}</span>
                  <strong>{league.name}</strong>
                  <small>{admin ? "Admin tools enabled" : "Member view"}</small>
                </div>
                <button className="profile-action" type="button" onClick={changeLeague}>Change league</button>
                <Link href="/app/join-invite">Join / Invite</Link>
                {admin ? <Link href="/app/admin-tools/add-result">Add Result</Link> : null}
                {admin ? <Link href="/app/admin-tools/player-management">Player Management</Link> : null}
                <button className="profile-action" type="button" onClick={signOut}>Sign out</button>
              </div>
            </details>
          </div>
        </div>
      </header>
      <main className={pathname?.startsWith("/app/matchday") ? "main matchday-main" : "main"}>
        {children}
      </main>
    </div>
  );
}
