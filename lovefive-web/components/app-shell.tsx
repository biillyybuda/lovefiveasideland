import Link from "next/link";
import { BrandMark } from "@/components/brand-mark";
import { ThemeToggle } from "@/components/theme-toggle";

const nav = [
  { href: "/demo", label: "Home", key: "overview" },
  { href: "/demo/matchday", label: "Matchday Hub", key: "matchday" },
  { href: "/demo/charts", label: "Charts & Stats", key: "charts" },
  { href: "/demo/season-review", label: "Season Review", key: "season" },
  { href: "/demo/info", label: "Info", key: "info" }
];

export function AppShell({
  active,
  children
}: {
  active:
    | "overview"
    | "matchday"
    | "dashboard"
    | "players"
    | "matches"
    | "charts"
    | "insights"
    | "chemistry"
    | "relationships"
    | "season"
    | "info"
    | "join"
    | "profile"
    | "admin";
  children: React.ReactNode;
}) {
  return (
    <div className="shell">
      <header className="site-header">
        <div className="site-header-inner">
          <Link className="site-brand" href="/">
            <BrandMark />
          </Link>
          <nav className="site-nav" aria-label="Demo league pages">
            {nav.map((item) => (
              <Link className={`site-nav-link ${active === item.key ? "active" : ""}`} href={item.href} key={item.href}>
                {item.label}
              </Link>
            ))}
            <Link className="site-league-link" href="/demo">
              Love Five Demo
            </Link>
          </nav>
          <div className="header-actions">
            <ThemeToggle />
            <details className="profile-menu">
              <summary>
                <span>Demo</span>
                <strong>LF</strong>
              </summary>
              <div className="profile-panel">
                <div>
                  <span>Demo League</span>
                  <strong>Love Five Demo</strong>
                  <small>Read-only preview</small>
                </div>
                <Link href="/demo/join-invite">Join / Invite</Link>
                <Link href="/demo/profile-settings">Profile Settings</Link>
                <Link href="/login">Sign in</Link>
                <Link href="/">Create league</Link>
                <Link href="/demo/admin-tools">Admin tools</Link>
              </div>
            </details>
          </div>
        </div>
      </header>
      <main className="main">
        {children}
      </main>
    </div>
  );
}
