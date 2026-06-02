import Link from "next/link";
import { AppShell } from "@/components/app-shell";

export default function DemoAdminToolsPage() {
  return (
    <AppShell active="admin">
      <div className="page-head">
        <div>
          <div className="eyebrow">Demo League</div>
          <h1>Admin Tools</h1>
          <p className="lead">The admin workflows from Streamlit are represented here in demo-safe read-only form.</p>
        </div>
      </div>

      <div className="tool-grid">
        <div className="panel">
          <h2>Add Result</h2>
          <div className="profile-grid">
            <div><span>Date</span><strong>Match date picker</strong></div>
            <div><span>Lineups</span><strong>Team A / Team B selectors</strong></div>
            <div><span>Score</span><strong>Auto result from score</strong></div>
            <div><span>Processing</span><strong>MMR + streak recalculation</strong></div>
          </div>
          <p className="muted">Enabled once website auth and league admin permissions are wired in.</p>
          <div className="button-row"><Link className="button" href="/demo/admin-tools/add-result">Open Add Result</Link></div>
        </div>

        <div className="panel">
          <h2>Player Management</h2>
          <div className="profile-grid">
            <div><span>Edit</span><strong>Name, display name and fitness</strong></div>
            <div><span>Add</span><strong>Single player form</strong></div>
            <div><span>Status</span><strong>Archive / unarchive</strong></div>
            <div><span>Ratings</span><strong>Starting MMR control</strong></div>
          </div>
          <p className="muted">The website will reuse the same display-name-first rules as the app.</p>
          <div className="button-row"><Link className="button" href="/demo/admin-tools/player-management">Open Player Management</Link></div>
        </div>

        <div className="panel">
          <h2>League Admin</h2>
          <div className="profile-grid">
            <div><span>Settings</span><strong>League name and join code</strong></div>
            <div><span>Members</span><strong>Player account links</strong></div>
            <div><span>Plan</span><strong>Subscription limits</strong></div>
            <div><span>Shortcuts</span><strong>Admin navigation</strong></div>
          </div>
          <p className="muted">Demo visitors can see the tool map without being able to change league data.</p>
          <div className="button-row"><Link className="button" href="/demo/admin-tools/league-admin">Open League Admin</Link></div>
        </div>
      </div>
    </AppShell>
  );
}
