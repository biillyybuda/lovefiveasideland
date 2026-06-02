import Link from "next/link";
import { AppShell } from "@/components/app-shell";

export default function DemoLeagueAdminPage() {
  return (
    <AppShell active="admin">
      <div className="page-head">
        <div>
          <div className="eyebrow">Admin</div>
          <h1>League Admin</h1>
          <p className="lead">League settings, subscription state and player-account links.</p>
        </div>
      </div>

      <div className="tool-grid">
        <div className="panel">
          <h2>League Settings</h2>
          <div className="control-bar compact">
            <label><span>League name</span><input disabled value="Love Five Demo League" readOnly /></label>
            <button className="button" disabled>Save</button>
          </div>
        </div>
        <div className="panel">
          <h2>Admin Shortcuts</h2>
          <div className="button-row">
            <Link className="button" href="/demo/admin-tools/add-result">Add Result</Link>
            <Link className="button" href="/demo/admin-tools/player-management">Player Management</Link>
          </div>
        </div>
        <div className="panel">
          <h2>Player Links</h2>
          <div className="profile-grid">
            <div><span>Linked accounts</span><strong>Demo locked</strong></div>
            <div><span>Assign account</span><strong>Auth required</strong></div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
