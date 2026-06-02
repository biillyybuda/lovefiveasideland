import { AppShell } from "@/components/app-shell";

export default function DemoProfileSettingsPage() {
  return (
    <AppShell active="profile">
      <div className="page-head">
        <div>
          <div className="eyebrow">Account</div>
          <h1>Profile Settings</h1>
          <p className="lead">Account display name and player-link settings from the Streamlit app.</p>
        </div>
      </div>

      <div className="two-col">
        <div className="panel">
          <h2>Display Name</h2>
          <div className="control-bar compact">
            <label><span>Name</span><input disabled placeholder="Your display name" /></label>
            <button className="button" disabled>Save</button>
          </div>
          <p className="muted">This will update the account display name once web auth is live.</p>
        </div>
        <div className="panel">
          <h2>Player Link</h2>
          <div className="profile-grid">
            <div><span>Status</span><strong>Demo viewer</strong></div>
            <div><span>Linked player</span><strong>None</strong></div>
          </div>
          <p className="muted">Real members will pick their player profile here, matching the current quick setup gate.</p>
        </div>
      </div>
    </AppShell>
  );
}
