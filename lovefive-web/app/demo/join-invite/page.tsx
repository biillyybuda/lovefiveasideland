import { AppShell } from "@/components/app-shell";

export default function DemoJoinInvitePage() {
  return (
    <AppShell active="join">
      <div className="page-head">
        <div>
          <div className="eyebrow">League</div>
          <h1>Join / Invite</h1>
          <p className="lead">The website version of league invites, join codes and create-league onboarding.</p>
        </div>
      </div>

      <div className="tool-grid">
        <div className="panel">
          <h2>Invite Players</h2>
          <div className="profile-grid">
            <div><span>League code</span><strong>DEMO2026</strong></div>
            <div><span>Share link</span><strong>Invite URL</strong></div>
          </div>
          <p className="muted">In real leagues this will copy the active league code and invite link.</p>
        </div>
        <div className="panel">
          <h2>Join League</h2>
          <div className="control-bar compact">
            <label><span>League code</span><input disabled placeholder="Paste league code" /></label>
            <button className="button" disabled>Join league</button>
          </div>
          <p className="muted">Disabled in demo so visitors cannot join or change data.</p>
        </div>
        <div className="panel">
          <h2>Create League</h2>
          <div className="control-bar compact">
            <label><span>League name</span><input disabled placeholder="Your weekly football group" /></label>
            <button className="button primary" disabled>Create league</button>
          </div>
          <p className="muted">This mirrors the Streamlit create-league flow and can sit behind a paywall later.</p>
        </div>
      </div>
    </AppShell>
  );
}
