"use client";

import { LiveLeagueLoader } from "@/components/live-league-loader";

export default function AppInfoPage() {
  return (
    <LiveLeagueLoader active="info">
      {({ league }) => (
        <>
          <div className="page-head">
            <div>
              <div className="eyebrow">Live League</div>
              <h1>Info</h1>
            </div>
          </div>
          <div className="info-grid">
            <details open>
              <summary>{league.name}</summary>
              <p>Your live league is connected to the existing Love Five database. Admin tools are enabled from the profile menu for league admins.</p>
            </details>
            <details>
              <summary>Current migration status</summary>
              <p>Read-only league pages are now online. Result and player admin tools are being migrated into the web app in admin-only routes.</p>
            </details>
          </div>
        </>
      )}
    </LiveLeagueLoader>
  );
}
