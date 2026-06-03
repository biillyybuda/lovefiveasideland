"use client";

import Link from "next/link";
import { LiveLeagueLoader } from "@/components/live-league-loader";

export default function AppAdminToolsPage() {
  return (
    <LiveLeagueLoader active="admin" requireAdmin>
      {() => (
        <>
          <div className="page-head">
            <div>
              <div className="eyebrow">Admin</div>
              <h1>Admin Tools</h1>
            </div>
          </div>
          <div className="tool-grid">
            <Link className="panel tool-card-link" href="/app/admin-tools/add-result">
              <span>Match Management</span>
              <strong>Add, correct and rebuild match results.</strong>
            </Link>
            <Link className="panel tool-card-link" href="/app/admin-tools/player-management">
              <span>Player Management</span>
              <strong>Add, edit and archive players.</strong>
            </Link>
            <Link className="panel tool-card-link" href="/app/join-invite">
              <span>Join / Invite</span>
              <strong>Share your league code or create another league.</strong>
            </Link>
          </div>
        </>
      )}
    </LiveLeagueLoader>
  );
}
