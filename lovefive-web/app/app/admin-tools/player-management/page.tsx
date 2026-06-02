"use client";

import { LiveLeagueLoader } from "@/components/live-league-loader";
import { LivePlayerManagement } from "@/components/live-player-management";

export default function AppPlayerManagementPage() {
  return (
    <LiveLeagueLoader active="admin" requireAdmin>
      {({ league, players, matches, refresh }) => (
        <>
          <div className="page-head">
            <div>
              <div className="eyebrow">Admin</div>
              <h1>Player Management</h1>
            </div>
          </div>
          <LivePlayerManagement leagueId={league.id} players={players} matches={matches} refresh={refresh} />
        </>
      )}
    </LiveLeagueLoader>
  );
}
