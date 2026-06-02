"use client";

import { LiveAddResult } from "@/components/live-add-result";
import { LiveLeagueLoader } from "@/components/live-league-loader";

export default function AppAddResultPage() {
  return (
    <LiveLeagueLoader active="admin" requireAdmin>
      {({ league, players, matches, refresh }) => (
        <>
          <div className="page-head">
            <div>
              <div className="eyebrow">Admin</div>
              <h1>Add Result</h1>
            </div>
          </div>
          <LiveAddResult leagueId={league.id} players={players} matches={matches} refresh={refresh} />
        </>
      )}
    </LiveLeagueLoader>
  );
}
