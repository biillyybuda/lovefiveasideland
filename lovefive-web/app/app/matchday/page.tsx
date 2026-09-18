"use client";

import { InteractiveMatchday } from "@/components/interactive-matchday";
import { LiveLeagueLoader } from "@/components/live-league-loader";
import { isAdminRole } from "@/lib/live-data";

export default function AppMatchdayPage() {
  return (
    <LiveLeagueLoader active="matchday">
      {({ league, players, modelPlayers, matches }) => (
        <>
          <div className="page-head">
            <div>
              <div className="eyebrow">Live League</div>
              <h1>Matchday Hub</h1>
            </div>
          </div>
          <InteractiveMatchday key={league.id} leagueKey={`league-${league.id}`} players={players} modelPlayers={modelPlayers} matches={matches} adminLeagueId={isAdminRole(league.role) ? league.id : undefined} />
        </>
      )}
    </LiveLeagueLoader>
  );
}
