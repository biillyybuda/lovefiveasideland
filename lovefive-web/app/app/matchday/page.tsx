"use client";

import { InteractiveMatchday } from "@/components/interactive-matchday";
import { LiveLeagueLoader } from "@/components/live-league-loader";

export default function AppMatchdayPage() {
  return (
    <LiveLeagueLoader active="matchday">
      {({ players, matches }) => (
        <>
          <div className="page-head">
            <div>
              <div className="eyebrow">Live League</div>
              <h1>Matchday Hub</h1>
            </div>
          </div>
          <InteractiveMatchday players={players} matches={matches} />
        </>
      )}
    </LiveLeagueLoader>
  );
}
