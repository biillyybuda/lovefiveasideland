"use client";

import { InteractiveSeason } from "@/components/interactive-season";
import { LiveLeagueLoader } from "@/components/live-league-loader";

export default function AppSeasonReviewPage() {
  return (
    <LiveLeagueLoader active="season">
      {({ players, modelPlayers, matches, mmrHistory }) => (
        <>
          <div className="page-head">
            <div>
              <div className="eyebrow">Live League</div>
              <h1>Season Review</h1>
            </div>
          </div>
          <InteractiveSeason players={players} historicalPlayers={modelPlayers} matches={matches} mmrHistory={mmrHistory} />
        </>
      )}
    </LiveLeagueLoader>
  );
}
