"use client";

import { LeagueInfoContent } from "@/components/league-info-content";
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
          <LeagueInfoContent liveLeagueName={league.name} />
        </>
      )}
    </LiveLeagueLoader>
  );
}
