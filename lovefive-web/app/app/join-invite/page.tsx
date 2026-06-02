"use client";

import { LiveJoinInvite } from "@/components/live-join-invite";
import { LiveLeagueLoader } from "@/components/live-league-loader";

export default function AppJoinInvitePage() {
  return (
    <LiveLeagueLoader active="join">
      {({ league }) => (
        <>
          <div className="page-head">
            <div>
              <div className="eyebrow">League</div>
              <h1>Join / Invite</h1>
            </div>
          </div>
          <LiveJoinInvite leagueId={league.id} leagueName={league.name} joinCode={league.join_code} />
        </>
      )}
    </LiveLeagueLoader>
  );
}
