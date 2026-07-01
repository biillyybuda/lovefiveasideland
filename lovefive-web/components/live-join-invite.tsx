"use client";

import { AccountLeagueOnboarding } from "@/components/account-league-onboarding";

export function LiveJoinInvite({
  leagueName,
  joinCode
}: {
  leagueId: number;
  leagueName: string;
  joinCode: string | null;
}) {
  return (
    <div className="tool-grid">
      <div className="panel">
        <h2>Invite Players</h2>
        <div className="profile-grid">
          <div><span>League</span><strong>{leagueName}</strong></div>
          <div><span>League code</span><strong>{joinCode || "-"}</strong></div>
        </div>
      </div>
      <AccountLeagueOnboarding compact />
    </div>
  );
}
