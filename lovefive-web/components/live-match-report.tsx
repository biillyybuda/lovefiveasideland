"use client";

import { notFound, useParams, useSearchParams } from "next/navigation";
import { LiveLeagueLoader } from "@/components/live-league-loader";
import { MatchReportContent } from "@/components/match-report-content";
import { MatchNoteEditor } from "@/components/match-note-editor";
import { MatchActivityPanel } from "@/components/match-activity-panel";
import { buildMatchReport } from "@/lib/match-report";
import { isAdminRole } from "@/lib/live-data";

export function LiveMatchReport() {
  const params = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  return <LiveLeagueLoader active="matches">{({ league, players, modelPlayers, matches, mmrHistory, refresh }) => {
    const match = matches.find(item => String(item.id) === String(params?.id));
    if (!match) notFound();
    const report = buildMatchReport({ match, matches, players: modelPlayers, awardPlayers: players, mmrHistory, leagueId: league.id });
    return <MatchReportContent report={report} share={searchParams?.get("share") === "1"}
      activitySection={<MatchActivityPanel key={`${league.id}:${match.id}`} leagueId={league.id} match={match} players={modelPlayers} canEdit={isAdminRole(league.role)}/>}
      activityEditable={isAdminRole(league.role)}
      noteEditor={isAdminRole(league.role) ? <MatchNoteEditor key={`${match.id}:${match.review_note || ""}`} leagueId={league.id} matchId={match.id} note={match.review_note || ""} refresh={refresh}/> : undefined}/>;
  }}</LiveLeagueLoader>;
}
