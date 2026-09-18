import { notFound } from "next/navigation";
import { AppShell } from "@/components/app-shell";
import { MatchReportContent } from "@/components/match-report-content";
import { buildMatchReport } from "@/lib/match-report";
import { getDemoSummary, getDemoPlayers } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoMatchReportPage({ params, searchParams }: {
  params: Promise<{ id: string }>; searchParams: Promise<{ share?: string }>;
}) {
  const [{ id }, query, data] = await Promise.all([params, searchParams, getDemoSummary()]);
  const match = data.matches.find(item => String(item.id) === id);
  if (!match) notFound();
  const players = await getDemoPlayers(data.league.id, true);
  const report = buildMatchReport({ match, matches: data.matches, players, awardPlayers: data.players, mmrHistory: data.mmrHistory, leagueId: data.league.id });
  return <AppShell active="matches"><MatchReportContent report={report} basePath="/demo" share={query.share === "1"}/></AppShell>;
}
