import { AppShell } from "@/components/app-shell";
import { InteractiveInsights } from "@/components/interactive-insights";
import { Stat } from "@/components/stats";
import { buildPlayerSummaries, getDemoSummary } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoInsightsPage() {
  const { players, matches } = await getDemoSummary();
  const summaries = buildPlayerSummaries(players, matches);
  const featured = [...summaries].sort((a, b) => (b.mmr || 0) - (a.mmr || 0))[0];
  const climber = [...summaries].sort((a, b) => b.goalDiff - a.goalDiff)[0];

  return (
    <AppShell active="insights">
      <div className="page-head">
        <div>
          <div className="eyebrow">Demo League</div>
          <h1>Player Insights</h1>
          <p className="lead">Player profiles with form, records and deeper match history.</p>
        </div>
      </div>

      <div className="stats-grid">
        <Stat label="Featured player" value={featured?.label || "-"} />
        <Stat label="Rating" value={featured ? Math.round(featured.mmr || 0) : "-"} />
        <Stat label="Goal diff leader" value={climber?.label || "-"} />
        <Stat label="Recent form" value={featured?.form.join(" ") || "-"} />
      </div>

      <InteractiveInsights players={summaries} rawPlayers={players} matches={matches} />
    </AppShell>
  );
}
