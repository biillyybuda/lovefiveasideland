import { AppShell } from "@/components/app-shell";
import { InteractiveChemistry } from "@/components/interactive-chemistry";
import { Stat } from "@/components/stats";
import { duoChemistry, getDemoSummary } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoChemistryPage() {
  const { players, matches } = await getDemoSummary();
  const teammateRows = duoChemistry(players, matches, "team");
  const matchupRows = duoChemistry(players, matches, "opponent");
  const best = teammateRows[0];
  const rivalry = matchupRows[0];

  return (
    <AppShell active="chemistry">
      <div className="page-head">
        <div>
          <div className="eyebrow">Demo League</div>
          <h1>Chemistry Lab</h1>
          <p className="lead">Teammate history, head-to-head patterns and the matchups that keep repeating.</p>
        </div>
      </div>

      <div className="stats-grid">
        <Stat label="Best duo" value={best ? `${best.a} + ${best.b}` : "-"} />
        <Stat label="Games together" value={best?.matches || 0} />
        <Stat label="Main matchup" value={rivalry ? `${rivalry.a} v ${rivalry.b}` : "-"} />
        <Stat label="Meetings" value={rivalry?.matches || 0} />
      </div>

      <InteractiveChemistry teammateRows={teammateRows} matchupRows={matchupRows} players={players} matches={matches} />
    </AppShell>
  );
}
