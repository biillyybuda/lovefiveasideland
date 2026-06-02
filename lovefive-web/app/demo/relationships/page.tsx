import { AppShell } from "@/components/app-shell";
import { InteractiveChemistry } from "@/components/interactive-chemistry";
import { Stat } from "@/components/stats";
import { duoChemistry, getDemoSummary } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoRelationshipsPage() {
  const { players, matches } = await getDemoSummary();
  const teammateRows = duoChemistry(players, matches, "team");
  const matchupRows = duoChemistry(players, matches, "opponent");
  const topChem = teammateRows[0];
  const topRival = matchupRows[0];

  return (
    <AppShell active="relationships">
      <div className="page-head">
        <div>
          <div className="eyebrow">Demo League</div>
          <h1>Relationships & Rivalries</h1>
          <p className="lead">Filter teammate pairs, rivalries and formula-style relationship scores.</p>
        </div>
      </div>

      <div className="stats-grid">
        <Stat label="Teammate pairs" value={teammateRows.length} />
        <Stat label="Rivalries" value={matchupRows.length} />
        <Stat label="Top chemistry" value={topChem ? `${topChem.a} + ${topChem.b}` : "-"} />
        <Stat label="Top rivalry" value={topRival ? `${topRival.a} v ${topRival.b}` : "-"} />
      </div>

      <InteractiveChemistry teammateRows={teammateRows} matchupRows={matchupRows} players={players} matches={matches} />

      <div className="panel">
        <h2>Chemistry & Intensity</h2>
        <div className="story-list">
          <div>
            <span>Chemistry</span>
            <strong>How well two teammates perform together</strong>
            <small>Games together, win rate and goal difference all shape the signal.</small>
          </div>
          <div>
            <span>Intensity</span>
            <strong>How strong a repeated rivalry feels</strong>
            <small>More meetings, balanced outcomes and close games raise the score.</small>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
