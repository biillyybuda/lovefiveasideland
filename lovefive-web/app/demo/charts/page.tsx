import { AppShell } from "@/components/app-shell";
import { InteractiveStatsHub } from "@/components/interactive-stats-hub";
import { getDemoLeague, getDemoMatches, getDemoMmrHistory, getDemoPlayers } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoChartsPage({
  searchParams
}: {
  searchParams?: Promise<{ view?: string | string[]; season?: string | string[]; player?: string | string[] }>;
}) {
  const params = searchParams ? await searchParams : {};
  const initialView = valueFromParam(params.view);
  const initialSeason = valueFromParam(params.season);
  const initialPlayer = valueFromParam(params.player);
  const league = await getDemoLeague();
  const [players, matches, mmrHistory] = await Promise.all([
    getDemoPlayers(league.id),
    getDemoMatches(league.id, 80),
    getDemoMmrHistory(league.id)
  ]);

  return (
    <AppShell active="charts">
      <div className="page-head">
        <div>
          <div className="eyebrow">Demo League</div>
          <h1>Charts & Stats</h1>
        </div>
      </div>

      <InteractiveStatsHub
        players={players}
        matches={matches}
        mmrHistory={mmrHistory}
        initialPlayer={initialPlayer}
        initialSeason={initialSeason}
        initialView={initialView}
      />
    </AppShell>
  );
}

function valueFromParam(value: string | string[] | undefined) {
  return Array.isArray(value) ? value[0] : value;
}
