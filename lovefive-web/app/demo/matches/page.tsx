import { AppShell } from "@/components/app-shell";
import { InteractiveMatches } from "@/components/interactive-matches";
import { getDemoLeague, getDemoMatches, getDemoPlayers } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoMatchesPage() {
  const league = await getDemoLeague();
  const [matches, players] = await Promise.all([getDemoMatches(league.id, 50), getDemoPlayers(league.id)]);

  return (
    <AppShell active="matches">
      <div className="page-head">
        <div>
          <div className="eyebrow">Demo League</div>
          <h1>Match History</h1>
          <p className="lead">Recent games with lineups, scores and outcomes.</p>
        </div>
      </div>

      <InteractiveMatches matches={matches} players={players} />
    </AppShell>
  );
}
