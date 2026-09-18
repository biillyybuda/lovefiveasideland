import { AppShell } from "@/components/app-shell";
import { InteractiveMatchday } from "@/components/interactive-matchday";
import { getDemoLeague, getDemoMatches, getDemoPlayers } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoMatchdayPage() {
  const league = await getDemoLeague();
  const [players, matches] = await Promise.all([getDemoPlayers(league.id, true), getDemoMatches(league.id, 80)]);

  return (
    <AppShell active="matchday">
      <div className="page-head">
        <div>
          <div className="eyebrow">Demo League</div>
          <h1>Matchday Hub</h1>
        </div>
      </div>

      <InteractiveMatchday leagueKey={`demo-${league.id}`} players={players.filter(p => p.is_active === 1)} modelPlayers={players} matches={matches} />
    </AppShell>
  );
}
