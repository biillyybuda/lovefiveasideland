import { AppShell } from "@/components/app-shell";
import { InteractivePlayers } from "@/components/interactive-players";
import { Stat } from "@/components/stats";
import { buildPlayerSummaries, displayName, getDemoLeague, getDemoMatches, getDemoPlayers } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoPlayersPage() {
  const league = await getDemoLeague();
  const [players, matches] = await Promise.all([getDemoPlayers(league.id), getDemoMatches(league.id, 80)]);
  const summaries = buildPlayerSummaries(players, matches);
  const topPlayer = players[0];
  const totalMatches = players.reduce((sum, player) => sum + Number(player.matches_played || 0), 0);
  const hotStreak = [...players].sort((a, b) => (b.win_streak || 0) - (a.win_streak || 0))[0];

  return (
    <AppShell active="players">
      <div className="page-head">
        <div>
          <div className="eyebrow">Demo League</div>
          <h1>Players</h1>
          <p className="lead">Ratings, records and form for the demo player pool.</p>
        </div>
      </div>

      <div className="stats-grid">
        <Stat label="Player pool" value={players.length} />
        <Stat label="Top player" value={topPlayer ? displayName(topPlayer) : "-"} />
        <Stat label="Total appearances" value={totalMatches} />
        <Stat label="Best streak" value={hotStreak ? `${displayName(hotStreak)} (${hotStreak.win_streak || 0})` : "-"} />
      </div>

      <InteractivePlayers players={summaries} />
    </AppShell>
  );
}
