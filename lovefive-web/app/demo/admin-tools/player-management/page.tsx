import { AppShell } from "@/components/app-shell";
import { InteractivePlayers } from "@/components/interactive-players";
import { buildPlayerSummaries, getDemoLeague, getDemoMatches, getDemoPlayers } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoPlayerManagementPage() {
  const league = await getDemoLeague();
  const [players, matches] = await Promise.all([getDemoPlayers(league.id), getDemoMatches(league.id, 80)]);

  return (
    <AppShell active="admin">
      <div className="page-head">
        <div>
          <div className="eyebrow">Admin</div>
          <h1>Player Management</h1>
          <p className="lead">Edit player records, add single players and manage archive state.</p>
        </div>
      </div>

      <div className="panel">
        <h2>Add Player</h2>
        <div className="control-bar">
          <label><span>DB name</span><input disabled placeholder="canonical player name" /></label>
          <label><span>Display name</span><input disabled placeholder="Displayed name" /></label>
          <label><span>Starting MMR</span><input disabled placeholder="1000" /></label>
          <label><span>Fitness</span><select disabled><option>Average</option></select></label>
        </div>
        <p className="muted">The display-name field is included so the web flow keeps the fix from the Streamlit app.</p>
      </div>

      <InteractivePlayers players={buildPlayerSummaries(players, matches)} showFitness />
    </AppShell>
  );
}
