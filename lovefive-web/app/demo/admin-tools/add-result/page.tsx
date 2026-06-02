import { AppShell } from "@/components/app-shell";
import { getDemoLeague, getDemoPlayers } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoAddResultPage() {
  const league = await getDemoLeague();
  const players = await getDemoPlayers(league.id);

  return (
    <AppShell active="admin">
      <div className="page-head">
        <div>
          <div className="eyebrow">Admin</div>
          <h1>Add Result</h1>
          <p className="lead">The match entry workflow from Streamlit, shown safely for demo visitors.</p>
        </div>
      </div>
      <div className="panel">
        <div className="control-bar">
          <label><span>Date</span><input disabled value="28/05/2026" readOnly /></label>
          <label><span>Team A</span><select disabled>{players.slice(0, 5).map((p) => <option key={p.id}>{p.display_name || p.name}</option>)}</select></label>
          <label><span>Team B</span><select disabled>{players.slice(5, 10).map((p) => <option key={p.id}>{p.display_name || p.name}</option>)}</select></label>
          <label><span>Score</span><input disabled placeholder="10-8" /></label>
        </div>
        <p className="muted">Real admins will be able to save, edit, delete, process unprocessed matches and reprocess seasons here.</p>
      </div>
    </AppShell>
  );
}
