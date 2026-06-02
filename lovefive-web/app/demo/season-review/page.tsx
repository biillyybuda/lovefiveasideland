import { AppShell } from "@/components/app-shell";
import { InteractiveSeason } from "@/components/interactive-season";
import { getDemoSummary } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoSeasonReviewPage() {
  const { players, matches, mmrHistory } = await getDemoSummary();

  return (
    <AppShell active="season">
      <div className="page-head">
        <div>
          <h1>Season Review</h1>
          <p className="lead">Awards, yearly trends, big swings and the stories hiding inside each season.</p>
        </div>
      </div>

      <InteractiveSeason players={players} matches={matches} mmrHistory={mmrHistory} />
    </AppShell>
  );
}
