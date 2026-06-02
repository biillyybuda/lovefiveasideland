import { AppShell } from "@/components/app-shell";
import { MatchCard } from "@/components/match-card";
import { Stat } from "@/components/stats";
import { buildPlayerSummaries, duoChemistry, formatUkDate, getDemoSummary, scoreParts } from "@/lib/demo-data";

export const dynamic = "force-dynamic";

export default async function DemoDashboardPage() {
  const { players, matches, nameMap, latestMatch } = await getDemoSummary();
  const summaries = buildPlayerSummaries(players, matches);
  const topPlayers = summaries.slice(0, 5);
  const highScoring = matches
    .map((match) => ({ match, total: scoreParts(match.score)?.reduce((sum, value) => sum + value, 0) || 0 }))
    .sort((a, b) => b.total - a.total)[0];
  const bestDuo = duoChemistry(players, matches, "team")[0];
  const avgGoals =
    matches.reduce((sum, match) => {
      const score = scoreParts(match.score);
      return sum + (score ? score[0] + score[1] : 0);
    }, 0) / Math.max(matches.length, 1);

  return (
    <AppShell active="dashboard">
      <div className="page-head">
        <div>
          <div className="eyebrow">Demo League</div>
          <h1>Dashboard</h1>
          <p className="lead">The quick read: form, leaders, recent action and league trends.</p>
        </div>
      </div>

      <div className="stats-grid">
        <Stat label="Avg goals" value={avgGoals.toFixed(1)} />
        <Stat label="Highest total" value={highScoring?.total || 0} />
        <Stat label="Top duo" value={bestDuo ? `${bestDuo.a} + ${bestDuo.b}` : "-"} />
        <Stat label="Latest match" value={formatUkDate(latestMatch?.date)} />
      </div>

      <section className="two-col">
        <div className="panel">
          <h2>Power Rankings</h2>
          <div className="rank-list">
            {topPlayers.map((player, index) => (
              <div className="rank-row" key={player.id}>
                <span>{index + 1}</span>
                <strong>{player.label}</strong>
                <em>{Math.round(player.mmr || 0)}</em>
                <div className="form-dots">
                  {player.form.map((result, resultIndex) => (
                    <b className={`form ${result.toLowerCase()}`} key={`${player.id}-${resultIndex}`}>
                      {result}
                    </b>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>Latest Fixture</h2>
          {latestMatch ? <MatchCard match={latestMatch} nameMap={nameMap} /> : <p className="muted">No matches yet.</p>}
        </div>
      </section>
    </AppShell>
  );
}
