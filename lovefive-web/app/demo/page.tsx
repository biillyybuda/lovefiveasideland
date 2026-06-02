import Link from "next/link";
import { AppShell } from "@/components/app-shell";
import { MatchCard } from "@/components/match-card";
import { Stat } from "@/components/stats";
import { StatsTicker } from "@/components/stats-ticker";
import {
  buildPlayerSummaries,
  displayName,
  formatUkDate,
  getDemoSummary,
  makeNameMap,
  type PlayerSummary
} from "@/lib/demo-data";
import { compareFormPlayers } from "@/lib/form-engine";
import { applyPeriodMmr, findMostImprovedPlayer } from "@/lib/mmr-engine";

export const dynamic = "force-dynamic";

export default async function DemoPage({
  searchParams
}: {
  searchParams?: Promise<{ season?: string | string[] }>;
}) {
  const params = searchParams ? await searchParams : {};
  const rawSeason = Array.isArray(params.season) ? params.season[0] : params.season;
  const { players, matches, mmrHistory, firstMatch, latestMatch } = await getDemoSummary();
  const seasons = Array.from(new Set(matches.map((match) => String(match.date || "").slice(0, 4)).filter(Boolean))).sort((a, b) =>
    b.localeCompare(a)
  );
  const selectedSeason = rawSeason && seasons.includes(rawSeason) ? rawSeason : "all";
  const scopedMatches = selectedSeason === "all" ? matches : matches.filter((match) => String(match.date || "").startsWith(selectedSeason));
  const scopedFirstMatch = [...scopedMatches].sort((a, b) => String(a.date).localeCompare(String(b.date)))[0];
  const scopedLatestMatch = scopedMatches[0];
  const playerSummaries = applyPeriodMmr(buildPlayerSummaries(players, scopedMatches), mmrHistory, selectedSeason);
  const rankedPlayers = [...playerSummaries]
    .filter((player) => selectedSeason === "all" || player.allMatches.length > 0)
    .sort((a, b) => Number(b.mmr || 0) - Number(a.mmr || 0));
  const nameMap = makeNameMap(players);
  const recentResults = scopedMatches.slice(0, 5);
  const formGuide = [...playerSummaries]
    .filter((player) => player.allMatches.length > 0)
    .sort(compareFormPlayers)
    .slice(0, 6);
  const mostPlayed = [...playerSummaries].sort((a, b) => b.allMatches.length - a.allMatches.length)[0];
  const goalDiffLeader = [...playerSummaries].filter((player) => player.allMatches.length > 0).sort((a, b) => b.goalDiff - a.goalDiff)[0];
  const mostImproved = findMostImprovedPlayer(playerSummaries, mmrHistory, seasons, selectedSeason);
  const unbeatenLeader = [...playerSummaries]
    .filter((player) => player.allMatches.length > 0)
    .sort((a, b) => streakLength(b, "W") - streakLength(a, "W"))[0];
  const roughRunLeader = [...playerSummaries]
    .filter((player) => player.allMatches.length > 0)
    .sort((a, b) => streakLength(b, "L") - streakLength(a, "L"))[0];
  const seasonLabel = selectedSeason === "all" ? "All time" : selectedSeason;
  const chartsSeason = selectedSeason === "all" ? "" : `&season=${selectedSeason}`;
  const matchHistoryHref = `/demo/charts?view=matches${chartsSeason}#match-history`;
  const playersHref = `/demo/charts?view=player${chartsSeason}#player-insights`;
  const latestReportHref = scopedLatestMatch ? `/demo/matches/${scopedLatestMatch.id}` : matchHistoryHref;
  const tickerItems = [
    rankedPlayers[0] ? `Current MVP: ${rankedPlayers[0].label} (${Math.round(rankedPlayers[0].mmr || 0)} MMR)` : "",
    mostImproved ? `Breakout: ${mostImproved.player.label} ${signed(mostImproved.improvementScore)} vs previous seasons` : "",
    unbeatenLeader ? `Hot streak: ${unbeatenLeader.label} ${streakLength(unbeatenLeader, "W")} unbeaten wins in form` : "",
    roughRunLeader && streakLength(roughRunLeader, "L") > 1 ? `Needs a bounce: ${roughRunLeader.label} ${streakLength(roughRunLeader, "L")} losses in form` : "",
    goalDiffLeader ? `Goal swing leader: ${goalDiffLeader.label} ${signed(goalDiffLeader.goalDiff)}` : "",
    mostPlayed ? `Ever-present: ${mostPlayed.label} ${mostPlayed.allMatches.length} matches` : ""
  ];

  return (
    <AppShell active="overview">
      <StatsTicker items={tickerItems} />

      <div className="home-hero">
        <div className="home-hero-copy">
          <div className="eyebrow">League Home</div>
          <h1>Home</h1>
          <div className="home-hero-meta">
            <span>{players.length} players</span>
            <span>{scopedMatches.length} matches</span>
            <span>{seasonLabel}</span>
          </div>
          <div className="season-switch" aria-label="Home season view">
            <Link className={selectedSeason === "all" ? "active" : ""} href="/demo">
              All time
            </Link>
            {seasons.map((season) => (
              <Link className={selectedSeason === season ? "active" : ""} href={`/demo?season=${season}`} key={season}>
                {season}
              </Link>
            ))}
          </div>
        </div>
        <div className="home-hero-card">
          <span>Latest Match</span>
          <strong>{scopedLatestMatch?.score || "-"}</strong>
          <small>{scopedLatestMatch ? `Team A vs Team B - ${formatUkDate(scopedLatestMatch.date)}` : "No matches yet"}</small>
          <Link className="mini-link" href={latestReportHref}>Match report</Link>
        </div>
      </div>

      <div className="stats-grid">
        <Stat label="Players" value={players.length} />
        <Stat label="Matches" value={scopedMatches.length} />
        <Stat label="First match" value={formatUkDate(scopedFirstMatch?.date || firstMatch?.date)} />
        <Stat label="Latest match" value={formatUkDate(scopedLatestMatch?.date || latestMatch?.date)} />
      </div>

      <section className="home-main-grid">
        <div className="panel feature-panel">
          <div className="section-title-row">
            <div>
              <span>Result Centre</span>
              <h2>Latest Result</h2>
            </div>
            <Link className="mini-link" href={latestReportHref}>Match report</Link>
          </div>
          {scopedLatestMatch ? <MatchCard match={scopedLatestMatch} nameMap={nameMap} /> : <p className="muted">No matches yet.</p>}
        </div>

        <div className="panel feature-panel">
          <div className="section-title-row">
            <div>
              <span>Players</span>
              <h2>Power Rankings</h2>
            </div>
            <Link className="mini-link" href={playersHref}>All players</Link>
          </div>
          <div className="rank-list">
            {rankedPlayers.slice(0, 5).map((player, index) => (
              <div className="rank-row" key={player.id}>
                <span>{index + 1}</span>
                <strong>{player.label}</strong>
                <em>{Math.round(player.mmr || 0)}</em>
                <div className="form-dots">
                  {player.form.slice(0, 5).map((result, resultIndex) => (
                    <b className={`form ${result.toLowerCase()}`} key={`${player.id}-${resultIndex}`}>
                      {result}
                    </b>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="home-support-grid">
        <div className="panel">
          <div className="section-title-row">
            <div>
              <span>Match Centre</span>
              <h2>Recent Results</h2>
            </div>
            <Link className="mini-link" href={matchHistoryHref}>All matches</Link>
          </div>
          <div className="compact-result-list">
            {recentResults.map((match) => (
              <Link className="compact-result" href={`/demo/matches/${match.id}`} key={match.id}>
                <span>{formatUkDate(match.date)}</span>
                <strong>Team A {match.score} Team B</strong>
              </Link>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="section-title-row">
            <div>
              <span>Form Table</span>
              <h2>In Form</h2>
            </div>
            <Link className="mini-link" href={playersHref}>All players</Link>
          </div>
          <div className="mini-leaderboard">
            {formGuide.map((player, index) => (
              <div className="mini-player-row" key={player.id}>
                <span>{index + 1}</span>
                <strong>{player.label}</strong>
                <div className="form-dots">
                  {player.form.slice(0, 5).map((result, resultIndex) => (
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
          <div className="section-title-row">
            <div>
              <span>League Leaders</span>
              <h2>Key Stats</h2>
            </div>
          </div>
          <div className="story-list">
            <div>
              <span>Most appearances</span>
              <strong>{mostPlayed ? displayName(mostPlayed) : "-"}</strong>
              <small>{mostPlayed ? `${mostPlayed.allMatches.length} matches` : ""}</small>
            </div>
            <div>
              <span>Most improved MMR</span>
              <strong>{mostImproved?.player.label || "-"}</strong>
              <small>
                {mostImproved
                  ? `${signed(mostImproved.currentGain)} in ${mostImproved.season}, ${signed(mostImproved.improvementScore)} vs prior average`
                  : ""}
              </small>
            </div>
            <div>
              <span>Best goal swing</span>
              <strong>{goalDiffLeader?.label || "-"}</strong>
              <small>{goalDiffLeader ? `${signed(goalDiffLeader.goalDiff)} goal difference` : ""}</small>
            </div>
            <div>
              <span>Best form streak</span>
              <strong>{unbeatenLeader?.label || "-"}</strong>
              <small>{unbeatenLeader ? `${streakLength(unbeatenLeader, "W")} wins in recent form` : ""}</small>
            </div>
          </div>
        </div>
      </section>

    </AppShell>
  );
}

function signed(value: number) {
  return value > 0 ? `+${value}` : String(value);
}

function streakLength(player: PlayerSummary | undefined, result: "W" | "L") {
  if (!player) return 0;
  let count = 0;
  for (const item of player.form) {
    if (item !== result) break;
    count += 1;
  }
  return count;
}
