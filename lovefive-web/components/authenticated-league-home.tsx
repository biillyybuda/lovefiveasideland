"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import type { User } from "@supabase/supabase-js";
import { AccountLeagueOnboarding } from "@/components/account-league-onboarding";
import { LiveAppShell } from "@/components/live-app-shell";
import { MatchCard } from "@/components/match-card";
import { Stat } from "@/components/stats";
import { StatsTicker } from "@/components/stats-ticker";
import { getBrowserSupabase } from "@/lib/auth-client";
import {
  buildPlayerSummaries,
  displayName,
  formatUkDate,
  makeNameMap,
  scoreParts,
  type Match,
  type MmrHistory,
  type Player,
  type PlayerSummary
} from "@/lib/demo-data";
import { compareFormPlayers } from "@/lib/form-engine";
import { applyPeriodMmr, findMostImprovedPlayer } from "@/lib/mmr-engine";
import {
  loadLeagueData,
  loadMyLeagues,
  saveSelectedLeague,
  selectedLeagueFromStorage,
  isAdminRole,
  type LeagueOption
} from "@/lib/live-data";

type LoadState = "auth" | "select" | "loading" | "ready" | "empty" | "error";

export function AuthenticatedLeagueHome() {
  const router = useRouter();
  const params = useSearchParams();
  const [user, setUser] = useState<User | null>(null);
  const [leagues, setLeagues] = useState<LeagueOption[]>([]);
  const [leagueId, setLeagueId] = useState<number | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [mmrHistory, setMmrHistory] = useState<MmrHistory[]>([]);
  const [state, setState] = useState<LoadState>("auth");
  const [message, setMessage] = useState("");

  const selectedLeague = leagues.find((league) => league.id === leagueId) || null;
  const admin = isAdminRole(selectedLeague?.role);

  const bootstrap = useCallback(async (activeUser: User) => {
    const supabase = getBrowserSupabase();
    setState("loading");
    setMessage("");

    try {
      const nextLeagues = await loadMyLeagues(supabase, activeUser);
      setLeagues(nextLeagues);

      if (!nextLeagues.length) {
        setState("empty");
        return;
      }

      const initialLeagueId = selectedLeagueFromStorage(nextLeagues);
      if (!initialLeagueId) {
        setState("select");
        return;
      }

      setLeagueId(initialLeagueId);
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Could not load your leagues.");
    }
  }, []);

  const loadSelectedLeague = useCallback(async (activeLeagueId: number) => {
    const supabase = getBrowserSupabase();
    setState("loading");
    setMessage("");

    try {
      const data = await loadLeagueData(supabase, activeLeagueId, 160);
      setPlayers(data.players);
      setMatches(data.matches);
      setMmrHistory(data.mmrHistory);
      setState("ready");
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Could not load league data.");
    }
  }, []);

  useEffect(() => {
    const supabase = getBrowserSupabase();
    let mounted = true;

    supabase.auth.getUser().then(({ data, error }) => {
      if (!mounted) return;
      if (error || !data.user) {
        router.replace("/login?next=/app");
        return;
      }
      setUser(data.user);
      bootstrap(data.user);
    });

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session?.user) router.replace("/login?next=/app");
    });

    return () => {
      mounted = false;
      listener.subscription.unsubscribe();
    };
  }, [bootstrap, router]);

  useEffect(() => {
    if (!leagueId) return;
    saveSelectedLeague(leagueId);
    loadSelectedLeague(leagueId);
  }, [leagueId, loadSelectedLeague]);

  function chooseLeague(id: number) {
    setLeagueId(id);
  }

  const seasons = useMemo(
    () => Array.from(new Set(matches.map((match) => String(match.date || "").slice(0, 4)).filter(Boolean))).sort((a, b) =>
      b.localeCompare(a)
    ),
    [matches]
  );
  const rawSeason = params.get("season");
  const selectedSeason = rawSeason && seasons.includes(rawSeason) ? rawSeason : "all";
  const seasonLabel = selectedSeason === "all" ? "All time" : selectedSeason;
  const scopedMatches = useMemo(
    () => selectedSeason === "all" ? matches : matches.filter((match) => String(match.date || "").startsWith(selectedSeason)),
    [matches, selectedSeason]
  );
  const nameMap = useMemo(() => makeNameMap(players), [players]);
  const playerSummaries = useMemo(
    () => applyPeriodMmr(buildPlayerSummaries(players, scopedMatches), mmrHistory, selectedSeason),
    [mmrHistory, players, scopedMatches, selectedSeason]
  );
  const rankedPlayers = useMemo(
    () => [...playerSummaries]
      .filter((player) => selectedSeason === "all" || player.allMatches.length > 0)
      .sort((a, b) => Number(b.mmr || 0) - Number(a.mmr || 0)),
    [playerSummaries, selectedSeason]
  );
  const formGuide = useMemo(
    () => [...playerSummaries]
      .filter((player) => player.allMatches.length > 0)
      .sort(compareFormPlayers)
      .slice(0, 6),
    [playerSummaries]
  );
  const latestMatch = scopedMatches[0];
  const firstMatch = [...matches].sort((a, b) => String(a.date).localeCompare(String(b.date)))[0];
  const scopedFirstMatch = [...scopedMatches].sort((a, b) => String(a.date).localeCompare(String(b.date)))[0];
  const highScoring = scopedMatches
    .map((match) => ({ match, total: scoreParts(match.score)?.reduce((sum, value) => sum + value, 0) || 0 }))
    .sort((a, b) => b.total - a.total)[0];
  const recentResults = scopedMatches.slice(0, 5);
  const mostPlayed = [...playerSummaries].sort((a, b) => b.allMatches.length - a.allMatches.length)[0];
  const goalDiffLeader = [...playerSummaries]
    .filter((player) => player.allMatches.length > 0)
    .sort((a, b) => b.goalDiff - a.goalDiff)[0];
  const mostImproved = findMostImprovedPlayer(playerSummaries, mmrHistory, seasons, selectedSeason);
  const unbeatenLeader = [...playerSummaries]
    .filter((player) => player.allMatches.length > 0)
    .sort((a, b) => streakLength(b, "W") - streakLength(a, "W"))[0];
  const roughRunLeader = [...playerSummaries]
    .filter((player) => player.allMatches.length > 0)
    .sort((a, b) => streakLength(b, "L") - streakLength(a, "L"))[0];
  const chartsSeason = selectedSeason === "all" ? "" : `&season=${selectedSeason}`;
  const matchHistoryHref = `/app/charts?view=matches${chartsSeason}#match-history`;
  const playersHref = `/app/charts?view=player${chartsSeason}#player-insights`;
  const latestReportHref = latestMatch ? `/app/matches/${latestMatch.id}` : matchHistoryHref;
  const tickerItems = [
    rankedPlayers[0] ? `Current MVP: ${rankedPlayers[0].label} (${Math.round(rankedPlayers[0].mmr || 0)} MMR)` : "",
    mostImproved ? `Breakout: ${mostImproved.player.label} ${signed(mostImproved.improvementScore)} vs previous seasons` : "",
    unbeatenLeader ? `Hot streak: ${unbeatenLeader.label} ${streakLength(unbeatenLeader, "W")} unbeaten wins in form` : "",
    roughRunLeader && streakLength(roughRunLeader, "L") > 1 ? `Needs a bounce: ${roughRunLeader.label} ${streakLength(roughRunLeader, "L")} losses in form` : "",
    goalDiffLeader ? `Goal swing leader: ${goalDiffLeader.label} ${signed(goalDiffLeader.goalDiff)}` : "",
    mostPlayed ? `Ever-present: ${mostPlayed.label} ${mostPlayed.allMatches.length} matches` : ""
  ];

  if (state === "select") {
    return (
      <main className="main">
        <div className="auth-layout">
          <div className="panel auth-panel">
            <div>
              <div className="eyebrow">League</div>
              <h1>Choose your league</h1>
              <p className="lead">Pick the league you want to open. You can switch later from the profile menu.</p>
            </div>
            <div className="league-choice-grid">
              {leagues.map((league) => (
                <button className="league-choice" type="button" onClick={() => chooseLeague(league.id)} key={league.id}>
                  <strong>{league.name}</strong>
                  <span>{league.role || "member"}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </main>
    );
  }

  if (state === "auth" || state === "loading") {
    return (
      <main className="main">
        <div className="app-loading">
          <div>
            <span>Love Five</span>
            <strong>Opening your league</strong>
          </div>
        </div>
      </main>
    );
  }

  if (state === "empty") {
    return (
      <main className="main">
        <AccountLeagueOnboarding email={user?.email} />
      </main>
    );
  }

  if (state === "error" || !selectedLeague) {
    return (
      <main className="main">
        <div className="panel">
          <h1>Could not load your league</h1>
          <p className="muted">{message}</p>
          <Link className="button primary" href="/app">Try again</Link>
          <Link className="button" href="/login">Sign in again</Link>
        </div>
      </main>
    );
  }

  return (
    <LiveAppShell active="overview" league={selectedLeague}>
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
            <Link className={selectedSeason === "all" ? "active" : ""} href="/app">
              All time
            </Link>
            {seasons.map((season) => (
              <Link className={selectedSeason === season ? "active" : ""} href={`/app?season=${season}`} key={season}>
                {season}
              </Link>
            ))}
          </div>
        </div>
        <div className="home-hero-card">
          <span>Latest Match</span>
          <strong>{latestMatch?.score || "-"}</strong>
          <small>{latestMatch ? `Team A vs Team B - ${formatUkDate(latestMatch.date)}` : "No matches yet"}</small>
          {latestMatch ? <Link className="mini-link" href={latestReportHref}>Match report</Link> : null}
          {!latestMatch && admin ? <Link className="mini-link" href="/app/admin-tools/add-result">Add result</Link> : null}
        </div>
      </div>

      <div className="stats-grid">
        <Stat label="Players" value={players.length} />
        <Stat label="Matches" value={scopedMatches.length} />
        <Stat label="First match" value={formatUkDate(scopedFirstMatch?.date || firstMatch?.date)} />
        <Stat label="Latest match" value={formatUkDate(latestMatch?.date)} />
      </div>

      <section className="home-main-grid">
        <div className="panel feature-panel">
          <div className="section-title-row">
            <div>
              <span>Result Centre</span>
              <h2>Latest Result</h2>
            </div>
            {latestMatch ? <Link className="mini-link" href={latestReportHref}>Match report</Link> : null}
          </div>
          {latestMatch ? <MatchCard match={latestMatch} nameMap={nameMap} /> : <p className="muted">No matches yet.</p>}
        </div>

        <div className="panel feature-panel">
          <div className="section-title-row">
            <div>
              <span>Players</span>
              <h2>Power Rankings</h2>
            </div>
            {admin ? <Link className="mini-link" href="/app/admin-tools/player-management">Manage players</Link> : null}
          </div>
          <div className="rank-list">
            {rankedPlayers.slice(0, 5).map((player, index) => (
              <div className="rank-row" key={player.id}>
                <span>{index + 1}</span>
                <strong>{player.label}</strong>
                <em>{Math.round(player.mmr || 0)}</em>
                <div className="form-dots">
                  {player.form.slice(0, 5).map((result, resultIndex) => (
                    <b className={`form ${result.toLowerCase()}`} key={`${player.id}-${resultIndex}`}>{result}</b>
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
              <Link className="compact-result" href={`/app/matches/${match.id}`} key={match.id}>
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
              <span>Highest total</span>
              <strong>{highScoring?.total || 0} goals</strong>
              <small>{highScoring?.match ? formatUkDate(highScoring.match.date) : ""}</small>
            </div>
          </div>
        </div>
      </section>
    </LiveAppShell>
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
