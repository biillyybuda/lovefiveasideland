"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import type { User } from "@supabase/supabase-js";
import { BrandMark } from "@/components/brand-mark";
import { MatchCard } from "@/components/match-card";
import { Stat } from "@/components/stats";
import { ThemeToggle } from "@/components/theme-toggle";
import { getBrowserSupabase } from "@/lib/auth-client";
import {
  buildPlayerSummaries,
  formatUkDate,
  getDemoMatches,
  getDemoMmrHistory,
  getDemoPlayers,
  makeNameMap,
  scoreParts,
  type League,
  type Match,
  type MmrHistory,
  type Player
} from "@/lib/demo-data";

type Membership = {
  league_id: number;
  role: string | null;
  status: string | null;
};

type LeagueOption = League & {
  role: string | null;
};

type LoadState = "loading" | "ready" | "empty" | "error";

export function AuthenticatedLeagueHome() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [leagues, setLeagues] = useState<LeagueOption[]>([]);
  const [leagueId, setLeagueId] = useState<number | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [mmrHistory, setMmrHistory] = useState<MmrHistory[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [message, setMessage] = useState("");

  const currentAppUrl = process.env.NEXT_PUBLIC_CURRENT_APP_URL || "https://lovefiveasideland.onrender.com";
  const selectedLeague = leagues.find((league) => league.id === leagueId) || leagues[0];

  const loadLeagues = useCallback(async (activeUser: User) => {
    const supabase = getBrowserSupabase();
    setState("loading");
    setMessage("");

    const { data: memberships, error: membershipError } = await supabase
      .from("league_members")
      .select("league_id,role,status")
      .eq("user_id", activeUser.id)
      .eq("status", "active");

    if (membershipError) {
      setState("error");
      setMessage(membershipError.message);
      return;
    }

    const rows = (memberships || []) as Membership[];
    const leagueIds = rows.map((row) => row.league_id).filter(Boolean);

    if (!leagueIds.length) {
      setLeagues([]);
      setLeagueId(null);
      setPlayers([]);
      setMatches([]);
      setMmrHistory([]);
      setState("empty");
      return;
    }

    const { data: leagueRows, error: leagueError } = await supabase
      .from("leagues")
      .select("id,name,join_code")
      .in("id", leagueIds)
      .order("name", { ascending: true });

    if (leagueError) {
      setState("error");
      setMessage(leagueError.message);
      return;
    }

    const roleMap = new Map(rows.map((row) => [row.league_id, row.role]));
    const nextLeagues = ((leagueRows || []) as League[]).map((league) => ({
      ...league,
      role: roleMap.get(league.id) || "member"
    }));

    setLeagues(nextLeagues);
    setLeagueId((existing) => {
      if (existing && nextLeagues.some((league) => league.id === existing)) {
        return existing;
      }
      const saved = Number(window.localStorage.getItem("lovefive-selected-league") || 0);
      if (saved && nextLeagues.some((league) => league.id === saved)) {
        return saved;
      }
      return nextLeagues[0]?.id || null;
    });
  }, []);

  const loadLeagueData = useCallback(async (activeLeagueId: number) => {
    setState("loading");
    setMessage("");

    try {
      const [nextPlayers, nextMatches, nextHistory] = await Promise.all([
        getDemoPlayers(activeLeagueId),
        getDemoMatches(activeLeagueId, 80),
        getDemoMmrHistory(activeLeagueId)
      ]);
      setPlayers(nextPlayers);
      setMatches(nextMatches);
      setMmrHistory(nextHistory);
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
      loadLeagues(data.user);
    });

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session?.user) {
        router.replace("/login?next=/app");
      }
    });

    return () => {
      mounted = false;
      listener.subscription.unsubscribe();
    };
  }, [loadLeagues, router]);

  useEffect(() => {
    if (!leagueId) return;
    window.localStorage.setItem("lovefive-selected-league", String(leagueId));
    loadLeagueData(leagueId);
  }, [leagueId, loadLeagueData]);

  const nameMap = useMemo(() => makeNameMap(players), [players]);
  const summaries = useMemo(() => buildPlayerSummaries(players, matches), [players, matches]);
  const rankedPlayers = useMemo(() => [...summaries].sort((a, b) => Number(b.mmr || 0) - Number(a.mmr || 0)), [summaries]);
  const latestMatch = matches[0];
  const firstMatch = [...matches].sort((a, b) => String(a.date).localeCompare(String(b.date)))[0];
  const highScoring = matches
    .map((match) => ({ match, total: scoreParts(match.score)?.reduce((sum, value) => sum + value, 0) || 0 }))
    .sort((a, b) => b.total - a.total)[0];
  const activePlayers = players.length;
  const avgGoals =
    matches.reduce((sum, match) => {
      const score = scoreParts(match.score);
      return sum + (score ? score[0] + score[1] : 0);
    }, 0) / Math.max(matches.length, 1);
  const historyPoints = mmrHistory.length;

  async function signOut() {
    const supabase = getBrowserSupabase();
    await supabase.auth.signOut();
    router.replace("/login");
  }

  return (
    <div className="shell">
      <header className="site-header">
        <div className="site-header-inner">
          <Link className="site-brand" href="/">
            <BrandMark />
          </Link>
          <nav className="site-nav" aria-label="League pages">
            <Link className="site-nav-link active" href="/app">
              Home
            </Link>
            <Link className="site-nav-link" href="/app/charts">
              Charts & Stats
            </Link>
            {selectedLeague ? (
              <span className="site-league-link">
                {selectedLeague.name}
              </span>
            ) : null}
          </nav>
          <div className="header-actions">
            <ThemeToggle />
            <button className="button compact-button" type="button" onClick={signOut}>
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="home-hero">
          <div className="home-hero-copy">
            <div className="eyebrow">Live League</div>
            <h1>{selectedLeague?.name || "Your Love Five league"}</h1>
            <div className="home-hero-meta">
              <span>{user?.email}</span>
              <span>{selectedLeague?.role || "member"}</span>
              <span>{state === "ready" ? "Connected to existing database" : "Loading"}</span>
            </div>
            {leagues.length > 1 ? (
              <div className="auth-inline-control">
                <label>
                  <span>League</span>
                  <select value={leagueId || ""} onChange={(event) => setLeagueId(Number(event.target.value))}>
                    {leagues.map((league) => (
                      <option value={league.id} key={league.id}>
                        {league.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            ) : null}
          </div>
          <div className="home-hero-card">
            <span>Latest Match</span>
            <strong>{latestMatch?.score || "-"}</strong>
            <small>{latestMatch ? formatUkDate(latestMatch.date) : "No matches loaded yet"}</small>
            <a className="mini-link" href={currentAppUrl}>
              Admin in current app
            </a>
          </div>
        </div>

        {state === "loading" ? <div className="panel"><p className="muted">Loading your league data...</p></div> : null}

        {state === "error" ? (
          <div className="panel">
            <h2>Could not load your league</h2>
            <p className="muted">{message}</p>
            <a className="button" href={currentAppUrl}>
              Open current app
            </a>
          </div>
        ) : null}

        {state === "empty" ? (
          <div className="panel">
            <h2>No leagues linked yet</h2>
            <p className="muted">Your account is signed in, but this browser app could not find an active league membership.</p>
            <a className="button primary" href={currentAppUrl}>
              Open current app
            </a>
          </div>
        ) : null}

        {state === "ready" ? (
          <>
            <div className="stats-grid">
              <Stat label="Players" value={activePlayers} />
              <Stat label="Matches" value={matches.length} />
              <Stat label="Avg goals" value={avgGoals.toFixed(1)} />
              <Stat label="MMR records" value={historyPoints} />
            </div>

            <section className="home-main-grid">
              <div className="panel feature-panel">
                <div className="section-title-row">
                  <div>
                    <span>Result Centre</span>
                    <h2>Latest Result</h2>
                  </div>
                </div>
                {latestMatch ? <MatchCard match={latestMatch} nameMap={nameMap} /> : <p className="muted">No matches yet.</p>}
              </div>

              <div className="panel feature-panel">
                <div className="section-title-row">
                  <div>
                    <span>Players</span>
                    <h2>Power Rankings</h2>
                  </div>
                </div>
                <div className="rank-list">
                  {rankedPlayers.slice(0, 7).map((player, index) => (
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
                </div>
                <div className="compact-result-list">
                  {matches.slice(0, 8).map((match) => (
                    <div className="compact-result" key={match.id}>
                      <span>{formatUkDate(match.date)}</span>
                      <strong>Team A {match.score} Team B</strong>
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel">
                <div className="section-title-row">
                  <div>
                    <span>Database Link</span>
                    <h2>Connected Tables</h2>
                  </div>
                </div>
                <div className="story-list">
                  <div>
                    <span>League</span>
                    <strong>{selectedLeague?.name}</strong>
                    <small>{selectedLeague?.role || "member"}</small>
                  </div>
                  <div>
                    <span>First match</span>
                    <strong>{formatUkDate(firstMatch?.date)}</strong>
                    <small>{matches.length} processed matches loaded</small>
                  </div>
                  <div>
                    <span>Highest total</span>
                    <strong>{highScoring?.total || 0} goals</strong>
                    <small>{highScoring?.match ? formatUkDate(highScoring.match.date) : ""}</small>
                  </div>
                </div>
              </div>
            </section>
          </>
        ) : null}
      </main>
    </div>
  );
}
