"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import type { User } from "@supabase/supabase-js";
import { BrandMark } from "@/components/brand-mark";
import { InteractiveStatsHub } from "@/components/interactive-stats-hub";
import { ThemeToggle } from "@/components/theme-toggle";
import { getBrowserSupabase } from "@/lib/auth-client";
import { getDemoMatches, getDemoMmrHistory, getDemoPlayers, type League, type Match, type MmrHistory, type Player } from "@/lib/demo-data";

type Membership = {
  league_id: number;
  role: string | null;
  status: string | null;
};

type LeagueOption = League & {
  role: string | null;
};

type LoadState = "loading" | "ready" | "empty" | "error";

export function AuthenticatedLeagueCharts() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [leagues, setLeagues] = useState<LeagueOption[]>([]);
  const [leagueId, setLeagueId] = useState<number | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [mmrHistory, setMmrHistory] = useState<MmrHistory[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [message, setMessage] = useState("");

  const selectedLeague = leagues.find((league) => league.id === leagueId) || leagues[0];
  const currentAppUrl = process.env.NEXT_PUBLIC_CURRENT_APP_URL || "https://lovefiveasideland.onrender.com";

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
      if (existing && nextLeagues.some((league) => league.id === existing)) return existing;
      const saved = Number(window.localStorage.getItem("lovefive-selected-league") || 0);
      if (saved && nextLeagues.some((league) => league.id === saved)) return saved;
      return nextLeagues[0]?.id || null;
    });
  }, []);

  const loadLeagueData = useCallback(async (activeLeagueId: number) => {
    setState("loading");
    setMessage("");
    try {
      const [nextPlayers, nextMatches, nextHistory] = await Promise.all([
        getDemoPlayers(activeLeagueId),
        getDemoMatches(activeLeagueId, 160),
        getDemoMmrHistory(activeLeagueId)
      ]);
      setPlayers(nextPlayers);
      setMatches(nextMatches);
      setMmrHistory(nextHistory);
      setState("ready");
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Could not load charts.");
    }
  }, []);

  useEffect(() => {
    const supabase = getBrowserSupabase();
    let mounted = true;

    supabase.auth.getUser().then(({ data, error }) => {
      if (!mounted) return;
      if (error || !data.user) {
        router.replace("/login?next=/app/charts");
        return;
      }
      setUser(data.user);
      loadLeagues(data.user);
    });

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session?.user) router.replace("/login?next=/app/charts");
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
            <Link className="site-nav-link" href="/app">
              Home
            </Link>
            <Link className="site-nav-link active" href="/app/charts">
              Charts & Stats
            </Link>
            {selectedLeague ? <span className="site-league-link">{selectedLeague.name}</span> : null}
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
        <div className="page-head">
          <div>
            <div className="eyebrow">Live League</div>
            <h1>Charts & Stats</h1>
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

        {state === "loading" ? <div className="panel"><p className="muted">Loading charts...</p></div> : null}

        {state === "error" ? (
          <div className="panel">
            <h2>Charts could not load</h2>
            <p className="muted">{message}</p>
            <a className="button" href={currentAppUrl}>
              Open current app
            </a>
          </div>
        ) : null}

        {state === "empty" ? (
          <div className="panel">
            <h2>No active league found</h2>
            <p className="muted">{user?.email ? `${user.email} is signed in, but has no active league membership.` : "Sign in again to load charts."}</p>
          </div>
        ) : null}

        {state === "ready" ? <InteractiveStatsHub players={players} matches={matches} mmrHistory={mmrHistory} /> : null}
      </main>
    </div>
  );
}
