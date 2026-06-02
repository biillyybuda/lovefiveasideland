"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import type { User } from "@supabase/supabase-js";
import { InteractiveStatsHub } from "@/components/interactive-stats-hub";
import { LiveAppShell } from "@/components/live-app-shell";
import { getBrowserSupabase } from "@/lib/auth-client";
import { type Match, type MmrHistory, type Player } from "@/lib/demo-data";
import {
  loadLeagueData,
  loadMyLeagues,
  selectedLeagueFromStorage,
  type LeagueOption
} from "@/lib/live-data";

type LoadState = "loading" | "ready" | "select" | "empty" | "error";

export function AuthenticatedLeagueCharts() {
  const router = useRouter();
  const params = useSearchParams();
  const [leagues, setLeagues] = useState<LeagueOption[]>([]);
  const [leagueId, setLeagueId] = useState<number | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [mmrHistory, setMmrHistory] = useState<MmrHistory[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [message, setMessage] = useState("");

  const selectedLeague = leagues.find((league) => league.id === leagueId) || null;

  const bootstrap = useCallback(async (activeUser: User) => {
    const supabase = getBrowserSupabase();
    try {
      const nextLeagues = await loadMyLeagues(supabase, activeUser);
      setLeagues(nextLeagues);
      if (!nextLeagues.length) {
        setState("empty");
        return;
      }
      const initialLeagueId = selectedLeagueFromStorage(nextLeagues);
      if (!initialLeagueId) {
        router.replace("/app");
        return;
      }
      setLeagueId(initialLeagueId);
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Could not load leagues.");
    }
  }, [router]);

  const loadSelectedLeague = useCallback(async (activeLeagueId: number) => {
    const supabase = getBrowserSupabase();
    setState("loading");
    try {
      const data = await loadLeagueData(supabase, activeLeagueId, 220);
      setPlayers(data.players);
      setMatches(data.matches);
      setMmrHistory(data.mmrHistory);
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
      bootstrap(data.user);
    });
    return () => {
      mounted = false;
    };
  }, [bootstrap, router]);

  useEffect(() => {
    if (leagueId) loadSelectedLeague(leagueId);
  }, [leagueId, loadSelectedLeague]);

  if (state === "loading") return <main className="main"><div className="panel"><p className="muted">Loading charts...</p></div></main>;
  if (state === "empty") return <main className="main"><div className="panel"><h1>No active league found</h1></div></main>;
  if (state === "error" || !selectedLeague) return <main className="main"><div className="panel"><h1>Charts could not load</h1><p className="muted">{message}</p></div></main>;

  return (
    <LiveAppShell active="charts" league={selectedLeague}>
      <div className="page-head">
        <div>
          <div className="eyebrow">Live League</div>
          <h1>Charts & Stats</h1>
        </div>
      </div>
      <InteractiveStatsHub
        initialPlayer={params.get("player") || undefined}
        initialSeason={params.get("season") || undefined}
        initialView={params.get("view") || undefined}
        reportBasePath="/app/matches"
        players={players}
        matches={matches}
        mmrHistory={mmrHistory}
      />
    </LiveAppShell>
  );
}
