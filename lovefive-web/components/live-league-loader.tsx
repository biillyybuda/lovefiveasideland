"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import type { User } from "@supabase/supabase-js";
import { LiveAppShell } from "@/components/live-app-shell";
import { getBrowserSupabase } from "@/lib/auth-client";
import type { Match, MmrHistory, Player } from "@/lib/demo-data";
import {
  loadLeagueData,
  loadMyLeagues,
  selectedLeagueFromStorage,
  type LeagueOption
} from "@/lib/live-data";

type LivePageActive = "overview" | "matchday" | "charts" | "season" | "info" | "admin" | "players" | "matches" | "join";
type LoadState = "loading" | "ready" | "empty" | "error";

export function LiveLeagueLoader({
  active,
  children,
  matchLimit = 220,
  requireAdmin = false
}: {
  active: LivePageActive;
  children: (props: {
    league: LeagueOption;
    players: Player[];
    matches: Match[];
    mmrHistory: MmrHistory[];
    refresh: () => Promise<void>;
  }) => React.ReactNode;
  matchLimit?: number;
  requireAdmin?: boolean;
}) {
  const router = useRouter();
  const [league, setLeague] = useState<LeagueOption | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [mmrHistory, setMmrHistory] = useState<MmrHistory[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [message, setMessage] = useState("");

  const loadData = useCallback(async (targetLeague: LeagueOption) => {
    const supabase = getBrowserSupabase();
    setState("loading");
    try {
      const data = await loadLeagueData(supabase, targetLeague.id, matchLimit);
      setPlayers(data.players);
      setMatches(data.matches);
      setMmrHistory(data.mmrHistory);
      setState("ready");
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Could not load league data.");
    }
  }, [matchLimit]);

  const bootstrap = useCallback(async (activeUser: User) => {
    const supabase = getBrowserSupabase();
    try {
      const leagues = await loadMyLeagues(supabase, activeUser);
      if (!leagues.length) {
        setState("empty");
        return;
      }
      const selectedId = selectedLeagueFromStorage(leagues);
      if (!selectedId) {
        router.replace("/app");
        return;
      }
      const selected = leagues.find((item) => item.id === selectedId);
      if (!selected) {
        router.replace("/app");
        return;
      }
      setLeague(selected);
      await loadData(selected);
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Could not load your league.");
    }
  }, [loadData, router]);

  useEffect(() => {
    const supabase = getBrowserSupabase();
    let mounted = true;
    supabase.auth.getUser().then(({ data, error }) => {
      if (!mounted) return;
      if (error || !data.user) {
        router.replace(`/login?next=${encodeURIComponent(window.location.pathname)}`);
        return;
      }
      bootstrap(data.user);
    });
    return () => {
      mounted = false;
    };
  }, [bootstrap, router]);

  async function refresh() {
    if (league) await loadData(league);
  }

  if (state === "loading") return <main className="main"><div className="panel"><p className="muted">Loading...</p></div></main>;
  if (state === "empty") return <main className="main"><div className="panel"><h1>No active league found</h1></div></main>;
  if (state === "error" || !league) return <main className="main"><div className="panel"><h1>Could not load page</h1><p className="muted">{message}</p></div></main>;
  if (requireAdmin && !["admin", "owner"].includes(String(league.role || "").toLowerCase())) {
    return (
      <LiveAppShell active="overview" league={league}>
        <div className="panel">
          <h1>Admin only</h1>
          <p className="muted">Only league admins can use this page.</p>
        </div>
      </LiveAppShell>
    );
  }

  return (
    <LiveAppShell active={active} league={league}>
      {children({ league, players, matches, mmrHistory, refresh })}
    </LiveAppShell>
  );
}
