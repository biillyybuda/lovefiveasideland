"use client";

import { useCallback, useEffect, useState } from "react";
import { getBrowserSupabase } from "@/lib/auth-client";
import { loadLeagueActivity, type HealthLoadState } from "@/lib/health-stats";

export function useLeagueActivity(leagueId: number | null) {
  const [attempt, setAttempt] = useState(0);
  const [loaded, setLoaded] = useState<HealthLoadState & { leagueId: number | null }>({ leagueId: null, status: "loading", records: [] });
  const reload = useCallback(() => setAttempt(value => value + 1), []);
  useEffect(() => {
    let cancelled = false;
    if (leagueId === null) return;
    setLoaded({ leagueId, status: "loading", records: [] });
    loadLeagueActivity(getBrowserSupabase(), leagueId).then(records => {
      if (!cancelled) setLoaded({ leagueId, status: "ready", records });
    }).catch(() => {
      if (!cancelled) setLoaded({ leagueId, status: "error", records: [], message: "Activity readings could not be loaded. Try again in a moment." });
    });
    return () => { cancelled = true; };
  }, [leagueId, attempt]);
  // Never show a previous league's readings while a new request is starting.
  const state: HealthLoadState = loaded.leagueId === leagueId ? loaded : { status: "loading", records: [] };
  return { state, reload };
}
