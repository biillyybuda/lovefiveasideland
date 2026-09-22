import type { SupabaseClient } from "@supabase/supabase-js";
import type { Match } from "./demo-data";
import { loadAllPages } from "./pagination";

/** Old databases remain readable during rollout. Only a missing competition column permits fallback. */
export async function loadCompletedMatches(client: SupabaseClient, leagueId: number): Promise<Match[]> {
  async function read(competition: boolean) {
    return loadAllPages<Match>(async (from, to) => {
      const query = competition
        ? client.from("matches").select("id,date,team_a,team_b,score,result,processed,review_note,competition")
        : client.from("matches").select("id,date,team_a,team_b,score,result,processed,review_note");
      const result = await query
        .eq("league_id", leagueId).eq("processed", 1).order("date", { ascending: false }).order("id", { ascending: false }).range(from, to);
      if (result.error) throw Object.assign(new Error(result.error.message), { code: result.error.code });
      return result;
    });
  }
  // Keep the structured database error long enough to distinguish missing schema from access/network failures.
  try { return await read(true); }
  catch (error) {
    const cause = error as { code?: string; message: string };
    if (["42703", "PGRST204"].includes(cause.code || "") && /competition/.test(cause.message)) return read(false);
    throw error;
  }
}
